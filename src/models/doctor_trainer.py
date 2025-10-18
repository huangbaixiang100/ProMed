import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
import deepspeed
import swanlab
import torch
import torch.nn.functional as F
from accelerate import Accelerator

from peft import (PeftModel, get_peft_model_state_dict, LoraConfig, 
                  get_peft_model)
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.models.doctor_reward import overall_reward, overall_reward_with_token_allocation
from src.utils.utils import optimize_model_memory
from src.utils.patient_model import PatientModel
try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None
try:
    from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
except ImportError:
    BnbQuantizationConfig = None
    load_and_quantize_model = None


def _is_poor_quality_generation(text: str) -> bool:
    """
    Detect whether the generated text quality is poor (contains too much repetitive content)
    """
    if not text or len(text.strip()) == 0:
        return True
    
    # Detect the proportion of repeated characters
    char_counts = {}
    for char in text:
        char_counts[char] = char_counts.get(char, 0) + 1
    
    # If a character appears more than 30% of the total length, consider it poor quality
    max_char_ratio = max(char_counts.values()) / len(text)
    if max_char_ratio > 0.3:
        return True
    
    # Detect repeated phrases (consecutive identical words)
    words = text.split()
    if len(words) > 10:
        consecutive_same = 0
        max_consecutive = 0
        for i in range(1, len(words)):
            if words[i] == words[i-1]:
                consecutive_same += 1
                max_consecutive = max(max_consecutive, consecutive_same)
            else:
                consecutive_same = 0
        
        # If more than 5 consecutive identical words, consider it poor quality
        if max_consecutive > 5:
            return True
    
    # Detect excessively repeated phrases or sentences
    if len(text) > 50:
        common_repeat_patterns = [
            r'(.{2,10})\1{3,}',  # Phrases repeated more than 3 times
            r'([，。？！])\1{5,}',  # Punctuation marks repeated more than 5 times
            r'([a-zA-Z]+)\s+\1\s+\1',  # English words repeated 3 times
        ]
        
        import re
        for pattern in common_repeat_patterns:
            if re.search(pattern, text):
                return True
    
    return False


def extract_atomic_questions_from_batch(
    batch_samples: Dict[str, List[Any]], 
    num_generations: int
) -> List[str]:
    """
    Extract atomic questions from batch data for Shapley value calculation
    
    This function supports the expected Shapley workflow:
    Determine the atomic questions to evaluate before dialogue generation
    
    Args:
        batch_samples: Batch data containing prompt, question, answer, etc.
        num_generations: Number of generations per prompt
        
    Returns:
        List of atomic questions after repeated expansion
    """
    try:
        #  debug: Print all fields of batch_samples
        print(f" ===== extract_atomic_questions_from_batch debug info =====")
        print(f" All fields in batch_samples: {list(batch_samples.keys())}")
        
        for key, value in batch_samples.items():
            if isinstance(value, list):
                print(f"  {key}: type=lists, length={len(value)}")
                if value and len(value) > 0:
                    print(f"    type of first element: {type(value[0])}")
                    if isinstance(value[0], str):
                        print(f"    content of first element: '{value[0][:100]}...'")
                    else:
                        print(f"    content of first element: {str(value[0])[:100]}...")
            else:
                print(f"  {key}: type={type(value)}, content={str(value)[:50]}...")
        
        # Try to get question field from batch
        if 'question' in batch_samples:
            questions = batch_samples['question']
            print(f" Found question field, content:: {questions}")
        elif 'atomic_question' in batch_samples:
            questions = batch_samples['atomic_question']
            print(f"Found atomic_question field, content:: {questions}")
        else:
            # If no explicit question field, try to parse from prompt
            logging.warning(
                "Question field not found, trying to parse atomic questions from prompt"
            )
            questions = []
            for prompt in batch_samples.get('prompt', []):
                # Simple question extraction logic, can be adjusted according to actual data format
                if 'Question:' in prompt:
                    question_part = (
                        prompt.split('Question:')[-1].split('\n')[0].strip()
                    )
                    questions.append(question_part)
                elif 'Question:' in prompt:
                    question_part = (
                        prompt.split('Question:')[-1].split('\n')[0].strip()
                    )
                    questions.append(question_part)
                else:
                    # Default question
                    questions.append("Please diagnose based on patient information")
        
        # Repeat expansion according to num_generations
        repeated_questions = []
        for question in questions:
            repeated_questions.extend([question] * num_generations)
        
        logging.info(
            f"Successfully extracted{len(questions)} atomic questions, "
            f"expanded to{len(repeated_questions)}"
        )
        return repeated_questions
        
    except Exception as e:
        logging.error(f"Failed to extract atomic questions: {e}")
        # Return default question
        default_question = "Please diagnose based on patient information"
        prompt_count = len(batch_samples.get('prompt', []))
        total_questions = prompt_count * num_generations
        return [default_question] * total_questions


def create_completion_mask(
        completion_ids: torch.LongTensor,
        tokenizer: AutoTokenizer,
) -> torch.LongTensor:
    """
    Create a binary mask marking all content generated by the doctor model.
    
    Rules:
    1. All non-padding content is marked as 1 by default
    2. User input parts (between <|im_start|>user and <|im_end|>) are marked as 0
    3. <|endoftext|> all tokens after this are set to 0
    4. All <|im_start|>assistant and <|im_end|> tokens are set to 0
    
    Args:
        completion_ids: (seq_len,) Token IDs of the completion part
        tokenizer: The tokenizer used for encoding special tokens

    Returns:
        mask: (seq_len,) 0/1 tensor, 1 indicates tokens participating in training
    """
    seq_len = completion_ids.size(0)
    mask = torch.zeros(seq_len, dtype=torch.long, device=completion_ids.device)
    
    # Find the position of the first non-padding token
    start_pos = 0
    while start_pos < seq_len and completion_ids[start_pos] == 0:
        start_pos += 1
    
    # Mark all non-padding content as 1 by default
    mask[start_pos:] = 1
    
    # Exclude user input parts
    user_start_ids = tokenizer.encode("<|im_start|>user", add_special_tokens=False)
    user_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    
    # Exclude assistant token parts
    assistant_start_ids = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
    
    # Exclude all content after <|endoftext|>
    eos_ids = tokenizer.encode("<|endoftext|>", add_special_tokens=False)
    
    i = 0
    while i < seq_len:
        # Check if it's the start of user input
        if i + len(user_start_ids) <= seq_len and torch.all(
                completion_ids[i:i+len(user_start_ids)] == torch.tensor(user_start_ids, device=completion_ids.device)):
            user_start_pos = i  # Including <|im_start|>user tokens
            i += len(user_start_ids)
            
            # Find the end of user input
            while i < seq_len:
                if i + len(user_end_ids) <= seq_len and torch.all(
                        completion_ids[i:i+len(user_end_ids)] == torch.tensor(user_end_ids, device=completion_ids.device)):
                    user_end_pos = i + len(user_end_ids)  # Including <|im_end|> tokens
                    break
                i += 1
                
            if i < seq_len:  # Found the end of user input
                # Mark entire user input part tokens as 0
                mask[user_start_pos:user_end_pos] = 0
        
        # Check if it's the start of assistant tokens
        elif i + len(assistant_start_ids) <= seq_len and torch.all(
                completion_ids[i:i+len(assistant_start_ids)] == torch.tensor(assistant_start_ids, device=completion_ids.device)):
            # Mark assistant token parts as 0
            assistant_end_pos = i + len(assistant_start_ids)
            mask[i:assistant_end_pos] = 0
            i = assistant_end_pos
        
        # Check if it's an independent <|im_end|> token
        elif i + len(user_end_ids) <= seq_len and torch.all(
                completion_ids[i:i+len(user_end_ids)] == torch.tensor(user_end_ids, device=completion_ids.device)):
            # Set <|im_end|> tokens to 0
            mask[i:i+len(user_end_ids)] = 0
            i += len(user_end_ids)
        
        # Check if it's an EOS token
        elif i + len(eos_ids) <= seq_len and torch.all(
                completion_ids[i:i+len(eos_ids)] == torch.tensor(eos_ids, device=completion_ids.device)):
            # Find first EOS tokens, set it and all subsequent tokens to 0
            mask[i:] = 0
            break
            
        else:
            i += 1
            
    return mask


def _unwrap_peft(model):
    """
    Sequentially unwrap DeepSpeedEngine / model, and return the PeftModel.
    If not a PEFT model, return None instead of raising an exception.
    """
    if isinstance(model, deepspeed.DeepSpeedEngine):
        model = model.module  # --> Base model

    if hasattr(model, "model"):
        model = model.model  # --> PeftModel

    if not isinstance(model, PeftModel):
        logging.warning("The underlying model is not a PeftModel, may not have applied LoRA or used other methods")
        return None

    return model


def save_lora_only_in_zero2(engine, tokenizer, ckpt_dir):
    """
    save lora only for ZeRO-2
    If the model is not a PEFT model, use regular saving method
    """
    os.makedirs(ckpt_dir, exist_ok=True)

    peft_model = _unwrap_peft(engine)
    if peft_model is None:
        logging.warning("Model is not a PEFT model, using regular saving method")
        if isinstance(engine, deepspeed.DeepSpeedEngine):
            state_dict = engine.module.state_dict()
        else:
            state_dict = engine.state_dict()
        torch.save(state_dict, os.path.join(ckpt_dir, "pytorch_model.bin"))
        tokenizer.save_pretrained(ckpt_dir)
        return

    lora_params = [p for n, p in peft_model.named_parameters() if "lora" in n]
    if not lora_params:
        logging.warning("No LoRA parameters found, using regular saving method")
        if isinstance(engine, deepspeed.DeepSpeedEngine):
            state_dict = engine.module.state_dict()
        else:
            state_dict = engine.state_dict()
        torch.save(state_dict, os.path.join(ckpt_dir, "pytorch_model.bin"))
        tokenizer.save_pretrained(ckpt_dir)
        return

    enabled = isinstance(engine, deepspeed.DeepSpeedEngine) and engine.zero_optimization_stage() == 2

    with deepspeed.zero.GatheredParameters(lora_params, enabled=enabled):
        lora_state = get_peft_model_state_dict(peft_model)

    peft_model.save_pretrained(ckpt_dir, state_dict=lora_state)
    tokenizer.save_pretrained(ckpt_dir)



def generate_completions_multi_round(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    num_generations: int = 4,
    max_new_tokens: int = 128,
    max_length_for_gather: int = 2048,
    temperature: float = 0.7,
    do_sample: bool = True,
    max_generate_iterations: int = 8,
    patient_models: List['PatientModel'] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Multi-round generation with patient_models per sample.
    """
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"

    # step 1: Tokenize initial prompts
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    prompt_ids = inputs["input_ids"].to(device)
    prompt_mask = inputs["attention_mask"].to(device)

    # Repeat for multiple generations
    prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
    prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)

    # Expand patient_models if needed
    if patient_models is not None and num_generations > 1:
        expanded_patient_models = []
        for model_i in patient_models:
            expanded_patient_models.extend([model_i] * num_generations)
        patient_models = expanded_patient_models

    batch_size = prompt_ids.size(0)

    current_ids = prompt_ids.clone()
    current_mask = prompt_mask.clone()

    should_gen = torch.ones(batch_size, dtype=torch.bool, device=device)
    # Complete prompt + all generated content
    final_outputs: List[Optional[torch.LongTensor]] = [None] * batch_size
    completion_texts = [""] * batch_size

    # Different samples in the same batch decode together, need to pay attention to padding, re-pad after each round of generation
    for round_idx in range(max_generate_iterations):
        print("=" * 80)
        print(f"[Round {round_idx + 1}/{max_generate_iterations}] Start")
        print(f"  should_gen: {should_gen.tolist()}")
        print(f"  current_ids shape: {current_ids.shape}")

        if not should_gen.any():
            break

        active = torch.nonzero(should_gen).squeeze(1) #Get sample IDs that need to continue generation
        print(f"[Generation] active batch indices: {active.tolist()}")

        #Generate for active samples
        outputs = model.generate(
            input_ids=current_ids,
            attention_mask=current_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,  # Add repetition penalty
            no_repeat_ngram_size=3,  # Prevent 3-gram repetition
        )

        old_len = current_ids.size(1)
        history_ids = outputs[:, :old_len]
        new_generated_ids = outputs[:, old_len:]

        history_texts = tokenizer.batch_decode(history_ids, skip_special_tokens=False)
        generated_texts = tokenizer.batch_decode(new_generated_ids, skip_special_tokens=False)
        history_texts = [
            text.replace(tokenizer.pad_token, "").strip()
            for text in history_texts
        ]
        generated_texts = [
            text.replace(tokenizer.pad_token, "").strip()
            for text in generated_texts
        ]

        next_prompts = []

        for idx, text in enumerate(generated_texts):
            b = active[idx].item()
            print(f"\n[sample {b}] Generated text: {repr(text)}")

            # Detect generation quality - if contains too much repetitive content, mark as invalid
            if _is_poor_quality_generation(text):
                print(f"[Warning] sample {b} generated poor quality text, stopping generation")
                completion_texts[b] += "<invalid_generation>"
                merged_text = history_texts[idx] + "<invalid_generation>"
                final_outputs[b] = tokenizer.encode(merged_text, add_special_tokens=False, return_tensors="pt").to(
                    device).squeeze(0)
                should_gen[b] = False
                continue

            merged_text = history_texts[idx]

            # step2: Check answer or no question
            if "answer:" in text:
                completion_texts[b] += text
                merged_text += text
                final_outputs[b] = tokenizer.encode(merged_text, add_special_tokens=False, return_tensors="pt").to(
                    device).squeeze(0)
                should_gen[b] = False
                continue

            # step3: Handle question
            if "question:" in text and (round_idx < max_generate_iterations-1):
                start = text.index("question:")
                question = text[start:].strip()
                try:
                    # Each sample uses its own patient_models[b]
                    answer = patient_models[b].get_answer(question) if patient_models is not None else "No answer available."
                except Exception as exc:
                    answer = f"Get Patient Answer Error: {exc}"

                new_text = text + '\n<|im_start|>user\n' + answer + '<|im_end|>\n<|im_start|>assistant\n'
                completion_texts[b] += new_text
                merged_text += new_text

                next_prompt_ids = tokenizer.encode(merged_text, add_special_tokens=False, return_tensors="pt").to(
                    device).squeeze(0)
                next_prompts.append(next_prompt_ids)
            else:
                merged_text += text
                completion_texts[b] += text
                final_outputs[b] = tokenizer.encode(merged_text, add_special_tokens=False, return_tensors="pt").to(
                    device).squeeze(0)
                should_gen[b] = False

        if next_prompts:
            texts = [tokenizer.decode(t, skip_special_tokens=False) for t in next_prompts]
            tokenizer.padding_side = "left"
            enc = tokenizer(texts, add_special_tokens=False,return_tensors="pt", padding=True)
            current_ids = enc.input_ids.to(device)
            current_mask = enc.attention_mask.to(device)

    tokenizer.padding_side = "right"
    completion_ids=tokenizer(completion_texts,add_special_tokens=False,return_tensors="pt", padding=True).input_ids.to(
                    device)
    completion_masks=[]
    prompt_len = prompt_ids.size(1)  # Uniform fixed prompt length
    allowed_completion_len = max_length_for_gather - prompt_len

    print(" ===== generate_completions_multi_round debug info =====")
    print(f" prompt_len: {prompt_len}")
    print(f" max_length_for_gather: {max_length_for_gather}")
    print(f" allowed_completion_len: {allowed_completion_len}")
    print(f" Original completion_ids shape: {completion_ids.shape}")
    
    # Show completion_texts info for each sample
    for i, text in enumerate(completion_texts):
        original_tokens = tokenizer.encode(text, add_special_tokens=False)
        print(f" sample {i}: completion_text length={len(text)}characters, original token count={len(original_tokens)}")

    if completion_ids.size(1) > allowed_completion_len:
        # Uniformly truncate to allowed_completion_len
        print(f" Truncate completion_ids from{completion_ids.size(1)}to{allowed_completion_len}")
        completion_ids = completion_ids[:, :allowed_completion_len]
        print(f" completion_ids shape after truncation: {completion_ids.shape}")
    else:
        print(f" completion_ids needs no truncation, keep shape: {completion_ids.shape}")

    for b in range(batch_size):
        print(f" For sample{b}create completion_mask...")
        mask = create_completion_mask(
            completion_ids[b],
            tokenizer,
        )
        print(f" sample{b}'s mask: length={len(mask)}, non-zero count={mask.sum().item()}")
        completion_masks.append(mask)
    completion_masks = torch.stack(completion_masks, dim=0)
    
    print(f"🎯 Finalcompletion_masksshape: {completion_masks.shape}")
    print("=" * 60)

    return prompt_ids, prompt_mask, completion_ids,  completion_masks


def selective_log_softmax(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Compute log probabilities only for specified token IDs.

    Args:
        logits (torch.Tensor): Raw model logits (batch, seq_len, vocab_size).
        input_ids (torch.Tensor): Token IDs to select (batch, seq_len).

    Returns:
        torch.Tensor: Log probabilities for each input_id (batch, seq_len).
    """
    log_probs = F.log_softmax(logits, dim=-1)
    selected = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1))
    return selected.squeeze(-1)


def compute_log_probabilities(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    logits_to_keep: int,
) -> torch.Tensor:
    """
    Calculate the last logits_to_keep  tokens' log probabilities.
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        logits_to_keep=logits_to_keep + 1,
        obtain_logits=True,
    )
    
    if isinstance(outputs, tuple):
        logits = outputs[0]
    else:
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
    logits = logits[:, :-1, :]
    ids = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:, :]
    return selective_log_softmax(logits, ids)



def parse_dialog(completion: str) -> List[Dict[str, str]]:
    """parse dialogue content into structured format"""
    dialog = []
    pattern = re.compile(r"<\|im_start\|>(user|assistant)\s*(.*?)(?=<\|im_start\|>|$)", re.DOTALL)
    
    # Handle possible initial content
    initial_parts = completion.split("<|im_start|>", 1)
    if initial_parts[0].strip():
        dialog.append({
            "role": "assistant", 
            "content": initial_parts[0].strip()
        })
    
    for match in pattern.finditer("<|im_start|>" + completion):
        role = match.group(1).strip()
        content = match.group(2).strip()
        dialog.append({"role": role, "content": content})
    
    return dialog


def parse_dialog_simple(completion: str) -> List[Dict[str, str]]:
    """
    Simple parsing of dialogue content into structured format, supporting Chinese and English markers
    
    Args:
        completion: String containing dialogue content
        
    Returns:
        List[Dict[str, str]]: List of dictionaries containing roles and content
    """
    dialog = []
    # First clean up possible HTML tags
    completion = re.sub(r'<br\s*/?>', '\n', completion)
    completion = re.sub(r'</?(?:p|ol|ul|li|div|span|h\d|strong|em)[^>]*>', '', completion)
    
    # Split dialogue by special markers
    parts = completion.split("<|im_start|>")
    
    # Handle the first part (if not empty)
    if parts[0].strip():
        # Default first part is doctor's reply
        dialog.append({
            "role": "assistant",
            "content": parts[0].strip()
        })
    
    # Handle the remaining parts
    for part in parts[1:]:
        if not part.strip():
            continue
            
        try:
            # Extract role and content
            if part.startswith("user"):
                role = "user"
                content = part[4:].strip()  # 4 = len("user")
            elif part.startswith("assistant"):
                role = "assistant"
                content = part[9:].strip()  # 9 = len("assistant")
            else:
                # Unrecognized role, default to assistant
                role = "assistant"
                content = part.strip()
                
            # Handle end markers in content
            if "<|im_end|>" in content:
                content = content.split("<|im_end|>")[0].strip()
                
            # Remove empty dialogues
            if content.strip():
                dialog.append({"role": role, "content": content})
        except Exception as e:
            logging.warning(f"Error parsing dialogue part: {e}, Partial content: {part[:50]}...")
    
    # If no dialogue is extracted, try parsing using question/answer format
    if not dialog:
        try:
            # Try to identify question and answer format
            qa_parts = re.split(r'(Question:|question:|Answer:|answer:|Answer:)', completion, flags=re.IGNORECASE)
            current_role = "assistant"
            current_content = ""
            
            for i, part in enumerate(qa_parts):
                part = part.strip()
                if not part:
                    continue
                
                lower_part = part.lower()
                if lower_part in ['Question:', 'question:']:
                    # Save previous content
                    if current_content:
                        dialog.append({"role": current_role, "content": current_content.strip()})
                    current_role = "assistant"  # Questions are asked by the doctor
                    current_content = "question: "  # New content prefix
                elif lower_part in ['Answer:', 'answer:', 'Answer:']:
                    # Save previous content
                    if current_content:
                        dialog.append({"role": current_role, "content": current_content.strip()})
                    current_role = "assistant"  # Answers are given by the doctor
                    current_content = "answer: "  # New content prefix
                else:
                    # Add content to current part
                    if current_content or not dialog:
                        current_content += part
                    else:
                        # If no clear marker and dialogue exists, add as new reply
                        dialog.append({"role": "assistant", "content": part})
            
            # Add the last part
            if current_content:
                dialog.append({"role": current_role, "content": current_content.strip()})
        except Exception as e:
            logging.warning(f"Question-answer format parsing failed: {e}")
            # If all parsing fails, at least return the entire content as one dialogue turn
            if not dialog:
                dialog.append({"role": "assistant", "content": completion.strip()})
    
    return dialog



def generate_rollout_data(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    batch_samples: Dict[str, List[Any]],
    num_generations: int,
    max_new_tokens: int,
    max_length_for_gather: int,
    temperature: float,
    do_sample: bool,
    max_generate_iterations: int,
) -> Dict[str, Any]:
    """
    Generate completions and compute log-probabilities for rollouts.

    Args:
        model (torch.nn.Module): Current policy model.
        ref_model (torch.nn.Module): Reference (static) model.
        tokenizer (AutoTokenizer): Tokenizer for decoding.
        batch_samples (Dict[str, List[Any]]): Contains "prompt", "question", "answer" lists.
        num_generations (int): Completions per prompt.
        max_new_tokens (int): Maximum new tokens.
        max_length_for_gather (int): Maximum total length.
        temperature (float): Sampling temperature.
        do_sample (bool): Sampling flag.
        max_generate_iterations (int): Maximum generate iterations.
    Returns:
        Dict[str, Any]: Rollout data including IDs, masks, log-probs, completions, etc.
    """
    prompts = batch_samples["prompt"]
    answers = batch_samples["answer"]
    batch_facts=batch_samples['facts']

    patient_model_list = []
    for facts in batch_facts:  # batch_facts: List[List[str]]
        patient_model = PatientModel(facts)
        patient_model_list.append(patient_model)

    with torch.no_grad():
        p_ids, p_mask, c_ids, c_mask = generate_completions_multi_round(
            model,
            tokenizer,
            prompts,
            num_generations,
            max_new_tokens,
            max_length_for_gather,
            temperature,
            do_sample,
            max_generate_iterations,
            patient_model_list
        )
        input_ids = torch.cat([p_ids, c_ids], dim=1)
        attention_mask = torch.cat([p_mask, c_mask], dim=1)
        k = c_ids.size(1)

        old_log_probs = compute_log_probabilities(model, input_ids, attention_mask, k)
        ref_log_probs = compute_log_probabilities(ref_model, input_ids, attention_mask, k)

    # Modify the display of generated content to ensure format consistency
    completions = []
    for ids in c_ids:
        raw_text = tokenizer.decode(ids, skip_special_tokens=False).replace(tokenizer.pad_token, "").strip()
        # Clean HTML tags
        clean_text = re.sub(r'<br\s*/?>', '\n', raw_text)
        clean_text = re.sub(r'</?[a-zA-Z][^>]*>', '', clean_text)
        completions.append([{"content": clean_text}])
    
    # Record dialogue content
    logging.info("="*80)
    logging.info("Details of generated dialogue content:")
    for i, completion in enumerate(completions):
        content = completion[0]["content"]
        logging.info(f"sample {i} content:")
        logging.info(f"{content}")
        logging.info("-"*50)
        
        # Record dialogue content's mask info
        mask_sum = c_mask[i].sum().item()
        mask_percentage = (mask_sum / c_mask[i].size(0)) * 100
        logging.info(f"Maskinfo: Sum={mask_sum}, Percentage={mask_percentage:.2f}%, Total length={c_mask[i].size(0)}")
        
        # Analyze dialogue structure
        try:
            # More robust dialogue parsing
            dialog = parse_dialog_simple(content)
            
            # Find question and answer markers in the dialogue
            has_assistant_tag = "<|im_start|>assistant" in content
            has_question = "question:" in content.lower() or "Question:" in content.lower()
            has_answer = "answer:" in content.lower() or "Answer:" in content.lower() or "Answer:" in content.lower()
            
            # If there are multiple interactions in the dialogue, show complete dialogue
            if len(dialog) > 1:
                logging.info("Complete dialogue interaction:")
                for turn in dialog:
                    role = "doctor" if turn["role"] == "assistant" else "Patient"
                    turn_content = turn["content"]
                    logging.info(f"{role}: {turn_content}")
        except Exception as e:
            logging.warning(f"Error parsing dialogue: {e}")
        
        # Record marker check
        logging.info(f"Marker check: assistant marker={has_assistant_tag}, question marker={has_question}, answer marker={has_answer}")
    
    logging.info("="*80)
    
    repeated_facts = [f for f in batch_facts for _ in range(num_generations)]
    # Compatible with field names of two datasets: uniformly use options
    options_key = 'options' if 'options' in batch_samples else 'option'
    repeated_options = [o for o in batch_samples[options_key] for _ in range(num_generations)]
    repeated_prompts = [p for p in prompts for _ in range(num_generations)]
    repeated_answers = [a for a in answers for _ in range(num_generations)]

    # Print rollout data structure info
    print("\n" + "="*80)
    print("RolloutData structure information:")
    for key, value in {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "completion_mask": c_mask,
        "old_log_probs": old_log_probs,
        "ref_log_probs": ref_log_probs,
        "formatted_completions": completions,
        "repeated_prompts": repeated_prompts,
        "repeated_answers": repeated_answers,
        "repeated_facts": repeated_facts,
        "repeated_options": repeated_options,
        "logits_to_keep": k,
        "batch_size": len(prompts),
        "num_generations": num_generations,
    }.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: shape={value.shape}, type={value.dtype}")
        elif isinstance(value, list):
            print(f"{key}: type=lists, length={len(value)}")
            if value and hasattr(value[0], 'keys'):
                print(f"  - First element keys: {list(value[0].keys())}")
        else:
            print(f"{key}: type={type(value)}")

    # Simple content validity check function
    def simple_valid_content_check(content):
        # Check if content contains necessary tokens or text
        if not content or len(content) < 10:  # content too short
            return False
        # Check if contains question or answer
        has_q = "question:" in content.lower() or "Question:" in content.lower()
        has_a = "answer:" in content.lower() or "Answer:" in content.lower() or "Answer:" in content.lower()
        return has_q or has_a

    # Add quality check in generate_rollout_data function
    valid_completions = []
    for i, completion in enumerate(completions):
        content = completion[0]["content"]
        is_valid = simple_valid_content_check(content)
        if not is_valid:
            logging.warning(f"sample {i} failed quality check: {content[:50]}...")
            # Can regenerate or use alternative content here
        valid_completions.append(is_valid)

    # Update flags in return value
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "completion_mask": c_mask,
        "old_log_probs": old_log_probs,
        "ref_log_probs": ref_log_probs,
        "formatted_completions": completions,
        "repeated_prompts": repeated_prompts,
        "repeated_answers": repeated_answers,
        "repeated_facts": repeated_facts,
        "repeated_options":repeated_options,
        "logits_to_keep": k,
        "batch_size": len(prompts),
        "num_generations": num_generations,
        "valid_completions": valid_completions
    }


def compute_group_relative_advantages(
    rewards: torch.Tensor,
    num_generations: int,
) -> torch.Tensor:
    """
    Normalize rewards within each prompt group and handle degenerate cases.

    Args:
        rewards (torch.Tensor): Flat tensor of rewards (batch*num_gen,).
        num_generations (int): Number of completions per prompt.

    Returns:
        torch.Tensor: Advantages of shape (batch*num_gen, 1).
    """
    groups = rewards.view(-1, num_generations)
    means = groups.mean(dim=1)
    stds = groups.std(dim=1)
    mins = groups.min(dim=1).values
    maxs = groups.max(dim=1).values

    degenerate = (means == mins) | (means == maxs)
    exp_means = means.repeat_interleave(num_generations)
    exp_stds = stds.repeat_interleave(num_generations)
    mask = degenerate.repeat_interleave(num_generations)

    adv = (rewards - exp_means) / (exp_stds + 1e-4)
    # Random ±1 for degenerate groups
    rand = (torch.randint(0, 2, rewards.shape, device=rewards.device) * 2 - 1).float()
    adv[mask] = rand[mask]
    return adv.unsqueeze(1)



def maximize_grpo_objective(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    rollout_data: Dict[str, Any],
    tokenizer: AutoTokenizer,
    reward_function: Callable[..., Dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    beta: float,
    epsilon: float,
    accelerator: Accelerator,
    use_shapley: bool = False,  # New parameter: whether to use Shapley value weighting
    atomic_questions: List[str] = None,  # New parameter: atomic question lists
    use_token_level: bool = False,  # New parameter: Whether to use token-level reward allocation
    token_reward_mode: str = "token_baseline",  # New parameter: token reward mode
    alpha: float = 2.0,  # Question Shapley reward weight
    beta_reward: float = 1.0,  # QuestionResult reward weight
    gamma: float = 3.0,  # Answer correctness reward weight
    format_reward_weight: float = 1.0,  # Format reward weight
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Perform a single GRPO update step, computing loss and backpropagating.
    Enhanced version supporting token-level reward allocation.

    Args:
        model (torch.nn.Module): Policy model.
        ref_model (torch.nn.Module): Reference model.
        rollout_data (Dict[str, Any]): Output from generate_rollout_data.
        tokenizer (AutoTokenizer): For decoding completions.
        reward_function (Callable): Function to compute rewards.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        beta (float): KL penalty coefficient.
        epsilon (float): Clipping parameter.
        accelerator (Accelerator): For distributed training.
        use_shapley (bool): Whether to use Shapley value weighting for fact scores.
        atomic_questions (List[str]): List of atomic questions for Shapley calculation.
        use_token_level (bool): Whether to use token-level reward allocation.
        token_reward_mode (str): Token reward calculation mode:
                                - "token_baseline": Calculate baseline at each token position (original method)
                                - "rollout_baseline": Calculate total score for each rollout, then compare within group
        alpha (float): Process reward weight.
        beta_reward (float): Result reward weight. 
        gamma (float): Final answer reward weight.

    Returns:
        Tuple[float, float, Dict[str, Any]]: Loss value, average reward, full reward dict.
    """
    input_ids = rollout_data["input_ids"]
    attention_mask = rollout_data["attention_mask"]
    comp_mask = rollout_data["completion_mask"]
    old_lp = rollout_data["old_log_probs"]
    ref_lp = rollout_data["ref_log_probs"]
    k = rollout_data["logits_to_keep"]

    # Current policy log probs
    curr_lp = compute_log_probabilities(model, input_ids, attention_mask, k)
    ratio = torch.exp(curr_lp - old_lp)

    # Call different reward functions based on configuration
    if use_token_level:
        # Use token-level reward allocation
        from src.models.doctor_reward import overall_reward_with_token_allocation
        
        # Fix: pass truncation length to maintain consistency with completion_mask
        completion_length = rollout_data["completion_mask"].size(1) if rollout_data["completion_mask"].dim() > 1 else len(rollout_data["completion_mask"])
        
        print(" ===== Token-level Reward Allocation Debug Info =====")
        print(f" completion_maskshape: {rollout_data['completion_mask'].shape}")
        print(f" input_idsshape: {rollout_data['input_ids'].shape}")
        print(f" logits_to_keep: {k}")
        print(f" completion_length: {completion_length}")
        print(f" batch count: {len(rollout_data['formatted_completions'])}")
        print(f" Token reward mode: {token_reward_mode}")
        
        # Check original text length for each completion
        for i, completion_list in enumerate(rollout_data['formatted_completions']):
            for j, completion_dict in enumerate(completion_list):
                content = completion_dict.get('content', '')
                tokens = tokenizer.encode(content, add_special_tokens=False)
                print(f"📄 sample {i}-{j}: original text length={len(tokens)}, first 50 characters of content='{content[:50]}...'")
        
        #  Pure token-level reward calculation
        rewards_dict = overall_reward_with_token_allocation(
            model=model,
            tokenizer=tokenizer,
            facts=rollout_data["repeated_facts"],
            completions=rollout_data["formatted_completions"],
            options=rollout_data["repeated_options"],
            answers=rollout_data["repeated_answers"],
            use_shapley=use_shapley,
            atomic_questions=atomic_questions,
            use_token_level=True,
            alpha=alpha,
            beta=beta_reward,
            gamma=gamma,
            format_reward_weight=format_reward_weight,  
            max_completion_length=completion_length
        )
        
        # Handle token-level rewards
        if "token_rewards" in rewards_dict:
            print("Use pure token-level reward allocation mode")
            token_rewards_list = rewards_dict["token_rewards"]
            
            #  Record statistics of various token rewards to SwanLab
            if "question_token_rewards" in rewards_dict:
                question_token_mean = sum(rewards_dict["question_token_rewards"]) / len(rewards_dict["question_token_rewards"]) if rewards_dict["question_token_rewards"] else 0.0
                print(f" Question tokensAverage reward: {question_token_mean:.4f}")
                # SwanLab record - Will be recorded uniformly in train_with_grpo
            
            if "answer_token_rewards" in rewards_dict:
                answer_token_mean = sum(rewards_dict["answer_token_rewards"]) / len(rewards_dict["answer_token_rewards"]) if rewards_dict["answer_token_rewards"] else 0.0
                print(f" Answer tokensAverage reward: {answer_token_mean:.4f}")
            
            if "format_token_rewards" in rewards_dict:
                format_token_mean = sum(rewards_dict["format_token_rewards"]) / len(rewards_dict["format_token_rewards"]) if rewards_dict["format_token_rewards"] else 0.0
                print(f" Format tokensAverage reward: {format_token_mean:.4f}")
            
            if "token_rewards_mean" in rewards_dict:
                total_token_mean = sum(rewards_dict["token_rewards_mean"]) / len(rewards_dict["token_rewards_mean"]) if rewards_dict["token_rewards_mean"] else 0.0
                print(f" Total tokensAverage reward: {total_token_mean:.4f}")
            
            print(f" token_rewards_listlength: {len(token_rewards_list)}")
            for i, token_rewards in enumerate(token_rewards_list):
                print(f" sample {i}: token_rewardslength={len(token_rewards)}")
            
            # Token-level Group Baseline mode: each token advantage = token_reward - group_baseline
            print(" Token-levelGroup BaselineMode：token_advantage = token_reward - group_baseline")
            
            # Use new group baseline advantage calculation
            adv = compute_token_level_group_advantages(
                token_rewards_list, rollout_data["num_generations"], comp_mask
            )
            
            print(f" Token-level Group Baseline Advantage statistics:")
            print(f"  Mean: {adv.mean():.4f}, Standard deviation: {adv.std():.4f}")
            print(f"  shape: {adv.shape}, Number of non-zero elements: {(adv != 0).sum().item()}")
            print(f"  Number of positive advantages: {(adv > 0).sum().item()}")
            print(f"  Number of negative advantages: {(adv < 0).sum().item()}")
                
            # In token-level mode, avg_reward uses the average of all token rewards
            avg_reward = float(adv.mean().item())
                
        else:
            print("Token-level reward calculation failed, fallback to global reward")
            # Fallback to global reward
            rewards = torch.tensor(rewards_dict["total_scores"], dtype=torch.float32, device=curr_lp.device)
            avg_reward = float(rewards.mean())
            adv = compute_group_relative_advantages(rewards, rollout_data["num_generations"])
            
    else:
        # Traditional rollout-level reward calculation
        print(" Using traditional rollout-level reward calculation")
        
        # Call different reward functions based on configuration
        if use_shapley:
            # Use Shapley value weighted overall_reward
            from src.models.doctor_reward import overall_reward_with_token_allocation
            rewards_dict = overall_reward_with_token_allocation(
                model=model,
                tokenizer=tokenizer,
                facts=rollout_data["repeated_facts"],
                completions=rollout_data["formatted_completions"],
                options=rollout_data["repeated_options"],
                answers=rollout_data["repeated_answers"],
                use_shapley=True,
                atomic_questions=atomic_questions,
                use_token_level=False,  # Explicitly set to False
            )
        else:
            # Use traditional overall_reward
            rewards_dict = reward_function(
                model=model,
                tokenizer=tokenizer,
                facts=rollout_data["repeated_facts"],
                completions=rollout_data["formatted_completions"],
                options=rollout_data["repeated_options"],
                answers=rollout_data["repeated_answers"],
                use_shapley=False,
            )
        
        # Traditional rollout-level advantage calculation
        rewards = torch.tensor(rewards_dict["total_scores"], dtype=torch.float32, device=curr_lp.device)
        avg_reward = float(rewards.mean())
        adv = compute_group_relative_advantages(rewards, rollout_data["num_generations"])
    
    # GRPO loss calculation
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * adv
    surr = torch.min(surr1, surr2)

    kl = torch.exp(ref_lp - curr_lp) - (ref_lp - curr_lp) - 1
    per_token = surr - beta * kl
    loss = -((per_token * comp_mask).sum(dim=1) / comp_mask.sum(dim=1)).mean()

    optimizer.zero_grad()
    accelerator.backward(loss)
    optimizer.step()
    return float(loss), avg_reward, rewards_dict


def compute_token_level_advantages(
    token_rewards: torch.Tensor, 
    num_generations: int,
    completion_mask: torch.Tensor
) -> torch.Tensor:
    """
    Calculate token-level advantages
    
    Args:
        token_rewards: [batch_size, seq_len] Token-level rewards
        num_generations: Number of generations per prompt
        completion_mask: [batch_size, seq_len] Completion mask
        
    Returns:
        torch.Tensor: Token-level advantages
    """
    
    batch_size_rewards, seq_len_rewards = token_rewards.shape
    batch_size_mask, seq_len_mask = completion_mask.shape
    
    print(f" token_rewards: batch_size={batch_size_rewards}, seq_len={seq_len_rewards}")
    print(f" completion_mask: batch_size={batch_size_mask}, seq_len={seq_len_mask}")
    
    # Ensure batch_size matches
    if batch_size_rewards != batch_size_mask:
        raise ValueError(f"token_rewardsand completion_mask batch_size mismatch: {batch_size_rewards} vs {batch_size_mask}")
    
    # If sequence lengths mismatch, need to align
    if seq_len_rewards != seq_len_mask:
        print(f"⚠️ Detected length mismatch: token_rewards({seq_len_rewards}) vs completion_mask({seq_len_mask})")
        
        # Use shorter length for truncation
        min_seq_len = min(seq_len_rewards, seq_len_mask)
        
        print(f"🔧 Align length to: {min_seq_len}")
        
        # Truncate to same length
        token_rewards_orig = token_rewards.clone()
        completion_mask_orig = completion_mask.clone()
        
        token_rewards = token_rewards[:, :min_seq_len]
        completion_mask = completion_mask[:, :min_seq_len]
        
        print(f" After alignmenttoken_rewardsshape: {token_rewards.shape}")
        print(f" After alignmentcompletion_maskshape: {completion_mask.shape}")
        
        # Show truncation information
        for i in range(min(3, batch_size_rewards)):  # Only show first 3 samples
            orig_reward_nonzero = (token_rewards_orig[i] != 0).sum().item()
            new_reward_nonzero = (token_rewards[i] != 0).sum().item()
            orig_mask_nonzero = (completion_mask_orig[i] != 0).sum().item() 
            new_mask_nonzero = (completion_mask[i] != 0).sum().item()
            print(f" sample {i}: Non-zero rewards {orig_reward_nonzero}->{new_reward_nonzero}, Non-zero mask {orig_mask_nonzero}->{new_mask_nonzero}")
        
        seq_len = min_seq_len
    else:
        print(" Lengths match, no alignment needed")
        seq_len = seq_len_rewards
    
    batch_size = batch_size_rewards
    
    print(f" Final processing shape: batch_size={batch_size}, seq_len={seq_len}")
    
    # Reshape to[num_prompts, num_generations, seq_len]
    num_prompts = batch_size // num_generations
    print(f"Calculation parameters: num_prompts={num_prompts}, num_generations={num_generations}")
    
    reshaped_rewards = token_rewards.view(num_prompts, num_generations, seq_len)
    reshaped_mask = completion_mask.view(num_prompts, num_generations, seq_len)
    
    print(f" After reshapingrewardsshape: {reshaped_rewards.shape}")
    print(f" After reshapingmaskshape: {reshaped_mask.shape}")
    
    # CalculateEach tokenposition置的baseline（Average reward）
    masked_rewards = reshaped_rewards * reshaped_mask
    token_count = reshaped_mask.sum(dim=1, keepdim=True)  # [num_prompts, 1, seq_len]
    token_count = torch.clamp(token_count, min=1)  # Avoid division by zero
    
    baseline = masked_rewards.sum(dim=1, keepdim=True) / token_count  # [num_prompts, 1, seq_len]
    
    print(f" Calculate baseline: masked_rewardsSum={masked_rewards.sum():.4f}")
    print(f" token_countRange: min={token_count.min()}, max={token_count.max()}")
    print(f" baselineStatistics: mean={baseline.mean():.4f}, std={baseline.std():.4f}")
    
    # Calculate advantage
    advantage = reshaped_rewards - baseline  # [num_prompts, num_generations, seq_len]
    
    print(f" advantageStatistics: mean={advantage.mean():.4f}, std={advantage.std():.4f}")
    
    
    advantage = advantage.view(batch_size, seq_len)
    
    
    return advantage


def compute_rollout_total_advantages(
    token_rewards_list: List[List[float]], 
    num_generations: int,
    completion_mask: torch.Tensor
) -> torch.Tensor:
    """
    Token-level Rollout Baseline calculation method:
    1. Each token保持ownspecificrewardvalue（0-3分）
    2. Group baseline = 该组all tokens reward的平Mean
    3. Each token advantages = ownreward - group baseline
    
    Args:
        token_rewards_list: List[List[float]] - Each rollout的Token-level rewards
        num_generations: Number of generations per prompt
        completion_mask: [batch_size, seq_len] Completion mask
        
    Returns:
        torch.Tensor: [batch_size, seq_len] shape advantages
    """
    print(" ===== Token-levelRollout Baseline AdvantageCalculate =====")
    print(f" Inputtoken_rewards_listlength: {len(token_rewards_list)}")
    print(f" Inputcompletion_maskshape: {completion_mask.shape}")
    print(f" num_generations: {num_generations}")
    
    batch_size, seq_len = completion_mask.shape
    num_prompts = batch_size // num_generations
    
    # VerifyInputdataconsistency
    if len(token_rewards_list) != batch_size:
        print(f" token_rewards_listlength({len(token_rewards_list)}) with batch_size({batch_size}) mismatch")
        while len(token_rewards_list) < batch_size:
            token_rewards_list.append([])
    
    # Convert token rewards to tensor
    token_rewards_tensor = torch.zeros(batch_size, seq_len, device=completion_mask.device)
    
    for i, token_rewards in enumerate(token_rewards_list):
        if i < batch_size and len(token_rewards) > 0:
            # Fill token rewards into tensor, but not exceeding seq_len
            for j, reward in enumerate(token_rewards):
                if j < seq_len:
                    token_rewards_tensor[i, j] = reward
    
    print(f" Token rewards tensorshape: {token_rewards_tensor.shape}")
    
    # by groupCalculate baseline
    # Reshape to[num_prompts, num_generations, seq_len]
    grouped_token_rewards = token_rewards_tensor.view(num_prompts, num_generations, seq_len)
    grouped_completion_mask = completion_mask.view(num_prompts, num_generations, seq_len)
    
    print(f" After groupingtoken rewardsshape: {grouped_token_rewards.shape}")
    print(f" After groupingcompletion maskshape: {grouped_completion_mask.shape}")
    
    # Calculate token reward baseline for each group
    group_baselines = torch.zeros(num_prompts, device=completion_mask.device)
    
    for group_idx in range(num_prompts):
        # Get valid token rewards for all rollouts in the group
        group_rewards = grouped_token_rewards[group_idx]  # [num_generations, seq_len]
        group_mask = grouped_completion_mask[group_idx]   # [num_generations, seq_len]
        
        # Collect reward values for all valid tokens in the group
        valid_rewards = []
        for gen_idx in range(num_generations):
            for token_idx in range(seq_len):
                if group_mask[gen_idx, token_idx] == 1:  # Valid token
                    reward = group_rewards[gen_idx, token_idx].item()
                    valid_rewards.append(reward)
        
        # Calculate the average reward of all tokens in this group as the baseline
        if len(valid_rewards) > 0:
            group_baseline = sum(valid_rewards) / len(valid_rewards)
        else:
            group_baseline = 0.0
            
        group_baselines[group_idx] = group_baseline
        
        print(f" Group {group_idx}: {len(valid_rewards)}Valid token, baseline={group_baseline:.4f}")
        print(f"    valid reward distribution: min={min(valid_rewards) if valid_rewards else 0:.2f}, "
              f"max={max(valid_rewards) if valid_rewards else 0:.2f}, "
              f"mean={group_baseline:.2f}")
    
    print(f" Baselines for each group: {group_baselines}")
    
    # CalculateEach token advantages = token_reward - group_baseline
    token_advantages = torch.zeros(batch_size, seq_len, device=completion_mask.device)
    
    for group_idx in range(num_prompts):
        group_baseline = group_baselines[group_idx]
        
        for gen_idx in range(num_generations):
            # Calculate the index in the flattened batch
            batch_idx = group_idx * num_generations + gen_idx
            
            for token_idx in range(seq_len):
                if completion_mask[batch_idx, token_idx] == 1:  # Valid token
                    token_reward = token_rewards_tensor[batch_idx, token_idx]
                    token_advantage = token_reward - group_baseline
                    token_advantages[batch_idx, token_idx] = token_advantage
                    
        print(f" Group {group_idx} Token advantages calculation completed")
    
    # Debuginfo：Showsomespecific advantagesvalue
    print(" Token AdvantageDetailed analysis:")
    for i in range(min(4, batch_size)):  # Show first4rollout的info
        valid_mask = completion_mask[i] == 1
        if valid_mask.any():
            valid_rewards = token_rewards_tensor[i][valid_mask]
            valid_advantages = token_advantages[i][valid_mask]
            group_idx = i // num_generations
            
            print(f"  Rollout {i} (Group {group_idx}):")
            print(f"    Token rewards: {valid_rewards[:5].tolist()}...")  # Show first5
            print(f"    Token advantages: {valid_advantages[:5].tolist()}...")
            print(f"    Group baseline: {group_baselines[group_idx]:.4f}")
    
    print(f"Finaltoken_advantagesshape: {token_advantages.shape}")
    print(f"AdvantageStatistics: mean={token_advantages.mean():.4f}, std={token_advantages.std():.4f}")
    print(f"non-zeroadvantagecount: {(token_advantages != 0).sum().item()}")
    print("=" * 60)
    
    return token_advantages


def compute_token_level_group_advantages(
    token_rewards_list: List[List[float]], 
    num_generations: int,
    completion_mask: torch.Tensor
) -> torch.Tensor:
    """
    CalculateToken-levelGroup Baseline的Advantage
    
    Logic：
    1. Each tokenhasownreward (token_reward)
    2. Group baseline = 该within group所hasValid token的Average reward
    3. Each token advantages = token_reward - group_baseline
    
    Args:
        token_rewards_list: List[List[float]] - Each rollout的Token-level rewards
        num_generations: Number of generations per prompt
        completion_mask: [batch_size, seq_len] Completion mask
        
    Returns:
        torch.Tensor: [batch_size, seq_len] shape advantages
    """
    print(" ===== Token-level Group Baseline Advantage Calculation =====")
    
    batch_size, seq_len = completion_mask.shape
    num_prompts = batch_size // num_generations
    
    # VerifyInputdataconsistency
    if len(token_rewards_list) != batch_size:
        print(f" token_rewards_listlength({len(token_rewards_list)}) with batch_size({batch_size}) mismatch")
        while len(token_rewards_list) < batch_size:
            token_rewards_list.append([])
    
    # Convert token rewards to tensor
    token_rewards_tensor = torch.zeros(batch_size, seq_len, device=completion_mask.device)
    
    for i, token_rewards in enumerate(token_rewards_list):
        if i < batch_size and len(token_rewards) > 0:
            # Fill token rewards into tensor, but not exceeding seq_len
            for j, reward in enumerate(token_rewards):
                if j < seq_len:
                    token_rewards_tensor[i, j] = reward
    
    print(f" Token rewards tensorshape: {token_rewards_tensor.shape}")
    
    # by groupCalculategroup baseline
    group_baselines = torch.zeros(num_prompts, device=completion_mask.device)
    
    for group_idx in range(num_prompts):
        # Get the rollout index range for this group
        start_idx = group_idx * num_generations
        end_idx = (group_idx + 1) * num_generations
        
        # Collect reward values for all valid tokens in the group
        group_token_rewards = []
        
        for rollout_idx in range(start_idx, min(end_idx, batch_size)):
            # Get the Valid token reward for this rollout
            mask = completion_mask[rollout_idx]  # [seq_len]
            rewards = token_rewards_tensor[rollout_idx]  # [seq_len]
            
            # Only collect rewards for Valid tokens
            valid_positions = (mask == 1).nonzero().squeeze(-1)
            for pos in valid_positions:
                 pos_idx = pos.item()
                 if pos_idx < rewards.size(0):
                     group_token_rewards.append(rewards[pos_idx].item())
        
        # Calculate the average reward of all tokens in this group as the baseline
        if len(group_token_rewards) > 0:
            group_baseline = sum(group_token_rewards) / len(group_token_rewards)
        else:
            group_baseline = 0.0
            
        group_baselines[group_idx] = group_baseline
        
        print(f" Group {group_idx}: {len(group_token_rewards)}Valid token")
        print(f"    Token reward distribution: min={min(group_token_rewards) if group_token_rewards else 0:.3f}, "
              f"max={max(group_token_rewards) if group_token_rewards else 0:.3f}, "
              f"baseline={group_baseline:.3f}")
    
    print(f" Baselines for each group: {group_baselines}")
    
    # CalculateEach token advantages = token_reward - group_baseline
    token_advantages = torch.zeros(batch_size, seq_len, device=completion_mask.device)
    
    for group_idx in range(num_prompts):
        group_baseline = group_baselines[group_idx]
        start_idx = group_idx * num_generations
        end_idx = (group_idx + 1) * num_generations
        
        for rollout_idx in range(start_idx, min(end_idx, batch_size)):
            for token_idx in range(seq_len):
                if completion_mask[rollout_idx, token_idx] == 1:  # Valid token
                    token_reward = token_rewards_tensor[rollout_idx, token_idx]
                    token_advantage = token_reward - group_baseline
                    token_advantages[rollout_idx, token_idx] = token_advantage
                    
        print(f" Group {group_idx} (baseline={group_baseline:.3f}) Token advantages calculation completed")
    
    # Debuginfo：Showsomespecific advantagesvalue
    print(" Token AdvantageDetailed analysis:")
    for i in range(min(3, batch_size)):  # Show first3rollout's info
        valid_mask = completion_mask[i] == 1
        if valid_mask.any():
            valid_rewards = token_rewards_tensor[i][valid_mask]
            valid_advantages = token_advantages[i][valid_mask]
            group_idx = i // num_generations
            
            print(f"  Rollout {i} (Group {group_idx}):")
            print(f"    前5Token rewards: {valid_rewards[:5].tolist()}")
            print(f"    前5Token advantages: {valid_advantages[:5].tolist()}")
            print(f"    Group baseline: {group_baselines[group_idx]:.3f}")
    
    print(f" Finaltoken_advantagesshape: {token_advantages.shape}")
    print(f" AdvantageStatistics: mean={token_advantages.mean():.4f}, std={token_advantages.std():.4f}")
    print(f"Number of positive advantage tokens: {(token_advantages > 0).sum().item()}")
    print(f" Number of negative advantage tokens: {(token_advantages < 0).sum().item()}")
    print(f" Number of zero advantage tokens: {(token_advantages == 0).sum().item()}")
    print("=" * 60)
    
    return token_advantages


def build_model(
    config,
    device: torch.device,
):
    """
    Build and return a language model based on the provided configuration and device.
    This function handles tokenizer loading, (Q)LoRA application, and memory optimization.
    Supports DeepSpeed ZeRO-2 and ZeRO-3 distributed training.
    
    Returns:
        Tuple[torch.nn.Module, AutoTokenizer]: Returns (model, tokenizer) tuple
    """
    continue_training = config.training.continue_training
    checkpoint_step = config.training.current_step
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=getattr(torch, config.model.torch_dtype),
        trust_remote_code=True,
    ).to(device)
    
    logging.info(f"Base model loaded, type: {type(model)}")
    
    # Apply LoRA (if needed)
    if config.training.use_lora:
        logging.info("Starting to apply LoRA configuration...")
        lora_cfg = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            target_modules=config.lora.target_modules,
            lora_dropout=config.lora.lora_dropout,
            bias=config.lora.bias,
            task_type=config.lora.task_type,
        )
        logging.info(f"LoRA configuration: {lora_cfg}")
        
 
        model = get_peft_model(model, lora_cfg)
        
        # Verify if LoRA is correctly applied
        logging.info(f"Model type after applying LoRA: {type(model)}")
        lora_params = [n for n, _ in model.named_parameters() if "lora" in n]
        logging.info(f"Model contains {len(lora_params)} LoRA parameters")
        if lora_params:
            logging.info(f"LoRA parameter examples: {lora_params[:5]}")
        else:
            logging.warning("Warning: No LoRA parameters found!")
            
    # Quantization configuration (if using quantization)
    if config.training.use_quant:
        # Prefer accelerate's quantization configuration
        if BnbQuantizationConfig is not None and load_and_quantize_model is not None:
            logging.info("Using accelerate's quantization configuration")
            bnb_quantization_config = BnbQuantizationConfig(
                load_in_4bit=config.qlora.load_in_4bit,
                bnb_4bit_compute_dtype=getattr(torch, config.qlora.bnb_4bit_compute_dtype),
                bnb_4bit_use_double_quant=config.qlora.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=config.qlora.bnb_4bit_quant_type,
                load_in_8bit=config.qlora.load_in_8bit,
                llm_int8_threshold=config.qlora.llm_int8_threshold,
            )
            
            model = load_and_quantize_model(model, bnb_quantization_config=bnb_quantization_config, device_map="auto")
            logging.info(f"Using quantization: {config.qlora}")
        # Fallback to transformers' BitsAndBytesConfig
        elif BitsAndBytesConfig is not None:
            logging.info("Using transformers' quantization configuration")
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=config.qlora.load_in_4bit,
                    bnb_4bit_compute_dtype=getattr(torch, config.qlora.bnb_4bit_compute_dtype),
                    bnb_4bit_use_double_quant=config.qlora.bnb_4bit_use_double_quant,
                    bnb_4bit_quant_type=config.qlora.bnb_4bit_quant_type,
                    load_in_8bit=config.qlora.load_in_8bit,
                    llm_int8_threshold=config.qlora.llm_int8_threshold,
                )
                
                model = AutoModelForCausalLM.from_pretrained(
                    config.model.name,
                    quantization_config=bnb_config,
                    torch_dtype=getattr(torch, config.model.torch_dtype),
                    trust_remote_code=True,
                    device_map="auto"
                )
                
                logging.info(f"Using quantization: {config.qlora}")
            except Exception as e:
                logging.error(f"Error during quantization: {e}")
                logging.warning("Fallback to non-quantized model")
        else:
            logging.warning("Quantization configuration unavailable, please install transformers>=4.30.0 or bitsandbytes>=0.39.0")
            logging.warning("Skip quantization, use original model")
    else:
        logging.info("不Using quantization")
    
    # Optimize memory usage
    model = optimize_model_memory(model)
    
    return model, tokenizer


def train_with_grpo(
    config: Dict[str, Any],
    device: torch.device,
    policy_model: torch.nn.Module,
    ref_base_model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    accelerator: Optional[Accelerator] = None,
    dataloader: Optional[torch.utils.data.DataLoader] = None,
    num_iterations: int = 1,
    steps_per_iteration: int = 500,
    num_generations: int = 4,
    max_new_tokens: int = 128,
    max_length_for_gather: int = 2000,
    max_generate_iterations: int = 8,
    temperature: float = 0.7,
    do_sample: bool = True,
    beta: float = 0.1,
    learning_rate: float = 5e-6,
    mu: int = 1,
    epsilon: float = 0.2,
    reward_function: Callable[..., Dict[str, Any]] = overall_reward,
    checkpoint_dir: Optional[str] = None,
    current_step: int = 0,
    save_interval: int = 5,
    use_shapley: bool = False,  
    extract_atomic_questions_fn: Optional[Callable] = None, 
    # Token-level reward allocation parameters
    use_token_level: bool = False,
    token_reward_mode: str = "token_baseline",  
    alpha_reward: float = 2.0,  # Question Shapley reward weight
    beta_reward: float = 1.0,  # QuestionResult reward weight
    gamma_reward: float = 3.0,  # Answer correctness reward weight
    format_reward_weight: float = 1.0,  # Format reward weight
) -> None:
    """
    Parameters:
        token_reward_mode (str): Token reward calculation mode：
                                - "token_baseline": Calculate baseline at each token position (original method)
                                - "rollout_baseline": Calculate total score for each rollout, then compare within group
    """    
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=learning_rate)
    policy_model.train()
    policy_model, optimizer, dataloader = accelerator.prepare(policy_model, optimizer, dataloader)
    
    # Get zero_stage - try multiple methods to obtain
    zero_stage = None
    try:
        if hasattr(policy_model, 'config') and isinstance(policy_model.config, dict) and 'zero_optimization' in policy_model.config:
            zero_stage = policy_model.config['zero_optimization']['stage']
        elif isinstance(policy_model, deepspeed.DeepSpeedEngine):
            zero_stage = policy_model.zero_optimization_stage()
        else:
            # Check configuration in accelerator
            deepspeed_plugin = getattr(accelerator.state, 'deepspeed_plugin', None)
            if deepspeed_plugin is not None and hasattr(deepspeed_plugin, 'zero_stage'):
                zero_stage = deepspeed_plugin.zero_stage
    except Exception as e:
        logging.warning(f"Unable to get zero_stage value: {str(e)}")
    
    if zero_stage is None:
        zero_stage = 3  # Default value
    
    logging.info(f"Using DeepSpeed ZeRO-{zero_stage} for training")
    logging.info(f"Token reward mode: {token_reward_mode}")
    
    # Build Reference model (only create once at the beginning of training)
    if current_step == 0 or not hasattr(train_with_grpo, '_ref_model'):
        logging.info("Create Reference model (keep original base model state, no LoRA weights applied)...")
        ref_model = AutoModelForCausalLM.from_pretrained(
            config.model.name,
            torch_dtype=getattr(torch, config.model.torch_dtype),
            trust_remote_code=True,
        ).to(device)
        ref_model.eval()
        
        # Ensure Reference model does not require gradients
        for p in ref_model.parameters():
            p.requires_grad_(False)

        # GRPO Reference model keeps original Base model state, no LoRA weights applied
        logging.info("Reference model keeps original base model state, no LoRA configuration or weights applied")
        
        # Move Reference model to correct device but not wrapped with DeepSpeed
        ref_model = ref_model.to(accelerator.device)
        
        # Cache Reference model to avoid repeated creation
        train_with_grpo._ref_model = ref_model
        logging.info("Reference model cached, will reuse for subsequent iterations")
    else:
        # Reuse cached Reference model
        ref_model = train_with_grpo._ref_model
        logging.info("Reuse cached Reference model")

    sum_steps = current_step
    for it in range(1, num_iterations + 1):
        logging.info(f"Starting GRPO iteration {it}/{num_iterations}")
        torch.cuda.empty_cache()

        step = 0
        for batch in dataloader:
            logging.info(f"Starting to generate rollout data, step {step+1}/{min(steps_per_iteration, len(dataloader))}")
            # Ensure model is in evaluation mode for generation
            was_training = policy_model.training
            policy_model.eval()
            
            with torch.no_grad():
                rollout = generate_rollout_data(
                    policy_model,
                    ref_model,
                    tokenizer,  # Use passed tokenizer to maintain consistent original behavior
                    batch,
                    num_generations,
                    max_new_tokens,
                    max_length_for_gather,
                    temperature,
                    do_sample,
                    max_generate_iterations,
                )
            
            # Restore previous training state
            if was_training:
                policy_model.train()
            logging.info("Successfully generated rollout data")
            
            # Executemu GRPO updates
            for _ in range(mu):
                # Extract atomic questions (if Shapley enabled)
                atomic_questions = None
                if use_shapley and extract_atomic_questions_fn is not None:
                    try:
                        print(f" Starting to extract atomic questions, batch fields: {list(batch.keys())}")
                        atomic_questions = extract_atomic_questions_fn(batch, num_generations)
                        logging.info(f"Extracted {len(atomic_questions)} atomic questions for Shapley calculation")
                        print(f" Extracted atomic questions: {atomic_questions[:3]}...")  # Show first3
                    except Exception as e:
                        logging.warning(f"Failed to extract atomic questions: {e}, Will use traditional reward mode")
                        import traceback
                        print(f" Detailed error information: {traceback.format_exc()}")
                        atomic_questions = None
                else:
                    if not use_shapley:
                        print(f"use_shapley=False，Skip atomic question extraction")
                    elif extract_atomic_questions_fn is None:
                        print(f" extract_atomic_questions_fn为None，Skip atomic question extraction")
                
                # Ensure parameter order matches maximize_grpo_objective definition
                loss_val, avg_r, rdict = maximize_grpo_objective(
                    model=policy_model, 
                    ref_model=ref_model, 
                    rollout_data=rollout, 
                    tokenizer=tokenizer, 
                    reward_function=reward_function, 
                    optimizer=optimizer, 
                    beta=beta, 
                    epsilon=epsilon, 
                    accelerator=accelerator,
                    use_shapley=use_shapley,
                    atomic_questions=atomic_questions,
                    use_token_level=use_token_level,
                    token_reward_mode=token_reward_mode,  
                    alpha=alpha_reward,
                    beta_reward=beta_reward,
                    gamma=gamma_reward,
                    format_reward_weight=format_reward_weight
                )
            logging.info("Successfully maximized GRPO objective function")

            print(
                f"Iteration {it}/{num_iterations}, step {step+1}/{min(steps_per_iteration, len(dataloader))}, "
                f"Loss: {loss_val:.6f}, Average reward: {avg_r:.2f}, TokenMode: {token_reward_mode}"
            )
            if accelerator.is_local_main_process:
                try:
                    def safe_avg(scores_list):
                        """Safely calculate the mean of score lists"""
                        if scores_list and len(scores_list) > 0:
                            return sum(scores_list) / len(scores_list)
                        return 0.0
                    
                    format_reward = safe_avg(rdict.get("format_scores", []))
                    answer_reward = safe_avg(rdict.get("correctness_scores", []))
                    fact_reward = safe_avg(rdict.get("fact_scores", []))
                    
                    print(f" SwanLab record - Format: {format_reward:.3f}, Answer: {answer_reward:.3f}, Fact: {fact_reward:.3f}")
                    print(f" Raw data - Format scores: {rdict.get('format_scores', [])}")
                    
                    # Build SwanLab record dictionary
                    swanlab_data = {
                        "Iteration": it,
                        "step": step+1,
                        "Loss": loss_val,
                        "Avg Reward": avg_r,
                        "Format Reward": format_reward,
                        "Answer Reward": answer_reward,
                        "Fact Score Reward": fact_reward,
                        "Token Reward Mode": token_reward_mode,  
                    }
                    
                    # Add Token-level rewards statistics to SwanLab
                    if use_token_level and "question_token_rewards" in rdict:
                        question_token_mean = sum(rdict["question_token_rewards"]) / len(rdict["question_token_rewards"]) if rdict["question_token_rewards"] else 0.0
                        swanlab_data["token_rewards/question_token_mean"] = question_token_mean
                    
                    if use_token_level and "answer_token_rewards" in rdict:
                        answer_token_mean = sum(rdict["answer_token_rewards"]) / len(rdict["answer_token_rewards"]) if rdict["answer_token_rewards"] else 0.0
                        swanlab_data["token_rewards/answer_token_mean"] = answer_token_mean
                    
                    if use_token_level and "format_token_rewards" in rdict:
                        format_token_mean = sum(rdict["format_token_rewards"]) / len(rdict["format_token_rewards"]) if rdict["format_token_rewards"] else 0.0
                        swanlab_data["token_rewards/format_token_mean"] = format_token_mean
                    
                    if use_token_level and "token_rewards_mean" in rdict:
                        total_token_mean = sum(rdict["token_rewards_mean"]) / len(rdict["token_rewards_mean"]) if rdict["token_rewards_mean"] else 0.0
                        swanlab_data["token_rewards/total_token_mean"] = total_token_mean
                    
                    # Record to SwanLab
                    swanlab.log(swanlab_data)
                except Exception as e:
                    logging.warning(f"Failed to record SwanLab log: {str(e)}")

            sum_steps += 1
            step += 1
            
            # Save checkpoint
            if sum_steps % save_interval == 0 and sum_steps > current_step:
                if accelerator.is_local_main_process:
                    logging.info(f"Save checkpoint，step {sum_steps}")
                    ckpt = f"{checkpoint_dir}/step-{sum_steps:04d}"
                    os.makedirs(ckpt, exist_ok=True)
                    
                    # Improved LoRA detection and saving logic
                    try:
                        # Get model for inspection
                        model_to_check = policy_model
                        if isinstance(policy_model, deepspeed.DeepSpeedEngine):
                            model_to_check = policy_model.module
                        
                        # Multiple ways to detect if it's a PEFT model
                        is_peft_model = False
                        peft_model = None
                        
                        # Method 1: Check type directly
                        if "PeftModel" in str(type(model_to_check)):
                            is_peft_model = True
                            peft_model = model_to_check
                            logging.info("Method 1: Direct type check, Found PEFT model")
                        
                        # Method 2: Check if has model attribute and is PeftModel
                        elif hasattr(model_to_check, "model") and "PeftModel" in str(type(model_to_check.model)):
                            is_peft_model = True
                            peft_model = model_to_check.model
                            logging.info("Method 2: Found PEFT model through model attribute")
                        
                        # Method 3: Check if has peft-related attributes
                        elif hasattr(model_to_check, "peft_config") or hasattr(model_to_check, "get_peft_model"):
                            is_peft_model = True
                            peft_model = model_to_check
                            logging.info("Method 3: Found PEFT model through PEFT attributes")
                        
                        # Method 4: Check if there are lora parameters in named_parameters
                        else:
                            lora_param_names = [n for n, _ in model_to_check.named_parameters() if "lora" in n.lower()]
                            if lora_param_names:
                                is_peft_model = True
                                peft_model = model_to_check
                                logging.info(f"Method 4: Found LoRA parameters through parameter names, total{len(lora_param_names)}")
                        
                        logging.info(f"Final detection result - Whether it is a PEFT model: {is_peft_model}")
                        logging.info(f"Model type: {type(model_to_check)}")
                        
                        # If it's a PEFT model, save LoRA weights
                        if is_peft_model and peft_model is not None:
                            logging.info("Starting to save LoRA weights...")
                            
                            # Get LoRA parameters
                            lora_params = [p for n, p in peft_model.named_parameters() if "lora" in n.lower()]
                            logging.info(f"Found{len(lora_params)}LoRA parameters")
                            
                            if lora_params:
                                # Decide whether to gather parameters based on ZeRO level
                                need_gather = isinstance(policy_model, deepspeed.DeepSpeedEngine) and zero_stage >= 2
                                logging.info(f"Whether to gather parameters: {need_gather} (ZeRO level: {zero_stage})")
                                
                                if need_gather:
                                    with deepspeed.zero.GatheredParameters(lora_params, enabled=True):
                                        lora_state = get_peft_model_state_dict(peft_model)
                                else:
                                    lora_state = get_peft_model_state_dict(peft_model)
                                
                                # Save LoRA weights
                                peft_model.save_pretrained(ckpt, state_dict=lora_state)
                                logging.info(f"LoRA weights saved to: {ckpt}")
                                
                                # Ensure config.json is saved
                                if hasattr(peft_model, 'config') and hasattr(peft_model.config, 'to_dict'):
                                    import json
                                    config_path = os.path.join(ckpt, "adapter_config.json")
                                    if not os.path.exists(config_path):
                                        with open(config_path, 'w') as f:
                                            json.dump(peft_model.config.to_dict(), f, indent=2)
                                        logging.info(f"adapter_config.json saved to: {config_path}")
                            else:
                                logging.warning("Although detected to be a PEFT model, no LoRA parameters were found, using regular save method")
                                raise ValueError("No LoRA parameters found")
                        else:
                            logging.warning("PEFT model not detected, using regular save method")
                            raise ValueError("Not a PEFT model")
                            
                    except Exception as e:
                        logging.warning(f"LoRA save failed: {e}，Try to save using regular method")
                        # Fallback to regular save method
                        if isinstance(policy_model, deepspeed.DeepSpeedEngine):
                            state_dict = policy_model.module.state_dict()
                            torch.save(state_dict, os.path.join(ckpt, "pytorch_model.bin"))
                        else:
                            policy_model.save_pretrained(ckpt)
                        logging.info("Saved complete model using regular method")
                    
                    # Always save tokenizer
                    tokenizer.save_pretrained(ckpt)
                    logging.info(f"tokenizer saved to: {ckpt}")
                            
            if step >= steps_per_iteration:
                break

            # Wait for all processes
            accelerator.wait_for_everyone()

        # Only clear memory at the end of the last iteration
        if it == num_iterations:
            logging.info("Training completed, clearing Reference model cache")
            if hasattr(train_with_grpo, '_ref_model'):
                del train_with_grpo._ref_model
        torch.cuda.empty_cache()

    # Call swanlab.finish() at the end of training
    if accelerator.is_local_main_process:
        try:
            swanlab.finish()
            logging.info("SwanLab experiment completed")
        except Exception as e:
            logging.warning(f"Failed to call swanlab.finish(): {str(e)}")






if __name__ == '__main__':
    import json
    from src.data.doctor_patient_prompts import *
    from torch.utils.data import DataLoader
    from src.data.prepare_dataset import prepare_dataset
    from accelerate import Accelerator, init_empty_weights


    def custom_collate_fn(batch):
        """
        Collate a batch of dicts with potentially non-tensor and variable-length fields.
        This version preserves lists and dicts as-is without stacking.
        """
        collated = {key: [sample[key] for sample in batch] for key in batch[0]}
        return collated

    train_dataset, eval_dataset = prepare_dataset("train", 'cmb', eval_size=1)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
    accelerator = Accelerator()


    # dataset = prepare_dataset("train", 'cmb', eval_size=2)
    # train_dataloader=DataLoader(dataset, batch_size=2, shuffle=True,collate_fn=custom_collate_fn)
    # accelerator = Accelerator()

    model_name_or_path = ("")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation='eager'
    )
    num_generations=3
    max_new_tokens=512
    max_length_for_gather=2048
    temperature=0.7
    do_sample=True
    max_generate_iterations=4

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    for batch in train_dataloader:
        with torch.no_grad():
            rollout = generate_rollout_data(
                model,
                model,
                tokenizer,
                batch,
                num_generations,
                max_new_tokens,
                max_length_for_gather,
                temperature,
                do_sample,
                max_generate_iterations,
            )

        # Print rollout data structure info
        print("\n" + "="*80)
        print("RolloutData structure information:")
        for key, value in rollout.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: shape={value.shape}, type={value.dtype}")
            elif isinstance(value, list):
                print(f"{key}: type=lists, length={len(value)}")
                if value and hasattr(value[0], 'keys'):
                    print(f"  - First element keys: {list(value[0].keys())}")
            else:
                print(f"{key}: type={type(value)}")
        
        # Print completion_mask's content and Statistics info
        print("\n" + "="*80)
        print("Completion Mask detailed info:")
        c_mask = rollout["completion_mask"]
        
        # Print overall Statistics
        total_mask_sum = c_mask.sum().item()
        total_elements = c_mask.numel()
        print(f"Total Mask Statistics: non-zero elements={total_mask_sum}, total elements={total_elements}, Percentage={(total_mask_sum/total_elements)*100:.2f}%")
        
        # Print Statistics for each sample
        for i in range(c_mask.size(0)):
            mask_sum = c_mask[i].sum().item()
            mask_percentage = (mask_sum / c_mask[i].size(0)) * 100
            print(f"sample {i}: Number of non-zero elements量={mask_sum}, Total length={c_mask[i].size(0)}, Percentage={mask_percentage:.2f}%")
            
            
            if mask_sum > 0:
                first_one = (c_mask[i] == 1).nonzero()[0].item()
                last_one = (c_mask[i] == 1).nonzero()[-1].item()
                print(f" First mask=1 position: {first_one}, Last mask=1 position: {last_one}")
                
                
                completion_length = rollout["logits_to_keep"]
                text = tokenizer.decode(rollout["input_ids"][i][-completion_length:])
                print(f"  Complete completion text: {text}")
                
                # Get the first tokens that are 1 and their surrounding context
                c_ids = rollout["input_ids"][i, -completion_length:]
                start_idx = max(0, first_one - 5)
                end_idx = min(first_one + 10, c_ids.size(0))
                context_ids = c_ids[start_idx:end_idx]
                context_text = tokenizer.decode(context_ids)
                print(f"  Context around the first mask=1 position: {context_text}")
                
                # Print the first 10 tokens that are 1
                ones_indices = (c_mask[i] == 1).nonzero().squeeze().tolist()
                if not isinstance(ones_indices, list):
                    ones_indices = [ones_indices]  
                ones_indices = ones_indices[:10]  
                ones_tokens = [tokenizer.decode(c_ids[idx:idx+1]) for idx in ones_indices]
                print(f"  The first 10 tokens that are 1: {ones_tokens}")
                
                # Find specific patterns
                assistant_pattern = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
                start_pattern = tokenizer.encode("<|im_start|>", add_special_tokens=False)
                
                # Find these patterns in completion_ids
                for j in range(len(c_ids) - len(assistant_pattern) + 1):
                    if torch.all(c_ids[j:j+len(assistant_pattern)] == torch.tensor(assistant_pattern, device=c_ids.device)):
                        print(f"  Found<|im_start|>assistant in completion at position {j}")
                        # Check the mask value at this position
                        if j < len(c_mask[i]):
                            print(f"  The mask value at this position: {c_mask[i][j:j+len(assistant_pattern)].tolist()}")
                
                for j in range(len(c_ids) - len(start_pattern) + 1):
                    if torch.all(c_ids[j:j+len(start_pattern)] == torch.tensor(start_pattern, device=c_ids.device)):
                        print(f"  Found<|im_start|> in completion at position {j}")
                        # Check the mask value at this position
                        if j < len(c_mask[i]):
                            print(f"  The mask value at this position: {c_mask[i][j:j+len(start_pattern)].tolist()}")
                
                # Recompute mask using create_completion_mask function and compare
                print("\n  Recompute mask to verify original calculation is correct:")
                recomputed_mask = create_completion_mask(
                    c_ids,
                    tokenizer,
                )
                
            
                original_sum = c_mask[i].sum().item()
                recomputed_sum = recomputed_mask.sum().item()
                match_ratio = (recomputed_mask == c_mask[i]).sum().item() / len(c_mask[i])
                
                print(f"  Original maskSum: {original_sum}, Recomputed maskSum: {recomputed_sum}")
                print(f"  Match ratio between two masks: {match_ratio*100:.2f}%")
                
                # If mismatch, find mismatch positions and check the reason
                if match_ratio < 1.0:
                    diff_indices = torch.nonzero(recomputed_mask != c_mask[i]).squeeze().tolist()
                    if not isinstance(diff_indices, list):
                        diff_indices = [diff_indices]  
                    
                    print(f"  Found{len(diff_indices)} mismatches")
                    for diff_idx in diff_indices[:5]:  
                        original_val = c_mask[i][diff_idx].item()
                        recomputed_val = recomputed_mask[diff_idx].item()
                        token = tokenizer.decode(c_ids[diff_idx:diff_idx+1])
                        print(f"    position {diff_idx}: Original value={original_val}, New value={recomputed_val}, token='{token}'")
                    
                    # Decode nearby area
                    for diff_idx in diff_indices[:2]:  
                        start = max(0, diff_idx - 10)
                        end = min(diff_idx + 10, len(c_ids))
                        context = tokenizer.decode(c_ids[start:end])
                        print(f"    position{diff_idx}Context around: '{context}'")

        print("="*80)
        
        # Analyze dialogue structure
        print("\n" + "="*80)
        print("Dialogue structure analysis:")
        for i, completion in enumerate(rollout['formatted_completions']):
            content = completion[0]["content"]
            print(f"\nsample {i} 的Dialogue structure analysis:")
            
            # Parse dialogue
            dialog = parse_dialog_simple(content)
            print(f"Dialogue turn number: {len(dialog)}")
            
            # Print the role and content summary for each turn
            for j, turn in enumerate(dialog):
                role = turn["role"]
                turn_content = turn["content"]
                # Extract content summary
                content_preview = turn_content[:50] + "..." if len(turn_content) > 50 else turn_content
                print(f"  turn {j+1}: Role={role}, content summary='{content_preview}'")
                
                # Check special tokens
                has_question = "question:" in turn_content.lower() or "Question:" in turn_content.lower()
                has_answer = "answer:" in turn_content.lower() or "Answer:" in turn_content.lower() or "Answer:" in turn_content.lower()
                
                if has_question:
                    print(f"    Contains question marker")
                if has_answer:
                    print(f"    Contains answer marker")
                    
            # Check the format of tokens in the dialogue
            has_im_start = "<|im_start|>" in content
            has_im_end = "<|im_end|>" in content
            has_assistant = "<|im_start|>assistant" in content
            has_user = "<|im_start|>user" in content
            
            print(f"Special format check: im_start={has_im_start}, im_end={has_im_end}, assistant={has_assistant}, user={has_user}")
        
        print("="*80)
        
        print("Final Results:")
        for completion in rollout['formatted_completions']:
            print("*"*80)
            print(completion)

        print("="*80)
        print("Standard reward calculation:")
        rewards_dict = overall_reward(
            model=model,
            tokenizer=tokenizer,
            facts=rollout["repeated_facts"],
            completions=rollout["formatted_completions"],
            options=rollout["repeated_options"],
            answers=rollout["repeated_answers"]
        )
        print(rewards_dict)

        # Test two Token-level rewards CalculateMode
        print("\n" + "="*80)
        print(" Token-level reward allocation mode comparison test:")
            
        # Mode1: token_baseline
        print("\n Mode1: token_baseline - Each token position Calculate baseline")
        from src.models.doctor_reward import overall_reward_with_token_allocation
        
        completion_length = rollout["completion_mask"].size(1)
        
        token_rewards_1 = overall_reward_with_token_allocation(
            model=model,
            tokenizer=tokenizer,
            facts=rollout["repeated_facts"],
            completions=rollout["formatted_completions"],
            options=rollout["repeated_options"],
            answers=rollout["repeated_answers"],
            use_shapley=False,
            atomic_questions=None,
            use_token_level=True,
            max_completion_length=completion_length
        )
        
        print(f"Token_baselineMode results: {list(token_rewards_1.keys())}")
        if 'token_rewards' in token_rewards_1:
            print(f"Number of token rewards: {len(token_rewards_1['token_rewards'])}")
            for i, tr in enumerate(token_rewards_1['token_rewards'][:2]):  
                nonzero_count = sum(1 for r in tr if r != 0)
                total_reward = sum(tr)
                print(f"  sample {i}: Number of non-zero tokens={nonzero_count}/{len(tr)}, Total reward={total_reward:.3f}")
        
        # Mode2: rollout_baseline 
        print("\n Mode2: rollout_baseline - Each rollout Calculate total score then compare within group")
        
        # Test new advantages calculation method
        if 'token_rewards' in token_rewards_1:
            token_rewards_list = token_rewards_1['token_rewards']
            
            # Test new advantages calculation method
            rollout_advantages = compute_rollout_total_advantages(
                token_rewards_list, 
                rollout["num_generations"], 
                rollout["completion_mask"]
            )
            
            print(f"Rollout_baselineModeAdvantageshape: {rollout_advantages.shape}")
            print(f"AdvantageStatistics: mean={rollout_advantages.mean():.4f}, std={rollout_advantages.std():.4f}")
        
        break



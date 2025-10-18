from typing import Any, Dict, List, Tuple
import random
import numpy as np
import torch
from openai import OpenAI
from src.data.doctor_patient_prompts import *
from src.utils.utils import call_gpt
import re


def parse_dialog(completion: str) -> List[Dict[str, str]]:
    """
    Parses a model completion string into a list of dialog turns with roles and content,
    retaining the <|im_end|> marker at the end of each message.

    Each turn is represented as a dict:
    {
        "role": "user" or "assistant",
        "content": the actual content (including <|im_end|>)
    }

    Args:
        completion: A single completion string with <|im_start|>role and <|im_end|> markers.

    Returns:
        A list of dicts with keys "role" and "content", in order of appearance.
    """
    dialog = []
    pattern = re.compile(r"<\|im_start\|>(user|assistant)\s*(.*?)(<\|im_end\|>)", re.DOTALL)

    # Optional: handle initial assistant segment before first user block
    initial_parts = completion.split("<|im_start|>", 1)
    if initial_parts[0].strip():
        dialog.append({
            "role": "assistant",
            "content": initial_parts[0].strip()
        })

    for match in pattern.finditer(completion):
        role = match.group(1).strip()
        content = match.group(2).rstrip()
        content += match.group(3)  # append <|im_end|>
        dialog.append({"role": role, "content": content})

    return dialog

def format_dialog(dialog):
    result=""
    for message in dialog[1:]:
        if message['role']=='assistant':
            result+='doctor: '+message['content']+"\n"
        else:
            result+='patient: '+message['content']+'\n'
    return result


def get_fact_score(facts, context):
    fact_checker_client = OpenAI(api_key="",
                            base_url="")
    fact_checker_model = 'qwen2.5:72b'
    fact_num = len(facts)
    correct_facts = 0
    for fact in facts:
        prompt = check_fact_prompt.format(context=context, fact=fact)
        fact_check_messages = [{"role": "user", "content": prompt}]
        ans = call_gpt(fact_checker_client,fact_checker_model,fact_check_messages)
        if "True" in ans:
            correct_facts += 1
    fact_score = correct_facts / fact_num
    return fact_score


def match_choice(text,options_dict):
    option = ["A", "B", "C", "D", "E", "F", "G"]
    #res = re.search(r"(answer: |answer|correct option)(?:is|:|as|should be|should as)(.*?)(.|\.|$)", text, re.S)
    res = re.search(r"(answer: |answer|correct option|correct conclusion|correct judgment|correct answer)(?:is|:|as|should be|should as)\s*(.*)", text, re.S) #(.*?)(.|\.|$)
    pattern = r"(?:correct answer|answer|correct option|correct conclusion|correct judgment|correct answer)[::isasshould beshould as\s]*[[]?\s*([A-Ga-g]{1,7})\s*[]]?"
    matches = re.findall(pattern, text, re.IGNORECASE)
    if matches:
        # Multiple matches take only the first one; deduplicate and sort for standardization
        answer = matches[0].upper()
        answer = "".join(sorted(set(answer)))
        if res:
             res_answer="".join([x for x in res.group(2) if x in option])
            #if res_answer!= answer:
                #print(text)
                #print(answer,res_answer)
                #print('*'*30)
        return answer
    else:
        tmp=[]
        for op_letter, op_text in options_dict.items():
            # Add null value check to prevent None type errors
            if op_text is not None and isinstance(op_text, str) and op_text in text:
                #print(f"Found {op_letter}:{op_text}")
                tmp.append(op_letter)
        return "".join(tmp)


def correctness_reward(completions: List[List[Dict[str, Any]]],options:List[Dict[str,str]], answers: List[str]) -> List[float]:
    """
    Assigns a reward based on the correctness of the model's answers.

    For each prompt, compares the model's final answer to the expected answer
    using a match_choice to get the output option and compare to the true answer
    Returns 4.0 for correct model answer, 0.0 otherwise.

    Args:
        prompts: List of prompt strings to evaluate.
        completions: Nested list of completion dicts from the model; we use the first element's "content".
        answers: List of expected answer strings.

    Returns:
        A list of floats, one per prompt, where each value is either 3.0 (correct) or 0.0 (incorrect).
    """
    rewards = []

    for i,completion_group in enumerate(completions):
        content = completion_group[0]["content"]

        # Extract the last assistant response segment
        last_response = content.split("<|im_start|>assistant")[-1].strip()

        model_answer = match_choice(last_response,options[i])
        correct_answer = answers[i].strip()
        print(model_answer,correct_answer)

        reward = 4.0 if model_answer == correct_answer else 0.0
        rewards.append(reward)

    return rewards




def format_reward(completions: List[List[Dict[str, Any]]]) -> List[float]:
    """
    Computes a formatting reward based on the presence of specific tags
    in each model response.

    Tag scoring:
      - "question:" at the beginning: 1; present: 0.5
      - "answer:" at the beginning: 1; present: 0.5
    If a single response contains both or multiple of either, reward is 0.
    Final reward is the average over all responses.

    Args:
        completions: Nested list of completion dicts from the model;
                     we use the "content" field of the first dict in each sublist.

    Returns:
        A list of floats, one per completion, representing the format score.
    """
    scores = []

    for completion_group in completions:
        content = completion_group[0]["content"]

        print(f" Format Reward debug - original content: {content[:100]}...")

        # First try parsing with parse_dialog (standard dialogue format)
        dialog = parse_dialog(content)

        if len(dialog) > 0:
            print(f" Parsed{len(dialog)}  turns dialogue")
            # Standard dialogue format processing
            total_score = 0.0
            valid_count = 0

            for response in dialog:
                if response['role'] == 'user':
                    continue
                response_content = response['content']

                # Check how many times each keyword appears
                q_count = response_content.count("question:")
                a_count = response_content.count("answer:")
                
                print(f"  Assistant reply: {response_content[:50]}...")
                print(f"  Number of question markers: {q_count}, Number of answer markers: {a_count}")

                # Invalid if more than one or both present
                if q_count + a_count != 1:
                    score = 0.0
                    print(f"   marker count abnormal， score=0")
                else:
                    if response_content.startswith("question:") or response_content.startswith("answer:"):
                        score = 1.0
                        print(f"   starts with marker， score=1.0")
                    else:
                        score = 0.5
                        print(f"    contains marker but not at start， score=0.5")

                total_score += score
                valid_count += 1

            avg_score = total_score / valid_count if valid_count > 0 else 0.0
        else:
            # if parse_dialog parsing fails, directly analyze original content
            print(f" dialogue parsing failed, directly analyze original text")
            
            # directly analyze entire content
            content_clean = content.strip()
            q_count = content_clean.count("question:")
            a_count = content_clean.count("answer:")
            
            print(f"  Number of question markers: {q_count}, Number of answer markers: {a_count}")
            
            # check if meets format requirements
            if q_count + a_count != 1:
                avg_score = 0.0
                print(f"marker count abnormal({q_count + a_count})， score=0")
            else:
                if content_clean.startswith("question:") or content_clean.startswith("answer:"):
                    avg_score = 1.0
                    print(f" starts with marker， score=1.0")
                elif "question:" in content_clean or "answer:" in content_clean:
                    avg_score = 0.5
                    print(f" contains marker but not at start， score=0.5")
                else:
                    avg_score = 0.0
                    print(f" no valid markers， score=0")

        print(f" Final Format Reward: {avg_score}")
        scores.append(avg_score)

    return scores

# ================ The relevant function for calculating the Shapley value ================

@torch.no_grad()
def get_answer_logprob(model, tokenizer, full_prompt, target_answer, past_key_values=None):
    """
    Calculate the logarithmic probability of the target answer under the given prompt
    """
    input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(model.device)
    target_ids = tokenizer(target_answer, return_tensors="pt").input_ids.to(model.device)

    if past_key_values is not None:
        outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
    else:
        outputs = model(input_ids=input_ids, use_cache=True)

    logits = outputs.logits[:, -target_ids.size(-1):, :]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_probs_for_targets = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    avg_logprob = log_probs_for_targets.mean().item()

    return avg_logprob, outputs.past_key_values


def compute_equal_shapley_values(
    model, tokenizer,
    all_facts: List[str],
    atomic_question: str,
    target_answer: str,
    max_samples: int = 50,
    min_samples: int = 3,
    tol: float = 1e-2
) -> np.ndarray:
    """
    Calculate Shapley values for all facts completely equally using Monte Carlo method
    
    Args:
        model: Doctor model
        tokenizer: Tokenizer
        all_facts: List of all facts (all facts treated equally)
        atomic_question: Atomic question
        target_answer: Target answer
        max_samples: Maximum number of samples
        min_samples: Minimum number of samples
        tol: Convergence tolerance
        
    Returns:
        Shapley value array for all facts
    """
    n = len(all_facts)
    if n == 0:
        return np.array([])
        
    print(f" Calculate using Monte Carlo method{n} facts' Shapley values (all facts equal)...")
    print(f" All facts: {all_facts}")
    print(f" Atomic question: {atomic_question}")
    
    shapley_scores = np.zeros(n)
    shapley_scores_prev = np.zeros(n)

    def build_medical_prompt(selected_facts):
        """Build medical consultation prompt - do not use basic facts, purely based on selected facts"""
        info_text = '，'.join(selected_facts) + '.' if selected_facts else ''
        
        prompt = f"""You are a professional doctor with extensive medical knowledge. Please use your professional medical knowledge to directly give the correct answer:

Patient information: {info_text}

Question: {atomic_question}
answer:"""
        return prompt

    # Calculate empty set score (no facts)
    empty_prompt = build_medical_prompt([])
    v_empty, _ = get_answer_logprob(model, tokenizer, empty_prompt, target_answer)
    
    # calculate full score（All facts）
    full_prompt = build_medical_prompt(all_facts)
    full_score, _ = get_answer_logprob(model, tokenizer, full_prompt, target_answer)

    print(f" Empty set score (no facts): {v_empty:.4f}")
    print(f" full score（All facts）: {full_score:.4f}")

    # Monte Carlo sampling to calculate Shapley values
    for t in range(1, max_samples + 1):
        # Randomly arrange All facts（completely equal）
        perm = list(range(n))
        random.shuffle(perm)
        
        shapley_scores_prev = shapley_scores.copy()
        
        # Start from empty set and gradually add facts
        v_prev = v_empty
        active_facts = []

        for j, idx in enumerate(perm):
            # Add current fact
            active_facts.append(all_facts[idx])
            cur_prompt = build_medical_prompt(active_facts)
            v_j, _ = get_answer_logprob(model, tokenizer, cur_prompt, target_answer)

            # Calculate marginal contribution (completely based on current coalition vs previous coalition)
            marginal_contrib = v_j - v_prev
            phi_old = shapley_scores[idx]
            shapley_scores[idx] = (t - 1)/t * phi_old + (1/t) * marginal_contrib
            v_prev = v_j

            # Early stopping condition
            if abs(v_j - full_score) < 1e-2:
                break
        
        # Check convergence
        if t >= min_samples:
            shapley_diffs = np.abs(shapley_scores - shapley_scores_prev)
            avg_diff = np.mean(shapley_diffs)
            if avg_diff < tol:
                print(f" Shapley values converged at iteration {t}")
                break
    
    print(f"📈 Final Shapley values (equal calculation): {shapley_scores}")
    return shapley_scores


def normalize_shapley_weights(shapley_scores: np.ndarray, method: str = "softmax", temperature: float = 2.0) -> np.ndarray:
    """
    Normalize Shapley values to get weights, supporting multiple normalization methods
    
    Step 2:
    "Then normalize to get weights for each unknown information"
    
    Args:
        shapley_scores: Original Shapley values
        method: Normalization method:
                - "z_score": Z-score normalization, softmax based on standardized absolute values
                - "softmax": Softmax normalization, softmax based on original values
        temperature: Temperature parameter (only used in softmax method), controls sharpness of distribution
                    - temperature < 1: Make distribution sharper, highlight most important information
                    - temperature > 1: Make distribution smoother, weights more uniform
                    - temperature = 1: Standard softmax
        
    Returns:
        Normalized weights
    """
    if len(shapley_scores) == 0:
        return np.array([])
    
    # If there is only one Fact, Weight as 1
    if len(shapley_scores) == 1:
        weights = np.array([1.0])
        print(f" {method}normalization (single fact): {weights}")
        return weights
    
    # Handle case where all Shapley values are the same
    if np.std(shapley_scores) < 1e-8:
        # If all Shapley values are the same, use uniform weights
        weights = np.ones(len(shapley_scores)) / len(shapley_scores)
        print(f" {method}normalization (uniform distribution): {weights}")
        return weights
    
    try:
        if method == "z_score":
            # Z-score normalization method
            print(" Using Z-score normalization method...")
            
            # Step 1: Calculate Z-score standardization
            mean_score = np.mean(shapley_scores)
            std_score = np.std(shapley_scores)
            z_scores = (shapley_scores - mean_score) / std_score
            
            print(f"  Original Shapley values: {shapley_scores}")
            print(f"  Mean: {mean_score:.4f}, Standard deviation: {std_score:.4f}")
            print(f"  Z-score: {z_scores}")
            
            # Step 2: Take absolute value (importance regardless of sign)
            abs_z_scores = np.abs(z_scores)
            print(f"  |Z-score|: {abs_z_scores}")
            
            # Step 3: Softmax normalization on absolute values
            # For numerical stability, first subtract the maximum value
            shifted_abs_z = abs_z_scores - np.max(abs_z_scores)
            exp_scores = np.exp(shifted_abs_z)
            weights = exp_scores / np.sum(exp_scores)
            
            
            weights = weights / np.sum(weights)
            
            print(f" Z-scorenormalization result: {weights}")
            
        elif method == "softmax":
            # Softmax normalization method
            print(f" Using Softmax normalization method (temperature={temperature:.2f})...")
            
            # For numerical stability, first subtract the maximum value
            shifted_scores = shapley_scores - np.max(shapley_scores)
            
            # Apply temperature parameter and calculate exponential
            exp_scores = np.exp(shifted_scores / temperature)
            
            # Softmax normalization
            weights = exp_scores / np.sum(exp_scores)
            
            # Ensure Weight and as 1
            weights = weights / np.sum(weights)
            
            print(f" Softmaxnormalization result: {weights}")
            
        else:
            raise ValueError(f"Unsupported normalization method: {method}.Supported methods: 'z_score', 'softmax'")
        
        # Verify the validity of the Weight
        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            print(f" {method}calculation encountered numerical problems, fallback to uniform weights")
            weights = np.ones(len(shapley_scores)) / len(shapley_scores)
        
        return weights
        
    except (OverflowError, FloatingPointError, ZeroDivisionError) as e:
        print(f" {method}calculation error: {e}，fallback to uniform weights")
        weights = np.ones(len(shapley_scores)) / len(shapley_scores)
        return weights


def evaluate_weighted_fact_acquisition(
    model, tokenizer,
    all_facts: List[str],
    known_facts: List[str], 
    formatted_dialog: str,
    shapley_weights: np.ndarray
) -> float:
    """
        Process: First calculate Shapley values equally, then use the first 50% known facts and Shapley weighting in the reward stage
        
        Args:
            model: Doctor model
            tokenizer: Tokenizer  
            all_facts: All facts list
            known_facts: The first 50% known facts（used for generating doctor understanding in the reward stage）
            formatted_dialog: Formatted dialogue content
            shapley_weights: Shapley weight of all facts（calculated equally）
            
        Returns:
            Weighted reward score based on Shapley weights
    """
    print(f" Start calculating weighted Fact reward...")
    print(f" Total number of facts: {len(all_facts)}, Number of known facts: {len(known_facts)}")
    
    # Use the first 50% known facts as doctor's known information, generate doctor's understanding
    understanding_prompt = doctor_understanding_prompt.format(
        patient_information='，'.join(known_facts) + '.' if known_facts else '',
        dialogue=formatted_dialog
    )
    understanding_prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n" + understanding_prompt + "\n<|im_end|>\n<|im_start|>assistant\n"
    )
    inputs = tokenizer(understanding_prompt, return_tensors="pt").to(model.device)
    
    output = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id
    )
    
    context = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Statistics each Fact isAppears, then use Shapley weight to add weight
    print(" Check occurrence of each fact and perform Shapley weighting:")
    weighted_score = 0.0
    
    for i, fact in enumerate(all_facts):
        try:
            # Use fact checker to check if the Fact appears
            fact_checker_client = OpenAI(api_key="",
                                        base_url="")
            fact_checker_model = 'qwen2.5:72b'
            
            prompt = check_fact_prompt.format(context=context, fact=fact)
            fact_check_messages = [{"role": "user", "content": prompt}]
            ans = call_gpt(fact_checker_client, fact_checker_model, fact_check_messages)
            
            fact_appeared = 1.0 if "True" in ans else 0.0
            
        except Exception as e:
            print(f"Fact checking API failed: {e}, using string matching")
            fact_lower = fact.lower()
            context_lower = context.lower()
            
            if (fact_lower in context_lower or 
                any(key_word in context_lower for key_word in fact_lower.split() if len(key_word) > 2)):
                fact_appeared = 1.0
            else:
                fact_appeared = 0.0
        
        # Weight using Shapley weights
        weighted_contribution = shapley_weights[i] * fact_appeared
        weighted_score += weighted_contribution
        
        print(f"  📋 Fact {i+1}: {fact[:30]}...")
        print(f"    Appears: {'' if fact_appeared > 0 else ''} ({fact_appeared})")
        print(f"    Shapley weight: {shapley_weights[i]:.3f}")
        print(f"    Weighted contribution: {weighted_contribution:.3f}")
    
    final_score = weighted_score * 2  
    
    print(f" Total weighted score: {weighted_score:.3f}")
    print(f" Final score: {final_score:.3f}")
    print("=" * 60)
    
    return final_score


def compute_shapley_weighted_fact_score(
    model, tokenizer, fact_list, formatted_dialog, 
    atomic_question, target_answer, use_shapley=True
):
    """
        Change the old version of the overall evaluation points method: Calculate Shapley values for all facts and evaluate the overall points
        
        Complete process:
        1. Use the first 50% known facts as "doctor's known facts"
        2. Calculate Shapley values for all facts and add weight
        3. Evaluate the overall points for all facts
    """
    if not use_shapley:
        # Traditional mode: Completely follow the old version
        understanding_prompt = doctor_understanding_prompt.format(
            patient_information='，'.join(fact_list[:max(1, len(fact_list) // 2)]) + '.',
            dialogue=formatted_dialog
        )
        understanding_prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n" + understanding_prompt + "\n<|im_end|>\n<|im_start|>assistant\n"
        )
        inputs = tokenizer(understanding_prompt, return_tensors="pt").to(model.device)
        
        output = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id
        )
        
        context = tokenizer.decode(output[0], skip_special_tokens=True)
        score = get_fact_score(fact_list, context)
        return score * 3
    
    else:
        # Shapley overall scoring mode: Change the old version approach, calculate all
        split_point = max(1, len(fact_list) // 2)
        known_facts = fact_list[:split_point]  # The first 50% known facts as doctor's known facts
        
        print(f" Shapley overall scoring mode (old version approach)")
        print(f" Total number of facts: {len(fact_list)}, 50% known facts: {len(known_facts)}")
        
        try:
            # Step 1: Calculate Shapley values for all facts equally（Monte Carlo Method）
            print(" Start executing Shapley process...")
            print(" Step 1: Calculate Shapley values for all facts equally")
            shapley_scores = compute_equal_shapley_values(
                model, tokenizer,
                fact_list,  
                atomic_question, target_answer,
                max_samples=50,
                min_samples=3
            )
            
            # Step 2: Normalize to get weights
            print(" Step 2: Normalize to get weights")
            shapley_weights = normalize_shapley_weights(shapley_scores, method="softmax", temperature=2.0)
            
            # Step 3: Multi-round dialogue completed externally(formatted_dialog)
            print(" Step 3: Multi-round dialogue completed")
            
            # Step 4&5: Statistics FactAppears and add weight based on Shapley weights
            print(" Step 4&5: Statistics FactAppears and add weight based on Shapley weights")
            final_score = evaluate_weighted_fact_acquisition(
                model, tokenizer,
                fact_list, known_facts,  # All facts + the first 50% known facts
                formatted_dialog, shapley_weights
            )
            
            return final_score
            
        except Exception as e:
            print(f" Shapley value calculation error: {e}, fallback to traditional mode")
            # When an error occurs, fallback to traditional mode
            return compute_shapley_weighted_fact_score(
                model, tokenizer, fact_list, formatted_dialog, 
                atomic_question, target_answer, use_shapley=False
            )


def fact_score_reward(model, tokenizer, facts: List[List[str]], 
                     completions: List[List[Dict[str, Any]]],
                     use_shapley: bool = False,
                     atomic_questions: List[str] = None,
                     target_answers: List[str] = None) -> List[float]:
    """
    Calculate the reward score based on Facts, support Traditional mode and Shapley value weighted mode
    
    Args:
        model: Policy model
        tokenizer: Tokenizer  
        facts: Fact list
        completions: Model completed dialogue
        use_shapley: Whether to use Shapley value weighted mode
        atomic_questions: Atomic question list（Shapley mode required）
        target_answers: Target answer list（Shapley mode required）
    
    Returns:
        Fact score reward list
    """
    all_rewards = []

    for i in range(len(completions)):
        fact_list = facts[i]
        completion = completions[i][0]["content"]
        dialog = parse_dialog(completion)
        formatted_dialog = format_dialog(dialog)

        if use_shapley and atomic_questions and target_answers:
            # Using Shapley value weighted mode
            atomic_question = atomic_questions[i]
            target_answer = target_answers[i]
            
            score = compute_shapley_weighted_fact_score(
                model, tokenizer, fact_list, formatted_dialog,
                atomic_question, target_answer, use_shapley=True
            )
        else:
            # Traditional mode
            score = compute_shapley_weighted_fact_score(
                model, tokenizer, fact_list, formatted_dialog,
                "", "", use_shapley=False
            )
        
        all_rewards.append(score)

    return all_rewards




def overall_reward(model, tokenizer, facts: List[str], 
                  completions: List[List[Dict[str, Any]]],
                  options: List[Dict[str, str]], answers: List[str],
                  use_shapley: bool = False,
                  atomic_questions: List[str] = None) -> Dict[str, List[float]]:
    """
    Combine correctness, format, and Fact score reward comprehensive evaluation points
    
    Args:
        model: Policy model
        tokenizer: Tokenizer
        facts: Fact list
        completions: Model completed dialogue
        options: List of option dictionaries
        answers: answer list  
        use_shapley: Whether to use Shapley value weighted mode for fact score
        atomic_questions: Atomic question list
    Returns:
        Dictionary containing various scores
    """
    # Parameter validation
    n = len(facts)
    if not (n == len(completions) == len(answers)):
        raise ValueError("facts, completions, and answers must have the same length.")

    correctness_scores = correctness_reward(completions, options, answers)
    format_scores = format_reward(completions)
    
    # Build target answer（used for Shapley value calculation）
    target_answers = None
    if use_shapley and atomic_questions:
        target_answers = []
        for i, answer_keys in enumerate(answers):
            target_answer = '，'.join([options[i][c] for c in answer_keys]).strip('，')
            target_answers.append(target_answer)
    
    fact_scores = fact_score_reward(
        model, tokenizer, facts, completions,
        use_shapley=use_shapley,
        atomic_questions=atomic_questions,
        target_answers=target_answers
    )

    total_scores: List[float] = [c + f + r for c, f, r in 
                               zip(correctness_scores, format_scores, fact_scores)]

    return {
        "total_scores": total_scores,
        "correctness_scores": correctness_scores,
        "format_scores": format_scores,
        "fact_scores": fact_scores,
    }


# ================ Token-level reward allocation system ================

def extract_question_boundaries(completion_text: str, tokenizer) -> List[
    Tuple[int, int]
]:
    """
    Identify token boundaries for each question in the dialogue
    
    Args:
        completion_text: Complete dialogue text
        tokenizer: Tokenizer
        
    Returns:
        List[Tuple[int, int]]: Each question's (start_token_idx, end_token_idx)
    """
    # Parse dialogue
    dialog = parse_dialog(completion_text)
    
    boundaries = []
    current_pos = 0
    
    for turn in dialog:
        if turn['role'] == 'assistant':  # Doctor's question
            turn_text = turn['content']
            
            # Tokenize and find position in complete text
            turn_tokens = tokenizer.encode(
                turn_text, add_special_tokens=False
            )
            turn_length = len(turn_tokens)
            
            # Find corresponding position in complete text
            full_tokens = tokenizer.encode(
                completion_text, add_special_tokens=False
            )
            
            # Search for matching token sequence
            for i in range(current_pos, len(full_tokens) - turn_length + 1):
                if full_tokens[i:i + turn_length] == turn_tokens:
                    boundaries.append((i, i + turn_length))
                    current_pos = i + turn_length
                    break
    
    return boundaries


def compute_question_shapley_gains(
    model, tokenizer, fact_list: List[str], dialog: List[Dict[str, str]], 
    atomic_question: str, target_answer: str, shapley_weights: np.ndarray
) -> List[float]:
    """
    Calculate Shapley information gain brought by each question
    
    Args:
        model: Doctor model
        tokenizer: Tokenizer
        fact_list: Complete Fact list
        dialog: Parsed dialogue
        atomic_question: Atomic question
        target_answer: Target answer
        shapley_weights: Pre-calculated Shapley weight
        
    Returns:
        List[float]: Shapley gain for each question
    """
    question_gains = []
    
    # Create Fact checker - keep using external API
    fact_checker_client = OpenAI(api_key="",
                                base_url="")
    fact_checker_model = 'qwen2.5:72b'
    
    print(f" ===== Shapley gain calculation debug info =====")
    print(f" Input parameters:")
    print(f"  - Fact list: {len(fact_list)}Fact")
    print(f"  - Number of dialogue turns: {len(dialog)} turns")
    print(f"  - Shapley weight: {shapley_weights}")
    print(f"  - Atomic question: '{atomic_question}'")
    print(f"  - target answer: '{target_answer}'")
    
    # Print Fact list details
    for i, fact in enumerate(fact_list):
        print(f"  Fact{i}: '{fact}' (Weight: {shapley_weights[i] if i < len(shapley_weights) else 'N/A'})")
    
    assistant_count = 0
    for turn_idx, turn in enumerate(dialog):
        if turn['role'] == 'assistant':  # Doctor's question
            assistant_count += 1
            question_gain = 0.0
            turn_content = turn['content']
            
            print(f"\n Processing Assistant turn {assistant_count}:")
            print(f"  Turn content: '{turn_content[:200]}...'")
            
            # Check which Facts are obtained for this question
            acquired_facts = []
            api_success_count = 0
            api_fail_count = 0
            
            for i, fact in enumerate(fact_list):
                fact_acquired = False
                method_used = ""
                
                try:
                    # Check if the Fact is obtained in this turn's dialogue
                    prompt = check_fact_prompt.format(context=turn_content, fact=fact)
                    fact_check_messages = [{"role": "user", "content": prompt}]
                    
                    print(f"Check Fact{i}: '{fact[:50]}...'")
                    print(f"Fact Check Prompt: '{prompt[:100]}...'")
                    
                    ans = call_gpt(fact_checker_client, fact_checker_model, fact_check_messages)
                    api_success_count += 1
                    method_used = "API"
                    
                    print(f"API call successful, response: '{ans[:100]}...'")
                    
                    if "True" in ans:
                        fact_acquired = True
                        print(f"API confirmed Fact was obtained")
                    else:
                        print(f"API confirmed Fact was not obtained")
                        
                except Exception as e:
                    api_fail_count += 1
                    method_used = "String matching"
                    print(f"API call failed: {str(e)[:200]}")
                    
                    # Fallback to improved String matching
                    fact_lower = fact.lower()
                    turn_lower = turn_content.lower()
                    
                    # Improved String matching: Check Keywords
                    fact_keywords = [word for word in fact_lower.split() if len(word) > 2]
                    if len(fact_keywords) > 0:
                        match_count = sum(1 for keyword in fact_keywords if keyword in turn_lower)
                        match_ratio = match_count / len(fact_keywords)
                        
                        print(f"String matching points析:")
                        print(f"Keywords: {fact_keywords}")
                        print(f"Match count: {match_count}/{len(fact_keywords)}")
                        print(f"Match ratio: {match_ratio:.2f}")
                        
                        if match_ratio >= 0.5: 
                            fact_acquired = True
                            print(f"String matching confirmed Fact was obtained")
                        else:
                            print(f"String matching confirmed Fact was not obtained")
                    else:
                        print(f"Fact has no valid Keywords, default not obtained")
                
                # Accumulate Shapley value
                if fact_acquired:
                    if i < len(shapley_weights):
                        shapley_contribution = shapley_weights[i]
                        question_gain += shapley_contribution
                        acquired_facts.append({
                            'index': i,
                            'fact': fact[:50] + '...' if len(fact) > 50 else fact,
                            'weight': shapley_contribution,
                            'method': method_used
                        })
                        print(f" Accumulate Shapley value: {shapley_contribution:.4f}")
                    else:
                        print(f"Shapley weight index out of range: {i} >= {len(shapley_weights)}")
                else:
                    print(f"Fact not obtained, Shapley value as 0")
            
            print(f"\n Assistant turn {assistant_count} Statistics:")
            print(f" Successful API calls: {api_success_count} times")
            print(f" API failure fallbacks: {api_fail_count} times")
            print(f" Fact count obtained: {len(acquired_facts)}")
            print(f" Total Shapley gain: {question_gain:.4f}")
            
            if acquired_facts:
                print(f" Obtained Fact details:")
                for fact_info in acquired_facts:
                    print(f" Fact{fact_info['index']}: {fact_info['fact']} (Weight:{fact_info['weight']:.4f}, Method:{fact_info['method']})")
            else:
                print(f" No Fact obtained in this turn!")
            
            question_gains.append(question_gain)
    
    return question_gains


def compute_token_level_rewards(
    model, tokenizer, facts: List[str], 
    completions: List[List[Dict[str, Any]]],
    options: List[Dict[str, str]], answers: List[str],
    use_shapley: bool = True,
    atomic_questions: List[str] = None,
    alpha: float = 1.0,  # Question Shapley reward Weight
    beta: float = 1.0,   # Question result reward Weight  
    gamma: float = 3.0,  # Answer correctness reward Weight
    format_reward_weight: float = 1.0,  # Format reward Weight
    max_completion_length: int = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Calculate pure token-level reward allocation - integrate format and content rewards
    
    New reward system (max 4 points)::
    1. Question tokens: Shapley reward (0-3 points) + format reward (0-1 point) = max 4 points
    2. Answer tokens: correctness reward (0-3 points) + format reward (0-1 point) = max 4 points  
    3. Other tokens: 0 points
    
    Format reward rules:
    - Sentences starting with "question:", all tokens get 1 point format reward
    - Sentences starting with "answer:", all tokens get 1 point format reward
    - Other tokens get 0 points format reward
    
    Args:
        format_reward_weight: Format reward Weight（default 1.0）
        max_completion_length: Maximum completion length
    
    Returns:
        Dictcontains:
        - token_rewards: List[List[float]] - Token-level reward for each sample
        - question_token_rewards: List[float] - Question token reward Mean
        - answer_token_rewards: List[float] - Answer token reward Mean
        - format_token_rewards: List[float] - Format reward Mean
        - token_rewards_mean: List[float] - Total token reward Mean
    """
    print(" ===== Pure token-level reward allocation (integrate format rewards) =====")
    print(f" Input parameters: max_completion_length={max_completion_length}")
    print(f" completionscount: {len(completions)}")
    print(f" Reward Weight: alpha={alpha}, beta={beta}, gamma={gamma}, format_weight={format_reward_weight}")
    
    try:
        token_rewards_list = []
        question_token_rewards_list = [] 
        answer_token_rewards_list = []    
        format_token_rewards_list = []   
        token_rewards_mean_list = []     
        
        for i, completion_list in enumerate(completions):
            print(f" Processing completion group {i}: contains{len(completion_list)}samples")
            for j, completion_dict in enumerate(completion_list):
                # Get Complete dialogue text
                completion_text = completion_dict.get('content', '')
                print(f" Sample {i}-{j}: Original text length={len(completion_text)}characters")
                
                # Calculate answer correctness directly（without relying on base_rewards）
                last_response = completion_text.split("<|im_start|>assistant")[-1].strip()
                # Compatible handling of options field (unified use of options)
                current_options = options[i] if i < len(options) else {}
                model_answer = match_choice(last_response, current_options)
                correct_answer = answers[i].strip() if i < len(answers) else ""
                answer_correct = 1.0 if model_answer == correct_answer else 0.0
                
                print(f" Sample {i}-{j}: Model answer='{model_answer}', correct answer='{correct_answer}', answer_correct={answer_correct}")
                
                # Tokenize text and apply truncation
                tokens = tokenizer.encode(completion_text, add_special_tokens=False)
                original_token_length = len(tokens)
                
                if max_completion_length is not None and len(tokens) > max_completion_length:
                    tokens = tokens[:max_completion_length]
                    completion_text = tokenizer.decode(tokens, skip_special_tokens=False)
                    print(f" Sample {i}-{j}: Token sequence from{original_token_length}truncated to{len(tokens)}")
                else:
                    print(f" Sample {i}-{j}: No truncation needed, keep length={len(tokens)}")
                
                # Initialize token rewards and Statistics variables
                token_rewards = [0.0] * len(tokens)
                question_tokens_rewards = []  
                answer_tokens_rewards = []     
                format_tokens_rewards = []    
                
                # Step 1: Calculate format reward
                # Check if each sentence starts with "question:" or "answer:", if so, all tokens in the sentence get format reward
                format_question_boundaries, format_answer_boundaries = extract_format_boundaries(completion_text, tokenizer)
                
                print(f" Sample {i}-{j}: Format check - {len(format_question_boundaries)}question format, {len(format_answer_boundaries)}answer format")
                
                # Assign format reward to all tokens in the question sentence with correct format
                for start_idx, end_idx, base_reward in format_question_boundaries:
                    for token_idx in range(start_idx, min(end_idx, len(token_rewards))):
                        actual_reward = base_reward * format_reward_weight
                        token_rewards[token_idx] += actual_reward
                        format_tokens_rewards.append(actual_reward)
                        print(f" Question format reward: token[{token_idx}] += {actual_reward:.3f} (base={base_reward}, weight={format_reward_weight})")
                
                # Assign format reward to all tokens in the answer sentence with correct format
                for start_idx, end_idx, base_reward in format_answer_boundaries:
                    for token_idx in range(start_idx, min(end_idx, len(token_rewards))):
                        actual_reward = base_reward * format_reward_weight
                        token_rewards[token_idx] += actual_reward
                        format_tokens_rewards.append(actual_reward)
                        print(f" Answer format reward: token[{token_idx}] += {actual_reward:.3f} (base={base_reward}, weight={format_reward_weight})")
                
                # Step 2: Identify content boundaries（all questions and answers, regardless of format）
                question_boundaries, answer_boundaries = extract_question_answer_boundaries(completion_text, tokenizer)
                
                print(f" Sample {i}-{j}: Content identification - {len(question_boundaries)}, {len(answer_boundaries)}answer")
                
                # Step 3: Calculate QuestionContent reward（using Shapley values）
                question_gains = []
                
                print(f" Sample {i}-{j}: Step 3 - Start calculating question content rewards")
                print(f"  Parameter check: use_shapley={use_shapley}, atomic_questionsexists={atomic_questions is not None}, i<len(facts)={i < len(facts) if facts else False}")
                
                if use_shapley and atomic_questions and i < len(facts):
                    fact_list = facts[i]
                    base_idx = i * len(completion_list) + j
                    atomic_question = atomic_questions[base_idx] if base_idx < len(atomic_questions) else "Default question"
                    
                    print(f" Sample {i}-{j}: Start calculating Shapley gain for each question...")
                    
                    try:
                        print(f" Sample {i}-{j}: Start complete Shapley process...")
                        
                        # Parse dialogue
                        dialog = parse_dialog(completion_text)
                        print(f"   Dialogue parsing completed: {len(dialog)}  turns dialogue")
                        
                        # Calculate overall Shapley weight
                        print(f"   Start calculating Shapley values...")
                        shapley_scores = compute_equal_shapley_values(
                            model, tokenizer, fact_list, atomic_question, 
                            answers[i] if i < len(answers) else "Default answer",
                            max_samples=50, min_samples=3
                        )
                        print(f"   Shapley value calculation completed: {shapley_scores}")
                        
                        shapley_weights = normalize_shapley_weights(shapley_scores, method="softmax", temperature=2.0)
                        print(f"   Shapley weight normalization completed: {shapley_weights}")
                        
                        # Calculate Shapley gain for each question
                        print(f"   Start calculating Shapley gain for each question...")
                        question_gains = compute_question_shapley_gains(
                            model, tokenizer, fact_list, dialog, atomic_question,
                            answers[i] if i < len(answers) else "Default answer", shapley_weights
                        )
                        
                        print(f" Sample {i}-{j}: Question Shapley gains: {question_gains}")
                        
                    except Exception as e:
                        print(f" Sample {i}-{j}: Shapley calculation failed!")
                        print(f"   Error type: {type(e).__name__}")
                        print(f"   Error message: {str(e)}")
                        import traceback
                        print(f"   Error stack: {traceback.format_exc()}")
                        print(f"   Fallback to default gain 1.0")
                        question_gains = [1.0] * len(question_boundaries)
                        
                else:
                    if i < len(facts):
                        fact_list = facts[i]
                        dialog = parse_dialog(completion_text)
                        
                        print(f" Sample {i}-{j}: Using uniform Weight mode")
                        print(f"  Total number of facts: {len(fact_list)}")
                        print(f"  Question count: {len(question_boundaries)}")
                        
                        # Assign uniform Weight:1/Total number of facts to each question
                        uniform_gain_per_question = 1.0 / len(fact_list) if len(fact_list) > 0 else 1.0
                        question_gains = [uniform_gain_per_question] * len(question_boundaries)
                        
                        print(f" Sample {i}-{j}: Uniform gain allocation: {uniform_gain_per_question:.4f} * {len(question_boundaries)} = {question_gains}")
                    else:
                        # Fallback to default value
                        question_gains = [1.0] * len(question_boundaries)
                        print(f" Sample {i}-{j}: No Fact data, using default gain 1.0")
                
                # Step 4: Assign Content reward to Question tokens
                for q_idx, (start_idx, end_idx) in enumerate(question_boundaries):
                    if q_idx < len(question_gains):
                        shapley_gain = question_gains[q_idx]
                        
                        # QuestionContent reward: process reward + result reward（if answer correct）
                        question_content_reward = alpha * shapley_gain + beta * shapley_gain * answer_correct
                        
                        print(f" Sample {i}-{j} Question {q_idx}: Shapley gain={shapley_gain:.3f}, Content reward={question_content_reward:.3f}")
                        
                        for token_idx in range(start_idx, min(end_idx, len(token_rewards))):
                            token_rewards[token_idx] += question_content_reward
                            question_tokens_rewards.append(token_rewards[token_idx])  
                            print(f" Question token[{token_idx}]: Total reward={token_rewards[token_idx]:.3f}")
                
                # Step 5: Assign Content reward to Answer tokens
                for start_idx, end_idx in answer_boundaries:
                    answer_content_reward = gamma * answer_correct
                    print(f" Sample {i}-{j}: answerContent reward={answer_content_reward:.3f}")
                    
                    for token_idx in range(start_idx, min(end_idx, len(token_rewards))):
                        token_rewards[token_idx] += answer_content_reward
                        answer_tokens_rewards.append(token_rewards[token_idx])  
                        print(f"  Answer token[{token_idx}]: Total reward={token_rewards[token_idx]:.3f}")
                
                # Step 6: Statistics the Mean of various rewards for the current sample
                sample_question_mean = sum(question_tokens_rewards) / len(question_tokens_rewards) if question_tokens_rewards else 0.0
                sample_answer_mean = sum(answer_tokens_rewards) / len(answer_tokens_rewards) if answer_tokens_rewards else 0.0
                sample_format_mean = sum(format_tokens_rewards) / len(format_tokens_rewards) if format_tokens_rewards else 0.0
                sample_total_mean = sum(token_rewards) / len(token_rewards) if token_rewards else 0.0
                
                question_token_rewards_list.append(sample_question_mean)
                answer_token_rewards_list.append(sample_answer_mean)
                format_token_rewards_list.append(sample_format_mean)
                token_rewards_mean_list.append(sample_total_mean)
                
                token_rewards_list.append(token_rewards)
      
        # Return detailed Statistics results
        result = {
            'token_rewards': token_rewards_list,
            'question_token_rewards': question_token_rewards_list,
            'answer_token_rewards': answer_token_rewards_list,
            'format_token_rewards': format_token_rewards_list,
            'token_rewards_mean': token_rewards_mean_list
        }
        
        print(f" Statistics result:")
        print(f"   Question tokensMean: {sum(question_token_rewards_list)/len(question_token_rewards_list) if question_token_rewards_list else 0:.3f}")
        print(f"   Answer tokensMean: {sum(answer_token_rewards_list)/len(answer_token_rewards_list) if answer_token_rewards_list else 0:.3f}")
        print(f"   Format reward Mean: {sum(format_token_rewards_list)/len(format_token_rewards_list) if format_token_rewards_list else 0:.3f}")
        print(f"   Total token Mean: {sum(token_rewards_mean_list)/len(token_rewards_mean_list) if token_rewards_mean_list else 0:.3f}")
        
        return result
        
    except Exception as e:
        print(f" Token-level reward calculation failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Return empty token rewards
        total_samples = sum(len(comp_list) for comp_list in completions)
        return {
            'token_rewards': [[] for _ in range(total_samples)],
            'question_token_rewards': [0.0] * total_samples,
            'answer_token_rewards': [0.0] * total_samples,
            'format_token_rewards': [0.0] * total_samples,
            'token_rewards_mean': [0.0] * total_samples
        }


def extract_question_answer_boundaries(completion_text: str, tokenizer) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Identify the token boundaries of each "question:" and "answer:" in the dialogue
    
    Args:
        completion_text: Complete dialogue text  
        tokenizer: Tokenizer
        
    Returns:
        Tuple[question_boundaries, answer_boundaries]: 
            - question_boundaries: List[Tuple[int, int]] - Each question's (start_token_idx, end_token_idx)
            - answer_boundaries: List[Tuple[int, int]] - Each answer's (start_token_idx, end_token_idx)
    """
    print(f" Start identifying question and answer boundaries...")
    
    # Tokenize complete text
    full_tokens = tokenizer.encode(completion_text, add_special_tokens=False)
    full_text = tokenizer.decode(full_tokens, skip_special_tokens=False)
    
    print(f" Complete text content preview: {full_text[:200]}...")
    
    question_boundaries = []
    answer_boundaries = []
    
    # Use regex to find all question: and answer: positions
    question_pattern = r'question\s*:'
    answer_pattern = r'answer\s*:'
    
    question_matches = list(re.finditer(question_pattern, full_text, re.IGNORECASE))
    answer_matches = list(re.finditer(answer_pattern, full_text, re.IGNORECASE))
    
    print(f" Found{len(question_matches)}question, {len(answer_matches)}answer")
    
    # For each question, find its token boundaries
    for i, match in enumerate(question_matches):
        start_char = match.start()
        
        # Fix: Find the end position of question, should be the next <|im_start|> or answer marker
        end_char = len(full_text)
        
        # First find the next <|im_start|> marker (user reply start)
        next_im_start_pos = full_text.find('<|im_start|>', start_char + 1)
        if next_im_start_pos != -1:
            end_char = next_im_start_pos
            print(f"Found next <|im_start|> marker position: {next_im_start_pos}")
        
        # Then find the next answer marker, take the closer one
        for next_answer_match in answer_matches:
            if next_answer_match.start() > start_char:
                if next_answer_match.start() < end_char:
                    end_char = next_answer_match.start()
                    print(f"Found closer answer marker position: {next_answer_match.start()}")
                break
        
        # Convert characters positions to token positions
        prefix_text = full_text[:start_char]
        question_text = full_text[start_char:end_char]
        
        prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
        question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
        
        start_token = len(prefix_tokens)
        end_token = start_token + len(question_tokens)
        
        question_boundaries.append((start_token, end_token))
        print(f" Question {i}: characters[{start_char}:{end_char}] -> token[{start_token}:{end_token}]")
        print(f"   Question content preview: '{question_text[:100]}...'")
    
    # For each answer, find its token boundaries
    for i, match in enumerate(answer_matches):
        start_char = match.start()
        
        # Find the end position of answer, is the next <|im_start|> or question marker
        end_char = len(full_text)
        
        # First find the next <|im_start|> marker
        next_im_start_pos = full_text.find('<|im_start|>', start_char + 1)
        if next_im_start_pos != -1:
            end_char = next_im_start_pos
            print(f" Found next <|im_start|> marker position: {next_im_start_pos}")
        
        # Then find the next question marker, take the closer one
        for next_question_match in question_matches:
            if next_question_match.start() > start_char:
                if next_question_match.start() < end_char:
                    end_char = next_question_match.start()
                    print(f" Found closer question marker position: {next_question_match.start()}")
                break
        
        # Convert characters positions to token positions
        prefix_text = full_text[:start_char]
        answer_text = full_text[start_char:end_char]
        
        prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
        answer_tokens = tokenizer.encode(answer_text, add_special_tokens=False)
        
        start_token = len(prefix_tokens)
        end_token = start_token + len(answer_tokens)
        
        answer_boundaries.append((start_token, end_token))
        print(f" Answer {i}: characters[{start_char}:{end_char}] -> token[{start_token}:{end_token}]")
        print(f"   answerContent preview: '{answer_text[:100]}...'")
    
    #  Verify correctness of boundaries
    print(f" Boundary verification:")
    for i, (start_token, end_token) in enumerate(question_boundaries):
        if start_token < len(full_tokens) and end_token <= len(full_tokens):
            question_tokens = full_tokens[start_token:end_token]
            question_text = tokenizer.decode(question_tokens, skip_special_tokens=False)
            print(f"   Question {i} tokens[{start_token}:{end_token}]: '{question_text[:50]}...'")
            
            # Check if contains<|im_start|> marker
            if '<|im_start|>' in question_text:
                print(f"    Question {i} contains<|im_start|> marker, boundary may be incorrect!")
        else:
            print(f"    Question {i} Boundary out of range: [{start_token}:{end_token}] vs {len(full_tokens)}")
    
    for i, (start_token, end_token) in enumerate(answer_boundaries):
        if start_token < len(full_tokens) and end_token <= len(full_tokens):
            answer_tokens = full_tokens[start_token:end_token]
            answer_text = tokenizer.decode(answer_tokens, skip_special_tokens=False)
            print(f"   Answer {i} tokens[{start_token}:{end_token}]: '{answer_text[:50]}...'")
            
            # Check if contains<|im_start|> marker
            if '<|im_start|>' in answer_text:
                print(f"    Answer {i} contains<|im_start|> marker, boundary may be incorrect!")
        else:
            print(f"    Answer {i} Boundary out of range: [{start_token}:{end_token}] vs {len(full_tokens)}")
    
    return question_boundaries, answer_boundaries


def extract_format_boundaries(completion_text: str, tokenizer) -> Tuple[List[Tuple[int, int, float]], List[Tuple[int, int, float]]]:
    """
    Identify question and answer boundaries for format rewards, supporting graded rewards
    
    Format reward rules:
    - Starting with "question:": 1.0 points
    - contains"question:"but not at the beginning: 0.5 points  
    - Starting with "answer:": 1.0 points
    - contains"answer:"but not at the beginning: 0.5 points
    - No markers: 0.0 points
    
    Args:
        completion_text: Complete dialogue text
        tokenizer: Tokenizer
        
    Returns:
        Tuple[format_question_boundaries, format_answer_boundaries]:
        - format_question_boundaries: List[Tuple[int, int, float]] - (start_token, end_token, reward_score)
        - format_answer_boundaries: List[Tuple[int, int, float]] - (start_token, end_token, reward_score)
    """
    
    # Tokenize complete text
    full_tokens = tokenizer.encode(completion_text, add_special_tokens=False)
    full_text = tokenizer.decode(full_tokens, skip_special_tokens=False)
    
    format_question_boundaries = []
    format_answer_boundaries = []
    
    # Strategy: Analyze based on assistant turns
    # Found all assistant turns' start and end positions
    assistant_pattern = r'<\|im_start\|>assistant(.*?)(?=<\|im_start\||$)'
    assistant_matches = list(re.finditer(assistant_pattern, full_text, re.DOTALL))
    
    print(f" Found{len(assistant_matches)}assistant turns")
    
    for i, assistant_match in enumerate(assistant_matches):
        assistant_content = assistant_match.group(1).strip()
        assistant_start_char = assistant_match.start() + len('<|im_start|>assistant')
        assistant_end_char = assistant_match.end()
        
        print(f" Assistant turn {i}: Content preview='{assistant_content[:50]}...'")
        
        # Check question markers
        question_reward = 0.0
        if assistant_content.startswith('question:'):
            question_reward = 1.0
            print(f"  Starts with question:, reward=1.0")
        elif 'question:' in assistant_content:
            question_reward = 0.5
            print(f"   containsquestion: but not at the beginning, reward=0.5")
        else:
            print(f"   No question marker, reward=0.0")
        
        # Check answer markers  
        answer_reward = 0.0
        if assistant_content.startswith('answer:'):
            answer_reward = 1.0
            print(f"   Starts with answer:, reward=1.0")
        elif 'answer:' in assistant_content:
            answer_reward = 0.5
            print(f"   containsanswer: but not at the beginning, reward=0.5")
        else:
            print(f"   No answer marker, reward=0.0")
        
        # Convert to token positions
        if question_reward > 0.0 or answer_reward > 0.0:
            # Calculate token boundaries for assistant turn
            prefix_text = full_text[:assistant_start_char]
            assistant_text = full_text[assistant_start_char:assistant_end_char]
            
            prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
            assistant_tokens = tokenizer.encode(assistant_text, add_special_tokens=False)
            
            start_token = len(prefix_tokens)
            end_token = start_token + len(assistant_tokens)
            
            # Add to corresponding list based on content type
            if question_reward > 0.0:
                format_question_boundaries.append((start_token, end_token, question_reward))
                print(f"   Question format boundary: token[{start_token}:{end_token}], reward={question_reward}")
            
            if answer_reward > 0.0:
                format_answer_boundaries.append((start_token, end_token, answer_reward))
                print(f"   Answer format boundary: token[{start_token}:{end_token}], reward={answer_reward}")
    
    print(f" Format reward Statistics: {len(format_question_boundaries)}question boundaries, {len(format_answer_boundaries)}answer boundaries")
    return format_question_boundaries, format_answer_boundaries


def overall_reward_with_token_allocation(
    model, tokenizer, facts: List[str], 
    completions: List[List[Dict[str, Any]]],
    options: List[Dict[str, str]], answers: List[str],
    use_shapley: bool = False,
    atomic_questions: List[str] = None,
    use_token_level: bool = False,  
    max_completion_length: int = None,  
    **kwargs
) -> Dict[str, List[float]]:
    """
    Enhanced overall_reward, supporting token-level rewards allocation
    
    Args:
        use_token_level: Whether to use token-level rewards allocation
        Other parameters same as original overall_reward
        
    Returns:
        If use_token_level=True, return detailed results containing token_rewards
        Otherwise return standard format to maintain compatibility
    """
    if use_token_level:
        # Use new token-level reward calculation
        return compute_token_level_rewards(
            model, tokenizer, facts, completions, options, answers,
            use_shapley=use_shapley, atomic_questions=atomic_questions, 
            max_completion_length=max_completion_length, **kwargs
        )
    else:
        # Fallback to original overall_reward, not using token-level rewards
        return overall_reward(
            model, tokenizer, facts, completions, options, answers,
            use_shapley=use_shapley, atomic_questions=atomic_questions
        )

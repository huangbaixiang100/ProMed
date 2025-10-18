import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from typing import List
import numpy as np
import random
import os

# load model
model_name = ""
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
model.eval()


@torch.no_grad()
def get_answer_logprob(model, tokenizer, full_prompt, target_answer, past_key_values=None):
    """
    Compute the log-probability of the target answer given the prompt.
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


@torch.no_grad()
def get_answer_logprob_v2(model, tokenizer, full_prompt, target_answer):
    input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(model.device)
    target_ids = tokenizer(target_answer, return_tensors="pt").input_ids.to(model.device)

    full_input_ids = torch.cat([input_ids, target_ids], dim=-1)

    labels = full_input_ids.clone()
    # Mask the input
    labels[:, :input_ids.size(-1)] = -100

    outputs = model(input_ids=full_input_ids, labels=labels)
    loss = outputs.loss  # CrossEntropyLoss
    avg_logprob = -loss.item()

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
    Args:
    model: doctor model
    tokenizer: tokenizer
    all_facts: list of all facts (all facts are treated equally)
    atomic_question: atomic question
    target_answer: target answer
    max_samples: maximum number of samples
    min_samples: minimum number of samples
    tol: convergence tolerance

    Returns:
        Array of Shapley values for all facts
    """
    n = len(all_facts)
    if n == 0:
        return np.array([])

    print(f"Computing Shapley values for {n} facts using the Monte Carlo method (all facts treated equally)...")
    print(f"All facts: {all_facts}")
    print(f"Atomic question: {atomic_question}")
    print(f"Target answer: {target_answer}")

    shapley_scores = np.zeros(n)
    shapley_scores_prev = np.zeros(n)

    def build_medical_prompt(selected_facts):
        info_text = '，'.join(selected_facts) + '。' if selected_facts else ''

        prompt = f"""你是一名专业的医生，具备丰富的医疗知识。请根据以下患者信息回答问题：

患者信息：{info_text}

问题：{atomic_question}
答案："""

        return prompt

    # prob with zero fact
    empty_prompt = build_medical_prompt([])
    v_empty, _ = get_answer_logprob_v2(model, tokenizer, empty_prompt, target_answer)

    # prob with full facts
    full_prompt = build_medical_prompt(all_facts)
    full_score, _ = get_answer_logprob_v2(model, tokenizer, full_prompt, target_answer)


    # Monte Carlo Shapley
    for t in range(1, max_samples + 1):
        # randomly sample a permutation of all facts
        perm = list(range(n))
        random.shuffle(perm)

        shapley_scores_prev = shapley_scores.copy()

        v_prev = v_empty
        active_facts = []

        for j, idx in enumerate(perm):
            # add current fact
            active_facts.append(all_facts[idx])
            cur_prompt = build_medical_prompt(active_facts)
            v_j, _ = get_answer_logprob_v2(model, tokenizer, cur_prompt, target_answer)

            # calculate marginal contribution
            marginal_contrib = v_j - v_prev
            phi_old = shapley_scores[idx]
            shapley_scores[idx] = (t - 1) / t * phi_old + (1 / t) * marginal_contrib
            v_prev = v_j

            # early stop
            if abs(v_j - full_score) < 1e-2:
                break

        if t >= min_samples:
            shapley_diffs = np.abs(shapley_scores - shapley_scores_prev)
            avg_diff = np.mean(shapley_diffs)
            if avg_diff < tol:
                print(f"Shapley scores converge at {t} iterations")
                break

    print(f"Final Shapley scores: {shapley_scores}")
    return shapley_scores


input_path = ("../data/dataset/cmb_atomic_patient_example.json")
output_path = "softmax_shapley.jsonl"


with open(input_path, "r", encoding="utf-8") as f:
    data_list = json.load(f)


existing_ids = set()
if os.path.exists(output_path):
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                existing_ids.add(record["id"])
            except json.JSONDecodeError:
                continue

with open(output_path, "a", encoding="utf-8") as output_file:
    for data in tqdm(data_list):
        sample_id = data["id"]

        if sample_id in existing_ids:
            continue

        try:
            facts: List[str] = data["facts"]
            question: str = data["atomic_question"]
            answer: str = '，'.join([data['option'][c] for c in data['answer']]).strip('，')

            shapley_scores = compute_equal_shapley_values(
                model=model,
                tokenizer=tokenizer,
                all_facts=facts,
                atomic_question=question,
                target_answer=answer,
                max_samples=50,
                min_samples=3,
                tol=1e-2
            )

            # save results
            result = {
                "id": sample_id,
                "shapley": shapley_scores.tolist()
            }
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_file.flush()
        except Exception as e:
            print(f"error sample {sample_id}: {e}")

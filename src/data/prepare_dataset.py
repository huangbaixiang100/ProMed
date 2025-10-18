from datasets import load_dataset

#from src.data.prompt import SYSTEM_PROMPT_TOOLS as SYSTEM_PROMPT
#from src.data.prompt import build_prompt, build_system_tools
from src.data.doctor_patient_prompts import *
import json
import logging
import os

from datasets import load_dataset, Dataset


def prepare_dataset(split="train", name="gsm8k", eval_size=10, train_file=None, test_file=None):
    if name == "gsm8k":
        return prepare_dataset_gsm8k(split, eval_size)
    elif name == "cmb":
        return prepare_dataset_cmb(split, eval_size, train_file, test_file)
    elif name == "medmcqa":
        return prepare_dataset_medmcqa(split, eval_size)
    elif name == "medqa":
        return prepare_dataset_medqa(split, eval_size, train_file, test_file)
    else:
        raise ValueError(f"Unknown dataset name: {name}")


def prepare_dataset_cmb(split="train", eval_size=10, train_file=None, test_file=None):
    """
    Load CMB dataset
    
    Args:
        split: Specify whether to load "train" training set or "test" test set
        eval_size: Deprecated parameter, kept for compatibility with old code
        train_file: Training set file path, use default path if None
        test_file: Test set file path, use default path if None
        
    Returns:
        tuple: If split is "train", return (train_dataset, empty_dataset),
              If split is "test", return (empty_dataset, test_dataset)
    """
    # Set default data file paths
    default_train_file = 'src/data/cmb_atomic_patient_train.json'
    default_test_file = 'src/data/cmb_atomic_patient_test.json'
    
    # Use provided file path or default path
    train_data_file = train_file if train_file else default_train_file
    test_data_file = test_file if test_file else default_test_file
    
    # Determine which data file to load based on split parameter
    if split == "train" or split == "all":
        data_file = train_data_file
        logging.info(f"Loading training set data: {data_file}")
    elif split == "test" or split == "eval":
        data_file = test_data_file
        logging.info(f"Loading test set data: {data_file}")
    else:
        raise ValueError(f"Invalid split parameter: {split}, must be 'train' or 'test'")
    
    # Check if file exists
    if not os.path.exists(data_file):
        # If specified training/test file not found, try using default file
        fallback_file = default_test_file if os.path.exists(default_test_file) else default_train_file
        logging.warning(f"Cannot find data file: {data_file}, fallback to using: {fallback_file}")
        data_file = fallback_file
        
    # Check if fallback file exists
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Cannot find data file: {data_file}")
        
    # Load data file
    logging.info(f"正在Load data file: {data_file}")
    with open(data_file) as f:
        data = json.load(f)

    formatted_data = []

    for idx, example in enumerate(data):
        partial_question = '，'.join(example['facts'][:int(len(example['facts']) / 2)]) + '。' + example['atomic_question']
        option_str = "\n".join([f"{key}: {value}" for key, value in example['option'].items()])
        prompt_str = doctor_system_prompt.format(question_type=example['question_type'], question=partial_question,
                                                    option_str=option_str)
        final_prompt = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n" + prompt_str + "\n<|im_end|>\n<|im_start|>assistant\n"
        )
        formatted_example = {
            "id": idx + 1,
            'prompt': final_prompt,
            'facts': example['facts'],
            'answer': example['answer'],
            'options': example['option'],  # Rename option field to options for consistency
            'atomic_question': example['atomic_question']  # Add atomic question field
        }
        formatted_data.append(formatted_example)

    dataset = Dataset.from_list(formatted_data)

    # Create empty dataset to maintain interface compatibility
    empty_dataset = Dataset.from_list([])
    
    # Decide which dataset to return based on split parameter
    if split == "train" or split == "all":
        logging.info(f"Training set size: {len(dataset)}")
        return dataset, empty_dataset
    else:
        logging.info(f"Test set size: {len(dataset)}")
        return empty_dataset, dataset


def prepare_dataset_gsm8k(split="train", eval_size=10):
    """Load and prepare the GSM8K dataset for training with string prompts."""
    data = load_dataset("openai/gsm8k", "main")[split]
    formatted_data = []

    for example in data:
        # Convert list of messages to a single string prompt.
        prompt_str = build_prompt(
            [
                {"role": "system", "content": build_system_tools(SYSTEM_PROMPT)},
                {"role": "user", "content": example["question"]},
            ]
        )
        formatted_example = {
            "prompt": prompt_str,  # Now a string rather than a list.
            "answer": extract_answer_from_dataset(example["answer"]),
        }
        formatted_data.append(formatted_example)

    return formatted_data


def prepare_dataset_medmcqa(split="train"):
    # Load medical dataset (using medmcqa as example)
    data = load_dataset("medmcqa", split=split)
    formatted_data = []

    for example in data:
        # Construct prompt, assuming SYSTEM_PROMPT is a medical-related prompt
        question = f"""Question: {example["question"]}
            Options:
            A. {example["opa"]}
            B. {example["opb"]}
            C. {example["opc"]}
            D. {example["opd"]}"""

        prompt_str = "\n".join(
            [
                build_system_tools(SYSTEM_PROMPT).strip(),
                f"""Question: {example["question"]}
            Options:
            A. {example["opa"]}
            B. {example["opb"]}
            C. {example["opc"]}
            D. {example["opd"]}""",
            ]
        )
        # Extract correct answer (assuming answer is in "correct_answer" field)
        # Construct formatted data
        correct_answer_index = example["cop"]
        options = [example["opa"], example["opb"], example["opc"], example["opd"]]
        correct_answer = options[correct_answer_index]

        formatted_example = {
            "prompt": prompt_str,
            "question": question,
            "answer": str(correct_answer),  # Convert answer to string
        }
        formatted_data.append(formatted_example)

    return formatted_data


def prepare_dataset_medqa(split="train", eval_size=10, train_file=None, test_file=None):
    """
    Load MedQA dataset
    
    Args:
        split: Specify whether to load "train" training set or "test" test set
        eval_size: Deprecated parameter, kept for compatibility with old code
        train_file: Training set file path, use HuggingFace dataset if None
        test_file: Test set file path, use HuggingFace dataset if None
        
    Returns:
        tuple: (train_dataset, eval_dataset)
    """
    # If local file path is specified, use local file
    if test_file or train_file:
        # Set default data file paths
        default_test_file = '/home/xiaobei/Agentic-RAG-R1__under-construction/src/data/medqa_train_atomic_data.jsonl'
        
        # Use provided file path or default path
        test_data_file = test_file if test_file else default_test_file
        train_data_file = train_file if train_file else default_test_file
        
        # Determine which data file to load based on split parameter
        if split == "train" or split == "all":
            data_file = train_data_file
            logging.info(f"Loading training set data: {data_file}")
        elif split == "test" or split == "eval":
            data_file = test_data_file
            logging.info(f"Loading test set data: {data_file}")
        else:
            raise ValueError(f"Invalid split parameter: {split}, must be 'train' or 'test'")
        
        # Check if file exists
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Cannot find data file: {data_file}")
            
        # Load local jsonl file
        logging.info(f"Loading local jsonl data file: {data_file}")
        formatted_data = []
        
        with open(data_file, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                if line.strip():  # Skip empty lines
                    example = json.loads(line)
                    
                    # Extract partial facts as known information (similar to CMB processing)
                    if 'facts' in example and isinstance(example['facts'], list) and len(example['facts']) > 0:
                        # Take first half of facts as known information
                        facts_len = len(example['facts'])
                        partial_facts = example['facts'][:max(1, facts_len // 2)]
                        initial_info = '，'.join(partial_facts) + '。'
                    else:
                        initial_info = ""
                    
                    # Build partial question (known information + atomic question)
                    partial_question = initial_info + example['atomic_question']
                    
                    # Format options
                    option_str = "\n".join([f"{key}: {value}" for key, value in example['options'].items()])
                    
                    # Build prompt (using same format as CMB)
                    prompt_str = doctor_system_prompt.format(
                        question_type='multiple choice problem', 
                        question=partial_question,
                        option_str=option_str
                    )
                    
                    final_prompt = (
                        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                        "<|im_start|>user\n" + prompt_str + "\n<|im_end|>\n<|im_start|>assistant\n"
                    )
                    
                    formatted_example = {
                        "id": example['id'],
                        'prompt': final_prompt,
                        'facts': example.get('facts', []),  
                        'answer': example['answer_idx'], 
                        'options': example['options'],  
                        'atomic_question': example['atomic_question'] 
                    }
                    formatted_data.append(formatted_example)
        
        dataset = Dataset.from_list(formatted_data)
        empty_dataset = Dataset.from_list([])
        
        
        if split == "train" or split == "all":
            return dataset, empty_dataset
        else:
            return empty_dataset, dataset
    
    else:
  
        data = load_dataset("fzkuji/MedQA", "med_qa_zh_4options_bigbio_qa")[split]

        formatted_data = []

        for idx, example in enumerate(data):
            question = example["question"]
            choices = example["choices"]
            answer = example["answer"][0]

           
            options_text = ""
            for j, choice in enumerate(choices):  
                option_letter = chr(65 + j)  
                options_text += f"{option_letter}. {choice}\n"

            prompt_str = "\n".join(
                [
                    #build_system_tools(SYSTEM_PROMPT).strip(),
                    f"""Question: {question}f
                Options:
                {options_text}""",
                ]
            )

            formatted_data.append(
                {
                    "id": idx + 1,
                    "prompt": prompt_str,
                    "question": question + "\n" + options_text,
                    "answer": str(answer),
                }
            )

        eval_data = formatted_data[:eval_size]
        train_data = formatted_data[eval_size:]  # fixme here

        train_dataset = Dataset.from_list(train_data)
        eval_dataset = Dataset.from_list(eval_data)

        return train_dataset, eval_dataset


if __name__ == "__main__":
    data = prepare_dataset_medqa(split="train")

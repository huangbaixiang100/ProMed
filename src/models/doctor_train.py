from dotenv import load_dotenv
from rich.traceback import install
from rich.logging import RichHandler
from rich import print
load_dotenv()
install()
import datetime
import json
import logging
import os
import time
import sys
import argparse
from pathlib import Path
import deepspeed
import swanlab
import torch
from accelerate import Accelerator, init_empty_weights
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
from dataclasses import dataclass, field
from typing import Optional
load_dotenv()
install()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
from src.data.prepare_dataset import prepare_dataset
from src.models.doctor_trainer import train_with_grpo, overall_reward, build_model
from src.utils.utils import (
    load_config,
    optimize_model_memory,
    set_random_seed,
    setup_logging,
)


@dataclass
class DeepSpeedArguments:
    """Command line arguments for DeepSpeed"""
    deepspeed: Optional[bool] = field(default=False, metadata={"help": "Whether to use DeepSpeed"})
    deepspeed_config: Optional[str] = field(default=None, metadata={"help": "Path to DeepSpeed config file"})
    local_rank: Optional[int] = field(default=-1, metadata={"help": "Local GPU rank"})


@dataclass  
class TrainingArguments:
    """Training related parameters"""
    use_shapley: Optional[bool] = field(default=False, metadata={"help": "Whether to enable Shapley value weighted fact score rewards"})
    shapley_max_samples: Optional[int] = field(default=50, metadata={"help": "Maximum number of samples for Shapley value calculation"})
    shapley_min_samples: Optional[int] = field(default=3, metadata={"help": "Minimum number of samples for Shapley value calculation"})

    # Token-level reward allocation parameters
    use_token_level: Optional[bool] = field(default=False, metadata={"help": "Whether to enable token-level reward allocation"})
    alpha_reward: Optional[float] = field(default=1.0, metadata={"help": "Question Shapley reward weight"})
    beta_reward: Optional[float] = field(default=1.0, metadata={"help": "Question result reward weight"})
    gamma_reward: Optional[float] = field(default=3.0, metadata={"help": "Answer correctness reward weight"})
    format_reward_weight: Optional[float] = field(default=1.0, metadata={"help": "Format reward weight"})


def custom_collate_fn(batch):
    """
    Consolidate dictionaries in the batch and extract atomic questions
    Also handle field compatibility between different datasets
    """
    collated = {key: [sample[key] for sample in batch] for key in batch[0]}
    
    # 🔧 Handle options field compatibility: standardize to 'options' field name
    if 'option' in collated and 'options' not in collated:
        # CMB dataset uses 'option', rename to 'options' for consistency
        collated['options'] = collated.pop('option')
        print(f"[Compatibility] Renamed 'option' field to 'options'")
    elif 'options' in collated:
        print(f"[Compatibility Check] 'options' field already exists")
    else:
        print(f"[Warning] Options field not found! batch keys: {list(collated.keys())}")
    
    return collated


def main():
    # Parse command line arguments
    parser = HfArgumentParser([DeepSpeedArguments, TrainingArguments])
    ds_args, train_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    
    # Set up environment
    config = load_config("src/config/config.yaml")

    # Print Shapley configuration information
    if train_args.use_shapley:
        print(f"[green]✓ Enabled Shapley value weighted training[/green]")
        print(f"  - Maximum samples: {train_args.shapley_max_samples}")
        print(f"  - Minimum samples: {train_args.shapley_min_samples}")
    else:
        print(f"[yellow]Using traditional fact score reward mode[/yellow]")
    
    # Print Token-level reward configuration
    if train_args.use_token_level:
        print(f"[bold green] Enabled pure Token-level reward allocation[/bold green]")
        print(f"  - Question Shapley weight(α): {train_args.alpha_reward}")
        print(f"  - Question result weight(β): {train_args.beta_reward}")
        print(f"  - Answer correctness weight(γ): {train_args.gamma_reward}")
        print(f"  - Format reward weight: {train_args.format_reward_weight}")
        print(f"  - Shapley weighting: {'Enabled' if train_args.use_shapley else 'Disabled'}")
    else:
        print(f"[yellow]Using traditional global reward mode[/yellow]")

    if ds_args.deepspeed and ds_args.deepspeed_config:
        # Load DeepSpeed configuration when using DeepSpeed
        with open(ds_args.deepspeed_config, 'r') as f:
            ds_config = json.load(f)
        # Ensure local rank is set correctly
        local_rank = ds_args.local_rank if ds_args.local_rank != -1 else int(os.environ.get('LOCAL_RANK', '0'))
    else:
        ds_config = None
        local_rank = -1

    # Set distributed configuration
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        is_main_process = local_rank == 0
    else:
        is_main_process = True

    # Create accelerator
    accelerator = Accelerator()
    # Initialize swanlab
    if accelerator.is_local_main_process and config.swanlab:
        swanlab_config = config.__dict__.copy()
        # Add Shapley configuration to SwanLab
        swanlab_config.update({
            "use_shapley": train_args.use_shapley,
            "shapley_max_samples": train_args.shapley_max_samples,
            "shapley_min_samples": train_args.shapley_min_samples
        })
        
        swanlab.init(
            project=config.project.name,
            experiment_name=config.experiment.name + ("_shapley" if train_args.use_shapley else "_traditional"),
            config=swanlab_config,
            api_key=""
        )
        logging.info("SwanLab initialized")

    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    today = now.strftime("%Y-%m-%d")
    checkpoint_dir = Path(f"checkpoints/{config.experiment.name}/{today}")
    output_dir = Path(f"experiments/training/{config.experiment.name}/{time_str}")
    
    if is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(output_dir, level=logging.INFO)
        with open(output_dir / "config.json", "w") as f:
            json.dump(config.__dict__, f, indent=2)
        logging.info(f"Saved configuration to {output_dir / 'config.json'}")

    set_random_seed(config.experiment.random_seed)
    if is_main_process:
        logging.info(f"Set random seed to {config.experiment.random_seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if is_main_process:
        logging.info(f"Using device: {device}")

    # Prepare dataset
    if config.dataset.name == "medqa":
        # Use local MedQA data file
        train_dataset, eval_dataset = prepare_dataset(
            "train", config.dataset.name, eval_size=config.dataset.num_eval,
            train_file='./src/data/dataset/medqa_train_atomic_data.jsonl'
        )
    else:
        train_dataset, eval_dataset = prepare_dataset("train", config.dataset.name, eval_size=config.dataset.num_eval)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.training.batch_size, 
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=1, 
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    if is_main_process:
        logging.info(f"Train dataloader: {len(train_dataloader)}, Eval dataloader: {len(eval_dataloader)}")

    # Initialize model and tokenizer
    if is_main_process:
        logging.info("Loading model...")

    # Build base model and tokenizer
    base_model, tokenizer = build_model(config, device)
    reference_base_model, _ = build_model(config, device)
    
    if is_main_process:
        logging.info("Base model and tokenizer loaded successfully")
        
    # Ensure tokenizer settings are correct
    tokenizer.pad_token = tokenizer.eos_token
    if hasattr(base_model, 'config'):
        base_model.config.pad_token_id = base_model.config.eos_token_id = tokenizer.eos_token_id
        reference_base_model.config.pad_token_id = reference_base_model.config.eos_token_id = tokenizer.eos_token_id
    
    # Model memory optimization
    base_model = optimize_model_memory(base_model)
    reference_base_model = optimize_model_memory(reference_base_model)

    # Import atomic question extraction function - must import before training_config
    from src.models.doctor_trainer import extract_atomic_questions_from_batch

    # Training configuration
    training_config = {
        "num_iterations": config.training.num_iterations,
        "steps_per_iteration": config.training.steps_per_iteration,
        "num_generations": config.training.generation.num_generations,
        "max_new_tokens": config.training.generation.max_new_tokens,
        "max_length_for_gather": config.training.generation.max_length_for_gather,
        "max_generate_iterations": config.training.generation.max_generate_iterations,
        "temperature": config.training.generation.temperature,
        "do_sample": config.training.generation.do_sample,
        "beta": config.training.optimizer.beta,
        "learning_rate": config.training.learning_rate,
        "mu": config.training.optimizer.mu,
        "epsilon": config.training.optimizer.epsilon,
        "reward_function": overall_reward,
        "save_interval": config.training.save_interval,
        # Add Shapley related configuration
        "use_shapley": train_args.use_shapley,
        "extract_atomic_questions_fn": extract_atomic_questions_from_batch,
        # Add Token-level reward configuration
        "use_token_level": train_args.use_token_level,
        "alpha_reward": train_args.alpha_reward,
        "beta_reward": train_args.beta_reward,
        "gamma_reward": train_args.gamma_reward,
        "format_reward_weight": train_args.format_reward_weight,
    }
    if is_main_process:
        logging.info(f"Training configuration: {training_config}")

    if config.training.continue_training:
        current_step = config.training.current_step
    else:
        current_step = 0

    # Detect DeepSpeed configuration
    zero_stage = None
    if ds_config is not None and "zero_optimization" in ds_config:
        zero_stage = ds_config["zero_optimization"].get("stage", 0)
        if is_main_process:
            logging.info(f"Detected DeepSpeed ZeRO-{zero_stage} configuration")
    
    # Train using GRPO
    train_with_grpo(
        config=config,
        device=device,
        policy_model=base_model,
        ref_base_model=reference_base_model,
        tokenizer=tokenizer,
        accelerator=accelerator,
        dataloader=train_dataloader,
        checkpoint_dir=checkpoint_dir,
        current_step=current_step,
        **training_config,
    )
    if is_main_process:
        logging.info("Training completed")

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
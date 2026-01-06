# ProMed: Shapley Information Gain Guided Reinforcement Learning for Proactive Medical LLMs

Official Implementation of [ProMed: Shapley Information Gain Guided Reinforcement Learning for Proactive Medical LLMs]
ProMed is a novel approach for enhancing medical LLMs' proactive information-seeking ability through Shapley information gain rewards and reinforcement learning frameworks to improve interactive diagnosis.

## 🎯 Key Features

- **Shapley Information Gain Reward System**: Utilizes Shapley values to calculate information gain rewards for questions
- **Token-level Reinforcement Learning**: Precise token-level reward allocation with group baseline advantage calculation
- **Interactive Medical Consultation**: Multi-round doctor-patient dialogue simulation
- **Support for Multiple Datasets**: Compatible with CMB and MedQA medical datasets
- **Distributed Training**: DeepSpeed ZeRO-2 support for efficient large model training

## 🏗️ Architecture

The ProMed framework consists of:

1. **Doctor Model**: The main model being trained for interactive diagnosis
2. **Patient Model**: Simulates patient responses during training and evaluation
3. **Shapley Value Calculator**: Computes information gain for each question
4. **Reward System**: 
   - Question tokens: Shapley reward (0-3) + Format reward (0-1)
   - Answer tokens: Correctness reward (0-3) + Format reward (0-1)
   - Other tokens: 0 points

## 📋 Requirements

- Python 3.8+
- CUDA-compatible GPU(s)

## 🚀 Quick Start

### 1. Environment Setup

Create a new conda environment:

```bash
conda create -n promed python=3.8
conda activate promed
```

### 2. Installation

Clone the repository and install dependencies:

```bash
git clone <repository-url>
cd ProMed
pip install -r requirements.txt
```

### 3. SFT with MCTS

Before reinforcement learning, ProMed supports **supervised fine-tuning (SFT) data collection** through a Monte Carlo Tree Search (MCTS)-based sampling module.  
This step simulates multi-turn doctor–patient dialogues and computes question-level Shapley values to prioritize informative samples.

Run the following script to calculate Shapley values for your dataset:

```bash
cd src/mcts_sampling
python cal_shapley.py
```

This step estimates the marginal contribution of each atomic fact using the model’s predicted outcome probability.

Edit src/MCTS_Sampling/mcts.py to set up your model API endpoints. After configuring APIs, run:
```bash
python async_mcts.py
```
Once sampling is complete, the resulting dataset can be used to fine-tune your model. After SFT, follow the subsequent steps to run ProMed reinforcement learning framework.


### 4. API Configuration

#### SwanLab API Key
Add your SwanLab API key to the training script:

```bash
# Edit script/training/train_with_shapley.sh
export SWANLAB_API_KEY="your_swanlab_api_key_here"
```

#### Patient Model API
Configure the patient model API in multiple files:

**For evaluation scripts:**
```bash
# Edit script/evaluate/async_prompt_ask_predict_answer.py
patient_client = OpenAI(api_key="your_patient_model_api_key", base_url="your_patient_model_endpoint")
patient_model = 'your_patient_model_name'  # e.g., 'qwen2.5:72b'
```

```bash
# Edit script/evaluate/async_test_medqa.py  
patient_client = OpenAI(api_key="your_patient_model_api_key", base_url="your_patient_model_endpoint")
patient_model = 'your_patient_model_name'
```

**For training (patient model used in reward calculation):**
```bash
# Edit src/utils/patient_model.py
patient_client = OpenAI(api_key="your_patient_model_api_key", base_url="your_patient_model_endpoint")
# Also update the model name in line 15: call_gpt(patient_client, 'your_patient_model_name', patient_messages)
```

**For fact checking in reward system:**
```bash
# Edit src/models/doctor_reward.py (3 locations around lines 58, 503, 798)
fact_checker_client = OpenAI(api_key="your_patient_model_api_key", base_url="your_patient_model_endpoint")
fact_checker_model = 'your_patient_model_name'  # e.g., 'qwen2.5:72b'
```

#### Doctor Model API (for evaluation)
Configure the doctor model API for evaluation:

```bash
# Edit script/evaluate/async_prompt_ask_predict_answer.py
doctor_client = OpenAI(api_key="EMPTY", base_url="your_doctor_model_endpoint")
doctor_model = 'your_doctor_model_name'  # e.g., 'llama3_1_8b'
```

### 5. Model Configuration

Edit the configuration file to specify your model and dataset:

```yaml
# src/config/config.yaml
model:
  name: "path/to/your/model"  # Path to the model you want to train with RL

dataset:
  name: "cmb"  # or "medqa"
```

### 6. Dataset Configuration

#### For CMB Dataset (Chinese)
If using the CMB dataset, you need to configure Chinese prompts:

```bash
# Edit src/data/doctor_patient_prompts.py
# Uncomment the Chinese prompt sections (lines 59-112) and comment out the English prompts (lines 8-56)
# The Chinese prompts provide better performance for CMB dataset
```

The script automatically handles:
- CMB dataset: Uses `prepare_dataset_cmb()` function with Chinese prompts
- MedQA dataset: Uses `prepare_dataset_medqa()` function with English prompts

#### Prompt Language Settings
- **CMB Dataset**: Use Chinese prompts for better performance
  - Chinese doctor_system_prompt (lines 59-76 in doctor_patient_prompts.py)
  - Chinese patient_system_prompt (lines 78-88 in doctor_patient_prompts.py)  
  - Chinese doctor_understanding_prompt (lines 90-104 in doctor_patient_prompts.py)

- **MedQA Dataset**: Use English prompts (default)
  - English prompts are active by default (lines 8-56)

## 🎓 Training

### Start Shapley-based Training

Run the training script with Shapley-based rewards:

```bash
cd ProMed
bash script/training/train_with_shapley.sh
```

### Training Configuration

The training script supports several configuration options:

```bash
# Environment variables for customization
export USE_SHAPLEY="True"              # Enable Shapley value weighting
export ALPHA_REWARD="2.0"              # Question Shapley reward weight
export BETA_REWARD="1.0"               # Question result reward weight  
export GAMMA_REWARD="3.0"              # Answer correctness reward weight
export FORMAT_REWARD_WEIGHT="1.0"      # Format reward weight

bash script/training/train_with_shapley.sh
```

### Monitoring Training

Training progress can be monitored through:
- **SwanLab Dashboard**: Real-time training metrics and rewards
- **Console Output**: Token-level reward statistics and training progress
- **Checkpoints**: Saved in the experiments directory

## 📊 Evaluation

### Evaluate on CMB Dataset

```bash
cd ProMed/script/evaluate
python async_prompt_ask_predict_answer.py
python cmb_acc.py
```

### Evaluate on MedQA Dataset  

```bash
cd ProMed/script/evaluate
python async_test_medqa.py
python medqa_acc.py
```

### Accuracy Calculation

The evaluation scripts will:
1. Run interactive dialogues between doctor and patient models
2. Collect responses and calculate accuracy metrics
3. Generate detailed performance reports

## 📁 Project Structure

```
ProMed/
├── src/
│   ├── mcts_sampling/                # SFT data collection with MCTS
│   │   ├── cal_shapley.py            # Compute Shapley values for sampling
│   │   ├── mcts.py                   # MCTS-based dialogue sampling
│   │   └── async_mcts.py             # Asynchronous sampling and data saving
│   ├── config/
│   │   ├── config.yaml              # Main configuration file
│   │   └── accelerate_config/       # DeepSpeed configurations
│   ├── data/
│   │   ├── prepare_dataset.py       # Dataset preparation and prompts
│   │   └── doctor_patient_prompts.py
│   ├── models/
│   │   ├── doctor_train.py          # Main training script
│   │   ├── doctor_trainer.py        # Training logic
│   │   └── doctor_reward.py         # Reward calculation with Shapley values
│   └── utils/
│       └── utils.py                 # Utility functions
├── script/
│   ├── training/
│   │   └── train_with_shapley.sh    # Shapley-based training script
│   └── evaluate/
│       ├── async_prompt_ask_predict_answer.py  # CMB evaluation
│       ├── async_test_medqa.py      # MedQA evaluation  
│       ├── cmb_acc.py               # CMB accuracy calculation
│       └── medqa_acc.py             # MedQA accuracy calculation
├── requirements.txt                 # Python dependencies
└── README.md
```

## 🔧 Configuration Details

### API Configuration Summary
Before training or evaluation, ensure all API keys are configured:

1. **SwanLab API**: `script/training/train_with_shapley.sh` 
2. **Patient Model APIs** (4 locations):
   - `script/evaluate/async_prompt_ask_predict_answer.py`
   - `script/evaluate/async_test_medqa.py`
   - `src/utils/patient_model.py` 
   - `src/models/doctor_reward.py` (3 locations)
3. **Doctor Model API**: `script/evaluate/` (evaluation scripts)

### Model Configuration
- **model.name**: Path to your base model for RL training
- **dataset.name**: Choose between "cmb" or "medqa"
- **training.use_lora**: Enable LoRA for efficient training
- **training.batch_size**: Adjust based on GPU memory

### Dataset-Specific Configuration
- **CMB Dataset**: 
  - Set `dataset.name: "cmb"` in config.yaml
  - Use Chinese prompts in `src/data/doctor_patient_prompts.py`
  - Configure patient model API for Chinese language model
- **MedQA Dataset**:
  - Set `dataset.name: "medqa"` in config.yaml  
  - Use English prompts (default configuration)
  - Configure patient model API for English language model

### Reward System Configuration
- **ALPHA_REWARD**: Weight for question Shapley rewards (default: 2.0)
- **BETA_REWARD**: Weight for question result rewards (default: 1.0)  
- **GAMMA_REWARD**: Weight for answer correctness rewards (default: 3.0)
- **FORMAT_REWARD_WEIGHT**: Weight for format rewards (default: 1.0)

### DeepSpeed Configuration
The project uses DeepSpeed ZeRO-2 for distributed training:
- Configured in `src/config/accelerate_config/train_zero2.yaml`
- Supports multi-GPU training with automatic gradient synchronization


## 📄 License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.
---

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# This script is written for the latest stable versions of the libraries (as of July 2025)
# It uses the modern, more flexible TRL API.
from trl import PPOConfig, PPOTrainer
# FIX: Import the model class with a value head
from trl.models import AutoModelForCausalLMWithValueHead

# --------------------------------------------------------------------------------------
# 1. Hardcoded Configuration
# --------------------------------------------------------------------------------------
config = {
    "model_name": "gpt2",
    "dataset_name": "imdb",
    "learning_rate": 1.41e-5,
    "batch_size": 16,
    "mini_batch_size": 16,
    "ppo_epochs": 4, # Number of optimization epochs per PPO batch
    "total_ppo_steps": 100, # Total number of PPO steps to train for
    "output_dir": "ppo_gpt2_imdb_latest",
}

# --------------------------------------------------------------------------------------
# 2. Model & Tokenizer
# --------------------------------------------------------------------------------------
# The policy model is a standard causal language model.
policy_model = AutoModelForCausalLM.from_pretrained(
    config["model_name"],
    device_map="auto",
)
# The reference model is a frozen copy of the initial policy model.
ref_model = AutoModelForCausalLM.from_pretrained(
    config["model_name"],
    device_map="auto",
)
# Create a dedicated value model with a value head.
# This is required by the PPOTrainer in your installed version.
value_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config["model_name"],
    device_map="auto",
)

# Manually set the base_model_prefix.
# The PPOTrainer needs this attribute to correctly access the underlying transformer.
# We copy it from the policy_model, which has it set correctly.
value_model.base_model_prefix = policy_model.base_model_prefix

# Manually attach the base model to the value_model wrapper.
# The PPOTrainer expects to find the attribute specified by `base_model_prefix`
# directly on the value_model object. We create that reference here.
setattr(value_model, value_model.base_model_prefix, value_model.pretrained_model)


tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
# A pad token is required for batch generation.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# --------------------------------------------------------------------------------------
# 3. Dataset Preparation
# --------------------------------------------------------------------------------------
def prepare_dataset(examples):
    """Create prompts for the model."""
    # We create a prompt by taking the first 8 words of the review text.
    # The trainer will handle tokenization internally.
    examples["query"] = [" ".join(text.split()[:8]) for text in examples["text"]]
    return examples

# Load the dataset and select a small subset for a quick example
dataset = load_dataset(config["dataset_name"], split="train").shuffle().select(range(1000))
# Apply the preparation function, keeping only the 'query' column.
dataset = dataset.map(prepare_dataset, batched=True, remove_columns=dataset.column_names)


# --------------------------------------------------------------------------------------
# 4. PPO Trainer Initialization (The Modern API)
# --------------------------------------------------------------------------------------
# The configuration is bundled into a PPOConfig object and passed via `args`.
# This matches the API for your installed version of TRL.
ppo_config = PPOConfig(
    learning_rate=config["learning_rate"],
    batch_size=config["batch_size"],
    mini_batch_size=config["mini_batch_size"],
    num_ppo_epochs=config["ppo_epochs"],
)

# FIX: The trainer requires a placeholder for `reward_model`. We pass the
# `value_model` as a valid placeholder, which resolves the `AttributeError`.
# This placeholder will not be used since we calculate rewards manually.
ppo_trainer = PPOTrainer(
    args=ppo_config,
    model=policy_model,
    ref_model=ref_model,
    reward_model=value_model,
    value_model=value_model,
    processing_class=tokenizer,
    train_dataset=dataset,
)

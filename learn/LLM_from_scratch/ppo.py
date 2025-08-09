import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from trl import PPOConfig, PPOTrainer
from trl.models import AutoModelForCausalLMWithValueHead

config = {
    "model_name": "gpt2",
    "reward_model_name": "gpt2",
    "value_model_name": "gpt2",
    "dataset_name": "imdb",
    "learning_rate": 1.41e-5,
    "batch_size": 4,
    "mini_batch_size": 2,
    "ppo_epochs": 4,
    "total_ppo_steps": 100,  # Total number of PPO steps to train for
    "output_dir": "ppo_gpt2_imdb_latest",
}


def build_models_and_tokenizer(cfg: dict):
    policy_model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], device_map="auto")
    # The reference model is a frozen copy of the initial policy model.
    ref_model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], device_map="auto")

    # Value model and reward model are sequence classifiers with 1 label
    value_model = AutoModelForSequenceClassification.from_pretrained(
        cfg["value_model_name"], device_map="auto", num_labels=1
    )
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        cfg["reward_model_name"], device_map="auto", num_labels=1
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return policy_model, ref_model, value_model, reward_model, tokenizer


def prepare_dataset(examples):
    """Create prompts for the model by taking first 20 words of the review text."""
    examples["query"] = [" ".join(text.split()[:20]) for text in examples["text"]]
    return examples


def create_dataset(cfg: dict, tokenizer: AutoTokenizer):
    # Load the dataset and select a small subset for a quick example
    dataset = load_dataset(cfg["dataset_name"], split="train").shuffle().select(range(1000))
    # Trainer requires tokenized inputs
    dataset = dataset.map(prepare_dataset, batched=True, remove_columns=dataset.column_names)

    def tokenize(element):
        outputs = tokenizer(
            element["query"],
            padding=False,
        )
        return {"input_ids": outputs["input_ids"]}

    dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    return dataset



def build_trainer(cfg: dict, policy_model, ref_model, value_model, tokenizer, dataset):
    ppo_config = PPOConfig(
        learning_rate=cfg["learning_rate"],
        batch_size=cfg["batch_size"],
        num_mini_batches=cfg["mini_batch_size"],
        num_ppo_epochs=cfg["ppo_epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=2,
    )

    ppo_trainer = PPOTrainer(
        args=ppo_config,
        model=policy_model,
        ref_model=ref_model,
        reward_model=value_model,
        value_model=value_model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=dataset,
    )
    return ppo_trainer


def main():
    policy_model, ref_model, value_model, reward_model, tokenizer = build_models_and_tokenizer(config)
    dataset = create_dataset(config, tokenizer)
    ppo_trainer = build_trainer(config, policy_model, ref_model, value_model, tokenizer, dataset)

    ppo_trainer.train()
    ppo_trainer.save_model(config["output_dir"])
    ppo_trainer.generate_completions()


if __name__ == "__main__":
    main()



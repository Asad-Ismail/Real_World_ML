import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModelForSequenceClassification
from trl import PPOConfig, PPOTrainer
from trl.models import AutoModelForCausalLMWithValueHead


# Define configs better to pass as arguments or config file.
config = {
    "model_name": "gpt2",
    "reward_model_name": "gpt2",
    "value_model_name": "gpt2",
    "dataset_name": "imdb",
    "learning_rate": 1.41e-5,
    "batch_size": 4,
    "mini_batch_size": 2,
    "ppo_epochs": 4,
    "total_ppo_steps": 100, # Total number of PPO steps to train for
    "output_dir": "ppo_gpt2_imdb_latest",
}



# Models Initialization
policy_model = AutoModelForCausalLM.from_pretrained(config["model_name"],device_map="auto")
# The reference model is a frozen copy of the initial policy model.
ref_model = AutoModelForCausalLM.from_pretrained(config["model_name"],device_map="auto")


# The PPOTrainer needs this attribute to correctly access the underlying transformer.
# We copy it from the policy_model, which has it set correctly.
#value_model.base_model_prefix = policy_model.base_model_prefix
#setattr(value_model, value_model.base_model_prefix, value_model.pretrained_model)
value_model = AutoModelForSequenceClassification.from_pretrained(config["value_model_name"],device_map="auto",num_labels=1)
reward_model =  AutoModelForSequenceClassification.from_pretrained(config["reward_model_name"],device_map="auto",num_labels=1)
#reward_model = AutoModelForCausalLMWithValueHead(config["reward_model_name"])
#reward_model = AutoModelForCausalLMWithValueHead.from_pretrained(config["reward_model_name"])

tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def prepare_dataset(examples):
    """Create prompts for the model."""
    # We create a prompt by taking the first 20 words of the review text to see
    examples["query"] = [" ".join(text.split()[:20]) for text in examples["text"]]
    return examples

# Load the dataset and select a small subset for a quick example
dataset = load_dataset(config["dataset_name"], split="train").shuffle().select(range(1000))
# Trainer requires tokenized inputs
dataset = dataset.map(prepare_dataset, batched=True, remove_columns=dataset.column_names)


def tokenize(element):
    outputs = tokenizer(
        element["query"],
        padding=False
        
    )
    #"attention_mask": outputs["attention_mask"]
    return {"input_ids": outputs["input_ids"]}

dataset = dataset.map(tokenize,batched=True,remove_columns=dataset.column_names)


'''
Test what value and reward model returns
for item in dataset:
    #print(item)
    device= next(reward_model.parameters()).device
    input_ids = torch.tensor(item["input_ids"]).unsqueeze(0).to(device)  # [1, seq_len]
    attention_mask = torch.tensor(item["attention_mask"]).unsqueeze(0).to(device)

    with torch.no_grad():
        print(input_ids.shape)
        print(attention_mask.shape)
        outputs = reward_model(input_ids=input_ids, attention_mask=attention_mask)
        print(outputs[0].shape)
        print(outputs[2].shape)
        #print(len(outputs))
        #print(outputs.keys())
        #print(outputs['logits'])
    #x = reward_model(item['input_ids'])
    break

'''

ppo_config = PPOConfig(
    learning_rate=config["learning_rate"],
    batch_size=config["batch_size"],
    num_mini_batches=config["mini_batch_size"],
    num_ppo_epochs=config["ppo_epochs"],
    per_device_train_batch_size=config["batch_size"],
    gradient_accumulation_steps=2 
)

ppo_trainer = PPOTrainer(
    args=ppo_config,
    model=policy_model,
    ref_model=ref_model,
    reward_model=value_model,
    value_model=value_model,
    processing_class=tokenizer,
    train_dataset=dataset,
    eval_dataset=dataset
)


ppo_trainer.train()

ppo_trainer.save_model(config["output_dir"])
ppo_trainer.generate_completions()



import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
import random
import numpy as np

seed=42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    seed=seed

# Set Seed
set_seed(seed)  

## Define the models, We will define policy, reference and policy model normally we have a reward model trianed on human
## preference here we will write a simple length based reward function. Note currently TRL does not support passing any
## function as reward model it needs to be inhereted from nn.Module
## Ssing Qwen3 llm Qwen/Qwen3-0.6B
model="Qwen/Qwen3-0.6B"
device ="auto"

gen_config = {"max_new_tokens":50,
    "do_sample":True,
    "temperature":0.9,
    "top_k":50,
    "top_p":0.95}


policy_model = AutoModelForCausalLM.from_pretrained(model,device_map=device)
reference_model = AutoModelForCausalLM.from_pretrained(model,device_map=device)
# wrap policy modle to also output value output saves memory of creating a completely new value model, just add new value head
# and use it for both policy and value calculations
class PolicyValueModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config
        self.value_head = nn.Linear(self.config.hidden_size, 1, bias=False)

    def forward(self, **kwargs):
        outputs = self.model(**kwargs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # Last layer hidden state
        values = self.value_head(hidden_states).squeeze(-1)  # (batch, seq_len)
        return outputs, values

## infer device from policy models
infer_device = next(policy_model.parameters()).device
pvmodel = PolicyValueModel(policy_model).to(infer_device)
## Define Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model)
## Load Datasets
def prepare_dataset(examples):
    """Create prompts for the model."""
    # We create a prompt by taking the first 30 words of the review text to see
    examples["query"] = [" ".join(text.split()[:30]) for text in examples["text"]]
    return examples

# Select smaller sample for test
dataset = load_dataset("imdb", split="train").shuffle(seed=seed).select(range(1000))
dataset = dataset.map(prepare_dataset, batched=True, remove_columns=dataset.column_names)
## Pretokenize data to save on the fly tokenization
def tokenize(element):
    outputs = tokenizer(element["query"],padding=True,truncation=True)
    return outputs

dataset = dataset.map(tokenize,batched=True,remove_columns=dataset.column_names)
# convert tor torch tensors
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

for i,data in enumerate(dataset):
    data = {k: v.unsqueeze(0).to(infer_device) for k, v in data.items()}
    print(f"Input data {data}")
    print(f"*"*50)
    generated_ids = pvmodel.model.generate(**data,**gen_config)
    print(f"Generated Input IDS")
    print(generated_ids)
    print(f"*"*50)
    # Recompute attention_mask for the generated_ids, really needed for batched
    attention_mask = (generated_ids != tokenizer.pad_token_id).long()
    print(f"Generated Attention Mask")
    print(attention_mask)
    print(f"*"*50)
    # Forward pass with both input_ids and attention_mask
    outputs, values = pvmodel(input_ids=generated_ids,attention_mask=attention_mask)
    print("Logits shape:", outputs.logits.shape)
    print("Value shape: ", values.shape)
    break


## PPO has two stages 
## 1. roll out the experience
## 2. Update policy and Value Model



## Roll Out the experience

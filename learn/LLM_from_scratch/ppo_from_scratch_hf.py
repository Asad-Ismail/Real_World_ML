import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

## Define the models, We will define policy, reference and policy model normally we have a reward model trianed on human
## preference here we will write a simple length based reward function. Note currently TRL does not support passing any
## function as reward model it needs to be inhereted from nn.Module
## Ssing Qwen3 llm Qwen/Qwen3-0.6B
model="Qwen/Qwen3-0.6B"
device ="auto"
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
## Test the model
#inputs = tokenizer("Hello, my name is", return_tensors="pt").to("cuda")
#outputs, value = pvmodel(**inputs)
#print("Logits shape:", outputs.logits.shape)
#print("Value shape: ", value.shape)
## Load Datasets
def prepare_dataset(examples):
    """Create prompts for the model."""
    # We create a prompt by taking the first 50 words of the review text to see
    examples["query"] = [" ".join(text.split()[:50]) for text in examples["text"]]
    return examples

dataset = load_dataset("imdb", split="train").shuffle().select(range(1000))
dataset = dataset.map(prepare_dataset, batched=True, remove_columns=dataset.column_names)

def tokenize(element):
    outputs = tokenizer(element["query"],padding=False)
    return {"input_ids": outputs["input_ids"],"attention_mask": outputs["attention_mask"]}

dataset = dataset.map(tokenize,batched=True,remove_columns=dataset.column_names)





## PPO has two stages 
## 1. roll out the experience
## 2. Update policy and Value Model



## Roll Out the experience

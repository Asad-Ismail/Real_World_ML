import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from torch.utils.data import DataLoader
import random
import numpy as np

seed=42
minibatch= 2

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


dataloader = DataLoader(dataset, batch_size=minibatch, shuffle=True)

# Gather minibatch
for data in dataloader:
    responses = []
    logprobs_list = []
    ref_logprobs_list = []
    sequence_lengths = []
    values_list = []
    rewards_list = []

    for singleitem in range(data["input_ids"].shape[0]):
        
        data = {k: v.unsqueeze(0).to(infer_device) for k, v in data[singleitem].items()}
        context_length = data["input_ids"].shape[1]

        print(f"Input data {data}")
        print("*" * 50)

        with torch.no_grad():
            generated_ids = pvmodel.model.generate(**data, **gen_config)

        print("Generated Input IDS")
        print(generated_ids)
        print("*" * 50)

        attention_mask = (generated_ids != tokenizer.pad_token_id).long()

        with torch.no_grad():
            outputs = pvmodel(input_ids=generated_ids, attention_mask=attention_mask)
            logits, values_ = outputs[0].logits, outputs[1]

        print("Logits shape:", logits.shape)
        print("Value shape: ", values_.shape)

        print("**Input text**")
        print(tokenizer.decode(generated_ids[0][:context_length], skip_special_tokens=False))

        print("*Generated text*")
        print(tokenizer.decode(generated_ids[0][context_length:], skip_special_tokens=False))

        # Get logits of generated tokens
        gen_index = generated_ids[:, context_length:]  # [B, G]
        temperature = getattr(gen_config, "temperature", 1.0) + 1e-7
        select_logits = logits[:, context_length-1:-1, :] / temperature
        assert select_logits.shape[1] == gen_index.shape[1], "Selcted logits shape does not match the index shape"
        logprobs_ = torch.gather(select_logits.log_softmax(-1), dim=-1, index=gen_index.unsqueeze(-1)).squeeze(-1)

        # Reference model logprobs
        with torch.no_grad():
            ref_output = reference_model(input_ids=generated_ids, attention_mask=attention_mask)

        ref_logits = ref_output.logits[:, context_length-1:-1, :] / temperature
        ref_logprobs_ = torch.gather(ref_logits.log_softmax(-1), dim=-1, index=gen_index.unsqueeze(-1)).squeeze(-1)

        ## Get Rewards
        # Get token index just before the pad token to get valudable reward in generated text
        pad_mask = generated_ids[:, context_length:] == tokenizer.pad_token_id
        # get first occurance of pad token
        first_pad = pad_mask.float().argmax(dim=1)
        # if no pad token exist then take the len of generated sequence
        any_pad = pad_mask.any(dim=1)
        first_pad[~any_pad]= pad_mask.shape[1]
        last_useful_tokens = first_pad -1 + context_length # shape of [B]
        sequence_length = first_pad -1
        # Now we get reward using the last_useful_token_index from a model but for now we are harcoding reward to be equal to
        # normalized length of response to encourage smaller responses
        rewards_ = last_useful_tokens / gen_config["max_new_tokens"]  # shape of [B]
        print(f"Reward shape is {rewards_.shape}")
        ## Reduce reward of uncomplete sentence
        contain_eos_token = torch.any(generated_ids == tokenizer.eos_token_id, dim=-1)
        rewards_[~contain_eos_token] -= 0.1

        ## collect data
        logprobs_list.append(logprobs_)
        ref_logprobs_list.append(ref_logprobs_)
        values_list.append(values_)
        rewards_list.append(rewards_)
        sequence_lengths.append(sequence_length)
        responses.append(gen_index)
        break

    logprobs = torch.cat(logprobs_list, 0)
    ref_logprobs = torch.cat(ref_logprobs_list, 0)
    values = torch.cat(values_list, 0)
    scores = torch.cat(rewards_list, 0)
    responses = torch.cat(responses, 0)
    sequence_lengths = torch.cat(sequence_lengths, 0)

    ## get valid log probs, ref porbs and values, reward is already valid we could have done it above while getting 
    # valid reward but is more efficent here since its calcualted in batch
    INVALID_LOGPROB = 1
    response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
    padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
    logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
    ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
    sequence_lengths_p1 = sequence_lengths + 1
    padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))
    values = torch.masked_fill(values, padding_mask_p1, 0)

    #compute rewards
    # Formula used by http://joschu.net/blog/kl-approx.html for the k1 
    kl_coeff = 0.05
    logr = ref_logprobs - logprobs
    kl = -logr
    non_score_reward = -kl_coeff * kl
    rewards = non_score_reward.clone()
    actual_start = torch.arange(rewards.size(0), device=rewards.device)
    actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)
    rewards[[actual_start, actual_end]] += scores

    #compute advantages and returns
    gamma=1
    lam =0.95
    lastgaelam = 0
    advantages_reversed = []
    gen_length = responses.shape[1]
    for t in reversed(range(gen_length)):
        nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1], axis=1)
    returns = advantages + values
    advantages = masked_whiten(advantages, ~padding_mask)
    advantages = torch.masked_fill(advantages, padding_mask, 0)


    break


## PPO has two stages 
## 1. roll out the experience
## 2. Update policy and Value Model



## Roll Out the experience

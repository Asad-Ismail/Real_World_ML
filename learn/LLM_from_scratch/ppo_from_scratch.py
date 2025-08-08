import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import set_seed

seed=42
minibatch_size = 2

# Set seed
set_seed(seed)
## Define the models. We will define a policy, a reference, and a policy model. Normally we have a reward model trained on human
## preference; here we will write a simple length-based reward function. Note: currently TRL does not support passing any
## function as a reward model
## Using Qwen3 LLM Qwen/Qwen3-0.6B
#model="Qwen/Qwen3-0.6B"
model="Qwen/Qwen3-0.6B"
device ="auto"

gen_config = {"max_new_tokens":50,
    "do_sample":True,
    "temperature":0.9,
    "top_k":50,
    "top_p":0.95}

temperature_epsilon = 1e-7

# Optional lower precision to reduce memory if CUDA is available
use_fp16 = torch.cuda.is_available()
dtype_kwargs = {"torch_dtype": torch.float16} if use_fp16 else {}

## Policy and reward model
policy_model = AutoModelForCausalLM.from_pretrained(model,device_map=device, low_cpu_mem_usage=True, **dtype_kwargs)
reference_model = AutoModelForCausalLM.from_pretrained(model,device_map=device, low_cpu_mem_usage=True, **dtype_kwargs)
# wrap policy model to also output value output saves memory of creating a completely new value model, just add new value head
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

## infer device from policy model
infer_device = next(policy_model.parameters()).device
pvmodel = PolicyValueModel(policy_model).to(infer_device)
if use_fp16:
    pvmodel = pvmodel.to(dtype=torch.float16)
## Define tokenizer
tokenizer = AutoTokenizer.from_pretrained(model)
## Load datasets
def prepare_dataset(examples):
    """Create prompts for the model."""
    # We create a prompt by taking the first 30 words of the review text to see
    examples["query"] = [" ".join(text.split()[:30]) for text in examples["text"]]
    return examples

# Select a smaller sample for testing
dataset = load_dataset("imdb", split="train").shuffle(seed=seed).select(range(1000))
dataset = dataset.map(prepare_dataset, batched=True, remove_columns=dataset.column_names)
## Pre-tokenize data to avoid on-the-fly tokenization
def tokenize(element):
    outputs = tokenizer(element["query"],padding=True,truncation=True)
    return outputs

dataset = dataset.map(tokenize,batched=True,remove_columns=dataset.column_names)
# convert to torch tensors
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

dataloader = DataLoader(dataset, batch_size=minibatch_size, shuffle=True)
optimizer = torch.optim.Adam(pvmodel.parameters(), lr=5e-5)
## PPO has two stages 
## 1. roll out the experience
# Iterate over batches
for batch in dataloader:
    # query + responses needed for new policy value and logits generation
    query_response = []
    # only responses
    responses = []
    logprobs_list = []
    ref_logprobs_list = []
    sequence_lengths = []
    values_list = []
    rewards_list = []
    context_length_list=[]

    batch_size = batch["input_ids"].shape[0]  # current minibatch size

    for i in range(batch_size):
        # Extract single sample
        single_sample = {k: v[i].unsqueeze(0).to(infer_device) for k, v in batch.items()}
        context_length = single_sample["input_ids"].shape[1]

        print(f"Input sample {i}: {single_sample}")
        print("*" * 50)

        with torch.no_grad():
            generated_ids = pvmodel.model.generate(**single_sample, **gen_config)
            query_response.append(generated_ids)

        print("Generated Input IDS")
        print(generated_ids)

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
        temperature_scaled = float(gen_config.get("temperature", 1.0)) + temperature_epsilon
        select_logits = logits[:, context_length-1:-1, :] / temperature_scaled
        assert select_logits.shape[1] == gen_index.shape[1], "Selected logits shape does not match the indices shape"
        logprobs_ = torch.gather(select_logits.log_softmax(-1), dim=-1, index=gen_index.unsqueeze(-1)).squeeze(-1)

        # Reference model logprobs
        with torch.no_grad():
            ref_output = reference_model(input_ids=generated_ids, attention_mask=attention_mask)

        ref_logits = ref_output.logits[:, context_length-1:-1, :] / temperature_scaled
        ref_logprobs_ = torch.gather(ref_logits.log_softmax(-1), dim=-1, index=gen_index.unsqueeze(-1)).squeeze(-1)

        ## Get generated values
        values_ = values_[:, context_length - 1 : -1].squeeze(-1)


        ## Get rewards
        # Get token index just before the pad token to get valuable reward in generated text
        pad_mask = generated_ids[:, context_length:] == tokenizer.pad_token_id
        # get first occurrence of pad token
        first_pad = pad_mask.float().argmax(dim=1)
        # if no pad token exists then take the length of generated sequence
        any_pad = pad_mask.any(dim=1)
        first_pad[~any_pad]= pad_mask.shape[1]
        last_useful_tokens = first_pad -1 + context_length # shape of [B]
        sequence_length = first_pad -1
        # Now we get reward using the last_useful_token_index from a model but for now we are hardcoding reward to be equal to
        # normalized length of the response to encourage shorter responses
        rewards_ = last_useful_tokens / gen_config["max_new_tokens"]  # shape of [B]
        print(f"Reward shape is {rewards_.shape}")
        ## Reduce reward of incomplete sentence
        contain_eos_token = torch.any(generated_ids == tokenizer.eos_token_id, dim=-1)
        rewards_[~contain_eos_token] -= 0.1

        ## collect data
        logprobs_list.append(logprobs_)
        ref_logprobs_list.append(ref_logprobs_)
        values_list.append(values_)
        rewards_list.append(rewards_)
        sequence_lengths.append(sequence_length)
        responses.append(gen_index)
        context_length_list.append(int(context_length))

    logprobs = torch.cat(logprobs_list, 0)
    ref_logprobs = torch.cat(ref_logprobs_list, 0)
    values = torch.cat(values_list, 0)
    scores = torch.cat(rewards_list, 0)
    responses = torch.cat(responses, 0)
    query_responses = torch.cat(query_response,0)
    sequence_lengths = torch.cat(sequence_lengths, 0)

    ## get valid log probs, ref probs and values, reward is already valid we could have done it above while getting 
    # valid reward but it is more efficient here since it's calculated in batch
    INVALID_LOGPROB = 1
    response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
    padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
    logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
    ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
    # For a sequence of length T, we expect T + 1 values: one for each token plus one more for the bootstrap.
    sequence_lengths_p1 = sequence_lengths + 1
    padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))
    values = torch.masked_fill(values, padding_mask_p1, 0)

    #compute rewards; KL divergence is used here for reward shaping
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
    # Target for Value models
    returns = advantages + values
    #advantages = masked_whiten(advantages, ~padding_mask)
    advantages = torch.masked_fill(advantages, padding_mask, 0)

    cliprange=cliprange_value =0.2
    vf_coef=0.1
    num_ppo_epochs=200

    for ppo_epoch_idx in tqdm(range(num_ppo_epochs)):
        permutation = torch.randperm(batch_size)
        for i in range(0, batch_size, minibatch_size):
            minibatch_inds = permutation[i : i + minibatch_size]

            mb_advantage = advantages[minibatch_inds]
            mb_responses = responses[minibatch_inds]
            mb_query_responses = query_responses[minibatch_inds]
            mb_logprobs = logprobs[minibatch_inds]
            mb_return = returns[minibatch_inds]
            mb_values = values[minibatch_inds]

            attention_mask = (mb_query_responses != tokenizer.pad_token_id).long()

            outputs = pvmodel(input_ids=mb_query_responses, attention_mask=attention_mask)
            logits, vpred_temp = outputs[0].logits, outputs[1]

            # Handling context length differences of logits
            #logits=torch.stack([logits[i, context_length[i] - 1 : -1] for i in range(logits.size(0))])
            logits = logits[:, context_length - 1 : -1]
            logits /= (float(gen_config.get("temperature", 1.0)) + temperature_epsilon)

            new_logprobs = torch.gather(logits.log_softmax(-1), dim=-1, index=mb_responses.unsqueeze(-1)).squeeze(-1)
            new_logprobs = torch.masked_fill(new_logprobs, padding_mask[minibatch_inds], INVALID_LOGPROB)

            vpred = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
            vpred = torch.masked_fill(vpred, padding_mask_p1[minibatch_inds], 0)

            vpred_clipped = torch.clamp( vpred,mb_values - cliprange_value,mb_values + cliprange_value)
            vf_loss1 = (vpred - mb_return) ** 2
            vf_loss2 = (vpred_clipped - mb_return) ** 2
            vf_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

            logprobs_diff = new_logprobs - mb_logprobs
            ratio = torch.exp(logprobs_diff)

            pg_loss1 = -mb_advantage * ratio
            pg_loss2 = -mb_advantage * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            total_loss = pg_loss + vf_coef * vf_loss
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

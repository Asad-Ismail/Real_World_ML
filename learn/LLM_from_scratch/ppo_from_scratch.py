import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessorList, InfNanRemoveLogitsProcessor
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import set_seed, batch_generation
from transformers import GenerationConfig


SEED = 42
MINIBATCH_SIZE = 2
#MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_ID= "gpt2"
DEVICE_MAP = "auto"
TEMPERATURE_EPSILON = 1e-7


# Optional lower precision to reduce memory if CUDA is available
USE_FP16 = torch.cuda.is_available()
USE_FP16 = False
DTYPE_KWARGS = {"torch_dtype": torch.float16} if USE_FP16 else {}

gen_config = GenerationConfig(
max_new_tokens=50,
do_sample=True,
temperature=0.9,
top_k=50,
top_p=0.95,
) 

class PolicyValueModel(nn.Module):
    def __init__(self, model: AutoModelForCausalLM) -> None:
        super().__init__()
        self.model = model
        self.config = model.config
        self.value_head = nn.Linear(self.config.hidden_size, 1, bias=False)

    def forward(self, **kwargs):
        outputs = self.model(**kwargs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # Last layer hidden state
        values = self.value_head(hidden_states).squeeze(-1)  # (batch, seq_len)
        return outputs, values


def build_models_and_tokenizer():
    policy_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map=DEVICE_MAP, low_cpu_mem_usage=True, **DTYPE_KWARGS
    )
    reference_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map=DEVICE_MAP, low_cpu_mem_usage=True, **DTYPE_KWARGS
    )

    # Infer device from policy model
    infer_device = next(policy_model.parameters()).device

    # Wrap policy with a value head
    pvmodel = PolicyValueModel(policy_model).to(infer_device)
    if USE_FP16:
        pvmodel = pvmodel.to(dtype=torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    gen_config.pad_token_id=tokenizer.pad_token_id
    return policy_model, reference_model, pvmodel, tokenizer, infer_device


def prepare_dataset(examples):
    """Create prompts for the model."""
    # We create a prompt by taking the first 30 words of the review text to see
    examples["query"] = [" ".join(text.split()[:30]) for text in examples["text"]]
    return examples


def tokenize(element):
    outputs = AutoTokenizer.from_pretrained(MODEL_ID)(element["query"], padding=True, truncation=True)
    return outputs


def create_dataloader(tokenizer: AutoTokenizer) -> DataLoader:
    # Select a smaller sample for testing
    dataset = load_dataset("imdb", split="train").shuffle(seed=SEED).select(range(1000))
    dataset = dataset.map(prepare_dataset, batched=True, remove_columns=dataset.column_names)

    # Pre-tokenize data to avoid on-the-fly tokenization
    def _tokenize(element):
        return tokenizer(element["query"], padding=True, truncation=True)

    dataset = dataset.map(_tokenize, batched=True, remove_columns=dataset.column_names)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    dataloader = DataLoader(dataset, batch_size=MINIBATCH_SIZE, shuffle=True)
    return dataloader


def rollout_minibatch(pvmodel: PolicyValueModel, reference_model: AutoModelForCausalLM, tokenizer: AutoTokenizer, batch, infer_device):
    # Query + responses for new policy value and logits generation
    query_response = []
    responses = []
    logprobs_list = []
    ref_logprobs_list = []
    sequence_lengths = []
    values_list = []
    rewards_list = []

    batch_size = batch["input_ids"].shape[0]  # current minibatch size

    ## Generate batch of responses

    queries = batch["input_ids"].to(infer_device)
    # Context length stays same due to padding in queires
    context_length = queries.shape[1]
    # Do generations batch wise so generated lengths are padded to same length also 
    with torch.no_grad():
        generated_ids, _ = batch_generation(pvmodel.model, queries, pad_token_id=tokenizer.pad_token_id, generation_config=gen_config)
    # Do one at a time to generate values and rewards to reduce memory required will keep it that way for now can make 
    # better by doing mini batch instead of single item processing
    for i in range(batch_size):
        generated_id = generated_ids[i].unsqueeze(0)
        query_response.append(generated_id)

        attention_mask = (generated_id != tokenizer.pad_token_id).long()

        with torch.no_grad():
            outputs = pvmodel(input_ids=generated_id, attention_mask=attention_mask)
            logits, values_ = outputs[0].logits, outputs[1]

        #print("Logits shape:", logits.shape)
        #print("Value shape: ", values_.shape)
        #print("**Input text**")
        #print(tokenizer.decode(generated_ids[0][:context_length], skip_special_tokens=False))
        #print("*Generated text*")
        #print(tokenizer.decode(generated_ids[0][context_length:], skip_special_tokens=False))
        # Get logits of generated tokens
        gen_index = generated_id[:, context_length:]  # [B, G]
        temperature_scaled = gen_config.temperature + TEMPERATURE_EPSILON
        select_logits = logits[:, context_length - 1 : -1, :] / temperature_scaled
        assert (
            select_logits.shape[1] == gen_index.shape[1]
        ), "Selected logits shape does not match the indices shape"
        logprobs_ = torch.gather(select_logits.log_softmax(-1), dim=-1, index=gen_index.unsqueeze(-1)).squeeze(-1)

        # Reference model logprobs
        with torch.no_grad():
            ref_output = reference_model(input_ids=generated_id, attention_mask=attention_mask)

        ref_logits = ref_output.logits[:, context_length - 1 : -1, :] / temperature_scaled
        ref_logprobs_ = torch.gather(ref_logits.log_softmax(-1), dim=-1, index=gen_index.unsqueeze(-1)).squeeze(-1)

        # Get generated values
        values_ = values_[:, context_length - 1 : -1].squeeze(-1)

        # Get rewards 12,13,67,50000,5000,5000
        # Get token index just before the pad token to get valuable reward in generated text
        pad_mask = generated_id[:, context_length:] == tokenizer.pad_token_id
        # get first occurrence of pad token
        first_pad = pad_mask.float().argmax(dim=1)
        # if no pad token exists then take the length of generated sequence
        any_pad = pad_mask.any(dim=1)
        first_pad[~any_pad] = pad_mask.shape[1]
        last_useful_tokens = first_pad - 1 + context_length  # shape of [B]
        # sequence length is -1 since we will use it in 0 index tensors
        sequence_length = first_pad - 1
        # Now we get reward using the last_useful_token_index from a model but for now we are hardcoding reward to be equal to
        # normalized length of the response to encourage shorter responses
        rewards_ = 1-(sequence_length / gen_config.max_new_tokens)  # shape of [B]
        #print(f"Reward shape is {rewards_.shape}")
        # Reduce reward of incomplete sentence
        contain_eos_token = torch.any(generated_id == tokenizer.eos_token_id, dim=-1)
        rewards_[~contain_eos_token] -= 0.1

        # collect data
        logprobs_list.append(logprobs_)
        ref_logprobs_list.append(ref_logprobs_)
        values_list.append(values_)
        rewards_list.append(rewards_)
        sequence_lengths.append(sequence_length)
        responses.append(gen_index)

    logprobs = torch.cat(logprobs_list, 0)
    ref_logprobs = torch.cat(ref_logprobs_list, 0)
    values = torch.cat(values_list, 0)
    scores = torch.cat(rewards_list, 0)
    responses = torch.cat(responses, 0)
    query_responses = torch.cat(query_response, 0)
    sequence_lengths = torch.cat(sequence_lengths, 0)

    # get valid log probs, ref probs and values, reward is already valid we could have done it above while getting
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

    return (
        logprobs,
        ref_logprobs,
        values,
        scores,
        responses,
        query_responses,
        sequence_lengths,
        padding_mask,
        padding_mask_p1,
        context_length,
        batch_size,
    )


def compute_advantages_and_returns(
    logprobs,
    ref_logprobs,
    values,
    scores,
    responses,
    sequence_lengths,
    padding_mask,
):
    # compute rewards; KL divergence is used here for reward shaping
    kl_coeff = 0.1
    logr = ref_logprobs - logprobs
    kl = -logr
    non_score_reward = -kl_coeff * kl
    rewards = non_score_reward.clone()
    actual_start = torch.arange(rewards.size(0), device=rewards.device)
    sequence_lengths_p1 = sequence_lengths + 1
    actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)
    rewards[[actual_start, actual_end]] += scores

    # compute advantages and returns
    gamma = 1
    lam = 0.95
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

    advantages = torch.masked_fill(advantages, padding_mask, 0)
    return advantages, returns


def ppo_update(
    pvmodel: PolicyValueModel,
    tokenizer: AutoTokenizer,
    optimizer: torch.optim.Optimizer,
    advantages,
    returns,
    responses,
    query_responses,
    logprobs,
    values,
    padding_mask,
    padding_mask_p1,
    batch_size: int,
    context_length: int,
):
    cliprange = 0.2
    cliprange_value = 0.2
    vf_coef = 0.1
    num_ppo_epochs = 4
    clip_grad_norm = 0.5 

    for _ in tqdm(range(num_ppo_epochs)):
        permutation = torch.randperm(batch_size)
        for i in range(0, batch_size, MINIBATCH_SIZE):
            minibatch_inds = permutation[i : i + MINIBATCH_SIZE]

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
            logits = logits[:, context_length - 1 : -1]
            logits /= (gen_config.temperature + TEMPERATURE_EPSILON)

            new_logprobs = torch.gather(logits.log_softmax(-1), dim=-1, index=mb_responses.unsqueeze(-1)).squeeze(-1)
            new_logprobs = torch.masked_fill(new_logprobs, padding_mask[minibatch_inds], 1)

            vpred = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
            vpred = torch.masked_fill(vpred, padding_mask_p1[minibatch_inds], 0)

            vpred_clipped = torch.clamp(vpred, mb_values - cliprange_value, mb_values + cliprange_value)
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
            torch.nn.utils.clip_grad_norm_(pvmodel.parameters(), clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad()


def run_inference(pvmodel: PolicyValueModel, tokenizer: AutoTokenizer, infer_device) -> None:
    print(f"")
    pvmodel.model.eval()
    sample_prompts = [
        "I recently watched movie gladiator and",
        "The storyline of titanic was engaging because",
        "In my honest opinion, the film momento is ",
    ]
    with torch.no_grad():
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        inputs = tokenizer(sample_prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(infer_device) for k, v in inputs.items()}
        context_lengths = inputs["attention_mask"].sum(dim=1).tolist()

        generations = pvmodel.model.generate(
            **inputs,
            #**gen_config,
            generation_config=gen_config,
        )

    for i, prompt in enumerate(sample_prompts):
        start = int(context_lengths[i])
        generated_continuation = tokenizer.decode(generations[i][start:], skip_special_tokens=True)
        print("Prompt:", prompt)
        print("Generation:", generated_continuation)
        print("-" * 80)


def main() -> None:
    # Set seed
    set_seed(SEED)
    # Build models/tokenizerbatch_generat
    # Not using policy_model direct we wrap it in pv model to use single forward pass
    policyModel, reference_model, pvmodel, tokenizer, infer_device = build_models_and_tokenizer()
    # Data
    dataloader = create_dataloader(tokenizer)
    # Optimizer
    optimizer = torch.optim.Adam(pvmodel.parameters(), lr=5e-7)
    # PPO training loop
    for batch in dataloader:
        (
            logprobs,
            ref_logprobs,
            values,
            scores,
            responses,
            query_responses,
            sequence_lengths,
            padding_mask,
            padding_mask_p1,
            context_length,
            batch_size,
        ) = rollout_minibatch(pvmodel, reference_model, tokenizer, batch, infer_device)

        advantages, returns = compute_advantages_and_returns(
            logprobs,
            ref_logprobs,
            values,
            scores,
            responses,
            sequence_lengths,
            padding_mask,
        )

        ppo_update(
            pvmodel,
            tokenizer,
            optimizer,
            advantages,
            returns,
            responses,
            query_responses,
            logprobs,
            values,
            padding_mask,
            padding_mask_p1,
            batch_size,
            context_length,
        )
    # Run inference
    run_inference(pvmodel, tokenizer, infer_device)


if __name__ == "__main__":
    main()
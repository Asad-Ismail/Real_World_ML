## RLFH throgug PPO



## Loop through dataset

### Roll Out Phase: Collect experience from old policy models
## Generate response for queries using policy model/ training model, output is logits + genrated tokens (selcted via topk, min p and temprature settings)
## Generate response logits already contain logits of new tokens not the context
## response consist of input token + output response
## Select output respose only via known input length 
## Select intesting logits/of selected token only, logprobs = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1) where index are from outputresponse
## Pass input token + output response from reference model/ normally sft frozen version of same model
## Get the output logits to act as reference model, rememeber for autoregressive models we have to shift this logits logits[: context_len-1:-1]
'''
ref_logits = ref_output.logits[:, context_length - 1 : -1]
ref_logits /= args.temperature + 1e-7
ref_logprob = selective_log_softmax(ref_logits, response)
'''

## Reward Model
## Pass input + response to reward model, it returns a scalar
'''
## Get last valid token for the last valid token see -1
sequence_lengths = first_true_indices(query_responses[:, context_length:] == pad_token_id) - 1 + context_length

logits[:,sequence_lengths], # shape[BSx1]

'''
## Value Model
## pass input + response to value model adn get values for each response token
## vlaue shape is [B,response(no query),1]

## Get all responses in replay buffer
'''
responses.append(response)
logprobs.append(logprob)
ref_logprobs.append(ref_logprob)
sequence_lengths.append(sequence_length)
scores.append(score)
values.append(value)
'''


### Calulate reward and value
'''

response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
sequence_lengths_p1 = sequence_lengths + 1
padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))
values = torch.masked_fill(values, padding_mask_p1, 0)

# 4. compute rewards
# Formula used by http://joschu.net/blog/kl-approx.html for the k1 and k3 estimators
# ref logs and log probs both has shape of Bxgeneratedresponse
logr = ref_logprobs - logprobs
kl = -logr if args.kl_estimator == "k1" else (logr.exp() - 1) - logr  # Else statement is k3
non_score_reward = -args.kl_coef * kl
rewards = non_score_reward.clone()
actual_start = torch.arange(rewards.size(0), device=rewards.device)
actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)
rewards[[actual_start, actual_end]] += scores

# 6. compute advantages and returns
lastgaelam = 0
advantages_reversed = []
gen_length = responses.shape[1]
for t in reversed(range(gen_length)):
    nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
    delta = rewards[:, t] + args.gamma * nextvalues - values[:, t]
    lastgaelam = delta + args.gamma * args.lam * lastgaelam
    advantages_reversed.append(lastgaelam)
advantages = torch.stack(advantages_reversed[::-1], axis=1)
returns = advantages + values
advantages = masked_whiten(advantages, ~padding_mask)
advantages = torch.masked_fill(advantages, padding_mask, 0)
empty_cache()
'''





### Training phase of PPo
### Select random mini batch from the replay buffer
## mb_query_response are query+ response of selected mini batch
## mb_response is jsut the response
'''
output, vpred_temp = forward(model, mb_query_responses, processing_class.pad_token_id)
logits = output.logits[:, context_length - 1 : -1]
logits /= args.temperature + 1e-7
new_logprobs = selective_log_softmax(logits, mb_responses)
new_logprobs = torch.masked_fill(
    new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB
)
vpred = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
vpred = torch.masked_fill(vpred, padding_mask_p1[micro_batch_inds], 0)
vpredclipped = torch.clamp(
    vpred,
    mb_values - args.cliprange_value,
    mb_values + args.cliprange_value,
)


output, vpred_temp = forward(model, mb_query_responses, processing_class.pad_token_id)
logits = output.logits[:, context_length - 1 : -1]
logits /= args.temperature + 1e-7
new_logprobs = selective_log_softmax(logits, mb_responses)
new_logprobs = torch.masked_fill(
    new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB
)
vpred = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
vpred = torch.masked_fill(vpred, padding_mask_p1[micro_batch_inds], 0)
vpredclipped = torch.clamp(
    vpred,
    mb_values - args.cliprange_value,
    mb_values + args.cliprange_value,
)
vf_losses1 = torch.square(vpred - mb_return)
vf_losses2 = torch.square(vpredclipped - mb_return)
vf_loss_max = torch.max(vf_losses1, vf_losses2)
vf_loss = 0.5 * masked_mean(vf_loss_max, ~padding_mask_p1[micro_batch_inds])
vf_clipfrac = masked_mean(
    (vf_losses2 > vf_losses1).float(), ~padding_mask_p1[micro_batch_inds]
)
logprobs_diff = new_logprobs - mb_logprobs
ratio = torch.exp(logprobs_diff)
pg_losses = -mb_advantage * ratio
pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
pg_loss_max = torch.max(pg_losses, pg_losses2)
pg_loss = masked_mean(pg_loss_max, ~padding_mask[micro_batch_inds])
loss = pg_loss + args.vf_coef * vf_loss
accelerator.backward(loss)
optimizer.step()
optimizer.zero_grad()

'''

## Go to the first step


### generation step for testing

'''
def generate_completions(self, sampling: bool = False):
    args = self.args
    processing_class = self.processing_class
    generation_config = GenerationConfig(
        max_new_tokens=self.args.response_length,
        temperature=(0.01 + 1e-7),
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
    )

    table = defaultdict(list)
    with unwrap_model_for_generation(
        self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
    ) as unwrapped_model:
        for batch in self.eval_dataloader:
            query = batch["input_ids"]
            with torch.no_grad():
                context_length = query.shape[1]
                query_response, _ = batch_generation(
                    unwrapped_model.policy,
                    query,
                    query.shape[0],
                    processing_class.pad_token_id,
                    generation_config,
                )
                response = query_response[:, context_length:]
                postprocessed_response = response
                if self.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                    postprocessed_response = truncate_response(
                        self.stop_token_id, processing_class.pad_token_id, response
                    )
                table["query"].extend(
                    gather_object(processing_class.batch_decode(query, skip_special_tokens=True))
                )
                table["model response"].extend(
                    gather_object(processing_class.batch_decode(postprocessed_response))
                )

                postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                _, score, _ = get_reward(
                    self.reward_model, postprocessed_query_response, processing_class.pad_token_id, context_length
                )
                table["score"].extend(self.accelerator.gather_for_metrics(score).float().cpu().numpy())

            if sampling:
                break
    df = pd.DataFrame(table)

'''
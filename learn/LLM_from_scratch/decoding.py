import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "Hello, my name is"
inputs = tokenizer(text, return_tensors="pt")


def greedy_decoding(model, input_ids, max_generation=20):
    gen_tokens = 0
    while (input_ids[0, -1] != tokenizer.eos_token_id) and (gen_tokens < max_generation):
        logits = model(input_ids).logits[:, -1, :]  # last token logits
        predicted_token_id = torch.argmax(logits, dim=-1)  # shape (1,)
        input_ids = torch.cat([input_ids, predicted_token_id[:, None]], dim=1)
        gen_tokens += 1
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def beam_search(model, input_ids, max_generation=20, beam_width=2, alpha=0.7):
    beams = [(input_ids, 0.0)]  # (tokens, score)
    
    for _ in range(max_generation):
        new_beams = []
        for seq, score in beams:
            if seq[0, -1].item() == tokenizer.eos_token_id:
                new_beams.append((seq, score))
                continue
            
            logits = model(seq).logits[:, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            
            topk_log_probs, topk_ids = torch.topk(log_probs, beam_width, dim=-1)
            
            for k in range(beam_width):
                next_id = topk_ids[0, k].unsqueeze(0).unsqueeze(0)
                new_seq = torch.cat([seq, next_id], dim=1)
                new_score = score + topk_log_probs[0, k].item()
                
                # length penalty
                length = new_seq.size(1)
                lp = ((5 + length) / 6) ** alpha
                normalized_score = new_score / lp
                
                new_beams.append((new_seq, normalized_score))
        
        new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        beams = new_beams
    
    best_seq, best_score = beams[0]
    return tokenizer.decode(best_seq[0], skip_special_tokens=True), best_score


def topk_sampling(model, input_ids, k=40, temperature=0.7, max_generation=20):
    generated = input_ids
    for _ in range(max_generation):
        logits = model(generated).logits[:, -1, :]  # [B, V]
        logits = logits / temperature
        topk_vals, topk_indices = torch.topk(logits, k, dim=-1)
        probs = torch.softmax(topk_vals, dim=-1)
        next_token = topk_indices[0, torch.multinomial(probs[0], 1)]
        generated = torch.cat([generated, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated[0], skip_special_tokens=True)




def top_p_sampling(model, input_ids, p=0.9, temperature=0.7, max_generation=20):
    generated = input_ids
    for _ in range(max_generation):
        logits = model(generated).logits[:, -1, :]  # [B, V]
        logits = logits / temperature

        # sort logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)

        # mask tokens above cumulative probability
        sorted_logits[cumulative_probs > p] = float('-inf')
        probs = torch.softmax(sorted_logits, dim=-1)

        # sample next token
        next_token = sorted_indices[0, torch.multinomial(probs[0], 1)]
        generated = torch.cat([generated, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated[0], skip_special_tokens=True)



def min_p_sampling(model, input_ids, pbase=0.9, temperature=0.7, max_generation=20):
    generated = input_ids

    for _ in range(max_generation):
        logits = model(generated).logits[:, -1, :]  # [B, V]
        logits = logits / temperature

        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Find max probability
        pmax, _ = torch.max(probs, dim=-1, keepdim=True)  # shape [B,1]

        # Scale threshold
        pscaled = pbase * pmax  # [B,1]

        # Mask tokens below threshold
        mask = probs < pscaled
        logits = logits.clone()
        logits[mask] = float('-inf')

        # Recompute probabilities after masking
        probs = torch.softmax(logits, dim=-1)

        # Sample next token
        next_token = torch.multinomial(probs[0], 1)
        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated[0], skip_special_tokens=True)

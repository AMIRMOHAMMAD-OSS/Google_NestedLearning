import torch
import torch.nn.functional as F

@torch.no_grad()
def sample_autoregressive(model, tokenizer, prompt, max_new_tokens=100, temperature=1.0, top_k=50, top_p=1.0, apply_inner=False, inner_lr=0.0, inner_scale=1.0, device="cuda"):
    model.eval()
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    model.reset_fast_states(B=1, device=device)
    out = tokens.clone()
    for _ in range(max_new_tokens):
        logits = model(out[:, -tokenizer.model_max_length:])[:, -1, :] / max(1e-8, temperature)
        probs = F.softmax(logits, dim=-1)

        # nucleus/top-k
        if top_k > 0:
            topk_vals, topk_idx = torch.topk(probs, min(top_k, probs.size(-1)))
            mask = torch.ones_like(probs) * -float('inf')
            mask.scatter_(1, topk_idx, torch.log(topk_vals))
            logits = mask
            probs = F.softmax(logits, dim=-1)
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            mask = cum <= top_p
            cutoff = mask.sum(dim=-1, keepdim=True)
            keep_idx = sorted_idx[:, :cutoff.item()]
            new_mask = torch.zeros_like(probs)
            new_mask.scatter_(1, keep_idx, 1.0)
            probs = probs * new_mask
            probs = probs / probs.sum(dim=-1, keepdim=True)

        next_id = torch.multinomial(probs, num_samples=1)
        out = torch.cat([out, next_id], dim=1)

        if apply_inner and inner_lr > 0:
            model.apply_inner_updates(inner_lr, inner_scale)
    return tokenizer.decode(out[0].tolist())

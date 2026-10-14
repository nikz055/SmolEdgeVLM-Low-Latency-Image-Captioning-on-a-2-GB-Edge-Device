"""Training losses."""
import torch
import torch.nn.functional as F

from .modules import pool_grid


def relative_mse(a, b):
    """mean((a - b)^2) / Var(b): scale-free distance to the target b."""
    return (a - b).pow(2).mean() / b.var()


def restoration_loss(merged_out, full_out):
    """Stage 1 (R): match the full encoder per patch and after the connector's 2 x 2 pooling. No captions."""
    return relative_mse(merged_out, full_out) + relative_mse(pool_grid(merged_out), pool_grid(full_out))


def caption_loss(lm, z, prompt_emb, ids, mask):
    """Stage 2 (connector): next-token cross-entropy on [visual tokens][prompt][caption] (teacher forcing)."""
    B, Lc = ids.shape
    emb = lm.get_input_embeddings()
    seq = torch.cat([z, prompt_emb.expand(B, -1, -1), emb(ids)], 1)
    attn = torch.cat([torch.ones(B, z.shape[1] + prompt_emb.shape[1], dtype=torch.long, device=ids.device), mask], 1)
    labels = torch.full(seq.shape[:2], -100, dtype=torch.long, device=ids.device)
    labels[:, -Lc:] = torch.where(mask.bool(), ids, torch.full_like(ids, -100))
    return lm(inputs_embeds=seq, attention_mask=attn, labels=labels).loss

"""Shared helpers for the training and evaluation scripts."""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def lr_cosine(step, total, peak, warmup):
    if step < warmup:
        return peak * (step + 1) / warmup
    return peak * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def pack_captions(tok, captions, device, max_len=32):
    """Token ids (+ EOS) padded to the longest caption, with a mask."""
    ids = [tok(c, truncation=True, max_length=max_len - 1).input_ids + [tok.eos_token_id] for c in captions]
    L = max(len(x) for x in ids)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out = torch.full((len(ids), L), pad, dtype=torch.long, device=device)
    mask = torch.zeros((len(ids), L), dtype=torch.long, device=device)
    for b, x in enumerate(ids):
        out[b, :len(x)] = torch.tensor(x, device=device)
        mask[b, :len(x)] = 1
    return out, mask

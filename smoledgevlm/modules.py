"""The two small trained modules: the restoration module R and the pooling connector."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Restore(nn.Module):
    """Restoration module R:  m' = m + W2 GELU(W1 LN(m)).

    Applied to each merged token right after merging (between layers 3 and 4). The last layer starts at zero,
    so R begins as the identity and only learns a correction. 768 -> 192 -> 768, about 0.30 M parameters.
    """

    def __init__(self, dim=768, rank=192):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, rank)
        self.fc2 = nn.Linear(rank, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, m):
        return m + self.fc2(F.gelu(self.fc1(self.norm(m))))


class PoolConnector(nn.Module):
    """2 x 2 average pooling (32 x 32 -> 16 x 16 grid), then Linear -> GELU -> Linear into the language model's
    embedding width. 768 -> 576 -> 576, about 0.78 M parameters."""

    def __init__(self, d_vision=768, d_text=576, grid=16):
        super().__init__()
        self.grid = grid
        self.proj = nn.Sequential(nn.Linear(d_vision, d_text), nn.GELU(), nn.Linear(d_text, d_text))

    def forward(self, feats):
        B, n, d = feats.shape
        s = int(round(n ** 0.5))
        x = F.adaptive_avg_pool2d(feats.transpose(1, 2).reshape(B, d, s, s), (self.grid, self.grid))
        return self.proj(x.flatten(2).transpose(1, 2))


def pool_grid(feats, grid=16):
    """Average-pool [B, N, D] patch features on their square grid to [B, grid * grid, D]."""
    B, n, d = feats.shape
    s = int(round(n ** 0.5))
    x = F.adaptive_avg_pool2d(feats.transpose(1, 2).reshape(B, d, s, s), (grid, grid))
    return x.flatten(2).transpose(1, 2)


def load_state(path, device='cpu'):
    try:
        ck = torch.load(path, map_location=device, weights_only=True)
    except TypeError:                                   # older PyTorch without weights_only
        ck = torch.load(path, map_location=device)
    return ck.get('state_dict', ck) if isinstance(ck, dict) else ck

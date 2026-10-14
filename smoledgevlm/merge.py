"""Merge, size bias and unmerge: the weight-free part of SmolEdgeVLM.

After layer 3 of the frozen encoder, the 1,024 patch tokens (a 32 x 32 grid) are merged into 256 tokens:
  1. Anchors: 256 existing token positions on a fixed 16 x 16 lattice.
  2. Assign:  every other patch joins the most similar anchor (cosine similarity) inside a 7 x 7 window.
  3. Merge:   each group is averaged into one token; the group size n is kept.
Layers 4 to 12 then attend with a log(n) bias on each key, so a token that stands for n patches is weighted
like n patches. Finally each group's output is copied back to every patch in it (unmerge).
"""
import torch
import torch.nn.functional as F


def grid_coords(grid, device=None):
    """(row, col) of every patch on a grid x grid layout, shape [grid * grid, 2]."""
    rows = torch.arange(grid, device=device).repeat_interleave(grid)
    cols = torch.arange(grid, device=device).repeat(grid)
    return torch.stack([rows, cols], -1)


def lattice_anchors(grid, num_tokens, device=None):
    """Indices of the anchor positions: every second row and column first, then the odd-odd positions."""
    coords = grid_coords(grid, device)
    r, c = coords[:, 0], coords[:, 1]
    score = torch.where((r % 2 == 0) & (c % 2 == 0), 2.0, torch.where((r % 2 == 1) & (c % 2 == 1), 1.0, 0.0))
    score = score.float() - 1e-6 * torch.arange(grid * grid, device=device)       # deterministic tie-break
    return torch.topk(score, num_tokens).indices.sort().values


class TokenMerger:
    """Precomputes the anchor lattice and the 7 x 7 neighbourhood once, then merges any [1, N, D] tensor."""

    def __init__(self, grid=32, num_tokens=256, radius=3, device=None):
        self.grid, self.num_tokens, self.radius = grid, num_tokens, radius
        self.N = grid * grid
        coords = grid_coords(grid, device)
        self.anchors = lattice_anchors(grid, num_tokens, device)
        is_anchor = torch.zeros(self.N, dtype=torch.bool, device=device)
        is_anchor[self.anchors] = True
        self.others = (~is_anchor).nonzero().squeeze(1)
        dist = (coords[self.others][:, None, :] - coords[self.anchors][None, :, :]).abs().max(-1).values
        self.outside = dist > radius                         # [N - M, M]: anchors a patch may not join

    def assign(self, h):
        """Group index (0 .. M-1) of every patch. h: [1, N, D]."""
        hn = F.normalize(h[0].float(), dim=-1)
        sim = hn[self.others] @ hn[self.anchors].t()
        sim = sim.masked_fill(self.outside, float('-inf'))
        assign = torch.empty(self.N, dtype=torch.long, device=h.device)
        assign[self.anchors] = torch.arange(self.num_tokens, device=h.device)
        assign[self.others] = sim.argmax(1)
        return assign

    def merge(self, h):
        """Returns merged tokens [1, M, D], group sizes [M] and the assignment [N]."""
        assign = self.assign(h)
        x = h[0]
        sizes = torch.zeros(self.num_tokens, device=h.device, dtype=x.dtype).index_add_(
            0, assign, torch.ones(self.N, device=h.device, dtype=x.dtype))
        merged = torch.zeros(self.num_tokens, x.shape[-1], device=h.device, dtype=x.dtype).index_add_(0, assign, x)
        return (merged / sizes[:, None])[None], sizes, assign


def size_bias(sizes):
    """log(n) for every key token, shaped to add onto attention scores [B, heads, queries, keys]."""
    return torch.log(sizes)[None, None, None, :]


def unmerge(z, assign):
    """Copy each group's output back to all of its patch positions: [1, M, D] -> [1, N, D]."""
    return z[:, assign, :]

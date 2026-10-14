"""CPU checks of the weight-free merge step and the two trained modules (no model downloads)."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from smoledgevlm.merge import TokenMerger, lattice_anchors, size_bias, unmerge  # noqa: E402
from smoledgevlm.modules import PoolConnector, Restore, load_state  # noqa: E402

WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'weights')


def test_lattice_has_256_anchors_on_even_rows_and_columns():
    a = lattice_anchors(32, 256)
    assert a.numel() == 256 and torch.unique(a).numel() == 256
    assert bool(((a // 32) % 2 == 0).all() and ((a % 32) % 2 == 0).all())


def test_merge_keeps_every_patch_and_stays_local():
    torch.manual_seed(0)
    tm = TokenMerger(32, 256, 3)
    h = torch.randn(1, 1024, 768)
    m, sizes, assign = tm.merge(h)
    assert m.shape == (1, 256, 768) and int(sizes.sum()) == 1024 and bool((sizes >= 1).all())
    rows, cols = torch.arange(1024) // 32, torch.arange(1024) % 32
    ar, ac = tm.anchors[assign] // 32, tm.anchors[assign] % 32
    assert int(torch.maximum((rows - ar).abs(), (cols - ac).abs()).max()) <= 3
    j = int(assign[100])
    assert torch.allclose(m[0, j], h[0][assign == j].mean(0), atol=1e-5)


def test_unmerge_and_size_bias_shapes():
    tm = TokenMerger(32, 256, 3)
    m, sizes, assign = tm.merge(torch.randn(1, 1024, 768))
    y = unmerge(m, assign)
    assert y.shape == (1, 1024, 768) and torch.equal(y[0, tm.anchors], m[0])
    assert size_bias(sizes).shape == (1, 1, 1, 256)


def test_restore_starts_as_identity_and_has_0_30M_parameters():
    R = Restore(768, 192)
    x = torch.randn(1, 256, 768)
    assert torch.equal(R(x), x)
    assert sum(p.numel() for p in R.parameters()) == 297408


def test_trained_weights_load():
    R = Restore(768, 192)
    R.load_state_dict(load_state(os.path.join(WEIGHTS, 'restore_R.pt')))
    for name in ('connector_smoledgevlm.pt', 'connector_full_encoder.pt'):
        c = PoolConnector(768, 576, 16)
        c.load_state_dict(load_state(os.path.join(WEIGHTS, name)))
        assert c(torch.randn(1, 1024, 768)).shape == (1, 256, 576)
        assert sum(p.numel() for p in c.parameters()) == 775296

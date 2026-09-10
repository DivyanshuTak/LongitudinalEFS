"""Smoke tests for the EFS head -- no embeddings on disk needed."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch

from src.data.dataset import collate_trajectories
from src.data.clinical import CLINICAL_DIM
from src.models.mil_pooling import ABMILPool
from src.models.time_encoding import FourierTimeEmbedding
from src.models.trajectory_model import LongitudinalEFSModel

M = 3


# --- ABMILPool ---------------------------------------------------------------

def _packed(lens, dim=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randn(sum(lens), dim, generator=g)
    seg_idx = torch.repeat_interleave(torch.arange(len(lens)), torch.tensor(lens))
    return tokens, seg_idx


def test_pool_attention_sums_to_one_per_segment():
    tokens, seg_idx = _packed([5, 1, 40])
    _, attn = ABMILPool(dim=16, hidden=8)(tokens, seg_idx, 3)
    sums = torch.zeros(3).index_add_(0, seg_idx, attn)
    assert torch.allclose(sums, torch.ones(3), atol=1e-5)


def test_pool_is_permutation_invariant_within_a_segment():
    tokens, seg_idx = _packed([6, 6])
    pool = ABMILPool(dim=16, hidden=8).eval()
    perm = torch.cat([torch.randperm(6), 6 + torch.randperm(6)])
    with torch.no_grad():
        a, _ = pool(tokens, seg_idx, 2)
        b, _ = pool(tokens[perm], seg_idx, 2)
    assert torch.allclose(a, b, atol=1e-5)


def test_pool_segments_are_independent():
    """Perturbing segment 1's tokens must not move segment 0's pooled vector."""
    tokens, seg_idx = _packed([4, 4])
    pool = ABMILPool(dim=16, hidden=8).eval()
    other = tokens.clone()
    other[4:] = torch.randn(4, 16)
    with torch.no_grad():
        a, _ = pool(tokens, seg_idx, 2)
        b, _ = pool(other, seg_idx, 2)
    assert torch.allclose(a[0], b[0], atol=1e-6)
    assert not torch.allclose(a[1], b[1], atol=1e-3)


def test_pool_single_token_segment_gets_weight_one():
    tokens, seg_idx = _packed([1, 3])
    pooled, attn = ABMILPool(dim=16, hidden=8).eval()(tokens, seg_idx, 2)
    assert torch.allclose(attn[0], torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(pooled[0], tokens[0], atol=1e-5)


# --- FourierTimeEmbedding ----------------------------------------------------

def test_time_embedding_rejects_odd_dim():
    with pytest.raises(ValueError):
        FourierTimeEmbedding(dim=15)


def test_time_embedding_is_parameter_free_and_zero_is_fixed():
    embed = FourierTimeEmbedding(dim=16)
    assert list(embed.parameters()) == []
    out = embed(torch.zeros(2, 4))
    assert out.shape == (2, 4, 16)
    sin_half, cos_half = out.chunk(2, dim=-1)
    assert torch.allclose(sin_half, torch.zeros_like(sin_half), atol=1e-6)
    assert torch.allclose(cos_half, torch.ones_like(cos_half), atol=1e-6)


# --- collate -----------------------------------------------------------------

def _item(T, lens_per_series, dim=16):
    return {
        "tokens": torch.randn(sum(lens_per_series), dim),
        "seg_lens": torch.tensor(lens_per_series, dtype=torch.long),
        "days": torch.arange(T, dtype=torch.float32) * 100,
        "label": torch.tensor(1.0),
        "clinical": torch.zeros(CLINICAL_DIM),
        "num_scans": T,
    }


def test_collate_maps_segments_to_the_right_slots():
    T_max = 4
    batch = [_item(2, [3, 4, 5, 6, 7, 8]), _item(1, [2, 2, 2])]
    out = collate_trajectories(batch, max_seq_len=T_max, n_modalities=M)
    assert out["tokens"].shape[0] == 33 + 6
    assert out["seg_idx"].shape == (39,)
    assert out["seg_idx"].max().item() == 8          # 6 segments + 3 segments - 1
    # sample 0 fills slots 0..5, sample 1 starts at its own trajectory's offset
    assert out["seg_slot"][:6].tolist() == [0, 1, 2, 3, 4, 5]
    assert out["seg_slot"][6:].tolist() == [T_max * M, T_max * M + 1, T_max * M + 2]
    assert out["mask"].tolist() == [[1, 1, 0, 0], [1, 0, 0, 0]]


def test_collate_rejects_overlong_trajectory():
    with pytest.raises(ValueError, match="max_seq_len"):
        collate_trajectories([_item(5, [1] * 15)], max_seq_len=4, n_modalities=M)


# --- LongitudinalEFSModel ----------------------------------------------------

def _model(**kw):
    cfg = dict(feature_dim=16, n_modalities=M, pool_hidden=8, fusion_dim=12,
               time_dim=4, hidden=8, dropout=0.0)
    cfg.update(kw)
    return LongitudinalEFSModel(**cfg)


def _batch(T_max=4, dim=16, seed=0):
    torch.manual_seed(seed)
    batch = [_item(4, [3, 4, 5, 6, 7, 8, 2, 2, 2, 9, 9, 9], dim), _item(2, [4] * 6, dim)]
    return collate_trajectories(batch, max_seq_len=T_max, n_modalities=M)


@pytest.mark.parametrize("temporal", ["gru", "lstm", "transformer"])
@pytest.mark.parametrize("readout", ["last", "attn_pool", "mean"])
def test_model_shapes_and_backward(temporal, readout):
    model = _model(temporal=temporal, readout=readout, nhead=4)
    b = _batch()
    b["tokens"].requires_grad_(True)
    logit, weights, attn = model(b["tokens"], b["seg_idx"], b["seg_slot"], b["mask"], b["days"])
    assert logit.shape == (2,)
    assert weights.shape == (2, 4)
    assert attn.shape == (b["tokens"].shape[0],)

    logit.sum().backward()
    assert b["tokens"].grad.abs().sum().item() > 0
    assert model.pool.w.weight.grad.abs().sum().item() > 0


@pytest.mark.parametrize("temporal", ["gru", "lstm", "transformer"])
def test_readout_weights_are_zero_on_padding(temporal):
    model = _model(temporal=temporal, readout="attn_pool", nhead=4).eval()
    b = _batch()
    with torch.no_grad():
        _, weights, _ = model(b["tokens"], b["seg_idx"], b["seg_slot"], b["mask"], b["days"])
    assert torch.allclose(weights * (1 - b["mask"]), torch.zeros_like(weights), atol=1e-6)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-5)


def test_last_readout_picks_the_final_valid_timestep():
    """The label is defined as progression within 365 days of the LAST scan, so
    `last` must land on that scan and not on a padded slot."""
    model = _model(readout="last").eval()
    b = _batch()
    with torch.no_grad():
        _, weights, _ = model(b["tokens"], b["seg_idx"], b["seg_slot"], b["mask"], b["days"])
    assert weights.tolist() == [[0, 0, 0, 1], [0, 1, 0, 0]]


@pytest.mark.parametrize("temporal", ["gru", "lstm", "transformer"])
def test_padded_timesteps_cannot_influence_the_logit(temporal):
    """A shorter trajectory's logit must not change when the padding grows."""
    model = _model(temporal=temporal, readout="last", nhead=4).eval()
    torch.manual_seed(0)
    item = _item(2, [4] * 6, 16)
    with torch.no_grad():
        short = collate_trajectories([item], max_seq_len=2, n_modalities=M)
        long = collate_trajectories([item], max_seq_len=6, n_modalities=M)
        a, _, _ = model(short["tokens"], short["seg_idx"], short["seg_slot"], short["mask"], short["days"])
        c, _, _ = model(long["tokens"], long["seg_idx"], long["seg_slot"], long["mask"], long["days"])
    assert torch.allclose(a, c, atol=1e-5)


def test_model_without_time_features():
    model = _model(time_dim=0)
    b = _batch()
    logit, _, _ = model(b["tokens"], b["seg_idx"], b["seg_slot"], b["mask"], days=None)
    assert logit.shape == (2,)


def test_model_requires_days_when_time_dim_set():
    model = _model(time_dim=4)
    b = _batch()
    with pytest.raises(ValueError, match="days"):
        model(b["tokens"], b["seg_idx"], b["seg_slot"], b["mask"], days=None)


def test_concat_first_fusion_runs():
    model = _model(fusion="concat_first")
    b = _batch()
    logit, _, _ = model(b["tokens"], b["seg_idx"], b["seg_slot"], b["mask"], b["days"])
    assert logit.shape == (2,)
    assert model.proj.in_features == 16 * M


def test_rejects_bad_config():
    with pytest.raises(ValueError):
        _model(temporal="rnn")
    with pytest.raises(ValueError):
        _model(readout="max")
    with pytest.raises(ValueError, match="divide evenly"):
        _model(fusion_dim=10)
    with pytest.raises(ValueError, match="nhead"):
        _model(temporal="transformer", fusion_dim=12, time_dim=4, nhead=5)

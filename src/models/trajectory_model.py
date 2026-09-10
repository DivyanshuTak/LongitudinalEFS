"""Longitudinal EFS head over frozen NeuroVFM patch tokens.

    packed tokens -> pooler (shared across modalities) -> (B, T, M, 768)
      -> per-modality Linear -> concat -> (B, T, fusion_dim)
      -> [Fourier(days since first scan)]
      -> GRU / LSTM / TransformerEncoder -> readout -> Linear -> logit

The label is anchored to the last scan, so the default readout is the last valid
timestep and there is no deep supervision: earlier timesteps have their own
365-day windows and would be mislabeled.
"""
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .mil_pooling import ABMILPool
from .time_encoding import FourierTimeEmbedding
from .vfm_pooler import VFMPool

FUSIONS = ("per_modality", "concat_first")
TEMPORALS = ("gru", "lstm", "transformer")
READOUTS = ("last", "attn_pool", "mean")


class MaskedAttentionPool(nn.Module):
    """Learned-query attention pooling over the time dimension, masked to valid scans."""

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim) * dim ** -0.5)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        scores = (x @ self.query) / (x.shape[-1] ** 0.5)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(scores.float(), dim=1).to(x.dtype)
        return (weights.unsqueeze(-1) * x).sum(dim=1), weights


class LongitudinalEFSModel(nn.Module):
    def __init__(
        self,
        feature_dim: int = 768,
        n_modalities: int = 3,
        pool_hidden: int = 512,
        pool_gated: bool = False,
        pool_dropout: float = 0.0,
        pool_source: str = "abmil",
        vfm_repo: str = "mlinslab/neurovfm-dx-mri",
        vfm_label: str = "low_grade_glioma",
        vfm_freeze: bool = True,
        fusion: str = "per_modality",
        fusion_dim: int = 768,
        time_dim: int = 16,
        min_period_days: float = 30.0,
        max_period_days: float = 2000.0,
        temporal: str = "gru",
        hidden: int = 128,
        num_layers: int = 1,
        nhead: int = 8,
        readout: str = "last",
        dropout: float = 0.1,
        clinical_dim: int = 0,
        clinical_hidden: int = 32,
        clinical_dropout: float = 0.3,
        use_imaging: bool = True,
    ):
        super().__init__()
        for name, val, allowed in (("fusion", fusion, FUSIONS),
                                   ("temporal", temporal, TEMPORALS),
                                   ("readout", readout, READOUTS)):
            if val not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {val!r}")
        if fusion == "per_modality" and fusion_dim % n_modalities != 0:
            raise ValueError(
                f"fusion_dim ({fusion_dim}) must divide evenly across "
                f"n_modalities ({n_modalities}) for per-modality fusion"
            )
        if not use_imaging and clinical_dim <= 0:
            raise ValueError("a clinical-only model needs clinical_dim > 0")
        self.use_imaging = use_imaging
        self.n_modalities = n_modalities
        self.fusion = fusion
        self.temporal_kind = temporal
        self.readout_kind = readout

        # one pooler shared across modalities
        if pool_source == "vfm":
            self.pool = VFMPool(vfm_repo, vfm_label, vfm_freeze)
        elif pool_source == "abmil":
            self.pool = ABMILPool(feature_dim, pool_hidden, pool_gated, pool_dropout)
        else:
            raise ValueError(f"pool.source must be abmil|vfm, got {pool_source!r}")

        if fusion == "per_modality":
            per = fusion_dim // n_modalities
            self.proj = nn.ModuleList(nn.Linear(feature_dim, per) for _ in range(n_modalities))
        else:
            self.proj = nn.Linear(feature_dim * n_modalities, fusion_dim)

        self.time_embed = FourierTimeEmbedding(time_dim, min_period_days, max_period_days) \
            if time_dim > 0 else None
        seq_dim = fusion_dim + time_dim

        if temporal == "transformer":
            if seq_dim % nhead != 0:
                raise ValueError(f"nhead ({nhead}) must divide fusion_dim+time_dim ({seq_dim})")
            layer = nn.TransformerEncoderLayer(
                d_model=seq_dim, nhead=nhead, dim_feedforward=4 * hidden,
                dropout=dropout, batch_first=True, norm_first=True,
            )
            self.seq = nn.TransformerEncoder(layer, num_layers=num_layers)
            out_dim = seq_dim
        else:
            rnn = nn.GRU if temporal == "gru" else nn.LSTM
            self.seq = rnn(seq_dim, hidden, num_layers=num_layers, batch_first=True)
            out_dim = hidden

        self.time_pool = MaskedAttentionPool(out_dim) if readout == "attn_pool" else None
        self.dropout = nn.Dropout(dropout)
        if not use_imaging:
            # clinical-only: no imaging parameters at all
            self.pool = self.proj = self.seq = self.time_pool = self.time_embed = None
            out_dim = 0

        # static covariates join after the sequence model; LayerNorm on both
        # sides keeps either branch from dominating by scale
        if clinical_dim > 0:
            self.clinical = nn.Sequential(
                nn.Linear(clinical_dim, clinical_hidden), nn.ReLU(),
                nn.Dropout(clinical_dropout), nn.LayerNorm(clinical_hidden),
            )
            self.img_norm = nn.LayerNorm(out_dim) if use_imaging else None
            head_dim = out_dim + clinical_hidden
        else:
            self.clinical = self.img_norm = None
            head_dim = out_dim
        self.classifier = nn.Linear(head_dim, 1)

    def _scan_features(self, tokens, seg_idx, seg_slot, n_slots):
        """Packed tokens -> (B*T_max*M, feature_dim); padded slots stay zero."""
        pooled, token_attn = self.pool(tokens, seg_idx, seg_slot.numel())
        feats = torch.zeros(n_slots, pooled.shape[-1], dtype=pooled.dtype, device=pooled.device)
        feats.index_copy_(0, seg_slot, pooled)
        return feats, token_attn

    def _fuse(self, scan_feats):
        """(B, T, M, D) -> (B, T, fusion_dim)."""
        if self.fusion == "per_modality":
            return torch.cat([p(scan_feats[:, :, m]) for m, p in enumerate(self.proj)], dim=-1)
        return self.proj(scan_feats.flatten(start_dim=2))

    def _run_sequence(self, x, mask):
        if self.temporal_kind == "transformer":
            return self.seq(x, src_key_padding_mask=(mask == 0))
        lengths = mask.sum(dim=1).clamp(min=1).long().cpu()
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.seq(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=mask.shape[1])
        return out

    def _readout(self, seq_out, mask):
        if self.readout_kind == "attn_pool":
            return self.time_pool(seq_out, mask)
        if self.readout_kind == "mean":
            counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            weights = mask / counts
            return (seq_out * weights.unsqueeze(-1)).sum(dim=1), weights
        last = mask.sum(dim=1).clamp(min=1).long() - 1                   # (B,)
        weights = torch.zeros_like(mask).scatter_(1, last.unsqueeze(1), 1.0)
        return seq_out.gather(1, last.view(-1, 1, 1).expand(-1, 1, seq_out.shape[-1])).squeeze(1), weights

    def forward(self, tokens, seg_idx, seg_slot, mask, days: Optional[torch.Tensor] = None,
                clinical: Optional[torch.Tensor] = None):
        """
        tokens:   (N, feature_dim)  all series' patch tokens in the batch, packed
        seg_idx:  (N,)              which packed segment each token belongs to
        seg_slot: (n_seg,)          flat (b, t, m) slot each segment writes to
        mask:     (B, T), 1.0=valid scan, 0.0=padding
        days:     (B, T)            days since the trajectory's first scan

        Returns (logit: (B,), time_weights: (B, T), token_attn: (N,)).
        """
        if not self.use_imaging:
            if clinical is None:
                raise ValueError("clinical is required for a clinical-only model")
            return self.classifier(self.clinical(clinical)).squeeze(-1), None, None

        B, T = mask.shape
        M = self.n_modalities
        feats, token_attn = self._scan_features(tokens, seg_idx, seg_slot, B * T * M)
        x = self._fuse(feats.view(B, T, M, -1))

        if self.time_embed is not None:
            if days is None:
                raise ValueError("days is required when time_dim > 0")
            x = torch.cat([x, self.time_embed(days)], dim=-1)

        seq_out = self._run_sequence(x, mask)
        pooled, time_weights = self._readout(seq_out, mask)
        pooled = self.dropout(pooled)
        if self.clinical is not None:
            if clinical is None:
                raise ValueError("clinical is required when clinical_dim > 0")
            pooled = torch.cat([self.img_norm(pooled), self.clinical(clinical)], dim=-1)
        logit = self.classifier(pooled).squeeze(-1)
        return logit, time_weights, token_attn

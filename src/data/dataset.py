"""Trajectory dataset over precomputed NeuroVFM patch-token embeddings.

CSV: pat_id, scandate, label. scandate is dash-joined and ordered, either
YYYYMMDD or integer days (CBTN age in days, date_format="int").

Embeddings: {emb_dir}/{pat_id}_{scandate}_v{view}.npz with tokens (N, 768) and
series_cu_seqlens (M+1,) splitting them into t1ce, t2, flair. Views 1..n-1 are
spatial augmentations of view 0; train samples one per epoch, val pins view 0.

Tokens are packed, not padded: per-series counts span 5..2859.
"""
import os
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .clinical import CLINICAL_COLS, CLINICAL_DIM, encode_row

MODALITY_ORDER = ("t1ce", "t2", "flair")


class EmbeddingTrajectoryDataset(Dataset):
    def __init__(self, csv_path: str, emb_dir: str = None, n_views: int = 5,
                 random_view: bool = False, n_modalities: int = len(MODALITY_ORDER),
                 date_format: str = "yyyymmdd", cohorts: Dict[str, Dict] = None,
                 load_tokens: bool = True, modalities=None):
        """modalities: subset of MODALITY_ORDER to keep (e.g. ["flair"]); default all three."""
        self.df = pd.read_csv(csv_path, dtype={"pat_id": str, "scandate": str})
        self.df = self.df[self.df["pat_id"].str.strip().astype(bool)].reset_index(drop=True)
        self.default = {"emb_dir": emb_dir, "date_format": date_format, "n_views": n_views}
        # n_views is per-cohort: PBTC has v0 only
        self.cohorts = ({c: {**self.default, **s} for c, s in cohorts.items()}
                        if cohorts else None)

        if self.cohorts and "cohort" in self.df.columns:
            unknown = set(self.df["cohort"]) - set(self.cohorts)
            if unknown:
                raise ValueError(f"CSV has cohorts with no config entry: {sorted(unknown)}")
            specs = [self.cohorts[c] for c in self.df["cohort"].unique()]
        else:
            specs = [self.default]
        for s in specs:
            if s["date_format"] not in ("yyyymmdd", "int"):
                raise ValueError(f"date_format must be 'yyyymmdd' or 'int', got {s['date_format']!r}")
            if load_tokens and not s["emb_dir"]:
                raise ValueError("emb_dir is required")

        self.random_view = random_view
        self.load_tokens = load_tokens
        self.has_clinical = all(c in self.df.columns for c in CLINICAL_COLS)
        self.modalities = list(modalities) if modalities else list(MODALITY_ORDER)
        self.mod_idx = [MODALITY_ORDER.index(m) for m in self.modalities]
        self.n_modalities = len(self.mod_idx)
        if load_tokens:
            self._check_coverage()

    def _spec(self, row) -> Dict:
        if self.cohorts and "cohort" in self.df.columns:
            return self.cohorts[row["cohort"]]
        return self.default

    def _path(self, emb_dir: str, pat_id: str, scandate: str, view: int) -> str:
        return os.path.join(emb_dir, f"{pat_id}_{scandate}_v{view}.npz")

    def _check_coverage(self) -> None:
        missing = [
            f"{r.pat_id}_{d}" for _, r in self.df.iterrows()
            for d in str(r["scandate"]).split("-")
            if not os.path.exists(self._path(self._spec(r)["emb_dir"], r["pat_id"], d, 0))
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} scan(s) in the CSV have no embedding: "
                f"{missing[:10]}{' ...' if len(missing) > 10 else ''}. "
                f"Run scripts/check_coverage.py to list them all."
            )
        # catch a wrong n_views up front rather than mid-epoch
        if not self.random_view:
            return
        for _, r in self.df.drop_duplicates("cohort" if self.cohorts and
                                            "cohort" in self.df.columns else "pat_id").iterrows():
            s = self._spec(r)
            top = s["n_views"] - 1
            d = str(r["scandate"]).split("-")[0]
            if not os.path.exists(self._path(s["emb_dir"], r["pat_id"], d, top)):
                raise FileNotFoundError(
                    f"n_views={s['n_views']} but {r['pat_id']}_{d}_v{top}.npz is absent "
                    f"under {s['emb_dir']} -- set n_views for this cohort to the number "
                    f"of views actually extracted."
                )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        pat_id = str(row["pat_id"])
        spec = self._spec(row)
        emb_dir, date_format = spec["emb_dir"], spec["date_format"]
        # int scandates sort numerically
        key = int if date_format == "int" else None
        scandates: List[str] = sorted(str(row["scandate"]).split("-"), key=key)
        view = int(torch.randint(spec["n_views"], (1,))) if self.random_view else 0

        if date_format == "int":
            parsed = [int(d) for d in scandates]
            days = [float(p - parsed[0]) for p in parsed]
        else:
            parsed = [datetime.strptime(d, "%Y%m%d") for d in scandates]
            days = [float((p - parsed[0]).days) for p in parsed]

        if not self.load_tokens:
            return {
                "tokens": torch.zeros((0, 768), dtype=torch.float32),
                "seg_lens": torch.zeros(len(scandates) * self.n_modalities, dtype=torch.long),
                "days": torch.tensor(days, dtype=torch.float32),
                "label": torch.tensor(float(row["label"]), dtype=torch.float32),
                "clinical": torch.tensor(encode_row(row), dtype=torch.float32),
                "num_scans": len(scandates),
            }

        tokens, seg_lens = [], []
        for d in scandates:
            npz = np.load(self._path(emb_dir, pat_id, d, view))
            tok, cu = npz["tokens"], npz["series_cu_seqlens"]
            if len(cu) - 1 != len(MODALITY_ORDER):
                raise ValueError(
                    f"{pat_id}_{d}_v{view}.npz has {len(cu)-1} series, expected "
                    f"{len(MODALITY_ORDER)} ({', '.join(MODALITY_ORDER)})"
                )
            for m in self.mod_idx:
                tokens.append(torch.from_numpy(tok[cu[m]:cu[m + 1]].astype(np.float32)))
                seg_lens.append(int(cu[m + 1] - cu[m]))

        return {
            "tokens": torch.cat(tokens, dim=0),                        # (N_i, 768)
            "seg_lens": torch.tensor(seg_lens, dtype=torch.long),      # (T*M,)
            "days": torch.tensor(days, dtype=torch.float32),           # (T,)
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "clinical": torch.tensor(encode_row(row) if self.has_clinical
                                     else [0.0] * CLINICAL_DIM, dtype=torch.float32),
            "num_scans": len(scandates),
        }


def collate_trajectories(batch, max_seq_len: int, n_modalities: int = len(MODALITY_ORDER)):
    """Pack tokens across the batch; pad only the (much smaller) time dimension.

    Returns:
        tokens:   (N_total, 768)
        seg_idx:  (N_total,)  segment each token belongs to, in [0, n_seg)
        seg_slot: (n_seg,)    flat b*T_max*M + t*M + m slot each segment fills
        mask:     (B, T_max)
        days:     (B, T_max)
        label:    (B,)
    """
    B, T_max, M = len(batch), max_seq_len, n_modalities
    mask = torch.zeros((B, T_max), dtype=torch.float32)
    days = torch.zeros((B, T_max), dtype=torch.float32)
    labels = torch.zeros((B,), dtype=torch.float32)

    all_lens, slots = [], []
    for b, item in enumerate(batch):
        T = item["num_scans"]
        if T > T_max:
            raise ValueError(
                f"Trajectory has {T} scans but max_seq_len={T_max}. "
                f"Increase data.max_seq_len in config."
            )
        mask[b, :T] = 1.0
        days[b, :T] = item["days"]
        labels[b] = item["label"]
        all_lens.append(item["seg_lens"])
        slots.extend(b * T_max * M + t * M + m for t in range(T) for m in range(M))

    seg_lens = torch.cat(all_lens)
    seg_slot = torch.tensor(slots, dtype=torch.long)
    seg_idx = torch.repeat_interleave(torch.arange(seg_lens.numel()), seg_lens)

    return {
        "tokens": torch.cat([item["tokens"] for item in batch], dim=0),
        "seg_idx": seg_idx,
        "seg_slot": seg_slot,
        "mask": mask,
        "days": days,
        "label": labels,
        "clinical": torch.stack([item["clinical"] for item in batch]),
    }

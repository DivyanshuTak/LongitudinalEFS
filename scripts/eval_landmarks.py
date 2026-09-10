"""Score one checkpoint on a landmark CSV (one row per patient per landmark).
Reports AUROC/AUPRC per landmark year (row bootstrap) and pooled over all rows
(patient-clustered bootstrap).

  python scripts/eval_landmarks.py --ckpt X.ckpt --csv data/eval_csvs/cbtn/cbtn_efs1y_landmarks.csv \
      --emb_dir data/embeddings/cbtn --date_format int --out eval_out/landmarks_<name>
"""
import argparse
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import EmbeddingTrajectoryDataset, collate_trajectories  # noqa: E402
from src.infer import run_inference                                             # noqa: E402
from src.lightning_module import EFSLightningModule                             # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--csv", required=True)
ap.add_argument("--emb_dir", required=True)
ap.add_argument("--date_format", default="int", choices=["yyyymmdd", "int"])
ap.add_argument("--out", required=True)
ap.add_argument("--gpu", default="0")
ap.add_argument("--n_boot", type=int, default=2000)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--max_seq_len", type=int, default=6)
ap.add_argument("--pool_from_year", type=int, default=8, help="landmarks >= this are one bin")
ap.add_argument("--group_col", default="landmark_year", help="row-strata column")
a = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def boot_ci(y, p, groups, rng, n_boot):
    """AUROC/AUPRC CIs; resample patients (groups) with replacement, keeping all their rows."""
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    aucs, aps = [], []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by_g[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        if len(np.unique(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], p[idx])); aps.append(average_precision_score(y[idx], p[idx]))
    return np.percentile(aucs, [2.5, 97.5]), np.percentile(aps, [2.5, 97.5])


hp = torch.load(a.ckpt, map_location="cpu", weights_only=False)["hyper_parameters"]
model = EFSLightningModule.load_from_checkpoint(a.ckpt, config=dict(hp), map_location=device).eval().to(device)
n_mod = hp["model"].get("n_modalities", 3)
ds = EmbeddingTrajectoryDataset(a.csv, a.emb_dir, random_view=False, n_modalities=n_mod, date_format=a.date_format,
                                modalities=hp.get("data", {}).get("modalities"))
loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4,
                    collate_fn=partial(collate_trajectories, max_seq_len=a.max_seq_len, n_modalities=n_mod))
p, y = run_inference(model, loader, device)
df = ds.df.copy()
df["prob"] = p.numpy()
assert (df["label"].astype(int).values == y.numpy().astype(int)).all()
os.makedirs(a.out, exist_ok=True)
df.to_csv(f"{a.out}/predictions.csv", index=False)

rng = np.random.default_rng(a.seed)
yv, pv, gv = df["label"].astype(int).values, df["prob"].values, df["pat_id"].values
lines = [f"checkpoint: {Path(a.ckpt).stem}", f"csv: {a.csv}",
         f"{len(df)} rows, {df.pat_id.nunique()} patients, {yv.sum()} positives ({yv.mean():.1%})", ""]
lines.append(f"{a.group_col:>10} {'rows':>5} {'pos':>4} {'AUROC':>7} {'95% CI':>16} {'AUPRC':>7} {'95% CI':>16}")
year = df[a.group_col].values
bins = [(t, year == t) for t in range(1, a.pool_from_year)] + [(f"{a.pool_from_year}+", year >= a.pool_from_year)]
for name, m in bins:
    if m.sum() == 0 or len(np.unique(yv[m])) < 2:
        continue
    (lo, hi), (plo, phi) = boot_ci(yv[m], pv[m], gv[m], rng, a.n_boot)
    lines.append(f"{str(name):>10} {m.sum():5d} {yv[m].sum():4d} {roc_auc_score(yv[m], pv[m]):7.3f} "
                 f"[{lo:.3f}, {hi:.3f}] {average_precision_score(yv[m], pv[m]):7.3f} [{plo:.3f}, {phi:.3f}]")
(lo, hi), (plo, phi) = boot_ci(yv, pv, gv, rng, a.n_boot)
lines.append(f"{'pooled':>10} {len(yv):5d} {yv.sum():4d} {roc_auc_score(yv, pv):7.3f} [{lo:.3f}, {hi:.3f}] "
             f"{average_precision_score(yv, pv):7.3f} [{plo:.3f}, {phi:.3f}]   (patient-clustered bootstrap)")
open(f"{a.out}/metrics.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

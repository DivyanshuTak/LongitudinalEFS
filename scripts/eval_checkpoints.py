"""Score every checkpoint in a directory on one CSV. Reports per-checkpoint metrics
(AUROC with bootstrap CI) plus a mean-probability ensemble per architecture."""
import argparse
import glob
import json
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             roc_auc_score)
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import EmbeddingTrajectoryDataset, collate_trajectories
from src.infer import run_inference
from src.lightning_module import EFSLightningModule


def metrics(y, p, threshold, n_boot, seed):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    ppv = tp / (tp + fp) if tp + fp else float("nan")
    npv = tn / (tn + fn) if tn + fn else float("nan")

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            boot.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = (np.percentile(boot, [2.5, 97.5]) if boot else (float("nan"),) * 2)

    return {
        "auroc": roc_auc_score(y, p),
        "auroc_lo": lo, "auroc_hi": hi,
        "auprc": average_precision_score(y, p),
        "f1": f1_score(y, pred, zero_division=0),
        "accuracy": accuracy_score(y, pred),
        "balanced_acc": (sens + spec) / 2,
        "precision_ppv": ppv,
        "recall_sensitivity": sens,
        "specificity": spec,
        "npv": npv,
        "mcc": matthews_corrcoef(y, pred),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yml")
    ap.add_argument("--ckpt_dir", default="./checkpoints")
    ap.add_argument("--test_csv", default="./data/eval_csvs/pbtc/pbtc_efs1y.csv")
    ap.add_argument("--emb_dir", default="./data/embeddings/pbtc")
    ap.add_argument("--out_dir", default="./eval_out/pbtc")
    ap.add_argument("--date_format", default="yyyymmdd", choices=["yyyymmdd", "int"])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max_seq_len", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "*.ckpt")))
    if not ckpts:
        raise SystemExit(f"no checkpoints in {args.ckpt_dir}")
    print(f"{len(ckpts)} checkpoints, scoring {args.test_csv}\n", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, probs_by_arch, labels = [], {}, None
    per_patient = None

    for i, ckpt in enumerate(ckpts, 1):
        hp = torch.load(ckpt, map_location="cpu", weights_only=False)["hyper_parameters"]
        arch = hp["model"].get("temporal", "?")
        name = Path(ckpt).stem
        print(f"[{i}/{len(ckpts)}] {arch:<5} {name}", flush=True)

        model = EFSLightningModule.load_from_checkpoint(
            ckpt, config=dict(hp), map_location=device).eval().to(device)
        n_mod = hp["model"].get("n_modalities", 3)
        ds = EmbeddingTrajectoryDataset(args.test_csv, args.emb_dir,
                                        random_view=False, n_modalities=n_mod,
                                        date_format=args.date_format,
                                        modalities=hp.get("data", {}).get("modalities"))
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers,
                            collate_fn=partial(collate_trajectories,
                                               max_seq_len=args.max_seq_len,
                                               n_modalities=n_mod))
        p, y = run_inference(model, loader, device)
        p, y = p.numpy(), y.numpy().astype(int)
        labels = y
        if per_patient is None:
            per_patient = ds.df[["pat_id", "scandate", "label"]].copy()
        per_patient[name] = p

        probs_by_arch.setdefault(arch, []).append(p)
        m = metrics(y, p, args.threshold, args.n_boot, args.seed)
        rows.append({"model": name, "arch": arch, "kind": "single", **m})
        print(f"        AUROC {m['auroc']:.3f}  AUPRC {m['auprc']:.3f}  "
              f"F1 {m['f1']:.3f}", flush=True)

    for arch, ps in sorted(probs_by_arch.items()):
        if len(ps) < 2:
            continue
        mean_p = np.mean(ps, axis=0)
        per_patient[f"ensemble_{arch}"] = mean_p
        m = metrics(labels, mean_p, args.threshold, args.n_boot, args.seed)
        rows.append({"model": f"ensemble_{arch} (n={len(ps)})", "arch": arch,
                     "kind": "ensemble", **m})

    df = pd.DataFrame(rows).sort_values(["kind", "arch", "auroc"],
                                        ascending=[True, True, False])
    df.to_csv(out_dir / "checkpoint_metrics.csv", index=False)
    per_patient.to_csv(out_dir / "per_patient_probs.csv", index=False)

    cols = ["auroc", "auprc", "f1", "accuracy", "balanced_acc", "precision_ppv",
            "recall_sensitivity", "specificity", "npv", "mcc"]
    hdr = (f"{'model':<62} {'arch':<5} " +
           " ".join(f"{c[:9]:>9}" for c in cols) + f" {'AUROC 95% CI':>16}")
    lines = [
        f"cohort: {Path(args.test_csv).name}   n={len(labels)}  "
        f"positives={int(labels.sum())}  prevalence={labels.mean():.3f}  "
        f"threshold={args.threshold}", "", hdr, "-" * len(hdr),
    ]
    for r in df.itertuples():
        lines.append(f"{r.model:<62} {r.arch:<5} " +
                     " ".join(f"{getattr(r, c):>9.3f}" for c in cols) +
                     f"   [{r.auroc_lo:.3f}, {r.auroc_hi:.3f}]")
    table = "\n".join(lines)
    (out_dir / "checkpoint_metrics.txt").write_text(table + "\n")
    print("\n" + table)
    print(f"\nwrote {out_dir}/checkpoint_metrics.csv|.txt and per_patient_probs.csv")


if __name__ == "__main__":
    main()

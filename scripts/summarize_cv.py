"""Aggregate the per-fold JSONs train.py writes: per-fold metrics, mean +/- sd, and
pooled out-of-fold metrics. No checkpoints are loaded."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torchmetrics
import torch
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yml")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out_dir", default="./eval_out")
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    run_name = config["logger"]["run_name"]
    out = Path(args.out_dir)

    reports, oof = [], []
    for k in range(args.folds):
        path = out / f"{run_name}_fold{k}.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing -- did fold {k} finish training?")
        r = json.load(open(path))
        reports.append(r)
        oof.append(pd.DataFrame(r["predictions"]))
        print(f"fold{k}: n={r['n']:>3} pos={r['n_pos']:>2}  "
              f"AUROC={r['auroc']:.3f}  AUPRC={r['auprc']:.3f}  F1={r['f1']:.3f}")

    oof_df = pd.concat(oof, ignore_index=True)
    if oof_df["pat_id"].duplicated().any():
        raise ValueError("a patient appears in more than one fold's val set")
    p = torch.tensor(oof_df["prob"].values, dtype=torch.float32)
    y = torch.tensor(oof_df["label"].values, dtype=torch.long)
    pooled = {
        "n": len(y), "n_pos": int(y.sum()), "prevalence": y.float().mean().item(),
        "auroc": torchmetrics.functional.classification.binary_auroc(p, y).item(),
        "auprc": torchmetrics.functional.classification.binary_average_precision(p, y).item(),
        "f1": torchmetrics.functional.classification.binary_f1_score(p, y, threshold=0.5).item(),
    }
    aurocs = np.array([r["auroc"] for r in reports])
    auprcs = np.array([r["auprc"] for r in reports])
    f1s = np.array([r["f1"] for r in reports])

    oof_df.to_csv(out / f"{run_name}_oof_predictions.csv", index=False)
    summary = {
        "run_name": run_name,
        "folds": [{k: r[k] for k in ("fold", "n", "n_pos", "auroc", "auprc", "f1")}
                  for r in reports],
        "pooled_oof": pooled,
        "mean_auroc": aurocs.mean(), "sd_auroc": aurocs.std(ddof=1),
        "mean_auprc": auprcs.mean(), "sd_auprc": auprcs.std(ddof=1),
        "mean_f1": f1s.mean(), "sd_f1": f1s.std(ddof=1),
        # from the fold reports; the config may have changed since
        "model": reports[0]["model"],
    }
    with open(out / f"{run_name}_cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== {run_name} : {args.folds}-fold CV ===")
    print(f"  AUROC  {aurocs.mean():.3f} +/- {aurocs.std(ddof=1):.3f}   "
          f"(folds: {', '.join(f'{a:.3f}' for a in aurocs)})")
    print(f"  AUPRC  {auprcs.mean():.3f} +/- {auprcs.std(ddof=1):.3f}   "
          f"(baseline {pooled['prevalence']:.3f})")
    print(f"  F1     {f1s.mean():.3f} +/- {f1s.std(ddof=1):.3f}   @ threshold 0.5")
    print(f"  pooled out-of-fold (n={pooled['n']}, {pooled['n_pos']} pos): "
          f"AUROC {pooled['auroc']:.3f}  AUPRC {pooled['auprc']:.3f}  F1 {pooled['f1']:.3f}")
    print(f"  wrote {out}/{run_name}_cv_summary.json")


if __name__ == "__main__":
    main()

"""Score a CSV with a trained checkpoint: AUROC/AUPRC/F1 + per-patient probabilities.
Architecture comes from the checkpoint's hyper_parameters; config's model block is ignored."""
import argparse
import json
import os
from functools import partial
from pathlib import Path

import torch
import torchmetrics
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ is None or __package__ == "":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data.dataset import EmbeddingTrajectoryDataset, collate_trajectories
    from src.lightning_module import EFSLightningModule
else:
    from .data.dataset import EmbeddingTrajectoryDataset, collate_trajectories
    from .lightning_module import EFSLightningModule


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yml")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--test_csv", type=str, default=None)
    p.add_argument("--emb_dir", type=str, default=None)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    return p.parse_args()


@torch.no_grad()
def run_inference(model, loader, device):
    probs, labels = [], []
    for batch in tqdm(loader, desc="inference"):
        clinical = batch.get("clinical")
        logit, _, _ = model(
            batch["tokens"].to(device), batch["seg_idx"].to(device),
            batch["seg_slot"].to(device), batch["mask"].to(device), batch["days"].to(device),
            clinical.to(device) if clinical is not None else None,
        )
        probs.append(torch.sigmoid(logit).float().cpu())
        labels.append(batch["label"])
    return torch.cat(probs), torch.cat(labels)


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    inf = config.setdefault("infer", {})
    ckpt_path = args.checkpoint or inf["checkpoint"]
    test_csv = args.test_csv or inf["test_csv"]
    emb_dir = args.emb_dir or inf.get("emb_dir") or config["data"]["emb_dir"]
    threshold = args.threshold if args.threshold is not None else inf.get("threshold", 0.5)
    out_dir = Path(args.out_dir or inf.get("out_dir", "./eval_out"))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu"]["visible_device"])
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_config = torch.load(ckpt_path, map_location="cpu", weights_only=False)["hyper_parameters"]
    model = EFSLightningModule.load_from_checkpoint(
        ckpt_path, config=dict(ckpt_config), map_location=device
    )
    model.eval().to(device)

    # a `cohort` column resolves emb_dir / date_format per row
    dataset = EmbeddingTrajectoryDataset(
        csv_path=test_csv, emb_dir=emb_dir, random_view=False,
        n_modalities=ckpt_config["model"].get("n_modalities", 3),
        modalities=ckpt_config.get("data", {}).get("modalities"),
        date_format=config["data"].get("date_format", "yyyymmdd"),
        cohorts=config["data"].get("cohorts"),
    )
    loader = DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,      # predictions align with dataset.df
        num_workers=config["data"]["num_workers"],
        collate_fn=partial(
            collate_trajectories,
            max_seq_len=config["data"]["max_seq_len"],
            n_modalities=ckpt_config["model"].get("n_modalities", 3),
        modalities=ckpt_config.get("data", {}).get("modalities"),
        ),
    )

    probs, labels = run_inference(model, loader, device)
    labels_int = labels.long()
    auroc = torchmetrics.functional.classification.binary_auroc(probs, labels_int).item()
    auprc = torchmetrics.functional.classification.binary_average_precision(probs, labels_int).item()
    f1 = torchmetrics.functional.classification.binary_f1_score(probs, labels_int, threshold=threshold).item()
    preds = (probs >= threshold).long()
    acc = (preds == labels_int).float().mean().item()
    prevalence = labels_int.float().mean().item()

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df = dataset.df.copy()
    pred_df["prob"] = probs.numpy()
    pred_df["pred"] = preds.numpy()
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    summary = {
        "checkpoint": ckpt_path, "test_csv": test_csv,
        "n": int(labels_int.numel()), "n_positive": int(labels_int.sum().item()),
        "prevalence": prevalence, "threshold": threshold,
        "auroc": auroc, "auprc": auprc,
        "auprc_baseline": prevalence,   # random-classifier AP
        "f1": f1, "accuracy": acc,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== EFS inference: {Path(test_csv).name} ===")
    print(f"  n={summary['n']}  positives={summary['n_positive']}  prevalence={prevalence:.3f}")
    print(f"  AUROC = {auroc:.3f}")
    print(f"  AUPRC = {auprc:.3f}  (baseline {prevalence:.3f})")
    print(f"  F1    = {f1:.3f}  @ threshold {threshold}")
    print(f"  Acc   = {acc:.3f}")
    print(f"  wrote {out_dir/'predictions.csv'} and {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()

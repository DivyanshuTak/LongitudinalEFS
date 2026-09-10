"""Train the EFS head. After fitting, the best checkpoint is re-validated and its
metrics + per-patient predictions written to eval_out/{run_name}.json."""
import argparse
import json
import os
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

if __package__ is None or __package__ == "":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data_module import EFSDataModule
    from src.lightning_module import EFSLightningModule
else:
    from .data_module import EFSDataModule
    from .lightning_module import EFSLightningModule


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yml")
    p.add_argument("--fold", type=int, default=None,
                   help="Retarget data.train_csv/val_csv and the run name at splits/fold{k}_*.csv.")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.fold is not None:
        split_dir = config["data"].get("split_dir", "./splits")
        config["data"]["train_csv"] = f"{split_dir}/fold{args.fold}_TRAIN.csv"
        config["data"]["val_csv"] = f"{split_dir}/fold{args.fold}_VAL.csv"
        config["logger"]["run_name"] = f"{config['logger']['run_name']}_fold{args.fold}"
        # per-fold dir so save_last does not collide across folds
        config["logger"]["save_dir"] = f"{config['logger']['save_dir']}/{config['logger']['run_name']}"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu"]["visible_device"])
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(config["train"].get("seed", 42), workers=True)

    data_module = EFSDataModule(config)
    model = EFSLightningModule(config)

    wandb_logger = WandbLogger(
        project=config["logger"]["project_name"],
        name=config["logger"]["run_name"],
        config=config,
    )
    monitor = config["train"].get("checkpoint_monitor", "val_auroc")
    checkpoint_cb = ModelCheckpoint(
        dirpath=config["logger"]["save_dir"],
        filename=config["logger"]["save_name"],
        monitor=monitor,
        mode="min" if monitor == "val_loss" else "max",
        save_top_k=config["train"].get("save_top_k", 1),
        save_last=config["train"].get("save_last", False),
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks = [checkpoint_cb, lr_monitor]
    if config["train"].get("early_stopping_patience"):
        callbacks.append(EarlyStopping(
            monitor=config["train"].get("early_stopping_monitor", "val_auroc"),
            mode="max" if config["train"].get("early_stopping_monitor", "val_auroc") != "val_loss" else "min",
            patience=config["train"]["early_stopping_patience"],
        ))

    trainer = pl.Trainer(
        max_epochs=config["train"]["max_epochs"],
        logger=wandb_logger,
        callbacks=callbacks,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=config["train"].get("precision", "bf16-mixed"),
        gradient_clip_val=config["train"].get("grad_clip", 1.0),
    )
    trainer.fit(model, datamodule=data_module)
    write_fold_report(trainer, model, data_module, config, args.fold, checkpoint_cb)


def write_fold_report(trainer, model, data_module, config, fold, checkpoint_cb):
    """Re-validate the best checkpoint and dump its metrics + predictions."""
    metrics = trainer.validate(model, datamodule=data_module, ckpt_path="best")[0]
    probs = torch.cat(model.val_probs).numpy()
    labels = torch.cat(model.val_labels).numpy()

    preds = data_module.val_dataset.df.copy()
    if len(preds) != len(probs):
        raise RuntimeError(f"got {len(probs)} predictions for {len(preds)} val patients")
    preds["prob"] = probs
    preds["fold"] = fold

    out_dir = Path(config.get("infer", {}).get("out_dir", "./eval_out"))
    out_dir.mkdir(parents=True, exist_ok=True)
    name = config["logger"]["run_name"]
    report = {
        "run_name": name,
        "fold": fold,
        "val_csv": config["data"]["val_csv"],
        "n": len(preds),
        "n_pos": int(labels.sum()),
        "auroc": metrics["val_auroc"],
        "auprc": metrics["val_auprc"],
        "f1": metrics["val_f1"],
        "best_checkpoint": checkpoint_cb.best_model_path,
        "predictions": preds.to_dict("records"),
        "model": config["model"],
    }
    with open(out_dir / f"{name}.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nfold {fold}: AUROC {report['auroc']:.3f}  AUPRC {report['auprc']:.3f}  "
          f"-> {out_dir/f'{name}.json'}")


if __name__ == "__main__":
    main()

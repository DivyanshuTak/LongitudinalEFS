"""LightningModule for the EFS head. Embeddings are precomputed; everything here is trainable."""
from typing import Any, Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics

from .models.build import build_model


class EFSLightningModule(pl.LightningModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.model = build_model(config)
        # warm-start, optionally training only the named submodules
        init = config["train"].get("init_checkpoint")
        if init:
            state = torch.load(init, map_location="cpu")["state_dict"]
            self.model.load_state_dict({k[len("model."):]: v for k, v in state.items()
                                        if k.startswith("model.")})
        trainable = config["train"].get("trainable")
        if trainable:
            for name, p in self.model.named_parameters():
                p.requires_grad_(any(name.startswith(t + ".") for t in trainable))

        pos_weight = config["train"].get("pos_weight", None)
        pos_weight_t = torch.tensor(float(pos_weight)) if pos_weight is not None else None
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)

        self.train_auroc = torchmetrics.classification.BinaryAUROC()
        self.val_auroc = torchmetrics.classification.BinaryAUROC()
        self.val_auprc = torchmetrics.classification.BinaryAveragePrecision()
        self.val_f1 = torchmetrics.classification.BinaryF1Score()
        # last val pass, in loader order; train.py dumps these
        self.val_probs: list = []
        self.val_labels: list = []

    def forward(self, tokens, seg_idx, seg_slot, mask, days, clinical=None):
        return self.model(tokens, seg_idx, seg_slot, mask, days, clinical)

    def _shared_step(self, batch):
        logit, _, _ = self.model(
            batch["tokens"], batch["seg_idx"], batch["seg_slot"], batch["mask"], batch["days"],
            batch.get("clinical"),
        )
        loss = self.criterion(logit, batch["label"])
        return loss, torch.sigmoid(logit).detach().float()

    def training_step(self, batch, batch_idx):
        loss, probs = self._shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=batch["label"].numel())
        self.train_auroc.update(probs, batch["label"].long())
        return loss

    def on_train_epoch_end(self):
        self.log("train_auroc", self.train_auroc.compute(), prog_bar=True)
        self.train_auroc.reset()

    def on_validation_epoch_start(self):
        self.val_probs, self.val_labels = [], []

    def validation_step(self, batch, batch_idx):
        loss, probs = self._shared_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=batch["label"].numel())
        labels = batch["label"].long()
        self.val_probs.append(probs.cpu())
        self.val_labels.append(labels.cpu())
        self.val_auroc.update(probs, labels)
        self.val_auprc.update(probs, labels)
        self.val_f1.update(probs, labels)
        return loss

    def on_validation_epoch_end(self):
        self.log("val_auroc", self.val_auroc.compute(), prog_bar=True)
        self.log("val_auprc", self.val_auprc.compute(), prog_bar=True)
        self.log("val_f1", self.val_f1.compute(), prog_bar=True)
        self.val_auroc.reset()
        self.val_auprc.reset()
        self.val_f1.reset()

    def configure_optimizers(self):
        o = self.config["optim"]
        return torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=o["lr"], weight_decay=o.get("weight_decay", 0.0)
        )

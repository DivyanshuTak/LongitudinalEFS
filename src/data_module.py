"""DataModule wiring train/val EmbeddingTrajectoryDataset + collate_trajectories."""
from functools import partial
from typing import Any, Dict

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .data.dataset import EmbeddingTrajectoryDataset, collate_trajectories


class EFSDataModule(pl.LightningDataModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        d = self.config["data"]
        n_mod = self.config["model"].get("n_modalities", 3)
        common = dict(emb_dir=d.get("emb_dir"), n_views=d.get("n_views", 5),
                      n_modalities=n_mod, date_format=d.get("date_format", "yyyymmdd"),
                      cohorts=d.get("cohorts"), modalities=d.get("modalities"),
                      load_tokens=self.config["model"].get("mode", "imaging") != "clinical")
        self.train_dataset = EmbeddingTrajectoryDataset(
            csv_path=d["train_csv"], random_view=d.get("augment_views", True), **common
        )
        self.val_dataset = EmbeddingTrajectoryDataset(
            csv_path=d["val_csv"], random_view=False, **common
        )

    def _collate(self):
        return partial(
            collate_trajectories,
            max_seq_len=self.config["data"]["max_seq_len"],
            n_modalities=self.config["model"].get("n_modalities", 3),
        )

    def train_dataloader(self):
        d = self.config["data"]
        return DataLoader(
            self.train_dataset,
            batch_size=d["batch_size"],
            shuffle=True,
            num_workers=d["num_workers"],
            collate_fn=self._collate(),
            pin_memory=True,
            persistent_workers=d["num_workers"] > 0,
        )

    def val_dataloader(self):
        d = self.config["data"]
        workers = d["num_workers"] // 2
        return DataLoader(
            self.val_dataset,
            batch_size=d["batch_size"],
            shuffle=False,
            num_workers=workers,
            collate_fn=self._collate(),
            pin_memory=True,
            persistent_workers=workers > 0,
        )

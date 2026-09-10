"""NeuroVFM's released diagnostic head (mlinslab/neurovfm-dx-mri) as a frozen
patch pooler: a scan vector is the attention-weighted mean of a series' tokens
under the head's attention for one label. Interface matches ABMILPool."""
from typing import Optional

import torch
import torch.nn as nn


class VFMPool(nn.Module):
    def __init__(self, repo: str = "mlinslab/neurovfm-dx-mri",
                 label: str = "low_grade_glioma", freeze: bool = True):
        super().__init__()
        import json
        from importlib.resources import files
        from huggingface_hub import hf_hub_download
        from neurovfm.models.mil import ClassifyThenAggregate

        cfg = json.load(open(hf_hub_download(repo, "config.json")))
        if cfg["which"] != "classify_then_aggregate":
            raise ValueError(f"unexpected pooler type {cfg['which']!r}")
        self.head = ClassifyThenAggregate(dim=768, **cfg["params"])
        missing, _ = self.head.load_state_dict(
            torch.load(hf_hub_download(repo, "pytorch_model.bin"), map_location="cpu"),
            strict=False)
        if missing:
            raise RuntimeError(f"released head is missing {len(missing)} keys: {missing[:5]}")

        with (files("neurovfm.pipelines.resources") / "mri_label_names.txt").open() as f:
            names = [ln.strip() for ln in f if ln.strip()]
        self.label_idx: Optional[int] = None if label == "mean" else names.index(label)
        self.frozen = freeze
        if freeze:
            self.head.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.head.eval()
        return self

    def forward(self, tokens: torch.Tensor, seg_idx: torch.Tensor, n_seg: int):
        """tokens: (N, D), seg_idx: (N,) contiguous in [0, n_seg) -> (n_seg, D), attn (N,)."""
        counts = torch.bincount(seg_idx, minlength=n_seg)
        if not torch.equal(seg_idx, torch.repeat_interleave(
                torch.arange(n_seg, device=seg_idx.device), counts)):
            raise ValueError("VFMPool needs contiguous, ordered segments")
        cu = torch.cat([torch.zeros(1, dtype=torch.long, device=tokens.device),
                        counts.cumsum(0)])

        # fp32 head; torch_scatter segment ops are not autocast-safe
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            ctx = torch.no_grad() if self.frozen else torch.enable_grad()
            with ctx:
                _, attn, _ = self.head(tokens.float(), cu_seqlens=cu, return_logits=True)
            attn = attn.float().mean(-1) if self.label_idx is None else attn[:, self.label_idx].float()
            pooled = torch.zeros(n_seg, tokens.shape[-1], device=tokens.device)
            pooled.index_add_(0, seg_idx, attn.unsqueeze(-1) * tokens.float())
        return pooled.to(tokens.dtype), attn

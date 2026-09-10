"""Attention-MIL pooling (Ilse et al. 2018) over packed token sets: flat (N, D)
tokens plus a segment index, segment softmax in fp32."""
import torch
import torch.nn as nn


class ABMILPool(nn.Module):
    def __init__(self, dim: int = 768, hidden: int = 512, gated: bool = False,
                 dropout: float = 0.0):
        super().__init__()
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden) if gated else None
        self.w = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, seg_idx: torch.Tensor, n_seg: int):
        """tokens: (N, D), seg_idx: (N,) in [0, n_seg) -> (n_seg, D), attn (N,)."""
        h = torch.tanh(self.V(tokens))
        if self.U is not None:
            h = h * torch.sigmoid(self.U(tokens))
        scores = self.w(self.dropout(h)).squeeze(-1).float()

        peak = torch.full((n_seg,), float("-inf"), device=scores.device)
        peak.scatter_reduce_(0, seg_idx, scores, reduce="amax", include_self=False)
        weights = torch.exp(scores - peak[seg_idx])
        denom = torch.zeros(n_seg, device=scores.device).index_add_(0, seg_idx, weights)
        attn = weights / denom[seg_idx].clamp(min=1e-12)

        pooled = torch.zeros(n_seg, tokens.shape[-1], device=tokens.device)
        pooled.index_add_(0, seg_idx, attn.unsqueeze(-1) * tokens.float())
        return pooled.to(tokens.dtype), attn

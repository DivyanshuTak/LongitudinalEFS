"""Model construction from a config dict."""
from typing import Any, Dict

from ..data.clinical import CLINICAL_DIM
from .trajectory_model import LongitudinalEFSModel


MODES = ("imaging", "imaging_clinical", "clinical")


def build_model(config: Dict[str, Any]) -> LongitudinalEFSModel:
    m = config["model"]
    pool = m.get("pool", {})
    mode = m.get("mode", "imaging")
    if mode not in MODES:
        raise ValueError(f"model.mode must be one of {MODES}, got {mode!r}")
    return LongitudinalEFSModel(
        feature_dim=m.get("feature_dim", 768),
        n_modalities=m.get("n_modalities", 3),
        pool_hidden=pool.get("hidden", 512),
        pool_gated=pool.get("gated", False),
        pool_dropout=pool.get("dropout", 0.0),
        pool_source=pool.get("source", "abmil"),
        vfm_repo=pool.get("vfm_repo", "mlinslab/neurovfm-dx-mri"),
        vfm_label=pool.get("vfm_label", "low_grade_glioma"),
        vfm_freeze=pool.get("vfm_freeze", True),
        fusion=m.get("fusion", "per_modality"),
        fusion_dim=m.get("fusion_dim", 768),
        time_dim=m.get("time_dim", 16),
        min_period_days=m.get("min_period_days", 30.0),
        max_period_days=m.get("max_period_days", 2000.0),
        temporal=m.get("temporal", "gru"),
        hidden=m.get("hidden", 128),
        num_layers=m.get("num_layers", 1),
        nhead=m.get("nhead", 8),
        readout=m.get("readout", "last"),
        dropout=m.get("dropout", 0.1),
        clinical_dim=CLINICAL_DIM if mode in ("imaging_clinical", "clinical") else 0,
        use_imaging=mode != "clinical",
        clinical_hidden=m.get("clinical_hidden", 32),
        clinical_dropout=m.get("clinical_dropout", 0.3),
    )

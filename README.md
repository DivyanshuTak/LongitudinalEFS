# LongitudinalEFS

Progression prediction from longitudinal pediatric low-grade glioma MRI on frozen
NeuroVFM patch tokens. Trained on BCH + CBTN, externally tested on PBTC.

## Model

```
patch tokens per scan (t1ce, t2, flair)
  -> NeuroVFM diagnostic head as frozen attention pooler   (B, T, 3, 768)
  -> per-modality Linear(768, 256), concat                 (B, T, 768)
  -> GRU(128) -> last timestep -> Linear(128, 1)           logit
```

Two label definitions, same architecture:

| config | label |
|---|---|
| `*_1y.yml` | progression within 365 d of the last scan |
| `*_current_status.yml` | last scan is the progression scan (±15 d); non-progressors unchanged |

## Layout

| path | role |
|---|---|
| `src/` | dataset, model, Lightning module, `train.py`, `infer.py` |
| `configs/` | six models: {bch, cbtn, combined} × {1y, current_status} |
| `scripts/build_*.py`, `make_splits.py` | label CSVs, landmark CSVs, splits |
| `scripts/extract_neurovfm_embeddings.py` | offline token extraction |
| `scripts/eval_landmarks.py`, `pbtc_model_sweep.sh` | PBTC evaluation (single-row + landmark, bootstrap CIs) |
| `scripts/plot_*.py` | landmark AUROC and calibration figures |

## Environment

```bash
source ../neurovfm/.venv/bin/activate
```

## Pipeline

Run from this directory. Paths inside configs and scripts are relative to it.

```bash
python scripts/extract_neurovfm_embeddings.py ...               # tokens -> {pat}_{date}_v{k}.npz
python scripts/build_pbtc_efs_csv.py                             # 1-year labels
python scripts/build_cbtn_efs_csv.py
python scripts/build_current_status_csv.py cbtn|pbtc|bch        # current-status labels
python scripts/build_landmark_csv.py cbtn|pbtc                   # year-t landmark rows, 1-year labels
python scripts/build_current_status_landmark_csv.py cbtn|pbtc
python scripts/build_combined_splits.py                          # BCH DEV/TEST + CBTN 80/20

python src/train.py --config configs/combined_1y.yml
scripts/pbtc_model_sweep.sh                                      # every model x every PBTC CSV
python scripts/plot_landmark_single.py <metrics.txt> <out.png> <color>
python scripts/plot_calibration_from_preds.py <predictions.csv> <out.png> <title>
python -m pytest tests/ -q
```

Data CSVs, splits, embeddings, and checkpoints are not in this directory yet.

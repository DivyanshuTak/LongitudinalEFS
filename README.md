# LongitudinalEFS

Progression prediction from longitudinal pediatric low-grade glioma MRI.

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
| `*_current_status.yml` | last scan is the progression scan; non-progressors unchanged |

## Data

Data, checkpoints and results are not in this repo. Download the bundle
(**Dropbox link: TODO**), unzip it, and place it at `data/` in the repo root so
the tree looks like this. Every script and config resolves paths relative to
the repo root, so run everything from here.

```
data/
  raw/                  label sources
    bch/                efs1y_postsurg_max6.csv, longitudinal_from_pool.csv,
                        lgg_bch_metadata_deidentified.csv, main_data_pool_selected.csv
    cbtn/               cbtn_longitudinal.csv, cbtn_all_2023-09-15_LGG.csv
    pbtc/               pbtc_longitudinal.csv, efs_data_set1.csv
  splits/               training splits used by configs/
  eval_csvs/{cbtn,pbtc} 1-year, current-status, and landmark test CSVs
  checkpoints/          {bch,cbtn,combined}_{1y,current_status}.ckpt
  embeddings/{bch,cbtn,pbtc}   NeuroVFM token .npz files (not in the bundle, see below)
  results/
    train_runs/         per-model validation metrics + predictions
    pbtc_sweep/         every model x every PBTC CSV: metrics.txt, predictions.csv, summary.tsv
    figures/            landmark AUROC and calibration plots for the combined models
```

Embeddings are extracted from the NIfTIs with `scripts/extract_neurovfm_embeddings.py`
into `data/embeddings/<cohort>/{pat_id}_{scandate}_v{view}.npz`. They are not
part of the bundle.

## Environment

Reuses the NeuroVFM environment (torch, pytorch-lightning, torchmetrics,
huggingface_hub, `neurovfm`). `wandb login` once before training.

## Pipeline

All CSVs in `data/splits` and `data/eval_csvs` are shipped; the commands below
regenerate them from `data/raw`.

```bash
# labels
python scripts/build_pbtc_efs_csv.py
python scripts/build_cbtn_efs_csv.py
python scripts/build_current_status_csv.py pbtc          # 15 d tolerance
python scripts/build_current_status_csv.py cbtn 60
python scripts/build_current_status_csv.py bch 60 DEV
python scripts/build_current_status_csv.py bch 60 TEST
python scripts/build_landmark_csv.py pbtc                 # and cbtn
python scripts/build_current_status_landmark_csv.py pbtc  # and cbtn

# splits (BCH DEV/TEST from make_splits.py; CBTN 80/20 stratified)
python scripts/build_combined_splits.py
python scripts/build_combined_splits.py \
    --bch-train data/splits/DEV_current_status.csv --bch-val data/splits/TEST_current_status.csv \
    --cbtn data/eval_csvs/cbtn/cbtn_current_status.csv \
    --out-train data/splits/combined_bch_cbtn_current_status_train.csv \
    --out-val data/splits/combined_bch_cbtn_current_status_val.csv
# cbtn_only_csplit_*.csv are the CBTN rows of the combined splits

# train
python src/train.py --config configs/combined_1y.yml

# PBTC evaluation: every model x every test CSV -> eval_out/pbtc_sweep/
scripts/pbtc_model_sweep.sh 0
python scripts/plot_landmark_single.py eval_out/pbtc_sweep/combined_1y__1y_landmark/metrics.txt out.png "#d62728"
python scripts/plot_calibration_from_preds.py eval_out/pbtc_sweep/combined_1y__1y_single/predictions.csv out.png "combined 1y"

python -m pytest tests/ -q
```

Notes on the shipped files:
- Current-status CSVs were built with a 60 d scan-to-progression tolerance for
  BCH and CBTN and 15 d for PBTC.
- The combined current-status split's CBTN train/val draw is not bit-reproducible
  from `build_combined_splits.py` (same patients and labels, different random
  assignment); use the shipped split.

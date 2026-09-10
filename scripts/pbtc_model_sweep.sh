#!/bin/bash
# Every model x every PBTC test CSV (1-year / current-status, single-row / landmark).
# Pooled AUROC + CI -> eval_out/pbtc_sweep/summary.tsv.  Usage: scripts/pbtc_model_sweep.sh [gpu]
cd "$(dirname "$0")/.."
GPU=${1:-0}
E=data/embeddings/pbtc
declare -A CSV=( [1y_single]="data/eval_csvs/pbtc/pbtc_efs1y.csv|label|99"
                 [1y_landmark]="data/eval_csvs/pbtc/pbtc_efs1y_landmarks.csv|landmark_year|4"
                 [cs_single]="data/eval_csvs/pbtc/pbtc_current_status.csv|label|99"
                 [cs_landmark]="data/eval_csvs/pbtc/pbtc_current_status_landmarks.csv|landmark_year|4" )
mkdir -p eval_out/pbtc_sweep; out=eval_out/pbtc_sweep/summary.tsv
echo -e "model\ttest\tn\tpos\tauroc\tlo\thi\tauprc" > $out
for m in bch_1y bch_current_status cbtn_1y cbtn_current_status combined_1y combined_current_status; do
  for t in 1y_single 1y_landmark cs_single cs_landmark; do
    IFS='|' read csv col pfy <<< "${CSV[$t]}"
    line=$(python scripts/eval_landmarks.py --ckpt data/checkpoints/$m.ckpt --csv $csv --emb_dir $E --date_format yyyymmdd \
             --out eval_out/pbtc_sweep/${m}__${t} --gpu $GPU --group_col $col --pool_from_year $pfy 2>&1 | grep -E "^ *pooled")
    read _ n pos auroc lo hi auprc _ <<< "$(echo "$line" | tr -d '[],')"
    echo -e "$m\t$t\t$n\t$pos\t$auroc\t$lo\t$hi\t$auprc" | tee -a $out
  done
done

#!/bin/bash
# Every model x every PBTC test CSV (1-year / current-status, single-row / landmark). Pooled AUROC + CI -> eval_out/pbtc_sweep/summary.tsv
cd "$(dirname "$0")/.."; source ../neurovfm/.venv/bin/activate
E=/media/sdb/divyanshu/divyanshu/genai_prognosis/pbtc_neurovfm_embeddings
declare -A CK=(
 [bch_1y]="checkpoints_vfmpool_frozen/efs_vfmpool_frozen_dev_test_gru128-epoch=17-val_auroc=0.809.ckpt"
 [bch_cs]="checkpoints/efs_currentstatus_vfmpool_frozen_dev_test_gru128/efs_currentstatus_vfmpool_frozen_dev_test_gru128-epoch=11-val_auroc=0.793.ckpt"
 [cbtn_1y]="checkpoints/efs_cbtnonly_vfmpool_frozen_gru128/efs_cbtnonly_vfmpool_frozen_gru128-epoch=07-val_auroc=0.810.ckpt"
 [cbtn_cs]="checkpoints/efs_currentstatus_cbtnonly_vfmpool_frozen_gru128/efs_currentstatus_cbtnonly_vfmpool_frozen_gru128-epoch=08-val_auroc=0.847.ckpt"
 [combined_1y]="checkpoints/efs_combined_bch_cbtn_vfmpool_frozen_gru128/efs_combined_bch_cbtn_vfmpool_frozen_gru128-epoch=07-val_auroc=0.820.ckpt"
 [combined_cs]="checkpoints/efs_currentstatus_combined_bch_cbtn_vfmpool_frozen_gru128/efs_currentstatus_combined_bch_cbtn_vfmpool_frozen_gru128-epoch=09-val_auroc=0.850.ckpt")
declare -A CSV=( [1y_single]="eval_csvs/pbtc/pbtc_efs1y.csv|label|99" [1y_landmark]="eval_csvs/pbtc/pbtc_efs1y_landmarks.csv|landmark_year|4"
                 [cs_single]="eval_csvs/pbtc/pbtc_current_status.csv|label|99" [cs_landmark]="eval_csvs/pbtc/pbtc_current_status_landmarks.csv|landmark_year|4")
mkdir -p eval_out/pbtc_sweep; out=eval_out/pbtc_sweep/summary.tsv; echo -e "model\ttest\tn\tpos\tauroc\tlo\thi\tauprc" > $out
i=0
for m in bch_1y bch_cs cbtn_1y cbtn_cs combined_1y combined_cs; do for t in 1y_single 1y_landmark cs_single cs_landmark; do
  IFS='|' read csv col pfy <<< "${CSV[$t]}"; gpu=$(( (i++) % 3 )); gpu=$(( gpu == 1 ? 3 : gpu ))
  line=$(python scripts/eval_landmarks.py --ckpt "${CK[$m]}" --csv $csv --emb_dir $E --date_format yyyymmdd --out eval_out/pbtc_sweep/${m}__${t} --gpu $gpu --group_col $col --pool_from_year $pfy 2>&1 | grep -E "^ *pooled")
  read _ n pos auroc lo hi auprc _ <<< "$(echo "$line" | tr -d '[],')"; echo -e "$m\t$t\t$n\t$pos\t$auroc\t$lo\t$hi\t$auprc" | tee -a $out
done; done

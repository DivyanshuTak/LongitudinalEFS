"""Copy raw T1c/T2/FLAIR volumes that still need preprocessing into one flat pool,
named <cohort>_<pat_id>_<scandate>_<t1c|t2|flair>.nii.gz. Resumable; writes a
copy_manifest.csv next to the pool.

  python scripts/copy_raw_for_preprocessing.py [dst_dir]
"""
import os, re, shutil, glob, sys
import pandas as pd

DST = sys.argv[1] if len(sys.argv) > 1 else "/media/sdb/divyanshu/divyanshu/aidan_segmentation/longitudianl_efs_data_pool/raw"
G, B = "/media/sdb/divyanshu/divyanshu/genai_prognosis", "../bch_longitudinal_data_latest"
MOD_OUT = {"t1ce": "t1c", "t2": "t2", "flair": "flair"}
need = pd.read_csv("eval_out/paper_needed_timepoints.csv", dtype=str)

# --- raw source path per (cohort, pat, date, modality)
src = {}
man = pd.read_csv(f"{B}/main_data_pool_selected.csv", dtype=str)
for r in man.itertuples():
    src[("bch", r.pat_id1, r.scandate, r.modality)] = f"{B}/main_data_pool/{r.dest_name}"
for c, d in [("cbtn", f"{G}/cbtn_data"), ("pbtc", f"{G}/pbtc_multimodal_dataset_images")]:
    for f in os.listdir(d):
        m = re.match(r"(.+?)_(\d+)_(t1c|t2|flair)\.nii\.gz$", f)
        if m: src[(c, m.group(1), m.group(2), {"t1c": "t1ce"}.get(m.group(3), m.group(3)))] = f"{d}/{f}"

# --- already preprocessed (strict: same series as the NeuroVFM pool for BCH)
have = set()
stem2key = {r.dest_name.replace(".nii.gz", ""): ("bch", r.pat_id1, r.scandate, r.modality) for r in man.itertuples()}
for f in os.listdir(f"{B}/main_data_pool_preprocessed/T2W_reg"):
    k = stem2key.get(f.replace("_0000.nii.gz", "").replace(".nii.gz", ""))
    if k: have.add(k)
for f in os.listdir(f"{G}/data/cbtn/imagesTs"):
    m = re.match(r"(\d+)_(\d+)\.nii\.gz$", f)
    if m: have.add(("cbtn", "C" + m.group(1), m.group(2), "t2"))
for f in glob.glob(f"{G}/data/PBTC/all_t2_preprocessed/T2W_reg/*.nii.gz") + glob.glob(f"{G}/data/PBTC/set*_nifti_flair_preprocessed/T2W_reg/*.nii.gz"):
    m = re.search(r"Pbtc-029b-(\d+)-(\d{8})_(t2|flair)", os.path.basename(f))
    if m: have.add(("pbtc", m.group(1), m.group(2), m.group(3)))

os.makedirs(DST, exist_ok=True)
rows, missing_src = [], []
for r in need.itertuples():
    for mod in ["t1ce", "t2", "flair"]:
        k = (r.cohort, r.pat_id, r.scandate, mod)
        if k in have: continue
        if k not in src: missing_src.append(k); continue
        out = f"{DST}/{r.cohort}_{r.pat_id}_{r.scandate}_{MOD_OUT[mod]}.nii.gz"
        if not os.path.exists(out): shutil.copy2(src[k], out)
        rows.append(dict(cohort=r.cohort, pat_id=r.pat_id, scandate=r.scandate, modality=MOD_OUT[mod], src=src[k], dst=out))
df = pd.DataFrame(rows); df.to_csv(os.path.join(os.path.dirname(DST), "copy_manifest.csv"), index=False)
print(df.groupby(["cohort", "modality"]).size().unstack().to_string()); print("total copied:", len(df), "| no raw source found:", len(missing_src), missing_src[:5])

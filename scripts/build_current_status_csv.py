"""Current-status CSVs ("has the patient progressed by now?"), one row per patient.

Progressors: trajectory ends at the scan closest to the progression date
(|scan - event| <= tolerance, else dropped), label 1. Non-progressors: row copied
from the 1-year EFS CSV. Last 6 scans kept; CBTN inputs stay post-surgical.

  python scripts/build_current_status_csv.py cbtn|pbtc|bch [tolerance_days] [DEV|TEST]
"""
import sys
from datetime import datetime, timedelta
import pandas as pd
sys.path.insert(0, "scripts")

COHORT, MAX_LEN, MIN_LEN = sys.argv[1], 6, 2
BCH_SPLIT = sys.argv[3] if len(sys.argv) > 3 else "TEST"   # bch only: DEV or TEST
TOL = int(sys.argv[2]) if len(sys.argv) > 2 else 15
if COHORT == "bch":   # BCH TEST split; label source = clinical metadata progression date
    lon = pd.read_csv("data/raw/bch/longitudinal_from_pool.csv", header=None, names=["pat_id", "scandate"], dtype=str).set_index("pat_id")["scandate"]
    efs = pd.read_csv(f"data/splits/{BCH_SPLIT}.csv", dtype={"pat_id": str, "scandate": str})
    meta = pd.read_csv("data/raw/bch/lgg_bch_metadata_deidentified.csv", dtype={"BCH MRN": str}).groupby("BCH MRN", as_index=False).first().set_index("BCH MRN")
    efs["pfs"] = (meta.loc[efs.pat_id, "Progression"].values == "Yes").astype(int)
    efs["n_scans"] = efs.scandate.str.count("-") + 1
    efs["days_last_scan_to_event"] = None
else:
    lon = pd.read_csv(f"data/raw/{COHORT}/{COHORT}_longitudinal.csv", dtype=str).set_index("pat_id")["scandate"]
    efs = pd.read_csv(f"data/eval_csvs/{COHORT}/{COHORT}_efs1y.csv", dtype={"pat_id": str, "scandate": str})

if COHORT == "cbtn":
    from build_cbtn_efs_csv import build_anchors
    anchors = build_anchors("data/raw/cbtn/cbtn_all_2023-09-15_LGG.csv")
    def timeline(pid):
        a = anchors[pid]
        scans = sorted(int(d) for d in lon[pid].split("-"))
        return [s for s in scans if s >= a["diagnosis_age"] + 1], a["diagnosis_age"] + a["efs"], bool(a["had_event"])
    days, fmt = lambda s, e: s - e, str
elif COHORT == "bch":
    def timeline(pid):
        r = meta.loc[pid]
        scans = sorted(datetime.strptime(d, "%Y%m%d") for d in lon[pid].split("-"))
        if r["Surgical Resection"] == "Yes":
            scans = [s for s in scans if s > pd.to_datetime(r["Date of first surgery"])]
        return scans, pd.to_datetime(r["Date of First Progression"]).to_pydatetime(), True
    days, fmt = lambda s, e: (s - e).days, lambda d: d.strftime("%Y%m%d")
else:
    lab = pd.read_csv("data/raw/pbtc/efs_data_set1.csv", dtype={"accessionnumber": str}).drop_duplicates("accessionnumber").set_index("accessionnumber")
    def timeline(pid):
        scans = sorted(datetime.strptime(d, "%Y%m%d") for d in lon[pid].split("-"))
        return scans, scans[0] + timedelta(days=float(lab.at[pid, "pfs_time"])), int(lab.at[pid, "pfs"]) == 1
    days, fmt = lambda s, e: (s - e).days, lambda d: d.strftime("%Y%m%d")

rows, dropped = [], {"no_scan_within_tol": 0, "too_few_scans": 0}
n_prev_neg_now_pos = 0
for r in efs.itertuples():
    if not r.pfs:
        rows.append(dict(pat_id=r.pat_id, scandate=r.scandate, label=0, n_scans=r.n_scans,
                         days_last_scan_to_event=r.days_last_scan_to_event, pfs=0, status="censored_unchanged"))
        continue
    scans, event, _ = timeline(r.pat_id)
    near = [s for s in scans if abs(days(s, event)) <= TOL]
    if not near:
        dropped["no_scan_within_tol"] += 1
        continue
    prog = min(near, key=lambda s: abs(days(s, event)))
    kept = [s for s in scans if s <= prog][-MAX_LEN:]
    if len(kept) < MIN_LEN:
        dropped["too_few_scans"] += 1
        continue
    n_prev_neg_now_pos += int(r.label == 0)
    rows.append(dict(pat_id=r.pat_id, scandate="-".join(fmt(s) for s in kept), label=1, n_scans=len(kept),
                     days_last_scan_to_event=-days(prog, event), pfs=1, status="progression_scan"))

out = pd.DataFrame(rows)
OUT = f"data/splits/{BCH_SPLIT}_current_status.csv" if COHORT == "bch" else f"data/eval_csvs/{COHORT}/{COHORT}_current_status.csv"
out.to_csv(OUT, index=False)
print(f"{OUT}: {len(out)} patients, {int(out.label.sum())} positive ({out.label.mean():.1%}); "
      f"progressors dropped: {dropped}; progressors that were label 0 in the EFS csv and are now 1: {n_prev_neg_now_pos}")
pos = out[out.label == 1]
print(f"  positives: scans/patient median {pos.n_scans.median():.0f}, offset progression-scan minus event (days) "
      f"median {(-pos.days_last_scan_to_event).median():.0f} [{(-pos.days_last_scan_to_event).min()}, {(-pos.days_last_scan_to_event).max()}]")

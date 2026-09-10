"""Landmark (year-t) rows for the current-status question, CBTN / PBTC.

Per patient and t = 1, 2, ...:
  input = scans up to first_scan + 365*t, never past the progression scan (the
          scan within +-tol d of the progression date), thinned to 6 evenly spaced
  label = 1 if the last input scan is the progression scan, else 0
Progressors with no scan within tol contribute only label-0 rows. A landmark that
adds no new scan is skipped; rows need >= 2 scans.

  python scripts/build_current_status_landmark_csv.py cbtn|pbtc [tol_days]
"""
import sys
from datetime import datetime, timedelta
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")

COHORT, TOL, MAX_LEN = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 60, 6
lon = pd.read_csv(f"eval_csvs/{COHORT}/{COHORT}_longitudinal.csv", dtype=str).set_index("pat_id")["scandate"]
efs = pd.read_csv(f"eval_csvs/{COHORT}/{COHORT}_efs1y.csv", dtype={"pat_id": str})
if COHORT == "cbtn":
    from build_cbtn_efs_csv import build_anchors
    A = build_anchors("eval_csvs/cbtn/cbtn_all_2023-09-15_LGG.csv")
    def timeline(p):
        a = A[p]; s = sorted(int(d) for d in lon[p].split("-"))
        return [x for x in s if x >= a["diagnosis_age"] + 1], a["diagnosis_age"] + a["efs"], bool(a["had_event"])
    days, fmt, plus = lambda s, e: s - e, str, lambda s, n: s + n
else:
    lab = pd.read_csv("../DirectEFS_NeuroJEPA/endpoint_csvs/pbtc_set4/efs_data_set1.csv", dtype={"accessionnumber": str}).drop_duplicates("accessionnumber").set_index("accessionnumber")
    def timeline(p):
        s = sorted(datetime.strptime(d, "%Y%m%d") for d in lon[p].split("-"))
        return s, s[0] + timedelta(days=float(lab.at[p, "pfs_time"])), int(lab.at[p, "pfs"]) == 1
    days, fmt, plus = lambda s, e: (s - e).days, lambda d: d.strftime("%Y%m%d"), lambda s, n: s + timedelta(days=n)

rows = []
for p in efs.pat_id:
    scans, event, prog = timeline(p)
    prog_scan = None
    if prog:
        near = [s for s in scans if abs(days(s, event)) <= TOL]
        prog_scan = min(near, key=lambda s: abs(days(s, event))) if near else None
        scans = [s for s in scans if s <= prog_scan] if prog_scan else [s for s in scans if days(s, event) < -TOL]
    if len(scans) < 2:
        continue
    prev = None
    for t in range(1, 40):
        sel = [s for s in scans if days(s, plus(scans[0], 365 * t)) <= 0]
        if len(sel) < 2:
            continue
        if sel == prev:
            if days(scans[-1], plus(scans[0], 365 * t)) <= 0:
                break
            continue
        prev = sel
        keep = [sel[i] for i in np.round(np.linspace(0, len(sel) - 1, min(len(sel), MAX_LEN))).astype(int)]
        rows.append(dict(pat_id=p, scandate="-".join(map(fmt, keep)), label=int(prog_scan is not None and sel[-1] == prog_scan),
                         landmark_year=t, n_scans=len(keep), n_scans_available=len(sel), anchor_scan=fmt(sel[-1]), pfs=int(prog)))
df = pd.DataFrame(rows)
OUT = f"eval_csvs/{COHORT}/{COHORT}_current_status_landmarks.csv"
df.to_csv(OUT, index=False)
print(f"{OUT}: rows {len(df)}, patients {df.pat_id.nunique()}, positives {int(df.label.sum())} ({df.label.mean():.1%})")
print(df.groupby("landmark_year").agg(rows=("pat_id", "size"), positives=("label", "sum")).T.to_string())

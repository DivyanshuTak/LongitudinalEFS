"""Landmark (year-t) rows for CBTN / PBTC, each with its own 1-year EFS label.

Per patient and t = 1, 2, ...:
  input  = scans up to first_scan + 365*t, thinned to 6 evenly spaced (first and
           last kept); scans on/after the progression date are never inputs
  anchor = last input scan
  label  = 1 if progression within 365 d of the anchor, 0 if a full year is
           observed without progression, else the row is dropped
A landmark that adds no new scan over the previous one is skipped.

  python scripts/build_landmark_csv.py cbtn|pbtc
"""
import numpy as np
import sys
import pandas as pd

COHORT = sys.argv[1] if len(sys.argv) > 1 else "cbtn"
LON, EFS = f"data/raw/{COHORT}/{COHORT}_longitudinal.csv", f"data/eval_csvs/{COHORT}/{COHORT}_efs1y.csv"
OUT = f"data/eval_csvs/{COHORT}/{COHORT}_efs1y_landmarks.csv"
# CBTN dates are age-in-days; PBTC dates are YYYYMMDD -> work in days, print back in the input format
if COHORT == "cbtn":
    to_day, to_str = int, str
else:
    _epoch = pd.Timestamp("1900-01-01")
    to_day = lambda s: (pd.to_datetime(str(s), format="%Y%m%d") - _epoch).days
    to_str = lambda d: (_epoch + pd.Timedelta(days=int(d))).strftime("%Y%m%d")

MAX_LEN = 6

lon = pd.read_csv(LON, dtype=str)
lon = lon.set_index("pat_id")["scandate"].str.split("-").apply(lambda l: sorted(to_day(x) for x in l))
efs = pd.read_csv(EFS, dtype={"pat_id": str, "scandate": str})

rows = []
for r in efs.itertuples():
    last = to_day(r.scandate.split("-")[-1])
    event_day = last + int(r.days_last_scan_to_event)     # progression date, or end of follow-up
    pfs = int(r.pfs)
    scans = [s for s in lon[r.pat_id] if (s < event_day if pfs == 1 else s <= event_day)]
    if len(scans) < 2:
        continue
    first, prev = scans[0], None
    for t in range(1, 40):
        lm = first + 365 * t
        sel = [s for s in scans if s <= lm]
        if len(sel) < 2:
            continue
        if sel == prev:
            if lm > scans[-1]:
                break
            continue
        prev = sel
        s = sel[-1]
        gap = event_day - s
        if pfs == 1 and gap <= 365:
            label = 1
        elif gap > 365:
            label = 0
        else:
            continue
        keep = [sel[i] for i in np.round(np.linspace(0, len(sel) - 1, min(len(sel), MAX_LEN))).astype(int)]
        rows.append(dict(pat_id=r.pat_id, scandate="-".join(map(to_str, keep)), label=label,
                         landmark_year=t, n_scans=len(keep), n_scans_available=len(sel), anchor_scan=to_str(s),
                         days_anchor_to_event=gap, pfs=pfs))

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(f"rows {len(df)}, patients {df.pat_id.nunique()}, positives {int(df.label.sum())} "
      f"({df.label.mean():.1%}) -> {OUT}")
print(df.groupby("landmark_year").agg(patients=("pat_id", "nunique"), positives=("label", "sum")).T.to_string())

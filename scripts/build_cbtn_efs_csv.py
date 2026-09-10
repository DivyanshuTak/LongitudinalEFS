"""1-year EFS CSV for CBTN: label = 1 iff the event occurs within 365 d of the last scan kept.

CBTN times are age in days, as are the scandates. Anchors per patient from
cbtn_all_*_LGG.csv:
  diagnosis_age = min "Age at Event Days" over "Initial CNS Tumor" rows
  event_age     = diagnosis_age + "Event Free Survival"
EFS counts progression, recurrence, second malignancy and death as events, and
is the censoring time otherwise (both verified against the event rows).

Trajectory = scans in [diagnosis_age + postop_buffer, event_age), last --max-scans
kept. Censored back-off as in build_pbtc_efs_csv.py.
"""
import argparse

import numpy as np
import pandas as pd

EVENT_TYPES = ("Progressive", "Recurrence", "Second Malignancy", "Deceased")


def build_anchors(meta_path):
    m = pd.read_csv(meta_path, low_memory=False)
    m["age"] = pd.to_numeric(m["Age at Event Days"], errors="coerce")
    m["efs"] = pd.to_numeric(m["Event Free Survival"], errors="coerce")

    out = {}
    for pid, g in m.groupby("CBTN Subject ID"):
        init = g.loc[g["Event Type"] == "Initial CNS Tumor", "age"].dropna()
        efs = g["efs"].dropna()
        if init.empty or efs.empty:
            continue
        out[pid] = {
            "diagnosis_age": float(init.min()),
            "efs": float(efs.iloc[0]),
            "had_event": bool(g["Event Type"].isin(EVENT_TYPES).any()),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--longitudinal", default="./data/raw/cbtn/cbtn_longitudinal.csv")
    ap.add_argument("--meta", default="./data/raw/cbtn/cbtn_all_2023-09-15_LGG.csv")
    ap.add_argument("--output", default="./data/eval_csvs/cbtn/cbtn_efs1y.csv")
    ap.add_argument("--max-scans", type=int, default=6)
    ap.add_argument("--min-scans", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=365)
    ap.add_argument("--postop-buffer", type=int, default=1,
                    help="days after the diagnosis/surgery age before a scan counts as post-surgical")
    ap.add_argument("--keep-uncertain", action="store_true",
                    help="keep censored patients even when backing off cannot secure a full window")
    args = ap.parse_args()

    lon = pd.read_csv(args.longitudinal, dtype={"pat_id": str, "scandate": str})
    anchors = build_anchors(args.meta)

    rows = []
    dropped = {"no_label": 0, "no_postop_scan": 0, "too_few_scans": 0,
               "censored_short": 0}
    n_pre_op_scans = n_post_event_scans = n_backed_off = 0

    for r in lon.itertuples():
        a = anchors.get(r.pat_id)
        if a is None:
            dropped["no_label"] += 1
            continue

        scans = sorted(int(d) for d in str(r.scandate).split("-"))
        event = a["diagnosis_age"] + a["efs"]
        start = a["diagnosis_age"] + args.postop_buffer

        post_op = [s for s in scans if s >= start]
        n_pre_op_scans += len(scans) - len(post_op)
        # scans at/after the event carry the outcome -- never model inputs
        kept = [s for s in post_op if s < event]
        n_post_event_scans += len(post_op) - len(kept)
        if not kept:
            dropped["no_postop_scan"] += 1
            continue

        status = "observed" if a["had_event"] else "censored"
        if not a["had_event"] and event - kept[-1] < args.horizon:
            backed = [s for s in kept if event - s >= args.horizon]
            if len(backed) >= args.min_scans:
                n_backed_off += len(kept) - len(backed)
                kept, status = backed, "censored_backoff"
            elif args.keep_uncertain:
                status = "censored_short"
            else:
                dropped["censored_short"] += 1
                continue

        kept = kept[-args.max_scans:]
        if len(kept) < args.min_scans:
            dropped["too_few_scans"] += 1
            continue

        gap = int(event - kept[-1])
        label = int(gap <= args.horizon) if a["had_event"] else 0

        rows.append({
            "pat_id": r.pat_id,
            "scandate": "-".join(str(s) for s in kept),
            "label": label,
            "n_scans": len(kept),
            "days_last_scan_to_event": gap,
            "pfs": int(a["had_event"]),
            "status": status,
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)

    print(f"wrote {args.output}")
    print(f"  {len(out)} patients, {out.n_scans.sum()} scans, "
          f"{int(out.label.sum())} positive ({100*out.label.mean():.1f}%)")
    print(f"  scans/patient: min {out.n_scans.min()} median {out.n_scans.median():.0f} "
          f"max {out.n_scans.max()}")
    print(f"\n  dropped {len(lon) - len(out)} patients: "
          + ", ".join(f"{k}={v}" for k, v in dropped.items() if v))
    print(f"  pre-surgical scans removed: {n_pre_op_scans}")
    print(f"  post-event scans removed from kept trajectories: {n_post_event_scans}")
    print(f"  scans backed off to secure a full {args.horizon}d window: {n_backed_off}")
    print("\n  status breakdown:")
    for s, n in out.status.value_counts().items():
        pos = int(out[out.status == s].label.sum())
        print(f"     {s:<16} {n:>3}  ({pos} positive)")
    print("\n  days from last scan to event, by label:")
    for lb, g in out.groupby("label"):
        print(f"     label={lb}  n={len(g):>3}  median {g.days_last_scan_to_event.median():.0f}"
              f"  [{g.days_last_scan_to_event.min()}, {g.days_last_scan_to_event.max()}]")


if __name__ == "__main__":
    main()

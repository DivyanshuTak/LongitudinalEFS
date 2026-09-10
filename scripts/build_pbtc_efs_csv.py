"""1-year EFS CSV for PBTC: label = 1 iff progression within 365 d of the last scan kept.

event_date = first_scan + pfs_time. Scans on/after the event are trimmed. A
censored patient whose last scan has <365 d of follow-up is anchored at the
newest scan that does (`censored_backoff`); dropped if that leaves <--min-scans
unless --keep-uncertain. Trajectories keep the LAST --max-scans scans.
"""
import argparse
import csv
from datetime import datetime, timedelta

import pandas as pd

HORIZON = 365


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--longitudinal", default="./data/raw/pbtc/pbtc_longitudinal.csv")
    ap.add_argument("--labels", default="./data/raw/pbtc/efs_data_set1.csv")
    ap.add_argument("--output", default="./data/eval_csvs/pbtc/pbtc_efs1y.csv")
    ap.add_argument("--max-scans", type=int, default=6)
    ap.add_argument("--min-scans", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--keep-uncertain", action="store_true",
                    help="keep censored patients even when backing off cannot secure a full window")
    args = ap.parse_args()

    lon = pd.read_csv(args.longitudinal, dtype={"pat_id": str, "scandate": str})
    lab = pd.read_csv(args.labels, dtype={"accessionnumber": str})
    lab["pfs_time"] = pd.to_numeric(lab["pfs_time"], errors="coerce")
    lab = lab.drop_duplicates("accessionnumber").set_index("accessionnumber")

    rows, dropped = [], {"no_label": 0, "post_event_only": 0, "too_few_scans": 0,
                         "censored_short": 0}
    n_post_event_scans = n_backed_off = 0

    for r in lon.itertuples():
        if r.pat_id not in lab.index or pd.isna(lab.at[r.pat_id, "pfs_time"]):
            dropped["no_label"] += 1
            continue
        pfs = int(lab.at[r.pat_id, "pfs"])
        pfs_time = float(lab.at[r.pat_id, "pfs_time"])

        dates = sorted(datetime.strptime(d, "%Y%m%d") for d in str(r.scandate).split("-"))
        event = dates[0] + timedelta(days=pfs_time)

        # scans at/after the event carry the outcome -- never model inputs
        kept = [d for d in dates if d < event]
        n_post_event_scans += len(dates) - len(kept)
        if not kept:
            dropped["post_event_only"] += 1
            continue

        status = "observed" if pfs == 1 else "censored"
        if pfs == 0 and (event - kept[-1]).days < args.horizon:
            backed = [d for d in kept if (event - d).days >= args.horizon]
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

        gap = (event - kept[-1]).days
        label = int(gap <= args.horizon) if pfs == 1 else 0

        rows.append({
            "pat_id": r.pat_id,
            "scandate": "-".join(d.strftime("%Y%m%d") for d in kept),
            "label": label,
            "n_scans": len(kept),
            "days_last_scan_to_event": gap,
            "pfs": pfs,
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
    print(f"  post-event scans removed from kept trajectories: {n_post_event_scans}")
    print(f"  scans backed off to secure a full {args.horizon}d window: {n_backed_off}")
    print("\n  status breakdown:")
    for s, n in out.status.value_counts().items():
        pos = int(out[out.status == s].label.sum())
        print(f"     {s:<16} {n:>3}  ({pos} positive)")


if __name__ == "__main__":
    main()

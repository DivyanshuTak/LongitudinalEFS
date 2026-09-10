"""List scans in a label CSV with no embedding or fewer views than expected."""
import argparse
import collections
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="../bch_longitudinal_data_latest/efs1y_postsurg_max6.csv")
    ap.add_argument("--emb_dir", default="../bch_longitudinal_data_latest/neurovfm_embeddings")
    ap.add_argument("--n_views", type=int, default=5)
    args = ap.parse_args()

    views = collections.defaultdict(set)
    for f in os.listdir(args.emb_dir):
        if f.endswith(".npz"):
            pat, date, v = f[:-4].rsplit("_", 2)
            views[(pat, date)].add(v)

    df = pd.read_csv(args.csv, dtype={"pat_id": str, "scandate": str})
    df = df[df["pat_id"].str.strip().astype(bool)]

    missing, partial, n_scans = [], [], 0
    for r in df.itertuples():
        for d in str(r.scandate).split("-"):
            n_scans += 1
            have = views.get((r.pat_id, d))
            if not have:
                missing.append((r.pat_id, d, r.label))
            elif len(have) != args.n_views:
                partial.append((r.pat_id, d, len(have)))

    print(f"{args.csv}: {len(df)} patients, {n_scans} scans")
    print(f"  missing entirely : {len(missing)}")
    for pat, d, lab in missing:
        print(f"      {pat}_{d}  label={lab}")
    print(f"  fewer than {args.n_views} views: {len(partial)}")
    for pat, d, n in partial:
        print(f"      {pat}_{d}  views={n}")

    affected = {m[0] for m in missing} | {p[0] for p in partial}
    print(f"  patients affected: {len(affected)}")


if __name__ == "__main__":
    main()

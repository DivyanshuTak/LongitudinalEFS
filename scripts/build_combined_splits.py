"""Merge BCH and CBTN into joint train/val CSVs. A `cohort` column lets the dataset
resolve emb_dir / date_format per row. BCH keeps its DEV/TEST assignment; CBTN is
split label-stratified at --val-frac.
"""
import argparse

import pandas as pd

COLS = ["pat_id", "scandate", "label", "cohort"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bch-train", default="./splits/DEV.csv")
    ap.add_argument("--bch-val", default="./splits/TEST.csv")
    ap.add_argument("--cbtn", default="./eval_csvs/cbtn/cbtn_efs1y.csv")
    ap.add_argument("--out-train", default="./splits/combined_bch_cbtn_train.csv")
    ap.add_argument("--out-val", default="./splits/combined_bch_cbtn_val.csv")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    def load(path, cohort):
        d = pd.read_csv(path, dtype={"pat_id": str, "scandate": str})
        d["cohort"] = cohort
        return d[COLS]

    bch_tr, bch_va = load(args.bch_train, "bch"), load(args.bch_val, "bch")
    cbtn = load(args.cbtn, "cbtn")

    va = (cbtn.groupby("label", group_keys=False)
              .apply(lambda g: g.sample(frac=args.val_frac, random_state=args.seed)))
    cbtn_va = cbtn.loc[va.index]
    cbtn_tr = cbtn.drop(va.index)

    overlap = set(bch_tr.pat_id) | set(bch_va.pat_id)
    if overlap & set(cbtn.pat_id):
        raise SystemExit(f"pat_id collision across cohorts: {sorted(overlap & set(cbtn.pat_id))[:5]}")

    train = pd.concat([bch_tr, cbtn_tr]).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    val = pd.concat([bch_va, cbtn_va]).reset_index(drop=True)
    train.to_csv(args.out_train, index=False)
    val.to_csv(args.out_val, index=False)

    for name, d in [("train", train), ("val", val)]:
        print(f"{name:<6} n={len(d):>3}  pos={int(d.label.sum()):>3} ({100*d.label.mean():.1f}%)")
        for c, g in d.groupby("cohort"):
            print(f"         {c:<5} n={len(g):>3}  pos={int(g.label.sum()):>3} ({100*g.label.mean():.1f}%)")
    print(f"\nwrote {args.out_train}\n      {args.out_val}")


if __name__ == "__main__":
    main()

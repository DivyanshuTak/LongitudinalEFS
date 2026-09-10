"""Patient-level splits: a locked TEST holdout, then stratified k-fold CV over DEV."""
import argparse
import os

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def drop_missing_scans(df: pd.DataFrame, emb_dir: str) -> pd.DataFrame:
    """Drop scandates with no embedding. Refuses to drop a trajectory's final scan,
    since the label is anchored to it."""
    kept, dropped = [], 0
    for r in df.itertuples():
        dates = sorted(str(r.scandate).split("-"))
        have = [d for d in dates
                if os.path.exists(os.path.join(emb_dir, f"{r.pat_id}_{d}_v0.npz"))]
        if dates[-1] not in have:
            raise ValueError(
                f"patient {r.pat_id}: final scan {dates[-1]} has no embedding. "
                f"The label is anchored to it -- extract it or drop the patient."
            )
        if len(have) < 2:
            raise ValueError(f"patient {r.pat_id}: only {len(have)} scan(s) with embeddings")
        dropped += len(dates) - len(have)
        kept.append({"pat_id": r.pat_id, "scandate": "-".join(have), "label": r.label})
    if dropped:
        print(f"dropped {dropped} scan(s) with no embedding")
    return pd.DataFrame(kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="./data/raw/bch/efs1y_postsurg_max6.csv")
    ap.add_argument("--emb_dir", default="./data/embeddings/bch")
    ap.add_argument("--out_dir", default="./data/splits")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--test_frac", type=float, default=0.2,
                    help="Held out before folding; 0 disables the test set.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.csv, dtype={"pat_id": str, "scandate": str})
    df = df[df["pat_id"].str.strip().astype(bool)].reset_index(drop=True)
    if df["pat_id"].duplicated().any():
        raise ValueError("CSV has duplicate pat_id rows; splits must be one row per patient")
    df = drop_missing_scans(df, args.emb_dir)

    os.makedirs(args.out_dir, exist_ok=True)
    if args.test_frac > 0:
        dev, test = train_test_split(
            df, test_size=args.test_frac, stratify=df["label"], random_state=args.seed
        )
        dev, test = dev.reset_index(drop=True), test.reset_index(drop=True)
        test.to_csv(f"{args.out_dir}/TEST.csv", index=False)
        print(f"TEST : {len(test)} ({test['label'].sum()} pos)  -- score once, at the end")
    else:
        dev = df

    dev.to_csv(f"{args.out_dir}/DEV.csv", index=False)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for k, (tr, va) in enumerate(skf.split(dev, dev["label"])):
        dev.iloc[tr].to_csv(f"{args.out_dir}/fold{k}_TRAIN.csv", index=False)
        dev.iloc[va].to_csv(f"{args.out_dir}/fold{k}_VAL.csv", index=False)
        print(f"fold{k}: train {len(tr)} ({dev.iloc[tr]['label'].sum()} pos)  "
              f"val {len(va)} ({dev.iloc[va]['label'].sum()} pos)")


if __name__ == "__main__":
    main()

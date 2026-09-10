"""Extract NeuroVFM patch-token embeddings per (pat_id, scandate) timepoint: the
original volume plus N spatially augmented views, one .npz each, no pooling.

Augmentation = NeuroVFM's H/W permutation + per-axis flips, applied after
prepare_for_inference so the background mask and token count are unchanged.
The 16 possible transforms are drawn without replacement, seeded on pat_id, so
every scan of a patient gets the same transform for a given view.

  python extract_neurovfm_embeddings.py --gpu 0 --limit 5    # smoke test
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",
                   default=f"{ROOT}/bch_longitudinal_data_latest/main_data_pool_selected.csv")
    p.add_argument("--pool-dir", default=f"{ROOT}/bch_longitudinal_data_latest/main_data_pool")
    p.add_argument("--out-dir", default=f"{ROOT}/bch_longitudinal_data_latest/neurovfm_embeddings")
    p.add_argument("--model", default="mlinslab/neurovfm-encoder",
                   help="HF repo id, or local dir with config.json + pytorch_model.bin")
    p.add_argument("--gpu", default="0")
    p.add_argument("--views", type=int, default=4,
                   help="augmented views per timepoint, on top of the original (max 15)")
    p.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    p.add_argument("--limit", type=int, default=0, help="process only N timepoints")
    p.add_argument("--shard", type=int, default=0, help="this worker's index")
    p.add_argument("--num-shards", type=int, default=1,
                   help="split timepoints round-robin across N workers (one per GPU)")
    return p.parse_args()


args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu   # must precede torch import


def _preload_nvjitlink():
    """Load the pip-bundled nvJitLink before torch so a stale system CUDA lib on
    LD_LIBRARY_PATH cannot shadow it."""
    import ctypes
    import glob
    import sysconfig
    pat = os.path.join(sysconfig.get_paths()["purelib"], "nvidia/nvjitlink/lib/libnvJitLink.so*")
    for so in sorted(glob.glob(pat)):
        try:
            ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            pass


_preload_nvjitlink()

import collections                                          # noqa: E402
import csv                                                  # noqa: E402
import hashlib                                              # noqa: E402
import itertools                                            # noqa: E402
import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
from neurovfm.data.io import load_image                     # noqa: E402
from neurovfm.data.preprocess import prepare_for_inference, tokenize_volume  # noqa: E402
from neurovfm.pipelines import load_encoder                 # noqa: E402

PATCH = (4, 16, 16)
# (permute_hw, flip_axes) -- identity first, so view 0 is always un-augmented
TRANSFORMS = [(pm, fl) for pm in (False, True)
              for r in range(4)
              for fl in itertools.combinations((0, 1, 2), r)]
TRANSFORMS.sort(key=lambda t: (t[0], len(t[1]), t[1]))


def transforms_for(pat_id, n_views):
    """Identity first, then a patient-seeded permutation of the remaining 15."""
    rest = TRANSFORMS[1:]
    seed = int(hashlib.md5(pat_id.encode()).hexdigest()[:8], 16)
    order = np.random.default_rng(seed).permutation(len(rest))
    return [TRANSFORMS[0]] + [rest[i] for i in order[:n_views]]


def apply_transform(img, mask, tf):
    """H/W permutation then per-axis flips, on [D, H, W] arrays."""
    permute_hw, flip_axes = tf
    if permute_hw:
        img, mask = img.transpose(0, 2, 1), mask.transpose(0, 2, 1)
    if flip_axes:
        img, mask = np.flip(img, flip_axes), np.flip(mask, flip_axes)
    return np.ascontiguousarray(img), np.ascontiguousarray(mask)


MODALITY_ORDER = ("t1ce", "t2", "flair")


def timepoints_from_manifest(path, pool_dir):
    """Manifest (one series per modality) -> {(pat_id, scandate): [(path, modality)]} in MODALITY_ORDER."""
    tps = collections.OrderedDict()
    for r in csv.DictReader(open(path)):
        tps.setdefault((r["pat_id1"], r["scandate"]), {})[r["modality"]] = \
            os.path.join(pool_dir, r["dest_name"])
    out = collections.OrderedDict()
    for k, mods in tps.items():
        out[k] = [(mods[m], m) for m in MODALITY_ORDER if m in mods]
    return out


def build_batch(entries, tf):
    """Preprocess + augment + tokenize one timepoint -> encoder batch dict."""
    toks, coords, lengths, modes, paths, sizes, cats = [], [], [], [], [], [], []
    for path, cat in entries:
        # every modality must load, or series index != modality downstream
        img_sitk = load_image(path, preprocess=True)
        if img_sitk is None:
            raise RuntimeError(f"unreadable volume ({cat}): {os.path.basename(path)}")
        prepared = prepare_for_inference(img_sitk, mode="mri")
        if prepared is None:            # rejected: D<4, H<16 or W<16
            raise RuntimeError(f"rejected by prepare_for_inference ({cat}): {os.path.basename(path)}")
        img_arrs, bg_mask, _ = prepared
        img, mask = apply_transform(img_arrs[0], bg_mask, tf)

        t, c, _ = tokenize_volume(img, mask, patch_size=PATCH, remove_background=True)
        if len(t) == 0:
            raise RuntimeError(f"no foreground tokens ({cat}): {os.path.basename(path)}")
        toks.append(torch.from_numpy(t).float())
        coords.append(torch.from_numpy(c).long())
        lengths.append(len(t))
        modes.append("mri")
        paths.append(path)
        sizes.append(img.shape)
        cats.append(cat)

    if not toks:
        return None, None, None

    cu = torch.zeros(len(lengths) + 1, dtype=torch.int32)
    cu[1:] = torch.tensor(lengths, dtype=torch.int32).cumsum(0)
    batch = {
        "img": torch.cat(toks, 0),
        "coords": torch.cat(coords, 0),
        "series_masks_indices": torch.tensor([]),   # background already removed
        "series_cu_seqlens": cu,
        "series_max_len": max(lengths),
        "study_cu_seqlens": torch.tensor([0, cu[-1]], dtype=torch.int32),
        "study_max_len": len(lengths),
        "mode": modes,
        "path": paths,
        "size": sizes,
    }
    return batch, paths, cats


def main():
    tps = timepoints_from_manifest(args.manifest, args.pool_dir)
    if args.limit:
        tps = collections.OrderedDict(list(tps.items())[:args.limit])
    if args.num_shards > 1:
        items = list(tps.items())[args.shard::args.num_shards]
        tps = collections.OrderedDict(items)
        print(f"shard {args.shard}/{args.num_shards}: {len(tps)} timepoints", flush=True)
    if not 0 <= args.views <= len(TRANSFORMS) - 1:
        sys.exit(f"--views must be 0..{len(TRANSFORMS) - 1}")
    os.makedirs(args.out_dir, exist_ok=True)
    np_dtype = np.float16 if args.dtype == "fp16" else np.float32
    print(f"timepoints={len(tps)} views={args.views}+1 dtype={args.dtype}", flush=True)

    encoder, _ = load_encoder(args.model, device="cuda")

    n_done = n_skip = n_err = 0
    for i, ((pid, sd), entries) in enumerate(tps.items(), 1):
        for v, tf in enumerate(transforms_for(pid, args.views)):
            out = os.path.join(args.out_dir, f"{pid}_{sd}_v{v}.npz")
            if os.path.exists(out):
                n_skip += 1
                continue
            try:
                batch, paths, cats = build_batch(entries, tf)
                if batch is None:
                    raise RuntimeError("no loadable volumes")
                embs = encoder.embed(batch).float().cpu().numpy().astype(np_dtype)
                np.savez_compressed(
                    out,
                    tokens=embs,                                    # (N_tokens, D)
                    series_cu_seqlens=batch["series_cu_seqlens"].numpy(),
                    coords=batch["coords"].numpy().astype(np.int16),
                    paths=np.array([os.path.basename(p) for p in paths], dtype=object),
                    categories=np.array(cats, dtype=object),
                    pat_id=pid, scandate=sd, view=v,
                    permute_hw=tf[0], flip_axes=np.array(tf[1], dtype=np.int8),
                )
                n_done += 1
            except Exception as e:
                n_err += 1
                print(f"ERR {pid}_{sd}_v{v}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        if i % 100 == 0:
            print(f"[{i}/{len(tps)}] done={n_done} skipped={n_skip} err={n_err}", flush=True)

    print(f"\nDONE done={n_done} skipped={n_skip} errors={n_err} -> {args.out_dir}")


if __name__ == "__main__":
    main()

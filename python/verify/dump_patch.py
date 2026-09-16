"""
dump_patch.py - save one real, normalised HYPSO patch to patch0.npy for
forward_driver.py --input-npy.

Mirrors train.py's dataset construction EXACTLY (legacy_raw_mode honoured),
so the patch is normalised identically to what the model saw in training:

    index      = load_capture_index(data_root, cube_suffix, legacy_raw_mode)
    split      = get_or_create_split(list(index.keys()), split_path)
    stats      = load_stats(stats_path)                 # cached train-split stats
    normaliser = Normaliser(stats) if normalisation.enabled else None
    ds         = HYPSOPatchDataset(index, split[<split>], patch_size=...,
                                   normaliser=normaliser, augmentation=None,
                                   random_offset=False, band_indices=...)

All the optional normalisation flags (per_capture_norm, per_pixel_l2,
magnitude_channel, magnitude_standardise) default False and are absent from
the scratch config, so they're off here, exactly as in train.py.

Run from the repo root (same cwd you launch scripts/train.py from):

  python3 dump_patch.py \
      --config configs/bnn_unet_cpba_a2_dualskip_bireal_student_scratch.yaml \
      --split test --idx 0 --out patch0.npy
"""

import argparse
import os
import sys

# make the training framework importable regardless of cwd: point HYPSO_REPO
# at the training repo root (same var the exporter uses).
_hrepo = os.environ.get("HYPSO_REPO")
if _hrepo and os.path.abspath(_hrepo) not in map(os.path.abspath, sys.path):
    sys.path.insert(0, _hrepo)

import numpy as np
import yaml

from data.dataset import (load_capture_index, HYPSOPatchDataset,
                          CUBE_SUFFIX_L1A, CUBE_SUFFIX_L1B, parse_band_selection)
from data.splits import get_or_create_split
from data.preprocessing import load_stats, Normaliser


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--out", default="patch0.npy")
    ap.add_argument("--project-dir", default=".",
                    help="repo root; config paths resolve against it. train.py "
                         "uses the parent of scripts/, i.e. the repo root; run "
                         "this from there and the default '.' works")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    pd = args.project_dir
    data_root  = os.path.join(pd, cfg["data"]["root"])
    split_path = os.path.join(pd, cfg["data"]["split_path"])
    stats_path = os.path.join(pd, cfg["data"]["stats_path"])

    legacy_raw_mode = cfg["data"].get("legacy_raw_mode", False)
    if legacy_raw_mode:
        cube_suffix = None
    else:
        cube_variant = cfg["data"].get("cube_variant", "l1b").lower()
        cube_suffix = CUBE_SUFFIX_L1A if cube_variant == "l1a" else CUBE_SUFFIX_L1B

    index = load_capture_index(data_root, cube_suffix=cube_suffix,
                               legacy_raw_mode=legacy_raw_mode)
    split = get_or_create_split(list(index.keys()), split_path)

    band_selection_spec = cfg["data"].get("band_selection", None)
    band_indices = (parse_band_selection(band_selection_spec, total_bands=120)
                    if band_selection_spec is not None else None)

    if not os.path.exists(stats_path):
        raise SystemExit(
            f"stats file not found at {stats_path}. It is normally created by "
            f"a training run. Run training once (or point --project-dir at the "
            f"repo root) so splits/stats_v1.npz exists.")
    stats = load_stats(stats_path)
    normaliser = Normaliser(stats) if cfg["normalisation"]["enabled"] else None

    ds = HYPSOPatchDataset(
        index, split[args.split], patch_size=cfg["data"]["patch_size"],
        normaliser=normaliser, augmentation=None,
        per_capture_norm=cfg["normalisation"].get("per_capture", False),
        random_offset=False, band_indices=band_indices,
        per_pixel_l2=cfg["normalisation"].get("per_pixel_l2", False),
        magnitude_channel=cfg["normalisation"].get("magnitude_channel", False),
        magnitude_stats=None,
        magnitude_standardise=cfg["normalisation"].get("magnitude_standardise", False),
    )

    x, y = ds[args.idx]
    x = x.numpy().astype(np.float32)
    np.save(args.out, x)

    print(f"\nsaved {args.out}  shape {x.shape}  dtype {x.dtype}")
    print(f"value range: min={x.min():.4f} max={x.max():.4f} "
          f"mean={x.mean():.4f} std={x.std():.4f}")
    print(f"label values in patch: {y.unique().tolist()}")
    looks_raw = (x.min() >= -1e-3 and x.max() > 20) or x.std() > 10
    print("normalisation check:",
          "** looks UNNORMALISED, do not use **" if looks_raw
          else "range looks normalised (OK to use)")


if __name__ == "__main__":
    main()
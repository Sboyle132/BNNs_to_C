"""export.py - export a trained BNN checkpoint to weights.blob for the C engine.

Handles THREE:
  * the original bireal models (no stem)          -> v1 blob -> bnn_infer
  * the spectral-stem A1b variant (--stem a1b)    -> v2 blob -> bnn_infer_stem
  * the ReActNet real-highway variant (--react)   -> v3 blob -> bnn_infer_react

blob.write_blob auto-selects v1/v2 from the model (presence of a stem), so the
only thing --stem changes here is HOW THE MODEL IS BUILT: the stem wrapper
needs stem_weight/stem_out flags that the generic loader does not pass, so the
A1b model is constructed directly. The v1 path is unchanged.

  # v1 (unchanged):
  python3 export.py \
      --model-module models.bnn_unet_cpba_a2_dualskip_bireal_student \
      --model-class  BNN_UNet_CPBA_A2_DualSkip_BiReal_Student \
      --checkpoint   runs/.../best_model.pt \
      --upsampler    nearest --out artifacts/weights.blob

  # v2 (spectral stem A1b):
  python3 export.py --stem a1b \
      --checkpoint runs/bnn_unet_cpba_a2_dualskip_fp32_wbin_abin_stem/best_model.pt \
      --stem-channels 16 --out artifacts/weights_stem.blob

  # v3 (ReActNet real highway):
  python3 export.py --react \
      --checkpoint runs/bnn_unet_cpba_a2_dualskip_bireal_reactnet/best_model.pt \
      --out artifacts/weights_react.blob
"""

import argparse
import os
import sys
from types import SimpleNamespace

# make sibling packages importable regardless of cwd. Order matters: the
# port's own modules (this dir + python/verify) must win over same-named
# files left in the training repo, so HYPSO_REPO is inserted first (lowest
# priority) and the port dirs last (highest).
_here = os.path.dirname(os.path.abspath(__file__))
for _p in (os.environ.get("HYPSO_REPO"),
           os.path.abspath(os.path.join(_here, "..", "verify")),
           _here):
    if _p and os.path.abspath(_p) not in map(os.path.abspath, sys.path):
        sys.path.insert(0, _p)

import numpy as np

import blob as BLOB


def _snap_zero_weights(model, eps=1e-6):
    """Set any exact-zero HardBinaryConv weight to +eps so sign()->+1. Returns
    count snapped. XNOR-popcount is 1-bit and cannot represent a 0 weight; the
    model's sign(0)=0 would otherwise make the C accumulator differ by 1 at
    every output pixel of the affected channel. Covers stem AND body convs."""
    import torch
    n = 0
    with torch.no_grad():
        for m in model.modules():
            if type(m).__name__ == "HardBinaryConv":
                z = (m.weight == 0)
                c = int(z.sum())
                if c:
                    m.weight[z] = eps
                    n += c
    return n


def _load_checkpoint_direct(model, path):
    import torch
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=True)


def _build_stem_a1b(args):
    """Construct the A1b spectral-stem model with the correct (non-default)
    stem flags, which the generic loader cannot pass."""
    from models.bnn_unet_cpba_a2_dualskip_bireal_stem import (
        BNN_UNet_CPBA_A2_DualSkip_BiReal_Stem as Stem)
    model = Stem(n_bands=args.n_bands, n_classes=args.n_classes,
                 base_channels=args.base_channels,
                 stem_channels=args.stem_channels, stem_layers=1,
                 stem_weight="binary", stem_out="binary", stem_act="prelu")
    _load_checkpoint_direct(model, args.checkpoint)
    return model


def _build_react(args):
    """Construct the ReActNet model directly (same constructor args as B0), so
    this path never depends on dump_model_spec.load_real_model."""
    from models.bnn_unet_cpba_a2_dualskip_bireal_reactnet import (
        BNN_UNet_CPBA_A2_DualSkip_BiReal_ReActNet as ReAct)
    model = ReAct(n_bands=args.n_bands, n_classes=args.n_classes,
                  base_channels=args.base_channels)
    _load_checkpoint_direct(model, args.checkpoint)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-module")
    ap.add_argument("--model-class")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--upsampler", default="nearest",
                    choices=["nearest", "pixshuf", "nnrefine"])
    ap.add_argument("--out", default="artifacts/weights.blob")
    ap.add_argument("--use-registry", action="store_true")
    ap.add_argument("--config")
    ap.add_argument("--n-bands", type=int, default=120)
    ap.add_argument("--n-classes", type=int, default=3)
    ap.add_argument("--base-channels", type=int, default=16)
    ap.add_argument("--patch", type=int, default=32)
    # spectral-stem variant
    ap.add_argument("--stem", choices=["none", "a1b"], default="none",
                    help="'a1b' builds the binary-weight binarised-output "
                         "spectral-stem model (writes a v2 blob)")
    ap.add_argument("--stem-channels", type=int, default=16)
    # ReActNet real-highway variant
    ap.add_argument("--react", action="store_true",
                    help="build the ReActNet model and write a v3 blob "
                         "(consumed by csrc/bnn_model_react.c, bnn_infer_react)")
    args = ap.parse_args()

    import torch

    if args.react and args.stem != "none":
        raise SystemExit("--react and --stem are mutually exclusive")

    if args.react:
        model = _build_react(args)
    elif args.stem == "a1b":
        model = _build_stem_a1b(args)
    else:
        from dump_model_spec import load_real_model
        ns = SimpleNamespace(
            model_module=args.model_module, model_class=args.model_class,
            use_registry=args.use_registry, config=args.config,
            checkpoint=args.checkpoint, input_npy=None,
            n_bands=args.n_bands, n_classes=args.n_classes,
            base_channels=args.base_channels, patch=args.patch)
        model, _ = load_real_model(ns)
    model.eval()

    # sign(0)=0 in HardBinaryConv, but XNOR-popcount is 1-bit. Snap any exact-
    # zero weights to a tiny +epsilon IN THE MODEL so every derived quantity
    # (alpha, fold thresholds, the C accumulator) sees the same +1 sign.
    n_snapped = _snap_zero_weights(model)
    if n_snapped:
        print(f"snapped {n_snapped} exact-zero weight(s) to +1 (XNOR cannot "
              f"represent 0)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    if args.react:
        # v3: separate writer/reader (blob_react.py), v1/v2 write paths untouched.
        import blob_react as BR
        BR.write_blob_react(model, args.out, base_channels=args.base_channels,
                            n_bands=args.n_bands, n_classes=args.n_classes,
                            upsampler=args.upsampler)
        b = BR.read_blob_react(args.out)       # round-trip (asserts layout, no trailing bytes)
        assert b["version"] == 3 and b["n_bands"] == args.n_bands
        assert b["n_classes"] == args.n_classes
        assert len(b["blocks"]) == 9 and b["blocks"]["enc1"]["conv1"]["tag"] == "real"
        assert "tap1" not in b["blocks"]["enc1"], "enc1 has no tap1 (raw spectra in)"
        size = os.path.getsize(args.out)
        print(f"wrote {args.out}  ({size:,} bytes)  [v3 ReActNet]")
        print(f"  base_channels={b['base_channels']} n_bands={b['n_bands']} "
              f"n_classes={b['n_classes']} upsampler={args.upsampler}")
        print(f"  9 blocks (4 down + bottleneck + 4 dec), 18 convs, 9 projs, "
              f"22 epilogues, 17 taps, round-trip OK")
        return

    BLOB.write_blob(model, args.out, upsampler=args.upsampler,
                    n_bands=args.n_bands, patch=args.patch)

    # self-check: round-trip
    b = BLOB.read_blob(args.out)
    assert b["n_bands"] == args.n_bands and b["n_classes"] == args.n_classes
    assert len(b["convs"]) == 18 and len(b["acts"]) == 22
    size = os.path.getsize(args.out)
    if b["version"] == BLOB.VERSION_STEM:
        assert b["has_stem"] == 1 and b["n_stem_layers"] == 1
        assert b["stem_conv"]["tag"] == "real" and b["stem_conv"]["kh"] == 1
        assert b["stem_act"]["kind"] == "real"
        assert b["convs"][0]["tag"] == "bin", "enc1.conv1 should be BIN in A1b"
        assert b["acts"][0]["kind"] == "int", "enc1.act1 should be INT in A1b"
        print(f"wrote {args.out}  ({size:,} bytes)  [v2 spectral-stem A1b]")
        print(f"  base_channels={b['base_channels']} n_bands={b['n_bands']} "
              f"n_classes={b['n_classes']} stem_channels={args.stem_channels} "
              f"upsampler={args.upsampler}")
        print(f"  stem: 1x1 real-input binary conv + real fold; "
              f"enc1 role-swap OK; 18 convs, 22 folds, round-trip OK")
    else:
        print(f"wrote {args.out}  ({size:,} bytes)  [v1]")
        print(f"  base_channels={b['base_channels']} n_bands={b['n_bands']} "
              f"n_classes={b['n_classes']} upsampler={args.upsampler}")
        print(f"  18 convs, 22 activation folds, round-trip OK")


if __name__ == "__main__":
    main()

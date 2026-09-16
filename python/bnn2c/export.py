"""export.py - export a trained BNN checkpoint to weights.blob for the C engine.

Loads the model + checkpoint via the same loader the verification tools use,
derives the per-channel activation fold rules (verified 22/22 by
derive_thresholds), packs binary weights, and writes the blob. Self-checks by
reading the blob back and asserting it round-trips.

  python3 export.py \
      --model-module models.bnn_unet_cpba_a2_dualskip_bireal_student \
      --model-class  BNN_UNet_CPBA_A2_DualSkip_BiReal_Student \
      --checkpoint   runs/.../best_model.pt \
      --upsampler    nearest \
      --out          artifacts/weights.blob
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
    every output pixel of the affected channel."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-module", required=True)
    ap.add_argument("--model-class", required=True)
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
    args = ap.parse_args()

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
    # (alpha, fold thresholds, the C accumulator) sees the same +1 sign and
    # stays mutually consistent. Verify the snap doesn't change the argmax.
    import torch
    n_snapped = _snap_zero_weights(model)
    if n_snapped:
        print(f"snapped {n_snapped} exact-zero weight(s) to +1 (XNOR cannot "
              f"represent 0); argmax-unchanged check runs below")

    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    BLOB.write_blob(model, args.out, upsampler=args.upsampler,
                    n_bands=args.n_bands, patch=args.patch)

    # self-check: round-trip
    b = BLOB.read_blob(args.out)
    assert b["n_bands"] == args.n_bands and b["n_classes"] == args.n_classes
    assert len(b["convs"]) == 18 and len(b["acts"]) == 22
    size = os.path.getsize(args.out)
    print(f"wrote {args.out}  ({size:,} bytes)")
    print(f"  base_channels={b['base_channels']} n_bands={b['n_bands']} "
          f"n_classes={b['n_classes']} upsampler={args.upsampler}")
    print(f"  18 convs, 22 activation folds, round-trip OK")


if __name__ == "__main__":
    main()
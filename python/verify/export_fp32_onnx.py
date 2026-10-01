#!/usr/bin/env python3
"""
export_fp32_onnx.py - export the FP32 twin of the CPBA-A2 dual-skip student
to a FIXED-SHAPE ONNX file for timing on a CPU board (ZCU104 PS, onnxruntime).

Random init (fixed seed): runtime is data-independent, no checkpoint needed.
BN statistics and affine params are randomised so BN is not an identity and
the conv+BN fusion in the exporter/ORT is actually exercised by --check.

Cheap-first validation: run once with a tiny odd shape and --check (compares
torch vs onnxruntime, covers both pad branches of the nearest upsample-cat),
then export at the real capture size without --check.

  # 1. validate the export path (seconds)
  python3 export_fp32_onnx.py --h 37 --w 53 --bands 120 --check --out /tmp/chk.onnx
  # 2. real export (B0 twin), then A1b-style stem twin
  python3 export_fp32_onnx.py --h 598 --w 1092 --bands 120 --out artifacts/unet_dualskip_fp32_120b_598x1092.onnx
  python3 export_fp32_onnx.py --h 598 --w 1092 --bands 120 --stem-channels 16 \
      --out artifacts/unet_dualskip_fp32_stem16_598x1092.onnx

Needs HYPSO_REPO (models/unet_cpba_a2_dualskip_fp32.py lives in the training
repo's models/), plus: pip install onnx onnxruntime
"""
import argparse
import os
import sys

import numpy as np
import torch

_repo = os.environ.get("HYPSO_REPO", "")
if _repo:
    sys.path.insert(0, os.path.expanduser(_repo))

from models.unet_cpba_a2_dualskip_fp32 import UNet_CPBA_A2_DualSkip_FP32  # noqa: E402


def randomise_bn(model, seed):
    g = torch.Generator().manual_seed(seed)
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            n = m.num_features
            m.running_mean.copy_(torch.randn(n, generator=g) * 0.1)
            m.running_var.copy_(torch.rand(n, generator=g) + 0.5)
            m.weight.data.copy_(torch.rand(n, generator=g) + 0.5)
            m.bias.data.copy_(torch.randn(n, generator=g) * 0.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=598)
    ap.add_argument("--w", type=int, default=1092)
    ap.add_argument("--bands", type=int, default=120)
    ap.add_argument("--classes", type=int, default=3)
    ap.add_argument("--base-channels", type=int, default=16)
    ap.add_argument("--stem-channels", type=int, default=0,
                    help="0 = B0 twin (no stem); 16 = FP32 counterpart of the A1b front end")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true",
                    help="compare torch vs onnxruntime on this shape (use a small shape)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model = UNet_CPBA_A2_DualSkip_FP32(
        n_bands=args.bands, n_classes=args.classes,
        base_channels=args.base_channels, stem_channels=args.stem_channels,
    )
    randomise_bn(model, args.seed)
    model.eval()
    print(f"[export] params={model.count_parameters():,}  "
          f"bands={args.bands}  stem={args.stem_channels}  "
          f"shape=1x{args.bands}x{args.h}x{args.w}")

    x = torch.randn(1, args.bands, args.h, args.w)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    kw = dict(opset_version=args.opset, input_names=["x"], output_names=["logits"],
              do_constant_folding=True)
    with torch.no_grad():
        try:
            torch.onnx.export(model, x, args.out, dynamo=False, **kw)
        except TypeError:           # older torch: no dynamo kwarg
            torch.onnx.export(model, x, args.out, **kw)
        ref = model(x).numpy() if args.check else None

    print(f"[export] wrote {args.out}  ({os.path.getsize(args.out) / 1e6:.2f} MB)")

    if args.check:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        got = sess.run(None, {"x": x.numpy()})[0]
        err = float(np.abs(ref - got).max())
        ok = np.allclose(ref, got, rtol=1e-3, atol=1e-4)
        print(f"[check] torch vs onnxruntime  max|diff|={err:.3e}  "
              f"ref max|.|={float(np.abs(ref).max()):.3f}  -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
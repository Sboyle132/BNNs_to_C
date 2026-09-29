"""verify_stem.py - independent structural + fold verification for the A1b
spectral-stem model, reusing derive_thresholds' generic (name-agnostic) chain
machinery. No existing verifier is modified.

Confirms, by MEASUREMENT (independent of the exporter's forced choice):
  - the stem act is a REAL accumulator chain (fed by the real-input 1x1 conv),
  - body.enc1.act1 is now an INTEGER-P chain (fed by the XNOR enc1.conv1),
  - every chain's fold reproduces the model's own sign bits (0 mismatches).

This is the check that catches the real/binary role-swap if the exporter's
force_real set is ever wired wrong: verify_stem uses measurement, export_stem
uses the structural force set; they must agree.

  python3 verify_stem.py \
      --checkpoint runs/bnn_unet_cpba_a2_dualskip_fp32_wbin_abin_stem/best_model.pt \
      --stem-channels 16
"""

import argparse
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
for _p in (os.environ.get("HYPSO_REPO"), _here):
    if _p and os.path.abspath(_p) not in map(os.path.abspath, sys.path):
        sys.path.insert(0, _p)

import torch
import derive_thresholds as DT


def _load_checkpoint(model, path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-bands", type=int, default=120)
    ap.add_argument("--n-classes", type=int, default=3)
    ap.add_argument("--base-channels", type=int, default=16)
    ap.add_argument("--stem-channels", type=int, default=16)
    ap.add_argument("--patch", type=int, default=32)
    ap.add_argument("--input-npy", default=None,
                    help="a normalised patch .npy for a real golden reference "
                         "(random input is fine for structure/fold checks)")
    args = ap.parse_args()

    from models.bnn_unet_cpba_a2_dualskip_bireal_stem import (
        BNN_UNet_CPBA_A2_DualSkip_BiReal_Stem as Stem)
    model = Stem(n_bands=args.n_bands, n_classes=args.n_classes,
                 base_channels=args.base_channels,
                 stem_channels=args.stem_channels, stem_layers=1,
                 stem_weight="binary", stem_out="binary", stem_act="prelu")
    _load_checkpoint(model, args.checkpoint)
    model.eval()

    if args.input_npy:
        import numpy as np
        arr = np.load(args.input_npy)
        if arr.ndim == 3:
            arr = arr[None]
        x = torch.from_numpy(arr).float()
    else:
        x = torch.randn(1, args.n_bands, args.patch, args.patch)

    trace = DT.trace_model(model, x)
    chains = DT.group_chains(trace)
    results = [DT.process_chain(ch) for ch in chains]

    n_int = sum(r["accumulator_integer"] for r in results)
    n_real = len(results) - n_int
    n_fail = sum(not r["verify"]["pass"] for r in results)

    print(f"\nchains: {len(results)}   integer-P: {n_int}   real-accum: {n_real}   "
          f"fold-mismatch chains: {n_fail}\n")
    print(f"{'chain':40s} {'accum':8s} {'conv':28s} {'bits':>10s} pass")
    stem_ok = enc1_ok = None
    for r in results:
        acc = "integer" if r["accumulator_integer"] else "REAL"
        mism = r["verify"]["total_bit_mismatches"]
        print(f"{r['name']:40s} {acc:8s} {str(r['conv']):28s} "
              f"{mism:>10d} {'OK' if r['verify']['pass'] else 'FAIL'}")
        nm = r["name"]
        if nm.split(".", 1)[0] == "stem":
            stem_ok = (not r["accumulator_integer"])   # stem act must be REAL
        if nm.endswith("enc1.act1"):
            enc1_ok = r["accumulator_integer"]          # enc1.act1 must be INTEGER

    print("\n--- role-swap assertions (A1b) ---")
    print(f"  stem act is REAL accumulator : {stem_ok}")
    print(f"  enc1.act1 is INTEGER-P       : {enc1_ok}")
    ok = (n_fail == 0) and bool(stem_ok) and bool(enc1_ok)
    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT — do not export'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

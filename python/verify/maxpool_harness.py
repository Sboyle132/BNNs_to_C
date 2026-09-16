"""
maxpool_harness.py - verify maxpool_P (integer 2x2 max on the accumulator P)
reproduces the model's down-branch pooled accumulator, for the four encoder
DualSkipDownBlocks.

For each enc{1,2,3,4}: the block computes conv2_out = alpha * P, then the down
branch does pool = MaxPool2d(2)(conv2_out). This harness captures conv2's
output and the pool's output via hooks, recovers the integer P = round(
conv2_out / alpha), runs the C integer max-pool on P, and checks

    alpha * maxpool_P(P)  ==  model's pool output    (float precision)

and that maxpool_P(P) is integer. This is the +-1 -> integer-P -> pooled-P
path the port's down branch uses, verified against the model.

  --self-test : checks maxpool_P against a numpy reference on synthetic
                integer P (no model needed).
  real run    : loads the model, hooks each enc*.conv2 and enc*.pool.

  python3 maxpool_harness.py \
      --model-module models.bnn_unet_cpba_a2_dualskip_bireal_student \
      --model-class  BNN_UNet_CPBA_A2_DualSkip_BiReal_Student \
      --checkpoint   runs/.../best_model.pt
"""

import argparse
import ctypes
import os
import subprocess
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSRC = os.path.abspath(os.path.join(HERE, "..", "..", "csrc"))
LIB = os.path.join(CSRC, "libbnnmaxpool.so")
SRC = os.path.join(CSRC, "bnn_maxpool.c")


def build():
    subprocess.run(["gcc", "-O3", "-march=native", "-shared", "-fPIC",
                    "-o", LIB, SRC], check=True)
    lib = ctypes.CDLL(LIB)
    i32 = ctypes.POINTER(ctypes.c_int32)
    ci = ctypes.c_int
    lib.maxpool_P.argtypes = [i32, i32, ci, ci, ci, ci, ci]
    return lib


def p_i32(a):
    a = np.ascontiguousarray(a, dtype=np.int32)
    return a, a.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def run_maxpool(lib, P, k=2, stride=2):
    C, H, W = P.shape
    Hout = (H - k) // stride + 1
    Wout = (W - k) // stride + 1
    P_c, P_p = p_i32(P)
    out, out_p = p_i32(np.zeros((C, Hout, Wout)))
    lib.maxpool_P(P_p, out_p, C, H, W, k, stride)
    return out


def numpy_maxpool(P, k=2, stride=2):
    C, H, W = P.shape
    Hout = (H - k) // stride + 1
    Wout = (W - k) // stride + 1
    out = np.zeros((C, Hout, Wout), dtype=np.int64)
    for c in range(C):
        for i in range(Hout):
            for j in range(Wout):
                out[c, i, j] = P[c, i*stride:i*stride+k, j*stride:j*stride+k].max()
    return out


def self_test(lib):
    rng = np.random.default_rng(0)
    all_ok = True
    for C, H, W in [(8, 8, 8), (16, 6, 10), (4, 4, 4)]:
        P = rng.integers(-300, 300, size=(C, H, W)).astype(np.int32)
        ref = numpy_maxpool(P)
        got = run_maxpool(lib, P)
        d = int(np.abs(got.astype(np.int64) - ref).max())
        ok = d == 0
        all_ok &= ok
        print(f"case C={C} H={H} W={W}: maxdiff={d} {'OK' if ok else '** FAIL **'}")
    print("\nSELF-TEST", "PASS" if all_ok else "FAIL")
    return all_ok


def real_run(lib, args):
    import torch
    from dump_model_spec import load_real_model
    ns = SimpleNamespace(
        model_module=args.model_module, model_class=args.model_class,
        use_registry=args.use_registry, config=args.config,
        checkpoint=args.checkpoint, input_npy=args.input_npy,
        n_bands=args.n_bands, n_classes=args.n_classes,
        base_channels=args.base_channels, patch=args.patch)
    model, x = load_real_model(ns)

    id2name = {id(m): n for n, m in model.named_modules()}
    conv2_out = {}
    pool_io = {}

    def mk_conv(m):
        def hook(mod, inp, out):
            name = id2name[id(mod)]
            if name.endswith(".conv2") and name.startswith("enc"):
                conv2_out[name.split(".")[0]] = (mod, out.detach().cpu().numpy())
        return hook

    def mk_pool(m):
        def hook(mod, inp, out):
            name = id2name[id(mod)]
            if name.endswith(".pool"):
                pool_io[name.split(".")[0]] = (inp[0].detach().cpu().numpy(),
                                               out.detach().cpu().numpy())
        return hook

    handles = []
    for n, m in model.named_modules():
        if type(m).__name__ == "HardBinaryConv":
            handles.append(m.register_forward_hook(mk_conv(m)))
        if type(m).__name__ == "MaxPool2d":
            handles.append(m.register_forward_hook(mk_pool(m)))
    model.eval()
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()

    stages = sorted(pool_io.keys())
    if not stages:
        print("WARNING: no encoder pool layers found")
        return
    print(f"{'stage':<7} {'P-shape':<16} {'pooled-shape':<16} {'P_int':>6} "
          f"{'reconΔ':>10} result")
    n_pass = 0
    for st in stages:
        mod, co = conv2_out[st]              # conv2 module + output (alpha*P)
        pin, pout = pool_io[st]              # pool input (== co) and output
        w = mod.weight.detach().cpu().numpy()
        alpha = np.abs(w).mean(axis=(1, 2, 3))
        Pf = co[0] / alpha[:, None, None]
        P = np.round(Pf).astype(np.int32)
        p_int_ok = float(np.abs(Pf - P).max()) < 1e-2
        Ppool = run_maxpool(lib, P)          # integer max-pool
        recon = alpha[:, None, None] * Ppool  # back to alpha*P_pooled
        d = float(np.abs(recon - pout[0]).max())
        ok = (d / (float(np.abs(pout[0]).max()) + 1e-12) < 1e-4) and p_int_ok
        n_pass += ok
        print(f"{st:<7} {str(P.shape):<16} {str(Ppool.shape):<16} "
              f"{str(p_int_ok):>6} {d:>10.3e} {'PASS' if ok else '** FAIL **'}")
    print(f"\nencoder down-branch pools checked: {len(stages)}   PASS: {n_pass}/{len(stages)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--model-module")
    ap.add_argument("--model-class")
    ap.add_argument("--use-registry", action="store_true")
    ap.add_argument("--config")
    ap.add_argument("--checkpoint")
    ap.add_argument("--input-npy")
    ap.add_argument("--n-bands", type=int, default=120)
    ap.add_argument("--n-classes", type=int, default=3)
    ap.add_argument("--base-channels", type=int, default=16)
    ap.add_argument("--patch", type=int, default=32)
    args = ap.parse_args()

    lib = build()
    if args.self_test or not args.model_module:
        self_test(lib)
    else:
        real_run(lib, args)


if __name__ == "__main__":
    main()

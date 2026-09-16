"""
bconv_harness.py - verify the C binary-conv kernels reproduce the model's
integer accumulator P for the 13 pure-binary convs, exactly.

Builds bnn_conv.c into a shared lib, then:
  --self-test : checks both kernels against a NumPy reference on synthetic
                +-1 data (no model needed). Proves packing/popcount/padding.
  real run    : loads the model (same loader as the dumper), runs one forward
                pass with hooks, and for every conv whose input is +-1 compares
                naive-C and xnor-C accumulators against round(conv_out / alpha),
                which is the model's own P.

Reference for P per pure-binary conv: the model computes conv_out = alpha * P
with alpha = mean(|w|) per out channel, so P = round(conv_out / alpha).

  python3 bconv_harness.py \
      --model-module models.bnn_unet_cpba_a2_dualskip_bireal_student \
      --model-class  BNN_UNet_CPBA_A2_DualSkip_BiReal_Student \
      --checkpoint   runs/bnn_unet_cpba_a2_dualskip_bireal_student_scratch/best_model.pt
"""

import argparse
import ctypes
import os
import subprocess
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSRC = os.path.abspath(os.path.join(HERE, "..", "..", "csrc"))
LIB = os.path.join(CSRC, "libbnnconv.so")
SRC = os.path.join(CSRC, "bnn_conv.c")


def build():
    subprocess.run(["gcc", "-O3", "-march=native", "-shared", "-fPIC",
                    "-o", LIB, SRC], check=True)
    lib = ctypes.CDLL(LIB)
    i8 = ctypes.POINTER(ctypes.c_int8)
    i32 = ctypes.POINTER(ctypes.c_int32)
    u64 = ctypes.POINTER(ctypes.c_uint64)
    ci = ctypes.c_int
    lib.bconv_naive_int.argtypes = [i8, i8, i32, ci, ci, ci, ci, ci, ci, ci, ci]
    lib.pack_weights.argtypes = [i8, u64, ci, ci, ci, ci]
    lib.bconv_xnor.argtypes = [i8, u64, i32, ci, ci, ci, ci, ci, ci, ci, ci]
    return lib


def p_i8(a):
    a = np.ascontiguousarray(a, dtype=np.int8)
    return a, a.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))


def p_i32(a):
    a = np.ascontiguousarray(a, dtype=np.int32)
    return a, a.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def p_u64(a):
    a = np.ascontiguousarray(a, dtype=np.uint64)
    return a, a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))


def run_naive(lib, a, wsign, pad=1, stride=1):
    Cin, H, W = a.shape
    Cout, _, kh, kw = wsign.shape
    Hout = (H + 2 * pad - kh) // stride + 1
    Wout = (W + 2 * pad - kw) // stride + 1
    a_c, a_p = p_i8(a)
    w_c, w_p = p_i8(wsign)
    P, P_p = p_i32(np.zeros((Cout, Hout, Wout)))
    lib.bconv_naive_int(a_p, w_p, P_p, Cin, H, W, Cout, kh, kw, pad, stride)
    return P


def run_xnor(lib, a, wsign, pad=1, stride=1):
    Cin, H, W = a.shape
    Cout, _, kh, kw = wsign.shape
    K = Cin * kh * kw
    nwords = (K + 63) // 64
    Hout = (H + 2 * pad - kh) // stride + 1
    Wout = (W + 2 * pad - kw) // stride + 1
    w_c, w_p = p_i8(wsign)
    packed, packed_p = p_u64(np.zeros(Cout * nwords))
    lib.pack_weights(w_p, packed_p, Cout, Cin, kh, kw)
    a_c, a_p = p_i8(a)
    P, P_p = p_i32(np.zeros((Cout, Hout, Wout)))
    lib.bconv_xnor(a_p, packed_p, P_p, Cin, H, W, Cout, kh, kw, pad, stride)
    return P


def numpy_reference(a, wsign, pad=1, stride=1):
    """Ground-truth P via zero-padded +-1 convolution, vectorised enough."""
    Cin, H, W = a.shape
    Cout, _, kh, kw = wsign.shape
    ap = np.zeros((Cin, H + 2 * pad, W + 2 * pad), dtype=np.int32)
    ap[:, pad:pad + H, pad:pad + W] = a
    Hout = (H + 2 * pad - kh) // stride + 1
    Wout = (W + 2 * pad - kw) // stride + 1
    P = np.zeros((Cout, Hout, Wout), dtype=np.int32)
    for oy in range(Hout):
        for ox in range(Wout):
            patch = ap[:, oy * stride:oy * stride + kh, ox * stride:ox * stride + kw]
            # (Cout,Cin,kh,kw) * (Cin,kh,kw) summed
            P[:, oy, ox] = np.tensordot(wsign.astype(np.int32), patch,
                                        axes=([1, 2, 3], [0, 1, 2]))
    return P


def self_test(lib):
    rng = np.random.default_rng(0)
    cases = [
        (8, 8, 8, 4, 3, 3, 1, 1),
        (16, 16, 16, 32, 3, 3, 1, 1),
        (32, 7, 5, 16, 3, 3, 1, 1),   # odd spatial, exercises borders
        (3, 6, 6, 5, 3, 3, 0, 1),     # no padding
    ]
    all_ok = True
    for Cin, H, W, Cout, kh, kw, pad, stride in cases:
        a = rng.choice([-1, 1], size=(Cin, H, W)).astype(np.int8)
        wsign = rng.choice([-1, 1], size=(Cout, Cin, kh, kw)).astype(np.int8)
        ref = numpy_reference(a, wsign, pad, stride)
        Pn = run_naive(lib, a, wsign, pad, stride)
        Px = run_xnor(lib, a, wsign, pad, stride)
        dn = int(np.abs(Pn - ref).max())
        dx = int(np.abs(Px - ref).max())
        ok = (dn == 0 and dx == 0)
        all_ok &= ok
        print(f"case Cin={Cin} H={H} W={W} Cout={Cout} k={kh}x{kw} pad={pad} "
              f"stride={stride}: naive_maxdiff={dn} xnor_maxdiff={dx} "
              f"{'OK' if ok else '** FAIL **'}")
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

    # capture per binary-conv module: input activation and output
    id2name = {id(m): n for n, m in model.named_modules()}
    caught = []

    def mk(m):
        def hook(mod, inp, out):
            caught.append((id2name[id(mod)], mod,
                           inp[0].detach().cpu().numpy(),
                           out.detach().cpu().numpy()))
        return hook

    handles = [m.register_forward_hook(mk(m)) for m in model.modules()
               if type(m).__name__ == "HardBinaryConv"]
    model.eval()
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()

    print(f"{'conv':<20} {'in-domain':<11} {'zeros':>6} {'naiveΔ':>7} {'xnorΔ':>7}  result")
    n_pure = n_pass = 0
    for name, mod, xin, xout in caught:
        a = xin[0]  # [Cin,H,W]
        uniq = np.unique(np.round(a).astype(np.int64))
        is_bin = set(uniq.tolist()).issubset({-1, 0, 1})
        if not (is_bin and set(uniq.tolist()).issubset({-1, 1})):
            print(f"{name:<20} {'real/mixed':<11} {'-':>6} {'-':>7} {'-':>7}  skipped (not pure binary)")
            continue
        n_pure += 1
        zeros = int((np.round(a) == 0).sum())
        ai = np.round(a).astype(np.int8)
        w = mod.weight.detach().cpu().numpy()
        wsign = np.sign(w).astype(np.int8)
        alpha = np.abs(w).mean(axis=(1, 2, 3))
        pad = mod.padding if isinstance(mod.padding, int) else mod.padding[0]
        stride = mod.stride if isinstance(mod.stride, int) else mod.stride[0]
        P_ref = np.round(xout[0] / alpha[:, None, None]).astype(np.int32)
        Pn = run_naive(lib, ai, wsign, pad, stride)
        Px = run_xnor(lib, ai, wsign, pad, stride)
        dn = int(np.abs(Pn - P_ref).max())
        dx = int(np.abs(Px - P_ref).max())
        ok = (dn == 0 and dx == 0 and zeros == 0)
        n_pass += ok
        print(f"{name:<20} {'binary_pm1':<11} {zeros:>6} {dn:>7} {dx:>7}  "
              f"{'PASS' if ok else '** FAIL **'}")
    print(f"\npure-binary convs: {n_pure}   PASS: {n_pass}/{n_pure}")


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

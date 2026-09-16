"""
realin_conv_harness.py - verify conv_realin_naive (real input, binary
sign(w) weights, alpha-scaled) against the model's own output, for every
conv whose input is NOT +-1: enc1.conv1 and dec{4,3,2,1}.conv1.

These are architecturally identical: HardBinaryConv fed a real-valued
activation. Confirmed exactly (float-precision) against HardBinaryConv's own
forward on synthetic real input before this harness was written. This
harness checks it holds on the TRAINED model's actual weights and actual
forward-pass activations at these five convs.

  --self-test  : builds the kernel, checks it against a standalone
                 HardBinaryConv on synthetic real input (no model needed).
  real run     : loads the model, captures each of the five convs' true
                 input/output via hooks, runs the C kernel on the captured
                 input+weights, compares to the model's captured output.

  python3 realin_conv_harness.py \
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
LIB = os.path.join(CSRC, "libbnnconv_realin.so")
SRC = os.path.join(CSRC, "bnn_conv_realin.c")

TARGET_CONVS = {"enc1.conv1", "dec4.conv1", "dec3.conv1", "dec2.conv1", "dec1.conv1"}


def build():
    subprocess.run(["gcc", "-O3", "-march=native", "-shared", "-fPIC",
                    "-o", LIB, SRC], check=True)
    lib = ctypes.CDLL(LIB)
    f32 = ctypes.POINTER(ctypes.c_float)
    i8 = ctypes.POINTER(ctypes.c_int8)
    ci = ctypes.c_int
    lib.conv_realin_naive.argtypes = [f32, i8, f32, f32, ci, ci, ci, ci, ci, ci, ci, ci]
    return lib


def p_f32(a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    return a, a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def p_i8(a):
    a = np.ascontiguousarray(a, dtype=np.int8)
    return a, a.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))


def run_realin(lib, x, wsign, alpha, pad=1, stride=1):
    Cin, H, W = x.shape
    Cout, _, kh, kw = wsign.shape
    Hout = (H + 2 * pad - kh) // stride + 1
    Wout = (W + 2 * pad - kw) // stride + 1
    x_c, x_p = p_f32(x)
    w_c, w_p = p_i8(wsign)
    a_c, a_p = p_f32(alpha)
    out, out_p = p_f32(np.zeros((Cout, Hout, Wout)))
    lib.conv_realin_naive(x_p, w_p, a_p, out_p, Cin, H, W, Cout, kh, kw, pad, stride)
    return out


def self_test(lib):
    import torch
    from bireal_blocks import HardBinaryConv
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    all_ok = True
    cases = [(5, 6, 6, 4, 3, 3, 1), (3, 9, 7, 6, 3, 3, 1), (2, 5, 5, 3, 3, 3, 0)]
    for Cin, H, W, Cout, kh, kw, pad in cases:
        conv = HardBinaryConv(Cin, Cout, kh, 1, pad)
        x_t = torch.randn(1, Cin, H, W)
        with torch.no_grad():
            y_t = conv(x_t)
        w = conv.weight.detach().numpy()
        alpha = np.abs(w).mean(axis=(1, 2, 3)).astype(np.float32)
        wsign = np.sign(w).astype(np.int8)
        x = x_t.numpy()[0]
        out = run_realin(lib, x, wsign, alpha, pad=pad, stride=1)
        d = float(np.abs(out - y_t.numpy()[0]).max())
        ok = d < 1e-4
        all_ok &= ok
        print(f"case Cin={Cin} H={H} W={W} Cout={Cout} k={kh}x{kw} pad={pad}: "
              f"maxdiff={d:.3e} {'OK' if ok else '** FAIL **'}")
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
    caught = {}

    def mk(m):
        def hook(mod, inp, out):
            name = id2name[id(mod)]
            if name in TARGET_CONVS:
                caught[name] = (mod, inp[0].detach().cpu().numpy(),
                                out.detach().cpu().numpy())
        return hook

    handles = [m.register_forward_hook(mk(m)) for m in model.modules()
               if type(m).__name__ == "HardBinaryConv"]
    model.eval()
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()

    missing = TARGET_CONVS - set(caught.keys())
    if missing:
        print(f"WARNING: expected convs not found/triggered: {missing}")

    print(f"{'conv':<14} {'in-domain':<12} {'in-shape':<18} {'maxdiff':>10} "
          f"{'rel(maxout)':>12}  result")
    n_pass = 0
    n_real = 0
    for name in sorted(caught, key=lambda n: (n != "enc1.conv1", n)):
        mod, xin, xout = caught[name]
        x0 = xin[0]
        # classify the captured input: strictly +-1 (XNOR-able elsewhere) or
        # genuinely real (this kernel is the ONLY one that handles it)
        uniq = np.unique(np.round(x0).astype(np.int64))
        is_pm1 = set(uniq.tolist()).issubset({-1, 1}) and \
            float(np.abs(x0 - np.round(x0)).max()) < 1e-4
        domain = "binary_pm1" if is_pm1 else "real/mixed"
        if not is_pm1:
            n_real += 1
        w = mod.weight.detach().cpu().numpy()
        alpha = np.abs(w).mean(axis=(1, 2, 3)).astype(np.float32)
        wsign = np.sign(w).astype(np.int8)
        pad = mod.padding if isinstance(mod.padding, int) else mod.padding[0]
        stride = mod.stride if isinstance(mod.stride, int) else mod.stride[0]
        out = run_realin(lib, x0.astype(np.float32), wsign, alpha, pad=pad, stride=stride)
        ref = xout[0]
        d = float(np.abs(out - ref).max())
        rel = d / (float(np.abs(ref).max()) + 1e-12)
        ok = rel < 1e-4
        n_pass += ok
        print(f"{name:<14} {domain:<12} {str(x0.shape):<18} {d:>10.3e} {rel:>12.3e}  "
              f"{'PASS' if ok else '** FAIL **'}")
    print(f"\nreal-input convs checked: {len(caught)}   PASS: {n_pass}/{len(caught)}")
    print(f"genuinely real-input (need this kernel): {n_real}   "
          f"binary-input (XNOR is the right path, this kernel also passes): "
          f"{len(caught) - n_real}")


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

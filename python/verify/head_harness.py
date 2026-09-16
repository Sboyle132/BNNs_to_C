"""
head_harness.py - verify head_1x1 (FP32 1x1 conv + bias) against the model's
outc layer, the network's final output head.

  --self-test  : checks the kernel against nn.Conv2d(1x1, bias=True) on
                 synthetic +-1 input (no model needed).
  real run     : loads the model, captures outc's true input/output via a
                 hook, runs the C kernel on the captured input + outc's
                 weight/bias, compares to the model's captured logits.

  python3 head_harness.py \
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
LIB = os.path.join(CSRC, "libbnnhead.so")
SRC = os.path.join(CSRC, "bnn_head.c")


def build():
    subprocess.run(["gcc", "-O3", "-march=native", "-shared", "-fPIC",
                    "-o", LIB, SRC], check=True)
    lib = ctypes.CDLL(LIB)
    f32 = ctypes.POINTER(ctypes.c_float)
    ci = ctypes.c_int
    lib.head_1x1.argtypes = [f32, f32, f32, f32, ci, ci, ci, ci]
    return lib


def p_f32(a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    return a, a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def run_head(lib, a, W, bias):
    Cin, H, Wd = a.shape
    Cout = W.shape[0]
    a_c, a_p = p_f32(a)
    w_c, w_p = p_f32(W)
    b_c, b_p = p_f32(bias)
    out, out_p = p_f32(np.zeros((Cout, H, Wd)))
    lib.head_1x1(a_p, w_p, b_p, out_p, Cin, H, Wd, Cout)
    return out


def self_test(lib):
    import torch
    import torch.nn as nn
    torch.manual_seed(0)
    all_ok = True
    for Cin, H, Wd, Cout in [(16, 8, 8, 3), (32, 5, 7, 4), (8, 16, 16, 2)]:
        outc = nn.Conv2d(Cin, Cout, 1)
        x = torch.sign(torch.randn(1, Cin, H, Wd))
        x[x == 0] = 1
        with torch.no_grad():
            y = outc(x)
        W = outc.weight.detach().numpy()[:, :, 0, 0]
        bias = outc.bias.detach().numpy()
        out = run_head(lib, x.numpy()[0], W, bias)
        d = float(np.abs(out - y.numpy()[0]).max())
        ok = d < 1e-4
        all_ok &= ok
        print(f"case Cin={Cin} H={H} W={Wd} Cout={Cout}: maxdiff={d:.3e} "
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

    id2name = {id(m): n for n, m in model.named_modules()}
    caught = {}

    def mk(m):
        def hook(mod, inp, out):
            caught[id2name[id(mod)]] = (mod, inp[0].detach().cpu().numpy(),
                                        out.detach().cpu().numpy())
        return hook

    # outc is the only nn.Conv2d with kernel_size 1x1 in this net
    handles = [m.register_forward_hook(mk(m)) for n, m in model.named_modules()
               if isinstance(m, torch.nn.Conv2d) and m.kernel_size == (1, 1)]
    model.eval()
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()

    if not caught:
        print("WARNING: no 1x1 Conv2d head found")
        return
    print(f"{'layer':<10} {'in-domain':<12} {'in-shape':<18} {'maxdiff':>10} "
          f"{'rel(maxout)':>12}  result")
    for name, (mod, xin, xout) in caught.items():
        a = xin[0]
        uniq = np.unique(np.round(a).astype(np.int64))
        is_pm1 = set(uniq.tolist()).issubset({-1, 1}) and \
            float(np.abs(a - np.round(a)).max()) < 1e-4
        domain = "binary_pm1" if is_pm1 else "real/mixed"
        W = mod.weight.detach().cpu().numpy()[:, :, 0, 0]
        bias = (mod.bias.detach().cpu().numpy() if mod.bias is not None
                else np.zeros(W.shape[0], dtype=np.float32))
        out = run_head(lib, a.astype(np.float32), W, bias)
        ref = xout[0]
        d = float(np.abs(out - ref).max())
        rel = d / (float(np.abs(ref).max()) + 1e-12)
        ok = rel < 1e-4
        print(f"{name:<10} {domain:<12} {str(a.shape):<18} {d:>10.3e} "
              f"{rel:>12.3e}  {'PASS' if ok else '** FAIL **'}")


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

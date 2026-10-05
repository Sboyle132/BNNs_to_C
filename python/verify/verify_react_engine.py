"""verify_react_engine.py - checks the SHIPPED react engine (blob v3 +
bnn_infer_react binary) against the independent numpy oracle, on synthetic
random weights/data. No torch, no checkpoint, seconds to run. Run this after
ANY change to csrc/bnn_model_react.c, bnn_react_ops.c or blob_react.py.

Place in python/verify/ (next to verify_react.py, which it imports).
Needs, both built from csrc/:
    make -f Makefile.react lib      -> csrc/libreact.so      (kernel A/B)
    make -f Makefile.react          -> csrc/bnn_infer_react  (the engine)
Override locations with BNN_REACT_LIB / BNN_REACT_ENGINE if needed.

  cd python/verify && python3 verify_react_engine.py

What it does:
  1. react_proj_1x1 vs head_1x1 (the validated oracle) on 4 shapes.
  2. For 3 sizes (64x64, plus two ODD sizes that exercise floor-pool and the
     padded up_cat): random model -> v3 blob -> engine binary -> logits,
     compared with the numpy forward. PASS = argmax agreement >= 99.5%.
This verifies the engine/blob/kernels. It does NOT test your checkpoint; that
is infer_capture.py --react (C vs torch on real captures).
"""
import os, subprocess, sys, tempfile
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "bnn2c"))
sys.path.insert(0, _here)
import verify_react as VR
import blob_react as BR

ENGINE = os.environ.get("BNN_REACT_ENGINE",
                        os.path.abspath(os.path.join(_here, "..", "..", "csrc", "bnn_infer_react")))

# --- odd-dim oracle rules (PyTorch semantics) patched into the shared oracle ---
def maxpool_np(x):
    C, H, W = x.shape; h, w = H // 2, W // 2
    return x[:, :h*2, :w*2].reshape(C, h, 2, w, 2).max(axis=(2, 4))
def upcat_np(skip, x):
    up = np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)
    sC, sH, sW = skip.shape; uC, uH, uW = up.shape
    pad = np.zeros((uC, sH, sW), np.float32)
    oh, ow = max(sH - uH, 0) // 2, max(sW - uW, 0) // 2
    cH, cW = min(uH, sH), min(uW, sW)
    pad[:, oh:oh+cH, ow:ow+cW] = up[:, :cH, :cW]
    return np.concatenate([skip, pad], 0).astype(np.float32)
VR.maxpool_np = maxpool_np; VR.upcat_np = upcat_np

def proj_ab():
    rng = np.random.default_rng(3)
    VR.LIB.react_proj_1x1.argtypes = [VR.f32p, VR.f32p, VR.f32p] + [VR.ci]*4
    for (Cin, Cout, H, W) in [(120, 16, 64, 64), (48, 16, 33, 47), (7, 5, 1, 1), (24, 16, 20, 300)]:
        x = rng.standard_normal((Cin, H, W), np.float32); Wm = rng.standard_normal((Cout, Cin), np.float32)
        _, xp = VR.F(x); _, wp = VR.F(Wm)
        o = np.zeros((Cout, H, W), np.float32); _, op = VR.F(o)
        VR.LIB.react_proj_1x1(xp, wp, op, Cin, H, W, Cout)
        got = np.ctypeslib.as_array(op, (Cout, H, W)).copy()
        ref = VR.c_head(x, Wm, np.zeros(Cout, np.float32))
        print(f"  proj {Cin:>3}->{Cout:<2} {H}x{W:<4} max|d| vs head_1x1 {np.abs(got-ref).max():.2e}")

def engine_vs_oracle(H, W, seed, td):
    np.random.seed(seed)
    raw = VR.make_raw(); fold = VR.extract_react_params(raw)
    blob, inp, out = (os.path.join(td, n) for n in ("t.blob", "in.bin", "out.bin"))
    BR.write_blob_react_from_fold(fold, blob)
    x0 = np.random.randn(120, H, W).astype(np.float32); x0.tofile(inp)
    r = subprocess.run([ENGINE, blob, inp, out, str(H), str(W), "1"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    lc = np.fromfile(out, np.float32).reshape(3, H, W)
    lo, _ = VR.numpy_forward(raw, x0)
    d = float(np.abs(lc - lo).max()); rel = d / float(np.abs(lo).max())
    ag = float((lc.argmax(0) == lo.argmax(0)).mean())
    print(f"  {H}x{W:<3} logits max|d| {d:.3e} rel {rel:.2e}  argmax agree {ag*100:.3f}%")
    return ag

if __name__ == "__main__":
    if not os.path.exists(ENGINE):
        sys.exit(f"engine not found: {ENGINE}\n  build it: make -f Makefile.react (in csrc/)")
    print("== react_proj_1x1 vs head_1x1 =="); proj_ab()
    print("== standalone engine vs numpy oracle ==")
    with tempfile.TemporaryDirectory() as td:
        a = [engine_vs_oracle(64, 64, 11, td), engine_vs_oracle(44, 60, 12, td),
             engine_vs_oracle(50, 38, 13, td)]
    print("GATE:", "PASS" if min(a) >= 0.995 else "FAIL")
    sys.exit(0 if min(a) >= 0.995 else 1)

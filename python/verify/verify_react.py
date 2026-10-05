"""
verify_react.py - step-1 gate for the ReActNet (v3) port.

Proves the react dataflow + fold derivation + new C kernels on synthetic data,
with no torch and no checkpoint. Strategy:

  raw random params  --extract_react_params-->  folded params
         |                                            |
   numpy_forward (oracle, model-faithful)      run_chain_react (real C kernels)
         |                                            |
         +------------------  compare  ----------------+

numpy_forward does it the MODEL way (alpha*sign(w) conv on the tap, BN, +res,
RPReLU). run_chain_react does it the ENGINE way (bconv integer P, A*P+B fold
with alpha folded into A, real add, RPReLU). Agreement validates the
derivation, the kernels, and the chaining at once. Taps are expected
bit-exact bar float-boundary sign flips; logits within float tol.

Also A/B's each NEW kernel (epilogue_int/real, tap, maxpool_real) standalone
against a numpy oracle before the chain, so a chain failure localizes.
"""
import ctypes
import os
import numpy as np

np.random.seed(0)
EPS = 1e-4  # BatchNorm2d(eps=1e-4) in the model
_here = os.path.dirname(os.path.abspath(__file__))
_default_lib = os.path.abspath(os.path.join(_here, "..", "..", "csrc", "libreact.so"))
LIB = ctypes.CDLL(os.environ.get("BNN_REACT_LIB", _default_lib))   # build: make -f Makefile.react lib

f32p = ctypes.POINTER(ctypes.c_float)
i8p = ctypes.POINTER(ctypes.c_int8)
i32p = ctypes.POINTER(ctypes.c_int32)
u64p = ctypes.POINTER(ctypes.c_uint64)
ci = ctypes.c_int

LIB.bconv_dispatch.argtypes = [i8p, u64p, i32p, ci, ci, ci, ci, ci, ci, ci, ci]
LIB.conv_realin_naive.argtypes = [f32p, i8p, f32p, f32p, ci, ci, ci, ci, ci, ci, ci, ci]
LIB.head_1x1.argtypes = [f32p, f32p, f32p, f32p, ci, ci, ci, ci]
LIB.react_epilogue_int.argtypes = [i32p, f32p, f32p, ci, ci, ci, f32p, f32p, f32p, f32p]
LIB.react_epilogue_real.argtypes = [f32p, f32p, f32p, ci, ci, ci, f32p, f32p, f32p, f32p]
LIB.react_tap.argtypes = [f32p, i8p, ci, ci, ci, f32p]
LIB.maxpool_real.argtypes = [f32p, f32p, ci, ci, ci, ci, ci]


# ---------- ptr helpers ----------
def F(a): a = np.ascontiguousarray(a, np.float32); return a, a.ctypes.data_as(f32p)
def I8(a): a = np.ascontiguousarray(a, np.int8); return a, a.ctypes.data_as(i8p)
def I32(a): a = np.ascontiguousarray(a, np.int32); return a, a.ctypes.data_as(i32p)
def U64(a): a = np.ascontiguousarray(a, np.uint64); return a, a.ctypes.data_as(u64p)


# ---------- weight packing (matches blob.pack_weights_np / bnn_conv pack_weights) ----------
def pack_weights_np(wsign):
    Cout, Cin, kh, kw = wsign.shape
    K = Cin * kh * kw
    nwords = (K + 63) // 64
    packed = np.zeros((Cout, nwords), np.uint64)
    for co in range(Cout):
        flat = wsign[co].reshape(-1)           # C-order == kidx = (ci*kh+ky)*kw+kx
        for k in np.nonzero(flat > 0)[0]:
            packed[co, k >> 6] |= np.uint64(1) << np.uint64(int(k) & 63)
    return packed, nwords


# ---------- C-op wrappers ----------
def c_bconv(a_pm1, wsign):
    Cout, Cin, kh, kw = wsign.shape
    _, H, W = a_pm1.shape
    packed, nwords = pack_weights_np(wsign)
    _, ap = I8(a_pm1); _, pp = U64(packed)
    P = np.zeros((Cout, H, W), np.int32); _, Pp = I32(P)
    LIB.bconv_dispatch(ap, pp, Pp, Cin, H, W, Cout, kh, kw, 1, 1)
    return np.ctypeslib.as_array(Pp, (Cout, H, W)).copy()

def c_realin(x_real, wsign, alpha):
    Cout, Cin, kh, kw = wsign.shape
    _, H, W = x_real.shape
    _, xp = F(x_real); _, wp = I8(wsign); _, alp = F(alpha)
    out = np.zeros((Cout, H, W), np.float32); _, op = F(out)
    LIB.conv_realin_naive(xp, wp, alp, op, Cin, H, W, Cout, kh, kw, 1, 1)
    return np.ctypeslib.as_array(op, (Cout, H, W)).copy()

def c_head(a_real, Wm, bias):
    Cout, Cin = Wm.shape
    _, H, W = a_real.shape
    _, ap = F(a_real); _, wp = F(Wm); _, bp = F(bias)
    out = np.zeros((Cout, H, W), np.float32); _, op = F(out)
    LIB.head_1x1(ap, wp, bp, op, Cin, H, W, Cout)
    return np.ctypeslib.as_array(op, (Cout, H, W)).copy()

def c_epi_int(P, res, A, B, m1, pw):
    C, H, W = P.shape
    _, Pp = I32(P); _, rp = F(res)
    out = np.zeros((C, H, W), np.float32); _, op = F(out)
    _, Ap = F(A); _, Bp = F(B); _, mp = F(m1); _, wp = F(pw)
    LIB.react_epilogue_int(Pp, rp, op, C, H, W, Ap, Bp, mp, wp)
    return np.ctypeslib.as_array(op, (C, H, W)).copy()

def c_epi_real(cin, res, A, B, m1, pw):
    C, H, W = cin.shape
    _, cp = F(cin); _, rp = F(res)
    out = np.zeros((C, H, W), np.float32); _, op = F(out)
    _, Ap = F(A); _, Bp = F(B); _, mp = F(m1); _, wp = F(pw)
    LIB.react_epilogue_real(cp, rp, op, C, H, W, Ap, Bp, mp, wp)
    return np.ctypeslib.as_array(op, (C, H, W)).copy()

def c_tap(x, rsign):
    C, H, W = x.shape
    _, xp = F(x); _, rp = F(rsign)
    out = np.zeros((C, H, W), np.int8); _, op = I8(out)
    LIB.react_tap(xp, op, C, H, W, rp)
    return np.ctypeslib.as_array(op, (C, H, W)).copy()

def c_maxpool_real(x):
    C, H, W = x.shape
    Ho, Wo = (H - 2) // 2 + 1, (W - 2) // 2 + 1
    _, xp = F(x)
    out = np.zeros((C, Ho, Wo), np.float32); _, op = F(out)
    LIB.maxpool_real(xp, op, C, H, W, 2, 2)
    return np.ctypeslib.as_array(op, (C, Ho, Wo)).copy()


# ---------- numpy oracle pieces (model-faithful) ----------
def conv2d_np(x, w, pad, stride=1):
    """x[Cin,H,W], w[Cout,Cin,kh,kw] -> [Cout,Hout,Wout], zero-pad like F.conv2d."""
    Cout, Cin, kh, kw = w.shape
    _, H, W = x.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    Ho = (H + 2 * pad - kh) // stride + 1
    Wo = (W + 2 * pad - kw) // stride + 1
    out = np.zeros((Cout, Ho, Wo), np.float32)
    for ky in range(kh):
        for kx in range(kw):
            sub = xp[:, ky:ky + stride * Ho:stride, kx:kx + stride * Wo:stride]
            out += np.einsum('oc,chw->ohw', w[:, :, ky, kx].astype(np.float32),
                             sub.astype(np.float32), optimize=True)
    return out.astype(np.float32)

def sign_pm1(w):  # sign with >=0 -> +1 (XNOR / snapped convention)
    return np.where(w >= 0, 1.0, -1.0).astype(np.float32)

def alpha_of(w):  # HardBinaryConv per-out-channel scale
    return np.abs(w).mean(axis=(1, 2, 3)).astype(np.float32)

def hbconv_np(x, w):  # model HardBinaryConv forward: conv2d(x, alpha*sign(w))
    a = alpha_of(w)
    wf = a[:, None, None, None] * sign_pm1(w)
    return conv2d_np(x, wf, pad=(w.shape[-1] // 2))

def bn_np(y, g, b, mean, var):
    s = (g / np.sqrt(var + EPS)).astype(np.float32)
    return (s[:, None, None] * (y - mean[:, None, None]) + b[:, None, None]).astype(np.float32)

def rprelu_np(h, move1, pw):
    t = h + move1[:, None, None]
    return np.where(t >= 0, t, pw[:, None, None] * t).astype(np.float32)

def tap_np(x, rsign):
    return np.where(x + rsign[:, None, None] > 0, 1, -1).astype(np.float32)

def proj_np(x, Wm):  # real 1x1 conv, Wm[out,in]
    return np.einsum('oc,chw->ohw', Wm, x.astype(np.float32), optimize=True).astype(np.float32)

def maxpool_np(x):   # 2x2 stride2, even dims here
    C, H, W = x.shape
    return x.reshape(C, H // 2, 2, W // 2, 2).max(axis=(2, 4))

def upcat_np(skip, x):  # nearest 2x then cat [skip, up] (exact-2x here, no pad)
    up = np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)
    return np.concatenate([skip, up], axis=0).astype(np.float32)


# ---------- random model (raw params) ----------
C0 = 16
# (name, in, out, kind) kind: 'down' | 'plain'; enc1 is input_real
ARCH = [
    ("enc1", 120, C0, "down"), ("enc2", C0, 2*C0, "down"),
    ("enc3", 2*C0, 4*C0, "down"), ("enc4", 4*C0, 8*C0, "down"),
    ("bottleneck", 8*C0, 16*C0, "plain"),
    ("dec4", 16*C0 + 8*C0, 8*C0, "plain"), ("dec3", 8*C0 + 4*C0, 4*C0, "plain"),
    ("dec2", 4*C0 + 2*C0, 2*C0, "plain"), ("dec1", 2*C0 + C0, C0, "plain"),
]

def rconv(co, cin, k=3): return (np.random.randn(co, cin, k, k) * 0.3).astype(np.float32)
def rbn(c): return (np.random.randn(c)*0.5+1).astype(np.float32), (np.random.randn(c)*0.3).astype(np.float32), \
                   (np.random.randn(c)*0.2).astype(np.float32), (np.abs(np.random.randn(c))*0.5+0.1).astype(np.float32)
def rvec(c, s=0.3): return (np.random.randn(c) * s).astype(np.float32)

def make_raw():
    raw = {}
    for name, cin, co, kind in ARCH:
        blk = {"kind": kind, "cin": cin, "co": co, "input_real": (name == "enc1")}
        blk["conv1_w"] = rconv(co, cin)
        blk["bn1"] = rbn(co)
        blk["proj1_w"] = (np.random.randn(co, cin) * 0.3).astype(np.float32)  # 1x1
        blk["rp1"] = (rvec(co), rvec(co, 0.5))
        blk["tap1_rsign"] = None if blk["input_real"] else rvec(cin)
        blk["conv2_w"] = rconv(co, co)
        blk["tap2_rsign"] = rvec(co)
        if kind == "down":
            blk["bn2_skip"] = rbn(co); blk["rp2_skip"] = (rvec(co), rvec(co, 0.5))
            blk["bn2_down"] = rbn(co); blk["rp2_down"] = (rvec(co), rvec(co, 0.5))
        else:
            blk["bn2"] = rbn(co); blk["rp2"] = (rvec(co), rvec(co, 0.5))
        raw[name] = blk
    raw["outc_w"] = (np.random.randn(3, C0) * 0.3).astype(np.float32)
    raw["outc_b"] = rvec(3)
    return raw


# ---------- derivation: raw -> folded (what build_rules_react / blob will store) ----------
def bn_affine(bn, alpha=None):
    g, b, mean, var = bn
    s = g / np.sqrt(var + EPS)
    A = (s * alpha).astype(np.float32) if alpha is not None else s.astype(np.float32)
    B = (b - s * mean).astype(np.float32)
    return A, B

def extract_react_params(raw):
    fold = {}
    for name, cin, co, kind in ARCH:
        blk = raw[name]; f = {"kind": kind, "cin": cin, "co": co,
                              "input_real": blk["input_real"]}
        w1 = blk["conv1_w"]; a1 = alpha_of(w1)
        f["conv1_wsign"] = sign_pm1(w1).astype(np.int8)
        f["conv1_alpha"] = a1
        if blk["input_real"]:
            A1, B1 = bn_affine(blk["bn1"], alpha=None)   # realin output already *alpha
        else:
            A1, B1 = bn_affine(blk["bn1"], alpha=a1)
        f["epi1"] = (A1, B1, blk["rp1"][0], blk["rp1"][1])
        f["proj1_W"] = blk["proj1_w"]
        f["tap1_rsign"] = blk["tap1_rsign"]
        w2 = blk["conv2_w"]; a2 = alpha_of(w2)
        f["conv2_wsign"] = sign_pm1(w2).astype(np.int8)
        f["conv2_alpha"] = a2
        f["tap2_rsign"] = blk["tap2_rsign"]
        if kind == "down":
            As, Bs = bn_affine(blk["bn2_skip"], alpha=a2)
            Ad, Bd = bn_affine(blk["bn2_down"], alpha=a2)
            f["epi_skip"] = (As, Bs, blk["rp2_skip"][0], blk["rp2_skip"][1])
            f["epi_down"] = (Ad, Bd, blk["rp2_down"][0], blk["rp2_down"][1])
        else:
            A2, B2 = bn_affine(blk["bn2"], alpha=a2)
            f["epi2"] = (A2, B2, blk["rp2"][0], blk["rp2"][1])
        fold[name] = f
    fold["outc_W"] = raw["outc_w"]; fold["outc_b"] = raw["outc_b"]
    return fold


# ---------- oracle forward (numpy, model-faithful) ----------
def numpy_forward(raw, x0):
    taps = {}
    def enc(name, x):
        b = raw[name]
        res1 = proj_np(x, b["proj1_w"])
        if b["input_real"]:
            c1 = hbconv_np(x, b["conv1_w"])
            h1 = rprelu_np(bn_np(c1, *b["bn1"]) + res1, *b["rp1"])
        else:
            a1 = tap_np(x, b["tap1_rsign"]); taps[name + ".tap1"] = a1
            c1 = hbconv_np(a1, b["conv1_w"])
            h1 = rprelu_np(bn_np(c1, *b["bn1"]) + res1, *b["rp1"])
        a2 = tap_np(h1, b["tap2_rsign"]); taps[name + ".tap2"] = a2
        c2 = hbconv_np(a2, b["conv2_w"])
        skip = rprelu_np(bn_np(c2, *b["bn2_skip"]) + h1, *b["rp2_skip"])
        down = rprelu_np(bn_np(maxpool_np(c2), *b["bn2_down"]) + maxpool_np(h1), *b["rp2_down"])
        return skip, down
    def plain(name, x):
        b = raw[name]
        res1 = proj_np(x, b["proj1_w"])
        a1 = tap_np(x, b["tap1_rsign"]); taps[name + ".tap1"] = a1
        c1 = hbconv_np(a1, b["conv1_w"])
        h1 = rprelu_np(bn_np(c1, *b["bn1"]) + res1, *b["rp1"])
        a2 = tap_np(h1, b["tap2_rsign"]); taps[name + ".tap2"] = a2
        c2 = hbconv_np(a2, b["conv2_w"])
        h2 = rprelu_np(bn_np(c2, *b["bn2"]) + h1, *b["rp2"])
        return h2
    s1, x = enc("enc1", x0); s2, x = enc("enc2", x); s3, x = enc("enc3", x); s4, x = enc("enc4", x)
    x = plain("bottleneck", x)
    x = plain("dec4", upcat_np(s4, x)); x = plain("dec3", upcat_np(s3, x))
    x = plain("dec2", upcat_np(s2, x)); x = plain("dec1", upcat_np(s1, x))
    logits = proj_np(x, raw["outc_w"]) + raw["outc_b"][:, None, None]
    return logits.astype(np.float32), taps


# ---------- engine forward (C kernels, folded params) ----------
def run_chain_react(fold, x0):
    taps = {}
    def enc(name, x):
        f = fold[name]
        res1 = c_head(x, f["proj1_W"], np.zeros(f["co"], np.float32))
        A1, B1, m1, pw = f["epi1"]
        if f["input_real"]:
            c1 = c_realin(x, f["conv1_wsign"], f["conv1_alpha"])
            h1 = c_epi_real(c1, res1, A1, B1, m1, pw)
        else:
            a1 = c_tap(x, f["tap1_rsign"]); taps[name + ".tap1"] = a1
            P1 = c_bconv(a1, f["conv1_wsign"])
            h1 = c_epi_int(P1, res1, A1, B1, m1, pw)
        a2 = c_tap(h1, f["tap2_rsign"]); taps[name + ".tap2"] = a2
        P2 = c_bconv(a2, f["conv2_wsign"])
        As, Bs, ms, ws = f["epi_skip"]; skip = c_epi_int(P2, h1, As, Bs, ms, ws)
        Ad, Bd, md, wd = f["epi_down"]
        P2d = maxpool_np(P2).astype(np.int32)          # == proven maxpool_P (int)
        h1d = c_maxpool_real(h1)                        # new kernel in the loop
        down = c_epi_int(P2d, h1d, Ad, Bd, md, wd)
        return skip, down
    def plain(name, x):
        f = fold[name]
        res1 = c_head(x, f["proj1_W"], np.zeros(f["co"], np.float32))
        A1, B1, m1, pw = f["epi1"]
        a1 = c_tap(x, f["tap1_rsign"]); taps[name + ".tap1"] = a1
        P1 = c_bconv(a1, f["conv1_wsign"]); h1 = c_epi_int(P1, res1, A1, B1, m1, pw)
        a2 = c_tap(h1, f["tap2_rsign"]); taps[name + ".tap2"] = a2
        P2 = c_bconv(a2, f["conv2_wsign"])
        A2, B2, m2, pw2 = f["epi2"]; h2 = c_epi_int(P2, h1, A2, B2, m2, pw2)
        return h2
    s1, x = enc("enc1", x0); s2, x = enc("enc2", x); s3, x = enc("enc3", x); s4, x = enc("enc4", x)
    x = plain("bottleneck", x)
    x = plain("dec4", upcat_np(s4, x)); x = plain("dec3", upcat_np(s3, x))
    x = plain("dec2", upcat_np(s2, x)); x = plain("dec1", upcat_np(s1, x))
    logits = c_head(x, fold["outc_W"], fold["outc_b"])
    return logits, taps


# ============================ run ============================
def kernel_ab():
    print("== per-kernel A/B (C vs numpy oracle) ==")
    rng = np.random.default_rng(1)
    # epilogue_int
    C, H, W = 8, 20, 24
    P = rng.integers(-30, 30, (C, H, W)).astype(np.int32); res = rng.standard_normal((C, H, W), np.float32)
    A = rng.standard_normal(C, np.float32); B = rng.standard_normal(C, np.float32)
    m1 = rng.standard_normal(C, np.float32); pw = rng.standard_normal(C, np.float32)
    o_c = c_epi_int(P, res, A, B, m1, pw)
    t = A[:, None, None]*P + B[:, None, None] + res + m1[:, None, None]
    o_n = np.where(t >= 0, t, pw[:, None, None]*t).astype(np.float32)
    print(f"  epilogue_int   max|d| {np.abs(o_c-o_n).max():.2e}")
    # epilogue_real
    cin = rng.standard_normal((C, H, W), np.float32)
    o_c = c_epi_real(cin, res, A, B, m1, pw)
    t = A[:, None, None]*cin + B[:, None, None] + res + m1[:, None, None]
    o_n = np.where(t >= 0, t, pw[:, None, None]*t).astype(np.float32)
    print(f"  epilogue_real  max|d| {np.abs(o_c-o_n).max():.2e}")
    # tap
    x = rng.standard_normal((C, H, W), np.float32); rs = rng.standard_normal(C, np.float32)
    print(f"  tap            mismatches {int((c_tap(x, rs) != tap_np(x, rs).astype(np.int8)).sum())}")
    # maxpool_real
    x = rng.standard_normal((C, 16, 18), np.float32)
    print(f"  maxpool_real   max|d| {np.abs(c_maxpool_real(x) - maxpool_np(x)).max():.2e}")

def full_chain():
    print("\n== full chain: C engine vs numpy oracle, synthetic [120,64,64] ==")
    raw = make_raw()
    fold = extract_react_params(raw)
    x0 = (np.random.randn(120, 64, 64) * 1.0).astype(np.float32)
    lo, to = numpy_forward(raw, x0)
    lc, tc = run_chain_react(fold, x0)
    print(f"{'tap':<18}{'shape':<16}{'bit mism':>10}  result")
    allok = True
    for k in to:
        mis = int((tc[k].astype(np.int8) != to[k].astype(np.int8)).sum())
        n = to[k].size
        ok = mis == 0
        allok &= ok
        print(f"{k:<18}{str(tuple(to[k].shape)):<16}{mis:>10}  {'OK' if ok else f'{100*mis/n:.3f}%'}")
    d = float(np.abs(lc - lo).max()); rel = d / (float(np.abs(lo).max()) + 1e-12)
    agree = float((lc.argmax(0) == lo.argmax(0)).mean())
    print("-" * 56)
    print(f"logits max|d| {d:.3e}  rel {rel:.3e}  argmax agree {agree*100:.3f}%")
    print(f"\nGATE: taps {'bit-exact' if allok else 'see above'}; "
          f"argmax {agree*100:.2f}% ({'PASS' if agree >= 0.995 else 'FAIL'})")
    return raw, fold

if __name__ == "__main__":
    kernel_ab()
    full_chain()

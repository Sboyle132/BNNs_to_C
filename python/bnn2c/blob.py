"""blob.py - single source of truth for the weights.blob binary layout.
Writer (write_blob) and reader (read_blob) live together so the format can
only change in one place. Little-endian. Mirror in csrc/bnn_model.h."""

import os
import sys
import struct

# make sibling packages importable regardless of cwd. Order matters: the
# port's own modules (python/verify) must win over anything with the same
# name left lying in the training repo, so verify goes on the path LAST-
# inserted (= highest priority), HYPSO_REPO first-inserted (= lowest).
_here = os.path.dirname(os.path.abspath(__file__))
_verify = os.path.abspath(os.path.join(_here, "..", "verify"))
_hrepo = os.environ.get("HYPSO_REPO")
for _p in (_hrepo, _verify):          # verify inserted last => searched first
    if _p and os.path.abspath(_p) not in map(os.path.abspath, sys.path):
        sys.path.insert(0, _p)

import numpy as np
import torch
import forward_driver as FD

MAGIC = b"BNNC"
VERSION = 1
UPSAMPLER = {"nearest": 0, "pixshuf": 1, "nnrefine": 2}

TAG_CONV_BIN = 1
TAG_CONV_REAL = 2
TAG_ACT = 3
TAG_HEAD = 4


def pack_weights_np(wsign):
    """Bitpack sign weights to match bnn_conv.c pack_weights EXACTLY:
    kidx=(ci*kh+ky)*kw+kx, bit=1 if w>0, word=kidx>>6, bit pos=kidx&63."""
    Cout, Cin, kh, kw = wsign.shape
    K = Cin * kh * kw
    nwords = (K + 63) // 64
    packed = np.zeros((Cout, nwords), np.uint64)
    for co in range(Cout):
        flat = wsign[co].reshape(-1)          # C-order == kidx order
        for k in np.nonzero(flat > 0)[0]:
            packed[co, k >> 6] |= np.uint64(1) << np.uint64(int(k) & 63)
    return packed, nwords


def _pad_stride(mod):
    pad = mod.padding[0] if isinstance(mod.padding, tuple) else mod.padding
    stride = mod.stride[0] if isinstance(mod.stride, tuple) else mod.stride
    return int(pad), int(stride)


def _w_conv_bin(f, mod):
    w = mod.weight.detach().cpu().numpy()
    # sign(0)=0 in the model but XNOR is 1-bit; exact-zero weights are snapped
    # to +1 at the model level before export (see _snap_zero_weights), so
    # np.sign is exact here. The >=0 form is a belt-and-suspenders guard.
    wsign = np.where(w >= 0, 1, -1).astype(np.int8)
    Cout, Cin, kh, kw = w.shape
    pad, stride = _pad_stride(mod)
    packed, nwords = pack_weights_np(wsign)
    f.write(struct.pack("<8i", TAG_CONV_BIN, Cout, Cin, kh, kw, pad, stride, nwords))
    f.write(np.ascontiguousarray(packed, np.uint64).tobytes())


def _w_conv_real(f, mod):
    w = mod.weight.detach().cpu().numpy()
    wsign = np.sign(w).astype(np.int8)
    alpha = np.abs(w).mean(axis=(1, 2, 3)).astype(np.float32)
    Cout, Cin, kh, kw = w.shape
    pad, stride = _pad_stride(mod)
    f.write(struct.pack("<7i", TAG_CONV_REAL, Cout, Cin, kh, kw, pad, stride))
    f.write(np.ascontiguousarray(wsign, np.int8).tobytes())
    f.write(np.ascontiguousarray(alpha, np.float32).tobytes())


def _w_act(f, rule):
    kind = 0 if rule[0] == "int" else 1
    if kind == 0:
        _, alpha, typ, t, lo, hi = rule
        C = len(typ)
        f.write(struct.pack("<3i", TAG_ACT, kind, C))
        for arr in (typ, t, lo, hi):
            f.write(np.ascontiguousarray(arr, np.int32).tobytes())
    else:
        _, alpha, A, B, slope, rsign = rule
        C = len(A)
        f.write(struct.pack("<3i", TAG_ACT, kind, C))
        for arr in (A, B, slope, rsign):
            f.write(np.ascontiguousarray(arr, np.float32).tobytes())


def _w_head(f, mod):
    W = mod.weight.detach().cpu().numpy()[:, :, 0, 0].astype(np.float32)  # (Cout,Cin)
    bias = (mod.bias.detach().cpu().numpy().astype(np.float32)
            if mod.bias is not None else np.zeros(W.shape[0], np.float32))
    Cout, Cin = W.shape
    f.write(struct.pack("<3i", TAG_HEAD, Cout, Cin))
    f.write(np.ascontiguousarray(W, np.float32).tobytes())
    f.write(np.ascontiguousarray(bias, np.float32).tobytes())


ENC = ["enc1", "enc2", "enc3", "enc4"]
DEC = ["dec4", "dec3", "dec2", "dec1"]


def write_blob(model, path, upsampler="nearest", n_bands=120, patch=32):
    golden = FD.capture_golden(model, torch.randn(1, n_bands, patch, patch))
    # real-accumulator chains are structural, not measured: enc1.act1 always
    # (first conv sees raw float spectra). Under bilinear the four dec*.act1
    # are also real (upsample injects real values); nearest/pixshuf/nnrefine
    # keep the decoder strictly +-1 so only enc1.act1 is real. Passing the
    # set explicitly avoids a per-input numeric heuristic misfiring.
    real_acts = {"enc1.act1"}
    if upsampler == "bilinear":
        real_acts |= {"dec4.act1", "dec3.act1", "dec2.act1", "dec1.act1"}
    rules = FD.build_rules(model, golden, force_real=real_acts)
    named = dict(model.named_modules())
    n_classes = named["outc"].weight.shape[0]
    base_channels = named["enc1.conv1"].weight.shape[0]
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<5i", VERSION, base_channels, n_bands, n_classes,
                            UPSAMPLER[upsampler]))
        for i, st in enumerate(ENC):
            if i == 0:
                _w_conv_real(f, named[f"{st}.conv1"])
            else:
                _w_conv_bin(f, named[f"{st}.conv1"])
            _w_act(f, rules[f"{st}.act1"])
            _w_conv_bin(f, named[f"{st}.conv2"])
            _w_act(f, rules[f"{st}.act2_skip"])
            _w_act(f, rules[f"{st}.act2_down"])
        _w_conv_bin(f, named["bottleneck.conv1"]); _w_act(f, rules["bottleneck.act1"])
        _w_conv_bin(f, named["bottleneck.conv2"]); _w_act(f, rules["bottleneck.act2"])
        for st in DEC:
            _w_conv_bin(f, named[f"{st}.conv1"]); _w_act(f, rules[f"{st}.act1"])
            _w_conv_bin(f, named[f"{st}.conv2"]); _w_act(f, rules[f"{st}.act2"])
        _w_head(f, named["outc"])
    return path


# ---- reader, for round-trip self-check in Python ----
def read_blob(path):
    with open(path, "rb") as f:
        assert f.read(4) == MAGIC, "bad magic"
        version, base_channels, n_bands, n_classes, ups = struct.unpack("<5i", f.read(20))
        out = {"version": version, "base_channels": base_channels,
               "n_bands": n_bands, "n_classes": n_classes, "upsampler": ups,
               "convs": [], "acts": [], "head": None}

        def rd_conv():
            tag = struct.unpack("<i", f.read(4))[0]
            if tag == TAG_CONV_BIN:
                Cout, Cin, kh, kw, pad, stride, nwords = struct.unpack("<7i", f.read(28))
                packed = np.frombuffer(f.read(Cout * nwords * 8), np.uint64).reshape(Cout, nwords).copy()
                return {"tag": "bin", "Cout": Cout, "Cin": Cin, "kh": kh, "kw": kw,
                        "pad": pad, "stride": stride, "packed": packed}
            elif tag == TAG_CONV_REAL:
                Cout, Cin, kh, kw, pad, stride = struct.unpack("<6i", f.read(24))
                wsign = np.frombuffer(f.read(Cout * Cin * kh * kw), np.int8).reshape(Cout, Cin, kh, kw).copy()
                alpha = np.frombuffer(f.read(Cout * 4), np.float32).copy()
                return {"tag": "real", "Cout": Cout, "Cin": Cin, "kh": kh, "kw": kw,
                        "pad": pad, "stride": stride, "wsign": wsign, "alpha": alpha}
            raise ValueError(f"expected conv tag, got {tag}")

        def rd_act():
            tag, kind, C = struct.unpack("<3i", f.read(12))
            assert tag == TAG_ACT
            if kind == 0:
                a = [np.frombuffer(f.read(C * 4), np.int32).copy() for _ in range(4)]
                return {"kind": "int", "C": C, "type": a[0], "t": a[1], "lo": a[2], "hi": a[3]}
            else:
                a = [np.frombuffer(f.read(C * 4), np.float32).copy() for _ in range(4)]
                return {"kind": "real", "C": C, "A": a[0], "B": a[1], "slope": a[2], "rsign": a[3]}

        for i, st in enumerate(ENC):
            out["convs"].append(rd_conv()); out["acts"].append(rd_act())
            out["convs"].append(rd_conv()); out["acts"].append(rd_act()); out["acts"].append(rd_act())
        for _ in range(2):
            out["convs"].append(rd_conv()); out["acts"].append(rd_act())
        for st in DEC:
            out["convs"].append(rd_conv()); out["acts"].append(rd_act())
            out["convs"].append(rd_conv()); out["acts"].append(rd_act())
        tag, Cout, Cin = struct.unpack("<3i", f.read(12))
        assert tag == TAG_HEAD
        W = np.frombuffer(f.read(Cout * Cin * 4), np.float32).reshape(Cout, Cin).copy()
        bias = np.frombuffer(f.read(Cout * 4), np.float32).copy()
        out["head"] = {"Cout": Cout, "Cin": Cin, "W": W, "bias": bias}
        extra = f.read()
        assert extra == b"", f"{len(extra)} trailing bytes, layout mismatch"
    return out
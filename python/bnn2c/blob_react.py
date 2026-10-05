"""blob_react.py - VERSION 3 (ReActNet real-highway) weights blob.

Additive beside blob.py's v1/v2: a distinct record stream, NOT v1+append. The
real highway means the per-unit epilogue (BN affine + real add + RPReLU) and
the binarising taps are stored explicitly; short2 shortcuts are structural
Identity and store nothing.

Layout (little-endian). Mirror in csrc/bnn_model_react.h.
  MAGIC "BNNC"
  <5i> VERSION_REACT(3), base_channels, n_bands, n_classes, upsampler
  per encoder DownBlock (enc1..enc4):
      [TAP]  tap1            (omitted for enc1: input_real)
      CONV   conv1           (REAL for enc1, else BIN)
      PROJ   short1.proj     (fp32 1x1, real weights, no bias)
      EPI    epi1            (kind=real for enc1, else int)
      TAP    tap2
      CONV   conv2           (BIN)
      EPI    epi_skip        (int; shortcut = identity highway h1)
      EPI    epi_down        (int; shortcut = maxpool(h1))
  per PlainBlock (bottleneck, dec4..dec1):
      TAP    tap1
      CONV   conv1           (BIN)
      PROJ   short1.proj     (fp32 1x1)
      EPI    epi1            (int)
      TAP    tap2
      CONV   conv2           (BIN)
      EPI    epi2            (int; shortcut = identity highway h1)
  HEAD  outc

Records:
  TAG_CONV_BIN  1  <8i> tag,Cout,Cin,kh,kw,pad,stride,nwords ; u64[Cout*nwords]
  TAG_CONV_REAL 2  <7i> tag,Cout,Cin,kh,kw,pad,stride ; i8[Cout*Cin*kh*kw] ; f32[Cout] alpha
  TAG_HEAD      4  <3i> tag,Cout,Cin ; f32[Cout*Cin] ; f32[Cout] bias
  TAG_PROJ      5  <3i> tag,Cout,Cin ; f32[Cout*Cin]                 (real 1x1, no bias)
  TAG_EPI       6  <3i> tag,kind,C ; f32[C]*4  (A,B,move1,prelu_w)   kind 0=int 1=real
  TAG_TAP       7  <2i> tag,C ; f32[C] rsign

Epilogue A/B fold (per out channel):
  BIN conv:  A = alpha*gamma/sqrt(var+eps) , B = beta - gamma*mean/sqrt(var+eps)
  REAL conv: A =       gamma/sqrt(var+eps) , B = beta - gamma*mean/sqrt(var+eps)
  (enc1.conv1 output already carries alpha via conv_realin, so A omits it.)
"""
import struct
import numpy as np

MAGIC = b"BNNC"
VERSION_REACT = 3
UPSAMPLER = {"nearest": 0, "pixshuf": 1, "nnrefine": 2}

TAG_CONV_BIN, TAG_CONV_REAL, TAG_HEAD, TAG_PROJ, TAG_EPI, TAG_TAP = 1, 2, 4, 5, 6, 7

C0_DEFAULT = 16
def arch(c):
    return [("enc1", 120, c, "down"), ("enc2", c, 2*c, "down"),
            ("enc3", 2*c, 4*c, "down"), ("enc4", 4*c, 8*c, "down"),
            ("bottleneck", 8*c, 16*c, "plain"),
            ("dec4", 16*c+8*c, 8*c, "plain"), ("dec3", 8*c+4*c, 4*c, "plain"),
            ("dec2", 4*c+2*c, 2*c, "plain"), ("dec1", 2*c+c, c, "plain")]


def pack_weights_np(wsign):
    """kidx=(ci*kh+ky)*kw+kx packing, identical to blob.pack_weights_np and
    bnn_conv.c pack_weights."""
    Cout, Cin, kh, kw = wsign.shape
    nwords = (Cin*kh*kw + 63) // 64
    packed = np.zeros((Cout, nwords), np.uint64)
    for co in range(Cout):
        flat = wsign[co].reshape(-1)
        for k in np.nonzero(flat > 0)[0]:
            packed[co, k >> 6] |= np.uint64(1) << np.uint64(int(k) & 63)
    return packed, nwords


# --------------------------- record emitters ---------------------------
def _emit_conv_bin(f, wsign, pad, stride):
    Cout, Cin, kh, kw = wsign.shape
    packed, nwords = pack_weights_np(wsign)
    f.write(struct.pack("<8i", TAG_CONV_BIN, Cout, Cin, kh, kw, pad, stride, nwords))
    f.write(np.ascontiguousarray(packed, np.uint64).tobytes())

def _emit_conv_real(f, wsign, alpha, pad, stride):
    Cout, Cin, kh, kw = wsign.shape
    f.write(struct.pack("<7i", TAG_CONV_REAL, Cout, Cin, kh, kw, pad, stride))
    f.write(np.ascontiguousarray(wsign, np.int8).tobytes())
    f.write(np.ascontiguousarray(alpha, np.float32).tobytes())

def _emit_proj(f, W):
    Cout, Cin = W.shape
    f.write(struct.pack("<3i", TAG_PROJ, Cout, Cin))
    f.write(np.ascontiguousarray(W, np.float32).tobytes())

def _emit_epi(f, kind, epi):
    A, B, m1, pw = epi
    f.write(struct.pack("<3i", TAG_EPI, kind, len(A)))
    for arr in (A, B, m1, pw):
        f.write(np.ascontiguousarray(arr, np.float32).tobytes())

def _emit_tap(f, rsign):
    f.write(struct.pack("<2i", TAG_TAP, len(rsign)))
    f.write(np.ascontiguousarray(rsign, np.float32).tobytes())

def _emit_head(f, W, bias):
    Cout, Cin = W.shape
    f.write(struct.pack("<3i", TAG_HEAD, Cout, Cin))
    f.write(np.ascontiguousarray(W, np.float32).tobytes())
    f.write(np.ascontiguousarray(bias, np.float32).tobytes())


# --------------------------- writer (from folded params) ---------------------------
def write_blob_react_from_fold(fold, path, base_channels=C0_DEFAULT,
                               n_bands=120, n_classes=3, upsampler="nearest"):
    """Serialize the folded-param structure extract_react_params produces.
    This is the format-only path (shared by the torch front-end below)."""
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<5i", VERSION_REACT, base_channels, n_bands,
                            n_classes, UPSAMPLER[upsampler]))
        for name, cin, co, kind in arch(base_channels):
            b = fold[name]
            if kind == "down":
                if not b["input_real"]:
                    _emit_tap(f, b["tap1_rsign"])
                if b["input_real"]:
                    _emit_conv_real(f, b["conv1_wsign"], b["conv1_alpha"], 1, 1)
                else:
                    _emit_conv_bin(f, b["conv1_wsign"], 1, 1)
                _emit_proj(f, b["proj1_W"])
                _emit_epi(f, 1 if b["input_real"] else 0, b["epi1"])
                _emit_tap(f, b["tap2_rsign"])
                _emit_conv_bin(f, b["conv2_wsign"], 1, 1)
                _emit_epi(f, 0, b["epi_skip"])
                _emit_epi(f, 0, b["epi_down"])
            else:
                _emit_tap(f, b["tap1_rsign"])
                _emit_conv_bin(f, b["conv1_wsign"], 1, 1)
                _emit_proj(f, b["proj1_W"])
                _emit_epi(f, 0, b["epi1"])
                _emit_tap(f, b["tap2_rsign"])
                _emit_conv_bin(f, b["conv2_wsign"], 1, 1)
                _emit_epi(f, 0, b["epi2"])
        _emit_head(f, fold["outc_W"], fold["outc_b"])
    return path


# --------------------------- writer (from a trained torch model) ---------------------------
def _bn_affine(bn, alpha=None):
    import numpy as np
    g = bn.weight.detach().cpu().numpy(); b = bn.bias.detach().cpu().numpy()
    mean = bn.running_mean.detach().cpu().numpy(); var = bn.running_var.detach().cpu().numpy()
    s = g / np.sqrt(var + bn.eps)
    A = (s * alpha) if alpha is not None else s
    return A.astype(np.float32), (b - s * mean).astype(np.float32)

def _hbc(conv):
    w = conv.weight.detach().cpu().numpy()
    wsign = np.where(w >= 0, 1, -1).astype(np.int8)
    alpha = np.abs(w).mean(axis=(1, 2, 3)).astype(np.float32)
    return wsign, alpha

def _bias_vec(mod):   # LearnableBias .bias [1,C,1,1]
    return mod.bias.detach().cpu().numpy().reshape(-1).astype(np.float32)

def _prelu(mod):      # nn.PReLU .weight [C]
    return mod.weight.detach().cpu().numpy().reshape(-1).astype(np.float32)

def extract_react_params(model, base_channels=C0_DEFAULT):
    """torch model -> folded-param structure (same structure the numpy test
    builds). Arithmetic matches _emit_* expectations; verified numerically by
    verify_react.py's chain test."""
    n = dict(model.named_modules())
    fold = {}
    for name, cin, co, kind in arch(base_channels):
        input_real = (name == "enc1")
        b = {"input_real": input_real, "co": co, "cin": cin, "kind": kind}
        w1s, a1 = _hbc(n[f"{name}.conv1"])
        b["conv1_wsign"] = w1s; b["conv1_alpha"] = a1
        A1, B1 = _bn_affine(n[f"{name}.bn1"], alpha=None if input_real else a1)
        b["epi1"] = (A1, B1, _bias_vec(n[f"{name}.rprelu1.move1"]), _prelu(n[f"{name}.rprelu1.prelu"]))
        b["proj1_W"] = n[f"{name}.short1.proj"].weight.detach().cpu().numpy()[:, :, 0, 0].astype(np.float32)
        b["tap1_rsign"] = None if input_real else _bias_vec(n[f"{name}.tap1.rsign_bias"])
        w2s, a2 = _hbc(n[f"{name}.conv2"])
        b["conv2_wsign"] = w2s; b["conv2_alpha"] = a2
        b["tap2_rsign"] = _bias_vec(n[f"{name}.tap2.rsign_bias"])
        if kind == "down":
            As, Bs = _bn_affine(n[f"{name}.bn2_skip"], alpha=a2)
            Ad, Bd = _bn_affine(n[f"{name}.bn2_down"], alpha=a2)
            b["epi_skip"] = (As, Bs, _bias_vec(n[f"{name}.rprelu2_skip.move1"]), _prelu(n[f"{name}.rprelu2_skip.prelu"]))
            b["epi_down"] = (Ad, Bd, _bias_vec(n[f"{name}.rprelu2_down.move1"]), _prelu(n[f"{name}.rprelu2_down.prelu"]))
        else:
            A2, B2 = _bn_affine(n[f"{name}.bn2"], alpha=a2)
            b["epi2"] = (A2, B2, _bias_vec(n[f"{name}.rprelu2.move1"]), _prelu(n[f"{name}.rprelu2.prelu"]))
        fold[name] = b
    fold["outc_W"] = n["outc"].weight.detach().cpu().numpy()[:, :, 0, 0].astype(np.float32)
    fold["outc_b"] = n["outc"].bias.detach().cpu().numpy().astype(np.float32)
    return fold

def write_blob_react(model, path, base_channels=C0_DEFAULT, n_bands=120,
                     n_classes=3, upsampler="nearest"):
    fold = extract_react_params(model, base_channels)
    return write_blob_react_from_fold(fold, path, base_channels, n_bands,
                                      n_classes, upsampler)


# --------------------------- reader (round-trip self-check) ---------------------------
def read_blob_react(path):
    with open(path, "rb") as f:
        assert f.read(4) == MAGIC, "bad magic"
        ver, base, nb, ncl, ups = struct.unpack("<5i", f.read(20))
        assert ver == VERSION_REACT, f"version {ver} != 3"
        out = {"version": ver, "base_channels": base, "n_bands": nb,
               "n_classes": ncl, "upsampler": ups, "blocks": {}}

        def rconv():
            tag = struct.unpack("<i", f.read(4))[0]
            if tag == TAG_CONV_BIN:
                Cout, Cin, kh, kw, pad, st, nw = struct.unpack("<7i", f.read(28))
                packed = np.frombuffer(f.read(Cout*nw*8), np.uint64).reshape(Cout, nw).copy()
                return {"tag": "bin", "Cout": Cout, "Cin": Cin, "kh": kh, "kw": kw,
                        "pad": pad, "stride": st, "packed": packed}
            if tag == TAG_CONV_REAL:
                Cout, Cin, kh, kw, pad, st = struct.unpack("<6i", f.read(24))
                wsign = np.frombuffer(f.read(Cout*Cin*kh*kw), np.int8).reshape(Cout, Cin, kh, kw).copy()
                alpha = np.frombuffer(f.read(Cout*4), np.float32).copy()
                return {"tag": "real", "Cout": Cout, "Cin": Cin, "kh": kh, "kw": kw,
                        "pad": pad, "stride": st, "wsign": wsign, "alpha": alpha}
            raise ValueError(f"expected conv tag, got {tag}")

        def rproj():
            tag, Cout, Cin = struct.unpack("<3i", f.read(12)); assert tag == TAG_PROJ
            W = np.frombuffer(f.read(Cout*Cin*4), np.float32).reshape(Cout, Cin).copy()
            return {"Cout": Cout, "Cin": Cin, "W": W}

        def repi():
            tag, kind, C = struct.unpack("<3i", f.read(12)); assert tag == TAG_EPI
            a = [np.frombuffer(f.read(C*4), np.float32).copy() for _ in range(4)]
            return {"kind": kind, "C": C, "A": a[0], "B": a[1], "move1": a[2], "prelu_w": a[3]}

        def rtap():
            tag, C = struct.unpack("<2i", f.read(8)); assert tag == TAG_TAP
            return {"C": C, "rsign": np.frombuffer(f.read(C*4), np.float32).copy()}

        for name, cin, co, kind in arch(base):
            blk = {}
            if kind == "down":
                if name != "enc1":
                    blk["tap1"] = rtap()
                blk["conv1"] = rconv()
                blk["proj1"] = rproj()
                blk["epi1"] = repi()
                blk["tap2"] = rtap()
                blk["conv2"] = rconv()
                blk["epi_skip"] = repi()
                blk["epi_down"] = repi()
            else:
                blk["tap1"] = rtap(); blk["conv1"] = rconv(); blk["proj1"] = rproj()
                blk["epi1"] = repi(); blk["tap2"] = rtap(); blk["conv2"] = rconv()
                blk["epi2"] = repi()
            out["blocks"][name] = blk
        tag, Cout, Cin = struct.unpack("<3i", f.read(12)); assert tag == TAG_HEAD
        W = np.frombuffer(f.read(Cout*Cin*4), np.float32).reshape(Cout, Cin).copy()
        bias = np.frombuffer(f.read(Cout*4), np.float32).copy()
        out["head"] = {"Cout": Cout, "Cin": Cin, "W": W, "bias": bias}
        extra = f.read()
        assert extra == b"", f"{len(extra)} trailing bytes, layout mismatch"
    return out

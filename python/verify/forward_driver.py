"""
forward_driver.py - Option B: the whole network run through the verified C
ops, chained (each stage's C output feeds the next), diffed stage-by-stage
against the model's own golden activations, ending in a logits + argmax
comparison.

This is the verification instrument, not the deployment binary: the arithmetic
all happens in the C kernels (binary conv, real-input conv, integer max-pool,
integer/real activation fold, FP32 head); Python only moves buffers between
them and diffs each stage against the model. Because the C output of stage N
is the C input of stage N+1, this catches chaining bugs (layout, fan-out,
fold wiring) that the per-op harnesses cannot, and localizes any divergence to
the first stage that breaks.

Targets the nearest-neighbour variant (decoder upsample = pixel copy, so the
decoder path is strictly +-1). enc1.act1 is the single real-accumulator fold;
every other activation is an integer-P threshold.

  --self-test : builds the real Bi-Real student locally (random weights),
                runs the full chained C forward, diffs against the model.
  real run    : same, against your trained checkpoint.

  python3 forward_driver.py \
      --model-module models.bnn_unet_cpba_a2_dualskip_bireal_student \
      --model-class  BNN_UNet_CPBA_A2_DualSkip_BiReal_Student \
      --checkpoint   runs/.../best_model.pt [--input-npy patch.npy]
"""

import argparse
import ctypes
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

# make the training framework importable regardless of cwd (needed by
# --self-test's direct `from models...` import below, and transitively by
# dump_model_spec.load_real_model for the --checkpoint path).
_hrepo = os.environ.get("HYPSO_REPO")
if _hrepo and os.path.abspath(_hrepo) not in map(os.path.abspath, sys.path):
    sys.path.insert(0, _hrepo)

import bconv_harness as BC
import realin_conv_harness as RI
import maxpool_harness as MP
import head_harness as HD
import derive_thresholds as DT

HERE = os.path.dirname(os.path.abspath(__file__))
CSRC = os.path.abspath(os.path.join(HERE, "..", "..", "csrc"))


def build_fold():
    lib_path = os.path.join(CSRC, "libbnnfold.so")
    subprocess.run(["gcc", "-O3", "-march=native", "-shared", "-fPIC",
                    "-o", lib_path, os.path.join(CSRC, "bnn_fold.c")], check=True)
    lib = ctypes.CDLL(lib_path)
    i8 = ctypes.POINTER(ctypes.c_int8)
    i32 = ctypes.POINTER(ctypes.c_int32)
    f32 = ctypes.POINTER(ctypes.c_float)
    ci = ctypes.c_int
    lib.apply_fold_int.argtypes = [i32, i8, ci, ci, ci, i32, i32, i32, i32]
    lib.apply_fold_real.argtypes = [f32, i8, ci, ci, ci, f32, f32, f32, f32]
    return lib


def _i32(a):
    a = np.ascontiguousarray(a, dtype=np.int32)
    return a, a.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def _i8(a):
    a = np.ascontiguousarray(a, dtype=np.int8)
    return a, a.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))


def _f32(a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    return a, a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


# ---- fold rule tables, derived once from the model via derive_thresholds ----
RULE_CODE = {"const": 0, "ge": 1, "lt": 2, "band_in": 3, "band_out": 4}


def build_rules(model, golden, force_real=None):
    """For every activation chain, produce either an integer-P rule set
    (type/t/lo/hi arrays) or real-fold params (A/B/slope/rsign).

    int vs real is structural, not measured: pass force_real as the set of
    activation names that are real-accumulator chains (their feeding conv has
    real input). If force_real is None, fall back to measuring whether the
    feeding conv's golden output / alpha is integer (fine for a single
    verification run, but the exporter passes the explicit set so the
    decision never depends on one input)."""
    import torch
    x = torch.randn(1, 120, 32, 32)
    trace = DT.trace_model(model, x)
    chains = DT.group_chains(trace)
    rules = {}
    for ch in chains:
        act_name = ch.binarize.name.rsplit(".", 1)[0]
        C = ch.bn.mod.num_features
        g, b, m, v, eps = DT.bn_params(ch.bn.mod)
        scale = g / np.sqrt(v + eps)
        shift = b - scale * m
        move1 = DT.bias_vec(ch.move1.mod) if ch.move1 else np.zeros(C)
        slope = DT.prelu_slopes(ch.prelu.mod) if ch.prelu else np.ones(C)
        rsign = DT.bias_vec(ch.rsign.mod) if ch.rsign else np.zeros(C)
        alpha = DT.conv_alpha(ch.conv.mod) if ch.conv else np.ones(C)
        p = {"bn_scale": scale, "bn_shift": shift, "move1": move1,
             "slope": slope, "rsign": rsign, "alpha": alpha}
        N = DT.conv_fanin_N(ch.conv.mod) if ch.conv else None
        # int vs real: structural if force_real given, else measured
        if force_real is not None:
            is_int = act_name not in force_real
        else:
            conv_out = golden["conv_out"][ch.conv.name]
            Pf = conv_out[0] / alpha[:, None, None]
            is_int = float(np.abs(Pf - np.round(Pf)).max()) < 1e-2
        if is_int:
            typ = np.zeros(C, np.int32); tt = np.zeros(C, np.int32)
            lo = np.zeros(C, np.int32); hi = np.zeros(C, np.int32)
            for k in range(C):
                r, _, _ = DT.derive_rule_for_channel(k, p, N)
                typ[k] = RULE_CODE[r["type"]]
                if r["type"] == "const":
                    tt[k] = 1 if r["value"] else -1
                elif r["type"] in ("ge", "lt"):
                    tt[k] = r["t"]
                elif r["type"] in ("band_in", "band_out"):
                    lo[k], hi[k] = r["lo"], r["hi"]
            rules[act_name] = ("int", alpha, typ, tt, lo, hi)
        else:
            A = scale.astype(np.float32)
            B = (shift + move1).astype(np.float32)
            rules[act_name] = ("real", alpha, A, B,
                               slope.astype(np.float32), rsign.astype(np.float32))
    return rules


# ---- C op wrappers used by the chain ----
def conv_xnor_P(bclib, a_pm1, wsign):
    return BC.run_xnor(bclib, a_pm1, wsign, pad=1, stride=1)


def conv_realin(rilib, x_real, wsign, alpha):
    return RI.run_realin(rilib, x_real, wsign, alpha, pad=1, stride=1)


def fold_int(foldlib, P, rule):
    _, alpha, typ, tt, lo, hi = rule
    C, H, W = P.shape
    Pc, Pp = _i32(P)
    out, outp = _i8(np.zeros((C, H, W)))
    _, typp = _i32(typ); _, ttp = _i32(tt); _, lop = _i32(lo); _, hip = _i32(hi)
    foldlib.apply_fold_int(Pp, outp, C, H, W, typp, ttp, lop, hip)
    return out


def fold_real(foldlib, x_real, rule):
    _, alpha, A, B, slope, rsign = rule
    C, H, W = x_real.shape
    xc, xp = _f32(x_real)
    out, outp = _i8(np.zeros((C, H, W)))
    _, Ap = _f32(A); _, Bp = _f32(B); _, sp = _f32(slope); _, rp = _f32(rsign)
    foldlib.apply_fold_real(xp, outp, C, H, W, Ap, Bp, sp, rp)
    return out


def apply_act(foldlib, accum, rule):
    if rule[0] == "int":
        return fold_int(foldlib, accum, rule)
    return fold_real(foldlib, accum, rule)


def nearest_up_cat(skip_pm1, x_pm1):
    """decoder upsample-cat for the nearest variant: nearest-2x then concat
    [skip, upsampled] on channel axis, all +-1 int8. Matches the model's
    _upsample_cat: F.interpolate(x, scale_factor=2, mode='nearest') then, if
    the skip is larger (odd dims), F.pad(x, [dw//2, dw-dw//2, dh//2, dh-dh//2])
    -> pad the UPSAMPLED tensor up to the skip's size (H via dh, W via dw)."""
    C, H, W = x_pm1.shape
    up = np.repeat(np.repeat(x_pm1, 2, axis=1), 2, axis=2)  # nearest 2x
    sh, sw = skip_pm1.shape[1], skip_pm1.shape[2]
    dh, dw = sh - up.shape[1], sw - up.shape[2]
    if dh or dw:
        # height axis gets dh, width axis gets dw (PyTorch F.pad order:
        # [left,right,top,bottom] == [w_before,w_after,h_before,h_after])
        up = np.pad(up, ((0, 0),
                         (dh // 2, dh - dh // 2),
                         (dw // 2, dw - dw // 2)))
    return np.concatenate([skip_pm1, up], axis=0).astype(np.int8)


def wsign_of(mod):
    # sign(0)->+1 (XNOR is 1-bit; exact-zero weights are a rare artifact). This
    # matches the exporter's convention and a snapped model.
    w = mod.weight.detach().cpu().numpy()
    return np.where(w >= 0, 1, -1).astype(np.int8)


def run_chain(model, golden, libs, x_input):
    """Run the whole net through C ops, chained. Returns dict of stage->C
    output for diffing, plus final logits."""
    bclib, rilib, mplib, hdlib, foldlib = libs
    # int-vs-real is structural, not measured: only enc1.act1 is a real-
    # accumulator chain in the nearest variant (bilinear would add the four
    # dec*.act1). The measured heuristic can misfire on saturated captures, so
    # pass the set explicitly, same as the exporter.
    rules = build_rules(model, golden, force_real={"enc1.act1"})
    G = {}

    def enc(name, x_pm1_or_real, first_layer=False):
        blk = dict(model.named_modules())[name]
        # conv1
        if first_layer:
            w = wsign_of(blk.conv1)
            alpha = np.abs(blk.conv1.weight.detach().cpu().numpy()).mean(axis=(1, 2, 3)).astype(np.float32)
            co1 = conv_realin(rilib, x_pm1_or_real.astype(np.float32), w, alpha)
            a1 = apply_act(foldlib, co1, rules[f"{name}.act1"])
        else:
            P1 = conv_xnor_P(bclib, x_pm1_or_real, wsign_of(blk.conv1))
            a1 = apply_act(foldlib, P1, rules[f"{name}.act1"])
        G[f"{name}.act1"] = a1
        # conv2 (shared)
        P2 = conv_xnor_P(bclib, a1, wsign_of(blk.conv2))
        skip = apply_act(foldlib, P2, rules[f"{name}.act2_skip"])
        Pd = MP.run_maxpool(mplib, P2)
        down = apply_act(foldlib, Pd, rules[f"{name}.act2_down"])
        G[f"{name}.act2_skip"] = skip
        G[f"{name}.act2_down"] = down
        return skip, down

    def plain(name, x_pm1):
        blk = dict(model.named_modules())[name]
        P1 = conv_xnor_P(bclib, x_pm1, wsign_of(blk.conv1))
        a1 = apply_act(foldlib, P1, rules[f"{name}.act1"])
        P2 = conv_xnor_P(bclib, a1, wsign_of(blk.conv2))
        a2 = apply_act(foldlib, P2, rules[f"{name}.act2"])
        G[f"{name}.act1"] = a1
        G[f"{name}.act2"] = a2
        return a2

    s1, x = enc("enc1", x_input[0], first_layer=True)
    s2, x = enc("enc2", x)
    s3, x = enc("enc3", x)
    s4, x = enc("enc4", x)
    x = plain("bottleneck", x)
    x = plain("dec4", nearest_up_cat(s4, x))
    x = plain("dec3", nearest_up_cat(s3, x))
    x = plain("dec2", nearest_up_cat(s2, x))
    x = plain("dec1", nearest_up_cat(s1, x))
    # head
    outc = model.outc
    W = outc.weight.detach().cpu().numpy()[:, :, 0, 0]
    bias = outc.bias.detach().cpu().numpy()
    logits = HD.run_head(hdlib, x.astype(np.float32), W, bias)
    G["logits"] = logits
    return G


def capture_golden(model, x):
    """Run the model once, capturing each conv output and each activation
    (BinaryActivation) output, plus final logits."""
    import torch
    id2name = {id(m): n for n, m in model.named_modules()}
    conv_out, act_out = {}, {}

    def mk(kind):
        def hook(mod, inp, out):
            name = id2name[id(mod)]
            (conv_out if kind == "conv" else act_out)[name] = out.detach().cpu().numpy()
        return hook

    handles = []
    for n, m in model.named_modules():
        if type(m).__name__ == "HardBinaryConv":
            handles.append(m.register_forward_hook(mk("conv")))
        if type(m).__name__ == "BinaryActivation":
            handles.append(m.register_forward_hook(mk("act")))
    model.eval()
    with torch.no_grad():
        logits = model(x).detach().cpu().numpy()
    for h in handles:
        h.remove()
    # map BinaryActivation names (…act1.binarize) to act names (…act1)
    acts = {k.rsplit(".", 1)[0]: v for k, v in act_out.items()}
    return {"conv_out": conv_out, "act": acts, "logits": logits}


def diff_report(G, golden):
    order = ["enc1.act1", "enc1.act2_skip", "enc1.act2_down",
             "enc2.act1", "enc2.act2_skip", "enc2.act2_down",
             "enc3.act1", "enc3.act2_skip", "enc3.act2_down",
             "enc4.act1", "enc4.act2_skip", "enc4.act2_down",
             "bottleneck.act1", "bottleneck.act2",
             "dec4.act1", "dec4.act2", "dec3.act1", "dec3.act2",
             "dec2.act1", "dec2.act2", "dec1.act1", "dec1.act2"]
    print(f"{'stage':<20} {'shape':<16} {'bit mismatches':>15}  result")
    acts_ok = True
    for name in order:
        c = G[name].reshape(-1)
        ref = np.sign(golden["act"][name][0]).reshape(-1).astype(np.int8)
        ref[ref == 0] = -1
        mis = int(np.sum(c != ref))
        ok = mis == 0
        acts_ok &= ok
        print(f"{name:<20} {str(G[name].shape):<16} {mis:>15}  "
              f"{'PASS' if ok else '** FAIL **'}")
    # logits
    lc = G["logits"]; lref = golden["logits"][0]
    d = float(np.abs(lc - lref).max()); rel = d / (float(np.abs(lref).max()) + 1e-12)
    seg_c = lc.argmax(0); seg_ref = lref.argmax(0)
    seg_agree = float((seg_c == seg_ref).mean())
    print("-" * 60)
    print(f"logits maxdiff {d:.3e}  rel {rel:.3e}   "
          f"argmax agreement {seg_agree*100:.2f}%")
    all_ok = acts_ok and rel < 1e-4 and seg_agree == 1.0
    print(f"\nFULL FORWARD: {'PASS' if all_ok else '** FAIL **'}  "
          f"(acts {'bit-exact' if acts_ok else 'MISMATCH'}, "
          f"logits {'within tol' if rel < 1e-4 else 'OFF'}, "
          f"argmax {seg_agree*100:.0f}%)")
    return all_ok


def load_model(args):
    if args.self_test:
        import torch
        from models.bnn_unet_cpba_a2_dualskip_bireal_student import (
            BNN_UNet_CPBA_A2_DualSkip_BiReal_Student as M)
        m = M(args.n_bands, args.n_classes)
        m.train()
        with torch.no_grad():
            for _ in range(3):
                m(torch.randn(4, args.n_bands, args.patch, args.patch))
        m.eval()
        x = torch.randn(1, args.n_bands, args.patch, args.patch)
        return m, x
    from dump_model_spec import load_real_model
    ns = SimpleNamespace(
        model_module=args.model_module, model_class=args.model_class,
        use_registry=args.use_registry, config=args.config,
        checkpoint=args.checkpoint, input_npy=args.input_npy,
        n_bands=args.n_bands, n_classes=args.n_classes,
        base_channels=args.base_channels, patch=args.patch)
    return load_real_model(ns)


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

    libs = (BC.build(), RI.build(), MP.build(), HD.build(), build_fold())
    import torch
    model, x = load_model(args)
    golden = capture_golden(model, x)
    G = run_chain(model, golden, libs, x.detach().cpu().numpy())
    diff_report(G, golden)


if __name__ == "__main__":
    main()
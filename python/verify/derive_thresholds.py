"""
derive_thresholds.py - fold each activation chain into a per-channel decision
on the integer accumulator P, and verify it reproduces the model's own sign
bits exactly.

For every binary conv the accumulator is P = 2*popcount - N (integer). The
chain that turns P into a 1-bit activation is, per output channel k, a scalar
map:

    x   = alpha_k * P                       (conv output; alpha_k = mean|w|)
    z   = (gamma_k/sqrt(var_k+eps))*(x-mean_k) + beta_k    (BatchNorm, eval)
    z  += move1_bias_k                      (RPReLU.move1)
    z   = z if z>=0 else slope_k * z        (RPReLU.prelu)
    r   = z + rsign_bias_k                  (RSign bias)
    bit = (r > 0)                           (sign)

Because everything is a per-channel scalar, the output bit only changes at the
P values where r crosses zero. This script tabulates r(P) over every
achievable P (P steps by 2 with the parity of N) and reads off the flips:

    0 flips  -> const           (channel always 0 or always 1)
    1 flip   -> single threshold ('ge': bit = P>=t, or 'lt': bit = P<t)
    2 flips  -> band            ('band_in': t_lo<=P<t_hi, or 'band_out': outside)

The 'band' cases are exactly the channels with a negative PReLU slope whose
second zero-crossing lands inside the operating range; channels with a
negative slope but only one crossing in range collapse to a single threshold,
which the tabulation discovers automatically.

Verification: reconstruct P from the captured accumulator (P = round(x/alpha)),
apply the derived rule, and compare to the model's recorded sign bits. Reports
per-chain mismatch counts; anything nonzero is a real convention bug.

Chains whose accumulator is NOT integer (the first layer, and the four decoder
conv1 fed by bilinear upsampling) are detected by measurement and reported as
'real' accumulator chains: the fold formula is still verified against the
model bits, but no integer-P threshold is emitted (the conv there is not
XNOR-popcount).

Run in your env, after dump_model_spec.py has confirmed the model loads:

  python3 derive_thresholds.py \
      --model-module models.bnn_unet_cpba_a2_dualskip_bireal_student \
      --model-class  BNN_UNet_CPBA_A2_DualSkip_BiReal_Student \
      --checkpoint   runs/bnn_unet_cpba_a2_dualskip_bireal_student_scratch/best_model.pt \
      --out-dir thr_out

Self-test the derivation/verification logic on a synthetic model from the real
blocks (no checkpoint needed):

  python3 derive_thresholds.py --self-test --out-dir thr_selftest
"""

import argparse
import json
import math
import os
import sys
from types import SimpleNamespace

# make the training framework importable regardless of cwd (see dump_model_spec.py)
_hrepo = os.environ.get("HYPSO_REPO")
if _hrepo and os.path.abspath(_hrepo) not in map(os.path.abspath, sys.path):
    sys.path.insert(0, _hrepo)

import numpy as np
import torch
import torch.nn as nn

CONV_CLASSES = ("HardBinaryConv", "BrevitasQuantConv", "QuantConv2d")
BINARIZER_CLASSES = ("BinaryActivation", "QuantIdentity")


# ----------------------------------------------------------------------------
# per-channel parameter extraction
# ----------------------------------------------------------------------------
def conv_alpha(mod):
    """Per-out-channel scale. HardBinaryConv: mean(|w|). Others: 1.0."""
    cls = type(mod).__name__
    if cls == "HardBinaryConv":
        w = mod.weight.detach()
        return w.abs().mean(dim=(1, 2, 3)).cpu().numpy()
    if cls == "BrevitasQuantConv":
        return np.ones(mod.conv.weight.shape[0], dtype=np.float64)
    return np.ones(mod.weight.shape[0], dtype=np.float64)


def conv_fanin_N(mod):
    cls = type(mod).__name__
    w = mod.conv.weight if cls == "BrevitasQuantConv" else mod.weight
    _, cin, kh, kw = w.shape
    return int(cin * kh * kw)


def bn_params(mod):
    g = mod.weight.detach().cpu().numpy()
    b = mod.bias.detach().cpu().numpy()
    m = mod.running_mean.detach().cpu().numpy()
    v = mod.running_var.detach().cpu().numpy()
    return g, b, m, v, float(mod.eps)


def bias_vec(mod):
    return mod.bias.detach().reshape(-1).cpu().numpy()


def prelu_slopes(mod):
    return mod.weight.detach().reshape(-1).cpu().numpy()


# ----------------------------------------------------------------------------
# the fold, per channel, vectorised over an array of P (or real x)
# ----------------------------------------------------------------------------
def presign_from_x(x, k, p):
    """r(x) for channel k. p is a dict of per-channel numpy arrays."""
    A = p["bn_scale"][k]
    B = p["bn_shift"][k] + p["move1"][k]
    z = A * x + B
    slope = p["slope"][k]
    z = np.where(z >= 0.0, z, slope * z)
    return z + p["rsign"][k]


def presign_from_P(P, k, p):
    return presign_from_x(p["alpha"][k] * P, k, p)


# ----------------------------------------------------------------------------
# derive the per-channel rule from the bit pattern over achievable P
# ----------------------------------------------------------------------------
def derive_rule_for_channel(k, p, N):
    Pvals = np.arange(-N, N + 1, 2, dtype=np.int64)     # achievable P, parity of N
    r = presign_from_P(Pvals.astype(np.float64), k, p)
    bits = r > 0.0
    flips = np.nonzero(bits[1:] != bits[:-1])[0]         # index i => flip between i, i+1
    n = len(flips)
    if n == 0:
        return {"type": "const", "value": bool(bits[0])}, Pvals, bits
    if n == 1:
        i = int(flips[0])
        t = int(Pvals[i + 1])                            # first P of the new segment
        if bits[i + 1]:                                  # False -> True going up
            return {"type": "ge", "t": t}, Pvals, bits   # bit = P >= t
        return {"type": "lt", "t": t}, Pvals, bits       # bit = P <  t
    if n == 2:
        i1, i2 = int(flips[0]), int(flips[1])
        t_lo = int(Pvals[i1 + 1])
        t_hi = int(Pvals[i2 + 1])
        if bits[i1 + 1]:                                 # F,T,F  -> True inside
            return {"type": "band_in", "lo": t_lo, "hi": t_hi}, Pvals, bits
        return {"type": "band_out", "lo": t_lo, "hi": t_hi}, Pvals, bits  # T,F,T
    # more than 2 crossings should be impossible for a 2-piece PReLU; keep a
    # full lookup as a safe fallback and flag it.
    return ({"type": "lookup", "P": Pvals.tolist(), "bits": [bool(x) for x in bits],
             "flag": "unexpected >2 crossings"}, Pvals, bits)


def apply_rule(P, rule):
    """P: int numpy array. Returns bool array of predicted bits."""
    t = rule["type"]
    if t == "const":
        return np.full(P.shape, rule["value"], dtype=bool)
    if t == "ge":
        return P >= rule["t"]
    if t == "lt":
        return P < rule["t"]
    if t == "band_in":
        return (P >= rule["lo"]) & (P < rule["hi"])
    if t == "band_out":
        return (P < rule["lo"]) | (P >= rule["hi"])
    if t == "lookup":
        table = {pp: bb for pp, bb in zip(rule["P"], rule["bits"])}
        return np.array([table.get(int(pp), False) for pp in P], dtype=bool)
    raise ValueError(t)


# ----------------------------------------------------------------------------
# trace the model and group activation chains
# ----------------------------------------------------------------------------
ATOMIC = set(CONV_CLASSES) | set(BINARIZER_CLASSES) | {"HardBinaryConv"}


def trace_model(model, x):
    id2name = {id(m): n for n, m in model.named_modules()}
    trace = []

    def mk(m):
        def hook(mod, inp, out):
            in0 = inp[0] if isinstance(inp, (tuple, list)) and inp else None
            trace.append(SimpleNamespace(
                name=id2name.get(id(mod), "?"), cls=type(mod).__name__, mod=mod,
                x_in=in0.detach() if torch.is_tensor(in0) else None,
                x_out=out.detach() if torch.is_tensor(out) else None))
        return hook

    skip = set()
    for m in model.modules():
        if type(m).__name__ in ATOMIC:
            for c in m.modules():
                if c is not m:
                    skip.add(id(c))
    handles = []
    for m in model.modules():
        if id(m) in skip:
            continue
        if len(list(m.children())) == 0 or type(m).__name__ in ATOMIC:
            handles.append(m.register_forward_hook(mk(m)))
    model.eval()
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    return trace


def group_chains(trace):
    """Return a list of chains. Each chain is the slice of ops from just after
    the previous binarizer up to and including a binarizer, with the feeding
    BN, the most-recent binary conv (for alpha), and the move1/prelu/rsign
    ops identified by position."""
    binarizer_idx = [i for i, o in enumerate(trace) if o.cls in BINARIZER_CLASSES]
    chains = []
    prev = -1
    for bi in binarizer_idx:
        sl = list(range(prev + 1, bi + 1))
        prev = bi
        bn = next((trace[i] for i in reversed(sl) if trace[i].cls == "BatchNorm2d"), None)
        if bn is None:
            continue  # binarizer with no preceding BN in its slice; skip
        prelu = next((trace[i] for i in reversed(sl) if trace[i].cls == "PReLU"), None)
        lbias = [trace[i] for i in sl if trace[i].cls == "LearnableBias"]
        prelu_pos = next((i for i in sl if trace[i].cls == "PReLU"), None)
        move1 = rsign = None
        for i in sl:
            if trace[i].cls != "LearnableBias":
                continue
            if prelu_pos is not None and i < prelu_pos:
                move1 = trace[i]
            else:
                rsign = trace[i]
        bn_global = trace.index(bn)
        conv = next((trace[i] for i in range(bn_global, -1, -1)
                     if trace[i].cls in CONV_CLASSES), None)
        chains.append(SimpleNamespace(
            binarize=trace[bi], bn=bn, prelu=prelu, move1=move1, rsign=rsign, conv=conv))
    return chains


# ----------------------------------------------------------------------------
# per-chain: build params, derive rules, verify against model bits
# ----------------------------------------------------------------------------
def process_chain(ch):
    C = ch.bn.mod.num_features
    g, b, m, v, eps = bn_params(ch.bn.mod)
    scale = g / np.sqrt(v + eps)
    shift = b - scale * m
    p = {
        "bn_scale": scale, "bn_shift": shift,
        "move1": bias_vec(ch.move1.mod) if ch.move1 else np.zeros(C),
        "slope": prelu_slopes(ch.prelu.mod) if ch.prelu else np.ones(C),
        "rsign": bias_vec(ch.rsign.mod) if ch.rsign else np.zeros(C),
        "alpha": conv_alpha(ch.conv.mod) if ch.conv else np.ones(C),
    }
    N = conv_fanin_N(ch.conv.mod) if ch.conv else None

    # accumulator captured at the BN input; reconstruct P per channel
    x = ch.bn.x_in.cpu().numpy()                 # [1, C, H, W]
    alpha = p["alpha"].reshape(1, -1, 1, 1)
    Pf = x / alpha
    int_dev = float(np.max(np.abs(Pf - np.round(Pf))))
    is_integer = int_dev < 1e-2
    model_bits = (ch.binarize.x_out.cpu().numpy() > 0)   # [1, C, H, W]

    result = {
        "name": ch.binarize.name.rsplit(".", 1)[0],
        "conv": ch.conv.name if ch.conv else None,
        "bn": ch.bn.name, "channels": int(C), "fan_in_N": N,
        "accumulator_integer": bool(is_integer),
        "accum_int_max_dev": int_dev,
        "has_rprelu": ch.prelu is not None, "has_rsign": ch.rsign is not None,
        "channel_rules": [], "rule_type_counts": {},
        "verify": {},
    }

    if is_integer:
        P = np.round(Pf).astype(np.int64)        # [1, C, H, W]
        total_mis = 0
        counts = {}
        for k in range(C):
            rule, _, _ = derive_rule_for_channel(k, p, N)
            result["channel_rules"].append(rule)
            counts[rule["type"]] = counts.get(rule["type"], 0) + 1
            pred = apply_rule(P[0, k].reshape(-1), rule)
            mis = int(np.sum(pred != model_bits[0, k].reshape(-1)))
            total_mis += mis
        result["rule_type_counts"] = counts
        result["verify"] = {"kind": "integer_P", "total_bit_mismatches": total_mis,
                            "pixels_per_channel": int(model_bits[0, 0].size),
                            "pass": total_mis == 0}
    else:
        # real accumulator: verify the fold formula reproduces the model bits
        # directly from x; emit per-channel affine + slope for the C port.
        total_mis = 0
        for k in range(C):
            r = presign_from_x(x[0, k].reshape(-1).astype(np.float64), k, p)
            pred = r > 0
            total_mis += int(np.sum(pred != model_bits[0, k].reshape(-1)))
        result["real_affine"] = {
            "note": "conv is not XNOR here; C computes real MAC then this fold",
            "bn_scale_range": [float(scale.min()), float(scale.max())],
            "slope_range": [float(p["slope"].min()), float(p["slope"].max())],
        }
        result["verify"] = {"kind": "real_accumulator", "total_bit_mismatches": total_mis,
                            "pixels_per_channel": int(model_bits[0, 0].size),
                            "pass": total_mis == 0}
    return result


# ----------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------
def write_report(results, path):
    L = ["=" * 78, "THRESHOLD / INTERVAL DERIVATION + VERIFICATION", "=" * 78]
    n_int = sum(1 for r in results if r["accumulator_integer"])
    n_real = len(results) - n_int
    n_pass = sum(1 for r in results if r["verify"]["pass"])
    L.append(f"chains: {len(results)}   integer-P: {n_int}   real-accum: {n_real}   "
             f"PASS: {n_pass}/{len(results)}")
    L.append("")
    for r in results:
        v = r["verify"]
        status = "PASS" if v["pass"] else "**FAIL**"
        L.append("-" * 78)
        L.append(f"{r['name']}   [{status}]  ({v['kind']})")
        L.append(f"  conv={r['conv']}  bn={r['bn']}  C={r['channels']}  N={r['fan_in_N']}")
        L.append(f"  accumulator integer: {r['accumulator_integer']}  "
                 f"(max dev {r['accum_int_max_dev']:.2e})")
        if r["rule_type_counts"]:
            L.append(f"  rule types: {r['rule_type_counts']}")
        L.append(f"  bit mismatches vs model: {v['total_bit_mismatches']} "
                 f"(of {v['pixels_per_channel']} px/channel x {r['channels']} ch)")
        # show the non-trivial channels (bands) explicitly
        bands = [(k, cr) for k, cr in enumerate(r["channel_rules"])
                 if cr["type"] in ("band_in", "band_out", "lookup")]
        for k, cr in bands[:12]:
            L.append(f"    ch{k:>3}: {cr}")
        if len(bands) > 12:
            L.append(f"    ... {len(bands) - 12} more band channels")
    txt = "\n".join(L)
    with open(path, "w") as f:
        f.write(txt)
    return txt


# ----------------------------------------------------------------------------
# loading / self-test
# ----------------------------------------------------------------------------
def load_real_model(args):
    from dump_model_spec import load_real_model as _load
    ns = SimpleNamespace(
        model_module=args.model_module, model_class=args.model_class,
        use_registry=args.use_registry, config=args.config,
        checkpoint=args.checkpoint, input_npy=args.input_npy,
        n_bands=args.n_bands, n_classes=args.n_classes,
        base_channels=args.base_channels, patch=args.patch)
    return _load(ns)


def build_selftest_model():
    from bireal_blocks import HardBinaryConv, ActBlock

    class MiniVerify(nn.Module):
        def __init__(self, n_bands=120, n_classes=3, c=8):
            super().__init__()
            self.c1 = HardBinaryConv(n_bands, c, 3, 1, 1)   # real input
            self.b1 = nn.BatchNorm2d(c, eps=1e-4)
            self.a1 = ActBlock(c, use_rsign=True, use_rprelu=True, act_impl="bireal")
            self.c2 = HardBinaryConv(c, c, 3, 1, 1)          # binary input -> integer P
            self.b2 = nn.BatchNorm2d(c, eps=1e-4)
            self.a2 = ActBlock(c, use_rsign=True, use_rprelu=True, act_impl="bireal")
            self.head = nn.Conv2d(c, n_classes, 1)

        def forward(self, x):
            x = self.a1(self.b1(self.c1(x)))
            x = self.a2(self.b2(self.c2(x)))
            return self.head(x)

    m = MiniVerify()
    m.train()
    with torch.no_grad():
        for _ in range(8):
            m(torch.randn(16, 120, 32, 32))
        # force channel 0 of a2 into a genuine band: negative slope + bias so
        # r(P) dips below zero in the middle while both extremes stay positive.
        m.a2.rprelu.prelu.weight[0] = -0.8
        m.a2.rsign_bias.bias[0, 0, 0, 0] = -0.025
        m.a2.rprelu.move1.bias[0, 0, 0, 0] = 0.0
    m.eval()
    return m, torch.randn(1, 120, 32, 32)


def main():
    ap = argparse.ArgumentParser()
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
    ap.add_argument("--out-dir", default="thr_out")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        model, x = build_selftest_model()
    else:
        model, x = load_real_model(args)

    os.makedirs(args.out_dir, exist_ok=True)
    trace = trace_model(model, x)
    chains = group_chains(trace)
    results = [process_chain(ch) for ch in chains]

    with open(os.path.join(args.out_dir, "thresholds.json"), "w") as f:
        json.dump(results, f, indent=2)
    txt = write_report(results, os.path.join(args.out_dir, "thresholds.txt"))
    print(txt)
    print(f"\nwrote {args.out_dir}/thresholds.json and thresholds.txt")


if __name__ == "__main__":
    main()
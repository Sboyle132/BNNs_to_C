"""
dump_model_spec.py - full verifiable spec of a trained BNN, for planning a
from-scratch C inference port.

What it produces, from the TRAINED model (eval mode, real checkpoint):
  1. Static parameter inventory: every module, class, param name/shape/dtype,
     param count and byte cost, split into binary vs real.
  2. Per-block extraction of the values a C port must load:
       HardBinaryConv -> per-out-channel scale alpha = mean(|w|), sign(w)
       BatchNorm2d    -> gamma, beta, running_mean, running_var, eps
       RSign bias, RPReLU (move1 bias + PReLU slopes), BinaryActivation
  3. Execution trace from a real forward pass: ordered list of every leaf op
     with input/output shapes, dtypes, and the DOMAIN of each tensor
     (binary {-1,+1}, binary {0,1}, or real). This is how block wiring, BN
     placement, skips and pooling are discovered, not assumed.
  4. Accumulator verification: for every binary conv whose input tensor is
     integer-valued (binary or int8), confirm P = conv_out / alpha is an
     integer tensor, report N = Cin*kh*kw, and the observed P range. This is
     the property the XNOR-popcount kernel depends on; if it is not clean,
     the sign/scale convention is wrong and must be fixed before any C.
  5. Monotonicity flag: report all learned PReLU slopes; any negative slope
     means BN->RPReLU->RSign->sign is NOT monotonic in the accumulator, so a
     single per-channel threshold does not suffice for that channel.

Outputs written to --out-dir:
  model_spec.json    machine-readable full spec
  model_spec.txt     human-readable report
  golden/*.npy       (optional, --save-golden) per-op input/output tensors
                     plus per-layer alpha/sign/BN exports, the reference the
                     C harness diffs against.

Run in YOUR environment (has the model class + checkpoint + deps):

  python3 dump_model_spec.py \
      --config configs/bnn_unet_cpba_a2_dualskip_bireal_student_scratch.yaml \
      --checkpoint path/to/best.ckpt \
      --out-dir spec_out --save-golden

Or self-test the introspection core with a synthetic model built from the
real blocks (no checkpoint, no dataset needed):

  python3 dump_model_spec.py --self-test --out-dir spec_selftest
"""

import argparse
import json
import os
import sys

# make the training framework importable regardless of cwd: point HYPSO_REPO
# at the training repo root (same var export.py / dump_patch.py use).
_hrepo = os.environ.get("HYPSO_REPO")
if _hrepo and os.path.abspath(_hrepo) not in map(os.path.abspath, sys.path):
    sys.path.insert(0, _hrepo)

import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# tensor domain classification
# ----------------------------------------------------------------------------
def classify_domain(t, max_sample=200000, tol=1e-4):
    """Return one of 'binary_pm1', 'binary_01', 'ternary_pm1', 'int', 'real'
    plus supporting stats, from a sample of tensor values."""
    if t is None or not torch.is_tensor(t) or t.numel() == 0:
        return {"domain": "none"}
    x = t.detach().reshape(-1).float()
    if x.numel() > max_sample:
        idx = torch.linspace(0, x.numel() - 1, max_sample).long()
        x = x[idx]
    vmin = float(x.min())
    vmax = float(x.max())
    uniq = torch.unique(x)
    n_uniq = int(uniq.numel())
    is_int = bool(torch.all(torch.abs(x - torch.round(x)) < tol))

    def subset_of(vals):
        return bool(torch.all(torch.min(torch.stack(
            [torch.abs(x - v) for v in vals]), dim=0).values < tol))

    if n_uniq <= 2 and subset_of([-1.0, 1.0]):
        dom = "binary_pm1"
    elif n_uniq <= 2 and subset_of([0.0, 1.0]):
        dom = "binary_01"
    elif n_uniq <= 3 and subset_of([-1.0, 0.0, 1.0]):
        dom = "ternary_pm1"
    elif is_int:
        dom = "int"
    else:
        dom = "real"
    return {
        "domain": dom,
        "min": vmin,
        "max": vmax,
        "n_unique_sampled": n_uniq,
        "is_integer_valued": is_int,
    }


# ----------------------------------------------------------------------------
# static parameter inventory
# ----------------------------------------------------------------------------
def dtype_bits(p):
    return {torch.float32: 32, torch.float16: 16, torch.int8: 8,
            torch.int32: 32, torch.int64: 64}.get(p.dtype, 32)


def module_param_record(m):
    params = {}
    for name, p in m.named_parameters(recurse=False):
        params[name] = {"shape": list(p.shape), "dtype": str(p.dtype),
                        "numel": int(p.numel())}
    for name, b in m.named_buffers(recurse=False):
        params[name + " (buffer)"] = {"shape": list(b.shape),
                                      "dtype": str(b.dtype),
                                      "numel": int(b.numel())}
    return params


def extract_special(m):
    """Pull the values a C port must load, by module class."""
    cls = type(m).__name__
    out = {}
    if cls == "HardBinaryConv":
        w = m.weight.detach()
        alpha = w.abs().mean(dim=(1, 2, 3))          # per out channel
        Cout, Cin, kh, kw = w.shape
        n_exact_zero = int((torch.sign(w) == 0).sum())
        out.update({
            "role": "binary_conv_weights",
            "weight_shape": [int(Cout), int(Cin), int(kh), int(kw)],
            "fan_in_N": int(Cin * kh * kw),
            "stride": m.stride, "padding": m.padding,
            "alpha_per_out_channel": {
                "min": float(alpha.min()), "max": float(alpha.max()),
                "mean": float(alpha.mean()),
            },
            "sign_zero_weight_count": n_exact_zero,
            "note": "inference weight = alpha_k * sign(w); conv out = alpha_k * P_k",
        })
    elif cls == "RealConv":
        w = m.weight.detach()
        out.update({"role": "real_conv_weights",
                    "weight_shape": list(w.shape),
                    "stride": m.stride, "padding": m.padding})
    elif cls == "BrevitasQuantConv":
        w = m.weight.detach()
        out.update({"role": "brevitas_binary_conv (scale=1.0 fixed)",
                    "weight_shape": list(w.shape)})
    elif cls in ("Conv2d",):
        out.update({"role": "real_conv (head/other)",
                    "weight_shape": list(m.weight.shape),
                    "stride": m.stride, "padding": m.padding,
                    "has_bias": m.bias is not None})
    elif cls == "BatchNorm2d":
        out.update({
            "role": "batchnorm (folds into threshold/affine)",
            "num_features": int(m.num_features), "eps": float(m.eps),
            "affine": bool(m.affine),
            "gamma_min_max": [float(m.weight.detach().min()), float(m.weight.detach().max())] if m.affine else None,
            "beta_min_max": [float(m.bias.detach().min()), float(m.bias.detach().max())] if m.affine else None,
            "running_mean_min_max": [float(m.running_mean.min()), float(m.running_mean.max())],
            "running_var_min_max": [float(m.running_var.min()), float(m.running_var.max())],
        })
    elif cls == "LearnableBias":
        b = m.bias.detach().reshape(-1)
        out.update({"role": "rsign_or_move_bias (real add before sign)",
                    "channels": int(b.numel()),
                    "bias_min_max": [float(b.min()), float(b.max())]})
    elif cls == "PReLU":
        s = m.weight.detach().reshape(-1)
        n_neg = int((s < 0).sum())
        out.update({"role": "prelu_slopes",
                    "channels": int(s.numel()),
                    "slope_min_max": [float(s.min()), float(s.max())],
                    "negative_slope_count": n_neg,
                    "monotonic_ok": n_neg == 0,
                    "note": "negative slope breaks single-threshold fold for that channel"})
    elif cls == "BinaryActivation":
        out.update({"role": "sign activation (out set {-1,+1})", "params": "none"})
    elif cls == "Identity":
        out.update({"role": "identity shortcut (0 params)"})
    return out


# ----------------------------------------------------------------------------
# execution trace via forward hooks
# ----------------------------------------------------------------------------
# Modules traced as a single op even though they have children (so Brevitas
# QuantIdentity/QuantConv2d show up as one step, not their internals).
ATOMIC = {"QuantConv2d", "QuantIdentity", "HardBinaryConv", "BrevitasQuantConv",
          "BinaryActivation", "RealConv"}


def is_leaf(m):
    return len(list(m.children())) == 0


def run_trace(model, x, save_golden_dir=None):
    id2name = {id(m): n for n, m in model.named_modules()}
    trace = []
    handles = []
    order = {"i": 0}

    def make_hook(m):
        def hook(mod, inp, out):
            step = order["i"]; order["i"] += 1
            cls = type(mod).__name__
            name = id2name.get(id(mod), "?")
            in0 = inp[0] if isinstance(inp, (tuple, list)) and len(inp) else None
            rec = {
                "step": step, "name": name, "class": cls,
                "in_shape": list(in0.shape) if torch.is_tensor(in0) else None,
                "in_dtype": str(in0.dtype) if torch.is_tensor(in0) else None,
                "in_domain": classify_domain(in0),
                "out_shape": list(out.shape) if torch.is_tensor(out) else None,
                "out_dtype": str(out.dtype) if torch.is_tensor(out) else None,
                "out_domain": classify_domain(out) if torch.is_tensor(out) else None,
            }
            # accumulator verification for binary convs
            if cls in ("HardBinaryConv", "BrevitasQuantConv", "QuantConv2d") and torch.is_tensor(out) and torch.is_tensor(in0):
                rec["accum_check"] = verify_accumulator(mod, in0, out)
            if save_golden_dir is not None and torch.is_tensor(out):
                safe = name.replace(".", "_") or "root"
                np.save(os.path.join(save_golden_dir, f"step{step:03d}_{safe}_out.npy"),
                        out.detach().cpu().numpy())
                if torch.is_tensor(in0):
                    np.save(os.path.join(save_golden_dir, f"step{step:03d}_{safe}_in.npy"),
                            in0.detach().cpu().numpy())
            trace.append(rec)
        return hook

    # collect descendants of atomic modules, to skip
    skip_ids = set()
    for m in model.modules():
        if type(m).__name__ in ATOMIC:
            for c in m.modules():
                if c is not m:
                    skip_ids.add(id(c))
    for m in model.modules():
        if id(m) in skip_ids:
            continue
        if is_leaf(m) or type(m).__name__ in ATOMIC:
            handles.append(m.register_forward_hook(make_hook(m)))
    model.eval()
    with torch.no_grad():
        y = model(x)
    for h in handles:
        h.remove()
    return trace, y


def verify_accumulator(mod, in0, out):
    """For a binary conv, recover P = out / alpha and check it is integer.
    Only meaningful when the input is integer-valued (binary or int8)."""
    dom = classify_domain(in0)
    res = {"input_domain": dom["domain"]}
    cls = type(mod).__name__
    if cls == "HardBinaryConv":
        w = mod.weight.detach()
        alpha = w.abs().mean(dim=(1, 2, 3)).reshape(1, -1, 1, 1)
    elif cls == "BrevitasQuantConv":
        w = mod.conv.weight.detach()
        alpha = torch.ones(1, out.shape[1], 1, 1)   # CommonWeightQuant scale = 1.0
    else:  # QuantConv2d or other, treat per-tensor scale 1.0
        w = mod.weight.detach()
        alpha = torch.ones(1, out.shape[1], 1, 1)
    P = out.detach() / alpha
    dev = float((P - torch.round(P)).abs().max())
    res.update({
        "alpha_divided_P_max_int_deviation": dev,
        "P_is_integer": dev < 1e-3,
        "P_min": float(P.min()), "P_max": float(P.max()),
        "fan_in_N": int(w.shape[1] * w.shape[2] * w.shape[3]),
        "meaningful": dom["domain"] in ("binary_pm1", "binary_01", "int"),
        "note": "P clean-integer only expected when input is integer-valued",
    })
    return res


# ----------------------------------------------------------------------------
# report assembly
# ----------------------------------------------------------------------------
def build_static_inventory(model):
    modules = []
    tot = {"real_params": 0, "binary_params": 0, "real_bytes": 0, "binary_bytes": 0}
    for name, m in model.named_modules():
        if not is_leaf(m) and len(list(m.parameters(recurse=False))) == 0 and len(list(m.buffers(recurse=False))) == 0:
            continue
        rec = {"name": name, "class": type(m).__name__,
               "params": module_param_record(m),
               "special": extract_special(m)}
        modules.append(rec)
        binary_cls = type(m).__name__ in ("HardBinaryConv", "BrevitasQuantConv")
        for p in m.parameters(recurse=False):
            b = p.numel() * dtype_bits(p) // 8
            if binary_cls:
                tot["binary_params"] += p.numel(); tot["binary_bytes"] += p.numel() // 8
            else:
                tot["real_params"] += p.numel(); tot["real_bytes"] += b
    return modules, tot


def write_text_report(spec, path):
    L = []
    L.append("=" * 78)
    L.append("MODEL SPEC  (trained, eval mode)")
    L.append("=" * 78)
    t = spec["totals"]
    L.append(f"real params   : {t['real_params']:>12,}   ({t['real_bytes']/1e6:.3f} MB as stored)")
    L.append(f"binary params : {t['binary_params']:>12,}   ({t['binary_bytes']/1e6:.3f} MB at 1 bit)")
    L.append(f"input spec    : shape={spec['input']['shape']} dtype={spec['input']['dtype']} domain={spec['input']['domain']['domain']}")
    L.append("")
    L.append("-" * 78)
    L.append("EXECUTION TRACE  (order, class, in->out shape, domains)")
    L.append("-" * 78)
    for r in spec["trace"]:
        acc = r.get("accum_check")
        flag = ""
        if acc and acc["meaningful"]:
            flag = "  P=int OK" if acc["P_is_integer"] else "  ** P NOT integer **"
        od = r["out_domain"]["domain"] if r["out_domain"] else "-"
        idd = r["in_domain"]["domain"] if r["in_domain"] else "-"
        L.append(f"[{r['step']:>3}] {r['class']:<18} {str(r['in_shape']):<22}->{str(r['out_shape']):<22} "
                 f"in:{idd:<11} out:{od:<11}{flag}")
        L.append(f"      {r['name']}")
    L.append("")
    L.append("-" * 78)
    L.append("CONV INPUT DOMAINS  (which convs are NOT pure XNOR-popcount)")
    L.append("-" * 78)
    for r in spec["trace"]:
        if r["class"] not in ("HardBinaryConv", "BrevitasQuantConv", "QuantConv2d", "Conv2d"):
            continue
        idd = r["in_domain"]["domain"] if r["in_domain"] else "-"
        xnor = idd in ("binary_pm1", "binary_01")
        tag = "XNOR-able" if xnor else "MIXED/REAL input -> not pure XNOR"
        L.append(f"[{r['step']:>3}] {r['name']:<42} in:{idd:<12} {tag}")
    L.append("")
    L.append("-" * 78)
    L.append("BINARY CONV ACCUMULATOR CHECKS")
    L.append("-" * 78)
    for r in spec["trace"]:
        acc = r.get("accum_check")
        if not acc:
            continue
        L.append(f"[{r['step']:>3}] {r['name']}")
        L.append(f"      input domain    : {acc['input_domain']}   (meaningful={acc['meaningful']})")
        L.append(f"      fan-in N        : {acc['fan_in_N']}")
        L.append(f"      P range         : [{acc['P_min']:.1f}, {acc['P_max']:.1f}]")
        L.append(f"      max int deviation of (out/alpha): {acc['alpha_divided_P_max_int_deviation']:.3e}  "
                 f"-> P integer: {acc['P_is_integer']}")
    L.append("")
    L.append("-" * 78)
    L.append("MONOTONICITY (PReLU slopes) - negative slope breaks single-threshold fold")
    L.append("-" * 78)
    any_prelu = False
    for m in spec["modules"]:
        sp = m["special"]
        if sp.get("role") == "prelu_slopes":
            any_prelu = True
            L.append(f"  {m['name']:<40} slopes in [{sp['slope_min_max'][0]:.3f}, "
                     f"{sp['slope_min_max'][1]:.3f}]  neg={sp['negative_slope_count']}  "
                     f"monotonic_ok={sp['monotonic_ok']}")
    if not any_prelu:
        L.append("  (no PReLU modules found)")
    L.append("")
    L.append("-" * 78)
    L.append("PER-CHANNEL VALUES A C PORT MUST LOAD  (per module)")
    L.append("-" * 78)
    for m in spec["modules"]:
        sp = m["special"]
        if not sp:
            continue
        L.append(f"  {m['name']:<40} {m['class']:<16} {sp.get('role','')}")
        for k, v in sp.items():
            if k in ("role", "note"):
                continue
            L.append(f"        {k}: {v}")
    with open(path, "w") as f:
        f.write("\n".join(L))
    return "\n".join(L)


def dump(model, x, out_dir, save_golden=False):
    os.makedirs(out_dir, exist_ok=True)
    golden_dir = None
    if save_golden:
        golden_dir = os.path.join(out_dir, "golden")
        os.makedirs(golden_dir, exist_ok=True)
    modules, totals = build_static_inventory(model)
    trace, y = run_trace(model, x, save_golden_dir=golden_dir)
    spec = {
        "input": {"shape": list(x.shape), "dtype": str(x.dtype),
                  "domain": classify_domain(x)},
        "output": {"shape": list(y.shape) if torch.is_tensor(y) else None,
                   "dtype": str(y.dtype) if torch.is_tensor(y) else None,
                   "domain": classify_domain(y) if torch.is_tensor(y) else None},
        "totals": totals,
        "modules": modules,
        "trace": trace,
    }
    with open(os.path.join(out_dir, "model_spec.json"), "w") as f:
        json.dump(spec, f, indent=2)
    txt = write_text_report(spec, os.path.join(out_dir, "model_spec.txt"))
    return spec, txt


# ----------------------------------------------------------------------------
# model loading (YOUR env) and self-test (here)
# ----------------------------------------------------------------------------
def _load_checkpoint(model, path):
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict):
        sd = sd.get("model_state_dict", sd.get("state_dict", sd))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded checkpoint '{path}'  missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  first missing:", missing[:6])
    if unexpected:
        print("  first unexpected:", unexpected[:6])


def load_real_model(args):
    """Build the model by importing its class directly (no registry signature
    assumptions), load the checkpoint, and return (model, input tensor).

    Config is optional and only used to fill hyperparameters. Run from your
    repo root so `models.` imports resolve."""
    import importlib

    mcfg = {}
    if args.config:
        import yaml
        with open(args.config) as f:
            mcfg = (yaml.safe_load(f) or {}).get("model", {})

    if args.use_registry:
        from models.registry import build_model
        name = args.model_class or mcfg.get("name")
        model = build_model(name, n_bands=args.n_bands, n_classes=args.n_classes,
                            base_channels=mcfg.get("base_channels", args.base_channels),
                            weight_bit_width=mcfg.get("weight_bit_width", 1),
                            act_bit_width=mcfg.get("act_bit_width", 1),
                            in_bit_width=mcfg.get("in_bit_width", 8))
    else:
        if not (args.model_module and args.model_class):
            raise SystemExit(
                "Need --model-module and --model-class (or --use-registry).\n"
                "Example for your student:\n"
                "  python3 dump_model_spec.py \\\n"
                "    --model-module models.bnn_unet_cpba_a2_dualskip_bireal_student \\\n"
                "    --model-class  BNN_UNet_CPBA_A2_DualSkip_BiReal_Student \\\n"
                "    --checkpoint   path/to/best.ckpt \\\n"
                "    --out-dir spec_out --save-golden")
        mod = importlib.import_module(args.model_module)
        Cls = getattr(mod, args.model_class)
        model = Cls(n_bands=args.n_bands, n_classes=args.n_classes,
                    base_channels=mcfg.get("base_channels", args.base_channels),
                    weight_bit_width=mcfg.get("weight_bit_width", 1),
                    act_bit_width=mcfg.get("act_bit_width", 1),
                    in_bit_width=mcfg.get("in_bit_width", 8))

    if args.checkpoint:
        _load_checkpoint(model, args.checkpoint)
    else:
        print("WARNING: no --checkpoint given. alpha/BN/threshold values will be "
              "from random init and are NOT meaningful. Load the trained checkpoint.")
    model.eval()

    if args.input_npy:
        arr = np.load(args.input_npy)
        if arr.ndim == 3:
            arr = arr[None]
        x = torch.from_numpy(arr).float()
        print(f"using real input sample {tuple(x.shape)} from {args.input_npy}")
    else:
        x = torch.randn(1, args.n_bands, args.patch, args.patch)
        print(f"using random input {tuple(x.shape)} (fine for structure/shapes/domains; "
              "pass --input-npy a normalised patch for a golden reference)")
    return model, x


def build_selftest_model():
    """Synthetic model from the REAL blocks in bireal_blocks.py, to verify the
    dumper logic: first conv takes real input, later conv takes binary input
    so the accumulator check is exercised on a genuine {-1,+1} tensor."""
    from bireal_blocks import HardBinaryConv, ActBlock

    class Mini(nn.Module):
        def __init__(self, n_bands=120, n_classes=3, c=16):
            super().__init__()
            self.enc_conv = HardBinaryConv(n_bands, c, 3, 1, 1)   # real input
            self.enc_bn = nn.BatchNorm2d(c)
            self.act1 = ActBlock(c, use_rsign=True, use_rprelu=True, act_impl="bireal")
            self.mid_conv = HardBinaryConv(c, c, 3, 1, 1)          # binary input
            self.mid_bn = nn.BatchNorm2d(c)
            self.act2 = ActBlock(c, use_rsign=True, use_rprelu=True, act_impl="bireal")
            self.head = nn.Conv2d(c, n_classes, 1)                 # FP32 head

        def forward(self, x):
            x = self.act1(self.enc_bn(self.enc_conv(x)))
            x = self.act2(self.mid_bn(self.mid_conv(x)))
            return self.head(x)

    m = Mini()
    # move BN running stats off their init values and set a negative PReLU
    # slope on one channel so the monotonicity flag is exercised
    m.train()
    with torch.no_grad():
        for _ in range(5):
            m(torch.randn(8, 120, 32, 32))
        m.act1.rprelu.prelu.weight[0] = -0.1
    m.eval()
    return m, torch.randn(1, 120, 32, 32)


def main():
    ap = argparse.ArgumentParser(description="Dump a verifiable spec of a trained BNN.")
    ap.add_argument("--model-module", help="e.g. models.bnn_unet_cpba_a2_dualskip_bireal_student")
    ap.add_argument("--model-class", help="e.g. BNN_UNet_CPBA_A2_DualSkip_BiReal_Student")
    ap.add_argument("--use-registry", action="store_true",
                    help="build via models.registry.build_model instead of direct import")
    ap.add_argument("--config", help="optional YAML, only read for model hyperparameters")
    ap.add_argument("--checkpoint", help="trained checkpoint (required for meaningful values)")
    ap.add_argument("--input-npy", help="optional real normalised patch (C,H,W) or (1,C,H,W)")
    ap.add_argument("--n-bands", type=int, default=120)
    ap.add_argument("--n-classes", type=int, default=3)
    ap.add_argument("--base-channels", type=int, default=16)
    ap.add_argument("--patch", type=int, default=32)
    ap.add_argument("--out-dir", default="spec_out")
    ap.add_argument("--save-golden", action="store_true",
                    help="save per-op input/output tensors to out-dir/golden for the C harness")
    ap.add_argument("--self-test", action="store_true",
                    help="run introspection on a synthetic model from bireal_blocks.py")
    args = ap.parse_args()

    if args.self_test:
        model, x = build_selftest_model()
    else:
        model, x = load_real_model(args)
    _, txt = dump(model, x, args.out_dir, save_golden=args.save_golden)
    print(txt)
    print(f"\nwrote {args.out_dir}/model_spec.json and model_spec.txt"
          + ("  and golden/*.npy" if args.save_golden else ""))


if __name__ == "__main__":
    main()
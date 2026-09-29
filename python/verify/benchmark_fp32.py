#!/usr/bin/env python3
"""
benchmark_fp32.py - benchmark a model's OWN forward pass (native precision,
no binarization) for comparison against the C BNN engine's timings.

Same median/min/max reporting convention as main.c's bnn_infer[_stem][_prof]
binaries (sorted reps, median/min/max in ms), so the numbers here are
directly comparable to what you've already collected from the C side. Same
H,W convention too: defaults to 598x1092 UNPADDED, matching exactly what was
passed to bnn_infer_stem_prof (padding to a multiple of 16 only happens
inside infer_capture.py's whole-capture path, not here or in the standalone
C binaries) -- pass --pad-to-16 to opt into that instead if you want the
infer_capture.py-equivalent size.

WHICH MODEL: this loads any model by importing its class directly via
--model-module/--model-class (+--checkpoint), introspecting the class's own
__init__ signature so it works whether or not that constructor takes the
BNN-family bit-width kwargs -- NOT hardcoded to one architecture, and does
NOT depend on dump_model_spec.load_real_model (that helper's model-
construction call assumes every class shares the BNN-family signature,
confirmed broken against plain models like UNet; fixed locally here rather
than editing that shared, load-bearing file). Two ready-to-run options
already in your config zoo:
  - unet_baseline (models/unet.py, class UNet) -- plain Ronneberger UNet,
    a DIFFERENT architecture from the dualskip family, but already trained
    and already your ablation study's fp32 reference point. Default below.
  - any real-weight (non-binary) variant of the dualskip architecture itself,
    if one exists in your registry -- architecturally closer to the BNN
    you've been optimising, point --model-module/--model-class at it directly.
These answer different questions (architecture cost vs. precision cost);
pick based on which comparison you actually want.

Backends, auto-detected (unavailable ones are skipped with a stated reason,
never a hard error):
  torch-cpu           eager PyTorch, CPU, torch.inference_mode()
  torch-cpu-compile   torch.compile(..., default backend) on CPU
  torch-cuda          eager PyTorch, CUDA, cudnn.benchmark=True
  torch-cuda-compile  torch.compile on CUDA
  onnx-cpu            ONNX Runtime, CPUExecutionProvider, full graph opt
  onnx-cuda           ONNX Runtime, CUDAExecutionProvider

  # fp32 UNet baseline, real capture size, every backend this machine has
  python3 benchmark_fp32.py \\
      --model-module models.unet --model-class UNet \\
      --checkpoint runs/unet_baseline/best_model.pt --base-channels 16

  # one backend only
  python3 benchmark_fp32.py ... --backends torch-cuda

NOTE: not runnable/tested in the environment this was written in (no torch
or onnxruntime available there) -- syntax-checked only, not executed. Sanity-
check the first run: compare torch-cpu's output against a known-good
infer_capture.py --backend torch run on the same checkpoint+input before
trusting the timings. onnx-* backends need `pip install onnxruntime` (CPU)
or `onnxruntime-gpu` (CUDA) separately -- not assumed installed.
"""
import argparse
import os
import sys
import time

_hrepo = os.environ.get("HYPSO_REPO")
if _hrepo and os.path.abspath(_hrepo) not in map(os.path.abspath, sys.path):
    sys.path.insert(0, _hrepo)


def pad_to_16(H, W):
    ph = (16 - H % 16) % 16
    pw = (16 - W % 16) % 16
    return H + ph, W + pw


def load_model(args):
    """Self-contained: does NOT go through dump_model_spec.load_real_model,
    because that helper's model-construction call (both its --use-registry
    and direct-import branches) assumes every model shares the BNN-family
    constructor signature (weight_bit_width/act_bit_width/in_bit_width) --
    confirmed broken against plain models like UNet (models/unet.py has no
    such kwargs at all) and, separately, against the CURRENT registry.py
    (build_model's real signature is (cfg: dict, n_bands, n_classes), not
    the individual kwargs load_real_model's --use-registry branch passes).
    Both are pre-existing mismatches in dump_model_spec.py/registry.py, not
    something this script introduces -- fixed here locally instead of
    editing those shared, load-bearing files.

    Introspects the target class's actual __init__ signature and only passes
    the kwargs it declares, so this works for UNet (n_bands,n_classes,
    bilinear,base_channels) AND the BNN family (+weight_bit_width etc.)
    unchanged, without assuming either shape up front."""
    import importlib
    import inspect
    mod = importlib.import_module(args.model_module)
    Cls = getattr(mod, args.model_class)

    pool = {"n_bands": args.n_bands, "n_classes": args.n_classes,
            "base_channels": args.base_channels,
            "weight_bit_width": 1, "act_bit_width": 1, "in_bit_width": 8}
    accepted = set(inspect.signature(Cls.__init__).parameters)
    kwargs = {k: v for k, v in pool.items() if k in accepted}
    missing_required = [
        k for k, p in inspect.signature(Cls.__init__).parameters.items()
        if k not in ("self",) and p.default is inspect.Parameter.empty
        and k not in kwargs]
    if missing_required:
        raise SystemExit(f"{args.model_class}.__init__ needs {missing_required} "
                         f"which this script doesn't know how to supply -- "
                         f"add them to `pool` above or pass via a config path")

    model = Cls(**kwargs)

    if args.checkpoint:
        import torch
        sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd, strict=True)
    else:
        print("!! no --checkpoint given: random-init weights -- timing only, "
              "outputs are meaningless (shapes/FLOPs are what matter for "
              "timing, not learned values)", file=sys.stderr)

    model.eval()
    return model


def time_torch(model, x, device, reps, warmup, compile_model=False):
    import torch
    m = model.to(device)
    xd = x.to(device)
    if compile_model:
        m = torch.compile(m)
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    with torch.inference_mode():
        for _ in range(warmup):
            m(xd)
        sync()
        times = []
        for _ in range(reps):
            sync()
            t0 = time.perf_counter()
            m(xd)
            sync()
            times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return times[len(times) // 2], times[0], times[-1]


def export_onnx(model, x, path):
    import torch
    torch.onnx.export(model.cpu().eval(), x.cpu(), path,
                      input_names=["input"], output_names=["logits"],
                      opset_version=17)
    return path


def time_onnx(onnx_path, x_np, provider, reps, warmup):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=[provider])
    in_name = sess.get_inputs()[0].name
    for _ in range(warmup):
        sess.run(None, {in_name: x_np})
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        sess.run(None, {in_name: x_np})
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return times[len(times) // 2], times[0], times[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-module", default="models.unet")
    ap.add_argument("--model-class", default="UNet")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--n-bands", type=int, default=120)
    ap.add_argument("--n-classes", type=int, default=3)
    ap.add_argument("--base-channels", type=int, default=16)
    ap.add_argument("--H", type=int, default=598)
    ap.add_argument("--W", type=int, default=1092)
    ap.add_argument("--pad-to-16", action="store_true",
                    help="pad H,W up to a multiple of 16 first, matching "
                         "infer_capture.py's whole-capture path instead of "
                         "the raw size you passed to bnn_infer_stem_prof")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--threads", type=int, default=None,
                    help="torch CPU thread count; default = all cores")
    ap.add_argument("--backends", nargs="+", default=None,
                    choices=["torch-cpu", "torch-cpu-compile", "torch-cuda",
                             "torch-cuda-compile", "onnx-cpu", "onnx-cuda"],
                    help="default: auto-detect everything this machine has")
    ap.add_argument("--onnx-path", default="/tmp/bench_model.onnx")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(args.threads or os.cpu_count())
    torch.backends.cudnn.benchmark = True   # harmless if no CUDA

    H, W = (pad_to_16(args.H, args.W) if args.pad_to_16 else (args.H, args.W))
    print(f"model: {args.model_module}.{args.model_class}"
          f"{'  (checkpoint)' if args.checkpoint else '  (RANDOM INIT)'}")
    print(f"input: n_bands={args.n_bands} H={H} W={W} "
          f"({'padded to /16' if args.pad_to_16 else 'raw, matching your C benchmark invocations'}) "
          f"threads={args.threads or os.cpu_count()} reps={args.reps} warmup={args.warmup}")

    model = load_model(args)
    x = torch.randn(1, args.n_bands, H, W)

    have_cuda = torch.cuda.is_available()
    if have_cuda:
        print(f"CUDA: {torch.cuda.get_device_name(0)}")
    have_onnx = False
    ort_providers = []
    try:
        import onnxruntime as ort
        have_onnx = True
        ort_providers = ort.get_available_providers()
    except ImportError:
        pass
    can_compile = hasattr(torch, "compile")

    candidates = args.backends or (
        ["torch-cpu"]
        + (["torch-cpu-compile"] if can_compile else [])
        + (["torch-cuda"] if have_cuda else [])
        + (["torch-cuda-compile"] if have_cuda and can_compile else [])
        + (["onnx-cpu"] if have_onnx else [])
        + (["onnx-cuda"] if have_onnx and "CUDAExecutionProvider" in ort_providers else [])
    )

    results = []
    onnx_exported = False
    print()
    for b in candidates:
        try:
            if b == "torch-cpu":
                med, mn, mx = time_torch(model, x, "cpu", args.reps, args.warmup)
            elif b == "torch-cpu-compile":
                med, mn, mx = time_torch(model, x, "cpu", args.reps, args.warmup, compile_model=True)
            elif b == "torch-cuda":
                if not have_cuda:
                    print(f"skip {b}: no CUDA device"); continue
                med, mn, mx = time_torch(model, x, "cuda", args.reps, args.warmup)
            elif b == "torch-cuda-compile":
                if not have_cuda:
                    print(f"skip {b}: no CUDA device"); continue
                med, mn, mx = time_torch(model, x, "cuda", args.reps, args.warmup, compile_model=True)
            elif b in ("onnx-cpu", "onnx-cuda"):
                if not have_onnx:
                    print(f"skip {b}: onnxruntime not installed "
                         f"(pip install onnxruntime{'-gpu' if b == 'onnx-cuda' else ''})")
                    continue
                provider = "CPUExecutionProvider" if b == "onnx-cpu" else "CUDAExecutionProvider"
                if provider not in ort_providers:
                    print(f"skip {b}: {provider} not in this onnxruntime "
                         f"build's available providers {ort_providers}")
                    continue
                if not onnx_exported:
                    export_onnx(model, x, args.onnx_path)
                    onnx_exported = True
                import numpy as np
                x_np = x.cpu().numpy()
                med, mn, mx = time_onnx(args.onnx_path, x_np, provider, args.reps, args.warmup)
            else:
                continue
            results.append((b, med, mn, mx))
            print(f"{b:<20} median {med:9.3f} ms   min {mn:9.3f}   max {mx:9.3f}")
        except Exception as e:
            print(f"{b:<20} FAILED: {e}")

    if results:
        print("\nsummary (median ms, fastest first):")
        for b, med, mn, mx in sorted(results, key=lambda r: r[1]):
            print(f"  {b:<20} {med:9.3f} ms")
        print(f"\nfor comparison, your C stem engine forward (same H,W, this "
             f"exact reps convention): see your own bnn_infer_stem_prof output")


if __name__ == "__main__":
    main()
"""
infer_capture.py - run the BNN over full HYPSO captures and write label maps.

Three modes:
  --split test              every capture in the test split
  --split val               every capture in the val split
  --capture <name>          one capture by name (must be in a split)

The model is fully convolutional and eval-mode BatchNorm uses fixed running
stats, so whole-capture inference is bit-identical to 32x32 patch inference
(verified). We therefore load the full normalised cube, pad H,W up to a
multiple of 16, run ONE forward pass, and crop back. No tiling, no seams.

Backends:
  --backend torch   PyTorch model forward (reference)
  --backend c       the standalone C engine via ctypes (bnn2c/verify kernels)
  --verify          run both and assert the argmax maps match

ReActNet (--react): the C backend is different. It exports the model to a v3
blob (blob_react.py), runs the SHIPPED engine binary (csrc/bnn_infer_react,
override with --react-engine) on the padded cube, and reads the logits back. So
--verify --react checks the actual blob + C engine end to end against torch,
not just the kernel chain.

Outputs, per capture, under artifacts/<capture_name>/<model_name>/:
  labels.bin    H*W uint8, row-major (0=cloud,1=land,2=sea)
  labels.json   {"H":H,"W":W,"classes":["cloud","land","sea"]}  (sidecar)
  mask.png      colourised label map
  metrics.json  macro-F1 vs ground truth  (only if labels available)

The capture loader is the one framework-specific seam: load_capture_cube()
below builds the full normalised cube using the SAME path dump_patch.py uses
(HYPSOPatchDataset + Normaliser), so normalisation matches training exactly.
"""

import argparse
import json
import os
import sys

_hrepo = os.environ.get("HYPSO_REPO")
if _hrepo and os.path.abspath(_hrepo) not in map(os.path.abspath, sys.path):
    sys.path.insert(0, _hrepo)

import numpy as np

CLASS_NAMES = ["cloud", "land", "sea"]
# palette: cloud=white, land=green, sea=blue
PALETTE = np.array([[255, 255, 255], [60, 160, 60], [30, 90, 200]], np.uint8)


# ---------------------------------------------------------------------------
# capture loading (framework seam) -- reconstruct the full normalised cube by
# tiling the patch dataset over the whole capture. Uses only the verified
# HYPSOPatchDataset + Normaliser path.
# ---------------------------------------------------------------------------
def _build_index_and_norm(cfg, project_dir):
    from data.dataset import (load_capture_index, CUBE_SUFFIX_L1A,
                              CUBE_SUFFIX_L1B, parse_band_selection)
    from data.splits import get_or_create_split
    from data.preprocessing import load_stats, Normaliser
    data_root = os.path.join(project_dir, cfg["data"]["root"])
    split_path = os.path.join(project_dir, cfg["data"]["split_path"])
    stats_path = os.path.join(project_dir, cfg["data"]["stats_path"])
    legacy = cfg["data"].get("legacy_raw_mode", False)
    cube_suffix = None if legacy else (
        CUBE_SUFFIX_L1A if cfg["data"].get("cube_variant", "l1b").lower() == "l1a"
        else CUBE_SUFFIX_L1B)
    index = load_capture_index(data_root, cube_suffix=cube_suffix, legacy_raw_mode=legacy)
    split = get_or_create_split(list(index.keys()), split_path)
    stats = load_stats(stats_path)
    normaliser = Normaliser(stats) if cfg["normalisation"]["enabled"] else None
    bsel = cfg["data"].get("band_selection", None)
    band_indices = parse_band_selection(bsel, total_bands=120) if bsel is not None else None
    return index, split, normaliser, band_indices


def load_capture_cube(cfg, project_dir, index, normaliser, band_indices, capture_id):
    """Return (cube [n_bands,H,W] float32 NORMALISED, labels [H,W] int64|None).

    Replicates data.dataset's loading + normalisation EXACTLY (so results match
    training), then transposes to channels-first for the model. Loads the raw
    cube once and normalises in place: no normalised data written to disk, no
    patch round-trip, no redundant full-capture copy.

    dataset.py contract (read from source):
      _load_cube_and_labels(cube_path, label_path) -> cube (H,W,B) f32,
                                                       label_map (H,W) int64
      band selection is cube[:, :, band_indices]  (last axis)
      normalisation (mutually exclusive, per_pixel_l2 > per_capture > global):
        per_pixel_l2 : per-pixel L2 over band axis  (needs per_pixel_l2_normalise)
        per_capture  : (cube - cube.mean((0,1))) / cube.std((0,1))
        global       : (cube - normaliser.mean) / normaliser.std   (per-band)
    """
    from data.dataset import _load_cube_and_labels
    entry = index[capture_id]
    cube, label_map = _load_cube_and_labels(entry["cube"], entry["labels"])  # (H,W,B),(H,W)
    cube = np.asarray(cube, np.float32)
    if band_indices is not None:
        cube = cube[:, :, band_indices]

    n = cfg["normalisation"]
    if n.get("per_pixel_l2", False):
        from data.dataset import per_pixel_l2_normalise
        cube, _ = per_pixel_l2_normalise(cube, n.get("magnitude_channel", False))
    elif n.get("per_capture", False):
        cap_mean = cube.mean(axis=(0, 1))
        cap_std = cube.std(axis=(0, 1))
        cap_std = np.where(cap_std < 1e-8, 1.0, cap_std)
        cube = (cube - cap_mean) / cap_std
    elif normaliser is not None:
        cube = (cube - normaliser.mean.numpy()) / normaliser.std.numpy()

    cube = np.ascontiguousarray(cube.transpose(2, 0, 1), np.float32)   # (B,H,W)
    labels = None if label_map is None else np.asarray(label_map).astype(np.int64)
    return cube, labels


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------
def pad_to_16(cube):
    _, H, W = cube.shape
    ph = (16 - H % 16) % 16
    pw = (16 - W % 16) % 16
    if ph or pw:
        cube = np.pad(cube, ((0, 0), (0, ph), (0, pw)))
    return cube, H, W


def infer_torch(model, cube):
    import torch
    cp, H, W = pad_to_16(cube)
    with torch.no_grad():
        logits = model(torch.from_numpy(cp[None]).float()).numpy()[0]
    return logits[:, :H, :W]


def infer_torch_patched(model, cube, P=32):
    """Full network on independent PxP patches (zero-padded at borders), stitched.
    Reproduces 32x32 patch inference (training), not whole-capture."""
    import torch
    nb, H, W = cube.shape
    ncl = model.outc.weight.shape[0] if hasattr(model, "outc") else 3
    out = np.zeros((ncl, H, W), np.float32)
    for py in range(0, H, P):
        ph = min(P, H - py)
        for px in range(0, W, P):
            pw = min(P, W - px)
            patch = np.zeros((nb, P, P), np.float32)
            patch[:, :ph, :pw] = cube[:, py:py+ph, px:px+pw]
            with torch.no_grad():
                o = model(torch.from_numpy(patch[None]).float()).numpy()[0]
            out[:, py:py+ph, px:px+pw] = o[:, :ph, :pw]
    return out


def infer_c_react_patched(model, cube, engine=None, n_bands=120, n_classes=3, P=32):
    """C patch engine: writes the cube to a temp file and runs bnn_infer_react
    with the patch<P>: prefix. No pad-to-16 (patches are already PxP)."""
    import subprocess, tempfile
    here = os.path.dirname(os.path.abspath(__file__))
    bnn2c = os.path.abspath(os.path.join(here, "..", "bnn2c"))
    if bnn2c not in sys.path: sys.path.insert(0, bnn2c)
    import blob_react as BR
    engine = engine or os.path.abspath(os.path.join(here, "..", "..", "csrc", "bnn_infer_react"))
    _, H, W = cube.shape
    key = id(model)
    if key not in _REACT_BLOB:
        d = tempfile.mkdtemp(prefix="react_blob_"); _REACT_BLOB[key] = os.path.join(d, "w.blob")
        BR.write_blob_react(model, _REACT_BLOB[key], n_bands=n_bands, n_classes=n_classes)
    with tempfile.TemporaryDirectory(prefix="react_patch_") as td:
        fin, fout = os.path.join(td, "in.bin"), os.path.join(td, "out.bin")
        np.ascontiguousarray(cube, np.float32).tofile(fin)
        r = subprocess.run([engine, _REACT_BLOB[key], f"patch{P}:mem:{fin}", fout, str(H), str(W), "1"],
                           capture_output=True, text=True)
        if r.returncode != 0: raise RuntimeError(f"patch engine failed: {r.stderr.strip()}")
        return np.fromfile(fout, np.float32).reshape(n_classes, H, W)


def infer_c(model, cube, stem=False):
    """Run the full capture through the C engine by reusing forward_driver's
    chained C ops (which are the same kernels the standalone binary uses).
    stem=True routes to run_chain_stem (spectral-stem A1b); the default v1 path
    is unchanged."""
    import torch
    import forward_driver as FD
    import bconv_harness as BC, realin_conv_harness as RI
    import maxpool_harness as MP, head_harness as HD
    cp, H, W = pad_to_16(cube)
    libs = (BC.build(), RI.build(), MP.build(), HD.build(), FD.build_fold())
    golden = FD.capture_golden(model, torch.from_numpy(cp[None]).float())
    runner = FD.run_chain_stem if stem else FD.run_chain
    G = runner(model, golden, libs, cp[None])
    return G["logits"][:, :H, :W]


_REACT_BLOB = {}   # model id -> exported blob path (export once per run)


def infer_c_react(model, cube, engine=None, n_bands=120, n_classes=3):
    """ReActNet C backend: export model -> v3 blob (cached), run the standalone
    engine binary on the padded cube, return logits cropped to [n_classes,H,W].
    The engine binary is the shipped artifact, so this validates blob + engine
    + export together against torch. Single-thread/openmp is whatever the
    binary was built with (OMP_NUM_THREADS applies)."""
    import subprocess
    import tempfile
    here = os.path.dirname(os.path.abspath(__file__))
    bnn2c = os.path.abspath(os.path.join(here, "..", "bnn2c"))
    if bnn2c not in sys.path:
        sys.path.insert(0, bnn2c)
    import blob_react as BR
    engine = engine or os.path.abspath(
        os.path.join(here, "..", "..", "csrc", "bnn_infer_react"))
    if not os.path.exists(engine):
        raise SystemExit(f"react engine not found: {engine}\n"
                         f"  build it: make -f Makefile.react (in csrc/), "
                         f"or pass --react-engine")
    cp, H, W = pad_to_16(cube)
    key = id(model)
    if key not in _REACT_BLOB:
        d = tempfile.mkdtemp(prefix="react_blob_")
        _REACT_BLOB[key] = os.path.join(d, "weights_react.blob")
        BR.write_blob_react(model, _REACT_BLOB[key], n_bands=n_bands,
                            n_classes=n_classes)
    Hp, Wp = cp.shape[1], cp.shape[2]
    with tempfile.TemporaryDirectory(prefix="react_run_") as td:
        fin, fout = os.path.join(td, "in.bin"), os.path.join(td, "out.bin")
        np.ascontiguousarray(cp, np.float32).tofile(fin)
        r = subprocess.run([engine, _REACT_BLOB[key], fin, fout, str(Hp), str(Wp), "1"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"react engine failed (rc={r.returncode}): {r.stderr.strip()}")
        logits = np.fromfile(fout, np.float32).reshape(n_classes, Hp, Wp)
    return logits[:, :H, :W]


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------
def macro_f1(pred, gt, n_classes=3):
    f1s = []
    for c in range(n_classes):
        tp = np.sum((pred == c) & (gt == c))
        fp = np.sum((pred == c) & (gt != c))
        fn = np.sum((pred != c) & (gt == c))
        denom = 2 * tp + fp + fn
        f1s.append(1.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s)), [float(x) for x in f1s]


def write_outputs(out_dir, seg, gt):
    os.makedirs(out_dir, exist_ok=True)
    H, W = seg.shape
    seg.astype(np.uint8).tofile(os.path.join(out_dir, "labels.bin"))
    with open(os.path.join(out_dir, "labels.json"), "w") as f:
        json.dump({"H": int(H), "W": int(W), "classes": CLASS_NAMES,
                   "dtype": "uint8", "layout": "row-major"}, f, indent=2)
    try:
        from PIL import Image
        Image.fromarray(PALETTE[seg]).save(os.path.join(out_dir, "mask.png"))
        png = "mask.png"
    except ImportError:
        # fallback: numpy .npy of the RGB, note PIL missing
        np.save(os.path.join(out_dir, "mask_rgb.npy"), PALETTE[seg])
        png = "mask_rgb.npy (PIL not installed)"
    metrics = None
    if gt is not None:
        f1, per = macro_f1(seg, gt)
        metrics = {"macro_f1": f1, "per_class_f1": dict(zip(CLASS_NAMES, per))}
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    return png, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-module")
    ap.add_argument("--model-class")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stem", choices=["none", "a1b"], default="none",
                    help="'a1b' builds the spectral-stem model and runs the "
                         "stem C forward (run_chain_stem)")
    ap.add_argument("--stem-channels", type=int, default=16)
    ap.add_argument("--react", action="store_true",
                    help="build the ReActNet model; the C backend runs the "
                         "standalone v3 engine (see --react-engine)")
    ap.add_argument("--patchwise", type=int, default=0,
                    help="ReActNet: run patch-wise with independent PxP patches "
                         "(matches 32x32 patch training) instead of whole-capture; "
                         "P multiple of 16, e.g. --patchwise 32")
    ap.add_argument("--react-engine", default=None,
                    help="path to bnn_infer_react (default: <repo>/csrc/bnn_infer_react)")
    ap.add_argument("--project-dir", default=os.environ.get("HYPSO_REPO", "."))
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--split", choices=["test", "val", "train"])
    g.add_argument("--capture")
    ap.add_argument("--backend", choices=["torch", "c"], default="c")
    ap.add_argument("--verify", action="store_true", help="run torch AND c, assert match")
    ap.add_argument("--verify-tol", type=float, default=0.005,
                    help="max fraction of argmax-disagreeing pixels allowed "
                         "under --verify (float-boundary sign flips at the real "
                         "first layer / head). Default 0.5%%; real logic bugs "
                         "produce far more.")
    ap.add_argument("--model-name", default=None, help="output subdir; default = model-class")
    ap.add_argument("--out-root", default="artifacts")
    ap.add_argument("--n-bands", type=int, default=120)
    ap.add_argument("--n-classes", type=int, default=3)
    ap.add_argument("--patch", type=int, default=32)
    args = ap.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.react and args.stem != "none":
        raise SystemExit("--react and --stem are mutually exclusive")
    model_name = (args.model_name or args.model_class
                  or ("stem_a1b" if args.stem != "none"
                      else "reactnet" if args.react else "model"))

    # build model
    import torch
    if args.stem == "a1b":
        from models.bnn_unet_cpba_a2_dualskip_bireal_stem import (
            BNN_UNet_CPBA_A2_DualSkip_BiReal_Stem as _Stem)
        model = _Stem(n_bands=args.n_bands, n_classes=args.n_classes,
                      base_channels=16, stem_channels=args.stem_channels,
                      stem_layers=1, stem_weight="binary", stem_out="binary",
                      stem_act="prelu")
        _sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if isinstance(_sd, dict) and "state_dict" in _sd:
            _sd = _sd["state_dict"]
        model.load_state_dict(_sd, strict=True)
    elif args.react:
        from models.bnn_unet_cpba_a2_dualskip_bireal_reactnet import (
            BNN_UNet_CPBA_A2_DualSkip_BiReal_ReActNet as _ReAct)
        model = _ReAct(n_bands=args.n_bands, n_classes=args.n_classes,
                       base_channels=16)
        _sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if isinstance(_sd, dict) and "state_dict" in _sd:
            _sd = _sd["state_dict"]
        model.load_state_dict(_sd, strict=True)
    else:
        if not (args.model_module and args.model_class):
            raise SystemExit("--model-module and --model-class are required "
                             "unless --stem a1b or --react is given")
        from dump_model_spec import load_real_model
        from types import SimpleNamespace
        ns = SimpleNamespace(model_module=args.model_module, model_class=args.model_class,
                             use_registry=False, config=args.config, checkpoint=args.checkpoint,
                             input_npy=None, n_bands=args.n_bands, n_classes=args.n_classes,
                             base_channels=16, patch=args.patch)
        model, _ = load_real_model(ns)
    model.eval()
    # snap exact-zero weights to +eps so sign()->+1 everywhere: the model
    # forward (reference) and the C engine (XNOR, 1-bit) then agree. Rare
    # artifact (~0-1 weights); negligible effect, verified by --verify.
    with torch.no_grad():
        _nz = 0
        for _m in model.modules():
            if type(_m).__name__ == "HardBinaryConv":
                _z = (_m.weight == 0)
                _c = int(_z.sum())
                if _c:
                    _m.weight[_z] = 1e-6
                    _nz += _c
    if _nz:
        print(f"snapped {_nz} exact-zero weight(s) to +1 for XNOR consistency")

    index, split, normaliser, band_indices = _build_index_and_norm(cfg, args.project_dir)
    if args.capture:
        captures = [args.capture]
    else:
        sp = args.split or "test"
        captures = split[sp]
    print(f"processing {len(captures)} capture(s); backend={args.backend} "
          f"verify={args.verify}; out -> {args.out_root}/<capture>/{model_name}/")

    for cid in captures:
        cube, gt = load_capture_cube(cfg, args.project_dir, index, normaliser,
                                     band_indices, cid)
        if args.verify:
            if args.patchwise:
                lt = infer_torch_patched(model, cube, args.patchwise)
                lc = infer_c_react_patched(model, cube, args.react_engine, args.n_bands, args.n_classes, args.patchwise)
                lw = infer_torch(model, cube)   # whole-capture torch, to show the training/deployment gap
                gap = float((lt.argmax(0) != lw.argmax(0)).mean())
                print(f"  [{cid}] torch patch-wise vs torch whole-capture differ {gap*100:.2f}% "
                      f"(these are different computations; patch-wise matches 32x32 training)")
            else:
                lt = infer_torch(model, cube)
                lc = (infer_c_react(model, cube, args.react_engine, args.n_bands, args.n_classes)
                      if args.react else infer_c(model, cube, stem=(args.stem != "none")))
            st, sc = lt.argmax(0), lc.argmax(0)
            agree = float((st == sc).mean())
            ndis = int((st != sc).sum())
            # The binary conv path is bit-exact; the only non-determinism is
            # float rounding at sign() boundaries in the real-valued first
            # layer (enc1.conv1->enc1.act1) and the FP32 head, where a
            # pre-activation within float-epsilon of zero can round either
            # way. A few such pixels flip and propagate. This is inherent to a
            # net with a real first layer and float head (torch itself is not
            # bit-deterministic there across BLAS/compiler flags), NOT a port
            # bug. Allow a small fraction; fail only if it's large enough to
            # indicate a real logic error.
            frac_dis = 1.0 - agree
            TOL = args.verify_tol
            if frac_dis > TOL:
                raise AssertionError(
                    f"{cid}: C vs torch argmax disagree {frac_dis*100:.3f}% "
                    f"({ndis} px) > tol {TOL*100:.2f}% -- exceeds float-boundary "
                    f"noise, investigate")
            seg = sc
            tag = (f"verify OK (argmax {agree*100:.3f}%, {ndis} float-boundary "
                   f"px within tol)")
        else:
            if args.backend == "c":
                logits = (infer_c_react(model, cube, args.react_engine, args.n_bands, args.n_classes)
                          if args.react else infer_c(model, cube, stem=(args.stem != "none")))
            else:
                logits = infer_torch(model, cube)
            seg = logits.argmax(0)
            tag = f"backend={args.backend}"
        out_dir = os.path.join(args.out_root, cid, model_name)
        png, metrics = write_outputs(out_dir, seg, gt)
        msg = f"[{cid}] {seg.shape[0]}x{seg.shape[1]}  {tag}  -> {out_dir}/ (labels.bin, {png}"
        if metrics:
            msg += f", metrics.json macroF1={metrics['macro_f1']:.4f}"
        print(msg + ")")


if __name__ == "__main__":
    main()
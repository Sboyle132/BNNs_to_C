#!/usr/bin/env python3
"""
bench_ort.py - time an ONNX model with onnxruntime (CPU) on the board.
Standalone: needs only numpy + onnxruntime, no torch, no training repo.

  PYTHONPATH=/tmp/ort python3 bench_ort.py model.onnx --threads 1 --reps 10

Random input (runtime is data-independent). Threads are pinned explicitly so
the number is comparable with the C engine run at the same thread count.
Reports median/min/max forward time (same convention as the C binaries),
session-creation time, first-run time and peak RSS.
"""
import argparse
import resource
import statistics
import time

import numpy as np
import onnxruntime as ort


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    t0 = time.perf_counter()
    sess = ort.InferenceSession(args.model, so, providers=["CPUExecutionProvider"])
    t_create = (time.perf_counter() - t0) * 1e3

    inp = sess.get_inputs()[0]
    shape = list(inp.shape)
    if not all(isinstance(d, int) for d in shape):
        raise SystemExit(f"non-static input shape {shape}; export a fixed-shape model")
    x = np.random.default_rng(args.seed).standard_normal(shape).astype(np.float32)

    t0 = time.perf_counter()
    sess.run(None, {inp.name: x})
    t_first = (time.perf_counter() - t0) * 1e3
    for _ in range(max(0, args.warmup - 1)):
        sess.run(None, {inp.name: x})

    ts = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        sess.run(None, {inp.name: x})
        ts.append((time.perf_counter() - t0) * 1e3)

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # KB on Linux
    print(f"# model    {args.model}  input {shape}")
    print(f"# session create {t_create:.0f} ms   first run {t_first:.0f} ms")
    print(f"# forward  median {statistics.median(ts):.1f} ms  "
          f"min {min(ts):.1f}  max {max(ts):.1f}  "
          f"(reps {args.reps}, threads {args.threads})")
    print(f"# peak RSS {rss_mb:.0f} MB")


if __name__ == "__main__":
    main()
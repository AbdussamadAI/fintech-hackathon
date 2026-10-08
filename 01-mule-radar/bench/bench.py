"""Throughput / latency benchmark: same algorithm, same batches, GPU (CUDA kernels) vs CPU (Numba, all
allocated cores). Methodology: warm-up excluded, contiguous slice of the stream with a full look-back
window, median of repeats, end-to-end per batch (window CSR rebuild + all feature kernels).

Usage: python bench/bench.py --name HI-Small
Writes data/processed/<name>/bench.json
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import cupy as cp
import numba
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline_cpu.features_cpu import CpuFeatureEngine  # noqa: E402
from engine.features import GpuFeatureEngine  # noqa: E402


def run(engine, start, n, bs, sync):
    lat = []
    t0 = time.perf_counter()
    for b0 in range(start, start + n, bs):
        s = time.perf_counter()
        engine.batch_features(b0, b0 + bs)
        sync()
        lat.append((time.perf_counter() - s) * 1e3)
    return n / (time.perf_counter() - t0), float(np.median(lat)), float(np.percentile(lat, 99))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="HI-Small")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--start", type=int, default=2_000_000)
    ap.add_argument("--n", type=int, default=262_144)
    ap.add_argument("--cpu-n", type=int, default=65_536)
    args = ap.parse_args()
    d = Path(args.data) / args.name
    tx = pl.read_parquet(d / "transactions.parquet", columns=["src", "dst", "ts", "amt_usd"])
    n_acc = pl.read_parquet(d / "accounts.parquet", columns=["acct_id"]).height
    cols = [tx[c].to_numpy() for c in ("src", "dst", "ts", "amt_usd")]
    gpu = GpuFeatureEngine(*cols, n_accounts=n_acc)
    cpu = CpuFeatureEngine(*cols, n_accounts=n_acc)
    gsync = cp.cuda.Device().synchronize
    gpu.batch_features(args.start, args.start + 4096); gsync()
    cpu.batch_features(args.start, args.start + 256)

    rows = []
    for bs in (128, 512, 2048, 8192, 32768):
        g = run(gpu, args.start, args.n, bs, gsync)
        c = run(cpu, args.start, args.cpu_n, bs, lambda: None)
        rows.append({"batch": bs, "gpu_txn_s": g[0], "gpu_p50_ms": g[1], "gpu_p99_ms": g[2],
                     "cpu_txn_s": c[0], "cpu_p50_ms": c[1], "cpu_p99_ms": c[2], "speedup": g[0] / c[0]})
        print(f"batch {bs:6d} | GPU {g[0]:>10,.0f} txn/s p50 {g[1]:7.2f} ms | "
              f"CPU {c[0]:>9,.0f} txn/s p50 {c[1]:8.2f} ms | x{g[0] / c[0]:.1f}", flush=True)
    out = {"rows": rows, "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
           "cpu": platform.processor() or platform.machine(), "cpu_threads": numba.get_num_threads(),
           "slurm_cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
           "note": "GPU shared with another tenant's process during the run (no MIG/exclusive mode)."}
    (d / "bench.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

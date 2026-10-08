"""GPU kernels must reproduce the CPU reference on real data.

Usage (on a GPU node): python -m tests.test_features --name HI-Small [--batches 20]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl

from baseline_cpu.features_cpu import CpuFeatureEngine
from engine.features import EngineConfig, GpuFeatureEngine, feature_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="HI-Small")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=2048)
    args = ap.parse_args()
    d = Path(args.data) / args.name
    tx = pl.read_parquet(d / "transactions.parquet", columns=["src", "dst", "ts", "amt_usd"])
    n_acc = pl.read_parquet(d / "accounts.parquet", columns=["acct_id"]).height
    cols = [tx[c].to_numpy() for c in ("src", "dst", "ts", "amt_usd")]
    cfg = EngineConfig()
    gpu = GpuFeatureEngine(*cols, n_accounts=n_acc, cfg=cfg)
    cpu = CpuFeatureEngine(*cols, n_accounts=n_acc, cfg=cfg)
    names = feature_names()

    # spread test batches across the stream (incl. late ones with a full look-back window)
    rng = np.random.default_rng(0)
    starts = np.sort(rng.choice(tx.height // args.batch_size, args.batches, replace=False)) * args.batch_size
    cpu.batch_features(int(starts[0]), int(starts[0]) + 64)   # JIT warm-up
    worst, n_bad = {}, 0
    for b0 in starts:
        b0 = int(b0); b1 = min(b0 + args.batch_size, tx.height)
        g = gpu.batch_features(b0, b1).get()
        c = cpu.batch_features(b0, b1)
        ok = np.isclose(g, c, rtol=1e-4, atol=1e-2)
        if not ok.all():
            n_bad += int((~ok).sum())
            for j in np.where(~ok.all(axis=0))[0]:
                i = int(np.argmax(~ok[:, j]))
                worst.setdefault(names[j], (b0 + i, float(g[i, j]), float(c[i, j])))
    print(f"compared {args.batches} batches x {args.batch_size} txns x {len(names)} features")
    if n_bad:
        print(f"MISMATCH in {n_bad} values; first per feature (txn, gpu, cpu):")
        for k, v in worst.items():
            print(f"  {k}: {v}")
        raise SystemExit(1)
    print("OK: GPU == CPU")

    # quick timing on the same batches
    t0 = time.perf_counter()
    for b0 in starts:
        gpu.batch_features(int(b0), int(b0) + args.batch_size)
    import cupy as cp
    cp.cuda.Device().synchronize()
    tg = time.perf_counter() - t0
    t0 = time.perf_counter()
    for b0 in starts:
        cpu.batch_features(int(b0), int(b0) + args.batch_size)
    tc = time.perf_counter() - t0
    n = args.batches * args.batch_size
    print(f"GPU {n / tg:,.0f} txn/s | CPU {n / tc:,.0f} txn/s | x{tc / tg:.1f}")


if __name__ == "__main__":
    main()

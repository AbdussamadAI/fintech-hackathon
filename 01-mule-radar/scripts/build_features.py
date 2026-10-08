"""Replay the whole stream through the GPU engine and save per-transaction features.

Usage: python scripts/build_features.py --name HI-Small [--batch-size 2048]
Writes data/processed/<name>/features.npy (float32 [n_txns, n_features]) + features.json (names, timing).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.features import EngineConfig, GpuFeatureEngine, feature_names  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="HI-Small")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--batch-size", type=int, default=2048)
    args = ap.parse_args()
    d = Path(args.data) / args.name
    tx = pl.read_parquet(d / "transactions.parquet", columns=["src", "dst", "ts", "amt_usd"])
    n_acc = pl.read_parquet(d / "accounts.parquet", columns=["acct_id"]).height
    cfg = EngineConfig()
    eng = GpuFeatureEngine(*(tx[c].to_numpy() for c in ("src", "dst", "ts", "amt_usd")), n_accounts=n_acc, cfg=cfg)
    names = feature_names()
    out = cp.empty((tx.height, len(names)), cp.float32)

    eng.batch_features(0, min(args.batch_size, tx.height))   # compile + warm-up
    cp.cuda.Device().synchronize()
    lat = []
    start = cp.cuda.Event(); end = cp.cuda.Event()
    t0 = time.perf_counter()
    for b0, b1, f in eng.replay(args.batch_size):
        out[b0:b1] = f
        if (b0 // args.batch_size) % 50 == 0:     # sample per-batch latency without stalling every batch
            start.record(); eng.batch_features(b0, b1); end.record(); end.synchronize()
            lat.append(cp.cuda.get_elapsed_time(start, end))
    cp.cuda.Device().synchronize()
    wall = time.perf_counter() - t0
    feats = out.get()
    np.save(d / "features.npy", feats)
    info = {"names": names, "config": cfg.__dict__, "batch_size": args.batch_size, "n_txns": tx.height,
            "wall_s": wall, "txn_per_s": tx.height / wall,
            "batch_latency_ms": {"p50": float(np.percentile(lat, 50)), "p99": float(np.percentile(lat, 99)),
                                 "n_sampled": len(lat)},
            "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "cycle_truncated_share": float((feats[:, names.index("cyc_truncated")] > 0).mean())}
    (d / "features.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({k: v for k, v in info.items() if k != "names"}, indent=2))


if __name__ == "__main__":
    main()

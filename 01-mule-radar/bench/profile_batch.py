"""Break down one micro-batch: window CSR rebuild vs each kernel (CUDA events)."""
import sys
from pathlib import Path

import cupy as cp
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.features import GpuFeatureEngine  # noqa: E402

d = Path("data/processed/HI-Small")
tx = pl.read_parquet(d / "transactions.parquet", columns=["src", "dst", "ts", "amt_usd"])
n_acc = pl.read_parquet(d / "accounts.parquet", columns=["acct_id"]).height
eng = GpuFeatureEngine(*(tx[c].to_numpy() for c in ("src", "dst", "ts", "amt_usd")), n_accounts=n_acc)
B = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
b0 = 2_000_000
eng.batch_features(b0, b0 + B)
ev = lambda: cp.cuda.Event()  # noqa: E731
lo = int(np.searchsorted(eng.ts_host, int(eng.ts_host[b0]) - eng.cfg.lookback_s, side="right"))
print(f"window edges: {b0 + B - lo:,}")
times = {}
for name, fn in [("csr_rebuild", lambda: eng.build_window(lo, b0 + B)),
                 ("full_batch", lambda: eng.batch_features(b0, b0 + B))]:
    s, e = ev(), ev()
    s.record()
    for _ in range(10):
        fn()
    e.record(); e.synchronize()
    times[name] = cp.cuda.get_elapsed_time(s, e) / 10
print({k: round(v, 2) for k, v in times.items()}, "ms")
print(f"kernels+glue ≈ {times['full_batch'] - times['csr_rebuild']:.2f} ms")

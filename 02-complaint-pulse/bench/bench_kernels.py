"""GPU (CUDA kernels) vs CPU (NumPy/Numba reference) for K1, K2, K3: identical inputs, outputs checked equal,
median of repeats, warm-up excluded. Scales to the sizes a bank-wide deployment would need.

Usage: python bench/bench_kernels.py [--quick]   -> data/bench_kernels.json
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.engine import CpuOps, GpuOps  # noqa: E402


def timeit(fn, sync, reps=5):
    fn(); sync()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    gpu, cpu = GpuOps(), CpuOps()
    cp = gpu.cp
    gs, cs = cp.cuda.Device().synchronize, (lambda: None)
    rng = np.random.default_rng(0)
    res = {"gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(), "cpu": platform.processor() or platform.machine(),
           "k1": [], "k2": [], "k3": []}

    # ---- K1: one 15-min bucket of complaints into the cube + score every series ----
    for D in ([(21, 7, 38, 6), (41, 7, 38, 6), (101, 7, 38, 11), (201, 13, 38, 21)] if not args.quick else [(21, 7, 38, 6)]):
        n_series = int(np.prod(D)); n = 2000
        cols = [rng.integers(0, D[k] - 1, n).astype(np.int32) for k in range(4)]
        gcols = [cp.asarray(c) for c in cols]
        st_c = [np.full(n_series, 1.0, np.float32), np.ones(n_series, np.float32), np.zeros(n_series, np.float32)]
        st_g = [cp.asarray(x) for x in st_c]

        def run_c():
            c = cpu.k1_count(*cols, D); return cpu.k1_score(c, 0.01, [x.copy() for x in st_c], 0.005, 1.0, 0.5, True)

        def run_g():
            c = gpu.k1_count(*gcols, D); return gpu.k1_score(c, 0.01, [x.copy() for x in st_g], 0.005, 1.0, 0.5, True)
        zc, scc = run_c(); zg, scg = run_g()
        ok = bool(np.allclose(scc, scg.get(), atol=1e-4))
        tc, tg = timeit(run_c, cs), timeit(run_g, gs)
        res["k1"].append({"series": n_series, "cpu_ms": tc * 1e3, "gpu_ms": tg * 1e3, "speedup": tc / tg, "match": ok})
        print(f"K1 {n_series:>10,} series  cpu {tc * 1e3:8.2f} ms  gpu {tg * 1e3:7.3f} ms  x{tc / tg:6.1f}  match={ok}", flush=True)

    # ---- K2: all single values + all pairs over a window of complaints ----
    for V, n in ([(100, 20000), (386, 20000), (800, 50000), (1500, 50000)] if not args.quick else [(386, 20000)]):
        W = (n + 31) // 32
        # sparse-ish attribute bitsets (each value present in ~2-20% of complaints)
        dens = rng.uniform(0.02, 0.2, V)
        bits_b = rng.random((V, W * 32)) < dens[:, None]
        pack = lambda m: np.packbits(m).view(np.uint32)  # noqa: E731
        bits = np.stack([pack(r) for r in bits_b])
        Sm = rng.random(W * 32) < 0.1
        S, B = pack(Sm), pack(~Sm)
        attr_of = (np.arange(V) // 12).astype(np.int32)
        g_in = [cp.asarray(x) for x in (bits, S, B, attr_of)]
        oc = cpu.k2(bits, S, B, attr_of); og = [x.get() for x in gpu.k2(*g_in)]
        ok = all(np.array_equal(a, b) for a, b in zip(oc, og))
        reps = 3 if V <= 400 else 1
        tc = timeit(lambda: cpu.k2(bits, S, B, attr_of), cs, reps)
        tg = timeit(lambda: gpu.k2(*g_in), gs)
        pairs = V * (V - 1) // 2
        res["k2"].append({"values": V, "complaints": n, "pairs": pairs, "cpu_ms": tc * 1e3, "gpu_ms": tg * 1e3, "speedup": tc / tg, "match": ok})
        print(f"K2 {V:>5} values ({pairs:>9,} pairs, {n:,} complaints)  cpu {tc * 1e3:9.1f} ms  gpu {tg * 1e3:7.2f} ms  x{tc / tg:6.1f}  match={ok}", flush=True)

    # ---- K3: MinHash signatures (64 hashes) ----
    words = ["abeg", "transfer", "debit", "una", "money", "reach", "account", "app", "otp", "ussd", "card", "atm", "naira",
             "since", "morning", "fix", "am", "o", "please", "refund", "bank", "no", "dey", "work"]
    for n in ([10000, 50000, 200000] if not args.quick else [10000]):
        texts = [" ".join(rng.choice(words, rng.integers(12, 30))).encode() for _ in range(n)]
        off = np.zeros(n + 1, np.int64); off[1:] = np.cumsum([len(t) for t in texts])
        chars = np.frombuffer(b"".join(texts), np.uint8)
        ha = (rng.integers(1, 2**31, 64, dtype=np.uint32) | 1).astype(np.uint32); hb = rng.integers(0, 2**31, 64, dtype=np.uint32).astype(np.uint32)
        g_in = [cp.asarray(x) for x in (chars, off, ha, hb)]
        ok = bool(np.array_equal(cpu.k3(chars, off, ha, hb), gpu.k3(*g_in).get()))
        tc = timeit(lambda: cpu.k3(chars, off, ha, hb), cs, 3)
        tg = timeit(lambda: gpu.k3(*g_in), gs)
        res["k3"].append({"docs": n, "cpu_ms": tc * 1e3, "gpu_ms": tg * 1e3, "speedup": tc / tg, "match": ok})
        print(f"K3 {n:>8,} docs  cpu {tc * 1e3:8.1f} ms  gpu {tg * 1e3:7.2f} ms  x{tc / tg:6.1f}  match={ok}", flush=True)

    import numba
    res["cpu_threads"] = numba.get_num_threads()
    out = Path(__file__).resolve().parents[1] / "data" / "bench_kernels.json"
    out.write_text(json.dumps(res, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()

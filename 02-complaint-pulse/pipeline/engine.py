"""K1/K2/K3 behind one interface with a GPU backend (CuPy + kernels.cu) and a CPU reference (NumPy/Numba).
The CPU versions implement the same maths and are used for correctness checks and the speed comparison."""
from pathlib import Path

import numba as nb
import numpy as np

KSRC = (Path(__file__).parent / "kernels" / "kernels.cu").read_text()
_POP8 = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def popcount_rows(a):
    """Row-wise popcount of a uint32 array (NumPy >= 2 has bitwise_count; fall back to a byte table)."""
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(a).sum(axis=-1, dtype=np.int64)
    return _POP8[np.ascontiguousarray(a).view(np.uint8)].sum(axis=-1, dtype=np.int64)


# ------------------------------------------------------------------------------------------------- CPU
class CpuOps:
    name = "cpu"
    xp = np

    def __init__(self):
        pass

    def k1_count(self, d0, d1, d2, d3, dims):
        D0, D1, D2, D3 = dims
        counts = np.zeros(D0 * D1 * D2 * D3, np.float32)
        a = (d0, d1, d2, d3)
        for m in range(16):
            c = [np.full_like(a[k], dims[k] - 1) if (m >> k) & 1 else a[k] for k in range(4)]
            idx = ((c[0] * D1 + c[1]) * D2 + c[2]) * D3 + c[3]
            counts += np.bincount(idx, minlength=counts.size).astype(np.float32)
        return counts

    def k1_score(self, counts, season, state, alpha, k_cusum, min_expected, warm):
        level, var, cusum = state
        expected = np.maximum(level * season, min_expected)
        sd = np.sqrt(expected + var * season * season)
        z = (counts - expected) / sd
        c = np.maximum(0, cusum + z - k_cusum) if warm else np.zeros_like(cusum)
        cusum[:] = c
        y = counts / max(season, 1e-3)
        a = np.where(z > 2, alpha * 0.05, alpha).astype(np.float32)
        d = y - level
        level += a * d
        var[:] = (1 - a) * (var + a * d * d)
        if not warm:
            return np.zeros_like(z), np.zeros_like(z)
        return z.astype(np.float32), np.maximum(z, c).astype(np.float32)

    def k2(self, bits, S, B, attr_of):
        V = bits.shape[0]
        sS1 = popcount_rows(bits & S).astype(np.int32)
        sB1 = popcount_rows(bits & B).astype(np.int32)
        iu, ju = np.triu_indices(V, k=1)
        sS2 = np.empty(len(iu), np.int32)
        sB2 = np.empty(len(iu), np.int32)
        step = 4096
        for s in range(0, len(iu), step):
            ab = bits[iu[s:s + step]] & bits[ju[s:s + step]]
            sS2[s:s + step] = popcount_rows(ab & S)
            sB2[s:s + step] = popcount_rows(ab & B)
        same = attr_of[iu] == attr_of[ju]
        sS2[same] = 0
        sB2[same] = 0
        return sS1, sB1, sS2, sB2

    def k3(self, chars, off, ha, hb):
        return _minhash_cpu(chars, off, ha, hb)


@nb.njit(parallel=True, cache=True)
def _minhash_cpu(chars, off, ha, hb):
    n, H = len(off) - 1, len(ha)
    sig = np.empty((n, H), np.uint32)
    for d in nb.prange(n):
        s, e = off[d], off[d + 1]
        for h in range(H):
            best = np.uint32(0xFFFFFFFF)
            A, Bc = ha[h], hb[h]
            for p in range(s, e - 4):
                x = np.uint32(2166136261)
                for q in range(5):
                    c = chars[p + q]
                    if 65 <= c <= 90:
                        c += 32
                    x = np.uint32((x ^ np.uint32(c)) * np.uint32(16777619))
                v = np.uint32(A * x + Bc)
                if v < best:
                    best = v
            sig[d, h] = best
    return sig


# ------------------------------------------------------------------------------------------------- GPU
class GpuOps:
    name = "gpu"

    def __init__(self):
        import cupy as cp
        self.cp = self.xp = cp
        mod = cp.RawModule(code=KSRC, options=("-std=c++17",))
        self.f = {n: mod.get_function(n) for n in ("k1_count", "k1_score", "k2_pairs", "k2_singles", "k3_minhash")}

    def k1_count(self, d0, d1, d2, d3, dims):
        cp = self.cp
        n = len(d0)
        counts = cp.zeros(int(np.prod(dims)), cp.float32)
        if n:
            self.f["k1_count"](((n + 255) // 256,), (256,), (d0, d1, d2, d3, np.int32(n), *[np.int32(x) for x in dims], counts))
        return counts

    def k1_score(self, counts, season, state, alpha, k_cusum, min_expected, warm):
        cp = self.cp
        level, var, cusum = state
        n = counts.size
        z = cp.empty(n, cp.float32)
        sc = cp.empty(n, cp.float32)
        self.f["k1_score"](((n + 255) // 256,), (256,), (counts, np.float32(season), np.int32(n), np.float32(alpha),
                                                         np.float32(k_cusum), np.float32(min_expected), level, var, cusum,
                                                         z, sc, np.int32(1 if warm else 0)))
        return z, sc

    def k2(self, bits, S, B, attr_of):
        cp = self.cp
        V, W = bits.shape
        sS1 = cp.empty(V, cp.int32); sB1 = cp.empty(V, cp.int32)
        self.f["k2_singles"](((V * 32 + 255) // 256,), (256,), (bits, S, B, np.int32(V), np.int32(W), sS1, sB1))
        P = V * (V - 1) // 2
        sS2 = cp.empty(P, cp.int32); sB2 = cp.empty(P, cp.int32)
        threads = P * 32
        self.f["k2_pairs"](((threads + 255) // 256,), (256,), (bits, S, B, np.int32(V), np.int32(W), np.int64(P),
                                                              attr_of, sS2, sB2))
        return sS1, sB1, sS2, sB2

    def k3(self, chars, off, ha, hb):
        cp = self.cp
        n, H = len(off) - 1, len(ha)
        sig = cp.empty((n, H), cp.uint32)
        self.f["k3_minhash"]((n,), (min(H, 128),), (chars, off, np.int32(n), np.int32(H), ha, hb, sig))
        return sig


def get_ops(name: str):
    return GpuOps() if name == "gpu" else CpuOps()

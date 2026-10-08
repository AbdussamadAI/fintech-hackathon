"""CPU reference for engine/kernels/features.cu: same algorithms, same window logic, multicore Numba.

Serves two purposes:
  * correctness oracle: GPU features must match these (tests/test_features.py)
  * honest speed baseline: a parallel compiled CPU implementation, not pandas
"""
import numba as nb
import numpy as np

from engine.features import EngineConfig, NF_CYC, NF_NODE, NF_PASS, NF_SG

MAX_DISTINCT = 64
MAX_DEPTH = 6
SG_CAP = 128


@nb.njit(inline="always")
def upper_ts(ts, lo, hi, t):
    while lo < hi:
        m = (lo + hi) >> 1
        if ts[m] <= t:
            lo = m + 1
        else:
            hi = m
    return lo


@nb.njit
def distinct_count(nbr, lo, hi):
    seen = set()
    for i in range(lo, hi):
        if len(seen) >= MAX_DISTINCT:
            break
        seen.add(nbr[i])
    return len(seen)


@nb.njit
def node_stats(x, t, W, o_off, o_nbr, o_ts, o_amt, i_off, i_nbr, i_ts, i_amt, f, base):
    olo = upper_ts(o_ts, o_off[x], o_off[x + 1], t - W)
    ohi = upper_ts(o_ts, olo, o_off[x + 1], t)
    ilo = upper_ts(i_ts, i_off[x], i_off[x + 1], t - W)
    ihi = upper_ts(i_ts, ilo, i_off[x + 1], t)
    oa = np.float32(0.0)
    ia = np.float32(0.0)
    for i in range(olo, ohi):
        oa += o_amt[i]
    for i in range(ilo, ihi):
        ia += i_amt[i]
    f[base + 0] = ohi - olo
    f[base + 1] = oa
    f[base + 2] = ihi - ilo
    f[base + 3] = ia
    f[base + 4] = distinct_count(o_nbr, olo, ohi)
    f[base + 5] = distinct_count(i_nbr, ilo, ihi)
    f[base + 6] = (ia - oa) / (ia + oa) if (ia + oa) > 0 else 0.0


@nb.njit(parallel=True)
def k_node_stats(qu, qv, qt, W, o_off, o_nbr, o_ts, o_amt, i_off, i_nbr, i_ts, i_amt, out):
    for q in nb.prange(len(qu)):
        node_stats(qu[q], qt[q], W, o_off, o_nbr, o_ts, o_amt, i_off, i_nbr, i_ts, i_amt, out[q], 0)
        node_stats(qv[q], qt[q], W, o_off, o_nbr, o_ts, o_amt, i_off, i_nbr, i_ts, i_amt, out[q], NF_NODE)


@nb.njit(parallel=True)
def k_pass_through(qu, qt, qa, dt, o_off, o_ts, o_amt, i_off, i_ts, i_amt, out):
    for q in nb.prange(len(qu)):
        u, t = qu[q], qt[q]
        ilo = upper_ts(i_ts, i_off[u], i_off[u + 1], t - dt)
        ihi = upper_ts(i_ts, ilo, i_off[u + 1], t)
        olo = upper_ts(o_ts, o_off[u], o_off[u + 1], t - dt)
        ohi = upper_ts(o_ts, olo, o_off[u + 1], t)
        ra = np.float32(0.0)
        sa = np.float32(0.0)
        for i in range(ilo, ihi):
            ra += i_amt[i]
        for i in range(olo, ohi):
            sa += o_amt[i]
        last = upper_ts(i_ts, i_off[u], i_off[u + 1], t) - 1
        since = (t - i_ts[last]) / 60.0 if last >= i_off[u] else 1e6
        out[q, 0] = ra
        out[q, 1] = ihi - ilo
        out[q, 2] = min(qa[q] / ra, 100.0) if ra > 0 else 100.0
        out[q, 3] = since
        out[q, 4] = min(sa / ra, 100.0) if ra > 0 else 100.0


@nb.njit(parallel=True)
def k_scatter_gather(qu, qv, qt, W, o_off, o_nbr, o_ts, i_off, i_nbr, i_ts, i_amt, out):
    for q in nb.prange(len(qu)):
        u, v, t = qu[q], qv[q], qt[q]
        olo = upper_ts(o_ts, o_off[u], o_off[u + 1], t - W)
        ohi = upper_ts(o_ts, olo, o_off[u + 1], t)
        ilo = upper_ts(i_ts, i_off[v], i_off[v + 1], t - W)
        ihi = upper_ts(i_ts, ilo, i_off[v + 1], t)
        first = dict()
        for i in range(olo, ohi):
            if len(first) >= SG_CAP:
                break
            w = o_nbr[i]
            if w == v or w == u:
                continue
            if w not in first:
                first[w] = o_ts[i]
        cnt = 0
        val = np.float32(0.0)
        hit = set()
        hit.add(-1)
        if len(first) > 0:
            for i in range(ilo, ihi):
                w = i_nbr[i]
                if w in first and first[w] <= i_ts[i]:
                    val += i_amt[i]
                    if w not in hit:
                        hit.add(w)
                        cnt += 1
        out[q, 0] = cnt
        out[q, 1] = val


@nb.njit(parallel=True)
def k_temporal_cycles(qu, qv, qt, W, max_work, o_off, o_nbr, o_ts, out):
    for q in nb.prange(len(qu)):
        u, v, t = qu[q], qv[q], qt[q]
        for k in range(NF_CYC):
            out[q, k] = 0
        if u == v:
            continue
        s_node = np.empty(MAX_DEPTH, np.int64)
        s_i = np.empty(MAX_DEPTH, np.int64)
        s_end = np.empty(MAX_DEPTH, np.int64)
        d = 0
        work = 0
        s_node[0] = v
        s_i[0] = upper_ts(o_ts, o_off[v], o_off[v + 1], t - W)
        s_end[0] = upper_ts(o_ts, s_i[0], o_off[v + 1], t)
        while d >= 0:
            if s_i[d] >= s_end[d]:
                d -= 1
                continue
            work += 1
            if work > max_work:
                out[q, NF_CYC - 1] = 1
                break
            e = s_i[d]
            s_i[d] += 1
            x = o_nbr[e]
            te = o_ts[e]
            if x == u:
                out[q, d] += 1
                continue
            if x == v or d + 1 >= MAX_DEPTH - 1:
                continue
            seen = False
            for k in range(d + 1):
                if s_node[k] == x:
                    seen = True
                    break
            if seen:
                continue
            d += 1
            s_node[d] = x
            s_i[d] = upper_ts(o_ts, o_off[x], o_off[x + 1], te - 1)
            s_end[d] = upper_ts(o_ts, s_i[d], o_off[x + 1], t)


class CpuFeatureEngine:
    """Mirror of engine.features.GpuFeatureEngine on the CPU."""

    def __init__(self, src, dst, ts, amt, n_accounts: int, cfg: EngineConfig = EngineConfig()):
        self.cfg, self.N = cfg, int(n_accounts)
        self.src = np.asarray(src, np.int32)
        self.dst = np.asarray(dst, np.int32)
        self.ts = np.asarray(ts, np.int64)
        self.amt = np.asarray(amt, np.float32)

    def _csr(self, key, nbr, ts, amt):
        order = np.argsort(key, kind="stable")
        off = np.searchsorted(key[order], np.arange(self.N + 1)).astype(np.int64)
        return off, nbr[order], ts[order], amt[order]

    def batch_features(self, b0: int, b1: int) -> np.ndarray:
        cfg = self.cfg
        lo = int(np.searchsorted(self.ts, int(self.ts[b0]) - cfg.lookback_s, side="right"))
        s, d, t, a = self.src[lo:b1], self.dst[lo:b1], self.ts[lo:b1], self.amt[lo:b1]
        o_off, o_nbr, o_ts, o_amt = self._csr(s, d, t, a)
        i_off, i_nbr, i_ts, i_amt = self._csr(d, s, t, a)
        qu, qv, qt, qa = self.src[b0:b1], self.dst[b0:b1], self.ts[b0:b1], self.amt[b0:b1]
        nq = b1 - b0
        node_L = np.empty((nq, 2 * NF_NODE), np.float32)
        node_S = np.empty((nq, 2 * NF_NODE), np.float32)
        k_node_stats(qu, qv, qt, cfg.window_s, o_off, o_nbr, o_ts, o_amt, i_off, i_nbr, i_ts, i_amt, node_L)
        k_node_stats(qu, qv, qt, cfg.short_window_s, o_off, o_nbr, o_ts, o_amt, i_off, i_nbr, i_ts, i_amt, node_S)
        pas = np.empty((nq, NF_PASS), np.float32)
        k_pass_through(qu, qt, qa, cfg.pass_dt_s, o_off, o_ts, o_amt, i_off, i_ts, i_amt, pas)
        sg = np.empty((nq, NF_SG), np.float32)
        k_scatter_gather(qu, qv, qt, cfg.window_s, o_off, o_nbr, o_ts, i_off, i_nbr, i_ts, i_amt, sg)
        cyc = np.empty((nq, NF_CYC), np.float32)
        k_temporal_cycles(qu, qv, qt, cfg.cycle_window_s, cfg.cycle_max_work, o_off, o_nbr, o_ts, cyc)
        return np.concatenate([node_L, node_S, pas, sg, cyc], axis=1)

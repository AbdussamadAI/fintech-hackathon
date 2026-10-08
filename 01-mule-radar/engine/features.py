"""GPU feature engine: window graph + custom CUDA kernels (engine/kernels/features.cu).

The stream is processed in micro-batches. For each batch with time range [t_first, t_last]:
  1. window = edges with ts in (t_first - W, t_last]   (edges are already time-ordered, so this is
     a contiguous slice of the global arrays: two binary searches)
  2. build CSR grouped by sender and by receiver with a stable sort, which keeps each account's
     edges time-sorted (CuPy / CUB radix sort)
  3. run the kernels for every transaction in the batch; each only looks at edges with ts <= its own t

Replaying the whole dataset through this loop yields exactly the features a live deployment would
have produced, so the same code serves offline training, evaluation, benchmarks and the live demo.
"""
from dataclasses import dataclass
from pathlib import Path

import cupy as cp
import numpy as np

KERNEL_SRC = (Path(__file__).parent / "kernels" / "features.cu").read_text()
NF_NODE, NF_PASS, NF_SG, NF_CYC = 7, 5, 2, 6
NODE_NAMES = ["out_cnt", "out_amt", "in_cnt", "in_amt", "out_distinct", "in_distinct", "net_flow"]
PASS_NAMES = ["pt_recv_amt", "pt_recv_cnt", "pt_amt_over_recv", "pt_min_since_in", "pt_sent_over_recv"]
SG_NAMES = ["sg_intermediaries", "sg_value"]
CYC_NAMES = ["cyc2", "cyc3", "cyc4", "cyc5", "cyc6", "cyc_truncated"]


@dataclass
class EngineConfig:
    # sized from EDA on HI-Small: attempts last ~75h (p50), hops are ~4.5h apart (p50, p90 21h),
    # and laundered money sits ~27h (p50) in an account before moving on
    window_s: int = 72 * 3600          # long window for account stats and scatter-gather
    short_window_s: int = 6 * 3600     # short window for velocity
    pass_dt_s: int = 48 * 3600         # pass-through look-back
    cycle_window_s: int = 96 * 3600    # temporal cycles
    cycle_max_work: int = 20000        # DFS edge visits per query before truncation
    block: int = 128

    @property
    def lookback_s(self) -> int:
        return max(self.window_s, self.short_window_s, self.pass_dt_s, self.cycle_window_s)


def feature_names() -> list[str]:
    names = []
    for w in ("L", "S"):
        names += [f"src_{n}_{w}" for n in NODE_NAMES] + [f"dst_{n}_{w}" for n in NODE_NAMES]
    return names + PASS_NAMES + SG_NAMES + CYC_NAMES


class GpuFeatureEngine:
    def __init__(self, src, dst, ts, amt, n_accounts: int, cfg: EngineConfig = EngineConfig()):
        """src/dst/ts/amt: full time-sorted edge arrays (host or device). In a live deployment these
        are an append-only ring buffer; for replay we simply hold the whole dataset on the GPU."""
        self.cfg = cfg
        self.N = int(n_accounts)
        self.src = cp.asarray(src, dtype=cp.int32)
        self.dst = cp.asarray(dst, dtype=cp.int32)
        self.ts = cp.asarray(ts, dtype=cp.int64)
        self.amt = cp.asarray(amt, dtype=cp.float32)
        self.ts_host = cp.asnumpy(self.ts)
        assert np.all(np.diff(self.ts_host) >= 0), "edges must be time-sorted"
        mod = cp.RawModule(code=KERNEL_SRC, options=("-std=c++17",))
        self.k_node = mod.get_function("k_node_stats")
        self.k_pass = mod.get_function("k_pass_through")
        self.k_sg = mod.get_function("k_scatter_gather")
        self.k_cyc = mod.get_function("k_temporal_cycles")

    # -- window graph -------------------------------------------------------------------------
    def _csr(self, key, nbr, ts, amt):
        order = cp.argsort(key, kind="stable")           # stable: keeps time order inside each group
        k = key[order]
        off = cp.searchsorted(k, cp.arange(self.N + 1, dtype=cp.int32)).astype(cp.int32)
        return off, nbr[order], ts[order], amt[order]

    def build_window(self, lo: int, hi: int):
        """CSR views of edges [lo, hi) (indices into the global time-sorted arrays)."""
        s, d, t, a = self.src[lo:hi], self.dst[lo:hi], self.ts[lo:hi], self.amt[lo:hi]
        self.out = self._csr(s, d, t, a)
        self.inn = self._csr(d, s, t, a)

    # -- features for a batch of query edges ----------------------------------------------------
    def batch_features(self, b0: int, b1: int) -> cp.ndarray:
        """Features for transactions [b0, b1). Returns a device array [b1-b0, n_features]."""
        cfg = self.cfg
        t_first = int(self.ts_host[b0])
        lo = int(np.searchsorted(self.ts_host, t_first - cfg.lookback_s, side="right"))
        # the window ends at the batch end; per-edge causality (ts <= t) is enforced in the kernels
        self.build_window(lo, b1)
        qu, qv, qt, qa = self.src[b0:b1], self.dst[b0:b1], self.ts[b0:b1], self.amt[b0:b1]
        nq = b1 - b0
        grid, blk = ((nq + cfg.block - 1) // cfg.block,), (cfg.block,)
        o_off, o_nbr, o_ts, o_amt = self.out
        i_off, i_nbr, i_ts, i_amt = self.inn

        node_L = cp.empty((nq, 2 * NF_NODE), cp.float32)
        node_S = cp.empty((nq, 2 * NF_NODE), cp.float32)
        for W, buf in ((cfg.window_s, node_L), (cfg.short_window_s, node_S)):
            self.k_node(grid, blk, (qu, qv, qt, np.int32(nq), np.int64(W),
                                    o_off, o_nbr, o_ts, o_amt, i_off, i_nbr, i_ts, i_amt, buf))
        pas = cp.empty((nq, NF_PASS), cp.float32)
        self.k_pass(grid, blk, (qu, qt, qa, np.int32(nq), np.int64(cfg.pass_dt_s),
                                o_off, o_ts, o_amt, i_off, i_ts, i_amt, pas))
        sg = cp.empty((nq, NF_SG), cp.float32)
        self.k_sg(grid, blk, (qu, qv, qt, np.int32(nq), np.int64(cfg.window_s),
                              o_off, o_nbr, o_ts, i_off, i_nbr, i_ts, i_amt, sg))
        cyc = cp.empty((nq, NF_CYC), cp.float32)
        self.k_cyc(grid, blk, (qu, qv, qt, np.int32(nq), np.int64(cfg.cycle_window_s),
                               np.int32(cfg.cycle_max_work), o_off, o_nbr, o_ts, cyc))
        return cp.concatenate([node_L, node_S, pas, sg, cyc], axis=1)

    def replay(self, batch_size: int = 2048, start: int = 0, stop: int | None = None):
        """Yield (b0, b1, features) over the stream in micro-batches of `batch_size` transactions."""
        stop = len(self.ts_host) if stop is None else stop
        for b0 in range(start, stop, batch_size):
            b1 = min(b0 + batch_size, stop)
            yield b0, b1, self.batch_features(b0, b1)

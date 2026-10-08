// MuleRadar feature kernels.
//
// All kernels read a *window graph*: the edges with ts in (t_lo, t_hi], grouped by account and
// time-sorted inside each group (CSR). `out_*` is grouped by sender, `in_*` by receiver.
//   out_off[N+1], out_nbr[E], out_ts[E], out_amt[E]   (edges u -> nbr)
//   in_off [N+1], in_nbr [E], in_ts [E], in_amt [E]   (edges nbr -> v)
// Query edges q = (u -> v at time t, amount a) are the transactions being scored. Every feature
// only looks at edges with ts <= t (causal: identical to what a live stream would have seen).
// Same-minute ties count as "before", matching the minute resolution of the data.

#define MAX_DISTINCT 64   // distinct-counterparty counts saturate here
#define MAX_DEPTH 6       // longest cycle searched = MAX_DEPTH edges incl. the query edge

// first index in [lo, hi) with ts[i] > t  (time-sorted segment)
__device__ __forceinline__ int upper_ts(const long long* ts, int lo, int hi, long long t) {
    while (lo < hi) { int m = (lo + hi) >> 1; if (ts[m] <= t) lo = m + 1; else hi = m; }
    return lo;
}
// first index in [lo, hi) with ts[i] > t  -- alias kept for readability at call sites
__device__ __forceinline__ int lower_after(const long long* ts, int lo, int hi, long long t) {
    return upper_ts(ts, lo, hi, t);
}

// Saturating distinct count of nbr[lo:hi) with a small open-addressing table in registers/local mem.
__device__ int distinct_count(const int* nbr, int lo, int hi) {
    int table[MAX_DISTINCT * 2];
    #pragma unroll 4
    for (int i = 0; i < MAX_DISTINCT * 2; ++i) table[i] = -1;
    int n = 0;
    for (int i = lo; i < hi && n < MAX_DISTINCT; ++i) {
        int x = nbr[i];
        unsigned h = ((unsigned)x * 2654435761u) & (MAX_DISTINCT * 2 - 1);
        while (table[h] != -1 && table[h] != x) h = (h + 1) & (MAX_DISTINCT * 2 - 1);
        if (table[h] == -1) { table[h] = x; ++n; }
    }
    return n;
}

// ---------------------------------------------------------------------------------------------
// K1: per-account window statistics for both endpoints of each query edge.
// One thread per query edge. Writes NF_NODE floats for u then NF_NODE for v:
//   [out_cnt, out_amt, in_cnt, in_amt, out_distinct, in_distinct, net_flow_ratio]
// over the window (t - W, t].
// ---------------------------------------------------------------------------------------------
#define NF_NODE 7
__device__ void node_stats(int x, long long t, long long W,
                           const int* out_off, const int* out_nbr, const long long* out_ts, const float* out_amt,
                           const int* in_off, const int* in_nbr, const long long* in_ts, const float* in_amt,
                           float* f) {
    int olo = lower_after(out_ts, out_off[x], out_off[x + 1], t - W);
    int ohi = upper_ts(out_ts, olo, out_off[x + 1], t);
    int ilo = lower_after(in_ts, in_off[x], in_off[x + 1], t - W);
    int ihi = upper_ts(in_ts, ilo, in_off[x + 1], t);
    float oa = 0.f, ia = 0.f;
    for (int i = olo; i < ohi; ++i) oa += out_amt[i];
    for (int i = ilo; i < ihi; ++i) ia += in_amt[i];
    f[0] = (float)(ohi - olo); f[1] = oa;
    f[2] = (float)(ihi - ilo); f[3] = ia;
    f[4] = (float)distinct_count(out_nbr, olo, ohi);
    f[5] = (float)distinct_count(in_nbr, ilo, ihi);
    f[6] = (ia + oa) > 0.f ? (ia - oa) / (ia + oa) : 0.f;   // +1 = only receives, -1 = only sends
}

extern "C" __global__ void k_node_stats(
        const int* q_u, const int* q_v, const long long* q_t, int nq, long long W,
        const int* out_off, const int* out_nbr, const long long* out_ts, const float* out_amt,
        const int* in_off, const int* in_nbr, const long long* in_ts, const float* in_amt,
        float* feat /* [nq, 2*NF_NODE] */) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nq) return;
    float* f = feat + (size_t)q * 2 * NF_NODE;
    node_stats(q_u[q], q_t[q], W, out_off, out_nbr, out_ts, out_amt, in_off, in_nbr, in_ts, in_amt, f);
    node_stats(q_v[q], q_t[q], W, out_off, out_nbr, out_ts, out_amt, in_off, in_nbr, in_ts, in_amt, f + NF_NODE);
}

// ---------------------------------------------------------------------------------------------
// K2: pass-through. For query u -> v (t, a): how much did u receive in (t - dt, t] and how fast
// is u forwarding it? Mules keep little and forward fast.
//   [recv_amt_dt, recv_cnt_dt, a / recv_amt_dt (clipped), minutes since last inbound, sent_amt_dt / recv_amt_dt]
// One thread per query edge.
// ---------------------------------------------------------------------------------------------
#define NF_PASS 5
extern "C" __global__ void k_pass_through(
        const int* q_u, const long long* q_t, const float* q_a, int nq, long long dt,
        const int* out_off, const long long* out_ts, const float* out_amt,
        const int* in_off, const long long* in_ts, const float* in_amt,
        float* feat /* [nq, NF_PASS] */) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nq) return;
    int u = q_u[q]; long long t = q_t[q];
    int ilo = lower_after(in_ts, in_off[u], in_off[u + 1], t - dt);
    int ihi = upper_ts(in_ts, ilo, in_off[u + 1], t);
    int olo = lower_after(out_ts, out_off[u], out_off[u + 1], t - dt);
    int ohi = upper_ts(out_ts, olo, out_off[u + 1], t);
    float ra = 0.f, sa = 0.f;
    for (int i = ilo; i < ihi; ++i) ra += in_amt[i];
    for (int i = olo; i < ohi; ++i) sa += out_amt[i];
    // last inbound at or before t (may be older than dt; search the whole window)
    int last = upper_ts(in_ts, in_off[u], in_off[u + 1], t) - 1;
    float since = last >= in_off[u] ? (float)(t - in_ts[last]) / 60.f : 1e6f;
    float* f = feat + (size_t)q * NF_PASS;
    f[0] = ra; f[1] = (float)(ihi - ilo);
    f[2] = ra > 0.f ? fminf(q_a[q] / ra, 100.f) : 100.f;
    f[3] = since;
    f[4] = ra > 0.f ? fminf(sa / ra, 100.f) : 100.f;
}

// ---------------------------------------------------------------------------------------------
// K3: scatter-gather / 2-hop. For query u -> v at t: number of distinct intermediaries w with
// u -> w at t1 and w -> v at t2, t - W < t1 <= t2 <= t (money split through middlemen and
// re-gathered), plus the total value routed that way.
// One thread per query edge; iterates u's out-edges (bounded) and binary-searches v's in-segment
// grouped by w is not available, so we scan v's in-edges with a small hash of w -> earliest t1.
// ---------------------------------------------------------------------------------------------
#define NF_SG 2
#define SG_CAP 128
extern "C" __global__ void k_scatter_gather(
        const int* q_u, const int* q_v, const long long* q_t, int nq, long long W,
        const int* out_off, const int* out_nbr, const long long* out_ts,
        const int* in_off, const int* in_nbr, const long long* in_ts, const float* in_amt,
        float* feat /* [nq, NF_SG] */) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nq) return;
    int u = q_u[q], v = q_v[q]; long long t = q_t[q];
    int olo = lower_after(out_ts, out_off[u], out_off[u + 1], t - W);
    int ohi = upper_ts(out_ts, olo, out_off[u + 1], t);
    int ilo = lower_after(in_ts, in_off[v], in_off[v + 1], t - W);
    int ihi = upper_ts(in_ts, ilo, in_off[v + 1], t);
    // table of (w, earliest t1) for u's out-neighbours, excluding v itself
    int keys[SG_CAP * 2]; long long t1s[SG_CAP * 2]; bool hit[SG_CAP * 2];
    for (int i = 0; i < SG_CAP * 2; ++i) { keys[i] = -1; hit[i] = false; }
    int n = 0;
    for (int i = olo; i < ohi && n < SG_CAP; ++i) {
        int w = out_nbr[i]; if (w == v || w == u) continue;
        unsigned h = ((unsigned)w * 2654435761u) & (SG_CAP * 2 - 1);
        while (keys[h] != -1 && keys[h] != w) h = (h + 1) & (SG_CAP * 2 - 1);
        if (keys[h] == -1) { keys[h] = w; t1s[h] = out_ts[i]; ++n; }   // time-sorted: first = earliest
    }
    float cnt = 0.f, val = 0.f;
    if (n > 0) {
        for (int i = ilo; i < ihi; ++i) {
            int w = in_nbr[i];
            unsigned h = ((unsigned)w * 2654435761u) & (SG_CAP * 2 - 1);
            while (keys[h] != -1 && keys[h] != w) h = (h + 1) & (SG_CAP * 2 - 1);
            if (keys[h] == w && t1s[h] <= in_ts[i]) {
                val += in_amt[i];
                if (!hit[h]) { cnt += 1.f; hit[h] = true; }   // count each intermediary once
            }
        }
    }
    feat[(size_t)q * NF_SG + 0] = cnt;
    feat[(size_t)q * NF_SG + 1] = val;
}

// ---------------------------------------------------------------------------------------------
// K4: temporal cycles closed by the query edge. Query u -> v at t closes a cycle of length L if
// there is a time-respecting path v -> x1 -> ... -> u with L-1 edges, all timestamps increasing,
// all in (t - W, t]. The query edge is the most recent hop, i.e. in time order the cycle is
// v -> x1 -> ... -> u -> v (the definition GFP uses). Counts simple paths of each length with a
// bounded DFS: one thread per query, explicit stack, per-query work cap.
// Output: [cyc2, cyc3, cyc4, cyc5, cyc6, truncated_flag]
// ---------------------------------------------------------------------------------------------
#define NF_CYC (MAX_DEPTH)
extern "C" __global__ void k_temporal_cycles(
        const int* q_u, const int* q_v, const long long* q_t, int nq, long long W, int max_work,
        const int* out_off, const int* out_nbr, const long long* out_ts,
        float* feat /* [nq, NF_CYC] */) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nq) return;
    int u = q_u[q], v = q_v[q]; long long t = q_t[q];
    float* f = feat + (size_t)q * NF_CYC;
    for (int i = 0; i < NF_CYC; ++i) f[i] = 0.f;
    if (u == v) return;
    long long t_lo = t - W;
    // stack of (node, next edge index, end edge index); depth d = number of edges taken from v
    int s_node[MAX_DEPTH], s_i[MAX_DEPTH], s_end[MAX_DEPTH];
    int d = 0, work = 0;
    s_node[0] = v;
    s_i[0] = lower_after(out_ts, out_off[v], out_off[v + 1], t_lo);
    s_end[0] = upper_ts(out_ts, s_i[0], out_off[v + 1], t);
    while (d >= 0) {
        if (s_i[d] >= s_end[d] || ++work > max_work) {
            if (work > max_work) { f[NF_CYC - 1] = 1.f; break; }
            --d; continue;
        }
        int e = s_i[d]++;
        int x = out_nbr[e]; long long te = out_ts[e];
        if (x == u) { f[d] += 1.f; continue; }          // path of d+1 edges + query edge = cycle d+2
        if (x == v) continue;                            // don't loop back through the start
        if (d + 1 >= MAX_DEPTH - 1) continue;             // max cycle length reached
        // avoid revisiting a node already on the path (simple cycles)
        bool seen = false;
        for (int k = 0; k <= d; ++k) if (s_node[k] == x) { seen = true; break; }
        if (seen) continue;
        ++d;
        s_node[d] = x;
        s_i[d] = lower_after(out_ts, out_off[x], out_off[x + 1], te - 1);   // next hop at or after te
        s_end[d] = upper_ts(out_ts, s_i[d], out_off[x + 1], t);
    }
}

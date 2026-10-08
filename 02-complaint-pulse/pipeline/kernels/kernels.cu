// ComplaintPulse custom CUDA kernels.
//
// K1  burst detection over a hierarchical complaint cube (topic x channel x region x segment, each with an
//     "ALL" member), one 15-minute bucket at a time:
//       k1_count  one thread per complaint, atomically adds it to all 16 roll-up cells it belongs to
//       k1_score  one thread per series: seasonal-adjusted Poisson surprise + one-sided CUSUM, then updates the
//                 series' EWMA baseline (level and variance)
// K2  root-cause contrast mining with bitsets: for every attribute value and every pair of values, count how
//     many complaints in the spike set S and in the baseline set B carry it (popcount on 32-bit words), then
//     lift and a two-proportion z-score. One warp per (value, value) pair.
// K3  near-duplicate detection: MinHash signatures over character 5-shingles, one block per document and one
//     thread per hash function.

typedef unsigned int uint32_t;  // NVRTC has no <cstdint>

// ------------------------------------------------------------------------------------------------- K1
// cube dims D0..D3 include the ALL member as the last index of each dim
extern "C" __global__ void k1_count(const int* d0, const int* d1, const int* d2, const int* d3, int n,
                                    int D0, int D1, int D2, int D3, float* counts) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int a[4] = {d0[i], d1[i], d2[i], d3[i]};
    int all[4] = {D0 - 1, D1 - 1, D2 - 1, D3 - 1};
    #pragma unroll
    for (int m = 0; m < 16; ++m) {
        int c0 = (m & 1) ? all[0] : a[0];
        int c1 = (m & 2) ? all[1] : a[1];
        int c2 = (m & 4) ? all[2] : a[2];
        int c3 = (m & 8) ? all[3] : a[3];
        atomicAdd(&counts[((c0 * D1 + c1) * D2 + c2) * D3 + c3], 1.0f);
    }
}

// season = expected share of daily volume in this bucket's hour (global profile, mean 1.0)
extern "C" __global__ void k1_score(const float* counts, float season, int n_series, float alpha, float k_cusum,
                                    float min_expected, float* level, float* var, float* cusum, float* z_out,
                                    float* score_out, int warm) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_series) return;
    float x = counts[s];
    float lv = level[s];
    float expected = fmaxf(lv * season, min_expected);
    // Poisson surprise with an over-dispersion term learned from the series' own residual variance
    float sd = sqrtf(expected + var[s] * season * season);
    float z = (x - expected) / sd;
    float c = warm ? fmaxf(0.f, cusum[s] + z - k_cusum) : 0.f;
    cusum[s] = c;
    z_out[s] = warm ? z : 0.f;
    score_out[s] = warm ? fmaxf(z, c) : 0.f;
    // update baseline on de-seasonalised count; damp updates while the series is surprising so a
    // sustained incident does not immediately become the new normal
    float y = x / fmaxf(season, 1e-3f);
    float a = (z > 2.f) ? alpha * 0.05f : alpha;
    float d = y - lv;
    level[s] = lv + a * d;
    var[s] = (1.f - a) * (var[s] + a * d * d);
}

// ------------------------------------------------------------------------------------------------- K2
// bits: [V, W] uint32 bitsets of attribute values over the analysis window (W words)
// S, B: [W] spike and baseline masks. pairs (i < j) are enumerated by a linear index.
// out per pair: sS (support in spike), sB (support in baseline)
__device__ __forceinline__ void pair_from_index(long long p, int V, int& i, int& j) {
    // invert p = i*V - i*(i+1)/2 + (j - i - 1)
    double b = 2.0 * V - 1.0;
    i = (int)floor((b - sqrt(b * b - 8.0 * (double)p)) / 2.0);
    long long start = (long long)i * V - (long long)i * (i + 1) / 2;
    while (start > p) { --i; start = (long long)i * V - (long long)i * (i + 1) / 2; }
    while ((long long)(i + 1) * V - (long long)(i + 1) * (i + 2) / 2 <= p) {
        ++i; start = (long long)i * V - (long long)i * (i + 1) / 2;
    }
    j = (int)(p - start) + i + 1;
}

extern "C" __global__ void k2_pairs(const uint32_t* bits, const uint32_t* S, const uint32_t* B, int V, int W,
                                    long long n_pairs, const int* attr_of, int* sS, int* sB) {
    long long warp = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= n_pairs) return;
    int i, j;
    pair_from_index(warp, V, i, j);
    if (attr_of[i] == attr_of[j]) {           // two values of the same attribute never co-occur
        if (lane == 0) { sS[warp] = 0; sB[warp] = 0; }
        return;
    }
    const uint32_t* a = bits + (size_t)i * W;
    const uint32_t* b = bits + (size_t)j * W;
    int cs = 0, cb = 0;
    for (int w = lane; w < W; w += 32) {
        uint32_t ab = a[w] & b[w];
        cs += __popc(ab & S[w]);
        cb += __popc(ab & B[w]);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        cs += __shfl_down_sync(0xffffffff, cs, o);
        cb += __shfl_down_sync(0xffffffff, cb, o);
    }
    if (lane == 0) { sS[warp] = cs; sB[warp] = cb; }
}

extern "C" __global__ void k2_singles(const uint32_t* bits, const uint32_t* S, const uint32_t* B, int V, int W,
                                      int* sS, int* sB) {
    int v = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (v >= V) return;
    const uint32_t* a = bits + (size_t)v * W;
    int cs = 0, cb = 0;
    for (int w = lane; w < W; w += 32) { cs += __popc(a[w] & S[w]); cb += __popc(a[w] & B[w]); }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        cs += __shfl_down_sync(0xffffffff, cs, o);
        cb += __shfl_down_sync(0xffffffff, cb, o);
    }
    if (lane == 0) { sS[v] = cs; sB[v] = cb; }
}

// ------------------------------------------------------------------------------------------------- K3
// texts concatenated into chars with offsets; signature [n_docs, H]; shingle length 5 over lower-cased bytes
extern "C" __global__ void k3_minhash(const unsigned char* chars, const long long* off, int n_docs, int H,
                                      const uint32_t* ha, const uint32_t* hb, uint32_t* sig) {
    int d = blockIdx.x;
    if (d >= n_docs) return;
    long long s = off[d], e = off[d + 1];
    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        uint32_t best = 0xffffffffu;
        uint32_t A = ha[h], Bc = hb[h];
        for (long long p = s; p + 5 <= e; ++p) {
            uint32_t x = 2166136261u;                     // FNV-1a over the 5-byte shingle
            #pragma unroll
            for (int q = 0; q < 5; ++q) {
                unsigned char c = chars[p + q];
                if (c >= 'A' && c <= 'Z') c += 32;
                x = (x ^ c) * 16777619u;
            }
            uint32_t v = A * x + Bc;                       // universal hash family (mod 2^32)
            best = v < best ? v : best;
        }
        sig[(size_t)d * H + h] = best;
    }
}

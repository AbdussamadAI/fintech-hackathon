# MuleRadar — Architecture

## 1. System overview

```
 ┌──────────────────┐   micro-batches    ┌─────────────────────────────── GPU ENGINE ───────────────────────────────┐
 │ Transaction      │  (e.g. 10k txns /  │                                                                           │
 │ Replay Streamer  │──── 100 ms) ──────▶│  [1] Ingest ──▶ [2] Sliding-Window Graph Store (edge ring buffer + CSR)   │
 │ (IBM AML / PaySim│                    │                        │                                                  │
 │  replayed at Nx) │                    │          ┌─────────────┼──────────────┬───────────────────┐               │
 └──────────────────┘                    │          ▼             ▼              ▼                   ▼               │
                                         │  [3a] Velocity &  [3b] Pass-through [3c] Temporal    [4] Graph analytics  │
                                         │   fan-in/out        ratio kernel     chain/cycle       (cuGraph: WCC,     │
                                         │   kernels                            search kernel     PPR, Louvain)      │
                                         │          └─────────────┴──────┬───────┴───────────────────┘               │
                                         │                               ▼                                           │
                                         │          [5] Feature assembly ──▶ [6] Risk scoring (XGBoost on GPU)       │
                                         │                               ▼                                           │
                                         │          [7] Ring aggregation + reason codes                              │
                                         └───────────────────────────────┬───────────────────────────────────────────┘
                                                                         ▼
                                         ┌──────────────────────────────────────────────┐      ┌────────────────────┐
                                         │ [8] Decision layer (tiered actions)          │─────▶│ Alerts store       │
                                         │ [9] API: FastAPI (REST + WebSocket)          │      │ (SQLite/Parquet)   │
                                         └───────────────────────┬──────────────────────┘      └────────────────────┘
                                                                 ▼
                                         ┌──────────────────────────────────────────────┐
                                         │ [10] Analyst dashboard (React)               │
                                         │  live ring graph · alerts · reason codes ·   │
                                         │  CPU-vs-GPU benchmark panel · metrics        │
                                         └──────────────────────────────────────────────┘

 Side path: [B] CPU baseline (pandas + NetworkX) runs the same detectors on the same data for benchmarking & correctness checks.
```

## 2. Data

**Primary:** IBM "Transactions for Anti Money Laundering" synthetic dataset (Kaggle).
- Sizes: HI-Small (~5M txns, ~0.5M accounts) for dev; HI-Medium / larger for benchmark scaling.
- Fields: timestamp, from bank/account, to bank/account, amounts & currencies, payment format, `Is Laundering` label.
- Laundering is generated from known typologies — **fan-out, fan-in, cycle, scatter-gather, gather-scatter,
  bipartite, stack** — which map directly to our detectors and give us ground truth.

**Secondary (optional):** PaySim (mobile-money simulation) — closer to the African mobile-money context.

**Localisation for the pitch:** convert amounts to NGN, map payment formats to NIP transfers / USSD / POS / card.

### Internal data model
| Structure | Location | Shape / type | Purpose |
|---|---|---|---|
| `acct_id` mapping | GPU | dense int32 0..N-1 | compact IDs for array indexing |
| Edge ring buffer | GPU | SoA: `src, dst int32; ts int64; amt float32; fmt int8` | all edges in the last window `W` |
| Window CSR (out) / CSC (in) | GPU | `offsets int32[N+1], nbr int32[E], eidx int32[E]` | rebuilt per micro-batch for traversal |
| Bucketed counters | GPU | `[N, B]` float32/int32 (in_cnt, in_amt, out_cnt, out_amt) | O(1) sliding-window velocity stats |
| Account features | GPU | `[N, F]` float32 | model input |
| Risk scores / ring IDs | GPU → host | `[N]` float32 / int32 | output (only changed accounts copied back) |

Memory estimate (HI-Small, N≈0.5M, B=64 buckets, 4 counter arrays): ~0.5 GB. On the A100 (40/80 GB HBM2e)
the largest IBM AML sets (~180M txns) can sit entirely in GPU memory — no host↔device paging during benchmarks.

## 3. Components

### [1] Ingest
- Pinned host buffers + async `cudaMemcpyAsync` on a dedicated stream; double-buffer so batch *k+1*
  copies while batch *k* is processed (overlap transfer and compute).
- Map raw account strings → dense IDs on the host once (pre-processing), so the stream carries ints only.

### [2] Sliding-window graph store
- Append batch edges into a GPU ring buffer; evict edges older than `now − W` by advancing the tail.
- Rebuild CSR/CSC for the window per batch: sort by `(src, ts)` (CUB radix sort / cuDF) + `searchsorted`
  for offsets. Sorting by time within each adjacency list makes time-respecting traversal cheap.
- Default `W` = 24 h of simulated time; bucket width = W / B.

### [3a] Velocity & fan-in / fan-out kernel (custom CUDA)
One thread per new edge; atomics into time-bucketed counters; a second pass sums the window per *touched* account.

```cuda
// counters laid out [N][B]; bucket = (ts / bucket_width) % B
__global__ void update_buckets(const int* src, const int* dst, const long long* ts,
                               const float* amt, int n_edges, long long bucket_width, int B,
                               int* out_cnt, float* out_amt, int* in_cnt, float* in_amt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_edges) return;
    int b = (int)((ts[i] / bucket_width) % B);
    atomicAdd(&out_cnt[src[i] * B + b], 1);
    atomicAdd(&out_amt[src[i] * B + b], amt[i]);
    atomicAdd(&in_cnt [dst[i] * B + b], 1);
    atomicAdd(&in_amt [dst[i] * B + b], amt[i]);
}
// + clear_expired_bucket kernel when the clock advances a bucket
// + window_reduce kernel: one warp per touched account, warp-shuffle sum over B buckets
```
- **Distinct counterparties** (key mule signal: "19 different senders"): per batch, gather window edges of
  touched accounts → sort by `(acct, counterparty)` → segmented unique-count. Stretch: per-account
  HyperLogLog sketches for O(1) updates.
- Outputs: in/out count, in/out amount, distinct senders/receivers, burstiness (max bucket / mean).

### [3b] Pass-through (rapid in→out) kernel
- For each touched account: fraction of inflow forwarded within Δt (e.g. 30 min) and balance-retention
  ratio `1 − out_amt/in_amt`. Mules keep little and forward fast.
- Implementation: one warp per account, two-pointer merge over its time-sorted in-edges and out-edges.

### [3c] Temporal chain / cycle search kernel (custom CUDA — the showcase)
Bounded-depth (k ≤ 4–5), **time-respecting** path expansion from seed accounts:
- Frontier item = `(origin, current, last_ts, last_amt, depth)`.
- One thread (or warp for high-degree nodes) per frontier item scans `current`'s out-edges in the CSR
  where `last_ts < ts ≤ last_ts + Δhop` and `amt ∈ [α·last_amt, last_amt]` (allowing for fees/splits).
- New items appended to the next frontier via `atomicAdd` on a counter (stream compaction).
- Hitting `origin` again → **cycle**; reaching depth k with value preserved → **layering chain**.
- Seeds = accounts flagged by 3a/3b (keeps the search small and fast).
- Guardrails: cap frontier size per origin; warp-per-node for hubs to avoid divergence.

### [4] Graph analytics (RAPIDS cuGraph)
- Subgraph of suspicious edges → **weakly connected components** = candidate rings.
- **Personalised PageRank** seeded on flagged accounts → risk propagates to likely collaborators.
- **Louvain** on the window graph (stretch) → community-level features.

### [5]–[6] Features & scoring
- Feature vector per account: outputs of 3a/3b/3c + PPR score + component size + account age/KYC tier
  (simulated) + payment-format mix.
- Model: **XGBoost with `device="cuda"`** trained on the `Is Laundering` labels (time-based train/test split
  to avoid leakage). Inference on GPU per batch for touched accounts only.
- Stretch: GraphSAGE (PyTorch Geometric) node embeddings as extra features.

### [7] Ring aggregation & reason codes
- Ring score = f(member scores, size, total value moved, cycle present).
- Reason codes generated from the feature values that drove the score (top SHAP contributions — XGBoost
  supports GPU SHAP), rendered as sentences, e.g.:
  > "Received ₦4.2M from 19 accounts in 14 min, forwarded 96% within 6 min to 3 accounts in the same ring."

### [7b] GenAI analyst copilot (on-prem LLM)
- Open-weight LLM served on the same A100 (e.g. vLLM), so no customer data leaves the bank.
- **STR drafting:** turns a ring's evidence (members, flows, timeline, reason codes) into a draft Suspicious
  Transaction Report narrative for NFIU; the analyst edits and approves it.
- **Ask-the-ring:** analyst asks "why is this account flagged?" / "summarise this ring" — answered only from
  structured evidence passed in the prompt (grounded; no free recall), with transaction IDs cited.
- Guardrail: the LLM never makes the hold/clear decision; it only explains and drafts.
- MIG or a second GPU can isolate LLM inference from the detection engine if latency interferes.

### [8] Decision layer — proportionate actions (inclusion angle)
| Tier | Trigger | Action |
|---|---|---|
| 0 | low risk | none |
| 1 | elevated | silent monitoring, lower thresholds |
| 2 | high, single account | step-up verification (in-app/BVN selfie) before next outbound transfer |
| 3 | high, part of ring | hold outbound on ring members, analyst review |
| 4 | confirmed ring | escalate: draft Suspicious Transaction Report (STR) for NFIU |

Goal: block the *ring*, not every unusual new user.

### [9] API (FastAPI)
- `WS /stream` — pushes new alerts, ring updates, throughput/latency stats each batch.
- `GET /alerts`, `GET /rings/{id}`, `GET /accounts/{id}` (features + reason codes + subgraph).
- `POST /replay {speed, dataset, engine: gpu|cpu}` — controls the demo.
- `GET /bench` — latest benchmark results.

### [10] Dashboard (React + Vite)
- **Live ring view:** graph visualisation (Sigma.js / Cytoscape.js) — rings light up as they're detected.
- **Alert queue** with tier, score, reason codes, one-click "hold / clear".
- **Benchmark panel:** CPU vs GPU throughput (txns/s) and detection latency (p50/p99) side by side, live.
- **Quality panel:** precision/recall, false-positive rate on legitimate accounts.

### [B] CPU baseline
- Same detectors in pandas + NetworkX (and/or NumPy), same window logic, same thresholds.
- Used for (a) correctness — GPU and CPU outputs must match on a sample — and (b) the speed-up story.

## 4. Tech stack
| Layer | Choice | Fallback |
|---|---|---|
| Custom kernels | CUDA C++ via **CuPy `RawKernel`** (fast iteration from Python) | Numba CUDA |
| GPU dataframes / sort | RAPIDS **cuDF**, CUB | CuPy |
| Graph algorithms | RAPIDS **cuGraph** | own kernels / NetworkX on CPU |
| Model | **XGBoost (CUDA)** | LightGBM CPU |
| GenAI | Open-weight instruct LLM served with **vLLM** on the A100 | Hosted LLM API with synthetic data only |
| API | FastAPI + uvicorn | Flask |
| Frontend | React + Vite + Sigma.js + Recharts | Streamlit (if time is short) |
| Compute | **NVIDIA A100 via Slurm** (compile for `sm_80`) | Kaggle/Colab T4 |
| Environment | conda/mamba env or **Apptainer** image of the RAPIDS container (Docker is usually unavailable on HPC) | pip + CuPy only |
| Profiling | Nsight Systems / Nsight Compute, CUDA events | `time.perf_counter` |

### Running on the Slurm cluster
```
 Laptop (Mac)                      Login node                    A100 compute node
 ┌──────────────┐   ssh -L tunnel  ┌───────────┐    ssh hop      ┌──────────────────────────────┐
 │ Dashboard    │─────────────────▶│           │────────────────▶│ GPU engine + FastAPI :8000   │
 │ (React/Vite) │◀── WebSocket ────│           │◀────────────────│ (srun / sbatch job)          │
 └──────────────┘                  └───────────┘                 └──────────────────────────────┘
```
- **Development:** interactive allocation, e.g. `salloc --gres=gpu:a100:1 --time=04:00:00`, then run notebooks/
  scripts there (Jupyter via tunnel if allowed).
- **Benchmarks:** batch jobs (`sbatch bench/run_bench.sbatch`) so results are reproducible and logged.
- **Demo:** the dashboard runs on the laptop; the engine + API run on the compute node and are reached through
  `ssh -L 8000:<node>:8000 user@login`. Check that the cluster allows port forwarding to compute nodes.
- Partition/GRES names (`gpu:a100:1`, partition names) vary per cluster — confirm with `sinfo` / cluster docs.
- **A100 specifics worth using:** large L2 cache (40 MB) helps the atomics-heavy bucket kernel; high memory
  bandwidth suits sort-heavy CSR rebuilds; TF32/FP16 tensor cores for the optional GNN.
- **Cost story:** also run the benchmark once on a T4 (Kaggle) to show it still works on a cheap, commodity GPU —
  the A100 shows the ceiling, the T4 shows affordability for smaller banks.

## 5. Benchmark methodology (judges will probe this)
- Same data, same window, same thresholds on CPU and GPU; verify identical flagged-account sets on a sample.
- Report: **end-to-end latency per batch** (ingest → alert), **throughput** (txns/s), and **kernel-only** times.
- Warm-up runs excluded; median of ≥5 runs; hardware listed (CPU model/cores, GPU model).
- Scaling curve: 1M → 5M → 30M → ~180M transactions, CPU vs A100 (plus one T4 run for the affordability point).
- CPU baseline runs on the same compute node's CPU cores, so the comparison is on the same machine.
- Be honest: show where GPU does *not* help (tiny batches) — credibility wins points.

## 6. Evaluation
- Account-level **precision, recall, PR-AUC** against `Is Laundering` labels.
- Ablation: rules only → + temporal patterns (3c) → + graph (4) → + model (6).
- **Inclusion metric:** false-positive rate on legitimate accounts and "innocent accounts frozen per 10k",
  compared with a naive threshold rule.
- Time-to-detect: simulated minutes from first laundering transaction to ring alert.

## 7. Repository layout (planned)
```
01-mule-radar/
├── data/                 # raw + preprocessed parquet (gitignored)
├── engine/
│   ├── kernels/          # .cu sources loaded by CuPy RawKernel
│   ├── ingest.py
│   ├── window_graph.py
│   ├── detectors.py      # 3a, 3b, 3c wrappers
│   ├── graph_analytics.py
│   ├── scoring.py
│   └── pipeline.py
├── baseline_cpu/         # pandas + NetworkX equivalents
├── api/                  # FastAPI app
├── dashboard/            # React app
├── bench/                # benchmark scripts + results
├── slurm/                # sbatch/salloc scripts, env setup, tunnel helper
├── notebooks/            # EDA, model training
└── pitch/                # slides, demo video
```

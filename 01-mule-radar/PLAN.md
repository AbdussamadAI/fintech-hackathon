# MuleRadar — Execution Plan

> Assumptions (update when known): hackathon of ~3–5 days build time, team of 1–4.
> **Compute: NVIDIA A100 via Slurm** (Kaggle/Colab T4 as backup). Phases are ordered by dependency; durations are relative.

## Phase 0 — Setup & data (≈10%)
- [ ] Check cluster: `sinfo` for A100 partition/GRES name, max job time, queue wait, `module avail cuda`.
- [ ] Build env on the cluster: RAPIDS via conda/mamba or `apptainer pull` of the RAPIDS image; CuPy,
      XGBoost-CUDA, FastAPI. Run a hello-kernel on the A100 (`nvidia-smi`, compile for `sm_80`).
- [ ] Write `slurm/` helpers: interactive `salloc` script, benchmark `sbatch` template, SSH tunnel command.
- [ ] Verify an SSH tunnel from the laptop to a port on a compute node works (needed for the live demo).
- [ ] Download IBM AML HI-Small; convert to Parquet; map accounts → dense int IDs.
- [ ] EDA notebook: laundering rate, typology examples, time span, degree distributions.
- **Exit:** data loads to GPU in seconds; one laundering ring visualised by hand.

## Phase 1 — CPU baseline (≈15%)
- [ ] Sliding window + fan-in/fan-out + pass-through in pandas.
- [ ] Temporal chain/cycle search in NetworkX/Python.
- [ ] Measure throughput and quality (precision/recall) → this is the "before" number.
- **Exit:** working, correct, slow reference implementation.

## Phase 2 — GPU engine core (≈25%) ← main technical edge
- [ ] Edge ring buffer + per-batch CSR rebuild on GPU.
- [ ] Kernel 3a: bucketed velocity / fan-in / fan-out + distinct-counterparty counts.
- [ ] Kernel 3b: pass-through ratio.
- [ ] Kernel 3c: time-respecting chain/cycle frontier expansion.
- [ ] Correctness test: GPU vs CPU outputs match on a sample.
- [ ] Profile with Nsight; fix the top bottleneck (likely transfers or atomics contention).
- **Exit:** GPU detectors are correct and much faster than baseline on HI-Small.

## Phase 3 — Graph analytics + scoring (≈15%)
- [ ] cuGraph WCC on suspicious subgraph → rings; personalised PageRank.
- [ ] Feature assembly; XGBoost-CUDA training with time-based split.
- [ ] Reason codes from GPU SHAP values.
- [ ] GenAI copilot: vLLM on the A100; STR-draft and "explain this ring" prompts grounded in ring evidence.
- [ ] Ablation table (rules → +temporal → +graph → +model).
- **Exit:** ranked rings with explanations; quality numbers beat the naive rule.

## Phase 4 — Streaming + API (≈10%)
- [ ] Replay streamer (configurable speed), double-buffered async ingest.
- [ ] FastAPI REST + WebSocket; engine switch `gpu|cpu` for the live comparison.
- **Exit:** alerts stream out in real time.

## Phase 5 — Dashboard (≈10%)
- [ ] Live ring graph, alert queue with tiers + reason codes.
- [ ] Side-by-side CPU vs GPU throughput / latency panel.
- [ ] Quality panel (precision/recall, false-positive rate).
- **Exit:** demo runs end to end from one command.

## Phase 6 — Benchmarks, polish, pitch (≈15%)
- [ ] Scaling benchmark (1M → 5M → 30M+), charts.
- [ ] Record a backup demo video (in case live GPU/Wi-Fi fails).
- [ ] Slides + 3-minute script; rehearse twice.

## MVP vs stretch
| MVP (must ship) | Stretch |
|---|---|
| Kernels 3a, 3b, 3c on GPU | HyperLogLog distinct counts |
| cuGraph WCC + PPR | Louvain communities |
| XGBoost-CUDA scoring + reason codes | GraphSAGE embeddings |
| CPU vs GPU live benchmark | Multi-GPU / larger datasets |
| Streamlit or React dashboard | Free-form analyst chat with the copilot |
| GenAI STR draft for one ring (required: submission asks for AI/GenAI) | |
| Tiered actions | PaySim / mobile-money variant |

If time runs short, cut in this order: Louvain → GraphSAGE → React (use Streamlit) → PPR.
**Never cut:** kernel 3c, the CPU-vs-GPU comparison, and the false-positive (inclusion) metric.

## Suggested roles (if team of 3–4)
- **GPU engineer (you):** kernels, window graph, profiling, benchmark.
- **ML/data:** EDA, baseline, features, XGBoost, evaluation.
- **Full-stack:** API, streamer, dashboard.
- **Pitch/domain:** Nigerian AML context (CBN, NFIU), slides, demo script, business model.

## Judging-criteria map
| Criterion | How MuleRadar scores |
|---|---|
| Innovation | Custom CUDA temporal-motif kernels + GPU graph analytics; detects *networks*, not single accounts |
| Feasibility | Working real-time demo on public data; honest, reproducible benchmarks; runs on one commodity GPU |
| Impact | Stops mule rings in minutes; tiered actions cut wrongful freezes of new/low-income users |
| Scalability | Throughput scaling curve; one GPU replaces a CPU cluster → affordable for tier-2 banks, MFBs, fintechs |
| Presentation | Live rings lighting up + CPU-vs-GPU race + plain-English reason codes |

## Demo script (3 minutes)
1. **Hook (20s):** "₦X billion moved through mule accounts last year; by the time batch AML runs, the money is gone."
2. **Live (90s):** Start replay. CPU engine falls behind; switch to GPU — keeps up at Nx real time.
   A ring lights up; click it → reason codes → tier-3 hold on the ring only.
3. **Evidence (40s):** Scaling chart, precision/recall, false-positive reduction vs naive rules.
4. **Close (30s):** Deployment path (sits beside existing core banking / NIP feed), cost, next steps.

## Risks & mitigations
| Risk | Mitigation |
|---|---|
| Slurm queue wait / no node on demo day | Book a reservation or start the job well before the slot; Kaggle T4 backup + recorded video |
| Job time limit kills the demo | Request enough `--time`; engine restarts from Parquet in seconds |
| Compute nodes block port forwarding / have no internet | Test tunnel in Phase 0; pre-download data and packages on the login node |
| RAPIDS install pain | Apptainer image of the RAPIDS container or conda env; CuPy-only fallback |
| "A100 isn't realistic for a bank" | Also show the T4 run: works on a cheap GPU, A100 shows headroom |
| Kernel bugs / race conditions | CPU baseline as oracle; `compute-sanitizer` |
| Synthetic data criticised | Explain why (privacy, labels); show how it plugs into real NIP/ISO 20022 feeds |
| Speed-up looks inflated | Publish methodology, same algorithms on both, show kernel vs end-to-end times |
| Too many false positives | Tier thresholds tuned on validation; report FPR explicitly |

## Open questions
- Exact hackathon duration and submission format (repo, video, slides)?
- Team size and skills?
- A100 40 GB or 80 GB? Max Slurm job length, and can you get a reservation for demo day?
- Will the demo be presented live from the cluster, or from a recording?
- Does the sponsor bank offer any API or sandbox data?

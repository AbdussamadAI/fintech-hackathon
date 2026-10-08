# MuleRadar — GPU-Accelerated Real-Time Money-Mule Network Detection

**Challenge:** #5 Combating Digital Fraud and Financial Crime
**One-liner:** MuleRadar watches the live transaction graph and uses custom CUDA kernels to expose
money-mule rings in milliseconds — not after the money is gone — while freezing fewer innocent accounts.

## The problem
- Fraud proceeds move through chains of "mule" accounts in minutes, split into many small transfers.
- Rule-based AML looks at accounts one at a time, runs in overnight batches, and produces floods of
  false positives — often freezing the accounts of new, low-income digital users.
- Nigerian banks have been penalised for weak AML controls; regulators (CBN, NFIU) are tightening oversight.

## Our approach
1. **Stream** transactions into a GPU-resident sliding-window graph.
2. **Detect** laundering *patterns* (fan-in, fan-out, rapid pass-through, chains, cycles) with custom CUDA kernels.
3. **Expand** from suspicious accounts to whole **rings** with GPU graph analytics (components, personalised PageRank, Louvain).
4. **Score & explain** each account and ring with a GPU-trained model plus human-readable reason codes.
5. **Act proportionately:** monitor → step-up verification → hold outbound → escalate — instead of blanket freezes.

6. **Draft & explain with GenAI:** an on-prem LLM writes STR narratives and answers analyst questions from the evidence.

## Results (IBM AML HI-Small, time-ordered 60/20/20 split, test = last 20% of the stream)
See [SUBMISSION.md](SUBMISSION.md) for the full table. Laundering-transaction F1 66.6% ± 0.3 (best published 64.8%),
90/95 known rings flagged, 99.6% fewer innocent accounts flagged than a threshold rule at equal recall,
GPU features bit-identical to the CPU reference.

## Reproduce (Slurm cluster)
```bash
./slurm/sync.sh                                   # laptop -> cluster (code only)
# on the login node, with a GPU job running (../slurm-notebook.sh -p A100 -g 1):
slurm/run.sh python scripts/prepare_data.py --name HI-Small   # CSV -> Parquet (needs data/raw/HI-Small_*)
slurm/run.sh python scripts/eda.py --name HI-Small
slurm/run.sh python -m tests.test_features --name HI-Small    # GPU == CPU check
slurm/run.sh python scripts/build_features.py --name HI-Small # replay stream through CUDA kernels
slurm/run.sh python scripts/train.py --name HI-Small --seeds 3 --ablation
slurm/run.sh python scripts/eval_rings.py --name HI-Small
slurm/run.sh python bench/bench.py --name HI-Small
slurm/run.sh python scripts/export_demo.py --name HI-Small && python dashboard/build.py <demo.json>
```
Environment: Python 3.13 venv (`~/envs/mule13`) with cupy-cuda12x, xgboost==2.1.4 (CUDA 12 build), numba,
polars, pyarrow, scikit-learn. `slurm/run.sh` sets `CUDA_PATH` to a shim so CuPy finds the system headers.

## Layout
- `engine/kernels/features.cu` — CUDA kernels (velocity, pass-through, scatter-gather, temporal cycles)
- `engine/features.py` — sliding-window graph + kernel launches; `engine/str_draft.py` — grounded STR drafting
- `baseline_cpu/features_cpu.py` — same algorithms in Numba (correctness oracle + speed baseline)
- `scripts/` — data prep, EDA, feature replay, training, ring evaluation, demo export
- `bench/` — throughput benchmark and per-batch profile; `dashboard/` — analyst console (single HTML file)

## Docs
- [../SUBMISSION_REQUIREMENTS.md](../SUBMISSION_REQUIREMENTS.md) — required submission sections + MuleRadar outline
- [PLAN.md](PLAN.md) — phases, milestones, MVP vs stretch, demo script, judging map, risks
- [ARCHITECTURE.md](ARCHITECTURE.md) — system design, data model, kernel designs, tech stack, benchmarks

## Status
- [x] Planning
- [x] Data + CPU baseline
- [x] GPU engine (verified against CPU)
- [x] Model, ring evaluation, STR drafting, dashboard
- [ ] Team section + pitch rehearsal

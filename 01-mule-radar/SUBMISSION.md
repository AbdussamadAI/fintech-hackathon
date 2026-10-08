# MuleRadar — catching money-mule rings while the money is still in the system

**Challenge #5 — Combating Digital Fraud and Financial Crime**

> MuleRadar watches the live transfer graph on a single GPU, recognises the *shapes* that laundering
> leaves behind (fan-in, fan-out, pass-through, scatter-gather, round-trip cycles), and raises a ring-level
> alert with plain-English reasons and a draft Suspicious Transaction Report. It holds the ring, not every
> unusual new customer.

---

## 1. Problem & Opportunity

**Fraud proceeds leave Nigerian banks through chains of mule accounts faster than today's controls can react.**

- Nigerian financial institutions lost **₦52.26 billion to fraud in 2024, three times the ₦17.67 billion of
  2023** and 4.5× the 2020 level; attempted fraud rose 338% year on year (NIBSS fraud report, 2024).
- The rails are fast and huge: NIBSS processed **11.2 billion e-payment transactions worth ₦1.07 quadrillion
  in 2024**, about 355 transfers every second on average. Stolen money is split, hopped across accounts and
  cashed out within hours.
- Most transaction monitoring still checks **one account at a time against fixed thresholds**, often in
  overnight batches. A mule ring is a *network* pattern: each account looks ordinary on its own; the crime is
  only visible in how money moves between them.
- Regulators are raising the cost of weak controls: Nigerian regulators fined seven of the largest banks a
  combined **US$10.7 million in 2024**, more than 60× the previous year, including AML failures, and CBN has
  introduced a fixed penalty regime for AML breaches.
- Blunt rules also **hurt financial inclusion**. Threshold rules flag huge numbers of legitimate customers,
  and the newly banked (students, small traders, first-time digital users) are the ones whose accounts get
  frozen while mule recruiters target exactly those groups.

**The opportunity:** detect rings (not single accounts) within the same window in which a hold can still
stop the money, and do it precisely enough that banks freeze far fewer innocent customers.

## 2. Innovative Solution

MuleRadar is a streaming detection engine plus an analyst copilot.

```
transfer stream ─▶ GPU sliding-window graph ─▶ 4 custom CUDA detectors ─▶ risk model ─▶ rings ─▶ tiered action
 (NIP / ISO 20022)   (rebuilt per micro-batch)   velocity · pass-through     (gradient     (connected   + reason codes
                                                  scatter-gather · cycles    boosted trees) components)  + GenAI STR draft
```

1. **GPU sliding-window transaction graph.** Transfers arrive in time order; for each micro-batch the engine
   rebuilds the last 96 hours of the graph on the GPU (≈2 million edges in ~1 ms) as time-sorted adjacency
   lists for every account.
2. **Four custom CUDA kernels** compute, for every incoming transfer and using only the past, the laundering
   signals that published research identifies as decisive:
   - *Account velocity:* counts, value and **distinct counterparties** sent/received in 6h and 72h windows.
   - *Pass-through:* how much the sender received in the previous 48h, how much of it is being forwarded, how
     soon after the last credit (mules keep little and forward fast).
   - *Scatter-gather:* how many middlemen sit on time-ordered two-hop paths between the same sender and receiver.
   - *Temporal cycles:* does this transfer close a time-respecting round trip of 2–6 accounts (money coming back
     to where it started)?
3. **AI risk scoring.** Gradient-boosted trees (XGBoost on GPU) score each transfer; alerted transfers are
   joined into **rings** (connected components), and every alert carries **reason codes** derived from the
   model's per-feature contributions (SHAP), e.g. *"receiver collected from 19 different accounts in 72h"*.
4. **GenAI analyst copilot.** For each ring an LLM drafts the **Suspicious Transaction Report narrative for
   the NFIU** from the ring's evidence only (accounts, transfers with IDs, timeline, reason codes). A validator
   rejects any draft that cites a transaction or account not in the evidence, and the LLM never makes the
   hold/clear decision. It runs against an open-weight model served on the bank's own GPU (OpenAI-compatible
   endpoint such as vLLM), so no customer data leaves the bank; a deterministic grounded draft is the fallback.
5. **Proportionate actions** instead of blanket freezes: monitor → step-up verification (BVN selfie /
   in-app confirmation) → hold outbound on ring members → escalate with STR.

**What is new:** the detectors are not generic graph libraries. They are purpose-built CUDA kernels for
*temporal* laundering motifs, verified bit-for-bit against a CPU reference, running in a streaming loop on
one GPU.

## 3. Impact & Value (measured)

All numbers below are measured by us on the public IBM "Transactions for Anti-Money Laundering" benchmark
(HI-Small: 5.08M transfers, 515k accounts, 18 days), replayed as a stream, with the **same time-ordered
60/20/20 train/validation/test split used by the published papers**. Test = the last 20% of the stream, never
seen during training.

| Result | MuleRadar | Reference |
|---|---|---|
| Laundering-transaction F1 (minority class) | **66.6% ± 0.3 (3 seeds)** | IBM Graph Feature Preprocessor + XGBoost (ICAIF'24): 63.2–64.8%; PNA graph neural network: 56.8%; GIN: 28.7%; plain XGBoost: 19.8% |
| Precision of alerts | **84.8% (recall 54.9%)** | |
| Known laundering rings flagged (test period) | **90 of 95 (95%)**; all cycle, scatter-gather, gather-scatter and stack rings | |
| Innocent accounts flagged per 10,000 active accounts | **7.6** | 1,950 for a "24h outbound ≥ $4,754" rule at the *same* recall → **99.6% fewer wrongful flags** |
| Throughput, one A100 GPU | **1.28 million transfers/s** at batch 32,768 (15× the same algorithms on 8 CPU cores; 23× at batch 2,048) | IBM GFP: ≈14k transfers/s on 64 CPU threads (batch 2048) |
| Latency per micro-batch of 2,048 transfers | **15.2 ms** | IBM GFP: 148 ms |

Ablation (each detector family adds accuracy):

| Features | F1 |
|---|---|
| Transfer fields only | 21.7 ± 0.1 |
| + account velocity | 63.4 ± 0.2 |
| + pass-through | 65.7 ± 0.5 |
| + scatter-gather | 66.5 ± 0.1 |
| + temporal cycles (full) | **66.6 ± 0.3** |

**What this means for a Nigerian bank**

- **Fewer wrongful freezes → inclusion.** To catch the same half of the mule accounts, a typical amount rule
  would put 70,616 accounts under suspicion; MuleRadar flags 1,401, of which 81% are truly involved.
  For students and small traders that is the difference between a working account and a frozen one.
- **Capacity to spare.** Nigeria's 2024 national average is ~355 e-payments per second; one GPU sustains
  hundreds of thousands per second, so a single server covers a tier-1 bank's peak load with large headroom,
  and the same software runs on a cheaper GPU for MFBs and fintechs.
- **Analyst time.** Each alert arrives as a ring with reasons and a drafted STR narrative instead of a list of
  isolated threshold hits.

**Honest limits.** The benchmark is synthetic (real bank labels are confidential). In it, laundering is
mostly generated as one payment format, which lets every model, ours included, alert on a ring's very first
transfer; our measured "95% of ring value still stoppable at first alert" is therefore an upper bound. On
real data the first alert will come a few hops in; the network detectors (velocity, pass-through,
scatter-gather, cycles) are what carry detection once that shortcut is gone, and the ablation shows they lift
F1 from 21.7% to 66.6%.

## 4. Target Users & Implementation

**Users.** Fraud and AML operations teams and compliance officers at commercial banks first; then
microfinance banks, payment service banks and fintechs, which face the same CBN/NFIU obligations with far
smaller teams. Customers benefit directly through fewer wrongful freezes and faster recovery of stolen funds.

**How it deploys.**
- Reads a *copy* of the transfer feed (NIP / ISO 20022 messages or the core-banking transaction log); no
  change to core systems.
- Runs on **one on-premises GPU server**; customer data never leaves the bank, and the copilot LLM is
  open-weight on the same hardware.
- Outputs alerts to the existing case-management tool via API; analysts approve every hold and STR.

**Rollout.**
1. *Back-test* on 3–6 months of the bank's historical transfers and confirmed fraud cases.
2. *Shadow mode*: live alerts to analysts only, measure precision and wrongful-flag rate against the current
   system.
3. *Analyst-in-the-loop holds* on tier-3/4 rings.
4. *Automated step-up verification* for tier-2 accounts.

**Feasibility today.** Everything in Section 3 runs end-to-end now: data pipeline, CUDA engine, CPU
reference, model training, ring evaluation, STR drafting and the analyst console. Built in Python with CuPy
(custom CUDA C++ kernels), Numba, XGBoost and Polars; reproducible with one script per step.

**Business model.** Annual licence per institution, tiered by transaction volume; managed appliance option
for MFBs that have no GPU infrastructure.

## 5. Team & Expertise

| Member | Role | Contribution to MuleRadar |
|---|---|---|
| **Abdussamad Ali** | CEO | Leads the venture and the GPU/AI build: problem framing, the CUDA detection engine and model evaluation; owns bank partnerships and the go-to-market plan. *[add one line of prior experience]* |
| **Abubakar Abdullahi** | CTO | Owns the technical architecture and deployment path: streaming ingestion from NIP / core-banking feeds, on-premises GPU server, security and integration with case-management tools. *[add one line of prior experience]* |
| **Umar Mohammed Bashir** | Product Manager | Turns detections into analyst workflow: tiered actions, the STR drafting flow for NFIU, CBN/NFIU compliance requirements, rollout plan with pilot banks. *[add one line of prior experience]* |
| **Abdulbasit Idris Ali** | UI Designer | Designs the analyst console: alert queue, ring visualisation, plain-English reason codes and the review-and-approve STR screen. *[add one line of prior experience]* |

Together the team covers the three things a bank needs to adopt this: a detection engine that works at
national scale (CEO/CTO), a workflow that fits compliance obligations (PM), and an interface analysts can
act on in seconds (design).

---

### Sources
- NIBSS 2024 fraud report via [21st Century Chronicle](https://21stcenturychronicle.com/nigerian-banks-lost-n52-26-billion-to-fraud-in-2024-report/) and [TechCabal](https://techcabal.com/2024/05/24/fraud-losses-decline-q1-2024/)
- NIBSS 2024 e-payment volumes via [TechCabal Insights](https://insights.techcabal.com/nigerian-payments-report-2024-online-transfers-dominate-atm-transactions-decline/)
- Regulatory fines 2024: [BusinessFront](https://businessfront.com/finance/insights/regulatory-fines-nigerias-top-lenders/), [Prime Business Africa](https://www.primebusiness.africa/?p=227110)
- Benchmark: Altman et al., *Realistic Synthetic Financial Transactions for AML Models*, NeurIPS 2023 (IBM AML dataset);
  Egressy et al., [*Provably Powerful GNNs for Directed Multigraphs*](https://arxiv.org/abs/2306.11586), AAAI 2024;
  Blanuša et al., [*Graph Feature Preprocessor*](https://arxiv.org/abs/2402.08593), ICAIF 2024

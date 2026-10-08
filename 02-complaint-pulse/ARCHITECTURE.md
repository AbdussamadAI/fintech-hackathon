# ComplaintPulse — Architecture

## 1. System overview

```
 CHANNELS                      ┌──────────────────────── A100 (Slurm) ─────────────────────────────────────────┐
 ┌──────────────┐              │                                                                               │
 │ Call audio   │──┐           │ [1] Ingest & normalise ─▶ [2] ASR (Whisper, batched) ─▶ [3] PII redaction     │
 │ Chat / email │  │  replay   │                                                              │                │
 │ Social posts │──┼─ stream ─▶│        ┌─────────────────────────────────────────────────────┘                │
 │ App reviews  │  │ (micro-   │        ▼                                                                      │
 │ Branch forms │──┘  batches) │ [4] GenAI extraction (vLLM) ──┐     [5] Embeddings (multilingual, batched)    │
 └──────────────┘              │   product, issue, amount,     │              │                                │
                               │   txn ref, sentiment, urgency │              ▼                                │
 OPERATIONAL EVENTS            │                               │   [6] Near-dup / repost removal (MinHash K3)  │
 ┌──────────────┐              │                               │              ▼                                │
 │ App releases │              │                               └─▶ [7] Online topic clustering (cuML)          │
 │ Outages/logs │──────────────▶ [10] Event correlator                        ▼                                │
 │ Failed txns  │              │        ▲                       [8] Burst detection over topic × channel ×     │
 └──────────────┘              │        │                           region × product series (custom kernel K1) │
                               │        │                                     ▼                                │
                               │        └───────────────────── [9] Root-cause contrast mining (custom kernel K2)│
                               │                                              ▼                                │
                               │                 [11] Systemic score + priority ─▶ [12] LLM incident brief     │
                               └──────────────────────────────────────────────┬────────────────────────────────┘
                                                                              ▼
                               [13] FastAPI (REST + WebSocket) ──▶ [14] Dashboard: live issue radar, root cause,
                                                                         briefs, routing, CPU-vs-GPU panel
 Side path: [B] CPU baseline for the same pipeline (benchmark + correctness).
 Cross-link: "unauthorised debit" complaints are sent as seed accounts to MuleRadar (01).
```

## 2. Data

No bank will hand over real complaints for a hackathon, so we combine public data (for scale) with
synthetic Nigerian data (for local relevance and ground truth).

| Source | Use |
|---|---|
| **CFPB Consumer Complaint Database** (US, millions of real complaints, with product/issue labels) | Scale benchmark; checks that extraction and clustering work on real complaint text |
| **Synthetic Nigerian complaint stream** (our generator) | Main demo: 5 languages, 6 channels, realistic products (NIP transfers, USSD, POS, cards, loans, app) |
| **Injected incidents** with known start time and cause (e.g. "app v4.2.1 OTP failure on Android", "NIP reversals delayed via one switch", "ATM cluster in Kano dispensing errors") | Ground truth for time-to-detect and root-cause accuracy |
| **Synthetic call audio** (TTS of generated complaints, Nigerian-accented voices) | Exercises the speech-recognition path |
| Optional: NaijaSenti (Hausa/Igbo/Yoruba/Pidgin sentiment), AfriSpeech (Nigerian-accented speech) | Validate sentiment and speech recognition on real local-language data |
| **Operational event log** (synthetic): app releases, outages, failed-transaction rates | Root-cause correlation |

The generator itself uses the LLM on the A100 to write varied complaints, including code-switching
("Abeg, I transfer 50k since morning, e no reach, una don debit me").

### Complaint record (after understanding)
`id, ts, channel, language, region, branch/ATM id, product, issue, sub_issue, amount, txn_ref,
app_version, device_os, partner_switch, sentiment, urgency, vulnerable_flag, embedding[1024], topic_id, dup_group`

## 3. Components

### [1] Ingest & normalise
- One schema for all channels; the replay streamer sends micro-batches (e.g. every 1 s of simulated time, sped up N×).

### [2] Speech recognition
- Whisper large-v3 via faster-whisper / CTranslate2 (FP16, batched) on the A100.
- Language ID from Whisper, then routed; low-confidence transcripts are flagged rather than trusted.

### [3] PII redaction
- Masks account numbers, BVN, phone numbers, card numbers (regex + NER) **before** anything is stored or sent to the LLM.
- Transaction references are kept, hashed, for linking.

### [4] GenAI extraction (vLLM)
- Open-weight instruct LLM on the A100; **JSON-schema-constrained output** (product, issue, amount, txn_ref,
  sentiment, urgency, language, summary_en).
- Large batches with continuous batching → high throughput; prompts few-shot in each language.
- Fallback: fine-tuned small classifier for product/issue if LLM throughput is the bottleneck.

### [5] Embeddings
- Multilingual embedding model (e.g. BGE-M3 or multilingual-E5), FP16, batched on the GPU.
- Complaints in different languages about the same problem land close together → one topic, not five.

### [6] Near-duplicate removal — custom kernel K3
- A viral social post reposted 2,000 times is one complaint signal, not 2,000.
- **MinHash** signatures over character shingles: one thread block per document, one thread per hash
  function; then LSH banding (sort by band hash, group equal keys) to find duplicate groups.
- Keeps `dup_group` and a count, so reach is still measured without inflating volume.

### [7] Online topic clustering
- Bootstrap: cuML UMAP + HDBSCAN on a historical window → initial topics.
- Streaming: each new complaint goes to its nearest topic centroid (GPU top-k cosine similarity on tensor cores).
  Far-from-everything complaints collect in a "novel" pool; re-cluster the pool every few minutes → **new topics appear on their own**.
- The LLM names and summarises each topic from a sample of its members.

### [8] Burst detection — custom kernel K1 (showcase #1)
- Counts complaints into a 4-D cube: **topic × channel × region × product**, per time bucket
  (e.g. 5 min). That is easily 10⁵–10⁶ separate time series.
- **One thread per series** updates a seasonal baseline (hour-of-week expected count, plus an exponentially
  weighted moving average (EWMA) and its variance) and computes a Poisson surprise score and a CUSUM statistic, all in one pass.
- Also aggregates up the hierarchy (e.g. topic × all regions) with parallel reductions, so both local
  (one region) and national spikes are caught.
- Output: series whose score crosses the threshold → candidate incidents.

```cuda
// one thread per series s; counts[s] = complaints in the newest bucket
__global__ void burst_score(const float* counts, const float* season, float* ewma, float* ewvar,
                            float* cusum, float* score, int n_series, float alpha, float k) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_series) return;
    float expected = fmaxf(season[s] * ewma[s], 0.5f);
    float x = counts[s];
    float z = (x - expected) / sqrtf(expected + ewvar[s]);        // Poisson-ish surprise
    cusum[s] = fmaxf(0.f, cusum[s] + z - k);                       // persistent drift
    score[s] = fmaxf(z, cusum[s]);
    float d = x / fmaxf(season[s], 1e-3f) - ewma[s];
    ewma[s] += alpha * d;                                          // update baseline
    ewvar[s] = (1 - alpha) * (ewvar[s] + alpha * d * d);
}
```

### [9] Root-cause contrast mining — custom kernel K2 (showcase #2)
Question: *what is different about the complaints in this spike compared with normal?*
- Every attribute value (app_version=4.2.1, device_os=Android, channel=USSD, switch=X, region=Kano, …) is stored as
  a **bitset** over complaints in the analysis window.
- Spike set S and baseline set B are also bitsets. For every single value and **every pair** of values:
  `support_S = popc(A & B2 & S)`, `support_B = popc(A & B2 & Base)` → lift and a significance test.
- **One warp per attribute pair**, `__popc` on 32-bit words, warp-shuffle reduction. Thousands of values →
  millions of pairs scored in milliseconds; the CPU version takes minutes.
- Output: ranked explanations, e.g. *"84% of the spike is app 4.2.1 on Android (baseline 9%) → lift 9.3"*.

### [10] Event correlator
- Lines up the spike's start time with operational events: release deployed 40 min before, failed-NIP rate
  up at the same time, outage ticket open. Measures lagged correlation and checks attribute overlap with K2's output.

### [11] Systemic vs isolated score + priority
- **Systemic score:** burst significance × spread across channels × spread across regions × concentration of the root-cause attributes.
  - High → "emerging systemic issue"; low with high severity → "isolated, handle case by case".
- **Priority:** affected customers (estimated from the trend) × money at stake (sum of extracted amounts)
  × share of vulnerable customers × regulatory risk (e.g. unreversed debits past the required timeline) × social virality.

### [12] LLM incident brief
- Grounded in structured evidence only: topic summary, volume chart data, top root-cause rules, correlated
  events, sample complaints (redacted) with IDs cited.
- Output: what's happening, likely cause, who is affected, recommended owner (app team / switch ops / ATM ops /
  card ops), suggested customer message, and an "isolated vs systemic" verdict with confidence.
- Also drafts proactive customer communication, in the customer's language, to deflect repeat complaints.

### [13] API · [14] Dashboard
- `WS /stream` (new incidents, score updates), `GET /incidents`, `GET /incidents/{id}` (brief, root cause, timeline),
  `GET /topics`, `POST /ask` (natural-language Q&A over complaints, retrieval-grounded).
- Dashboard: issue radar (topics sized by volume, coloured by burst score), incident timeline with injected-incident
  markers, root-cause table, brief viewer, language/channel breakdown, CPU-vs-GPU throughput panel.

### [B] CPU baseline
- Same pipeline on CPU: sentence-transformers on CPU, sklearn HDBSCAN, pandas burst loop, Python contrast mining,
  whisper on CPU for a sample. Used for correctness and the speed story.

## 4. Tech stack
| Layer | Choice | Fallback |
|---|---|---|
| Custom kernels (K1–K3) | CUDA C++ via CuPy `RawKernel`, `sm_80` | Numba CUDA |
| Speech recognition | faster-whisper (CTranslate2) | openai-whisper |
| LLM | Open-weight instruct model on **vLLM**, JSON-constrained | Smaller model, or hosted API on synthetic data only |
| Embeddings | BGE-M3 / multilingual-E5 (FP16) | MiniLM multilingual |
| Clustering / kNN | RAPIDS cuML (UMAP, HDBSCAN), FAISS-GPU or CuPy matmul top-k | sklearn |
| Dataframes | cuDF | pandas |
| API / UI | FastAPI + React (or Streamlit) | — |
| Compute | A100 via Slurm (shared setup with MuleRadar) | Kaggle T4 |

**GPU sharing on one A100:** vLLM takes a fixed memory fraction (`--gpu-memory-utilization`); embeddings, whisper and
kernels use the rest. If available, MIG slices isolate them. The same vLLM server can serve both projects.

## 5. Benchmark methodology
- End-to-end throughput (complaints/s) and latency from complaint to incident alert, CPU vs A100.
- Per-stage times (ASR, extraction, embeddings, clustering, K1, K2, K3), so the speed-up is not hidden in one stage.
- Kernel scaling: K1 from 10⁴ → 10⁶ series; K2 from 100 → 5,000 attribute values (≈12.5M pairs).
- Same inputs and thresholds on CPU and GPU; check identical incident and root-cause outputs on a sample.

## 6. Evaluation (measurable impact)
- **Time to detect** each injected incident vs a naive "daily ticket count > threshold" rule, and how long before the peak.
- **Root-cause accuracy:** is the injected cause in the top-1 / top-3 explanations?
- **Systemic vs isolated:** precision and recall on labelled incidents vs background noise.
- **Extraction quality:** product/issue F1 on CFPB labels and on synthetic data in each language (per-language
  breakdown, to show local-language customers aren't left behind).
- Deduplication: share of social-media volume removed as reposts.
- Business proxies: analyst triage hours saved, repeat complaints avoided by proactive messages.

## 7. Repository layout (planned)
```
02-complaint-pulse/
├── data/                 # CFPB extract, synthetic stream, audio, event logs (gitignored)
├── generator/            # synthetic complaints + incident injector + TTS
├── pipeline/
│   ├── kernels/          # k1_burst.cu, k2_contrast.cu, k3_minhash.cu
│   ├── asr.py  redact.py  extract_llm.py  embed.py
│   ├── dedup.py  topics.py  burst.py  rootcause.py  correlate.py
│   ├── scoring.py  brief_llm.py
│   └── stream.py
├── baseline_cpu/
├── api/  dashboard/  bench/  notebooks/  pitch/
└── slurm/                # shared with 01 (symlink or copy)
```

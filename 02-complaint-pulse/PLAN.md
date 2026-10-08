# ComplaintPulse — Execution Plan

> Assumptions: same hackathon window, team and A100 (Slurm) as MuleRadar. Shared pieces (Slurm scripts, environment,
> vLLM server, dashboard shell) are built once and reused. Phases are ordered by dependency; durations are relative.

## Phase 0 — Setup & data (≈15%)
- [ ] Reuse the MuleRadar environment; add vLLM, faster-whisper, sentence-transformers, cuML, FAISS.
- [ ] Pre-download models and the CFPB dataset on the login node (compute nodes may have no internet).
- [ ] Build the **synthetic generator**: 6 channels, 5 languages incl. code-switching, realistic Nigerian products,
      daily/weekly seasonality, plus an **incident injector** with known start time, cause attributes and growth curve.
- [ ] Synthetic event log (releases, outages, failed-transaction rates) aligned with the injected incidents.
- [ ] TTS a few hundred complaints to audio for the speech-recognition path.
- **Exit:** a replayable 30-day stream with ≥5 labelled incidents and lots of background noise.

## Phase 1 — Understanding pipeline (≈20%)
- [ ] PII redaction.
- [ ] Batched Whisper transcription.
- [ ] vLLM structured extraction (JSON schema); measure per-language F1.
- [ ] Batched multilingual embeddings.
- **Exit:** every complaint becomes a clean, structured, embedded record at high throughput.

## Phase 2 — Custom GPU kernels (≈20%) ← main technical edge
- [ ] K1 burst detection over the topic × channel × region × product cube (+ hierarchy roll-ups).
- [ ] K2 bitset contrast mining (single values + pairs) with lift and significance.
- [ ] K3 MinHash + LSH near-duplicate detection.
- [ ] CPU reference versions; correctness checks; Nsight profiling.
- **Exit:** kernels are correct and fast, with a scaling chart.

## Phase 3 — Topics, scoring, incident briefs (≈15%)
- [ ] cuML UMAP + HDBSCAN bootstrap; streaming nearest-centroid assignment; novel-pool re-clustering.
- [ ] LLM topic naming.
- [ ] Event correlator; systemic score; priority score.
- [ ] Grounded LLM incident brief + proactive customer message in the customer's language.
- **Exit:** injected incidents appear as ranked incidents with the correct cause and a readable brief.

## Phase 4 — API + dashboard (≈15%)
- [ ] Replay streamer; FastAPI REST + WebSocket.
- [ ] Issue radar, incident timeline, root-cause table, brief viewer, language/channel view, CPU-vs-GPU panel.
- [ ] "Ask your complaints" Q&A (stretch if time allows).

## Phase 5 — Evaluation, polish, pitch (≈15%)
- [ ] Time-to-detect vs naive rule, root-cause top-k accuracy, systemic/isolated precision and recall, per-language F1.
- [ ] Benchmarks (per stage and end to end), backup demo video, slides, rehearse.

## MVP vs stretch
| MVP (must ship) | Stretch |
|---|---|
| Text channels + LLM extraction + embeddings | Speech recognition on live audio |
| K1 burst detection, K2 contrast mining | K3 MinHash dedup (fallback: exact-hash dedup) |
| Streaming topics with LLM names | "Ask your complaints" chat |
| Systemic score, priority, incident brief | Proactive customer messages in 5 languages |
| CPU-vs-GPU panel, evaluation table | Cross-link of fraud complaints to MuleRadar |

Cut order if short on time: chat → audio → K3 → proactive messages.
**Never cut:** K1 + K2, the incident brief (it's the GenAI story), and per-language results (the inclusion story).

## Judging-criteria map
| Criterion | How ComplaintPulse scores |
|---|---|
| Innovation | Custom CUDA burst and root-cause kernels + GenAI extraction and briefs; multilingual and voice-aware |
| Feasibility | Working demo on public + synthetic data; open-weight models on-premises; plugs into existing CRM/contact-centre exports |
| Impact | Issues caught hours earlier; customers who complain by voice or in local languages are counted; fewer repeat complaints |
| Scalability | Handles every complaint across all channels in real time on one GPU; extends to other banks and languages |
| Presentation | Watch an incident appear on the radar, click it, see the cause and the brief written live |

## Demo script (3 minutes)
1. **Hook (20s):** "When the app broke last time, the bank found out on Twitter. 3,000 complaints in 5 languages, 6 channels, 0 connected."
2. **Live (100s):** Replay starts; background noise across channels. An injected incident (app 4.2.1 OTP failure)
   begins — show Pidgin tweets, Hausa calls and English emails merging into one topic. K1 flags it while volume is still small;
   K2 shows "Android + v4.2.1, lift 9×"; the correlator links it to the release 40 minutes earlier. The LLM brief appears
   and is routed to the app team, along with a customer message in Pidgin.
3. **Evidence (40s):** Detected X h before the peak vs the naive rule; root cause correct in top-1 for N of M incidents; per-language F1; CPU vs A100.
4. **Close (20s):** Deployment path (reads CRM, contact-centre and social exports; on-premises), and the link to MuleRadar.

## Risks & mitigations
| Risk | Mitigation |
|---|---|
| LLM extraction too slow for the stream | vLLM continuous batching; smaller model; classifier fallback for product/issue |
| Weak quality in Hausa/Yoruba/Igbo | Few-shot prompts per language; translate-then-extract fallback; report per-language results honestly |
| Synthetic data looks too easy | Add realistic noise, typos, code-switching, overlapping incidents; also show results on real CFPB text |
| vLLM + other models compete for A100 memory | Fixed vLLM memory fraction; MIG if available; run ASR offline before the demo |
| Clustering drift / topics merging | Periodic re-clustering; minimum distance between centroids; LLM checks topic names |
| Privacy concerns | Redaction before storage and the LLM; on-premises models; no customer data leaves the bank |

## Open questions
- Is there a sponsor sandbox or sample complaint export we can use?
- Do the judges expect voice/call-centre support, or is text enough?
- Which local languages matter most to the sponsor bank's customers?

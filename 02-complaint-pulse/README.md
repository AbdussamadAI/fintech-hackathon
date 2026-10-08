# ComplaintPulse — GPU-Accelerated, Multilingual Complaint Intelligence

**Challenge:** #2 AI-Powered Transaction Investigation and Resolution
**One-liner:** ComplaintPulse listens to every complaint channel (calls, chat, email, social, app reviews, branch
forms) in English, Pidgin, Hausa, Yoruba and Igbo, spots emerging systemic issues hours before they peak,
pinpoints the likely root cause, and hands the owning team a ready-to-act incident brief.

## The problem
- Complaints are scattered across branches, contact centres, email, social media and digital channels, in
  several languages, and stored in separate systems.
- Banks see ticket volume, not causes: they can't tell one angry customer from the start of an outage, so
  problems are found late, after they have gone viral or reached the regulator.
- Customers who complain by voice, in local languages, or through agents and USSD (often the newly banked)
  are the least visible in text-only analytics.

## Our approach
1. **Unify** every channel into one stream; transcribe calls with GPU speech recognition; redact personal data.
2. **Understand** each complaint with GenAI: product, issue, amount, transaction reference, sentiment, urgency, in any of 5 languages.
3. **Group** complaints into live topics using GPU embeddings, clustering and near-duplicate removal.
4. **Detect** emerging issues with a custom CUDA kernel that watches thousands of topic × channel × region series at once.
5. **Explain** the root cause with a custom CUDA kernel that finds which attributes (app version, device, channel,
   partner switch, location) are over-represented in the spike, linked to operational events.
6. **Prioritise & act:** a "systemic vs isolated" score, an impact-based priority, and an LLM-written incident brief routed to the owning team.

## Docs
- [PLAN.md](PLAN.md) — phases, MVP vs stretch, demo script, judging map, risks
- [ARCHITECTURE.md](ARCHITECTURE.md) — system design, data, kernels, GenAI components, benchmarks
- [../SUBMISSION_REQUIREMENTS.md](../SUBMISSION_REQUIREMENTS.md) — required submission sections + outline

## Status
- [ ] Planning (this folder)
- [ ] Synthetic data + incident injector
- [ ] GPU understanding pipeline (ASR, extraction, embeddings)
- [ ] Detection + root-cause kernels
- [ ] Dashboard + incident briefs
- [ ] Evaluation + pitch

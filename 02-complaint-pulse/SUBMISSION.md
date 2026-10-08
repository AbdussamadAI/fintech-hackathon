# ComplaintPulse: hearing every customer, in every language, before the problem goes viral

**Challenge #2: AI-Powered Transaction Investigation and Resolution**

> ComplaintPulse listens to every complaint channel (call centre, app chat, social media, email, app reviews,
> branch forms) in English, Pidgin, Hausa, Yoruba and Igbo. It spots an emerging systemic problem hours before it
> peaks, names the likely root cause and the operational event behind it, and hands the owning team a ready-to-act
> brief, written by Nigeria's own language model, N-ATLaS, with a message to affected customers in their own language.

---

## 1. Problem & Opportunity

**Nigerian banks count complaints; they don't hear them.**

- Complaints escalated to the Central Bank keep rising: **23,129 in 2025, up from 20,925 in 2024**, and those are only
  the cases customers took to the regulator. CBN fined banks **₦1.69 billion in 2025**, including **₦430 million**
  specifically for ignoring customer complaints, and has forced back **₦115 billion** of failed-transaction money over
  ten years.
- The underlying causes are operational: failed transfers that are not reversed, switch and acquirer outages, ATM
  faults, USSD and OTP failures after app or telco changes. One fault produces thousands of complaints across
  channels, but each channel's team sees only its own slice.
- Banks monitor **ticket counts per day**. That rule fires a day late, raises many false alarms, and never says why.
- Customers who complain **in Hausa, Yoruba, Igbo or Pidgin, or by phone at a branch**, are the least visible in
  English-only analytics, and they are often the newly banked whose trust is easiest to lose.

**The opportunity:** treat every complaint, in every language, as a live sensor of the bank's operations, so the
bank fixes the cause within hours and tells affected customers before they complain again.

## 2. Innovative Solution

```
all channels ─▶ PII redaction ─▶ N-ATLaS understanding ─▶ topics ─▶ K3 repost dedup ─▶ K1 burst cube ─▶ K2 root cause ─▶ event link ─▶ N-ATLaS brief
(5 languages)    (BVN, phone,      (issue, English summary,  (BGE-M3    (MinHash, GPU)     (33,516 series,   (all attribute    (releases,      + customer message
                  account, amount)  amount, urgency)          embeddings)                  GPU)              pairs, GPU)       outages)        in their language
```

1. **Privacy first.** Account numbers, BVN/NIN, phone numbers, card numbers and amounts are masked before any text is
   stored, embedded or shown to a model.
2. **Understanding every language with N-ATLaS.** Nigeria's national multilingual LLM (Llama-3 8B, trained on Hausa,
   Igbo, Yoruba and English) reads each complaint and returns structured JSON: issue, English summary, amount,
   urgency, language. It runs **on the bank's own GPU**; no complaint leaves the bank.
3. **Topics that ignore language.** Complaints are grouped by meaning with multilingual embeddings (BGE-M3), so new
   kinds of problems form new topics on their own.
4. **Three custom CUDA kernels:**
   - **K3 repost detection (MinHash + LSH):** a viral post copied 900 times counts once, not 900 times.
   - **K1 burst detection over a complaint cube:** topic × channel × state × customer segment, with every roll-up:
     33,516 time series scored together every 15 minutes (seasonal baseline, Poisson surprise, CUSUM drift). It catches
     a spike in one state as well as a national one.
   - **K2 root-cause mining:** every attribute value (app version, device, payment switch, telco, ATM, card scheme,
     state, channel, segment, language) and **every pair of values** is scored by how much of the spike's *excess*
     it explains: "app 4.2.1 on Android explains 95% of the extra complaints, lift 25×".
5. **Event correlation:** the spike is matched to operational events (app releases, switch degradation, maintenance)
   that happened shortly before it.
6. **Systemic or not:** a spike without a concentrated cause (a salary-week volume rise, a random blip) is checked and
   **not escalated**. A concentrated one becomes an incident with an owner (app team, switch operations, ATM
   operations, card operations, billing, fraud).
7. **GenAI brief and customer message (N-ATLaS).** For each incident, N-ATLaS writes a brief for the owning team from
   the evidence only, citing complaint IDs; a validator rejects any citation not in the evidence. It also drafts a
   proactive message to affected customers **in Hausa, Pidgin, Yoruba, Igbo or English**.

## 3. Impact & Value (measured)

We built a 30-day synthetic stream of **249,553 complaints** modelled on a large Nigerian bank (diurnal and weekly
patterns, salary-week surge, 37 states, 6 channels, 5 languages with code-switching), with **8 injected incidents**
of known cause and timing, plus decoys: unrelated operational events, isolated high-value complaints and a viral
repost storm. Real complaint data is confidential; the generator lets us measure against ground truth. To guard
against tuning to one dataset, every result is also reported on a **held-out stream** (new random seed, incident
times shifted by up to ±1.5 days, no retuning).

| Result | ComplaintPulse | Daily-count rule (what banks use) |
|---|---|---|
| Incidents detected | **8 / 8** (held-out **8 / 8**) | 8 / 8 |
| Median time from onset to alert | **4.6 h** (held-out **5.8 h**) | 18.6 h (held-out 22.6 h) |
| Alert before the incident peaks | **8 / 8** (held-out **8 / 8**), median 8.9 h early | 2 / 8 (held-out 0 / 8) |
| False alarms | **0** (held-out **0**) | 8 of 19 alerts (held-out 7 of 17) |
| Root cause ranked first | **8 / 8** (held-out **8 / 8**) | none given |
| Correct operational event linked | **8 / 8** (held-out **8 / 8**) | none given |

- **Repost storm:** K3 collapsed 895 of the 900 viral copies. Without it, the storm raises a false incident.
- **Salary-week surge and other blips:** 6 volume rises were checked and correctly not escalated.
- **Fully unsupervised topics** (no labels at all, topics discovered from embeddings): on the held-out stream
  **8 of 8 incidents, 0 false alarms**, root cause and event correct for all 8 (the daily-count rule on the same topics:
  35 false alarms out of 57 alerts). On the main stream 7 of 8 with 1 false alarm; the miss is the smallest incident,
  a fraud campaign, which the language finding below explains.

**GPU acceleration (A100 vs the same algorithms on CPU, outputs identical):**

| Kernel | Scale | CPU | GPU | Speed-up |
|---|---|---|---|---|
| K1 burst scoring | 2.1 million series | 51.7 ms | 0.16 ms | **328×** |
| K2 root-cause mining | 1.1 million attribute pairs | 6.35 s | 2.9 ms | **2,180×** |
| K3 MinHash | 200,000 posts | 1.27 s | 6.7 ms | **189×** |

K2 is what makes it possible to test every pair of attributes for every alert, across every topic, in real time.

**Inclusion finding.** Multilingual embeddings alone group complaints by language as well as by problem: only
17–37% of Hausa, Pidgin, Igbo and Yoruba complaints landed in the same topic as English complaints about the same
problem. That silently splits local-language customers' signal. Normalising each complaint with N-ATLaS first fixes
this: __LLM_TOPIC_RESULT__.

**N-ATLaS understanding, per language:** __LLM_TABLE__

## 4. Target Users & Implementation

**Who benefits.** Customer-experience and complaints teams, IT and operations incident teams, and compliance
(CBN complaint reporting); then microfinance banks, payment service banks and fintechs. Customers benefit directly:
problems are fixed hours earlier, refunds start sooner, and they hear from the bank in their own language before
they need to complain again.

**How it deploys.**
- Reads exports or streams the bank already has: CRM tickets, contact-centre logs, email, social listening, app-store
  reviews, branch forms, plus the change and incident calendar.
- Runs on **one on-premises GPU server**, with open-weight models (N-ATLaS, BGE-M3). No customer data leaves the bank.
- Sends incidents to the existing service-management tool with an owner, a brief and a customer message for approval.

**Rollout.** (1) Back-test on 6–12 months of historical complaints and past incidents. (2) Live dashboard for one
channel. (3) All channels with routed incidents and approved customer messages.

**Feasibility today.** The full pipeline runs end to end in our prototype on an A100: generator, redaction, N-ATLaS
and BGE-M3 served on-premises with llama.cpp, the three CUDA kernels with CPU reference implementations, detection,
evaluation, and a live console that replays the complaint stream and writes briefs on demand.

**Business model.** Annual licence per institution tiered by complaint volume. N-ATLaS is free for up to 1,000 active
users, which covers a bank's complaints and operations staff; larger deployments take a commercial licence from its
publishers.

**Link to MuleRadar.** "Unauthorised debit" incidents (like the fraud campaign in our test stream) are passed to
MuleRadar as seed accounts for mule-network analysis.

## 5. Team & Expertise

| Member | Role | Contribution |
|---|---|---|
| **Abdussamad Ali** | CEO | Leads the venture and the AI and GPU build: detection kernels, evaluation, N-ATLaS integration; owns bank partnerships. |
| **Abubakar Abdullahi** | CTO | Architecture and deployment: ingestion from CRM, contact-centre and social feeds, on-premises GPU serving, security and PII redaction. |
| **Umar Mohammed Bashir** | Product Manager | Turns incidents into workflow: ownership routing, CBN complaint-handling requirements, customer-message approval, pilot plan. |
| **Abdulbasit Idris Ali** | UI Designer | Designs the console: issue radar, root-cause view, multilingual complaint feed, brief and message review. |

---

### Sources
- CBN complaints and fines 2024–2025: [Prime Business Africa](https://www.primebusiness.africa/cbn-hits-banks-with-n1-69bn-in-fines-as-consumer-complaints-hit-23129-in-2025/), [Nairametrics](https://nairametrics.com/2026/07/29/cbn-slams-banks-with-n430-million-fines-for-ignoring-customer-complaints/)
- Failed-transaction refunds over 10 years: [Blueprint](https://blueprint.ng/cbn-returns-n115-45bn-22-7mn-in-failed-transactions-to-bank-customers-in-10-years/)
- N-ATLaS model card and terms: [NCAIR1/N-ATLaS](https://huggingface.co/NCAIR1/N-ATLaS)

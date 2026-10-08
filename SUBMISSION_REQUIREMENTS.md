# Submission Requirements (applies to both projects)

## Required sections
1. **Problem & Opportunity** — clearly define the challenge and why it matters.
2. **Innovative Solution** — the idea, how it works, and how it uses **AI, GenAI, or emerging tech** to solve the problem.
3. **Impact & Value** — what makes it unique and the **measurable** benefits for users or the financial ecosystem.
4. **Target Users & Implementation** — who benefits, and a feasible approach for building or deploying it.
5. **Team & Expertise** — team members, their roles, and how their skills contribute to success.

## Judging criteria (for reference)
Innovation & Creativity · Feasibility · Impact (financial inclusion) · Scalability · Presentation

## How each section maps to the judging criteria
| Submission section | Criteria it mainly feeds |
|---|---|
| Problem & Opportunity | Impact, Presentation |
| Innovative Solution | Innovation, Feasibility |
| Impact & Value | Impact, Scalability |
| Target Users & Implementation | Feasibility, Scalability |
| Team & Expertise | Feasibility, Presentation |

---

## 01 MuleRadar — draft outline per section

### 1. Problem & Opportunity
- Fraud proceeds leave through mule-account chains within minutes, while batch AML runs overnight and looks at accounts one at a time.
- Nigerian banks have faced penalties for weak AML controls, and CBN and NFIU are tightening oversight.
- Mule recruiters target students and newly banked users, and blunt rules freeze innocent new customers.
- *Numbers to find:* NIBSS fraud losses for the latest year, recent AML fines, mule-recruitment cases.

### 2. Innovative Solution
- A real-time transaction graph that lives on the GPU, with custom CUDA kernels that detect laundering patterns (fan-in/out, pass-through, chains, cycles).
- GPU graph analytics to expose whole rings, and AI scoring (XGBoost, with an optional GNN) plus reason codes.
- **GenAI:** an on-premises LLM on the A100 drafts Suspicious Transaction Report (STR) narratives and answers analyst questions ("why is this ring flagged?"), grounded only in the detected evidence.
- **Emerging tech:** GPU-accelerated streaming graph analytics.

### 3. Impact & Value (measurable)
- Time from the first laundering transfer to a ring alert, in minutes versus the next-day batch.
- Precision and recall against labelled data, compared with naive rules.
- Fewer innocent accounts frozen per 10,000 customers (the inclusion metric).
- Throughput (transactions per second) and the speed-up over CPU, plus the cost per million transactions.
- Analyst hours saved per STR thanks to drafted narratives.

### 4. Target Users & Implementation
- **Users:** bank fraud/AML teams and compliance officers, then tier-2 banks, microfinance banks and fintechs. Customers benefit indirectly through fewer wrongful freezes.
- **Deployment:** runs alongside the core banking system and the NIP transaction feed, reading a copy of transactions without changing core systems. It runs on one GPU server on-premises, so no customer data leaves the bank.
- **Rollout:** shadow mode (alerts only) → analyst-in-the-loop holds → automated tier-2 step-up verification.

### 5. Team & Expertise
- *To fill in:* names, roles (GPU engineer, ML/data, full-stack, domain/pitch), and one line each on the relevant skills and experience.

---

## 02 ComplaintPulse — draft outline per section

### 1. Problem & Opportunity
- Complaints are spread across branches, contact centres, email, social media and apps, in several languages, and stored in separate systems.
- Banks count tickets instead of finding causes, so systemic issues (app bugs, delayed reversals, ATM faults) are found late, after they go viral or reach the regulator.
- Customers who complain by voice or in local languages, often the newly banked, are invisible to text-only English analytics.
- *Numbers to find:* CBN consumer-complaint statistics, recent outage incidents that went viral.

### 2. Innovative Solution
- **GenAI:** an on-premises LLM extracts structured facts from complaints in 5 languages (including Pidgin and code-switching), names topics, and writes grounded incident briefs and customer messages.
- **AI:** multilingual embeddings, streaming topic clustering, speech recognition for call-centre audio.
- **Emerging tech / GPU:** custom CUDA kernels that score a million complaint time series for spikes at once (K1) and mine millions of attribute combinations for the root cause (K2), plus near-duplicate detection for viral reposts (K3).

### 3. Impact & Value (measurable)
- Hours of earlier detection vs a daily ticket-count rule; detection before the peak.
- Root cause correctly identified in the top 1 / top 3 explanations.
- Precision and recall for telling systemic issues from isolated ones.
- Per-language extraction F1, showing local-language and voice customers are counted.
- Analyst triage hours saved; repeat complaints avoided through proactive messages.

### 4. Target Users & Implementation
- **Users:** customer-experience and complaints teams, operations and IT incident teams, and compliance (CBN complaint reporting). Customers benefit from faster fixes and proactive updates in their own language.
- **Deployment:** reads exports or streams from the CRM, contact-centre recordings, email, social listening and app-store reviews. Runs on one on-premises GPU server with open-weight models, and redacts personal data first.
- **Rollout:** historical back-test → live dashboard for one channel → all channels with routed incidents.
- Synergy: "unauthorised debit" complaints feed MuleRadar as fraud seeds.

### 5. Team & Expertise
- *Shared with MuleRadar — to fill in.*

# Fintech Hackathon: MuleRadar & ComplaintPulse

Two GPU-accelerated AI systems for Nigerian banks, built by Abdussamad Ali (CEO), Abubakar Abdullahi (CTO),
Umar Mohammed Bashir (Product Manager) and Abdulbasit Idris Ali (UI Designer).

| Project | Challenge | What it does | Submission |
|---|---|---|---|
| [**MuleRadar**](01-mule-radar/) | #5 Combating Digital Fraud and Financial Crime | Detects money-mule rings in the live transfer stream with custom CUDA kernels; 66.7% F1 on the IBM AML benchmark (best published 64.8%), 99.6% fewer innocent accounts flagged than a threshold rule, 1.28M transfers/s on one A100 | [PDF](01-mule-radar/report/MuleRadar_Submission.pdf) |
| [**ComplaintPulse**](02-complaint-pulse/) | #2 AI-Powered Transaction Investigation and Resolution | Hears every complaint channel in English, Pidgin, Hausa, Yoruba and Igbo; detects emerging incidents ~14 h earlier than a daily-count rule with 0 false alarms, names the root cause and operational event, and writes briefs and customer messages with N-ATLaS, Nigeria's multilingual LLM | [PDF](02-complaint-pulse/report/ComplaintPulse_Submission.pdf) |

Each project folder has its own README (how to run), `SUBMISSION.md` (the five required sections) and `report/`
(the formatted submission). Data, model weights and large intermediate files are not stored in the repository;
the scripts regenerate or download them.

**Hardware used:** NVIDIA A100 80 GB via Slurm (`slurm-notebook.sh` opens a GPU job and an SSH tunnel).

"""GenAI analyst copilot: draft a Suspicious Transaction Report (STR) narrative for a ring.

Grounding contract: the model only sees the structured evidence built by MuleRadar (members, transactions
with IDs, timeline, reason codes) and must cite transaction IDs; `validate()` rejects any draft that
mentions a transaction ID or account that is not in the evidence. The LLM never decides hold/clear.

Backends (first available wins):
  * OpenAI-compatible endpoint, e.g. vLLM serving an open-weight model on the bank's own GPU:
        LLM_BASE_URL=http://gpu-node:8000/v1  LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
  * Anthropic API (only for synthetic/demo data):  ANTHROPIC_API_KEY=...  LLM_MODEL=claude-sonnet-5
  * Deterministic template (no LLM): always grounded, used as fallback
"""
import json
import os
import re
import urllib.request

SYSTEM = (
    "You are an AML analyst assistant at a Nigerian bank. Draft the narrative section of a Suspicious "
    "Transaction Report for the NFIU using ONLY the evidence JSON provided. Cite transaction IDs in square "
    "brackets, e.g. [T123]. Do not invent facts, names, amounts or IDs. Structure: 1) Summary, 2) Parties, "
    "3) Pattern and timeline, 4) Red flags, 5) Action taken (as given in the evidence). Under 300 words."
)


def _post(url: str, payload: dict, headers: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def llm_draft(evidence: dict) -> tuple[str, str] | None:
    user = "Evidence JSON:\n" + json.dumps(evidence, indent=1)
    if os.environ.get("LLM_BASE_URL"):
        model = os.environ.get("LLM_MODEL", "default")
        out = _post(os.environ["LLM_BASE_URL"].rstrip("/") + "/chat/completions",
                    {"model": model, "temperature": 0.2, "max_tokens": 700,
                     "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]},
                    {"Content-Type": "application/json"})
        return out["choices"][0]["message"]["content"], model
    if os.environ.get("ANTHROPIC_API_KEY"):
        model = os.environ.get("LLM_MODEL", "claude-sonnet-5")
        out = _post("https://api.anthropic.com/v1/messages",
                    {"model": model, "max_tokens": 700, "system": SYSTEM,
                     "messages": [{"role": "user", "content": user}]},
                    {"Content-Type": "application/json", "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                     "anthropic-version": "2023-06-01"})
        return "".join(b.get("text", "") for b in out["content"]), model
    return None


def template_draft(ev: dict) -> str:
    tl = ev["timeline"]
    lines = [
        f"1) Summary. MuleRadar flagged ring {ev['ring_id']}: {ev['n_accounts']} accounts exchanged "
        f"{ev['n_transactions']} transactions worth USD {ev['total_usd']:,.0f} (≈ NGN {ev['total_ngn']:,.0f}) "
        f"between {ev['first_seen']} and {ev['last_seen']}. Ring risk score {ev['ring_score']:.2f}.",
        "2) Parties. " + "; ".join(f"{a['account']} ({a['role']}, bank {a['bank']}, {a['entity_type'] or 'unknown type'})"
                                     for a in ev["accounts"][:8]) + ("; …" if len(ev["accounts"]) > 8 else "."),
        "3) Pattern and timeline. " + " ".join(
            f"{t['time']}: {t['from']} → {t['to']} USD {t['usd']:,.0f} via {t['format']} [{t['txn']}]." for t in tl[:8])
        + (" …" if len(tl) > 8 else ""),
        "4) Red flags. " + " ".join(f"{r}." for r in ev["reason_codes"]),
        f"5) Action taken. {ev['action']}",
    ]
    return "\n".join(lines)


def validate(text: str, ev: dict) -> list[str]:
    """Return problems: cited txn IDs / accounts that are not in the evidence."""
    known_tx = {t["txn"] for t in ev["timeline"]}
    known_acct = {a["account"] for a in ev["accounts"]}
    bad = [f"unknown txn {t}" for t in re.findall(r"T\d+", text) if t not in known_tx]
    bad += [f"unknown account {a}" for a in re.findall(r"\b8[0-9A-F]{8}\b", text) if a not in known_acct]
    return bad


def draft(ev: dict) -> dict:
    try:
        res = llm_draft(ev)
    except Exception as e:   # network / endpoint failure -> grounded template
        res, err = None, str(e)
    else:
        err = None
    if res:
        text, model = res
        problems = validate(text, ev)
        if not problems:
            return {"text": text, "backend": model, "validated": True}
        err = "; ".join(problems[:5])
    return {"text": template_draft(ev), "backend": "template", "validated": True,
            **({"llm_error": err} if err else {})}

"""GenAI layer on N-ATLaS (Nigeria's multilingual LLM) + BGE-M3 embeddings, both served by llama.cpp on-premises.

  extract()        complaint -> JSON (issue, product, amount, sentiment, urgency, language, English summary),
                   constrained by a JSON schema so output always parses
  paraphrase()     rewrite a complaint freely in a given language (builds a harder, LLM-written test set)
  incident_brief() grounded brief for the owning team; every claim must cite complaint IDs from the evidence
  customer_message() proactive message to affected customers in their own language
  embed()          multilingual sentence embeddings (BGE-M3)

Servers (OpenAI-compatible llama-server):
  LLM_URL=http://127.0.0.1:8081   (N-ATLaS)     EMB_URL=http://127.0.0.1:8082   (bge-m3, --embedding)
"""
import json
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor

LLM_URL = os.environ.get("LLM_URL", "http://127.0.0.1:8081")
EMB_URL = os.environ.get("EMB_URL", "http://127.0.0.1:8082")
ISSUES = ["transfer_debited_not_received", "reversal_delayed", "wrong_beneficiary", "ussd_session_timeout",
          "ussd_wrong_charges", "atm_debited_no_cash", "card_declined", "card_blocked", "pos_debited_no_value",
          "app_login_failure", "otp_not_received", "app_crash", "bvn_nin_update", "account_restricted",
          "unauthorised_debit", "sms_charges", "maintenance_fee", "loan_deduction", "long_wait_time", "staff_attitude"]
LANGS = ["English", "Pidgin", "Hausa", "Yoruba", "Igbo"]
ISSUE_HELP = {
    "transfer_debited_not_received": "transfer debited but beneficiary did not receive", "reversal_delayed": "failed transaction not yet reversed/refunded",
    "wrong_beneficiary": "money sent to wrong account", "ussd_session_timeout": "USSD code times out / session ended",
    "ussd_wrong_charges": "charged for failed USSD", "atm_debited_no_cash": "ATM debited without dispensing cash",
    "card_declined": "card declined", "card_blocked": "card blocked", "pos_debited_no_value": "POS payment debited but merchant not credited",
    "app_login_failure": "cannot log in to mobile app", "otp_not_received": "OTP/token not received", "app_crash": "app crashes/closes",
    "bvn_nin_update": "BVN/NIN/KYC update problem", "account_restricted": "account restricted/frozen",
    "unauthorised_debit": "debit the customer did not authorise (possible fraud)", "sms_charges": "SMS alert charges",
    "maintenance_fee": "account maintenance fee", "loan_deduction": "wrong loan deduction", "long_wait_time": "long wait / no response",
    "staff_attitude": "rude or unhelpful staff",
}
EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "issue": {"type": "string", "enum": ISSUES + ["other"]},
        "amount_naira": {"type": ["number", "null"]},
        "sentiment": {"type": "string", "enum": ["very_negative", "negative", "neutral"]},
        "urgency": {"type": "string", "enum": ["high", "medium", "low"]},
        "language": {"type": "string", "enum": LANGS},
        "summary_en": {"type": "string", "maxLength": 160},
    },
    "required": ["issue", "amount_naira", "sentiment", "urgency", "language", "summary_en"],
}


def _post(url, payload, timeout=180):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def chat(messages, max_tokens=400, temperature=0.1, schema=None):
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    if schema:
        body["response_format"] = {"type": "json_object", "schema": schema}
    out = _post(f"{LLM_URL}/v1/chat/completions", body)
    return out["choices"][0]["message"]["content"]


def pmap(fn, items, workers=16):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))


# --------------------------------------------------------------------------------------------- extraction
EXTRACT_SYS = ("You read Nigerian bank customer complaints written in English, Nigerian Pidgin, Hausa, Yoruba or Igbo "
               "(often mixed). Return JSON only. Choose the single best issue code:\n" +
               "\n".join(f"- {k}: {v}" for k, v in ISSUE_HELP.items()) +
               "\nUse amount_naira for the money amount mentioned (e.g. '50k' = 50000), or null. "
               "language is the main language of the text. summary_en is a short English summary.")


def extract_one(text):
    try:
        raw = chat([{"role": "system", "content": EXTRACT_SYS}, {"role": "user", "content": text}],
                   max_tokens=160, temperature=0.0, schema=EXTRACT_SCHEMA)
        return json.loads(raw)
    except Exception as e:  # keep the batch going; the record is marked as failed
        return {"error": str(e)[:200]}


def extract(texts, workers=16):
    return pmap(extract_one, texts, workers)


# --------------------------------------------------------------------------------------------- paraphrase
def paraphrase_one(args):
    text, lang = args
    style = {"Pidgin": "Nigerian Pidgin", "Hausa": "Hausa", "Yoruba": "Yoruba", "Igbo": "Igbo", "English": "Nigerian English"}[lang]
    try:
        return chat([{"role": "system", "content": "You rewrite bank customer complaints the way a real Nigerian customer would "
                                                   "write them: different wording, natural and informal, same facts and amount. "
                                                   "Output only the rewritten complaint."},
                     {"role": "user", "content": f"Rewrite this complaint in {style}. Keep the same problem and amount.\n\n{text}"}],
                    max_tokens=120, temperature=0.9).strip().strip('"')
    except Exception as e:
        return f"__error__ {e}"


def paraphrase(pairs, workers=16):
    return pmap(paraphrase_one, pairs, workers)


# --------------------------------------------------------------------------------------------- briefs
BRIEF_SYS = ("You are a bank operations analyst. Write an incident brief for the owning team using ONLY the evidence JSON. "
             "Cite complaint IDs in square brackets like [C123] for every claim about customers. Do not invent numbers, "
             "names or causes. Sections: Summary; Likely cause (with lift numbers); Who is affected; Recommended owner and "
             "first actions; Systemic or isolated. Under 220 words.")


def incident_brief(evidence: dict):
    text = chat([{"role": "system", "content": BRIEF_SYS}, {"role": "user", "content": json.dumps(evidence, ensure_ascii=False)}],
                max_tokens=450, temperature=0.2)
    known = {f"C{c['id']}" for c in evidence.get("sample_complaints", [])}
    cited = set(re.findall(r"C\d+", text))
    return {"text": text, "cited": sorted(cited), "uncited_or_unknown": sorted(cited - known), "grounded": cited <= known and bool(cited)}


def customer_message(evidence: dict, language: str):
    style = {"Pidgin": "Nigerian Pidgin", "Hausa": "Hausa", "Yoruba": "Yoruba", "Igbo": "Igbo", "English": "English"}[language]
    return chat([{"role": "system", "content": "You write short, warm SMS/app notifications from a Nigerian bank to customers. "
                                               "Acknowledge the problem, say it is being fixed, say money will be refunded where "
                                               "relevant, and give no false promises or exact times. Max 300 characters."},
                 {"role": "user", "content": f"Write the notification in {style}. Facts: {json.dumps(evidence['customer_facts'], ensure_ascii=False)}"}],
                max_tokens=160, temperature=0.4).strip()


# --------------------------------------------------------------------------------------------- embeddings
def embed(texts, batch=128, workers=8):
    """Batches are sent concurrently so every server slot stays busy."""
    def one(i):
        r = _post(f"{EMB_URL}/v1/embeddings", {"input": texts[i:i + batch]}, timeout=600)
        return [d["embedding"] for d in sorted(r["data"], key=lambda d: d["index"])]
    out = []
    for chunk in pmap(one, list(range(0, len(texts), batch)), workers):
        out.extend(chunk)
    return out


def health():
    s = {}
    for name, url in (("llm", LLM_URL), ("emb", EMB_URL)):
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=5) as r:
                s[name] = json.loads(r.read()).get("status", "ok")
        except Exception as e:
            s[name] = f"down ({e.__class__.__name__})"
    return s

"""GenAI layer on N-ATLaS (Nigeria's multilingual LLM) + BGE-M3 embeddings, both served by llama.cpp on-premises.

  extract()        complaint -> JSON (issue, product, amount, sentiment, urgency, language, English summary),
                   constrained by a JSON schema so output always parses
  paraphrase()     rewrite a complaint freely in a given language (builds a harder, LLM-written test set)
  incident_brief() grounded brief for the owning team; every claim must cite complaint IDs from the evidence
  customer_message() proactive message to affected customers in their own language
  embed()          multilingual sentence embeddings (BGE-M3)

Servers (OpenAI-compatible llama-server):
  LLM_URL=http://127.0.0.1:18081  (N-ATLaS)     EMB_URL=http://127.0.0.1:8082   (bge-m3, --embedding)
"""
import json
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor

LLM_URLS = os.environ.get("LLM_URL", "http://127.0.0.1:18081").split(",")   # comma-separated list = round-robin
LLM_URL = LLM_URLS[0]
_rr = __import__("itertools").count()
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
    if temperature == 0.0:      # deterministic: skip the full-vocabulary (128k) sampler chain, which is CPU-bound
        body.update(samplers=["top_k"], top_k=1)
    if schema:
        body["response_format"] = {"type": "json_object", "schema": schema}
    url = LLM_URLS[next(_rr) % len(LLM_URLS)]          # round-robin over N-ATLaS server processes
    out = _post(f"{url}/v1/chat/completions", body)
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


FAST = os.environ.get("EXTRACT_MODE", "fast") == "fast"


def _coerce(d):
    """Validate a parsed extraction against the schema enums; unknown values become 'other'/None."""
    out = {"issue": d.get("issue") if d.get("issue") in ISSUES else "other",
           "language": d.get("language") if d.get("language") in LANGS else None,
           "sentiment": d.get("sentiment"), "urgency": d.get("urgency"), "summary_en": str(d.get("summary_en", ""))[:200]}
    a = d.get("amount_naira")
    try:
        out["amount_naira"] = float(str(a).replace(",", "")) if a not in (None, "", "null") else None
    except ValueError:
        out["amount_naira"] = None
    return out


def extract_one(text, fast=None):
    """fast: plain prompt + tolerant JSON parse (grammar-constrained decoding is CPU-bound in llama.cpp);
    falls back to schema-constrained decoding only when the fast output does not parse."""
    fast = FAST if fast is None else fast
    msgs = [{"role": "system", "content": EXTRACT_SYS + " Reply with one JSON object with keys issue, amount_naira, "
             "sentiment, urgency, language, summary_en and nothing else."}, {"role": "user", "content": text}]
    try:
        if fast:
            raw = chat(msgs, max_tokens=120, temperature=0.0)
            m = re.search(r"\{.*\}", raw, re.S)
            if m:
                try:
                    return _coerce(json.loads(m.group(0)))
                except json.JSONDecodeError:
                    pass
        raw = chat(msgs, max_tokens=160, temperature=0.0, schema=EXTRACT_SCHEMA)
        return _coerce(json.loads(raw))
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
             "Support claims about customers by citing at least two of the listed complaint IDs in square brackets, exactly as "
             "written in 'citable_ids' (for example [{ex}]). Never invent IDs, numbers, names or causes. Use the recommended "
             "owner given in the evidence. Sections: Summary; Likely cause (with lift numbers); Who is affected; Recommended "
             "owner and first actions; Systemic or isolated. Under 220 words.")


def incident_brief(evidence: dict):
    known = [f"C{c['id']}" for c in evidence.get("sample_complaints", [])]
    ev = {**evidence, "citable_ids": known,
          "sample_complaints": [{**c, "id": f"C{c['id']}"} for c in evidence.get("sample_complaints", [])]}
    msgs = [{"role": "system", "content": BRIEF_SYS.format(ex=known[0] if known else "C0")},
            {"role": "user", "content": json.dumps(ev, ensure_ascii=False)}]
    attempts = []
    for _ in range(3):                      # generate, check every number against the evidence, send back errors
        text = chat(msgs, max_tokens=450, temperature=0.2)
        chk = check_grounding(text, evidence)
        attempts.append({"unsupported_numbers": chk["unsupported_numbers"], "unknown_ids": chk["unknown_ids"]})
        if chk["grounded"]:
            break
        problems = ", ".join([f"{n:,.2f}".rstrip("0").rstrip(".") for n in chk["unsupported_numbers"]] + chk["unknown_ids"])
        msgs += [{"role": "assistant", "content": text},
                 {"role": "user", "content": f"These values are not in the evidence: {problems}. Rewrite the brief using only "
                                             "numbers and IDs exactly as they appear in the evidence JSON (amounts in naira "
                                             "as given in money_at_stake_naira)."}]
    return {"text": text, "evidence_ids": known, "attempts": attempts, **chk}


def _numbers(s):
    return [float(x.replace(",", "")) for x in re.findall(r"(?<![\d.])\d[\d,]*(?:\.\d+)?(?![\w.]*\d\.\d)", s)]


def check_grounding(text, evidence):
    """Every number in the brief must be supported by the evidence (allowing %, rounding and N/₦ scaling);
    any cited complaint ID must be one of the evidence IDs."""
    ev_nums = set()
    for v in _numbers(json.dumps(evidence, ensure_ascii=False)):
        for x in (v, v * 100, v / 1e6, v / 1e3):
            ev_nums.add(round(x, 1)); ev_nums.add(round(x)); ev_nums.add(round(x, 2))
    ignore = {float(i) for i in range(0, 11)}                      # list numbering, small counts like "2 channels"
    near = lambda n: n >= 100 and any(abs(n - e) <= 0.01 * e for e in ev_nums if e >= 100)  # noqa: E731  (rounded amounts)
    unsupported = sorted({n for n in _numbers(text) if n not in ignore and round(n, 2) not in ev_nums
                          and round(n, 1) not in ev_nums and round(n) not in ev_nums and not near(n)})
    ids = {f"C{c['id']}" for c in evidence.get("sample_complaints", [])}
    bad_ids = sorted(set(re.findall(r"\bC\d+\b", text)) - ids)
    return {"unsupported_numbers": unsupported, "unknown_ids": bad_ids, "grounded": not unsupported and not bad_ids}


MESSAGE_LANG = {"English": "English", "Hausa": "Hausa", "Yoruba": "Yoruba", "Igbo": "Igbo",
                "Pidgin": "simple Nigerian English"}   # N-ATLaS is not trained on Pidgin; its Pidgin output was unusable


def customer_message(evidence: dict, language: str):
    style = MESSAGE_LANG[language]
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
    for name, url in [(f"llm{i}", u) for i, u in enumerate(LLM_URLS)] + [("emb", EMB_URL)]:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=5) as r:
                s[name] = json.loads(r.read()).get("status", "ok")
        except Exception as e:
            s[name] = f"down ({e.__class__.__name__})"
    return s

"""N-ATLaS evaluation + GenAI outputs.

  1. stratified sample of background complaints (n per language)
  2. N-ATLaS rewrites each one freely in the same language -> harder, LLM-written variants
  3. N-ATLaS structured extraction on both sets -> per-language issue accuracy / macro-F1, amount accuracy, language ID
  4. grounded incident brief + customer messages (top languages of affected customers) for every raised incident
Writes data/<name>/llm_eval.json and briefs.json.

Usage: python pipeline/eval_llm.py --data data/synthetic --per-lang 60 [--workers 16]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import genai  # noqa: E402

HOUR = 3600


def macro_f1(y, p):
    labels = sorted(set(y))
    f = []
    for lab in labels:
        tp = sum(1 for a, b in zip(y, p) if a == lab and b == lab)
        fp = sum(1 for a, b in zip(y, p) if a != lab and b == lab)
        fn = sum(1 for a, b in zip(y, p) if a == lab and b != lab)
        f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f))


def score(df, preds):
    out = {}
    for lang in ["English", "Pidgin", "Hausa", "Yoruba", "Igbo", "ALL"]:
        m = [i for i in range(df.height) if lang == "ALL" or df["language"][i] == lang]
        ok = [i for i in m if "error" not in preds[i]]
        y = [df["issue"][i] for i in ok]
        p = [preds[i].get("issue") for i in ok]
        amt = [i for i in ok if df["amount"][i] > 0]
        amt_ok = sum(1 for i in amt if preds[i].get("amount_naira") and abs(preds[i]["amount_naira"] - df["amount"][i]) <= 0.01 * df["amount"][i])
        out[lang] = {"n": len(m), "parsed": len(ok), "issue_accuracy": float(np.mean([a == b for a, b in zip(y, p)])) if ok else None,
                     "issue_macro_f1": macro_f1(y, p) if ok else None,
                     "amount_accuracy": amt_ok / len(amt) if amt else None,
                     "language_id_accuracy": float(np.mean([preds[i].get("language") == df["language"][i] for i in ok])) if ok else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--per-lang", type=int, default=60)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()
    d = Path(args.data)
    print("servers:", genai.health(), flush=True)
    res = {}
    if not args.skip_eval:
        df = pl.read_parquet(d / "complaints.parquet").filter((pl.col("incident_id") == -1) & (pl.col("repost_of") == -1))
        smp = pl.concat([df.filter(pl.col("language") == L).sample(args.per_lang, seed=3) for L in ["English", "Pidgin", "Hausa", "Yoruba", "Igbo"]])
        texts = smp["text"].to_list()
        t0 = time.perf_counter()
        para = genai.paraphrase(list(zip(texts, smp["language"].to_list())), args.workers)
        t_para = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred_t = genai.extract(texts, args.workers)
        t_ext = time.perf_counter() - t0
        pred_p = genai.extract([p if not p.startswith("__error__") else t for p, t in zip(para, texts)], args.workers)
        res = {"n": smp.height, "template": score(smp, pred_t), "rewritten": score(smp, pred_p),
               "extract_per_s": smp.height / t_ext, "paraphrase_per_s": smp.height / t_para,
               "examples": [{"language": smp["language"][i], "original": texts[i], "rewritten": para[i], "gold_issue": smp["issue"][i],
                             "pred_on_rewritten": pred_p[i]} for i in range(0, smp.height, max(1, smp.height // 15))]}
        (d / "llm_eval.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
        for k in ("template", "rewritten"):
            print(k, {L: (round(v["issue_accuracy"] or 0, 3), round(v["amount_accuracy"] or 0, 3) if v["amount_accuracy"] is not None else None)
                      for L, v in res[k].items()}, flush=True)
        print(f"extraction {res['extract_per_s']:.1f}/s, paraphrase {res['paraphrase_per_s']:.1f}/s", flush=True)

    # ---- briefs + customer messages for every raised incident ----
    det = json.loads((d / "detect_gpu.json").read_text() if (d / "detect_gpu.json").exists() else (d / "detect_cpu.json").read_text())
    briefs = []
    for k, inc in enumerate(det["raised"]):
        rc = inc["root_cause"][:3]
        ev = {"incident_id": f"I{k + 1}", "topics": inc["topics"], "detected_at_sim_hours": round(inc["alert_ts"] / HOUR, 1),
              "complaints_last_3h": inc["n_spike"], "expected_last_3h": round(inc["expected"], 1),
              "root_cause": [{"attributes": e["items"], "share_of_spike": round(e["support_spike"], 2),
                              "share_normally": round(e["support_base"], 2), "lift": round(e["lift"], 1),
                              "share_of_excess_explained": round(e["explained"], 2)} for e in rc],
              "correlated_event": inc["linked_event"], "regions_affected": inc["n_regions"], "channels_affected": inc["n_channels"],
              "money_at_stake_naira": round(inc["money_at_stake"]), "vulnerable_customer_share": round(inc["vulnerable_share"], 2),
              "languages": inc["languages"][:5],
              "sample_complaints": [{"id": s["id"], "language": s["language"], "channel": s["channel"], "region": s["region"], "text": s["text"]}
                                    for s in inc["samples"]],
              "customer_facts": {"problem": " / ".join(inc["topics"]), "cause_attributes": rc[0]["items"] if rc else [],
                                 "event": inc["linked_event"]}}
        t0 = time.perf_counter()
        b = genai.incident_brief(ev)
        msgs = {L["language"]: genai.customer_message(ev, L["language"]) for L in inc["languages"][:3]}
        briefs.append({"evidence": ev, "brief": b, "customer_messages": msgs, "seconds": time.perf_counter() - t0})
        print(f"I{k + 1} brief grounded={b['grounded']} cited={len(b['cited'])} langs={list(msgs)} ({briefs[-1]['seconds']:.1f}s)", flush=True)
    (d / "briefs.json").write_text(json.dumps(briefs, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

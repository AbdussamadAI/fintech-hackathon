"""N-ATLaS normalisation of every distinct (redacted) complaint: issue code, English summary, language, amount.

The English summaries make topic discovery language-independent (a Hausa and an English complaint about the same
problem land in the same topic), and the issue codes give a large per-language accuracy measurement.
Writes data/<name>/normalized.parquet and normalize.json (per-language accuracy, throughput).

Usage: python pipeline/normalize.py --data data/synthetic [--workers 32] [--limit N]
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
from pipeline.redact import redact  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    d = Path(args.data)
    df = pl.read_parquet(d / "complaints.parquet").with_columns(
        redacted=pl.col("text").map_elements(redact, return_dtype=pl.Utf8))
    uniq = df.group_by("redacted").agg(pl.len().alias("n"), pl.col("issue").mode().first().alias("gold_issue"),
                                       pl.col("language").mode().first().alias("gold_language")).sort("n", descending=True)
    if args.limit:
        uniq = uniq.head(args.limit)
    texts = uniq["redacted"].to_list()
    print(f"{len(texts):,} distinct redacted complaints; servers {genai.health()}", flush=True)
    t0 = time.perf_counter()
    out, done = [None] * len(texts), 0
    step = 2000
    for i in range(0, len(texts), step):
        out[i:i + step] = genai.extract(texts[i:i + step], args.workers)
        done += len(texts[i:i + step])
        print(f"  {done:,}/{len(texts):,}  {done / (time.perf_counter() - t0):.1f}/s", flush=True)
    secs = time.perf_counter() - t0
    norm = uniq.with_columns(
        issue_llm=pl.Series([o.get("issue") for o in out]), summary_en=pl.Series([o.get("summary_en") for o in out]),
        language_llm=pl.Series([o.get("language") for o in out]), amount_llm=pl.Series([o.get("amount_naira") for o in out], dtype=pl.Float64),
        urgency=pl.Series([o.get("urgency") for o in out]), sentiment=pl.Series([o.get("sentiment") for o in out]),
        error=pl.Series([o.get("error") for o in out]))
    norm.write_parquet(d / "normalized.parquet")
    # per-language accuracy weighted by how many complaints carry each text
    full = df.join(norm.select("redacted", "issue_llm", "language_llm"), on="redacted")
    rep = {"distinct": len(texts), "seconds": secs, "per_second": len(texts) / secs,
           "parse_errors": int(norm["error"].is_not_null().sum()), "by_language": {}}
    for L in ["English", "Pidgin", "Hausa", "Yoruba", "Igbo"]:
        f = full.filter(pl.col("language") == L)
        y, p = f["issue"].to_numpy(), f["issue_llm"].to_numpy()
        labs = np.unique(y)
        f1 = np.mean([2 * ((y == c) & (p == c)).sum() / max((y == c).sum() + (p == c).sum(), 1) for c in labs])
        rep["by_language"][L] = {"n": f.height, "issue_accuracy": float((y == p).mean()), "issue_macro_f1": float(f1),
                                 "language_id_accuracy": float((f["language_llm"] == L).mean())}
    (d / "normalize.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()

"""A/B: plain-prompt JSON extraction vs grammar-constrained (schema) extraction with N-ATLaS."""
import sys, time
import polars as pl
sys.path.insert(0, ".")
from pipeline import genai
from pipeline.redact import redact
d = pl.read_parquet("data/synthetic/complaints.parquet").filter(pl.col("incident_id") == -1)
s = pl.concat([d.filter(pl.col("language") == L).sample(40, seed=5) for L in ["English", "Pidgin", "Hausa", "Yoruba", "Igbo"]])
texts = [redact(t) for t in s["text"].to_list()]
for mode, name in ((True, "fast"), (False, "schema")):
    t = time.perf_counter()
    out = genai.pmap(lambda x: genai.extract_one(x, fast=mode), texts, 32)
    dt = time.perf_counter() - t
    acc = sum(o.get("issue") == g for o, g in zip(out, s["issue"].to_list())) / len(out)
    err = sum("error" in o for o in out)
    print(f"{name:6s}: {len(texts) / dt:5.1f}/s  issue accuracy {acc:.3f}  errors {err}", flush=True)

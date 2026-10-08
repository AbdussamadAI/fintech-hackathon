"""Streaming multilingual topic discovery on BGE-M3 embeddings (no labels used).

  1. embed every distinct complaint text once (cache: embeddings.npy + text index)
  2. bootstrap: spherical k-means on the warm-up days -> initial topics
  3. stream in time order: a complaint joins its nearest topic if cosine >= tau, else goes to a novelty pool;
     every 6 h the pool is clustered and dense groups become new topics
  4. evaluate against the generator's gold issue (purity, NMI) overall and per language: the inclusion check is
     whether a Hausa complaint about an issue lands in the same topic as the English ones
Writes data/<name>/topics.parquet (complaint_id, topic) and topics.json (sizes, samples, metrics).

Usage: python pipeline/topics.py --data data/synthetic [--k 40]
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

HOUR, DAY = 3600, 86400


def kmeans(X, k, iters=25, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        lab = np.argmax(X @ C.T, axis=1)
        for j in range(k):
            m = X[lab == j]
            if len(m):
                c = m.mean(0); C[j] = c / (np.linalg.norm(c) + 1e-9)
    return C, np.argmax(X @ C.T, axis=1)


def nmi(a, b):
    a = np.unique(a, return_inverse=True)[1]; b = np.unique(b, return_inverse=True)[1]
    n = len(a)
    cont = np.zeros((a.max() + 1, b.max() + 1)); np.add.at(cont, (a, b), 1)
    pa, pb, pab = cont.sum(1) / n, cont.sum(0) / n, cont / n
    nz = pab > 0
    mi = (pab[nz] * np.log(pab[nz] / (pa[:, None] * pb[None, :])[nz])).sum()
    ha, hb = -(pa[pa > 0] * np.log(pa[pa > 0])).sum(), -(pb[pb > 0] * np.log(pb[pb > 0])).sum()
    return float(2 * mi / (ha + hb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--tau", type=float, default=0.75)
    ap.add_argument("--warmup-days", type=float, default=3)
    ap.add_argument("--source", choices=["text", "summary"], default="text",
                    help="embed the redacted complaint, or N-ATLaS's English summary of it (language-independent)")
    args = ap.parse_args()
    d = Path(args.data)
    df = pl.read_parquet(d / "complaints.parquet").sort("ts")

    # ---- embeddings (cached per distinct text) ----
    red = [redact(t) for t in df["text"].to_list()]          # embed redacted text only (no PII leaves this step)
    if args.source == "summary":                             # N-ATLaS English summary of each redacted complaint
        nm = pl.read_parquet(d / "normalized.parquet").select("redacted", "summary_en")
        smap = dict(zip(nm["redacted"].to_list(), nm["summary_en"].to_list()))
        red = [smap.get(r) or r for r in red]
    texts = list(dict.fromkeys(red))
    cache = d / f"embeddings_{args.source}.npy"
    t0 = time.perf_counter()
    meta_f = d / f"embeddings_{args.source}.json"
    if cache.exists() and json.loads(meta_f.read_text())["n"] == len(texts):
        E = np.load(cache)
    else:
        E = np.asarray(genai.embed(texts), np.float32)
        E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
        np.save(cache, E)
        meta_f.write_text(json.dumps({"n": len(texts), "seconds": time.perf_counter() - t0}))
    emb_s = time.perf_counter() - t0
    idx = {t: i for i, t in enumerate(texts)}
    row_e = np.array([idx[t] for t in red])
    ts = df["ts"].to_numpy()

    # ---- bootstrap topics on warm-up days ----
    warm = ts < args.warmup_days * DAY
    uw = np.unique(row_e[warm])
    C, _ = kmeans(E[uw], args.k)
    # merge near-identical centroids (k-means over-splitting)
    keep = []
    for j in range(len(C)):
        if all(C[j] @ C[i] < 0.92 for i in keep):
            keep.append(j)
    C = C[keep]

    # ---- streaming assignment with novelty pool ----
    topic = np.full(len(df), -1)
    pool, next_check = [], args.warmup_days * DAY + 6 * HOUR
    counts = np.ones(len(C))
    for i in range(len(df)):
        e = E[row_e[i]]
        sims = C @ e
        j = int(np.argmax(sims))
        if sims[j] >= args.tau or ts[i] < args.warmup_days * DAY:
            topic[i] = j
            counts[j] += 1
            C[j] += (e - C[j]) / min(counts[j], 500)          # slow online centroid update
            C[j] /= np.linalg.norm(C[j]) + 1e-9
        else:
            pool.append(i)
        if ts[i] >= next_check:
            next_check += 6 * HOUR
            if len(pool) >= 60:
                P = E[row_e[pool]]
                kk = max(1, len(pool) // 150)
                Cp, lp = kmeans(P, kk, iters=15)
                for c in range(kk):
                    members = [pool[m] for m in np.where(lp == c)[0]]
                    if len(members) >= 40 and float((E[row_e[members]] @ Cp[c]).mean()) >= args.tau:
                        C = np.vstack([C, Cp[c]]); counts = np.append(counts, len(members))
                        topic[members] = len(C) - 1
                pool = [m for m in pool if topic[m] < 0]
    # leftovers: nearest topic (they are counted, just not trusted as new topics)
    for i in pool:
        topic[i] = int(np.argmax(C @ E[row_e[i]]))

    gold = df["issue"].to_numpy()
    lang = df["language"].to_numpy()

    def purity(t, g):
        tot = 0
        for tp in np.unique(t):
            _, c = np.unique(g[t == tp], return_counts=True); tot += c.max()
        return tot / len(t)

    by_lang = {}
    for L in np.unique(lang):
        m = lang == L
        # share of each gold issue's complaints (in this language) that land in that issue's majority topic overall
        hits = 0
        for g in np.unique(gold):
            maj = np.bincount(topic[gold == g]).argmax()
            hits += int(((gold == g) & m & (topic == maj)).sum())
        by_lang[L] = {"n": int(m.sum()), "same_topic_as_majority": hits / int(m.sum())}
    # distinctive words per topic (used to match operational events; no labels involved)
    import re
    from collections import Counter
    tok = lambda t: set(re.findall(r"[a-z][a-z0-9.]+", t.lower()))  # noqa: E731
    all_txt = red
    df_count = Counter(w for t in all_txt[::7] for w in tok(t))
    n_all = len(all_txt[::7])
    names = {}
    for tp in np.unique(topic):
        g, c = np.unique(gold[topic == tp], return_counts=True)
        mem = [all_txt[i] for i in np.where(topic == tp)[0][::5]]
        cw = Counter(w for t in mem for w in tok(t))
        kw = sorted(((cw[w] / len(mem)) / ((df_count[w] + 1) / n_all), w) for w in cw if cw[w] >= max(5, 0.15 * len(mem)))[::-1][:8]
        names[int(tp)] = {"size": int((topic == tp).sum()), "keywords": [w for _, w in kw],
                          "majority_issue": str(g[c.argmax()]), "purity": float(c.max() / c.sum()),
                          "samples": [red[i] for i in np.where(topic == tp)[0][:5]]}
    metrics = {"n_topics": int(len(np.unique(topic))), "bootstrap_topics": len(keep), "purity": purity(topic, gold),
               "nmi": nmi(topic, gold), "by_language": by_lang, "embed_seconds": emb_s, "distinct_texts": len(texts)}
    out = df.select("complaint_id").with_columns(topic=pl.Series(["T%02d" % t for t in topic]))
    suffix = "" if args.source == "text" else "_summary"
    out.write_parquet(d / f"topics{suffix}.parquet")
    (d / f"topics{suffix}.json").write_text(json.dumps({"metrics": metrics, "topics": {"T%02d" % k: v for k, v in names.items()}},
                                              indent=1, ensure_ascii=False))
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()

"""Replay the complaint stream, detect emerging incidents and explain them; evaluate against injected ground truth.

Stages per run:
  1. K3 MinHash + LSH over social posts -> repost cascades counted once
  2. K1 burst scoring over the topic x channel x region x segment cube, one 15-min bucket at a time
  3. alerts grouped into candidate incidents per topic
  4. K2 root-cause contrast mining for every candidate (spike vs the topic's previous 7 days)
  5. event correlation, systemic-vs-surge verdict, priority
  6. evaluation: time to detect vs a daily-count rule, root-cause top-1/top-3, event link, false alarms

Usage: python pipeline/detect.py --data data/synthetic [--ops gpu|cpu] [--topics gold|<topics.parquet>]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.engine import get_ops  # noqa: E402

HOUR, DAY, BUCKET = 3600, 86400, 900
K2_ATTRS = ["channel", "language", "region", "app_version", "device_os", "switch", "telco", "atm_id", "card_scheme", "segment"]
TOPIC_WORDS = {  # topic tokens used when matching operational events
    "otp": ["otp", "app"], "app": ["app"], "ussd": ["ussd"], "atm": ["atm"], "card": ["card"], "pos": ["pos", "acquir"],
    "sms": ["sms", "billing"], "reversal": ["switch", "settlement", "nip"], "transfer": ["switch", "nip"], "unauthorised": ["fraud"],
}


class P:  # tunables
    alpha, k_cusum, min_expected = 0.005, 1.0, 0.5    # ~50 h baseline memory: slow ramps must not become 'normal'
    score_th, min_count = 7.0, 3
    warmup_days = 3
    quiet_close_h = 12
    lsh_bands, lsh_rows, cascade_min, cascade_window_h = 8, 8, 50, 6
    rc_min_explained, rc_min_lift = 0.4, 1.5           # an explanation must account for >=40% of the excess
    conc_explained, conc_lift, conc_min_excess = 0.6, 1.8, 12
    alpha_warmup = 0.05                                # converge quickly during warm-up, then slow down
    recheck_every_h, recheck_for_h = 1, 24             # unresolved alerts are re-explained hourly for a day


# ------------------------------------------------------------------------------------------------- K3 dedup
def repost_cascades(df: pl.DataFrame, ops, H=64):
    soc = df.filter(pl.col("channel") == "social").select("complaint_id", "ts", "text")
    texts = [t.encode("utf-8") for t in soc["text"].to_list()]
    off = np.zeros(len(texts) + 1, np.int64)
    off[1:] = np.cumsum([len(t) for t in texts])
    chars = np.frombuffer(b"".join(texts), np.uint8)
    rng = np.random.default_rng(1)
    ha = (rng.integers(1, 2**31, H, dtype=np.uint32) | 1).astype(np.uint32)
    hb = rng.integers(0, 2**31, H, dtype=np.uint32).astype(np.uint32)
    t0 = time.perf_counter()
    if ops.name == "gpu":
        cp = ops.cp
        sig = ops.k3(cp.asarray(chars), cp.asarray(off), cp.asarray(ha), cp.asarray(hb)).get()
    else:
        sig = ops.k3(chars, off, ha, hb)
    t_sig = time.perf_counter() - t0
    # LSH: docs sharing any band key inside the same cascade window are linked
    n = len(texts)
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    win = (soc["ts"].to_numpy() // (P.cascade_window_h * HOUR)).astype(np.int64)
    for b in range(P.lsh_bands):
        band = sig[:, b * P.lsh_rows:(b + 1) * P.lsh_rows]
        key = np.ascontiguousarray(band).view(np.dtype((np.void, band.dtype.itemsize * P.lsh_rows))).ravel()
        keyh = np.array([hash(bytes(k)) for k in key], dtype=np.int64) ^ (win * 1000003)
        order = np.argsort(keyh, kind="stable")
        ks = keyh[order]
        same = np.where(ks[1:] == ks[:-1])[0]
        for i in same:
            a, c = find(order[i]), find(order[i + 1])
            if a != c:
                parent[a] = c
    roots = np.array([find(i) for i in range(n)])
    _, inv, cnt = np.unique(roots, return_inverse=True, return_counts=True)
    in_cascade = cnt[inv] >= P.cascade_min
    # keep the earliest post of each cascade, drop the rest
    first = {}
    ids = soc["complaint_id"].to_numpy()
    drop = []
    for i in np.argsort(soc["ts"].to_numpy()):
        if in_cascade[i]:
            r = roots[i]
            if r in first:
                drop.append(ids[i])
            else:
                first[r] = ids[i]
    return set(int(x) for x in drop), {"social_posts": n, "cascades": len(first), "dropped_reposts": len(drop),
                                        "minhash_seconds": t_sig}


# ------------------------------------------------------------------------------------------------- main replay
def run(data, ops_name="gpu", topics="gold", no_dedup=False):
    """Run the whole detector over a dataset; returns the summary dict (also used by the live server)."""
    args = argparse.Namespace(data=data, ops=ops_name, topics=topics, no_dedup=no_dedup)
    d = Path(args.data)
    ops = get_ops(args.ops)
    xp = ops.xp
    df = pl.read_parquet(d / "complaints.parquet")
    meta = json.loads((d / "incidents.json").read_text())
    events = pl.read_parquet(d / "events.parquet")
    topic_kw, topic_issue = {}, {}
    if args.topics == "gold":
        df = df.with_columns(topic=pl.col("issue"))
    else:
        df = df.join(pl.read_parquet(args.topics).select("complaint_id", "topic"), on="complaint_id")
        tj = json.loads(Path(args.topics).with_suffix(".json").read_text())["topics"]
        topic_kw = {k: v["keywords"] for k, v in tj.items()}             # used for event matching
        topic_issue = {k: v["majority_issue"] for k, v in tj.items()}    # evaluation only

    timings = {}
    t0 = time.perf_counter()
    dropped, dedup_info = (set(), {}) if args.no_dedup else repost_cascades(df, ops)
    timings["dedup_s"] = time.perf_counter() - t0
    df = df.with_columns(is_repost=pl.col("complaint_id").is_in(list(dropped)))
    live = df.filter(~pl.col("is_repost"))

    # ---- encode cube dims (last index of each dim = ALL) ----
    dims_names = ["topic", "channel", "region", "segment"]
    cats = {c: sorted(df[c].unique().to_list()) for c in dims_names}
    enc = {c: {v: i for i, v in enumerate(cats[c])} for c in dims_names}
    dims = tuple(len(cats[c]) + 1 for c in dims_names)
    n_series = int(np.prod(dims))
    codes = {c: live[c].replace_strict(enc[c]).cast(pl.Int32).to_numpy() for c in dims_names}
    ts = live["ts"].to_numpy()
    bucket = (ts // BUCKET).astype(np.int64)
    n_buckets = int(df["ts"].max() // BUCKET) + 1
    starts = np.searchsorted(bucket, np.arange(n_buckets + 1))

    # global diurnal profile from the warm-up days (share of daily volume per hour, mean 1.0)
    warm_mask = ts < P.warmup_days * DAY
    hod_counts = np.bincount(((ts[warm_mask] // HOUR) % 24).astype(int), minlength=24).astype(float)
    season_h = hod_counts / hod_counts.mean()

    state = [xp.full(n_series, 0.0, xp.float32), xp.zeros(n_series, xp.float32), xp.zeros(n_series, xp.float32)]
    dev = {c: xp.asarray(codes[c]) for c in dims_names}
    alerts = []
    t0 = time.perf_counter()
    for b in range(n_buckets):
        s, e = starts[b], starts[b + 1]
        counts = ops.k1_count(dev["topic"][s:e], dev["channel"][s:e], dev["region"][s:e], dev["segment"][s:e], dims)
        season = float(season_h[(b * BUCKET // HOUR) % 24]) / (HOUR / BUCKET)  # expected per bucket = level*season
        warm = b * BUCKET >= P.warmup_days * DAY
        z, sc = ops.k1_score(counts, season, state, P.alpha if warm else P.alpha_warmup, P.k_cusum, P.min_expected, warm)
        if warm:
            hit = xp.where((sc >= P.score_th) & (counts >= P.min_count))[0]
            if hit.size:
                hit_h = hit.get() if hasattr(hit, "get") else hit
                cz = (counts[hit].get() if hasattr(counts, "get") else counts[hit])
                ss = (sc[hit].get() if hasattr(sc, "get") else sc[hit])
                for h, c, s_ in zip(hit_h, cz, ss):
                    alerts.append((b, int(h), float(c), float(s_)))
    if ops.name == "gpu":
        ops.cp.cuda.Device().synchronize()
    timings["k1_replay_s"] = time.perf_counter() - t0
    timings["n_series"], timings["n_buckets"] = n_series, n_buckets

    def decode(idx):
        out = []
        for k in reversed(range(4)):
            out.append(idx % dims[k]); idx //= dims[k]
        out = out[::-1]
        return {dims_names[k]: (cats[dims_names[k]][out[k]] if out[k] < dims[k] - 1 else "ALL") for k in range(4)}

    topic_arr = live["topic"].to_numpy()
    topic_cache = {}

    def dominant_topic(b):
        """Topic with the largest excess over its 7-day rate in the 3 h before bucket b (for ALL-topic alerts)."""
        if b in topic_cache:
            return topic_cache[b]
        t1 = (b + 1) * BUCKET
        i0, i1, i2 = np.searchsorted(ts, [t1 - 3 * HOUR - 7 * DAY, t1 - 3 * HOUR, t1])
        recent = np.unique(topic_arr[i1:i2], return_counts=True)
        base = dict(zip(*np.unique(topic_arr[i0:i1], return_counts=True)))
        best, best_ex = None, 0.0
        for tp, c in zip(*recent):
            ex = c - base.get(tp, 0) * 3 / (7 * 24)
            if ex > best_ex:
                best, best_ex = tp, ex
        topic_cache[b] = best
        return best

    # ---- group alerts into candidate incidents per topic ----
    cands, open_ = [], {}
    for b, s_idx, c, sc_ in alerts:
        key = decode(s_idx)
        tkey = key["topic"]
        if tkey == "ALL":
            tkey = dominant_topic(b)
            if tkey is None:
                continue
            key["topic"] = tkey
        o = open_.get(tkey)
        if o is None or b - o["last_bucket"] > P.quiet_close_h * HOUR / BUCKET:
            o = {"topic": tkey, "first_bucket": b, "last_bucket": b, "max_score": 0.0, "series": {}, "alert_buckets": []}
            open_[tkey] = o
            cands.append(o)
        o["last_bucket"] = b
        if not o["alert_buckets"] or o["alert_buckets"][-1] != b:
            o["alert_buckets"].append(b)
        o["max_score"] = max(o["max_score"], sc_)
        lbl = " / ".join(f"{k}={v}" for k, v in key.items() if v != "ALL" and k != "topic") or "all channels/regions"
        o["series"][lbl] = max(o["series"].get(lbl, 0), sc_)

    # ---- K2 root cause per candidate ----
    attr_vals, attr_of = [], []
    for ai, a in enumerate(K2_ATTRS):
        for v in sorted(df[a].unique().to_list()):
            if v != "n/a":
                attr_vals.append((a, v)); attr_of.append(ai)
    V = len(attr_vals)
    attr_of = np.array(attr_of, np.int32)
    val_codes = {a: live[a].to_numpy() for a in K2_ATTRS}
    k2_time, k2_pairs = 0.0, 0
    iu, ju = np.triu_indices(V, k=1)
    pack = lambda m: np.packbits(m).view(np.uint32)  # bit order is irrelevant: only AND + popcount  # noqa: E731

    def explain(topic, t_alert):
        """K2: which attribute values (and pairs) account for the spike's excess over the topic's last 7 days?"""
        nonlocal k2_time, k2_pairs
        s0, b0 = t_alert - 3 * HOUR, t_alert - 3 * HOUR - 7 * DAY
        lo, hi = np.searchsorted(ts, b0), np.searchsorted(ts, t_alert)
        sel = np.arange(lo, hi)
        if topic != "ALL":
            sel = sel[topic_arr[sel] == topic]
        inS = ts[sel] >= s0
        nS, nB = int(inS.sum()), int((~inS).sum())
        hrs = ((np.arange(s0, t_alert, HOUR) // HOUR) % 24).astype(int)
        E = nB * (3 / (7 * 24)) * float(season_h[hrs].mean())       # complaints expected in the window without an incident
        excess = nS - E
        n = len(sel)
        W = max((n + 31) // 32, 1)
        pad = W * 32 - n
        bits_u = np.stack([pack(np.pad(val_codes[a][sel] == v, (0, pad))) for a, v in attr_vals])
        S = pack(np.pad(inS, (0, pad)))
        B = pack(np.pad(~inS, (0, pad)))
        tk = time.perf_counter()
        if ops.name == "gpu":
            cp = ops.cp
            r = ops.k2(cp.asarray(bits_u), cp.asarray(S), cp.asarray(B), cp.asarray(attr_of))
            sS1, sB1, sS2, sB2 = (x.get() for x in r)
        else:
            sS1, sB1, sS2, sB2 = ops.k2(bits_u, S, B, attr_of)
        k2_time += time.perf_counter() - tk
        k2_pairs += V * (V - 1) // 2
        expl = []
        if excess > 0:
            keep = np.where(sS2 >= P.rc_min_explained * excess)[0]
            cand = [((attr_vals[i],), sS1[i], sB1[i]) for i in range(V)]
            cand += [((attr_vals[iu[p]], attr_vals[ju[p]]), sS2[p], sB2[p]) for p in keep]
            for items, a_s, a_b in cand:
                pB_raw = a_b / max(nB, 1)
                explained = (a_s - E * pB_raw) / excess
                lift = (a_s / max(nS, 1)) / ((a_b + 0.5) / (nB + 1))
                if explained < P.rc_min_explained or lift < P.rc_min_lift:
                    continue
                expl.append({"items": [f"{a}={v}" for a, v in items], "support_spike": a_s / max(nS, 1),
                             "support_base": pB_raw, "lift": lift, "explained": min(float(explained), 1.0),
                             "score": min(float(explained), 1.0) * np.log2(lift)})
            expl.sort(key=lambda e: -e["score"])
        return expl[:5], nS, nB, E

    for o in cands:
        first = o["alert_buckets"][0]
        checks, last = [], -10**9
        for b in o["alert_buckets"]:                    # first alert, then at most hourly while alerts continue
            if b - first > P.recheck_for_h * HOUR / BUCKET:
                break
            if b - last >= P.recheck_every_h * HOUR / BUCKET:
                checks.append(b); last = b
        o["first_alert_ts"] = float((first + 1) * BUCKET)
        for b in checks:
            t_alert = float((b + 1) * BUCKET)
            expl, nS, nB, E = explain(o["topic"], t_alert)
            o.update(root_cause=expl, n_spike=nS, n_base=nB, expected=E, alert_ts=t_alert)
            top = expl[0] if expl else None
            if top and top["explained"] >= P.conc_explained and top["lift"] >= P.conc_lift and nS - E >= P.conc_min_excess:
                break

    # ---- event correlation + verdicts ----
    ev_rows = events.to_dicts()
    for o in cands:
        toks = set()
        for e in o["root_cause"][:3]:
            for it in e["items"]:
                toks.add(it.split("=", 1)[1].lower())
        for k, ws in TOPIC_WORDS.items():
            if k in o["topic"]:
                toks.update(ws)
        toks.update(w for w in topic_kw.get(o["topic"], []) if len(w) >= 3)
        best = None
        for ev in ev_rows:
            if o["alert_ts"] - 18 * HOUR <= ev["ts"] <= o["alert_ts"] + HOUR:
                desc = ev["description"].lower()
                m = sum(1 for tkn in toks if tkn and tkn in desc)
                if m and (best is None or (m, ev["ts"]) > (best[0], best[1]["ts"])):
                    best = (m, ev)
        o["linked_event"] = best[1] if best else None
        top = o["root_cause"][0] if o["root_cause"] else None
        concentrated = bool(top and top["explained"] >= P.conc_explained and top["lift"] >= P.conc_lift
                            and o["n_spike"] - o["expected"] >= P.conc_min_excess)
        o["verdict"] = "systemic incident" if concentrated else "volume surge, no concentrated cause"
        sub = live.filter((pl.col("ts") >= o["alert_ts"] - 3 * HOUR) & (pl.col("ts") < o["alert_ts"]) &
                          ((pl.col("topic") == o["topic"]) if o["topic"] != "ALL" else pl.lit(True)))
        o["n_regions"] = sub["region"].n_unique(); o["n_channels"] = sub["channel"].n_unique()
        o["money_at_stake"] = float(sub["amount"].sum())
        o["vulnerable_share"] = float(sub["segment"].is_in(["pensioner", "student"]).mean()) if sub.height else 0.0
        # evidence sample for the brief: up to 2 complaints per language from the spike window
        smp = sub.sort("ts", descending=True).group_by("language", maintain_order=True).head(2).head(10)
        o["samples"] = [{"id": int(r["complaint_id"]), "language": r["language"], "channel": r["channel"], "region": r["region"],
                         "text": r["text"]} for r in smp.iter_rows(named=True)]
        o["languages"] = sub["language"].value_counts().sort("count", descending=True).to_dicts()

    raised = [o for o in cands if o["verdict"] == "systemic incident"]
    # merge raised candidates that share their top root cause and start within 6h (e.g. OTP + login topics)
    merged = []
    for o in sorted(raised, key=lambda x: x["alert_ts"]):
        tgt = next((m for m in merged if abs(m["alert_ts"] - o["alert_ts"]) < 12 * HOUR and m["root_cause"] and o["root_cause"]
                    and set(m["root_cause"][0]["items"]) & set(o["root_cause"][0]["items"])), None)
        if tgt:
            tgt["topics"].append(o["topic"]); tgt["max_score"] = max(tgt["max_score"], o["max_score"])
        else:
            merged.append({**o, "topics": [o["topic"]]})

    # ---- naive rule: topic daily count > mean + 3 sd of previous 7 days, checked at end of day ----
    daily = (live.with_columns(day=(pl.col("ts") // DAY).cast(pl.Int32)).group_by("topic", "day").len()
                 .sort("topic", "day"))
    naive = {}
    for tp, g in daily.group_by("topic"):
        arr = np.zeros(meta["days"]); arr[g["day"].to_numpy()] = g["len"].to_numpy()
        for dd in range(P.warmup_days, meta["days"]):       # same warm-up as the detector; up to 7 days of history
            hist = arr[max(0, dd - 7):dd]
            mu, sd = hist.mean(), hist.std()
            if arr[dd] > mu + 3 * max(sd, 1):
                naive.setdefault(tp[0], []).append((dd + 1) * DAY)

    # ---- evaluation ----
    def cause_hit(inc, expl):
        items = set(expl["items"])
        need = []
        for a, v in inc["cause"].items():
            if isinstance(v, list):
                need.append(any(f"{a}={x}" in items for x in v) or f"region={inc['cause'].get('region')}" in items)
            else:
                need.append(f"{a}={v}" in items)
        return all(need) or (len(need) > 1 and sum(need) >= len(need) - 1 and len(items) == 1 and any(need))

    issues_of = lambda m: {topic_issue.get(t, t) for t in m["topics"]}  # noqa: E731
    rows, matched = [], set()
    for inc in meta["incidents"]:
        cand_m = [m for m in merged if issues_of(m) & set(inc["issues"]) and inc["start_ts"] - 3 * HOUR <= m["alert_ts"] <= inc["end_ts"]]
        m = min(cand_m, key=lambda x: x["alert_ts"]) if cand_m else None
        nv = min([t for tp, times in naive.items() if topic_issue.get(tp, tp) in inc["issues"] for t in times
                  if inc["start_ts"] <= t <= inc["end_ts"] + DAY], default=None)
        r = {"incident": inc["name"], "start": inc["start_ts"], "peak": inc["peak_ts"]}
        if m:
            matched.add(id(m))
            rc = m["root_cause"]
            r.update(detected=True, hours_to_detect=(m["alert_ts"] - inc["start_ts"]) / HOUR,
                     hours_before_peak=(inc["peak_ts"] - m["alert_ts"]) / HOUR,
                     top1=bool(rc and cause_hit(inc, rc[0])), top3=any(cause_hit(inc, e) for e in rc[:3]),
                     top_cause=" + ".join(rc[0]["items"]) if rc else None, lift=rc[0]["lift"] if rc else None,
                     support=rc[0]["support_spike"] if rc else None,
                     event_ok=(inc["event_id"] is not None and m["linked_event"] is not None and m["linked_event"]["event_id"] == inc["event_id"])
                     or (inc["event_id"] is None and m["linked_event"] is None),
                     linked_event=m["linked_event"]["description"] if m["linked_event"] else None)
        else:
            r.update(detected=False)
        r["naive_hours_to_detect"] = (nv - inc["start_ts"]) / HOUR if nv else None
        rows.append(r)
    # a raised incident is a true positive if it overlaps any injected incident on topic, time and cause
    for m in merged:
        for inc in meta["incidents"]:
            if (issues_of(m) & set(inc["issues"]) and inc["start_ts"] - 3 * HOUR <= m["alert_ts"] <= inc["end_ts"]
                    and m["root_cause"] and any(cause_hit(inc, e) for e in m["root_cause"][:3])):
                matched.add(id(m))
    false_raised = [m for m in merged if id(m) not in matched]
    det = [r for r in rows if r["detected"]]
    summary = {
        "ops": ops.name, "topics": args.topics, "dedup": dedup_info, "timings": timings,
        "k2": {"seconds": k2_time, "pairs_scored": k2_pairs, "values": V, "candidates": len(cands)},
        "incidents": rows,
        "detected": f"{len(det)}/{len(rows)}",
        "median_hours_to_detect": float(np.median([r["hours_to_detect"] for r in det])) if det else None,
        "median_hours_before_peak": float(np.median([r["hours_before_peak"] for r in det])) if det else None,
        "naive_detected": f"{sum(r['naive_hours_to_detect'] is not None for r in rows)}/{len(rows)}",
        "naive_median_hours_to_detect": float(np.median([r["naive_hours_to_detect"] for r in rows if r["naive_hours_to_detect"] is not None]))
        if any(r["naive_hours_to_detect"] is not None for r in rows) else None,
        "root_cause_top1": f"{sum(r.get('top1', False) for r in rows)}/{len(rows)}",
        "root_cause_top3": f"{sum(r.get('top3', False) for r in rows)}/{len(rows)}",
        "event_linked": f"{sum(r.get('event_ok', False) for r in rows)}/{len(rows)}",
        "naive_false_alarms": sum(1 for tp, times in naive.items() for t in times
                                  if not any(topic_issue.get(tp, tp) in inc["issues"] and inc["start_ts"] <= t <= inc["end_ts"] + DAY
                                             for inc in meta["incidents"])),
        "naive_alerts": sum(len(v) for v in naive.values()),
        "false_incidents": [{"topic": m["topics"], "at_day": m["alert_ts"] / DAY, "cause": m["root_cause"][0]["items"] if m["root_cause"] else None}
                            for m in false_raised],
        "suppressed_surges": [{"topic": o["topic"], "at_day": o["alert_ts"] / DAY} for o in cands if o["verdict"] != "systemic incident"],
        "raised": [{k: (v if k != "linked_event" else (v["description"] if v else None)) for k, v in m.items()
                    if k in ("topics", "alert_ts", "first_alert_ts", "max_score", "root_cause", "linked_event", "n_spike", "expected",
                             "n_regions", "n_channels", "money_at_stake", "vulnerable_share", "series", "samples", "languages")}
                   for m in merged],
        "suppressed": [{"topic": o["topic"], "alert_ts": o["alert_ts"], "n_spike": o["n_spike"], "expected": o["expected"],
                        "root_cause": o["root_cause"][:2]} for o in cands if o["verdict"] != "systemic incident"],
    }
    summary["_rows"] = rows
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--ops", default="gpu")
    ap.add_argument("--topics", default="gold", help="'gold' (issue labels) or a parquet with complaint_id, topic")
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    summary = run(args.data, args.ops, args.topics, args.no_dedup)
    rows = summary.pop("_rows")
    dedup_info, timings = summary["dedup"], summary["timings"]
    out = Path(args.out or Path(args.data) / f"detect_{args.ops}{'_nodedup' if args.no_dedup else ''}{'' if args.topics == 'gold' else '_topics'}.json")
    out.write_text(json.dumps(summary, indent=1, default=float))
    for r in rows:
        if r["detected"]:
            print(f"  ✔ {r['incident'][:48]:48s} +{r['hours_to_detect']:5.1f}h (peak in {r['hours_before_peak']:5.1f}h) "
                  f"naive {r['naive_hours_to_detect'] if r['naive_hours_to_detect'] is None else round(r['naive_hours_to_detect'], 1)}h | "
                  f"cause {r['top_cause']} (lift {r['lift']:.1f}) top1={r['top1']} | event={'ok' if r['event_ok'] else r['linked_event']}")
        else:
            print(f"  ✘ {r['incident'][:48]:48s} not detected; naive {r['naive_hours_to_detect']}")
    print({k: summary[k] for k in ("detected", "median_hours_to_detect", "median_hours_before_peak", "naive_detected",
                                   "naive_median_hours_to_detect", "naive_false_alarms", "naive_alerts", "root_cause_top1",
                                   "root_cause_top3", "event_linked")})
    print("false incidents:", summary["false_incidents"])
    print("suppressed surges:", len(summary["suppressed_surges"]), "| dedup:", dedup_info, "| timings:", {k: round(v, 3) if isinstance(v, float) else v for k, v in timings.items()},
          "| k2:", {k: round(v, 3) if isinstance(v, float) else v for k, v in summary["k2"].items()})


if __name__ == "__main__":
    main()

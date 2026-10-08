"""Assemble everything the dashboard / pitch needs into one JSON: metrics + top rings with
reason codes (XGBoost SHAP) and STR drafts.

Usage: python scripts/export_demo.py --name HI-Small [--rings 8]
Writes data/processed/<name>/demo.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.str_draft import draft  # noqa: E402
from scripts.eval_rings import union_find_components  # noqa: E402
from scripts.train import load  # noqa: E402

USD_NGN = 1535.0   # CBN rate on 31 Dec 2024, used only to present amounts in naira

# feature -> sentence template (value-dependent). Only features with a clear human reading are listed.
REASONS = {
    "cyc2": lambda v: f"money returned to the sender through a 2-hop round trip ({int(v)} path(s))",
    "cyc3": lambda v: f"funds travelled in a 3-account cycle back to the origin ({int(v)} path(s))",
    "cyc4": lambda v: f"funds travelled in a 4-account cycle back to the origin ({int(v)} path(s))",
    "cyc5": lambda v: f"funds travelled in a 5-account cycle back to the origin ({int(v)} path(s))",
    "cyc6": lambda v: f"funds travelled in a 6-account cycle back to the origin ({int(v)} path(s))",
    "sg_intermediaries": lambda v: f"value was split across {int(v)} intermediary account(s) and re-gathered",
    "sg_value": lambda v: f"USD {v:,.0f} routed through intermediaries between the same two accounts",
    "src_in_distinct_L": lambda v: f"sender received from {int(v)} different accounts in 72h",
    "dst_in_distinct_L": lambda v: f"receiver collected from {int(v)} different accounts in 72h",
    "src_out_distinct_L": lambda v: f"sender paid {int(v)} different accounts in 72h",
    "dst_out_distinct_L": lambda v: f"receiver paid out to {int(v)} different accounts in 72h",
    "pt_amt_over_recv": lambda v: f"outgoing amount was {v:.0%} of what the sender received in the prior 48h",
    "pt_sent_over_recv": lambda v: f"sender forwarded {v:.0%} of its 48h inflow",
    "pt_min_since_in": lambda v: (f"sent within {max(int(v), 1)} minute(s) of the last inbound credit" if v < 60
                                  else f"sent {v / 60:.1f}h after the last inbound credit"),
    "src_out_cnt_S": lambda v: f"sender made {int(v)} transfers in 6h",
    "dst_in_cnt_S": lambda v: f"receiver got {int(v)} transfers in 6h",
    "src_net_flow_L": lambda v: f"sender is a {'collector' if v > 0.3 else 'distributor' if v < -0.3 else 'pass-through'} account (net flow {v:+.2f})",
    "dst_net_flow_L": lambda v: f"receiver is a {'collector' if v > 0.3 else 'distributor' if v < -0.3 else 'pass-through'} account (net flow {v:+.2f})",
    "fmt": lambda v: None,   # filled in with the format name
    "log_amt_usd": lambda v: f"transfer size USD {np.expm1(v):,.0f}",
    "cross_currency": lambda v: "cross-currency transfer" if v > 0 else None,
}


def tier(score: float, n_acc: int) -> tuple[int, str]:
    if score >= 0.9 and n_acc >= 3:
        return 4, "Tier 4 – escalate: hold outbound on all ring members and file STR with NFIU after analyst review."
    if n_acc >= 3:
        return 3, "Tier 3 – hold outbound transfers on ring members pending analyst review."
    if score >= 0.9:
        return 2, "Tier 2 – step-up verification (BVN selfie / in-app confirmation) before next outbound transfer."
    return 1, "Tier 1 – silent monitoring with lowered thresholds."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="HI-Small")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--rings", type=int, default=8)
    args = ap.parse_args()
    d = Path(args.data) / args.name
    meta = json.loads((d / "meta.json").read_text())
    X, y, names, splits, tx = load(d)
    te = splits[2]
    th = json.loads((d / "train_results.json").read_text())["final"]["threshold"]
    booster = xgb.Booster()
    booster.load_model(d / "model.json")
    acc = pl.read_parquet(d / "accounts.parquet")
    att = pl.read_parquet(d / "attempts.parquet")

    scores = pl.read_parquet(d / "test_scores.parquet")
    T = tx.join(scores, on="txn_id").with_row_index("row")
    al = T.filter(pl.col("score") >= th)
    comp = union_find_components(al["src"].to_numpy(), al["dst"].to_numpy())
    ring_of = pl.DataFrame({"acct": list(comp.keys()), "ring": list(comp.values())},
                           schema={"acct": pl.Int32, "ring": pl.Int64})
    al = al.join(ring_of, left_on="src", right_on="acct")
    rings = (al.group_by("ring").agg(n_tx=pl.len(), usd=pl.col("amt_usd").sum(), score=pl.col("score").max(),
                                     mean_score=pl.col("score").mean(), true_share=pl.col("label").mean())
               .join(ring_of.group_by("ring").agg(accts=pl.len()), on="ring")
               .filter(pl.col("accts") >= 3)
               .with_columns(rank=pl.col("mean_score") * pl.col("usd").log1p())
               .sort("rank", descending=True).head(args.rings))

    out_rings = []
    for k, r in enumerate(rings.iter_rows(named=True)):
        e = al.filter(pl.col("ring") == r["ring"]).sort("ts")
        rows = e["row"].to_numpy()
        Xr = X[te][rows]
        contrib = booster.predict(xgb.DMatrix(Xr), pred_contribs=True)[:, :-1]
        # reason codes: features with the largest total positive SHAP across the ring's transactions
        tot = np.clip(contrib, 0, None).sum(axis=0)
        reasons = []
        for j in np.argsort(-tot):
            if tot[j] <= 0 or len(reasons) >= 5:
                break
            n = names[j]
            if n not in REASONS:
                continue
            i = int(np.argmax(contrib[:, j]))
            v = float(Xr[i, j])
            s = (f"payments made via {meta['formats'][int(v)]}" if n == "fmt" else REASONS[n](v))
            if s:
                reasons.append(s[0].upper() + s[1:])
        members = pl.concat([e.select(acct="src"), e.select(acct="dst")]).unique()
        flows = pl.concat([e.select(acct="src", out="amt_usd", inn=pl.lit(0.0)),
                           e.select(acct="dst", out=pl.lit(0.0), inn="amt_usd")]).group_by("acct").agg(
            pl.col("out").sum(), pl.col("inn").sum())
        members = members.join(flows, on="acct").join(acc, left_on="acct", right_on="acct_id", how="left")
        accounts = []
        for m in members.sort("inn", descending=True).iter_rows(named=True):
            role = "source" if m["inn"] < 0.1 * m["out"] else "collector" if m["out"] < 0.1 * m["inn"] else "pass-through"
            accounts.append({"id": int(m["acct"]), "account": m["account"], "bank": m["bank"],
                             "entity_type": m.get("entity_type"), "role": role, "in_usd": m["inn"],
                             "out_usd": m["out"], "is_laundering": bool(m["label"])})
        t0 = meta["t0"]
        base = np.datetime64(t0.replace(" ", "T"))
        fmt_time = lambda s: str(base + np.timedelta64(int(s), "s")).replace("T", " ")[:16]  # noqa: E731
        timeline = [{"txn": f"T{int(t['txn_id'])}", "time": fmt_time(t["ts"]),
                     "from": acc["account"][int(t["src"])], "to": acc["account"][int(t["dst"])],
                     "usd": float(t["amt_usd"]), "format": meta["formats"][int(t["fmt"])],
                     "score": float(t["score"]), "is_laundering": bool(t["label"])} for t in e.iter_rows(named=True)]
        att_ids = [a for a in e["attempt_id"].unique().to_list() if a >= 0]
        typ = att.filter(pl.col("attempt_id").is_in(att_ids))["typology"].unique().to_list()
        tnum, action = tier(r["score"], r["accts"])
        ev = {"ring_id": f"R{k + 1}", "n_accounts": r["accts"], "n_transactions": r["n_tx"],
              "total_usd": r["usd"], "total_ngn": r["usd"] * USD_NGN, "ring_score": r["mean_score"],
              "first_seen": timeline[0]["time"], "last_seen": timeline[-1]["time"],
              "accounts": [{k2: a[k2] for k2 in ("account", "bank", "entity_type", "role")} for a in accounts],
              "timeline": [{k2: t[k2] for k2 in ("txn", "time", "from", "to", "usd", "format")} for t in timeline],
              "reason_codes": reasons, "action": action}
        out_rings.append({**ev, "tier": tnum, "accounts_full": accounts, "timeline_full": timeline,
                          "ground_truth": {"laundering_txn_share": r["true_share"], "known_typologies": typ},
                          "str": draft(ev)})
        print(f"ring R{k + 1}: {r['accts']} accts, {r['n_tx']} txns, USD {r['usd']:,.0f}, "
              f"true-share {r['true_share']:.2f}, typologies {typ}, tier {tnum}")

    demo = {"dataset": meta["name"], "meta": {k: meta[k] for k in ("n_txns", "n_accounts", "n_laundering_txns",
                                                                   "n_attempts", "span_seconds")},
            "features": json.loads((d / "features.json").read_text()),
            "train": json.loads((d / "train_results.json").read_text()),
            "eval": json.loads((d / "eval_rings.json").read_text()),
            "bench": json.loads((d / "bench.json").read_text()) if (d / "bench.json").exists() else None,
            "rings": out_rings}
    demo["features"].pop("names", None)
    (d / "demo.json").write_text(json.dumps(demo, indent=1, default=str))
    print("wrote", d / "demo.json")


if __name__ == "__main__":
    main()

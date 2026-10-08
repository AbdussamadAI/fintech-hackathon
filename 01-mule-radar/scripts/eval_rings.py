"""Business-level evaluation on the test period: rings, stoppable money, and wrongful flags.

Needs: transactions.parquet, attempts.parquet, test_scores.parquet, train_results.json (threshold).

Reports
  * ring recall          share of known laundering attempts with >= 1 alerted transaction
  * time / value to alert how far into each attempt the first alert fires, and the share of the
                          attempt's USD value that moved *after* the alert (what a hold could have stopped)
  * account-level         precision / recall of flagged accounts
  * inclusion             innocent accounts flagged per 10k active accounts: MuleRadar vs a typical
                          velocity/threshold rule tuned to the same account recall
  * rings                 connected components of alerted transactions (what the analyst sees)

Usage: python scripts/eval_rings.py --name HI-Small
"""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl


def union_find_components(u: np.ndarray, v: np.ndarray) -> dict[int, int]:
    parent: dict[int, int] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(u.tolist(), v.tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return {x: find(x) for x in parent}


def account_flags(tx: pl.DataFrame, flag_col: str) -> pl.DataFrame:
    f = tx.filter(pl.col(flag_col))
    return pl.concat([f.select(acct="src"), f.select(acct="dst")]).unique()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="HI-Small")
    ap.add_argument("--data", default="data/processed")
    args = ap.parse_args()
    d = Path(args.data) / args.name
    scores = pl.read_parquet(d / "test_scores.parquet")
    th = json.loads((d / "train_results.json").read_text())["final"]["threshold"]
    tx = pl.read_parquet(d / "transactions.parquet").join(scores, on="txn_id")   # test period only
    tx = tx.with_columns(alert=pl.col("score") >= th)
    att = pl.read_parquet(d / "attempts.parquet")
    out = {"threshold": th, "test_txns": tx.height}

    # --- rings (known attempts fully inside the test period) ---
    t_start = tx["ts"].min()
    att_te = att.filter(pl.col("first_ts") >= t_start)
    a_tx = tx.filter(pl.col("attempt_id") >= 0).sort("ts")
    per = (a_tx.group_by("attempt_id").agg(
               usd=pl.col("amt_usd").sum(), n=pl.len(), first_ts=pl.col("ts").min(),
               alert_ts=pl.col("ts").filter(pl.col("alert")).min(),
               usd_after=pl.col("amt_usd").filter(pl.col("ts") >= pl.col("ts").filter(pl.col("alert")).min()).sum(),
               idx_first_alert=pl.col("alert").arg_max())
             .join(att_te.select("attempt_id", "typology"), on="attempt_id"))
    det = per.filter(pl.col("alert_ts").is_not_null())
    out["rings"] = {
        "known_attempts_in_test": per.height,
        "detected": det.height,
        "ring_recall": det.height / max(per.height, 1),
        "median_hours_to_first_alert": float(((det["alert_ts"] - det["first_ts"]) / 3600).median() or 0),
        "median_txns_before_first_alert": float(det["idx_first_alert"].median() or 0),
        "stoppable_value_share": float(det["usd_after"].sum() / max(per["usd"].sum(), 1e-9)),
        "by_typology": {r["typology"]: f"{r['det']}/{r['n']}" for r in
                        per.group_by("typology").agg(n=pl.len(), det=pl.col("alert_ts").is_not_null().sum())
                           .sort("typology").iter_rows(named=True)},
    }

    # --- account level + inclusion ---
    active = pl.concat([tx.select(acct="src"), tx.select(acct="dst")]).unique()
    bad = account_flags(tx.with_columns(lab=pl.col("label") == 1), "lab")
    n_active, bad_set = active.height, set(bad["acct"].to_list())

    def acct_metrics(flagged: pl.DataFrame) -> dict:
        fs = set(flagged["acct"].to_list())
        tp = len(fs & bad_set)
        fp = len(fs) - tp
        return {"flagged": len(fs), "precision": tp / max(len(fs), 1), "recall": tp / max(len(bad_set), 1),
                "innocent_flagged_per_10k": fp / n_active * 1e4}

    ours = acct_metrics(account_flags(tx, "alert"))
    # Typical rule: flag an account whose 24h outbound USD or outbound count exceeds a threshold.
    # Sweep the amount threshold to match our account recall, then compare wrongful flags.
    daily = (tx.with_columns(day=pl.col("ts") // 86400).group_by("src", "day")
               .agg(usd=pl.col("amt_usd").sum(), n=pl.len()))
    best = None
    for q in np.linspace(0.5, 0.9999, 400):
        cut_usd = daily["usd"].quantile(q)
        flagged = daily.filter((pl.col("usd") >= cut_usd)).select(acct="src").unique()
        m = acct_metrics(flagged)
        if m["recall"] >= ours["recall"]:
            best = (q, cut_usd, m)
    out["accounts"] = {"active_accounts": n_active, "laundering_accounts": len(bad_set), "muleradar": ours}
    if best:
        q, cut_usd, m = best
        out["accounts"]["threshold_rule_same_recall"] = {"rule": f"24h outbound >= ${cut_usd:,.0f} (q={q:.4f})", **m}
        out["accounts"]["wrongful_flags_reduction"] = 1 - ours["innocent_flagged_per_10k"] / max(m["innocent_flagged_per_10k"], 1e-9)

    # --- alert rings (connected components of alerted txns) ---
    al = tx.filter(pl.col("alert"))
    comp = union_find_components(al["src"].to_numpy(), al["dst"].to_numpy())
    sizes = pl.DataFrame({"acct": list(comp.keys()), "ring": list(comp.values())}).group_by("ring").len()
    out["alert_rings"] = {"n_rings": sizes.height, "accounts_in_rings": int(sizes["len"].sum()),
                          "median_ring_size": float(sizes["len"].median() or 0),
                          "max_ring_size": int(sizes["len"].max() or 0)}
    (d / "eval_rings.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

"""Quick EDA on processed data: the numbers that drive window size, thresholds and the pitch.

Usage: python scripts/eda.py --name HI-Small
"""
import argparse
import json
from pathlib import Path

import polars as pl

HOUR = 3600


def q(s: pl.Series, qs=(0.5, 0.9, 0.99)) -> str:
    return " ".join(f"p{int(x * 100)}={s.quantile(x):,.1f}" for x in qs) + f" max={s.max():,.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="HI-Small")
    ap.add_argument("--data", default="data/processed")
    args = ap.parse_args()
    d = Path(args.data) / args.name
    tx = pl.read_parquet(d / "transactions.parquet")
    acc = pl.read_parquet(d / "accounts.parquet")
    att = pl.read_parquet(d / "attempts.parquet")
    meta = json.loads((d / "meta.json").read_text())

    print(f"== {args.name}: {tx.height:,} txns, {acc.height:,} accounts, span {meta['span_seconds'] / 86400:.1f} days")
    print(f"laundering txns {meta['n_laundering_txns']:,} ({meta['laundering_rate']:.4%}), "
          f"of which in a known attempt {meta['laundering_txns_in_attempts']:,}; "
          f"laundering accounts {meta['n_laundering_accounts']:,} ({meta['n_laundering_accounts'] / acc.height:.3%})")

    # volume over time: is the tail of the span sparse? (affects the time split)
    per_day = tx.group_by((pl.col("ts") // 86400).alias("day")).agg(n=pl.len(), bad=pl.col("label").sum()).sort("day")
    print("\n-- txns per day (day, n, laundering)")
    print(per_day.to_pandas().to_string(index=False))

    # attempt durations by typology -> window size
    att = att.with_columns(hours=(pl.col("last_ts") - pl.col("first_ts")) / HOUR)
    print("\n-- laundering attempts by typology (duration hours, txns, USD)")
    print(att.group_by("typology").agg(
        n=pl.len(), med_hours=pl.col("hours").median(), p90_hours=pl.col("hours").quantile(0.9),
        med_txns=pl.col("n_txns").median(), med_usd=pl.col("usd").median()).sort("typology").to_pandas().to_string(index=False))
    print("all attempts duration hours:", q(att["hours"]))

    # gaps between consecutive txns inside an attempt -> hop window (delta t)
    gaps = (tx.filter(pl.col("attempt_id") >= 0).sort(["attempt_id", "ts"])
              .with_columns(gap=pl.col("ts").diff().over("attempt_id") / HOUR).drop_nulls("gap"))
    print("intra-attempt gap hours:", q(gaps["gap"]))

    # pass-through: for laundering accounts, delay from an incoming laundering txn to the next outgoing one
    bad = tx.filter(pl.col("label") == 1)
    ins = bad.select(acct="dst", t_in="ts", a_in="amt_usd").sort("t_in")
    outs = bad.select(acct="src", t_out="ts", a_out="amt_usd").sort("t_out")
    pt = ins.join_asof(outs, left_on="t_in", right_on="t_out", by="acct", strategy="forward").drop_nulls("t_out")
    pt = pt.with_columns(delay_h=(pl.col("t_out") - pl.col("t_in")) / HOUR, ratio=pl.col("a_out") / pl.col("a_in"))
    print(f"\n-- pass-through in->next-out on laundering edges ({pt.height:,} pairs)")
    print("delay hours:", q(pt["delay_h"]))
    print("out/in amount ratio:", q(pt["ratio"], (0.1, 0.5, 0.9)))

    # degree distributions (whole span): laundering vs clean accounts
    deg = (pl.concat([tx.select(acct="src", o=pl.lit(1), i=pl.lit(0)), tx.select(acct="dst", o=pl.lit(0), i=pl.lit(1))])
             .group_by("acct").agg(out_deg=pl.col("o").sum(), in_deg=pl.col("i").sum()))
    deg = deg.join(acc.select(acct="acct_id", label="label"), on="acct")
    for lab in (0, 1):
        s = deg.filter(pl.col("label") == lab)
        print(f"label={lab}: out_deg {q(s['out_deg'].cast(pl.Float64))} | in_deg {q(s['in_deg'].cast(pl.Float64))}")

    print("\n-- payment format mix (share of txns, laundering rate)")
    fm = tx.group_by("fmt").agg(n=pl.len(), rate=pl.col("label").mean()).sort("fmt")
    fm = fm.with_columns(name=pl.col("fmt").map_elements(lambda i: meta["formats"][i], return_dtype=pl.Utf8),
                         share=pl.col("n") / tx.height)
    print(fm.select("name", "share", "rate").to_pandas().to_string(index=False))

    self_loops = tx.filter(pl.col("src") == pl.col("dst")).height
    print(f"\nself-loop txns (same account): {self_loops:,} ({self_loops / tx.height:.2%})")
    print("split cut-offs (days):", {k: v / 86400 for k, v in meta["split"].items() if k.endswith("_ts")})


if __name__ == "__main__":
    main()

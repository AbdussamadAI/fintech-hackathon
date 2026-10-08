"""Convert an IBM AML dataset (e.g. HI-Small) into compact Parquet for the engine.

Outputs (in data/processed/<name>/):
  transactions.parquet  one row per transaction, time-sorted
      txn_id int64, ts int64 (seconds since first txn), src int32, dst int32,
      amt_paid, amt_recv, amt_usd float64, pay_cur, recv_cur, fmt int8,
      label int8, attempt_id int32 (-1 if not in a known laundering attempt)
  accounts.parquet      acct_id int32, bank, account, entity_id, entity_type,
                        label int8 (1 if involved in any laundering txn)
  attempts.parquet      attempt_id, typology, detail, n_txns, first_ts, last_ts
  meta.json             code tables (currencies, payment formats), split cut-offs, counts

Accounts are keyed by (bank, account): the raw account string alone is not unique.
Amounts are normalised to USD with rates derived from the data's own cross-currency txns.

Usage: python scripts/prepare_data.py --name HI-Small [--raw data/raw] [--out data/processed]
"""
import argparse
import json
import re
import time
from pathlib import Path

import polars as pl

TRANS_COLS = ["timestamp", "from_bank", "from_acct", "to_bank", "to_acct", "amt_recv",
              "recv_cur", "amt_paid", "pay_cur", "fmt", "label"]
TS_FMT = "%Y/%m/%d %H:%M"


def read_transactions(path: Path) -> pl.DataFrame:
    df = pl.read_csv(path, has_header=True, new_columns=TRANS_COLS,
                     schema_overrides={"from_bank": pl.Utf8, "to_bank": pl.Utf8,
                                       "from_acct": pl.Utf8, "to_acct": pl.Utf8})
    return df.with_columns(pl.col("timestamp").str.strptime(pl.Datetime, TS_FMT))


def read_patterns(path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Parse the Patterns file into (attempt table, attempt transactions)."""
    begin = re.compile(r"BEGIN LAUNDERING ATTEMPT - ([A-Z-]+)(?::\s*(.*))?")
    attempts, rows, cur = [], [], None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        m = begin.match(line)
        if m:
            cur = len(attempts)
            attempts.append({"attempt_id": cur, "typology": m.group(1),
                             "detail": (m.group(2) or "").strip()})
        elif line.startswith("END LAUNDERING ATTEMPT"):
            cur = None
        elif cur is not None:
            rows.append(line.split(",") + [cur])
    tx = pl.DataFrame(rows, schema=TRANS_COLS + ["attempt_id"], orient="row")
    tx = tx.with_columns(
        pl.col("timestamp").str.strptime(pl.Datetime, TS_FMT),
        pl.col("amt_recv").cast(pl.Float64), pl.col("amt_paid").cast(pl.Float64),
        pl.col("label").cast(pl.Int64), pl.col("attempt_id").cast(pl.Int32))
    return pl.DataFrame(attempts), tx


def usd_rates(df: pl.DataFrame) -> dict[str, float]:
    """USD value of one unit of each currency, from cross-currency USD transactions."""
    rates = {"US Dollar": 1.0}
    to_usd = (df.filter((pl.col("recv_cur") == "US Dollar") & (pl.col("pay_cur") != "US Dollar")
                        & (pl.col("amt_paid") > 0))
                .group_by("pay_cur").agg((pl.col("amt_recv") / pl.col("amt_paid")).median().alias("r")))
    rates.update(dict(zip(to_usd["pay_cur"], to_usd["r"])))
    from_usd = (df.filter((pl.col("pay_cur") == "US Dollar") & (pl.col("recv_cur") != "US Dollar")
                          & (pl.col("amt_recv") > 0))
                  .group_by("recv_cur").agg((pl.col("amt_paid") / pl.col("amt_recv")).median().alias("r")))
    for c, r in zip(from_usd["recv_cur"], from_usd["r"]):
        rates.setdefault(c, r)
    return rates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="HI-Small")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--splits", default="0.6,0.2", help="train,val fractions of txns (time-ordered)")
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df = read_transactions(raw / f"{args.name}_Trans.csv")
    print(f"read {df.height:,} txns in {time.time() - t0:.1f}s")

    # --- dense account ids keyed by (bank, account) ---
    keys = pl.concat([df.select(bank="from_bank", account="from_acct"),
                      df.select(bank="to_bank", account="to_acct")]).unique()
    acct_path = raw / f"{args.name}_accounts.csv"
    if acct_path.exists():
        info = pl.read_csv(acct_path, schema_overrides={"Bank ID": pl.Utf8, "Account Number": pl.Utf8}) \
                 .select(bank="Bank ID", account="Account Number", entity_id="Entity ID",
                         entity_name="Entity Name").unique(subset=["bank", "account"])
        # bank ids in the accounts file are not zero-padded; normalise both sides
        keys = keys.with_columns(bank_n=pl.col("bank").str.strip_chars_start("0"))
        info = info.with_columns(bank_n=pl.col("bank").str.strip_chars_start("0")).drop("bank")
        keys = keys.join(info, on=["bank_n", "account"], how="left").drop("bank_n")
        keys = keys.with_columns(entity_type=pl.col("entity_name").str.replace(r"\s*#\d+$", ""))
    keys = keys.sort(["bank", "account"]).with_row_index("acct_id").with_columns(pl.col("acct_id").cast(pl.Int32))

    df = (df.join(keys.select("bank", "account", src="acct_id"),
                  left_on=["from_bank", "from_acct"], right_on=["bank", "account"])
            .join(keys.select("bank", "account", dst="acct_id"),
                  left_on=["to_bank", "to_acct"], right_on=["bank", "account"]))

    # --- attach known laundering attempts ---
    attempts, ptx = read_patterns(raw / f"{args.name}_Patterns.txt")
    jk = ["timestamp", "from_bank", "from_acct", "to_bank", "to_acct", "amt_paid", "pay_cur", "fmt"]
    ptx = ptx.unique(subset=jk, keep="first").select(jk + ["attempt_id"])
    df = df.join(ptx, on=jk, how="left").with_columns(pl.col("attempt_id").fill_null(-1))

    # --- codes, USD amounts, time ---
    rates = usd_rates(df)
    currencies = sorted(set(df["pay_cur"].unique()) | set(df["recv_cur"].unique()))
    formats = sorted(df["fmt"].unique())
    t_min = df["timestamp"].min()
    df = (df.sort(["timestamp", "src", "dst"])
            .with_row_index("txn_id")
            .with_columns(
                ts=((pl.col("timestamp") - t_min).dt.total_seconds()).cast(pl.Int64),
                amt_usd=pl.col("amt_paid") * pl.col("pay_cur").replace_strict(rates, default=None),
                pay_cur=pl.col("pay_cur").replace_strict({c: i for i, c in enumerate(currencies)}).cast(pl.Int8),
                recv_cur=pl.col("recv_cur").replace_strict({c: i for i, c in enumerate(currencies)}).cast(pl.Int8),
                fmt=pl.col("fmt").replace_strict({f: i for i, f in enumerate(formats)}).cast(pl.Int8),
                label=pl.col("label").cast(pl.Int8),
                txn_id=pl.col("txn_id").cast(pl.Int64))
            .select("txn_id", "ts", "src", "dst", "amt_paid", "amt_recv", "amt_usd",
                    "pay_cur", "recv_cur", "fmt", "label", "attempt_id"))
    assert df["amt_usd"].null_count() == 0, "missing USD rate for some currency"

    # --- account labels & attempt stats ---
    bad = pl.concat([df.filter(pl.col("label") == 1).select(acct_id="src"),
                     df.filter(pl.col("label") == 1).select(acct_id="dst")]).unique()
    accounts = keys.join(bad.with_columns(label=pl.lit(1, pl.Int8)), on="acct_id", how="left") \
                   .with_columns(pl.col("label").fill_null(0))
    att = (df.filter(pl.col("attempt_id") >= 0).group_by("attempt_id")
             .agg(n_txns=pl.len(), first_ts=pl.col("ts").min(), last_ts=pl.col("ts").max(),
                  usd=pl.col("amt_usd").sum()))
    attempts = attempts.join(att, on="attempt_id", how="left").sort("attempt_id")

    # --- time-ordered split cut-offs (same convention as the IBM/Multi-GNN papers: 60/20/20 by txn) ---
    f_train, f_val = (float(x) for x in args.splits.split(","))
    ts_sorted = df["ts"].to_numpy()
    cut_train = int(ts_sorted[int(len(ts_sorted) * f_train)])
    cut_val = int(ts_sorted[int(len(ts_sorted) * (f_train + f_val))])

    df.write_parquet(out / "transactions.parquet")
    accounts.write_parquet(out / "accounts.parquet")
    attempts.write_parquet(out / "attempts.parquet")
    n_lab = int(df["label"].sum())
    meta = {
        "name": args.name, "t0": str(t_min), "span_seconds": int(ts_sorted[-1]),
        "n_txns": df.height, "n_accounts": accounts.height, "n_laundering_txns": n_lab,
        "laundering_rate": n_lab / df.height,
        "n_laundering_accounts": int(accounts["label"].sum()),
        "n_attempts": attempts.height,
        "laundering_txns_in_attempts": int(df.filter((pl.col("label") == 1) & (pl.col("attempt_id") >= 0)).height),
        "currencies": currencies, "usd_rates": rates, "formats": formats,
        "split": {"train_end_ts": cut_train, "val_end_ts": cut_val, "fractions": [f_train, f_val]},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps({k: v for k, v in meta.items() if k not in ("usd_rates",)}, indent=2, default=str))
    print(f"done in {time.time() - t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()

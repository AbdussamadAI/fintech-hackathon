"""Train + evaluate the transaction-level laundering classifier (XGBoost on GPU).

Protocol matches the published IBM AML benchmarks (Altman et al. 2023; Egressy et al. AAAI'24;
Blanuša et al. ICAIF'24): time-ordered 60/20/20 split by transaction, minority-class (laundering) F1
on the test split, decision threshold chosen on validation.

Usage: python scripts/train.py --name HI-Small [--seeds 3] [--ablation]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve

GROUPS = {   # ablation ladder: each row adds a feature family
    "edge only": ["edge"],
    "+ account velocity": ["edge", "node"],
    "+ pass-through": ["edge", "node", "pass"],
    "+ scatter-gather": ["edge", "node", "pass", "sg"],
    "+ temporal cycles (full)": ["edge", "node", "pass", "sg", "cyc"],
}


def family(name: str) -> str:
    if name.startswith(("src_", "dst_")):
        return "node"
    if name.startswith("pt_"):
        return "pass"
    if name.startswith("sg_"):
        return "sg"
    if name.startswith("cyc"):
        return "cyc"
    return "edge"


def load(d: Path):
    tx = pl.read_parquet(d / "transactions.parquet")
    meta = json.loads((d / "meta.json").read_text())
    finfo = json.loads((d / "features.json").read_text())
    F = np.load(d / "features.npy")
    hour = (tx["ts"].to_numpy() // 3600) % 24
    edge = np.column_stack([
        np.log1p(tx["amt_usd"].to_numpy()), tx["fmt"].to_numpy(), tx["pay_cur"].to_numpy(),
        tx["recv_cur"].to_numpy(), (tx["pay_cur"] != tx["recv_cur"]).to_numpy(),
        (tx["src"] == tx["dst"]).to_numpy(), hour]).astype(np.float32)
    edge_names = ["log_amt_usd", "fmt", "pay_cur", "recv_cur", "cross_currency", "self_loop", "hour"]
    X = np.concatenate([edge, F], axis=1)
    names = edge_names + finfo["names"]
    y = tx["label"].to_numpy().astype(np.int32)
    n = len(y)
    i_tr, i_va = int(n * meta["split"]["fractions"][0]), int(n * sum(meta["split"]["fractions"]))
    return X, y, names, (slice(0, i_tr), slice(i_tr, i_va), slice(i_va, n)), tx


def fit_eval(X, y, splits, cols, seed, device="cuda", lr=0.05):
    tr, va, te = splits
    pos = y[tr].mean()
    clf = xgb.XGBClassifier(
        device=device, tree_method="hist", n_estimators=3000, learning_rate=lr, max_depth=8,
        min_child_weight=1, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        scale_pos_weight=min((1 - pos) / pos, 50.0), eval_metric="aucpr", early_stopping_rounds=max(50, int(5 / lr)),
        random_state=seed)
    clf.fit(X[tr][:, cols], y[tr], eval_set=[(X[va][:, cols], y[va])], verbose=False)
    p_va = clf.predict_proba(X[va][:, cols])[:, 1]
    p_te = clf.predict_proba(X[te][:, cols])[:, 1]
    prec, rec, thr = precision_recall_curve(y[va], p_va)
    f1s = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    th = float(thr[np.argmax(f1s[:-1])])
    yhat = (p_te >= th).astype(int)
    return {"f1": f1_score(y[te], yhat), "pr_auc": average_precision_score(y[te], p_te),
            "precision": float((yhat & y[te]).sum() / max(yhat.sum(), 1)),
            "recall": float((yhat & y[te]).sum() / max(y[te].sum(), 1)),
            "threshold": th, "best_iter": int(clf.best_iteration)}, clf, p_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="HI-Small")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lr", type=float, default=0.05)
    args = ap.parse_args()
    d = Path(args.data) / args.name
    X, y, names, splits, tx = load(d)
    fam = np.array([family(n) for n in names])
    print(f"{len(y):,} txns, {len(names)} features; test laundering txns: {y[splits[2]].sum():,}", flush=True)

    ladder = GROUPS if args.ablation else {"+ temporal cycles (full)": GROUPS["+ temporal cycles (full)"]}
    results = {}
    for label, fams in ladder.items():
        cols = np.where(np.isin(fam, fams))[0]
        runs = [fit_eval(X, y, splits, cols, seed, args.device, args.lr)[0] for seed in range(args.seeds)]
        agg = {k: (float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs])))
               for k in ("f1", "pr_auc", "precision", "recall")}
        results[label] = agg
        print(f"{label:28s} F1 {agg['f1'][0] * 100:5.2f} ± {agg['f1'][1] * 100:4.2f} | "
              f"PR-AUC {agg['pr_auc'][0] * 100:5.2f} | P {agg['precision'][0] * 100:5.1f} R {agg['recall'][0] * 100:5.1f}", flush=True)

    # final model (seed 0, all features): save test scores + importances for downstream ring/alert eval
    cols = np.arange(len(names))
    res, clf, p_te = fit_eval(X, y, splits, cols, 0, args.device, args.lr)
    te = splits[2]
    pl.DataFrame({"txn_id": tx["txn_id"].to_numpy()[te], "score": p_te}).write_parquet(d / "test_scores.parquet")
    clf.get_booster().save_model(d / "model.json")
    imp = sorted(zip(names, clf.feature_importances_), key=lambda t: -t[1])[:15]
    print("top features:", ", ".join(f"{n} {v:.3f}" for n, v in imp))
    (d / "train_results.json").write_text(json.dumps({"ladder": results, "final": res,
                                                      "top_features": [(n, float(v)) for n, v in imp]}, indent=2))


if __name__ == "__main__":
    main()

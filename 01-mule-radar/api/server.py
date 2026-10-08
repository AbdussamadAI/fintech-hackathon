"""MuleRadar live server: replays the transaction stream through the CUDA engine + model on the GPU node
and streams alerts / rings / throughput to the browser.

Run on the GPU node (inside the Slurm allocation), then tunnel the port to your laptop:
    slurm/run.sh python api/server.py --port 8050
    ssh -O forward -L 8050:<node>:8050 slurm        # from the laptop (see slurm/live.sh)
    open http://localhost:8050

Endpoints
    GET  /                 live console (dashboard/live.html)
    GET  /results          static results page (dashboard/index.html)
    GET  /api/stream       Server-Sent Events: a snapshot ~4x per second
    POST /api/control      {"action": "play"|"pause"|"reset", "speed": <sim seconds per real second | 0 = max>,
                            "engine": "gpu"|"cpu", "batch": <txns per micro-batch>}
    GET  /api/ring/{root}  ring detail: members, transfers, reason codes, STR draft
"""
import argparse
import asyncio
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cupy as cp
import numpy as np
import polars as pl
import uvicorn
import xgboost as xgb
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline_cpu.features_cpu import CpuFeatureEngine  # noqa: E402
from engine.features import EngineConfig, GpuFeatureEngine, feature_names  # noqa: E402
from engine.str_draft import draft  # noqa: E402
from scripts.export_demo import REASONS, USD_NGN, tier  # noqa: E402

EDGE_NAMES = ["log_amt_usd", "fmt", "pay_cur", "recv_cur", "cross_currency", "self_loop", "hour"]


class Rings:
    """Union-find over alerted transfers; keeps per-ring aggregates up to date."""

    def __init__(self):
        self.parent, self.edges = {}, {}           # acct -> parent; root -> list of alert dicts

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def add(self, a: dict):
        ra, rb = self.find(a["src"]), self.find(a["dst"])
        if ra != rb:
            big, small = (ra, rb) if len(self.edges.get(ra, [])) >= len(self.edges.get(rb, [])) else (rb, ra)
            self.parent[small] = big
            self.edges.setdefault(big, []).extend(self.edges.pop(small, []))
            ra = big
        self.edges.setdefault(ra, []).append(a)

    def members(self, root):
        return {x for e in self.edges.get(root, []) for x in (e["src"], e["dst"])}

    def summary(self, root):
        es = self.edges[root]
        n_acc = len(self.members(root))
        score = max(e["score"] for e in es)
        return {"root": int(root), "accounts": n_acc, "txns": len(es), "usd": sum(e["usd"] for e in es),
                "score": score, "mean_score": float(np.mean([e["score"] for e in es])),
                "true_share": float(np.mean([e["label"] for e in es])),
                "first_ts": min(e["ts"] for e in es), "last_ts": max(e["ts"] for e in es),
                "tier": tier(score, n_acc)[0]}

    def top(self, k=12):
        rs = [self.summary(r) for r, es in self.edges.items() if self.find(r) == r and es]
        rs = [r for r in rs if r["accounts"] >= 3] or rs
        return sorted(rs, key=lambda r: (-r["tier"], -r["mean_score"] * np.log1p(r["usd"])))[:k]


class Live:
    def __init__(self, name: str, data: Path):
        d = data / name
        self.meta = json.loads((d / "meta.json").read_text())
        tx = pl.read_parquet(d / "transactions.parquet")
        self.acc = pl.read_parquet(d / "accounts.parquet")
        self.n = tx.height
        self.cols = {c: tx[c].to_numpy() for c in tx.columns}
        c = self.cols
        hour = (c["ts"] // 3600) % 24
        self.edge = np.column_stack([np.log1p(c["amt_usd"]), c["fmt"], c["pay_cur"], c["recv_cur"],
                                     c["pay_cur"] != c["recv_cur"], c["src"] == c["dst"], hour]).astype(np.float32)
        self.names = EDGE_NAMES + feature_names()
        args = (c["src"], c["dst"], c["ts"], c["amt_usd"])
        cfg = EngineConfig()
        self.engines = {"gpu": GpuFeatureEngine(*args, n_accounts=self.acc.height, cfg=cfg),
                        "cpu": CpuFeatureEngine(*args, n_accounts=self.acc.height, cfg=cfg)}
        self.models = {}
        for dev in ("gpu", "cpu"):
            b = xgb.Booster()
            b.load_model(d / "model.json")
            b.set_param({"device": "cuda" if dev == "gpu" else "cpu", "nthread": 8})
            self.models[dev] = b
        self.th = json.loads((d / "train_results.json").read_text())["final"]["threshold"]
        self.test_start = int(self.n * sum(self.meta["split"]["fractions"]))
        self.t0 = np.datetime64(self.meta["t0"].replace(" ", "T"))
        self.lock = threading.Lock()
        self.speed, self.engine, self.batch, self.running = 3600.0, "gpu", 2048, False
        self.reset()
        # warm up both paths (JIT / kernel compile) so the first live batch is representative
        for e in ("gpu", "cpu"):
            self._process(self.test_start, self.test_start + 256, e, record=False)
        threading.Thread(target=self._loop, daemon=True).start()

    # ---------------------------------------------------------------- state
    def reset(self):
        with self.lock:
            self.idx = self.test_start
            self.sim_origin = int(self.cols["ts"][self.test_start])
            self.sim_now = self.sim_origin
            self.rings, self.alerts, self.xrows = Rings(), deque(maxlen=20), {}
            self.n_alerts = self.tp = self.processed = 0
            self.lat = deque(maxlen=200)
            self.rate = deque(maxlen=120)               # (wall, processed, engine)
            self.busy = 0.0
            self.clock_wall = time.perf_counter()

    def control(self, body: dict):
        with self.lock:
            now = time.perf_counter()
            self.sim_now = self._target(now)
            self.clock_wall = now
            if "speed" in body:
                self.speed = float(body["speed"])
            if self.speed > 0 and ("speed" in body or body.get("action") == "play"):
                # restart the paced clock at the stream position (e.g. after "max"); engine switches keep the lag
                self.sim_now = min(self.sim_now, int(self.cols["ts"][min(self.idx, self.n - 1)]))
            if "engine" in body and body["engine"] in self.engines:
                self.engine = body["engine"]
            if "batch" in body:
                self.batch = int(np.clip(int(body["batch"]), 128, 65536))
            act = body.get("action")
            if act == "play":
                self.running = True
            elif act == "pause":
                self.running = False
        if act == "reset":
            self.running = False
            self.reset()

    def _target(self, now):
        """Simulated time the stream should have reached by wall time `now`."""
        if not self.running:
            return self.sim_now
        if self.speed <= 0:
            return 1 << 62
        return int(self.sim_now + (now - self.clock_wall) * self.speed)

    # ---------------------------------------------------------------- processing
    def _process(self, b0, b1, eng, record=True):
        t = time.perf_counter()
        F = self.engines[eng].batch_features(b0, b1)
        F = F.get() if isinstance(F, cp.ndarray) else F
        X = np.concatenate([self.edge[b0:b1], F], axis=1)
        p = self.models[eng].inplace_predict(X)
        p = cp.asnumpy(p) if isinstance(p, cp.ndarray) else np.asarray(p)
        dt = time.perf_counter() - t
        if not record:
            return
        c = self.cols
        hit = np.where(p >= self.th)[0]
        with self.lock:
            for i in hit:
                j = b0 + int(i)
                a = {"txn": int(c["txn_id"][j]), "src": int(c["src"][j]), "dst": int(c["dst"][j]),
                     "ts": int(c["ts"][j]), "usd": float(c["amt_usd"][j]), "fmt": int(c["fmt"][j]),
                     "score": float(p[i]), "label": int(c["label"][j])}
                self.rings.add(a)
                self.alerts.appendleft(a)
                self.xrows[a["txn"]] = X[i]
                self.n_alerts += 1
                self.tp += a["label"]
            self.processed += b1 - b0
            self.lat.append(dt * 1e3)
            self.busy += dt
            self.rate.append((time.perf_counter(), self.processed, eng))

    def _loop(self):
        ts = self.cols["ts"]
        while True:
            with self.lock:
                running, eng, bs, idx = self.running, self.engine, self.batch, self.idx
                target = self._target(time.perf_counter())
            if not running or idx >= self.n:
                time.sleep(0.05)
                continue
            # take up to `bs` transfers that are "due" by the simulated clock
            b1 = min(idx + bs, self.n)
            if target < (1 << 62):
                b1 = min(b1, int(np.searchsorted(ts, target, side="right")))
            if b1 <= idx:
                time.sleep(0.01)
                continue
            self._process(idx, b1, eng)
            with self.lock:
                self.idx = b1

    # ---------------------------------------------------------------- views
    def fmt_time(self, s):
        return str(self.t0 + np.timedelta64(int(s), "s")).replace("T", " ")[:16]

    def snapshot(self):
        with self.lock:
            now = time.perf_counter()
            target = self._target(now)
            stream_ts = int(self.cols["ts"][min(self.idx, self.n - 1)])
            r = [x for x in self.rate if x[0] >= now - 3.0]
            tps = (r[-1][1] - r[0][1]) / max(r[-1][0] - r[0][0], 1e-6) if len(r) >= 2 else 0.0
            lat = list(self.lat)
            due = int(np.searchsorted(self.cols["ts"], target, side="right")) if target < (1 << 62) else self.n
            snap = {
                "running": self.running, "engine": self.engine, "speed": self.speed, "batch": self.batch,
                "sim_time": self.fmt_time(stream_ts), "processed": self.processed,
                "remaining": self.n - self.idx, "done": self.idx >= self.n,
                "txn_per_s": tps,
                "capacity_txn_per_s": self.processed / self.busy if self.busy else 0.0,
                "lat_p50": float(np.percentile(lat, 50)) if lat else 0.0,
                "lat_p99": float(np.percentile(lat, 99)) if lat else 0.0,
                "backlog_txns": max(due - self.idx, 0) if self.running and self.speed > 0 else 0,
                "lag_minutes": max(target - stream_ts, 0) / 60 if self.running and self.speed > 0 and due > self.idx else 0.0,
                "alerts": self.n_alerts, "alert_precision": self.tp / self.n_alerts if self.n_alerts else None,
                "rings": [{**x, "first": self.fmt_time(x["first_ts"]), "last": self.fmt_time(x["last_ts"]),
                           "ngn": x["usd"] * USD_NGN} for x in self.rings.top()],
                "recent": [{**a, "time": self.fmt_time(a["ts"]), "from": self.acc["account"][a["src"]],
                            "to": self.acc["account"][a["dst"]], "format": self.meta["formats"][a["fmt"]]}
                           for a in list(self.alerts)[:12]],
            }
        return snap

    def ring(self, root: int):
        with self.lock:
            root = self.rings.find(root)
            es = sorted(self.rings.edges.get(root, []), key=lambda e: e["ts"])
            xs = np.array([self.xrows[e["txn"]] for e in es], dtype=np.float32)
        if not es:
            return None
        contrib = self.models["gpu"].predict(xgb.DMatrix(xs), pred_contribs=True)[:, :-1]
        tot = np.clip(contrib, 0, None).sum(axis=0)
        reasons = []
        for j in np.argsort(-tot):
            if tot[j] <= 0 or len(reasons) >= 5:
                break
            n = self.names[j]
            if n not in REASONS:
                continue
            v = float(xs[int(np.argmax(contrib[:, j])), j])
            s = f"payments made via {self.meta['formats'][int(v)]}" if n == "fmt" else REASONS[n](v)
            if s:
                reasons.append(s[0].upper() + s[1:])
        flows = {}
        for e in es:
            flows.setdefault(e["src"], [0.0, 0.0])[1] += e["usd"]
            flows.setdefault(e["dst"], [0.0, 0.0])[0] += e["usd"]
        accounts = []
        for a, (inn, out) in sorted(flows.items(), key=lambda kv: -kv[1][0]):
            row = self.acc.row(a, named=True)
            role = "source" if inn < 0.1 * out else "collector" if out < 0.1 * inn else "pass-through"
            accounts.append({"account": row["account"], "bank": row["bank"], "entity_type": row.get("entity_type"),
                             "role": role, "in_usd": inn, "out_usd": out, "is_laundering": bool(row["label"])})
        timeline = [{"txn": f"T{e['txn']}", "time": self.fmt_time(e["ts"]), "from": self.acc["account"][e["src"]],
                     "to": self.acc["account"][e["dst"]], "usd": e["usd"], "format": self.meta["formats"][e["fmt"]],
                     "score": e["score"], "is_laundering": bool(e["label"])} for e in es]
        s = self.rings.summary(root)
        tnum, action = tier(s["score"], s["accounts"])
        ev = {"ring_id": f"R{root}", "n_accounts": s["accounts"], "n_transactions": s["txns"], "total_usd": s["usd"],
              "total_ngn": s["usd"] * USD_NGN, "ring_score": s["mean_score"], "first_seen": timeline[0]["time"],
              "last_seen": timeline[-1]["time"],
              "accounts": [{k: a[k] for k in ("account", "bank", "entity_type", "role")} for a in accounts],
              "timeline": [{k: t[k] for k in ("txn", "time", "from", "to", "usd", "format")} for t in timeline],
              "reason_codes": reasons, "action": action}
        return {**ev, "tier": tnum, "accounts_full": accounts, "timeline_full": timeline[:200], "str": draft(ev)}


def build_app(live: Live) -> FastAPI:
    app = FastAPI(title="MuleRadar live")
    dash = ROOT / "dashboard"

    @app.get("/")
    def index():
        return FileResponse(dash / "live.html")

    @app.get("/results")
    def results():
        return FileResponse(dash / "index.html")

    @app.post("/api/control")
    def control(body: dict):
        live.control(body)
        return live.snapshot()

    @app.get("/api/ring/{root}")
    def ring(root: int):
        r = live.ring(root)
        return JSONResponse(r) if r else JSONResponse({"error": "unknown ring"}, status_code=404)

    @app.get("/api/stream")
    async def stream():
        async def gen():
            while True:
                yield f"data: {json.dumps(live.snapshot(), default=float)}\n\n"
                await asyncio.sleep(0.25)
        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="HI-Small")
    ap.add_argument("--data", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8050)
    args = ap.parse_args()
    print("loading data + compiling kernels ...", flush=True)
    live = Live(args.name, Path(args.data))
    print(f"ready: {live.n - live.test_start:,} test transfers to replay; threshold {live.th:.3f}", flush=True)
    uvicorn.run(build_app(live), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

"""
Flask API server.

Endpoints
---------
GET  /api/workloads          – list all workload types + their JSON schema
GET  /api/engines            – list available engines
POST /api/runs               – submit a new benchmark run
GET  /api/runs               – list runs (optional ?workload=, ?engine=, ?status=)
GET  /api/runs/<id>          – get a single run (poll for status)
GET  /api/summary            – aggregate statistics
GET  /api/compare_matrix     – benchmark matrix aggregated by (engine, ad_mode, config)

Background execution
--------------------
A single daemon thread pulls pending runs from the DB and executes them
sequentially.  For the expected load (<100 runs/day) this is sufficient.
Scale to a thread pool later by replacing _worker_loop with a thread pool.
"""

from __future__ import annotations

import json
import math
import sys
import threading
import time
import logging
from collections import defaultdict
from pathlib import Path

# ---- make the project importable when running `python -m benchmarking.api.server`
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, request, abort
from flask_cors import CORS

from benchmarking.core.config import WORKLOAD_REGISTRY, config_from_dict
from benchmarking.runner.runner import BenchmarkRunner
from benchmarking.storage.database import BenchmarkDB
from benchmarking.workloads.mc_cpu import CPUMonteCarloEngine
from benchmarking.workloads.mc_jax import JAXMonteCarloEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

try:
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "benchmarking" / "cpp"))
    from benchmarking.workloads.mc_cpp import CPPMonteCarloEngine
    _CPP_AVAILABLE = True
except Exception as _cpp_err:
    log.warning("C++ engine unavailable (build with: cd benchmarking/cpp && pip install -e .): %s", _cpp_err)
    CPPMonteCarloEngine = None  # type: ignore[assignment,misc]
    _CPP_AVAILABLE = False

try:
    from benchmarking.workloads.mc_cuda import CUDAMonteCarloEngine as _CUDAClass
    # Probe: compile kernel + run 128 paths to confirm the GPU is usable at startup.
    from benchmarking.core.config import EuropeanOptionConfig as _EurCfg
    _CUDAClass().run(_EurCfg(M=128, seed=0))
    _CUDA_AVAILABLE = True
    log.info("CUDA engine available.")
except Exception as _cuda_err:
    log.warning("CUDA engine unavailable: %s", _cuda_err)
    _CUDAClass = None  # type: ignore[assignment,misc]
    _CUDA_AVAILABLE = False


def _cuda_available() -> bool:
    """Return True if PyCUDA initialised successfully at server startup."""
    return _CUDA_AVAILABLE

app = Flask(__name__)
CORS(app)

db = BenchmarkDB()

# ------------------------------------------------------------------
# Engine registry  (add new engines here as you build them)
# ------------------------------------------------------------------

ENGINES = {
    "cpu": CPUMonteCarloEngine,
    "jax": JAXMonteCarloEngine,
    **({"cpp": CPPMonteCarloEngine} if _CPP_AVAILABLE else {}),
    **({"cuda": _CUDAClass} if _CUDA_AVAILABLE else {}),
}


def _get_engine(name: str):
    cls = ENGINES.get(name)
    if cls is None:
        raise ValueError(f"Unknown engine {name!r}. Available: {list(ENGINES)}")
    return cls()


# ------------------------------------------------------------------
# Background worker
# ------------------------------------------------------------------

def _execute_run(run_id: str, config_dict: dict, engine_name: str, ad_mode: str) -> None:
    """Execute one benchmark run, updating the DB throughout."""
    try:
        db.mark_running(run_id)
        config = config_from_dict(config_dict)
        config.validate()
        engine = _get_engine(engine_name)
        if not engine.supports(config.workload_type):
            raise NotImplementedError(
                f"Engine '{engine_name}' does not support workload '{config.workload_type}'"
            )
        runner = BenchmarkRunner(engine, name=f"{engine_name}/{config.workload_type}")
        result = runner.run(config, num_warmup=1, num_runs=5, ad_mode=ad_mode)
        db.mark_completed(
            run_id=run_id,
            result_value=result.result,
            mean_runtime_ms=result.mean_runtime * 1000,
            std_runtime_ms=result.std_runtime * 1000,
            ad_overhead_ratio=result.ad_overhead_ratio,
            greeks=result.greeks,
        )
        log.info("Run %s completed: price=%.4f  mean=%.2f ms",
                 run_id[:8], result.result, result.mean_runtime * 1000)
    except Exception as exc:
        log.error("Run %s failed: %s", run_id[:8], exc)
        db.mark_failed(run_id, str(exc))


def _worker_loop() -> None:
    """Poll for pending runs and execute them one at a time."""
    while True:
        try:
            pending = db.get_pending_runs()
            for row in pending:
                _execute_run(
                    run_id=row["id"],
                    config_dict=row["config"],
                    engine_name=row["engine"],
                    ad_mode=row["ad_mode"],
                )
        except Exception as exc:
            log.error("Worker loop error: %s", exc)
        time.sleep(1)


_worker = threading.Thread(target=_worker_loop, daemon=True, name="benchmark-worker")
_worker.start()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _schema_for(workload_type: str) -> list:
    cls = WORKLOAD_REGISTRY[workload_type]
    # Instantiate with defaults to read the SCHEMA field
    instance = cls()
    return instance.SCHEMA


def _throughput(row: dict) -> float | None:
    """Paths per second = M / mean_runtime_s.  Returns None when unavailable."""
    ms = row.get("mean_runtime_ms")
    if not ms:
        return None
    M = (row.get("config") or {}).get("M")
    if not M:
        return None
    return round(M / (ms / 1000.0), 1)


def _jsonify_run(row: dict) -> dict:
    """Strip internal fields and add friendly display values."""
    return {
        "id":                row["id"],
        "workload_type":     row["workload_type"],
        "engine":            row["engine"],
        "ad_mode":           row["ad_mode"],
        "status":            row["status"],
        "config":            row.get("config", {}),
        "result_value":      row["result_value"],
        "mean_runtime_ms":   row["mean_runtime_ms"],
        "std_runtime_ms":    row["std_runtime_ms"],
        "throughput_paths_per_sec": _throughput(row),
        "ad_overhead_ratio": row["ad_overhead_ratio"],
        "greeks":            row.get("greeks"),
        "error_message":     row["error_message"],
        "created_at":        row["created_at"],
        "started_at":        row["started_at"],
        "completed_at":      row["completed_at"],
    }


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.get("/api/workloads")
def list_workloads():
    """Return all registered workload types with their parameter schema."""
    workloads = {}
    for wtype, cls in WORKLOAD_REGISTRY.items():
        instance = cls()
        workloads[wtype] = {
            "label":       wtype.capitalize() + " Option",
            "workload_type": wtype,
            "schema":      instance.SCHEMA,
        }
    return jsonify(workloads)


@app.get("/api/engines")
def list_engines():
    """Return all registered engines."""
    result = {}
    for name, cls in ENGINES.items():
        engine = cls()
        result[name] = {
            "name": name,
            "supported_workloads": [
                wt for wt in WORKLOAD_REGISTRY if engine.supports(wt)
            ],
            "supported_ad_modes": list(engine.supported_ad_modes()),
        }
    return jsonify(result)


@app.post("/api/runs")
def submit_run():
    """
    Submit a new benchmark run.

    Body (JSON):
      {
        "workload_type": "european",
        "engine": "jax",
        "ad_mode": "none",
        "config": {
          "S0": 100, "K": 100, "r": 0.05, "sigma": 0.2,
          "T": 1.0, "option_type": "call", "M": 10000, "seed": 42
        }
      }
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        abort(400, "Request body must be JSON")

    engine_name   = data.get("engine", "cpu")
    ad_mode       = data.get("ad_mode", "none")
    config_fields = data.get("config", {})
    workload_type = data.get("workload_type", config_fields.get("workload_type", "european"))

    # Merge workload_type into config so config_from_dict knows which class to use
    config_fields["workload_type"] = workload_type

    # Validate before inserting
    try:
        config = config_from_dict(config_fields)
        config.validate()
    except (ValueError, TypeError) as exc:
        abort(400, str(exc))

    if engine_name not in ENGINES:
        abort(400, f"Unknown engine '{engine_name}'. Available: {list(ENGINES)}")
    if ad_mode not in ("none", "forward", "reverse"):
        abort(400, f"ad_mode must be none/forward/reverse, got '{ad_mode}'")

    # Verify the engine supports the requested AD mode
    engine_instance = _get_engine(engine_name)
    if ad_mode not in engine_instance.supported_ad_modes():
        abort(400, f"Engine '{engine_name}' does not support ad_mode='{ad_mode}'. "
                    f"Supported: {list(engine_instance.supported_ad_modes())}")

    run_id = db.create_run(config.to_dict(), engine_name, ad_mode)
    return jsonify({"id": run_id, "status": "pending"}), 201


@app.get("/api/runs")
def list_runs():
    workload = request.args.get("workload")
    engine   = request.args.get("engine")
    status   = request.args.get("status")
    limit    = min(int(request.args.get("limit", 200)), 1000)
    rows = db.get_all_runs(limit=limit, workload_type=workload,
                           engine=engine, status=status)
    return jsonify([_jsonify_run(r) for r in rows])


@app.get("/api/runs/<run_id>")
def get_run(run_id: str):
    row = db.get_run(run_id)
    if not row:
        abort(404, f"Run {run_id!r} not found")
    return jsonify(_jsonify_run(row))


@app.get("/api/capabilities")
def capabilities():
    """
    Report which compute backends are available on this machine.

    Response example::

        {"cpu": true, "cpp": true, "jax": true, "cuda": false}
    """
    return jsonify({
        "cpu":  True,
        "jax":  True,
        "cpp":  _CPP_AVAILABLE,
        "cuda": _CUDA_AVAILABLE,
    })


@app.get("/api/summary")
def summary():
    return jsonify(db.summary())


# ------------------------------------------------------------------
# Benchmark matrix
# ------------------------------------------------------------------

_EXCLUDE_FROM_COL_KEY = {"seed", "workload_type"}


def _col_key(run: dict) -> str:
    """
    Stable string key that uniquely identifies a (config, ad_mode) column.
    We exclude seed and workload_type because they don't affect the physics,
    and we include ad_mode so forward/reverse are separate columns.
    """
    cfg = {
        k: v
        for k, v in (run.get("config") or {}).items()
        if k not in _EXCLUDE_FROM_COL_KEY
    }
    cfg["ad_mode"] = run["ad_mode"]
    return json.dumps(cfg, sort_keys=True)


def _mean(vals: list) -> float | None:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _variance(vals: list) -> float | None:
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    return sum((v - m) ** 2 for v in vals) / (len(vals) - 1)


def _aggregate_cell(runs: list) -> dict:
    rts = [r["mean_runtime_ms"] for r in runs if r.get("mean_runtime_ms") is not None]
    tps = [_throughput(r) for r in runs]
    prices = [r["result_value"] for r in runs if r.get("result_value") is not None]
    overheads = [r["ad_overhead_ratio"] for r in runs if r.get("ad_overhead_ratio") is not None]
    var = _variance(rts)
    return {
        "mean_runtime_ms": _mean(rts),
        "std_runtime_ms": math.sqrt(var) if var is not None else None,
        "variance_runtime_ms": var,
        "throughput_paths_per_sec": _mean(tps),
        "result_value": _mean(prices),
        "ad_overhead_ratio": _mean(overheads),
        "memory_mb": None,  # reserved for future instrumentation
        "run_count": len(runs),
    }


@app.get("/api/compare_matrix")
def compare_matrix():
    """
    Build a benchmark matrix: rows = (engine, ad_mode), columns = unique configs.

    Query params
    ------------
    workload  : str   required  – workload_type to filter by
    engines   : str   optional  – comma-separated engine names (default: all available)
    baseline  : str   optional  – "engine/ad_mode" string for speedup reference
                                  (default: "cpu/none")

    Response
    --------
    {
      "workload": "european",
      "baseline": "cpu/none",
      "columns": [
        { "col_key": "...", "ad_mode": "none", "config": { "M": 1000, ... } },
        ...
      ],
      "rows": [
        {
          "engine": "cpu",
          "ad_mode": "none",
          "cells": [ { cell } | null, ... ]   // one entry per column, null if no data
        },
        ...
      ]
    }
    """
    workload = request.args.get("workload")
    if not workload:
        abort(400, "workload query parameter is required")

    engine_filter_raw = request.args.get("engines", "")
    engine_filter = set(engine_filter_raw.split(",")) if engine_filter_raw else None

    baseline = request.args.get("baseline", "cpu/none")

    # Fetch all completed runs for this workload
    all_runs = db.get_all_runs(limit=5000, workload_type=workload, status="completed")

    # Optionally filter by engine
    if engine_filter:
        all_runs = [r for r in all_runs if r["engine"] in engine_filter]

    # Group: (engine, ad_mode) → col_key → [runs]
    # Using defaultdict of defaultdict of list
    groups: dict = defaultdict(lambda: defaultdict(list))
    col_meta: dict = {}  # col_key → {config, ad_mode}

    for r in all_runs:
        ck = _col_key(r)
        row_key = (r["engine"], r["ad_mode"])
        groups[row_key][ck].append(r)

        if ck not in col_meta:
            cfg = {
                k: v
                for k, v in (r.get("config") or {}).items()
                if k not in _EXCLUDE_FROM_COL_KEY
            }
            col_meta[ck] = {"ad_mode": r["ad_mode"], "config": cfg}

    # Sort columns: primarily by M (if present), then by ad_mode, then by key
    def col_sort_key(ck: str) -> tuple:
        meta = col_meta[ck]
        M = meta["config"].get("M", 0)
        ad_order = {"none": 0, "forward": 1, "reverse": 2}
        return (int(M), ad_order.get(meta["ad_mode"], 99), ck)

    ordered_cols = sorted(col_meta.keys(), key=col_sort_key)

    columns = [
        {"col_key": ck, **col_meta[ck]}
        for ck in ordered_cols
    ]

    # Sort rows: engine canonical order, then ad_mode
    ENGINE_ORDER = ["cpu", "cpp", "jax", "cuda"]
    AD_ORDER = {"none": 0, "forward": 1, "reverse": 2}

    def row_sort_key(rk: tuple) -> tuple:
        eng, ad = rk
        return (ENGINE_ORDER.index(eng) if eng in ENGINE_ORDER else 99, AD_ORDER.get(ad, 99))

    ordered_row_keys = sorted(groups.keys(), key=row_sort_key)

    rows = []
    for (engine, ad_mode) in ordered_row_keys:
        col_map = groups[(engine, ad_mode)]
        cells = [
            _aggregate_cell(col_map[ck]) if ck in col_map else None
            for ck in ordered_cols
        ]
        rows.append({"engine": engine, "ad_mode": ad_mode, "cells": cells})

    return jsonify({
        "workload": workload,
        "baseline": baseline,
        "columns": columns,
        "rows": rows,
    })


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    log.info("Starting API server on %s:%d", args.host, args.port)
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)

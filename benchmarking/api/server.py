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

Background execution
--------------------
A single daemon thread pulls pending runs from the DB and executes them
sequentially.  For the expected load (<100 runs/day) this is sufficient.
Scale to a thread pool later by replacing _worker_loop with a thread pool.
"""

from __future__ import annotations

import sys
import threading
import time
import logging
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

app = Flask(__name__)
CORS(app)

db = BenchmarkDB()

# ------------------------------------------------------------------
# Engine registry  (add new engines here as you build them)
# ------------------------------------------------------------------

ENGINES = {
    "cpu": CPUMonteCarloEngine,
    "jax": JAXMonteCarloEngine,
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
        "ad_overhead_ratio": row["ad_overhead_ratio"],
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


@app.get("/api/summary")
def summary():
    return jsonify(db.summary())


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

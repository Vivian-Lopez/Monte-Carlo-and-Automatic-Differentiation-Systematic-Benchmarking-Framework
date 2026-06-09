"""
SQLite persistence layer for benchmark runs.

Schema (runs table)
-------------------
See _FULL_SCHEMA_COLUMNS below for the authoritative column list.

All timestamps are UTC ISO-8601 strings.
New columns are added via ALTER TABLE migrations so existing databases
continue to work without manual intervention.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_DB_PATH = Path(__file__).parent.parent.parent / "results" / "benchmarks.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Authoritative column list for the runs table.
# Format: (name, sql_type_and_constraints)
# ---------------------------------------------------------------------------
_FULL_SCHEMA_COLUMNS: list[tuple[str, str]] = [
    # Identity / grouping
    ("id",                    "TEXT PRIMARY KEY"),
    ("experiment_id",         "TEXT"),
    ("experiment_type",       "TEXT"),
    ("config_hash",           "TEXT"),
    # Status lifecycle
    ("status",                "TEXT NOT NULL DEFAULT 'pending'"),
    ("error_message",         "TEXT"),
    ("created_at",            "TEXT NOT NULL"),
    ("started_at",            "TEXT"),
    ("completed_at",          "TEXT"),
    # Workload definition
    ("workload_type",         "TEXT NOT NULL"),
    ("M",                     "INTEGER"),
    ("N",                     "INTEGER"),
    ("seed",                  "INTEGER"),
    ("config_json",           "TEXT NOT NULL"),
    # Engine / implementation
    ("engine",                "TEXT NOT NULL"),
    ("language",              "TEXT"),
    ("backend",               "TEXT"),
    # Parallelism / scalability (nullable for now)
    ("num_threads",           "INTEGER"),
    # Thread diagnostics (recorded per cell by run_thread_cell.py)
    ("requested_threads",                   "INTEGER"),
    ("observed_threads_before_engine_load", "INTEGER"),
    ("observed_threads_after_engine_load",  "INTEGER"),
    ("observed_threads_after_run",          "INTEGER"),
    ("observed_threads_max",                "INTEGER"),
    ("env_omp_num_threads",                 "TEXT"),
    ("env_xla_flags",                       "TEXT"),
    ("vectorization_flag",    "TEXT"),
    ("batch_size",            "INTEGER"),
    # Runtime / performance
    ("mean_runtime_ms",       "REAL"),
    ("std_runtime_ms",        "REAL"),
    ("min_runtime_ms",        "REAL"),
    ("max_runtime_ms",        "REAL"),
    ("throughput_paths_per_sec", "REAL"),
    # AD metadata
    ("ad_mode",               "TEXT NOT NULL DEFAULT 'none'"),
    ("baseline_mean_ms",      "REAL"),
    ("ad_overhead_ratio",     "REAL"),
    # Numerical results
    ("result_value",          "REAL"),
    ("greek_delta",           "REAL"),
    ("greek_vega",            "REAL"),
    ("greek_rho",             "REAL"),
    # Analytical reference (European closed-form)
    ("analytical_price",      "REAL"),
    ("analytical_delta",      "REAL"),
    ("analytical_vega",       "REAL"),
    ("analytical_rho",        "REAL"),
    # Error metrics
    ("abs_price_error",       "REAL"),
    ("rel_price_error",       "REAL"),
    ("abs_delta_error",       "REAL"),
    ("rel_delta_error",       "REAL"),
    ("abs_vega_error",        "REAL"),
    ("rel_vega_error",        "REAL"),
    ("abs_rho_error",         "REAL"),
    ("rel_rho_error",         "REAL"),
    # Resource utilisation
    ("memory_peak_mb",        "REAL"),
    # Environment / hardware
    ("cpu_model",             "TEXT"),
    ("cpu_architecture",      "TEXT"),
    ("cpu_count",             "INTEGER"),
    ("memory_gb",             "REAL"),
    ("platform",              "TEXT"),
    ("python_version",        "TEXT"),
    ("numpy_version",         "TEXT"),
    ("jax_version",           "TEXT"),
    ("blas_backend",          "TEXT"),
    # Cloud fields (nullable)
    ("cloud_provider",        "TEXT"),
    ("region",                "TEXT"),
    ("zone",                  "TEXT"),
    ("instance_type",         "TEXT"),
    ("machine_family",        "TEXT"),
    ("vcpu_count",            "INTEGER"),
    ("cost_per_run",          "REAL"),
    ("paths_per_dollar",      "REAL"),
    # Profiler fields (nullable)
    ("profiler_phase",        "TEXT"),   # probe | full | grid | sha_round_N
    ("profiler_decision",     "TEXT"),   # selected | pruned | full_grid_only
    ("profiler_reason",       "TEXT"),
    ("dominated",             "INTEGER"),  # 0/1 boolean
    ("git_commit_hash",       "TEXT"),
    # Successive Halving (SHA) fields (nullable)
    ("sha_round",             "INTEGER"),  # 0,1,2,… or NULL for full-grid
    ("sha_eliminated",        "INTEGER"),  # 0/1: was this config eliminated at sha_round?
    # Scaling-law fit: t(M) = alpha*M + beta (2-point linear fit)
    ("scaling_law_alpha",     "REAL"),  # slope (ms per path)
    ("scaling_law_beta",      "REAL"),  # intercept (startup cost ms)
    ("extrapolated_runtime_ms", "REAL"),  # predicted runtime at max_M
    ("extrapolation_error_pct", "REAL"),  # |predicted - actual| / actual * 100
    # Legacy: greeks_json kept for backward-compat with old rows / API
    ("greeks_json",           "TEXT"),
]

# Columns that must exist for legacy callers (non-nullable in old code)
_REQUIRED_COLS = {"id", "workload_type", "engine", "ad_mode", "status",
                  "config_json", "created_at"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BenchmarkDB:
    """
    Thread-safe SQLite wrapper.

    Uses a per-thread connection (check_same_thread=False + explicit locking)
    so the Flask background thread can share the same DB file as the main
    request thread safely.
    """

    def __init__(self, db_path: Path = _DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")   # concurrent reads during writes
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        # Build CREATE TABLE statement from the authoritative column list.
        col_defs = ",\n                    ".join(
            f"{name} {defn}" for name, defn in _FULL_SCHEMA_COLUMNS
        )
        with self._lock, self._conn() as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS runs (
                    {col_defs}
                )
            """)
            # Always-safe indexes (columns guaranteed to exist from original schema)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status   ON runs(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_workload ON runs(workload_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_engine   ON runs(engine)")
            # Migrate existing DBs: add any missing columns one by one.
            existing = {r[1] for r in conn.execute("PRAGMA table_info(runs)").fetchall()}
            for col_name, col_def in _FULL_SCHEMA_COLUMNS:
                if col_name not in existing and col_name != "id":
                    # Skip NOT NULL constraints on ALTER TABLE (SQLite restriction)
                    safe_def = col_def.replace(" NOT NULL", "").replace(" PRIMARY KEY", "")
                    try:
                        conn.execute(f"ALTER TABLE runs ADD COLUMN {col_name} {safe_def}")
                        existing.add(col_name)
                    except sqlite3.OperationalError:
                        pass  # column already exists (race between threads)
            # Index on experiment_id — safe now that migration has run
            if "experiment_id" in existing:
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_experiment ON runs(experiment_id)"
                )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def create_run(
        self,
        config_dict: Dict[str, Any],
        engine: str,
        ad_mode: str = "none",
    ) -> str:
        """Insert a new pending run and return its UUID."""
        run_id = str(uuid.uuid4())
        workload_type = config_dict.get("workload_type", "european")
        with self._lock, self._conn() as conn:
            conn.execute(
                """INSERT INTO runs
                   (id, workload_type, engine, ad_mode, status,
                    config_json, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (run_id, workload_type, engine, ad_mode, "pending",
                 json.dumps(config_dict), _now()),
            )
        return run_id

    def mark_running(self, run_id: str) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "UPDATE runs SET status=?, started_at=? WHERE id=?",
                ("running", _now(), run_id),
            )

    def mark_completed(
        self,
        run_id: str,
        result_value: float,
        mean_runtime_ms: float,
        std_runtime_ms: float,
        ad_overhead_ratio: float,
        greeks: dict | None = None,
    ) -> None:
        greeks_str = json.dumps(greeks) if greeks else None
        with self._lock, self._conn() as conn:
            conn.execute(
                """UPDATE runs
                   SET status=?, result_value=?, mean_runtime_ms=?,
                       std_runtime_ms=?, ad_overhead_ratio=?,
                       greeks_json=?, completed_at=?
                   WHERE id=?""",
                ("completed", result_value, mean_runtime_ms,
                 std_runtime_ms, ad_overhead_ratio, greeks_str, _now(), run_id),
            )

    def mark_failed(self, run_id: str, error_message: str) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "UPDATE runs SET status=?, error_message=?, completed_at=? WHERE id=?",
                ("failed", error_message[:2000], _now(), run_id),
            )

    def store_run_full(
        self,
        config_dict: Dict[str, Any],
        engine: str,
        ad_mode: str = "none",
        *,
        experiment_id: Optional[str] = None,
        experiment_type: Optional[str] = None,
        # runtime
        mean_runtime_ms: Optional[float] = None,
        std_runtime_ms: Optional[float] = None,
        min_runtime_ms: Optional[float] = None,
        max_runtime_ms: Optional[float] = None,
        throughput_paths_per_sec: Optional[float] = None,
        # AD
        baseline_mean_ms: Optional[float] = None,
        ad_overhead_ratio: Optional[float] = None,
        # result
        result_value: Optional[float] = None,
        greek_delta: Optional[float] = None,
        greek_vega: Optional[float] = None,
        greek_rho: Optional[float] = None,
        # analytical reference
        analytical_price: Optional[float] = None,
        analytical_delta: Optional[float] = None,
        analytical_vega: Optional[float] = None,
        analytical_rho: Optional[float] = None,
        # error metrics
        abs_price_error: Optional[float] = None,
        rel_price_error: Optional[float] = None,
        abs_delta_error: Optional[float] = None,
        rel_delta_error: Optional[float] = None,
        abs_vega_error: Optional[float] = None,
        rel_vega_error: Optional[float] = None,
        abs_rho_error: Optional[float] = None,
        rel_rho_error: Optional[float] = None,
        # resource
        memory_peak_mb: Optional[float] = None,
        # environment
        language: Optional[str] = None,
        backend: Optional[str] = None,
        num_threads: Optional[int] = None,
        requested_threads: Optional[int] = None,
        observed_threads_before_engine_load: Optional[int] = None,
        observed_threads_after_engine_load: Optional[int] = None,
        observed_threads_after_run: Optional[int] = None,
        observed_threads_max: Optional[int] = None,
        env_omp_num_threads: Optional[str] = None,
        env_xla_flags: Optional[str] = None,
        cpu_model: Optional[str] = None,
        cpu_architecture: Optional[str] = None,
        cpu_count: Optional[int] = None,
        memory_gb: Optional[float] = None,
        platform: Optional[str] = None,
        python_version: Optional[str] = None,
        numpy_version: Optional[str] = None,
        jax_version: Optional[str] = None,
        blas_backend: Optional[str] = None,
        vectorization_flag: Optional[str] = None,
        # cloud
        cloud_provider: Optional[str] = None,
        region: Optional[str] = None,
        zone: Optional[str] = None,
        instance_type: Optional[str] = None,
        machine_family: Optional[str] = None,
        vcpu_count: Optional[int] = None,
        cost_per_run: Optional[float] = None,
        paths_per_dollar: Optional[float] = None,
        # profiler
        profiler_phase: Optional[str] = None,
        profiler_decision: Optional[str] = None,
        profiler_reason: Optional[str] = None,
        dominated: Optional[int] = None,
        git_commit_hash: Optional[str] = None,
        # SHA / scaling law
        sha_round: Optional[int] = None,
        sha_eliminated: Optional[int] = None,
        scaling_law_alpha: Optional[float] = None,
        scaling_law_beta: Optional[float] = None,
        extrapolated_runtime_ms: Optional[float] = None,
        extrapolation_error_pct: Optional[float] = None,
    ) -> str:
        """
        Insert a fully-populated completed run in one shot.

        This is the preferred method for CLI scripts.  It creates the row,
        marks it running, and marks it completed atomically.  Returns the run UUID.
        """
        run_id = str(uuid.uuid4())
        workload_type = config_dict.get("workload_type", "european")
        config_hash = config_dict.get("config_hash", "")
        M = config_dict.get("M")
        N = config_dict.get("N")
        seed = config_dict.get("seed")
        greeks_dict = {}
        if greek_delta is not None:
            greeks_dict["delta"] = greek_delta
        if greek_vega is not None:
            greeks_dict["vega"] = greek_vega
        if greek_rho is not None:
            greeks_dict["rho"] = greek_rho
        greeks_json = json.dumps(greeks_dict) if greeks_dict else None
        ts = _now()
        with self._lock, self._conn() as conn:
            conn.execute(
                """INSERT INTO runs (
                    id, experiment_id, experiment_type, config_hash,
                    status, created_at, started_at, completed_at,
                    workload_type, M, N, seed, config_json,
                    engine, language, backend, num_threads,
                    mean_runtime_ms, std_runtime_ms, min_runtime_ms, max_runtime_ms,
                    throughput_paths_per_sec,
                    ad_mode, baseline_mean_ms, ad_overhead_ratio,
                    result_value, greek_delta, greek_vega, greek_rho,
                    analytical_price, analytical_delta, analytical_vega, analytical_rho,
                    abs_price_error, rel_price_error,
                    abs_delta_error, rel_delta_error,
                    abs_vega_error, rel_vega_error,
                    abs_rho_error, rel_rho_error,
                    memory_peak_mb,
                    cpu_model, cpu_architecture, cpu_count, memory_gb,
                    platform, python_version, numpy_version, jax_version, blas_backend,
                    vectorization_flag,
                    cloud_provider, region, zone, instance_type, machine_family, vcpu_count,
                    cost_per_run, paths_per_dollar,
                    profiler_phase, profiler_decision, profiler_reason, dominated, git_commit_hash,
                    sha_round, sha_eliminated,
                    scaling_law_alpha, scaling_law_beta,
                    extrapolated_runtime_ms, extrapolation_error_pct,
                    greeks_json,
                    requested_threads,
                    observed_threads_before_engine_load,
                    observed_threads_after_engine_load,
                    observed_threads_after_run,
                    observed_threads_max,
                    env_omp_num_threads,
                    env_xla_flags
                ) VALUES (
                    ?,?,?,?,
                    ?,?,?,?,
                    ?,?,?,?,?,
                    ?,?,?,?,
                    ?,?,?,?,
                    ?,
                    ?,?,?,
                    ?,?,?,?,
                    ?,?,?,?,
                    ?,?,
                    ?,?,
                    ?,?,
                    ?,?,
                    ?,
                    ?,?,?,?,
                    ?,?,?,?,?,
                    ?,
                    ?,?,?,?,?,?,
                    ?,?,
                    ?,?,?,?,?,
                    ?,?,
                    ?,?,?,?,
                    ?,
                    ?,?,?,?,?,?,?
                )""",
                (
                    run_id, experiment_id, experiment_type, config_hash,
                    "completed", ts, ts, ts,
                    workload_type, M, N, seed, json.dumps(config_dict),
                    engine, language, backend, num_threads,
                    mean_runtime_ms, std_runtime_ms, min_runtime_ms, max_runtime_ms,
                    throughput_paths_per_sec,
                    ad_mode, baseline_mean_ms, ad_overhead_ratio,
                    result_value, greek_delta, greek_vega, greek_rho,
                    analytical_price, analytical_delta, analytical_vega, analytical_rho,
                    abs_price_error, rel_price_error,
                    abs_delta_error, rel_delta_error,
                    abs_vega_error, rel_vega_error,
                    abs_rho_error, rel_rho_error,
                    memory_peak_mb,
                    cpu_model, cpu_architecture, cpu_count, memory_gb,
                    platform, python_version, numpy_version, jax_version, blas_backend,
                    vectorization_flag,
                    cloud_provider, region, zone, instance_type, machine_family, vcpu_count,
                    cost_per_run, paths_per_dollar,
                    profiler_phase, profiler_decision, profiler_reason, dominated, git_commit_hash,
                    sha_round, sha_eliminated,
                    scaling_law_alpha, scaling_law_beta,
                    extrapolated_runtime_ms, extrapolation_error_pct,
                    greeks_json,
                    requested_threads,
                    observed_threads_before_engine_load,
                    observed_threads_after_engine_load,
                    observed_threads_after_run,
                    observed_threads_max,
                    env_omp_num_threads,
                    env_xla_flags,
                ),
            )
        return run_id

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE id=?", (run_id,)
            ).fetchone()
        return _row_to_dict(row) if row else None

    def get_all_runs(
        self,
        limit: int = 200,
        workload_type: Optional[str] = None,
        engine: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        filters, params = [], []
        if workload_type:
            filters.append("workload_type=?"); params.append(workload_type)
        if engine:
            filters.append("engine=?"); params.append(engine)
        if status:
            filters.append("status=?"); params.append(status)
        where = ("WHERE " + " AND ".join(filters)) if filters else ""
        params.append(limit)
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM runs {where} ORDER BY created_at DESC LIMIT ?",
                params,
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def get_pending_runs(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM runs WHERE status='pending' ORDER BY created_at ASC"
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def summary(self) -> Dict[str, Any]:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            completed = conn.execute(
                "SELECT COUNT(*) FROM runs WHERE status='completed'"
            ).fetchone()[0]
            pending = conn.execute(
                "SELECT COUNT(*) FROM runs WHERE status IN ('pending','running')"
            ).fetchone()[0]
            failed = conn.execute(
                "SELECT COUNT(*) FROM runs WHERE status='failed'"
            ).fetchone()[0]
            fastest = conn.execute(
                "SELECT MIN(mean_runtime_ms) FROM runs WHERE status='completed'"
            ).fetchone()[0]
            by_workload = conn.execute(
                "SELECT workload_type, COUNT(*) as cnt FROM runs GROUP BY workload_type"
            ).fetchall()
            by_engine = conn.execute(
                "SELECT engine, COUNT(*) as cnt FROM runs GROUP BY engine"
            ).fetchall()
        return {
            "total": total,
            "completed": completed,
            "pending": pending,
            "failed": failed,
            "fastest_ms": fastest,
            "by_workload": {r["workload_type"]: r["cnt"] for r in by_workload},
            "by_engine":   {r["engine"]: r["cnt"] for r in by_engine},
        }


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    # Parse config_json back to a dict for convenience
    if isinstance(d.get("config_json"), str):
        d["config"] = json.loads(d["config_json"])
    # Parse greeks_json back to a dict
    gj = d.pop("greeks_json", None)
    d["greeks"] = json.loads(gj) if isinstance(gj, str) else None
    return d

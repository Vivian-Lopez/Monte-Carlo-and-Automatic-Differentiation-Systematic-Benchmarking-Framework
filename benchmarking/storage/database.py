"""
SQLite persistence layer for benchmark runs.

Schema
------
runs
  id            TEXT PRIMARY KEY   (UUID)
  workload_type TEXT               (european | asian | barrier | basket)
  engine        TEXT               (cpu | jax | cpp | gpu)
  ad_mode       TEXT               (none | forward | reverse)
  status        TEXT               (pending | running | completed | failed)
  config_json   TEXT               (full WorkloadConfig serialised as JSON)
  result_value  REAL               (option price, NULL until completed)
  mean_runtime_ms REAL
  std_runtime_ms  REAL
  ad_overhead_ratio REAL
  error_message TEXT               (NULL on success)
  created_at    TEXT               (ISO-8601)
  started_at    TEXT
  completed_at  TEXT

All timestamps are UTC ISO-8601 strings.
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
        with self._lock, self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id                TEXT PRIMARY KEY,
                    workload_type     TEXT NOT NULL,
                    engine            TEXT NOT NULL,
                    ad_mode           TEXT NOT NULL DEFAULT 'none',
                    status            TEXT NOT NULL DEFAULT 'pending',
                    config_json       TEXT NOT NULL,
                    result_value      REAL,
                    mean_runtime_ms   REAL,
                    std_runtime_ms    REAL,
                    ad_overhead_ratio REAL,
                    error_message     TEXT,
                    created_at        TEXT NOT NULL,
                    started_at        TEXT,
                    completed_at      TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status   ON runs(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_workload ON runs(workload_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_engine   ON runs(engine)")

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
    ) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                """UPDATE runs
                   SET status=?, result_value=?, mean_runtime_ms=?,
                       std_runtime_ms=?, ad_overhead_ratio=?, completed_at=?
                   WHERE id=?""",
                ("completed", result_value, mean_runtime_ms,
                 std_runtime_ms, ad_overhead_ratio, _now(), run_id),
            )

    def mark_failed(self, run_id: str, error_message: str) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "UPDATE runs SET status=?, error_message=?, completed_at=? WHERE id=?",
                ("failed", error_message[:2000], _now(), run_id),
            )

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
    return d

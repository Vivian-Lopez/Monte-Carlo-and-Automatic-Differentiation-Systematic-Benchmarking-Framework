#!/usr/bin/env python3
"""
Merge one or more source SQLite benchmark databases into the main database.

Usage
-----
  # Merge a single VM result into the main DB
  python scripts/merge_sqlite_results.py \\
      --source results/benchmarks_n2-standard-8_20260605.db \\
      --target results/benchmarks.db

  # Merge multiple VM results at once
  python scripts/merge_sqlite_results.py \\
      --source results/benchmarks_n2*.db results/benchmarks_t2d*.db \\
      --target results/benchmarks.db

  # Dry-run: show what would be inserted without writing anything
  python scripts/merge_sqlite_results.py \\
      --source results/benchmarks_n2-standard-8_*.db \\
      --target results/benchmarks.db \\
      --dry-run

Deduplication
-------------
A row is considered a duplicate if (experiment_id, engine, workload_type,
ad_mode, M, profiler_phase, instance_type, created_at) all match an existing
row in the target DB.  Duplicate rows are skipped; all other rows are inserted.

The target DB schema is migrated to include any missing columns before the
merge, using the same safe ALTER TABLE pattern as BenchmarkDB.

Safety
------
- The target DB is never truncated or overwritten.
- Source DBs are opened read-only.
- A summary of inserted / skipped / failed rows is printed.
"""

from __future__ import annotations

import argparse
import glob
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from benchmarking.storage.database import BenchmarkDB, _FULL_SCHEMA_COLUMNS


# Fields used to detect duplicates.  Chosen to be stable across re-runs
# while avoiding false positives from different VMs running the same config.
_DEDUP_FIELDS = (
    "experiment_id",
    "engine",
    "workload_type",
    "ad_mode",
    "M",
    "profiler_phase",
    "sha_round",
    "instance_type",
    "created_at",
)


def _migrate_target(conn: sqlite3.Connection) -> None:
    """Ensure target DB has all columns defined in _FULL_SCHEMA_COLUMNS."""
    existing = {r[1] for r in conn.execute("PRAGMA table_info(runs)").fetchall()}
    for col_name, col_def in _FULL_SCHEMA_COLUMNS:
        if col_name not in existing and col_name != "id":
            safe_def = col_def.replace(" NOT NULL", "").replace(" PRIMARY KEY", "")
            try:
                conn.execute(f"ALTER TABLE runs ADD COLUMN {col_name} {safe_def}")
                existing.add(col_name)
            except sqlite3.OperationalError:
                pass


def _get_column_names(conn: sqlite3.Connection) -> list[str]:
    return [r[1] for r in conn.execute("PRAGMA table_info(runs)").fetchall()]


def _build_dedup_set(conn: sqlite3.Connection) -> set[tuple]:
    """Return the set of dedup tuples already in the target DB."""
    cols = _get_column_names(conn)
    available = [f for f in _DEDUP_FIELDS if f in cols]
    if not available:
        return set()
    rows = conn.execute(
        f"SELECT {', '.join(available)} FROM runs"
    ).fetchall()
    return {tuple(r) for r in rows}


def merge_source(
    src_path: Path,
    tgt_conn: sqlite3.Connection,
    tgt_dedup: set[tuple],
    dry_run: bool,
) -> tuple[int, int, int]:
    """
    Merge runs from src_path into tgt_conn.
    Returns (inserted, skipped, failed).
    """
    try:
        src = sqlite3.connect(f"file:{src_path}?mode=ro", uri=True)
        src.row_factory = sqlite3.Row
    except sqlite3.OperationalError as e:
        print(f"  [ERROR] Cannot open {src_path}: {e}")
        return 0, 0, 1

    src_cols = _get_column_names(src)
    tgt_cols = _get_column_names(tgt_conn)

    # Only insert columns that exist in both source and target
    common_cols = [c for c in src_cols if c in tgt_cols]

    src_rows = src.execute(
        f"SELECT {', '.join(common_cols)} FROM runs WHERE status = 'completed'"
    ).fetchall()
    src.close()

    dedup_col_indices = [
        common_cols.index(f) for f in _DEDUP_FIELDS if f in common_cols
    ]
    dedup_fields_available = [f for f in _DEDUP_FIELDS if f in common_cols]

    inserted = skipped = failed = 0

    for row in src_rows:
        row_dict = dict(zip(common_cols, row))
        dedup_key = tuple(row_dict.get(f) for f in dedup_fields_available)

        if dedup_key in tgt_dedup:
            skipped += 1
            continue

        if dry_run:
            inserted += 1
            tgt_dedup.add(dedup_key)
            continue

        placeholders = ", ".join("?" * len(common_cols))
        col_list     = ", ".join(common_cols)
        try:
            tgt_conn.execute(
                f"INSERT INTO runs ({col_list}) VALUES ({placeholders})",
                tuple(row_dict[c] for c in common_cols),
            )
            tgt_dedup.add(dedup_key)
            inserted += 1
        except sqlite3.IntegrityError:
            # Primary key collision (id already exists) — treat as duplicate
            skipped += 1
        except Exception as e:
            print(f"  [WARN] Row insert failed: {e}")
            failed += 1

    return inserted, skipped, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge SQLite benchmark databases")
    parser.add_argument("--source", nargs="+", required=True,
                        help="Source DB path(s) or glob patterns")
    parser.add_argument("--target", default="results/benchmarks.db",
                        help="Target (main) DB path (default: results/benchmarks.db)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be inserted without writing anything")
    args = parser.parse_args()

    # Expand globs
    source_paths: list[Path] = []
    for pattern in args.source:
        expanded = glob.glob(pattern)
        if expanded:
            source_paths.extend(Path(p) for p in expanded)
        else:
            p = Path(pattern)
            if p.exists():
                source_paths.append(p)
            else:
                print(f"  [WARN] No files matched: {pattern}")

    if not source_paths:
        print("No source databases found. Exiting.")
        sys.exit(1)

    tgt_path = Path(args.target)
    tgt_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure target schema is up-to-date
    _ = BenchmarkDB(tgt_path)  # triggers _init_schema + migrations

    tgt_conn = sqlite3.connect(str(tgt_path))
    tgt_conn.row_factory = sqlite3.Row
    tgt_conn.execute("PRAGMA journal_mode=WAL")
    _migrate_target(tgt_conn)

    tgt_dedup = _build_dedup_set(tgt_conn)
    print(f"Target DB          : {tgt_path}")
    print(f"Existing rows      : {len(tgt_dedup)}")
    print(f"Sources to merge   : {len(source_paths)}")
    if args.dry_run:
        print("DRY-RUN mode       : no rows will be written")
    print()

    total_inserted = total_skipped = total_failed = 0

    for src_path in source_paths:
        print(f"  Merging: {src_path.name}")
        ins, skp, fail = merge_source(src_path, tgt_conn, tgt_dedup, args.dry_run)
        print(f"    inserted={ins}  skipped={skp}  failed={fail}")
        total_inserted += ins
        total_skipped  += skp
        total_failed   += fail

    if not args.dry_run:
        tgt_conn.commit()

    tgt_conn.close()

    print()
    print("=" * 50)
    print(f"MERGE COMPLETE")
    print(f"  Total inserted : {total_inserted}")
    print(f"  Total skipped  : {total_skipped}  (duplicates)")
    print(f"  Total failed   : {total_failed}")
    if args.dry_run:
        print("  (DRY-RUN: nothing written to disk)")
    print("=" * 50)


if __name__ == "__main__":
    main()

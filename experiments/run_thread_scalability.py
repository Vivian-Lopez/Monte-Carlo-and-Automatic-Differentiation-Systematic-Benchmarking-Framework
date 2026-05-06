"""
Thread-Scalability Benchmark — Orchestrator
=============================================
Measures how throughput scales with available CPU parallelism for the C++
(OpenMP) and JAX (XLA) engines across a range of thread counts and path
counts.

Two scaling regimes are measured:

  Strong scaling — fixed problem size (M), thread count varies.
                   Ideal: throughput doubles when threads double.
                   Reality: Amdahl's law limits the gain.

  Weak scaling   — M = weak_base_m × num_threads, so each thread always
                   prices the same number of paths.
                   Ideal: constant throughput per thread at all thread counts.

Subprocess-per-cell design
---------------------------
XLA reads its thread count from XLA_FLAGS BEFORE any JAX computation runs.
Once XLA has spawned its internal thread pool the flag is ignored for the
rest of the process lifetime.  OpenMP behaves the same: the thread count is
fixed on the first parallel region entry.

This orchestrator therefore launches a fresh Python subprocess for every
(engine, thread_count, M, regime) cell, setting the correct environment
variables before the process starts.  This guarantees each cell starts
with a clean process and the requested environment.

For C++/OpenMP, OMP_NUM_THREADS reliably controls the thread pool.
For JAX/XLA, subprocess isolation guarantees a clean runtime and the
requested environment, but exact CPU worker thread control is
runtime-dependent.  Each cell records observed OS thread counts (via
psutil) for honest post-hoc interpretation.

Usage
-----
    # Quick smoke-test on 4-core Codespaces
    python experiments/run_thread_scalability.py --threads 1 2 4 --runs 3 --warmup 1

    # Full experiment
    python experiments/run_thread_scalability.py \\
        --threads 1 2 4 8 16 \\
        --strong-m-values 50000 250000 1000000 \\
        --weak-base-m-values 10000 50000 250000 \\
        --runs 7 --warmup 2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

CELL_SCRIPT  = Path(__file__).parent / "run_thread_cell.py"
DEFAULT_THREADS    = [1, 2, 4]
DEFAULT_STRONG_M   = [50_000]
DEFAULT_WEAK_BASE  = [10_000]
NUM_WARMUP = 2
NUM_RUNS   = 5


# ---------------------------------------------------------------------------
# Environment construction
# ---------------------------------------------------------------------------

def _build_env(n_threads: int, engine: str) -> dict[str, str]:
    """Return a copy of os.environ with thread-count vars set for this cell.

    Note on JAX/XLA thread control (JAX 0.9.x)
    -------------------------------------------
    The legacy flag ``--xla_cpu_multi_thread_eigen_intra_op_parallelism_threads``
    was removed in XLA ≥ 0.4.x (JAX 0.4+).  In JAX 0.9.x, XLA's internal
    CPU thread pool no longer has a public XLA_FLAGS knob.

    For the JAX engine we still set ``OMP_NUM_THREADS`` because it may
    influence XLA's internal thread scheduling on some platforms, and it
    keeps the environment consistent between engines.

    For JAX/XLA, subprocess isolation guarantees a clean runtime and the
    requested environment, but exact CPU worker thread control is
    runtime-dependent.  Observed OS thread counts (recorded per cell via
    psutil) are needed for honest interpretation of JAX thread-scaling
    results.
    """
    env = os.environ.copy()

    # OpenMP / BLAS — always set
    env["OMP_NUM_THREADS"]      = str(n_threads)
    env["OMP_DYNAMIC"]          = "FALSE"
    env["OMP_PROC_BIND"]        = "close"
    env["OMP_PLACES"]           = "cores"
    env["OPENBLAS_NUM_THREADS"] = "1"   # NumPy/SciPy use 1 thread; engine owns parallelism
    env["MKL_NUM_THREADS"]      = "1"
    env["NUMEXPR_NUM_THREADS"]  = "1"

    # Remove any leftover (now-invalid) XLA thread flag from parent env
    if "XLA_FLAGS" in env:
        cleaned = re.sub(
            r"--xla_cpu_multi_thread_eigen_intra_op_parallelism_threads=\d+",
            "", env["XLA_FLAGS"],
        ).strip()
        if cleaned:
            env["XLA_FLAGS"] = cleaned
        else:
            del env["XLA_FLAGS"]

    return env


# ---------------------------------------------------------------------------
# Single cell dispatch
# ---------------------------------------------------------------------------

def _run_cell(
    engine: str,
    n_threads: int,
    M: int,
    regime: str,
    experiment_id: str,
    args: argparse.Namespace,
    oversubscribed: bool,
) -> Optional[dict]:
    """Launch one subprocess cell and return its parsed JSON summary."""
    cmd = [
        sys.executable, str(CELL_SCRIPT),
        "--engine",        engine,
        "--threads",       str(n_threads),
        "--M",             str(M),
        "--regime",        regime,
        "--experiment-id", experiment_id,
        "--runs",          str(args.runs),
        "--warmup",        str(args.warmup),
    ]
    if oversubscribed:
        cmd.append("--oversubscribed")

    env = _build_env(n_threads, engine)

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=600,   # 10-minute hard ceiling per cell
        )
    except subprocess.TimeoutExpired:
        print(f"    [{engine}, T={n_threads}, M={M:,}] TIMEOUT (>600 s)")
        return None
    except Exception as exc:
        print(f"    [{engine}, T={n_threads}, M={M:,}] LAUNCH ERROR: {exc}")
        return None

    if proc.stderr:
        # Forward stderr (includes JAX/XLA compilation logs) at one indent level
        for line in proc.stderr.strip().splitlines():
            # Suppress verbose XLA/absl INFO lines to keep output readable
            if any(tag in line for tag in ("I0", "W0", "E0", "absl", "xla", "XLA",
                                            "jax", "JAX", "WARNING")):
                continue
            print(f"    [stderr] {line}")

    # Parse the single JSON summary line from stdout
    for line in proc.stdout.strip().splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                data = json.loads(line)
                if not data.get("success"):
                    print(f"    [{engine}, T={n_threads}, M={M:,}] "
                          f"FAILED: {data.get('error', 'unknown error')}")
                    return None
                return data
            except json.JSONDecodeError:
                pass

    print(f"    [{engine}, T={n_threads}, M={M:,}] "
          f"No JSON output (exit {proc.returncode})")
    if proc.stdout.strip():
        print(f"    stdout: {proc.stdout.strip()[:200]}")
    return None


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _fmt(v, fmt=".3f") -> str:
    return f"{v:{fmt}}" if v is not None else "    n/a "


def _print_table(rows: list[dict], title: str) -> None:
    if not rows:
        return
    print(f"\n  {title}")
    print("  " + "-" * 94)
    hdr = (f"  {'Engine':<8}  {'T':>3}  {'M':>8}  {'Mean ms':>9}  {'Std ms':>7}  "
           f"{'Paths/s':>12}  {'Price':>9}  {'Rel Err%':>9}  {'Mem MB':>7}")
    print(hdr)
    print("  " + "-" * 94)
    for r in rows:
        sub = " *" if r.get("oversubscribed") else "  "
        re_s = f"{r['rel_err']*100:.4f}" if r.get("rel_err") is not None else "     n/a"
        print(
            f"  {r['engine']:<8}  {r['threads']:>3}  {r['M']:>8,}  "
            f"{r['mean_ms']:>9.3f}  {r['std_ms']:>7.3f}  "
            f"{r['throughput']:>12.0f}  {r['price']:>9.5f}  "
            f"{re_s:>9}  {_fmt(r['memory_peak_mb'], '.1f'):>7}{sub}"
        )
    if any(r.get("oversubscribed") for r in rows):
        print("  * oversubscribed run (threads > physical cores) — not scaling data")


def _print_speedup(strong_rows: list[dict], engines: list[str],
                   thread_counts: list[int]) -> None:
    if 1 not in thread_counts:
        print("\n  [warn] T=1 not in thread sweep — skipping speedup summary")
        return

    # Build per-engine baselines from T=1 non-oversubscribed cells
    baselines: dict[str, dict[int, float]] = {}  # engine -> strong_M -> throughput
    for r in strong_rows:
        if r["threads"] == 1 and not r.get("oversubscribed"):
            baselines.setdefault(r["engine"], {})[r["M"]] = r["throughput"]

    if not baselines:
        print("\n  [warn] No T=1 baseline found — skipping speedup summary")
        return

    # One speedup table per distinct strong-M value
    strong_m_values = sorted({r["M"] for r in strong_rows})
    for strong_M in strong_m_values:
        print(f"\n  Strong-scaling speedup  (M={strong_M:,}, throughput relative to T=1)")
        print("  " + "-" * 55)
        hdr = f"  {'Engine':<8}" + "".join(f"  T={t:>2}" for t in thread_counts)
        print(hdr)
        print("  " + "-" * 55)
        for eng in engines:
            base = baselines.get(eng, {}).get(strong_M)
            if base is None:
                continue
            row_s = f"  {eng:<8}"
            for t in thread_counts:
                match = next(
                    (r for r in strong_rows
                     if r["engine"] == eng and r["threads"] == t
                     and r["M"] == strong_M and not r.get("oversubscribed")),
                    None,
                )
                if match:
                    row_s += f"  {match['throughput'] / base:>5.2f}x"
                else:
                    row_s += "     n/a"
            print(row_s)


def _print_diag_table(rows: list[dict]) -> None:
    """Print per-cell OS thread diagnostics for result interpretation."""
    diag_rows = [r for r in rows if "observed_threads_max" in r]
    if not diag_rows:
        return
    print("\n  Thread diagnostics")
    print("  Note: For C++/OpenMP, T_env (OMP_NUM_THREADS) is the reliable control.")
    print("        For JAX/XLA, Obs_run shows OS threads after first computation.")
    print("        A match between T_req and Obs_max does NOT guarantee XLA used")
    print("        exactly that many compute threads \u2014 it reflects the process pool.")
    print("  " + "-" * 90)
    hdr = (
        f"  {'Engine':<8}  {'Regime':<8}  {'T_req':>5}  {'T_env':>5}  "
        f"{'Obs_bef':>7}  {'Obs_ld':>6}  {'Obs_run':>7}  {'Obs_max':>7}  "
        f"XLA_FLAGS"
    )
    print(hdr)
    print("  " + "-" * 90)
    for r in diag_rows:
        xla = r.get("env_xla_flags") or "-"
        if len(xla) > 35:
            xla = xla[:32] + "..."
        t_env = r.get("env_omp_num_threads") or "-"
        print(
            f"  {r['engine']:<8}  {r.get('regime', '?'):<8}  "
            f"{r.get('requested_threads', r['threads']):>5}  "
            f"{t_env:>5}  "
            f"{r.get('observed_threads_before_engine_load', -1):>7}  "
            f"{r.get('observed_threads_after_engine_load', -1):>6}  "
            f"{r.get('observed_threads_after_run', -1):>7}  "
            f"{r.get('observed_threads_max', -1):>7}  "
            f"{xla}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    n_cpus = os.cpu_count() or 1

    parser = argparse.ArgumentParser(
        description="Thread-scalability benchmark orchestrator (subprocess-per-cell)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--runs",   type=int, default=NUM_RUNS,
                        help="Timed repetitions per cell (default: %(default)s)")
    parser.add_argument("--warmup", type=int, default=NUM_WARMUP,
                        help="Warmup runs per cell (default: %(default)s)")
    parser.add_argument("--threads", type=int, nargs="+", default=DEFAULT_THREADS,
                        metavar="N",
                        help=f"Thread counts to sweep (default: {DEFAULT_THREADS})")
    parser.add_argument("--engines", nargs="+", default=["jax", "cpp"],
                        choices=["jax", "cpp"],
                        help="Engines to include (default: jax cpp)")
    parser.add_argument("--strong-m-values", type=int, nargs="+",
                        default=DEFAULT_STRONG_M, metavar="M",
                        help=f"Fixed M values for strong scaling (default: {DEFAULT_STRONG_M})")
    parser.add_argument("--weak-base-m-values", type=int, nargs="+",
                        default=DEFAULT_WEAK_BASE, metavar="M",
                        help=f"Per-thread M for weak scaling (default: {DEFAULT_WEAK_BASE})")
    parser.add_argument("--no-strong-scaling", action="store_true")
    parser.add_argument("--no-weak-scaling",   action="store_true")
    args = parser.parse_args()

    thread_counts = sorted(set(args.threads))

    # Warn on oversubscription
    oversubscribed_counts = [t for t in thread_counts if t > n_cpus]
    if oversubscribed_counts:
        print(f"\n  [warn] Thread counts {oversubscribed_counts} exceed "
              f"os.cpu_count()={n_cpus}.")
        print("         These cells will run but are labelled as oversubscription "
              "tests.\n         Do not include them in scaling conclusions.")

    experiment_id = str(uuid.uuid4())

    print()
    print("=" * 80)
    print("  Thread-Scalability Benchmark")
    print("=" * 80)
    try:
        import psutil as _psutil
        n_physical = _psutil.cpu_count(logical=False) or "unknown"
    except Exception:
        n_physical = "unknown"
    print(f"  Logical CPUs   : {n_cpus}")
    print(f"  Physical cores : {n_physical}")
    print(f"  Thread sweep   : {thread_counts}")
    print(f"  Engines        : {args.engines}")
    if not args.no_strong_scaling:
        print(f"  Strong M       : {args.strong_m_values}")
    if not args.no_weak_scaling:
        print(f"  Weak base-M    : {args.weak_base_m_values}")
    print(f"  Runs / Warmup  : {args.runs} / {args.warmup}")
    print(f"  Experiment ID  : {experiment_id}")
    print()

    strong_rows: list[dict] = []
    weak_rows:   list[dict] = []

    for engine in args.engines:
        for n_threads in thread_counts:
            oversubscribed = n_threads > n_cpus

            # Strong scaling
            if not args.no_strong_scaling:
                for strong_M in args.strong_m_values:
                    tag = f"[{engine}, T={n_threads}, M={strong_M:,}, strong]"
                    if oversubscribed:
                        tag += " (oversubscribed)"
                    print(f"  Running {tag} ...")
                    row = _run_cell(engine, n_threads, strong_M, "strong",
                                    experiment_id, args, oversubscribed)
                    if row:
                        row["oversubscribed"] = oversubscribed
                        strong_rows.append(row)
                        print(f"    => {row['mean_ms']:.3f} ms  "
                              f"{row['throughput']:,.0f} paths/s")

            # Weak scaling
            if not args.no_weak_scaling:
                for base_M in args.weak_base_m_values:
                    weak_M = base_M * n_threads
                    tag = f"[{engine}, T={n_threads}, M={weak_M:,}, weak]"
                    if oversubscribed:
                        tag += " (oversubscribed)"
                    print(f"  Running {tag} ...")
                    row = _run_cell(engine, n_threads, weak_M, "weak",
                                    experiment_id, args, oversubscribed)
                    if row:
                        row["oversubscribed"] = oversubscribed
                        weak_rows.append(row)
                        print(f"    => {row['mean_ms']:.3f} ms  "
                              f"{row['throughput']:,.0f} paths/s")

        print()  # blank line between engines

    # Summary tables — one per strong-M value
    for strong_M in (args.strong_m_values if not args.no_strong_scaling else []):
        subset = [r for r in strong_rows if r["M"] == strong_M]
        _print_table(subset, f"Strong scaling  (fixed M={strong_M:,})")

    for base_M in (args.weak_base_m_values if not args.no_weak_scaling else []):
        subset = [r for r in weak_rows if r["M"] == base_M * r["threads"]]
        _print_table(subset, f"Weak scaling  (M = {base_M:,} × threads)")

    # Speedup summary
    if strong_rows and not args.no_strong_scaling:
        _print_speedup(strong_rows, args.engines, thread_counts)

    # Thread diagnostics table (all cells, both regimes)
    _print_diag_table(strong_rows + weak_rows)

    from benchmarking.storage.database import BenchmarkDB
    db = BenchmarkDB()
    print()
    print(f"  Results stored in : {db.db_path}")
    print(f"  Experiment ID     : {experiment_id}")
    print()
    print("  ── Performance query (strong scaling) ────────────────────────────────────────")
    print(f'  sqlite3 results/benchmarks.db "')
    print(f"    SELECT engine, num_threads, mean_runtime_ms,")
    print(f"           throughput_paths_per_sec, rel_price_error")
    print(f"    FROM runs")
    print(f"    WHERE experiment_id='{experiment_id}'")
    print(f"      AND experiment_type='thread_scalability_strong'")
    print(f'    ORDER BY engine, num_threads;"')
    print()
    print("  ── Thread diagnostics query ──────────────────────────────────────────────────")
    print(f'  sqlite3 results/benchmarks.db "')
    print(f"    SELECT experiment_type, engine, num_threads,")
    print(f"           requested_threads, env_omp_num_threads,")
    print(f"           observed_threads_before_engine_load,")
    print(f"           observed_threads_after_engine_load,")
    print(f"           observed_threads_after_run,")
    print(f"           observed_threads_max,")
    print(f"           env_xla_flags,")
    print(f"           mean_runtime_ms, throughput_paths_per_sec")
    print(f"    FROM runs")
    print(f"    WHERE experiment_id='{experiment_id}'")
    print(f'    ORDER BY experiment_type, engine, num_threads;"')
    print()
    print("  ── Correctness check query ───────────────────────────────────────────────────")
    print(f'  sqlite3 results/benchmarks.db "')
    print(f"    SELECT engine, num_threads,")
    print(f"           json_extract(config_json, '$.M') AS M,")
    print(f"           requested_threads,")
    print(f"           env_omp_num_threads,")
    print(f"           observed_threads_max,")
    print(f"           round(rel_price_error * 100, 4)              AS rel_err_pct,")
    print(f"           round(mean_runtime_ms, 3)                    AS mean_ms,")
    print(f"           round(json_extract(config_json, '$.M')")
    print(f"                 / (mean_runtime_ms / 1000.0))          AS implied_paths_s,")
    print(f"           round(throughput_paths_per_sec)              AS recorded_paths_s")
    print(f"    FROM runs")
    print(f"    WHERE experiment_id='{experiment_id}'")
    print(f'    ORDER BY engine, num_threads;"')
    print()


if __name__ == "__main__":
    main()

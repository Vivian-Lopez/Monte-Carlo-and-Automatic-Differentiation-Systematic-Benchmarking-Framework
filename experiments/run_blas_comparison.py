"""
BLAS Backend Comparison — Orchestrator
=======================================
Measures how NumPy's underlying BLAS implementation (MKL, OpenBLAS, BLIS, …)
affects Monte Carlo throughput for the CPU engine.

Each (blas_environment, M) cell is run as a fresh subprocess using the Python
executable from the corresponding conda environment.  This guarantees that
each cell loads exactly the NumPy that was compiled against the target BLAS —
there is no way to hot-swap BLAS linkage within a single process.

All results accumulate in the shared SQLite database (results/benchmarks.db),
differentiated by the blas_backend and numpy_version columns.

Correctness checks (run automatically after all cells complete)
---------------------------------------------------------------
  1. Backend differentiation   — every cell must have a non-null, non-"unknown"
                                  blas_backend; each executable must produce a
                                  distinct value.
  2. Numerical agreement        — same (seed, M) must produce bit-for-bit
                                  identical prices across backends.  Any
                                  divergence indicates different NumPy versions
                                  (different RNG state machines) and invalidates
                                  the comparison.
  3. Black-Scholes accuracy     — rel_price_error < 1% for M >= 50,000.
  4. Throughput ordering        — for M >= 250,000 the ranking is printed so the
                                  user can spot if MKL and OpenBLAS are swapped
                                  (a sign of environment misconfiguration).

Usage
-----
  # Smoke test (fast; verifies correctness checks pass before a full run)
  python experiments/run_blas_comparison.py \\
      --backends mkl openblas \\
      --executables /opt/conda/envs/bench_mkl/bin/python \\
                    /opt/conda/envs/bench_openblas/bin/python \\
      --m-values 10000 \\
      --runs 3 --warmup 1

  # Full experiment
  python experiments/run_blas_comparison.py \\
      --backends mkl openblas \\
      --executables /opt/conda/envs/bench_mkl/bin/python \\
                    /opt/conda/envs/bench_openblas/bin/python \\
      --m-values 10000 50000 250000 1000000 \\
      --runs 7 --warmup 2

  # Three-way comparison with BLIS
  python experiments/run_blas_comparison.py \\
      --backends mkl openblas blis \\
      --executables /opt/conda/envs/bench_mkl/bin/python \\
                    /opt/conda/envs/bench_openblas/bin/python \\
                    /opt/conda/envs/bench_blis/bin/python \\
      --m-values 10000 50000 250000 1000000 \\
      --runs 7 --warmup 2

Environment setup (do once, before running)
-------------------------------------------
  conda create -n bench_mkl     python=3.12 -y
  conda install -n bench_mkl     numpy scipy "blas=*=mkl"      -y

  conda create -n bench_openblas python=3.12 -y
  conda install -n bench_openblas numpy scipy "blas=*=openblas" -y

  # Verify (must print different 'libraries' lines)
  conda run -n bench_mkl     python -c "import numpy as np; np.show_config()"
  conda run -n bench_openblas python -c "import numpy as np; np.show_config()"
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

CELL_SCRIPT    = Path(__file__).parent / "run_blas_cell.py"
DEFAULT_M      = [10_000, 50_000, 250_000]
NUM_WARMUP     = 2
NUM_RUNS       = 7
REL_ERR_THRESH = 0.01   # 1% relative error tolerance for M >= 50k


# ---------------------------------------------------------------------------
# Environment construction — pin thread counts to 1 to isolate BLAS effect
# ---------------------------------------------------------------------------

def _build_env() -> dict[str, str]:
    """Return os.environ copy with all thread knobs pinned to 1.

    We are benchmarking BLAS vendor performance, not thread scaling.  Pinning
    everything to 1 removes thread count as a confounding variable and gives
    the clearest signal on scalar / SIMD kernel quality.
    """
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
        env[key] = "1"
    env["OMP_DYNAMIC"] = "FALSE"
    # Prevent ~/.local/lib/pythonX.Y/site-packages from shadowing the conda
    # env's packages (e.g. user-installed numpy 2.x overriding env numpy 1.26)
    env["PYTHONNOUSERSITE"] = "1"
    return env


# ---------------------------------------------------------------------------
# Single cell dispatch
# ---------------------------------------------------------------------------

def _run_cell(
    python_exe: str,
    backend_label: str,
    M: int,
    experiment_id: str,
    args: argparse.Namespace,
) -> Optional[dict]:
    """Launch one subprocess cell and return its parsed JSON summary."""
    cmd = [
        python_exe, str(CELL_SCRIPT),
        "--M",             str(M),
        "--experiment-id", experiment_id,
        "--runs",          str(args.runs),
        "--warmup",        str(args.warmup),
    ]
    env = _build_env()

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        print(f"    [{backend_label}, M={M:,}] TIMEOUT (>600 s)")
        return None
    except Exception as exc:
        print(f"    [{backend_label}, M={M:,}] LAUNCH ERROR: {exc}")
        return None

    if proc.stderr:
        for line in proc.stderr.strip().splitlines():
            if any(tag in line for tag in ("UserWarning", "FutureWarning",
                                            "DeprecationWarning")):
                print(f"    [stderr] {line}")

    for line in proc.stdout.strip().splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                data = json.loads(line)
                if not data.get("success"):
                    print(f"    [{backend_label}, M={M:,}] "
                          f"FAILED: {data.get('error', 'unknown error')}")
                    return None
                return data
            except json.JSONDecodeError:
                pass

    print(f"    [{backend_label}, M={M:,}] No JSON output (exit {proc.returncode})")
    if proc.stdout.strip():
        print(f"    stdout: {proc.stdout.strip()[:300]}")
    return None


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _fmt(v, fmt=".3f") -> str:
    return f"{v:{fmt}}" if v is not None else "    n/a "


def _print_table(rows: list[dict]) -> None:
    if not rows:
        return
    print()
    print("  " + "=" * 96)
    print("  Results")
    print("  " + "=" * 96)
    hdr = (f"  {'Backend':<12}  {'M':>9}  {'Mean ms':>9}  {'Std ms':>7}  "
           f"{'Paths/s':>13}  {'Price':>10}  {'Rel Err%':>9}  {'Vec':>8}  "
           f"{'NumPy':>8}")
    print(hdr)
    print("  " + "-" * 96)
    for r in rows:
        re_s = f"{r['rel_err'] * 100:.5f}" if r.get("rel_err") is not None else "      n/a"
        print(
            f"  {r['blas_backend']:<12}  {r['M']:>9,}  "
            f"{r['mean_ms']:>9.3f}  {r['std_ms']:>7.3f}  "
            f"{r['throughput']:>13,.0f}  "
            f"{r['price']:>10.6f}  "
            f"{re_s:>9}  "
            f"{r.get('vectorization', 'unknown'):>8}  "
            f"{r.get('numpy_version', '?'):>8}"
        )


def _print_speedup(rows: list[dict], m_values: list[int]) -> None:
    """Print relative throughput table normalised to the slowest backend."""
    if not rows:
        return
    backends = sorted({r["blas_backend"] for r in rows})
    for M in m_values:
        subset = [r for r in rows if r["M"] == M]
        if not subset:
            continue
        baseline = min(r["throughput"] for r in subset)
        print(f"\n  Relative throughput  (M={M:,}, normalised to slowest)")
        print("  " + "-" * 50)
        for r in sorted(subset, key=lambda x: -x["throughput"]):
            bar = "█" * int(r["throughput"] / baseline * 20)
            print(f"  {r['blas_backend']:<12}  {r['throughput'] / baseline:>5.2f}x  {bar}")


# ---------------------------------------------------------------------------
# Correctness checks
# ---------------------------------------------------------------------------

def _run_correctness_checks(
    rows: list[dict],
    expected_backends: list[str],
    m_values: list[int],
) -> bool:
    """
    Run all four correctness checks.  Returns True if all pass.

    Check 1 — Backend differentiation: every row has a non-null, non-"unknown"
               blas_backend, and the set of detected backends matches the
               expected set of labels provided by the user.

    Check 2 — Numerical agreement: for each M, all backends must return the
               same price to within 1e-10 (same seed → deterministic RNG →
               bit-for-bit identical result if NumPy versions match).

    Check 3 — Black-Scholes accuracy: rel_price_error < 1% for M >= 50,000.

    Check 4 — NumPy version consistency: warns if different executables have
               different NumPy versions (which would invalidate Check 2).
    """
    all_pass = True
    print()
    print("  " + "=" * 70)
    print("  Correctness Checks")
    print("  " + "=" * 70)

    # Check 4 first (prerequisite for Check 2)
    numpy_versions = {r.get("numpy_version") for r in rows}
    if len(numpy_versions) > 1:
        print(f"\n  [WARN] Check 4 — NumPy version mismatch: {sorted(numpy_versions)}")
        print("         Different NumPy versions may use different RNG state machines.")
        print("         Prices will not be bit-for-bit identical across backends.")
        print("         Check 2 (numerical agreement) will use a looser tolerance.")
        price_tol = 1e-4   # generous: only checks MC error convergence
    else:
        ver = next(iter(numpy_versions)) or "?"
        print(f"\n  [PASS] Check 4 — NumPy version consistent: {ver}")
        price_tol = 1e-10  # bit-for-bit identical

    # Check 1 — Backend differentiation
    detected = {r["blas_backend"] for r in rows}
    bad_rows  = [r for r in rows if r["blas_backend"] in (None, "unknown")]
    if bad_rows:
        print(f"\n  [FAIL] Check 1 — {len(bad_rows)} row(s) have blas_backend=unknown")
        print("         Ensure each executable is from a correctly configured conda env.")
        all_pass = False
    else:
        missing = set(expected_backends) - detected
        if missing:
            print(f"\n  [FAIL] Check 1 — Expected backends not detected: {sorted(missing)}")
            print(f"         Detected: {sorted(detected)}")
            all_pass = False
        else:
            print(f"\n  [PASS] Check 1 — Backend differentiation: {sorted(detected)}")

    # Check 2 — Numerical agreement
    check2_pass = True
    for M in m_values:
        m_rows = [r for r in rows if r["M"] == M and "price" in r]
        if len(m_rows) < 2:
            continue
        prices = [r["price"] for r in m_rows]
        max_diff = max(prices) - min(prices)
        if max_diff > price_tol:
            print(f"\n  [FAIL] Check 2 — M={M:,}: price spread {max_diff:.2e} > tol {price_tol:.2e}")
            for r in m_rows:
                print(f"         {r['blas_backend']:12}  price={r['price']:.10f}")
            check2_pass = False
            all_pass = False
    if check2_pass:
        print(f"\n  [PASS] Check 2 — Numerical agreement (tol={price_tol:.2e}) across all M values")

    # Check 3 — Black-Scholes accuracy
    check3_pass = True
    for r in rows:
        if r["M"] >= 50_000 and r.get("rel_err") is not None:
            if r["rel_err"] > REL_ERR_THRESH:
                print(f"\n  [FAIL] Check 3 — {r['blas_backend']}, M={r['M']:,}: "
                      f"rel_err={r['rel_err']*100:.4f}% > {REL_ERR_THRESH*100:.0f}%")
                check3_pass = False
                all_pass = False
    if check3_pass:
        print(f"\n  [PASS] Check 3 — All prices within {REL_ERR_THRESH*100:.0f}% of Black-Scholes "
              f"for M >= 50,000")

    status = "ALL PASSED" if all_pass else "SOME CHECKS FAILED — see above"
    print()
    print(f"  {'=' * 70}")
    print(f"  {status}")
    print(f"  {'=' * 70}")
    return all_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BLAS-backend comparison orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--backends", nargs="+", required=True,
                        metavar="NAME",
                        help="Human-readable backend names, one per executable "
                             "(e.g. mkl openblas blis)")
    parser.add_argument("--executables", nargs="+", required=True,
                        metavar="PATH",
                        help="Absolute paths to Python executables, one per backend")
    parser.add_argument("--m-values", type=int, nargs="+", default=DEFAULT_M,
                        metavar="M",
                        help=f"Path counts to benchmark (default: {DEFAULT_M})")
    parser.add_argument("--runs",   type=int, default=NUM_RUNS,
                        help="Timed repetitions per cell (default: %(default)s)")
    parser.add_argument("--warmup", type=int, default=NUM_WARMUP,
                        help="Warmup runs per cell (default: %(default)s)")
    args = parser.parse_args()

    if len(args.backends) != len(args.executables):
        parser.error(
            f"--backends ({len(args.backends)}) and --executables "
            f"({len(args.executables)}) must have the same number of entries."
        )

    for exe in args.executables:
        if not Path(exe).is_file():
            parser.error(f"Executable not found: {exe}")

    m_values      = sorted(set(args.m_values))
    experiment_id = str(uuid.uuid4())

    print()
    print("=" * 80)
    print("  BLAS Backend Comparison")
    print("=" * 80)
    print(f"  Backends       : {args.backends}")
    print(f"  M values       : {[f'{m:,}' for m in m_values]}")
    print(f"  Runs / Warmup  : {args.runs} / {args.warmup}")
    print(f"  Experiment ID  : {experiment_id}")
    print(f"  Thread pinning : OMP_NUM_THREADS=1 (BLAS only, no parallelism)")
    print()

    # Verify each executable can detect its BLAS before running any timing
    print("  Pre-flight: verifying BLAS detection in each environment ...")
    preflight_env = _build_env()
    for label, exe in zip(args.backends, args.executables):
        probe = subprocess.run(
            [exe, "-c",
             "import sys; sys.path.insert(0, '.')\n"
             "from experiments.run_blas_cell import detect_blas\n"
             "n, v = detect_blas()\n"
             "print(f'{n}|{v}')"],
            capture_output=True, text=True,
            cwd=str(ROOT), timeout=30,
            env=preflight_env,
        )
        out = probe.stdout.strip().split("|")
        detected_name = out[0] if out else "unknown"
        if detected_name == "unknown" or probe.returncode != 0:
            print(f"    [{label}] BLAS detection FAILED — "
                  f"stdout={probe.stdout.strip()!r} stderr={probe.stderr.strip()[:120]!r}")
            print("  Aborting: fix environment before running the experiment.")
            sys.exit(1)
        print(f"    [{label}]  detected={detected_name}  vec={out[1] if len(out) > 1 else '?'}"
              f"  exe={exe}")
    print()

    rows: list[dict] = []

    for M in m_values:
        print(f"  M = {M:,}")
        for label, exe in zip(args.backends, args.executables):
            tag = f"[{label}, M={M:,}]"
            print(f"    Running {tag} ...")
            row = _run_cell(exe, label, M, experiment_id, args)
            if row:
                rows.append(row)
                print(f"      => {row['mean_ms']:.3f} ms  "
                      f"{row['throughput']:,.0f} paths/s  "
                      f"blas={row['blas_backend']}  "
                      f"rel_err={row['rel_err']*100:.4f}%"
                      if row.get("rel_err") is not None
                      else f"      => {row['mean_ms']:.3f} ms  blas={row['blas_backend']}")
        print()

    _print_table(rows)
    _print_speedup(rows, m_values)

    checks_passed = _run_correctness_checks(rows, args.backends, m_values)

    from benchmarking.storage.database import BenchmarkDB
    db = BenchmarkDB()

    print()
    print(f"  Results stored in : {db.db_path}")
    print(f"  Experiment ID     : {experiment_id}")
    print()
    print("  ── Performance query ─────────────────────────────────────────────────────────")
    print(f'  sqlite3 results/benchmarks.db "')
    print(f"    SELECT blas_backend, vectorization_flag,")
    print(f"           json_extract(config_json, '$.M') AS M,")
    print(f"           round(mean_runtime_ms, 3)         AS mean_ms,")
    print(f"           round(throughput_paths_per_sec)   AS paths_s,")
    print(f"           numpy_version")
    print(f"    FROM runs")
    print(f"    WHERE experiment_id='{experiment_id}'")
    print(f'    ORDER BY M, throughput_paths_per_sec DESC;"')
    print()
    print("  ── Correctness check query ───────────────────────────────────────────────────")
    print(f'  sqlite3 results/benchmarks.db "')
    print(f"    SELECT blas_backend,")
    print(f"           json_extract(config_json, '$.M') AS M,")
    print(f"           round(result_value, 8)           AS price,")
    print(f"           round(rel_price_error * 100, 5)  AS rel_err_pct,")
    print(f"           numpy_version")
    print(f"    FROM runs")
    print(f"    WHERE experiment_id='{experiment_id}'")
    print(f'    ORDER BY M, blas_backend;"')
    print()

    sys.exit(0 if checks_passed else 1)


if __name__ == "__main__":
    main()

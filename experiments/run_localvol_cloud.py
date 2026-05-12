"""
Local Volatility Cloud Profile Benchmark
==========================================
Profiles all available engines on the European local-vol workload across a
range of path counts, capturing GCP instance metadata and cost-per-run for
hardware cost/performance analysis.

This is the primary experiment for Stage 5 (Cloud Resource Profiling).

Engines
-------
  cpu   — NumPy baseline (no AD)
  jax   — JAX/XLA, sweeps all requested --ad-modes
  cpp   — C++ OpenMP (no AD, skipped if not built)
  rust  — Rust Rayon (no AD, skipped if not built)

Workload
--------
  EuropeanLocalVolConfig with default 4-parameter local vol surface,
  N=252 time steps, configurable M paths.

Cloud integration
-----------------
  On a GCP VM the instance type and zone are auto-detected from the metadata
  server.  Hourly rates are resolved from the GCP Billing Catalog API
  (GCP_PRICING_API_KEY) or from the bundled KNOWN_RATES table.  cost_per_run
  is computed per row and stored alongside timing data.

Usage
-----
  # Local smoke test (no cloud metadata, cost_per_run = NULL)
  python experiments/run_localvol_cloud.py \\
      --runs 2 --warmup 1 --m-values 1000 --ad-modes none

  # Full local vol profile on GCP VM
  python experiments/run_localvol_cloud.py \\
      --runs 7 --warmup 3

  # Override instance metadata manually (e.g. running on-prem for comparison)
  python experiments/run_localvol_cloud.py \\
      --instance-type n2-standard-8 --cloud-provider gcp --hourly-rate 0.3882
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from benchmarking.core.config import EuropeanLocalVolConfig
from benchmarking.runner.runner import BenchmarkRunner
from benchmarking.storage.database import BenchmarkDB
from benchmarking.cloud.metadata import get_instance_metadata
from benchmarking.cloud.pricing import get_hourly_rate, compute_cost_per_run

# Optional engines — silently skipped if not built
try:
    from benchmarking.workloads.mc_cpp import CPPMonteCarloEngine
    _CPP = CPPMonteCarloEngine()
except Exception:
    _CPP = None

try:
    from benchmarking.workloads.mc_rust import RustMonteCarloEngine
    _RUST = RustMonteCarloEngine()
except Exception:
    _RUST = None

from benchmarking.workloads.mc_cpu import CPUMonteCarloEngine
from benchmarking.workloads.mc_jax import JAXMonteCarloEngine

DEFAULT_M_VALUES  = [10_000, 50_000, 100_000]
DEFAULT_AD_MODES  = ["none", "forward", "reverse"]
NUM_WARMUP        = 3
NUM_RUNS          = 7

# Engine language/backend metadata
_ENGINE_META = {
    "cpu":  ("python", "numpy"),
    "jax":  ("python", "xla"),
    "cpp":  ("cpp",    "openmp"),
    "rust": ("rust",   "rayon"),
}


def _fmt(v, fmt=".4f") -> str:
    return f"{v:{fmt}}" if v is not None else "    n/a "


def _resolve_cloud_args(args: argparse.Namespace) -> tuple[str | None, str | None, float | None]:
    """
    Return (cloud_provider, instance_type, hourly_rate) after applying CLI
    overrides on top of auto-detected metadata.
    """
    meta = get_instance_metadata()

    cloud_provider = args.cloud_provider or meta.get("cloud_provider")
    instance_type  = args.instance_type  or meta.get("instance_type")

    if args.hourly_rate is not None:
        hourly_rate = args.hourly_rate
    elif instance_type:
        zone   = meta.get("zone") or ""
        region = "-".join(zone.split("-")[:2]) if zone else "us-central1"
        hourly_rate = get_hourly_rate(
            instance_type,
            region=region,
            api_key=args.gcp_api_key,
        )
    else:
        hourly_rate = None

    return cloud_provider, instance_type, hourly_rate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local volatility cloud profile benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--runs",   type=int, default=NUM_RUNS,
                        help="Timed repetitions per cell (default: %(default)s)")
    parser.add_argument("--warmup", type=int, default=NUM_WARMUP,
                        help="Warmup runs per cell (default: %(default)s)")
    parser.add_argument("--m-values", type=int, nargs="+", default=DEFAULT_M_VALUES,
                        metavar="M",
                        help=f"Path counts to sweep (default: {DEFAULT_M_VALUES})")
    parser.add_argument("--ad-modes", nargs="+", default=DEFAULT_AD_MODES,
                        choices=["none", "forward", "reverse"],
                        metavar="MODE",
                        help="AD modes for JAX engine (default: none forward reverse)")
    # Cloud metadata overrides
    parser.add_argument("--instance-type",  default=None,
                        help="GCP machine type (auto-detected on GCP VMs)")
    parser.add_argument("--cloud-provider", default=None,
                        help="Cloud provider name, e.g. 'gcp' (auto-detected)")
    parser.add_argument("--hourly-rate",    type=float, default=None,
                        help="Instance hourly rate USD (overrides pricing API)")
    parser.add_argument("--gcp-api-key",    default=None,
                        help="GCP API key for Billing Catalog (or set GCP_PRICING_API_KEY)")
    args = parser.parse_args()

    db            = BenchmarkDB()
    experiment_id = str(uuid.uuid4())

    # Resolve cloud context once for the whole run
    cloud_provider, instance_type, hourly_rate = _resolve_cloud_args(args)

    # Build engine list (cpu always present; jax always present; cpp/rust optional)
    engines = [("cpu", CPUMonteCarloEngine()), ("jax", JAXMonteCarloEngine())]
    if _CPP  is not None:
        engines.append(("cpp",  _CPP))
    if _RUST is not None:
        engines.append(("rust", _RUST))

    eng_names = [e for e, _ in engines]

    print()
    print("=" * 80)
    print("  Local Volatility Cloud Profile Benchmark")
    print("=" * 80)
    print(f"  M values       : {args.m_values}")
    print(f"  AD modes (JAX) : {args.ad_modes}")
    print(f"  Engines        : {eng_names}")
    print(f"  Runs           : {args.runs}  Warmup: {args.warmup}")
    if cloud_provider:
        print(f"  Cloud provider : {cloud_provider}")
    if instance_type:
        print(f"  Instance type  : {instance_type}")
    if hourly_rate is not None:
        print(f"  Hourly rate    : ${hourly_rate:.6f}/hr")
    else:
        print("  Hourly rate    : not available (cost_per_run will be NULL)")
    print()

    rows = []

    for M in args.m_values:
        config = EuropeanLocalVolConfig(M=M)

        for eng_name, engine in engines:
            # Non-JAX engines only support ad_mode="none"
            if eng_name != "jax":
                run_modes = ["none"]
            else:
                run_modes = args.ad_modes

            for ad_mode in run_modes:
                runner = BenchmarkRunner(
                    engine,
                    name=f"{eng_name}/local_vol/{ad_mode}/M{M}",
                )
                try:
                    res = runner.run(
                        config,
                        num_warmup=args.warmup,
                        num_runs=args.runs,
                        ad_mode=ad_mode,
                    )
                except Exception as exc:
                    print(f"  [{eng_name}, M={M}, {ad_mode}] FAILED: {exc}")
                    continue

                mean_ms    = res.mean_runtime * 1000
                std_ms     = res.std_runtime  * 1000
                min_ms     = res.min_runtime  * 1000
                max_ms     = res.max_runtime  * 1000
                price      = res.result
                throughput = res.throughput_paths_per_sec
                overhead   = res.ad_overhead_ratio
                baseline_ms = res.baseline_mean_ms
                greeks     = res.greeks  # dict or None
                language, backend = _ENGINE_META.get(eng_name, ("python", "cpu"))
                env        = res.metadata

                cost_per_run = (
                    compute_cost_per_run(mean_ms, hourly_rate)
                    if hourly_rate is not None else None
                )

                # Local vol has 6 sensitivities; store in greeks_json
                greeks_json = json.dumps(greeks) if greeks else None

                db.store_run_full(
                    config_dict=config.to_dict(),
                    engine=eng_name,
                    ad_mode=ad_mode,
                    experiment_id=experiment_id,
                    experiment_type="localvol_cloud_profile",
                    mean_runtime_ms=mean_ms,
                    std_runtime_ms=std_ms,
                    min_runtime_ms=min_ms,
                    max_runtime_ms=max_ms,
                    throughput_paths_per_sec=throughput,
                    baseline_mean_ms=baseline_ms,
                    ad_overhead_ratio=overhead,
                    result_value=price,
                    memory_peak_mb=res.memory_peak_mb,
                    language=language,
                    backend=backend,
                    cloud_provider=cloud_provider,
                    instance_type=instance_type,
                    cost_per_run=cost_per_run,
                    cpu_model=env.get("cpu_model"),
                    cpu_architecture=env.get("cpu_architecture"),
                    cpu_count=env.get("cpu_count"),
                    memory_gb=env.get("memory_gb"),
                    platform=env.get("platform"),
                    python_version=env.get("python_version"),
                    numpy_version=env.get("numpy_version"),
                    jax_version=env.get("jax_version"),
                )

                rows.append({
                    "M":          M,
                    "engine":     eng_name,
                    "ad_mode":    ad_mode,
                    "mean_ms":    mean_ms,
                    "std_ms":     std_ms,
                    "throughput": throughput,
                    "overhead":   overhead,
                    "price":      price,
                    "cost":       cost_per_run,
                    "mem_mb":     res.memory_peak_mb,
                })

                oh_str = (f"{overhead:.2f}x" if ad_mode != "none" else "  1.00x")
                cost_str = (f"${cost_per_run:.2e}" if cost_per_run is not None else "     n/a")
                print(
                    f"  [{eng_name:<4} / M={M:>7,} / {ad_mode:<8}]  "
                    f"{mean_ms:>9.3f} ms ± {std_ms:.3f}  "
                    f"overhead {oh_str}  "
                    f"price {price:.5f}  "
                    f"cost {cost_str}"
                )

    # --------------------------------------------------------------------------
    # Summary table
    # --------------------------------------------------------------------------
    print()
    print("  " + "=" * 100)
    print("  Summary")
    print("  " + "=" * 100)
    hdr = (
        f"  {'Engine':<6}  {'M':>7}  {'AD':>8}  {'Mean ms':>9}  {'Std ms':>7}  "
        f"{'Paths/s':>12}  {'Overhead':>8}  {'Price':>9}  "
        f"{'Cost/run':>10}  {'Mem MB':>7}"
    )
    print(hdr)
    print("  " + "-" * 100)
    for r in rows:
        oh  = f"{r['overhead']:.2f}x" if r["ad_mode"] != "none" else "  1.00x"
        c   = f"${r['cost']:.3e}" if r["cost"] is not None else "       n/a"
        print(
            f"  {r['engine']:<6}  {r['M']:>7,}  {r['ad_mode']:>8}  "
            f"{r['mean_ms']:>9.3f}  {r['std_ms']:>7.3f}  "
            f"{r['throughput']:>12.0f}  {oh:>8}  "
            f"{r['price']:>9.5f}  {c:>10}  "
            f"{_fmt(r['mem_mb'], '.1f'):>7}"
        )

    print()
    print(f"  Results stored in : {db.db_path}")
    print(f"  Experiment ID     : {experiment_id}")
    if instance_type:
        print(f"  Instance type     : {instance_type}")
    print()


if __name__ == "__main__":
    main()

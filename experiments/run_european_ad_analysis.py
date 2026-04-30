"""
European Option AD Analysis
=============================
Compare JAX none / forward / reverse AD modes over a range of path counts.

For each (M, ad_mode) combination:
  - times the run (warmup excluded)
  - computes Greeks
  - computes analytical Greeks for comparison
  - stores everything to SQLite

Prints:
  1. Overhead table: M | AD mode | runtime ms | overhead | price | δ | ν | ρ | rel errors
  2. Greek comparison table: Greek | Forward | Reverse | Analytical | Fwd Err | Rev Err

Usage
-----
    python experiments/run_european_ad_analysis.py
"""

import argparse
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from benchmarking.core.config import EuropeanOptionConfig
from benchmarking.runner.runner import BenchmarkRunner
from benchmarking.storage.database import BenchmarkDB
from benchmarking.workloads.mc_cpu import european_analytical_greeks
from benchmarking.workloads.mc_jax import JAXMonteCarloEngine

# Path counts to sweep
M_VALUES = [1_000, 5_000, 10_000, 50_000, 100_000]
AD_MODES  = ["none", "forward", "reverse"]
NUM_WARMUP = 2   # overridden by --warmup
NUM_RUNS   = 5   # overridden by --runs

BASE_CFG = dict(S0=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0,
                option_type="call", seed=42)


def _abs_err(x, ref):
    return abs(x - ref) if (x is not None and ref is not None) else None


def _rel_err(x, ref):
    if x is None or ref is None or ref == 0:
        return None
    return abs(x - ref) / abs(ref)


def _fmt(v, fmt=".4f"):
    return f"{v:{fmt}}" if v is not None else "   n/a  "


def main() -> None:
    parser = argparse.ArgumentParser(description="European option AD analysis")
    parser.add_argument("--runs",   type=int, default=NUM_RUNS,   help="Timed repetitions (default: %(default)s)")
    parser.add_argument("--warmup", type=int, default=NUM_WARMUP, help="Warmup runs (default: %(default)s)")
    args = parser.parse_args()

    db = BenchmarkDB()
    jax_engine = JAXMonteCarloEngine()
    experiment_id = str(uuid.uuid4())

    # -----------------------------------------------------------------------
    # 1. Run all (M, ad_mode) combinations and collect results
    # -----------------------------------------------------------------------
    records = {}  # (M, ad_mode) → result dict

    print()
    print("=" * 80)
    print("  European Option AD Analysis")
    print("=" * 80)
    print(f"  Path counts : {M_VALUES}")
    print(f"  AD modes    : {AD_MODES}")
    print(f"  Runs        : {args.runs}  Warmup: {args.warmup}")
    print()

    for M in M_VALUES:
        config = EuropeanOptionConfig(**BASE_CFG, M=M)
        analytical = european_analytical_greeks(config)
        ana_price, ana_delta, ana_vega, ana_rho = (
            analytical["price"], analytical["delta"],
            analytical["vega"],  analytical["rho"],
        )

        for ad_mode in AD_MODES:
            runner = BenchmarkRunner(jax_engine, name=f"jax/european/{ad_mode}/M{M}")
            try:
                res = runner.run(config, num_warmup=args.warmup, num_runs=args.runs,
                                 ad_mode=ad_mode)
            except Exception as exc:
                print(f"  [M={M}, {ad_mode}] FAILED: {exc}")
                continue

            mean_ms     = res.mean_runtime * 1000
            std_ms      = res.std_runtime  * 1000
            min_ms      = res.min_runtime  * 1000
            max_ms      = res.max_runtime  * 1000
            price       = res.result
            greeks      = res.greeks or {}
            throughput  = res.throughput_paths_per_sec
            overhead    = res.ad_overhead_ratio
            baseline_ms = res.baseline_mean_ms or mean_ms

            delta = greeks.get("delta")
            vega  = greeks.get("vega")
            rho   = greeks.get("rho")

            abs_p_err = _abs_err(price, ana_price)
            rel_p_err = _rel_err(price, ana_price)
            abs_d_err = _abs_err(delta, ana_delta)
            rel_d_err = _rel_err(delta, ana_delta)
            abs_v_err = _abs_err(vega,  ana_vega)
            rel_v_err = _rel_err(vega,  ana_vega)
            abs_r_err = _abs_err(rho,   ana_rho)
            rel_r_err = _rel_err(rho,   ana_rho)

            env = res.metadata
            db.store_run_full(
                config_dict=config.to_dict(),
                engine="jax",
                ad_mode=ad_mode,
                experiment_id=experiment_id,
                experiment_type="european_ad_analysis",
                mean_runtime_ms=mean_ms,
                std_runtime_ms=std_ms,
                min_runtime_ms=min_ms,
                max_runtime_ms=max_ms,
                throughput_paths_per_sec=throughput,
                baseline_mean_ms=baseline_ms,
                ad_overhead_ratio=overhead,
                result_value=price,
                greek_delta=delta,
                greek_vega=vega,
                greek_rho=rho,
                analytical_price=ana_price,
                analytical_delta=ana_delta,
                analytical_vega=ana_vega,
                analytical_rho=ana_rho,
                abs_price_error=abs_p_err,
                rel_price_error=rel_p_err,
                abs_delta_error=abs_d_err,
                rel_delta_error=rel_d_err,
                abs_vega_error=abs_v_err,
                rel_vega_error=rel_v_err,
                abs_rho_error=abs_r_err,
                rel_rho_error=rel_r_err,
                memory_peak_mb=res.memory_peak_mb,
                language="python",
                backend="xla",
                cpu_model=env.get("cpu_model"),
                cpu_architecture=env.get("cpu_architecture"),
                cpu_count=env.get("cpu_count"),
                memory_gb=env.get("memory_gb"),
                platform=env.get("platform"),
                python_version=env.get("python_version"),
                numpy_version=env.get("numpy_version"),
                jax_version=env.get("jax_version"),
            )

            records[(M, ad_mode)] = {
                "M": M, "ad_mode": ad_mode,
                "mean_ms": mean_ms, "std_ms": std_ms,
                "overhead": overhead, "baseline_ms": baseline_ms,
                "price": price, "delta": delta, "vega": vega, "rho": rho,
                "rel_p_err": rel_p_err,
                "rel_d_err": rel_d_err,
                "rel_v_err": rel_v_err,
                "rel_r_err": rel_r_err,
                "analytical": analytical,
            }

    # -----------------------------------------------------------------------
    # 2. Print overhead table
    # -----------------------------------------------------------------------
    print("  AD Overhead Table")
    print("  " + "-" * 108)
    hdr = (f"  {'M':>7}  {'AD mode':<8}  {'ms':>8}  {'±ms':>6}  "
           f"{'overhead':>8}  {'price':>9}  "
           f"{'delta':>8}  {'vega':>8}  {'rho':>8}  "
           f"{'rel_p%':>7}  {'rel_d%':>7}  {'rel_v%':>7}  {'rel_r%':>7}")
    print(hdr)
    print("  " + "-" * 108)

    for M in M_VALUES:
        for ad_mode in AD_MODES:
            r = records.get((M, ad_mode))
            if r is None:
                continue
            oh = f"{r['overhead']:.2f}x" if ad_mode != "none" else "  1.00x"
            rel_p = f"{r['rel_p_err']*100:.3f}" if r["rel_p_err"] is not None else "  n/a"
            rel_d = f"{r['rel_d_err']*100:.3f}" if r["rel_d_err"] is not None else "  n/a"
            rel_v = f"{r['rel_v_err']*100:.3f}" if r["rel_v_err"] is not None else "  n/a"
            rel_r = f"{r['rel_r_err']*100:.3f}" if r["rel_r_err"] is not None else "  n/a"
            print(
                f"  {r['M']:>7}  {r['ad_mode']:<8}  {r['mean_ms']:>8.3f}  "
                f"{r['std_ms']:>6.3f}  {oh:>8}  {r['price']:>9.5f}  "
                f"{_fmt(r['delta'], '.5f'):>8}  {_fmt(r['vega'], '.5f'):>8}  "
                f"{_fmt(r['rho'], '.5f'):>8}  "
                f"{rel_p:>7}  {rel_d:>7}  {rel_v:>7}  {rel_r:>7}"
            )

    # -----------------------------------------------------------------------
    # 3. Print Greek comparison (largest M with forward+reverse)
    # -----------------------------------------------------------------------
    # Use the largest M that has both forward and reverse
    best_M = max((M for M in M_VALUES
                  if (M, "forward") in records and (M, "reverse") in records),
                 default=None)

    if best_M is not None:
        fwd = records[(best_M, "forward")]
        rev = records[(best_M, "reverse")]
        ana = fwd["analytical"]

        print()
        print(f"  Greek Comparison at M={best_M:,}")
        print("  " + "-" * 80)
        print(f"  {'Greek':<8}  {'Forward':>12}  {'Reverse':>12}  {'Analytical':>12}  "
              f"{'Fwd Abs Err':>12}  {'Rev Abs Err':>12}")
        print("  " + "-" * 80)
        for greek, ana_val in [("delta", ana["delta"]), ("vega", ana["vega"]), ("rho", ana["rho"])]:
            fv = fwd.get(greek)
            rv = rev.get(greek)
            fa = _abs_err(fv, ana_val)
            ra = _abs_err(rv, ana_val)
            print(
                f"  {greek:<8}  {_fmt(fv, '.6f'):>12}  {_fmt(rv, '.6f'):>12}  "
                f"{ana_val:>12.6f}  {_fmt(fa, '.6f'):>12}  {_fmt(ra, '.6f'):>12}"
            )

    print()
    print(f"  Results stored in : {db.db_path}")
    print(f"  Experiment ID     : {experiment_id}")
    print()


if __name__ == "__main__":
    main()

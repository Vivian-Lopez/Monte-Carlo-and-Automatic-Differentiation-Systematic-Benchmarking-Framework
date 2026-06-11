#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarking.cloud.pricing import get_hourly_rate
from benchmarking.profiling.guarded_sha.analysis import attach_scores, oracle_map
from benchmarking.profiling.guarded_sha.candidates import (
    default_instance_configs,
    generate_candidates,
)
from benchmarking.profiling.guarded_sha.config import LocalVolProfilerConfig, Observation
from benchmarking.profiling.guarded_sha.profiler import GuardedSHAProfiler
from benchmarking.profiling.guarded_sha.reporting import read_observations, write_outputs
from benchmarking.profiling.guarded_sha.runner import CandidateRunner


DEFAULT_INSTANCES = [
    "e2-standard-4",
    "n2-standard-4",
    "n2d-standard-4",
    "c2d-standard-4",
    "t2d-standard-4",
]

# Conservative fallback rates for local smoke tests and offline analysis. Real
# GCP runs should prefer GCP_PRICING_API_KEY or --hourly-rates-json.
FALLBACK_HOURLY_RATES = {
    "e2-standard-4": 0.134,
    "n2-standard-4": 0.19410,
    "n2d-standard-4": 0.173,
    "c2d-standard-4": 0.224,
    "t2d-standard-4": 0.17008,
}


class _CapabilityEngine:
    def __init__(self, workloads, ad_modes):
        self._workloads = set(workloads)
        self._ad_modes = tuple(ad_modes)

    def supports(self, workload_type: str) -> bool:
        return workload_type in self._workloads

    def supported_ad_modes(self):
        return self._ad_modes


def _parse_csv_ints(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_csv_strings(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _load_hourly_rates(path: str | None) -> Dict[str, float]:
    if not path:
        return {}
    raw = json.loads(Path(path).read_text())
    return {str(key): float(value) for key, value in raw.items()}


def _load_engines(dry_run: bool = False) -> dict:
    if dry_run:
        return {
            "cpu": _CapabilityEngine(["european_local_vol"], ["none"]),
            "jax": _CapabilityEngine(["european_local_vol"], ["none", "forward", "reverse"]),
            "cpp": _CapabilityEngine(["european_local_vol"], ["none"]),
            "rust": _CapabilityEngine(["european_local_vol"], ["none"]),
        }
    from benchmarking.workloads.mc_cpu import CPUMonteCarloEngine

    try:
        from benchmarking.workloads.mc_jax import JAXMonteCarloEngine
        jax_engine = JAXMonteCarloEngine()
    except BaseException as exc:
        print(f"[WARN] JAX unavailable: {exc}")
        jax_engine = None

    try:
        from benchmarking.workloads.mc_cpp import CPPMonteCarloEngine
        cpp_engine = CPPMonteCarloEngine()
    except Exception:
        cpp_engine = None

    try:
        from benchmarking.workloads.mc_rust import RustMonteCarloEngine
        rust_engine = RustMonteCarloEngine()
    except Exception:
        rust_engine = None

    engines = {
        "cpu": CPUMonteCarloEngine(),
        "jax": jax_engine,
        "cpp": cpp_engine,
        "rust": rust_engine,
    }
    return engines


def _resolve_rates(
    instance_names: List[str],
    region: str,
    rates_json: str | None,
    gcp_api_key: str | None,
) -> Dict[str, float | None]:
    explicit_rates = _load_hourly_rates(rates_json)
    rates: Dict[str, float | None] = {}
    for name in instance_names:
        if name in explicit_rates:
            rates[name] = explicit_rates[name]
            continue
        rates[name] = get_hourly_rate(name, region=region, api_key=gcp_api_key)
        if rates[name] is None:
            rates[name] = FALLBACK_HOURLY_RATES.get(name)
    return rates


def _build_config(args: argparse.Namespace) -> LocalVolProfilerConfig:
    config = LocalVolProfilerConfig.from_json(args.config) if args.config else LocalVolProfilerConfig()
    instance_names = _parse_csv_strings(args.instances) if args.instances else DEFAULT_INSTANCES
    rates = _resolve_rates(instance_names, args.region, args.hourly_rates_json, args.gcp_api_key)
    instances = default_instance_configs(instance_names, rates, region=args.region)
    config = config.with_instances(instances)
    config.probe_budgets = _parse_csv_ints(args.probe_budgets)
    config.full_budget = int(args.full_budget)
    config.repeats = int(args.repeats)
    config.warmup = int(args.warmup)
    return config


def _timestamped_output_dir(base: str) -> Path:
    return Path(base) / datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> int:
    parser = argparse.ArgumentParser(description="Guarded SHA local-volatility profiler")
    parser.add_argument("--config", default=None, help="Optional JSON profiler config")
    parser.add_argument("--run-full-grid", action="store_true")
    parser.add_argument("--run-plain-sha", action="store_true")
    parser.add_argument("--run-guarded-sha", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--instances", default=",".join(DEFAULT_INSTANCES))
    parser.add_argument("--full-budget", default="100000")
    parser.add_argument("--probe-budgets", default="1000,5000,25000")
    parser.add_argument("--repeats", default="3")
    parser.add_argument("--warmup", default="1")
    parser.add_argument("--output-dir", default="results/guarded_sha_local_vol")
    parser.add_argument("--hourly-rates-json", default=None)
    parser.add_argument("--gcp-api-key", default=None)
    parser.add_argument("--region", default="europe-west1")
    parser.add_argument(
        "--input-observations",
        nargs="*",
        default=None,
        help="Existing candidate_observations.csv files to analyse instead of running benchmarks",
    )
    args = parser.parse_args()

    if not (args.run_full_grid or args.run_plain_sha or args.run_guarded_sha):
        args.run_full_grid = args.run_plain_sha = args.run_guarded_sha = True

    config = _build_config(args)
    engines = _load_engines(dry_run=args.dry_run or bool(args.input_observations))
    candidates, rejected = generate_candidates(config, engines)
    output_dir = _timestamped_output_dir(args.output_dir)

    profiler = GuardedSHAProfiler(config=config, candidates=candidates)

    if args.input_observations:
        observations = read_observations([Path(path) for path in args.input_observations])
    else:
        runner = CandidateRunner(
            engines=engines,
            dry_run=args.dry_run,
            num_warmup=config.warmup,
            num_runs=config.repeats,
        )
        profiler = GuardedSHAProfiler(config=config, candidates=candidates, runner=runner)
        observations = profiler.run_observation_pool()

    attach_scores(observations, config.objectives)
    full_observations = [obs for obs in observations if obs.budget_M == config.full_budget]
    oracle = oracle_map(config.objectives, full_observations, config.full_budget)

    recommendations = []
    decisions = []
    if args.run_full_grid:
        recommendations.extend(profiler.oracle_recommendations(observations))
    if args.run_plain_sha:
        plain_recs, plain_decisions = profiler.simulate_plain_sha(observations, oracle)
        recommendations.extend(plain_recs)
        decisions.extend(plain_decisions)
    if args.run_guarded_sha:
        guarded_recs, guarded_decisions = profiler.simulate_guarded_sha(observations, oracle)
        recommendations.extend(guarded_recs)
        decisions.extend(guarded_decisions)

    write_outputs(output_dir, observations, decisions, recommendations, rejected)
    print(f"Guarded SHA outputs written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

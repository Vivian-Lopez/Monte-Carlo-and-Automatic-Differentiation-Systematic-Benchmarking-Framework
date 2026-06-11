import subprocess
import sys
from pathlib import Path

from benchmarking.profiling.guarded_sha.config import (
    CandidateConfig,
    CloudInstanceConfig,
    LocalVolProfilerConfig,
)
from benchmarking.profiling.guarded_sha.profiler import GuardedSHAProfiler
from benchmarking.profiling.guarded_sha.runner import CandidateRunner


def _config():
    cfg = LocalVolProfilerConfig()
    cfg.instances = [
        CloudInstanceConfig("e2-standard-4", hourly_rate=0.1),
        CloudInstanceConfig("c2d-standard-4", hourly_rate=0.2),
    ]
    cfg.engines = ["cpu", "rust"]
    cfg.ad_modes = ["none"]
    cfg.probe_budgets = [1_000, 5_000]
    cfg.full_budget = 10_000
    cfg.repeats = 1
    cfg.warmup = 0
    cfg.include_ad_required = False
    return cfg


def _candidates(cfg):
    return [
        CandidateConfig("european_local_vol", "price_only", "cpu", "none", "e2-standard-4", 0.1),
        CandidateConfig("european_local_vol", "price_only", "rust", "none", "c2d-standard-4", 0.2),
    ]


def test_profiler_records_decision_trace_in_dry_run():
    cfg = _config()
    candidates = _candidates(cfg)
    runner = CandidateRunner({"cpu": None, "rust": None}, dry_run=True, num_warmup=0, num_runs=1)
    profiler = GuardedSHAProfiler(cfg, candidates, runner)
    observations = profiler.run_observation_pool()
    oracle = {key: value for key, value in []}
    recs, decisions = profiler.simulate_guarded_sha(observations, oracle)
    assert observations
    assert decisions
    assert recs
    assert all(decision.reason for decision in decisions)


def test_cli_dry_run_writes_structured_outputs(tmp_path):
    cmd = [
        sys.executable,
        "experiments/run_guarded_sha_local_vol.py",
        "--dry-run",
        "--run-full-grid",
        "--run-plain-sha",
        "--run-guarded-sha",
        "--instances",
        "e2-standard-4,c2d-standard-4",
        "--probe-budgets",
        "1000,5000",
        "--full-budget",
        "10000",
        "--repeats",
        "1",
        "--warmup",
        "0",
        "--output-dir",
        str(tmp_path),
    ]
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    out_dir = Path(result.stdout.strip().split()[-1])
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "candidate_observations.csv").exists()
    assert (out_dir / "sha_rounds.csv").exists()
    assert (out_dir / "recommendations.json").exists()
    assert (out_dir / "guarded_sha_report.md").exists()


from __future__ import annotations

import hashlib
import math
import random
from typing import Dict, Optional

from benchmarking.cloud.pricing import compute_cost_per_run
from benchmarking.core.engine import MonteCarloEngine
from benchmarking.profiling.guarded_sha.config import CandidateConfig, Observation


class CandidateRunner:
    def __init__(
        self,
        engines: Dict[str, MonteCarloEngine | None],
        dry_run: bool = False,
        num_warmup: int = 1,
        num_runs: int = 3,
    ) -> None:
        self.engines = engines
        self.dry_run = dry_run
        self.num_warmup = num_warmup
        self.num_runs = num_runs

    def run_candidate(self, candidate: CandidateConfig, workload, budget_M: int, stage: str) -> Observation:
        if candidate.hourly_rate is None:
            return Observation(
                candidate_id=candidate.candidate_id,
                task=candidate.task,
                engine=candidate.engine,
                ad_mode=candidate.ad_mode,
                instance_type=candidate.instance_type,
                budget_M=budget_M,
                stage=stage,
                valid=False,
                correctness_passed=False,
                failure_reason="missing hourly rate",
            )
        if self.dry_run:
            return self._dry_observation(candidate, budget_M, stage)

        engine = self.engines.get(candidate.engine)
        if engine is None:
            return Observation(
                candidate_id=candidate.candidate_id,
                task=candidate.task,
                engine=candidate.engine,
                ad_mode=candidate.ad_mode,
                instance_type=candidate.instance_type,
                budget_M=budget_M,
                stage=stage,
                valid=False,
                failure_reason="engine unavailable",
            )
        try:
            from benchmarking.runner.runner import BenchmarkRunner

            runner = BenchmarkRunner(engine, name=f"{candidate.engine}_{candidate.task}")
            result = runner.run(
                workload,
                num_warmup=self.num_warmup,
                num_runs=self.num_runs,
                ad_mode=candidate.ad_mode,
            )
            runtime_ms = result.mean_runtime * 1000.0
            cost = compute_cost_per_run(runtime_ms, candidate.hourly_rate)
            greeks = result.greeks or None
            if candidate.task == "ad_required" and not greeks:
                return Observation(
                    candidate_id=candidate.candidate_id,
                    task=candidate.task,
                    engine=candidate.engine,
                    ad_mode=candidate.ad_mode,
                    instance_type=candidate.instance_type,
                    budget_M=budget_M,
                    stage=stage,
                    valid=False,
                    runtime_ms=runtime_ms,
                    cost_per_run=cost,
                    price=result.result,
                    greeks=greeks,
                    memory_peak_mb=result.memory_peak_mb,
                    correctness_passed=False,
                    failure_reason="missing required greeks",
                )
            if result.result is None or not math.isfinite(float(result.result)):
                return Observation(
                    candidate_id=candidate.candidate_id,
                    task=candidate.task,
                    engine=candidate.engine,
                    ad_mode=candidate.ad_mode,
                    instance_type=candidate.instance_type,
                    budget_M=budget_M,
                    stage=stage,
                    valid=False,
                    runtime_ms=runtime_ms,
                    cost_per_run=cost,
                    price=result.result,
                    greeks=greeks,
                    memory_peak_mb=result.memory_peak_mb,
                    correctness_passed=False,
                    failure_reason="non-finite price",
                )
            return Observation(
                candidate_id=candidate.candidate_id,
                task=candidate.task,
                engine=candidate.engine,
                ad_mode=candidate.ad_mode,
                instance_type=candidate.instance_type,
                budget_M=budget_M,
                stage=stage,
                valid=True,
                runtime_ms=runtime_ms,
                cost_per_run=cost,
                price=result.result,
                greeks=greeks,
                memory_peak_mb=result.memory_peak_mb,
            )
        except Exception as exc:
            return Observation(
                candidate_id=candidate.candidate_id,
                task=candidate.task,
                engine=candidate.engine,
                ad_mode=candidate.ad_mode,
                instance_type=candidate.instance_type,
                budget_M=budget_M,
                stage=stage,
                valid=False,
                failure_reason=str(exc),
            )

    def _dry_observation(self, candidate: CandidateConfig, budget_M: int, stage: str) -> Observation:
        digest = hashlib.sha256(candidate.candidate_id.encode()).hexdigest()
        jitter_seed = int(digest[:8], 16) + int(budget_M)
        rng = random.Random(jitter_seed)
        engine_factor = {
            "cpu": 4.0,
            "jax": 1.8 if candidate.ad_mode == "none" else 3.2,
            "cpp": 1.25,
            "rust": 1.0,
        }.get(candidate.engine, 3.0)
        instance_factor = {
            "e2": 1.25,
            "n2": 1.0,
            "n2d": 0.9,
            "c2d": 0.82,
            "t2d": 0.86,
        }.get(candidate.instance_family, 1.0)
        ad_factor = {"none": 1.0, "forward": 2.4, "reverse": 1.7}.get(candidate.ad_mode, 1.0)
        fixed_ms = 18.0 if candidate.engine == "jax" else 2.0
        runtime_ms = fixed_ms + (budget_M / 1000.0) * engine_factor * instance_factor * ad_factor
        runtime_ms *= 0.95 + rng.random() * 0.10
        cost = compute_cost_per_run(runtime_ms, candidate.hourly_rate or 0.0)
        greeks = None
        if candidate.task == "ad_required":
            greeks = {"delta": 0.55, "vega": 0.02, "rho": 0.01}
        return Observation(
            candidate_id=candidate.candidate_id,
            task=candidate.task,
            engine=candidate.engine,
            ad_mode=candidate.ad_mode,
            instance_type=candidate.instance_type,
            budget_M=budget_M,
            stage=stage,
            valid=True,
            runtime_ms=runtime_ms,
            cost_per_run=cost,
            price=10.0 + rng.random() * 0.1,
            greeks=greeks,
            memory_peak_mb=64.0 + budget_M / 2000.0,
        )

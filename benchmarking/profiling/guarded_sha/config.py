from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmarking.core.config import EuropeanLocalVolConfig, WorkloadConfig, config_from_dict


@dataclass(frozen=True)
class CloudInstanceConfig:
    instance_type: str
    hourly_rate: Optional[float] = None
    cloud_provider: str = "gcp"
    region: str = "europe-west1"
    zone: Optional[str] = None

    @property
    def family(self) -> str:
        return self.instance_type.split("-", 1)[0]

    @property
    def broad_family(self) -> str:
        if self.family in {"c2", "c2d"}:
            return "compute_optimised"
        if self.family in {"e2", "n2", "n2d", "t2d"}:
            return "general_purpose"
        return self.family


@dataclass(frozen=True)
class ObjectiveWeights:
    name: str
    runtime_weight: float
    cost_weight: float


@dataclass(frozen=True)
class GuardConfig:
    near_tie_ratio: float = 0.15
    eta: float = 2.0
    engine_diversity_rounds: int = 2
    instance_diversity_rounds: int = 2
    scaling_guard_rounds: int = 99


@dataclass(frozen=True)
class CandidateConfig:
    workload_type: str
    task: str
    engine: str
    ad_mode: str
    instance_type: str
    hourly_rate: Optional[float]
    region: str = "europe-west1"
    zone: Optional[str] = None

    @property
    def candidate_id(self) -> str:
        return (
            f"{self.task}|{self.workload_type}|{self.engine}|"
            f"{self.ad_mode}|{self.instance_type}"
        )

    @property
    def engine_family(self) -> str:
        return self.engine

    @property
    def instance_family(self) -> str:
        return self.instance_type.split("-", 1)[0]

    @property
    def broad_instance_family(self) -> str:
        if self.instance_family in {"c2", "c2d"}:
            return "compute_optimised"
        if self.instance_family in {"e2", "n2", "n2d", "t2d"}:
            return "general_purpose"
        return self.instance_family


@dataclass
class Observation:
    candidate_id: str
    task: str
    engine: str
    ad_mode: str
    instance_type: str
    budget_M: int
    stage: str
    valid: bool
    runtime_ms: Optional[float] = None
    cost_per_run: Optional[float] = None
    score: Optional[float] = None
    price: Optional[float] = None
    greeks: Optional[Dict[str, float]] = None
    memory_peak_mb: Optional[float] = None
    correctness_passed: bool = True
    capacity_passed: bool = True
    failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["greeks"] = json.dumps(self.greeks or {}, sort_keys=True)
        return data

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Observation":
        raw_greeks = data.get("greeks")
        greeks: Optional[Dict[str, float]]
        if isinstance(raw_greeks, str) and raw_greeks.strip():
            greeks = json.loads(raw_greeks)
        elif isinstance(raw_greeks, dict):
            greeks = raw_greeks
        else:
            greeks = None

        def _float_or_none(value: Any) -> Optional[float]:
            if value in ("", None):
                return None
            return float(value)

        return Observation(
            candidate_id=str(data["candidate_id"]),
            task=str(data["task"]),
            engine=str(data["engine"]),
            ad_mode=str(data["ad_mode"]),
            instance_type=str(data["instance_type"]),
            budget_M=int(data["budget_M"]),
            stage=str(data.get("stage", "imported")),
            valid=str(data.get("valid", "true")).lower() in {"1", "true", "yes"},
            runtime_ms=_float_or_none(data.get("runtime_ms")),
            cost_per_run=_float_or_none(data.get("cost_per_run")),
            score=_float_or_none(data.get("score")),
            price=_float_or_none(data.get("price")),
            greeks=greeks,
            memory_peak_mb=_float_or_none(data.get("memory_peak_mb")),
            correctness_passed=str(data.get("correctness_passed", "true")).lower()
            in {"1", "true", "yes"},
            capacity_passed=str(data.get("capacity_passed", "true")).lower()
            in {"1", "true", "yes"},
            failure_reason=data.get("failure_reason") or None,
        )


@dataclass
class SHADecision:
    method: str
    objective: str
    task: str
    round_index: int
    budget_M: int
    candidate_id: str
    rank: Optional[int]
    score: Optional[float]
    promoted: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Recommendation:
    method: str
    objective: str
    task: str
    candidate_id: str
    engine: str
    ad_mode: str
    instance_type: str
    final_budget_M: int
    runtime_ms: Optional[float]
    cost_per_run: Optional[float]
    objective_score: Optional[float]
    correctness_passed: bool
    capacity_passed: bool
    full_runs_used: int
    full_runs_saved: int
    full_run_saving_pct: float
    objective_regret_pct: Optional[float] = None
    runtime_regret_pct: Optional[float] = None
    cost_regret_pct: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LocalVolProfilerConfig:
    workload: WorkloadConfig = field(default_factory=EuropeanLocalVolConfig)
    engines: List[str] = field(default_factory=lambda: ["cpu", "jax", "cpp", "rust"])
    ad_modes: List[str] = field(default_factory=lambda: ["none", "forward", "reverse"])
    instances: List[CloudInstanceConfig] = field(default_factory=list)
    probe_budgets: List[int] = field(default_factory=lambda: [1_000, 5_000, 25_000])
    full_budget: int = 100_000
    repeats: int = 3
    warmup: int = 1
    objectives: List[ObjectiveWeights] = field(
        default_factory=lambda: [
            ObjectiveWeights("speed_sensitive", 0.8, 0.2),
            ObjectiveWeights("balanced", 0.5, 0.5),
            ObjectiveWeights("cost_sensitive", 0.2, 0.8),
        ]
    )
    guards: GuardConfig = field(default_factory=GuardConfig)
    include_price_only: bool = True
    include_ad_required: bool = True

    def workload_at_budget(self, budget_M: int) -> WorkloadConfig:
        data = self.workload.to_dict()
        data["M"] = int(budget_M)
        return config_from_dict(data)

    @staticmethod
    def from_json(path: str | Path) -> "LocalVolProfilerConfig":
        data = json.loads(Path(path).read_text())
        workload_data = data.get("workload") or {"workload_type": "european_local_vol"}
        workload = config_from_dict(workload_data)
        instances = [
            CloudInstanceConfig(**item) for item in data.get("instances", [])
        ]
        objectives = [
            ObjectiveWeights(**item) for item in data.get("objectives", [])
        ] or LocalVolProfilerConfig().objectives
        guards = GuardConfig(**data.get("guards", {}))
        return LocalVolProfilerConfig(
            workload=workload,
            engines=list(data.get("engines", ["cpu", "jax", "cpp", "rust"])),
            ad_modes=list(data.get("ad_modes", ["none", "forward", "reverse"])),
            instances=instances,
            probe_budgets=list(data.get("probe_budgets", [1_000, 5_000, 25_000])),
            full_budget=int(data.get("full_budget", 100_000)),
            repeats=int(data.get("repeats", 3)),
            warmup=int(data.get("warmup", 1)),
            objectives=objectives,
            guards=guards,
            include_price_only=bool(data.get("include_price_only", True)),
            include_ad_required=bool(data.get("include_ad_required", True)),
        )

    def with_instances(self, instances: List[CloudInstanceConfig]) -> "LocalVolProfilerConfig":
        return replace(self, instances=instances)


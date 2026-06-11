from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from benchmarking.core.engine import MonteCarloEngine
from benchmarking.profiling.guarded_sha.config import (
    CandidateConfig,
    CloudInstanceConfig,
    LocalVolProfilerConfig,
)


def generate_candidates(
    config: LocalVolProfilerConfig,
    engines: Dict[str, MonteCarloEngine | None],
) -> Tuple[List[CandidateConfig], Dict[str, str]]:
    candidates: List[CandidateConfig] = []
    rejected: Dict[str, str] = {}
    workload_type = config.workload.workload_type

    for engine_name in config.engines:
        engine = engines.get(engine_name)
        if engine is None:
            rejected[f"{engine_name}|*|*"] = "engine unavailable"
            continue
        if not engine.supports(workload_type):
            rejected[f"{engine_name}|*|*"] = f"engine does not support {workload_type}"
            continue

        supported_modes = set(engine.supported_ad_modes())
        for ad_mode in config.ad_modes:
            if ad_mode not in supported_modes:
                rejected[f"{engine_name}|{ad_mode}|*"] = "unsupported AD mode"
                continue

            tasks: List[str] = []
            if ad_mode == "none" and config.include_price_only:
                tasks.append("price_only")
            if ad_mode != "none" and config.include_ad_required:
                tasks.append("ad_required")
            if not tasks:
                continue

            for instance in config.instances:
                for task in tasks:
                    candidates.append(
                        CandidateConfig(
                            workload_type=workload_type,
                            task=task,
                            engine=engine_name,
                            ad_mode=ad_mode,
                            instance_type=instance.instance_type,
                            hourly_rate=instance.hourly_rate,
                            region=instance.region,
                            zone=instance.zone,
                        )
                    )
    return candidates, rejected


def default_instance_configs(
    instance_names: Iterable[str],
    hourly_rates: Dict[str, float | None],
    region: str = "europe-west1",
) -> List[CloudInstanceConfig]:
    return [
        CloudInstanceConfig(
            instance_type=name,
            hourly_rate=hourly_rates.get(name),
            region=region,
        )
        for name in instance_names
    ]


from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

from benchmarking.profiling.guarded_sha.config import Observation, ObjectiveWeights


def valid_observations(observations: Iterable[Observation]) -> List[Observation]:
    return [
        obs for obs in observations
        if obs.valid
        and obs.correctness_passed
        and obs.capacity_passed
        and obs.runtime_ms is not None
        and obs.cost_per_run is not None
        and obs.runtime_ms > 0
        and obs.cost_per_run >= 0
        and math.isfinite(obs.runtime_ms)
        and math.isfinite(obs.cost_per_run)
    ]


def score_observations(
    observations: Iterable[Observation],
    weights: ObjectiveWeights,
) -> Dict[str, float]:
    valid = valid_observations(observations)
    if not valid:
        return {}

    fastest = min(obs.runtime_ms for obs in valid if obs.runtime_ms is not None)
    cheapest = min(obs.cost_per_run for obs in valid if obs.cost_per_run is not None)
    cheapest = max(cheapest, 1e-15)
    fastest = max(fastest, 1e-15)

    scores: Dict[str, float] = {}
    for obs in valid:
        runtime_norm = (obs.runtime_ms or 0.0) / fastest
        cost_norm = (obs.cost_per_run or 0.0) / cheapest
        scores[obs.candidate_id] = (
            weights.runtime_weight * runtime_norm
            + weights.cost_weight * cost_norm
        )
    return scores


def rank_observations(
    observations: Iterable[Observation],
    weights: ObjectiveWeights,
) -> List[tuple[Observation, float]]:
    scores = score_observations(observations, weights)
    by_id = {obs.candidate_id: obs for obs in observations}
    ranked = [(by_id[candidate_id], score) for candidate_id, score in scores.items()]
    return sorted(ranked, key=lambda item: (item[1], item[0].runtime_ms or float("inf")))


def best_observation(
    observations: Iterable[Observation],
    weights: ObjectiveWeights,
) -> Optional[tuple[Observation, float]]:
    ranked = rank_observations(observations, weights)
    return ranked[0] if ranked else None


def regret_pct(selected: Optional[float], oracle: Optional[float]) -> Optional[float]:
    if selected is None or oracle is None or oracle <= 0:
        return None
    return max(0.0, (selected - oracle) / oracle * 100.0)


def metric_regrets(
    selected: Observation,
    oracle: Observation,
    selected_score: Optional[float],
    oracle_score: Optional[float],
) -> Dict[str, Optional[float]]:
    return {
        "objective_regret_pct": regret_pct(selected_score, oracle_score),
        "runtime_regret_pct": regret_pct(selected.runtime_ms, oracle.runtime_ms),
        "cost_regret_pct": regret_pct(selected.cost_per_run, oracle.cost_per_run),
    }


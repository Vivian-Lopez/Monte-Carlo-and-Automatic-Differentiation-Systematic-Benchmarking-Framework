from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional

from benchmarking.profiling.guarded_sha.config import (
    CandidateConfig,
    Observation,
    ObjectiveWeights,
    Recommendation,
)
from benchmarking.profiling.guarded_sha.objective import (
    best_observation,
    metric_regrets,
    score_observations,
)


def observations_by_key(observations: Iterable[Observation]) -> Dict[tuple[str, int], Observation]:
    return {(obs.candidate_id, obs.budget_M): obs for obs in observations}


def group_by_task(observations: Iterable[Observation]) -> Dict[str, List[Observation]]:
    grouped: Dict[str, List[Observation]] = defaultdict(list)
    for obs in observations:
        grouped[obs.task].append(obs)
    return dict(grouped)


def make_recommendations(
    method: str,
    objectives: List[ObjectiveWeights],
    final_observations: List[Observation],
    candidates: List[CandidateConfig],
    full_budget: int,
    full_grid_candidate_count: int,
    full_runs_used: int,
    oracle_by_task_objective: Optional[Dict[tuple[str, str], tuple[Observation, float]]] = None,
    score_reference_observations: Optional[List[Observation]] = None,
) -> List[Recommendation]:
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    recs: List[Recommendation] = []
    grouped = group_by_task([obs for obs in final_observations if obs.budget_M == full_budget])

    for task, task_observations in grouped.items():
        for objective in objectives:
            best = best_observation(task_observations, objective)
            if best is None:
                continue
            selected_obs, selected_score = best
            if score_reference_observations is not None:
                reference_rows = [
                    obs for obs in score_reference_observations
                    if obs.task == task and obs.budget_M == full_budget
                ]
                reference_scores = score_observations(reference_rows, objective)
                selected_score = reference_scores.get(selected_obs.candidate_id, selected_score)
            candidate = candidate_by_id[selected_obs.candidate_id]
            oracle = None
            regrets = {
                "objective_regret_pct": None,
                "runtime_regret_pct": None,
                "cost_regret_pct": None,
            }
            if oracle_by_task_objective:
                oracle = oracle_by_task_objective.get((task, objective.name))
            if oracle:
                oracle_obs, oracle_score = oracle
                regrets = metric_regrets(selected_obs, oracle_obs, selected_score, oracle_score)
            full_runs_saved = max(0, full_grid_candidate_count - full_runs_used)
            saving_pct = (
                full_runs_saved / full_grid_candidate_count * 100.0
                if full_grid_candidate_count else 0.0
            )
            recs.append(
                Recommendation(
                    method=method,
                    objective=objective.name,
                    task=task,
                    candidate_id=selected_obs.candidate_id,
                    engine=candidate.engine,
                    ad_mode=candidate.ad_mode,
                    instance_type=candidate.instance_type,
                    final_budget_M=full_budget,
                    runtime_ms=selected_obs.runtime_ms,
                    cost_per_run=selected_obs.cost_per_run,
                    objective_score=selected_score,
                    correctness_passed=selected_obs.correctness_passed,
                    capacity_passed=selected_obs.capacity_passed,
                    full_runs_used=full_runs_used,
                    full_runs_saved=full_runs_saved,
                    full_run_saving_pct=saving_pct,
                    **regrets,
                )
            )
    return recs


def oracle_map(
    objectives: List[ObjectiveWeights],
    full_observations: List[Observation],
    full_budget: int,
) -> Dict[tuple[str, str], tuple[Observation, float]]:
    result: Dict[tuple[str, str], tuple[Observation, float]] = {}
    grouped = group_by_task([obs for obs in full_observations if obs.budget_M == full_budget])
    for task, observations in grouped.items():
        for objective in objectives:
            best = best_observation(observations, objective)
            if best is not None:
                result[(task, objective.name)] = best
    return result


def attach_scores(
    observations: List[Observation],
    objectives: List[ObjectiveWeights],
) -> None:
    grouped = defaultdict(list)
    for obs in observations:
        grouped[(obs.task, obs.budget_M, obs.stage)].append(obs)
    for _key, rows in grouped.items():
        for objective in objectives:
            if objective.name == "balanced":
                scores = score_observations(rows, objective)
                for obs in rows:
                    if obs.candidate_id in scores:
                        obs.score = scores[obs.candidate_id]

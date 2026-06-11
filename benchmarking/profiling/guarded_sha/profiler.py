from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set

from benchmarking.profiling.guarded_sha.analysis import make_recommendations, oracle_map
from benchmarking.profiling.guarded_sha.config import (
    CandidateConfig,
    LocalVolProfilerConfig,
    Observation,
    Recommendation,
    SHADecision,
)
from benchmarking.profiling.guarded_sha.guards import apply_guards, base_cutoff_keep
from benchmarking.profiling.guarded_sha.objective import rank_observations
from benchmarking.profiling.guarded_sha.runner import CandidateRunner


class GuardedSHAProfiler:
    def __init__(
        self,
        config: LocalVolProfilerConfig,
        candidates: List[CandidateConfig],
        runner: Optional[CandidateRunner] = None,
    ) -> None:
        self.config = config
        self.candidates = candidates
        self.runner = runner
        self.decisions: List[SHADecision] = []
        self.observations: List[Observation] = []

    def run_observation_pool(self) -> List[Observation]:
        if self.runner is None:
            raise ValueError("runner is required to collect observations")
        budgets = list(dict.fromkeys([*self.config.probe_budgets, self.config.full_budget]))
        rows: List[Observation] = []
        for budget in budgets:
            workload = self.config.workload_at_budget(budget)
            stage = "full_grid" if budget == self.config.full_budget else "probe"
            for candidate in self.candidates:
                rows.append(self.runner.run_candidate(candidate, workload, budget, stage))
        self.observations.extend(rows)
        return rows

    def simulate_plain_sha(
        self,
        observations: List[Observation],
        oracle_recs: Optional[Dict[tuple[str, str], tuple[Observation, float]]] = None,
    ) -> tuple[List[Recommendation], List[SHADecision]]:
        return self._simulate_sha(observations, guarded=False, oracle_recs=oracle_recs)

    def simulate_guarded_sha(
        self,
        observations: List[Observation],
        oracle_recs: Optional[Dict[tuple[str, str], tuple[Observation, float]]] = None,
    ) -> tuple[List[Recommendation], List[SHADecision]]:
        return self._simulate_sha(observations, guarded=True, oracle_recs=oracle_recs)

    def oracle_recommendations(self, observations: List[Observation]) -> List[Recommendation]:
        full_rows = [obs for obs in observations if obs.budget_M == self.config.full_budget]
        oracle = oracle_map(self.config.objectives, full_rows, self.config.full_budget)
        return make_recommendations(
            method="full_grid_oracle",
            objectives=self.config.objectives,
            final_observations=full_rows,
            candidates=self.candidates,
            full_budget=self.config.full_budget,
            full_grid_candidate_count=len(self.candidates),
            full_runs_used=len(self.candidates),
            oracle_by_task_objective=oracle,
            score_reference_observations=full_rows,
        )

    def _simulate_sha(
        self,
        observations: List[Observation],
        guarded: bool,
        oracle_recs: Optional[Dict[tuple[str, str], tuple[Observation, float]]] = None,
    ) -> tuple[List[Recommendation], List[SHADecision]]:
        method = "guarded_sha" if guarded else "plain_sha"
        obs_by_candidate_budget = {
            (obs.candidate_id, obs.budget_M): obs for obs in observations
        }
        candidates_by_task: Dict[str, List[CandidateConfig]] = defaultdict(list)
        for candidate in self.candidates:
            candidates_by_task[candidate.task].append(candidate)

        active_by_task: Dict[str, Set[str]] = {
            task: {candidate.candidate_id for candidate in candidates}
            for task, candidates in candidates_by_task.items()
        }
        previous_ranks_by_objective: Dict[tuple[str, str], Dict[str, int]] = {}
        decisions: List[SHADecision] = []

        for round_index, budget in enumerate(self.config.probe_budgets):
            for task, task_candidates in candidates_by_task.items():
                active_ids = active_by_task.get(task, set())
                active_candidates = [
                    candidate for candidate in task_candidates
                    if candidate.candidate_id in active_ids
                ]
                if not active_candidates:
                    continue

                promoted_union: Set[str] = set()
                reason_map: Dict[str, List[str]] = defaultdict(list)
                ranked_for_any: Dict[str, tuple[int, float]] = {}

                for objective in self.config.objectives:
                    rows = [
                        obs_by_candidate_budget[(candidate.candidate_id, budget)]
                        for candidate in active_candidates
                        if (candidate.candidate_id, budget) in obs_by_candidate_budget
                    ]
                    ranked = rank_observations(rows, objective)
                    ranked_scores = [
                        (obs.candidate_id, score) for obs, score in ranked
                    ]
                    current_ranks = {
                        candidate_id: rank
                        for rank, (candidate_id, _score) in enumerate(ranked_scores, start=1)
                    }
                    previous_ranks = previous_ranks_by_objective.get((task, objective.name), {})
                    if guarded:
                        guard_result = apply_guards(
                            active_candidates,
                            ranked_scores,
                            round_index,
                            self.config.guards,
                            previous_ranks=previous_ranks,
                            current_ranks=current_ranks,
                        )
                        objective_kept = guard_result.kept_ids
                        for candidate_id, reasons in guard_result.reasons.items():
                            reason_map[candidate_id].extend(reasons)
                    else:
                        ranked_ids = [candidate_id for candidate_id, _score in ranked_scores]
                        objective_kept = base_cutoff_keep(ranked_ids, self.config.guards.eta)
                        for candidate_id in objective_kept:
                            reason_map[candidate_id].append("base_cutoff")
                    promoted_union.update(objective_kept)
                    previous_ranks_by_objective[(task, objective.name)] = current_ranks
                    for rank, (candidate_id, score) in enumerate(ranked_scores, start=1):
                        ranked_for_any[candidate_id] = (rank, score)

                for candidate in active_candidates:
                    rank_score = ranked_for_any.get(candidate.candidate_id)
                    promoted = candidate.candidate_id in promoted_union
                    reason = ";".join(sorted(set(reason_map.get(candidate.candidate_id, []))))
                    if not promoted:
                        reason = "eliminated_by_rank"
                    decisions.append(
                        SHADecision(
                            method=method,
                            objective="union",
                            task=task,
                            round_index=round_index,
                            budget_M=budget,
                            candidate_id=candidate.candidate_id,
                            rank=rank_score[0] if rank_score else None,
                            score=rank_score[1] if rank_score else None,
                            promoted=promoted,
                            reason=reason,
                        )
                    )
                active_by_task[task] = promoted_union

        final_candidate_ids = set().union(*active_by_task.values()) if active_by_task else set()
        final_rows = [
            obs for obs in observations
            if obs.budget_M == self.config.full_budget
            and obs.candidate_id in final_candidate_ids
        ]
        full_runs_used = len({obs.candidate_id for obs in final_rows})
        recs = make_recommendations(
            method=method,
            objectives=self.config.objectives,
            final_observations=final_rows,
            candidates=self.candidates,
            full_budget=self.config.full_budget,
            full_grid_candidate_count=len(self.candidates),
            full_runs_used=full_runs_used,
            oracle_by_task_objective=oracle_recs,
            score_reference_observations=[
                obs for obs in observations if obs.budget_M == self.config.full_budget
            ],
        )
        return recs, decisions

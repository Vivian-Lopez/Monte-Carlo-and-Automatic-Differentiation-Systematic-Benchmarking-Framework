from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Set

from benchmarking.profiling.guarded_sha.config import CandidateConfig, GuardConfig


@dataclass
class GuardResult:
    kept_ids: Set[str]
    reasons: Dict[str, List[str]]


def _add(reason_map: Dict[str, List[str]], candidate_id: str, reason: str) -> None:
    reason_map.setdefault(candidate_id, []).append(reason)


def base_cutoff_keep(
    ranked_ids: List[str],
    eta: float,
) -> Set[str]:
    if not ranked_ids:
        return set()
    keep_n = max(1, math.ceil(len(ranked_ids) / eta))
    return set(ranked_ids[:keep_n])


def apply_guards(
    candidates: Iterable[CandidateConfig],
    ranked_scores: List[tuple[str, float]],
    round_index: int,
    guard_config: GuardConfig,
    previous_ranks: Mapping[str, int] | None = None,
    current_ranks: Mapping[str, int] | None = None,
) -> GuardResult:
    candidates_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    ranked_ids = [candidate_id for candidate_id, _score in ranked_scores]
    kept = base_cutoff_keep(ranked_ids, guard_config.eta)
    reasons: Dict[str, List[str]] = {}

    for candidate_id in kept:
        _add(reasons, candidate_id, "base_cutoff")

    cutoff_score = None
    if ranked_ids:
        cutoff_index = min(len(ranked_ids), max(1, math.ceil(len(ranked_ids) / guard_config.eta))) - 1
        cutoff_score = ranked_scores[cutoff_index][1]

    if cutoff_score is not None:
        threshold = cutoff_score * (1.0 + guard_config.near_tie_ratio)
        for candidate_id, score in ranked_scores:
            if score <= threshold:
                kept.add(candidate_id)
                _add(reasons, candidate_id, f"near_tie_within_{guard_config.near_tie_ratio:.0%}")

    if round_index < guard_config.engine_diversity_rounds:
        best_by_engine: Dict[str, tuple[str, float]] = {}
        for candidate_id, score in ranked_scores:
            engine = candidates_by_id[candidate_id].engine_family
            if engine not in best_by_engine or score < best_by_engine[engine][1]:
                best_by_engine[engine] = (candidate_id, score)
        for candidate_id, _score in best_by_engine.values():
            kept.add(candidate_id)
            _add(reasons, candidate_id, "engine_diversity")

    if round_index < guard_config.instance_diversity_rounds:
        best_by_family: Dict[str, tuple[str, float]] = {}
        for candidate_id, score in ranked_scores:
            family = candidates_by_id[candidate_id].broad_instance_family
            if family not in best_by_family or score < best_by_family[family][1]:
                best_by_family[family] = (candidate_id, score)
        for candidate_id, _score in best_by_family.values():
            kept.add(candidate_id)
            _add(reasons, candidate_id, "instance_diversity")

    if previous_ranks and current_ranks and round_index < guard_config.scaling_guard_rounds:
        for candidate_id, current_rank in current_ranks.items():
            previous_rank = previous_ranks.get(candidate_id)
            if previous_rank is not None and current_rank < previous_rank:
                kept.add(candidate_id)
                _add(reasons, candidate_id, "scaling_improved")

    return GuardResult(kept_ids=kept, reasons=reasons)


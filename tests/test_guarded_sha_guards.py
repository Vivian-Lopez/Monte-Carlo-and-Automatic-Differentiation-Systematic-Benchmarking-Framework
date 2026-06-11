from benchmarking.profiling.guarded_sha.config import CandidateConfig, GuardConfig
from benchmarking.profiling.guarded_sha.guards import apply_guards


def _candidate(engine, instance):
    return CandidateConfig(
        workload_type="european_local_vol",
        task="price_only",
        engine=engine,
        ad_mode="none",
        instance_type=instance,
        hourly_rate=0.1,
    )


def test_near_tie_guard_keeps_candidate_close_to_cutoff():
    candidates = [_candidate("cpu", "e2-standard-4"), _candidate("rust", "n2-standard-4"), _candidate("jax", "c2d-standard-4")]
    ranked = [
        (candidates[0].candidate_id, 1.0),
        (candidates[1].candidate_id, 1.10),
        (candidates[2].candidate_id, 2.0),
    ]
    result = apply_guards(
        candidates,
        ranked,
        round_index=3,
        guard_config=GuardConfig(near_tie_ratio=0.15, eta=3.0, engine_diversity_rounds=0, instance_diversity_rounds=0),
    )
    assert candidates[1].candidate_id in result.kept_ids
    assert any("near_tie" in reason for reason in result.reasons[candidates[1].candidate_id])


def test_engine_diversity_guard_keeps_one_per_engine():
    candidates = [_candidate("cpu", "e2-standard-4"), _candidate("rust", "n2-standard-4"), _candidate("jax", "c2d-standard-4")]
    ranked = [
        (candidates[0].candidate_id, 1.0),
        (candidates[1].candidate_id, 3.0),
        (candidates[2].candidate_id, 4.0),
    ]
    result = apply_guards(
        candidates,
        ranked,
        round_index=0,
        guard_config=GuardConfig(eta=10.0, engine_diversity_rounds=1, instance_diversity_rounds=0),
    )
    assert result.kept_ids == {candidate.candidate_id for candidate in candidates}


def test_instance_diversity_guard_keeps_compute_family():
    candidates = [_candidate("cpu", "e2-standard-4"), _candidate("rust", "n2-standard-4"), _candidate("jax", "c2d-standard-4")]
    ranked = [
        (candidates[0].candidate_id, 1.0),
        (candidates[1].candidate_id, 1.1),
        (candidates[2].candidate_id, 4.0),
    ]
    result = apply_guards(
        candidates,
        ranked,
        round_index=0,
        guard_config=GuardConfig(eta=10.0, engine_diversity_rounds=0, instance_diversity_rounds=1),
    )
    assert candidates[2].candidate_id in result.kept_ids
    assert "instance_diversity" in result.reasons[candidates[2].candidate_id]


def test_scaling_guard_keeps_improving_rank():
    candidates = [_candidate("cpu", "e2-standard-4"), _candidate("rust", "n2-standard-4"), _candidate("jax", "c2d-standard-4")]
    ranked = [
        (candidates[0].candidate_id, 1.0),
        (candidates[1].candidate_id, 1.1),
        (candidates[2].candidate_id, 1.2),
    ]
    result = apply_guards(
        candidates,
        ranked,
        round_index=2,
        guard_config=GuardConfig(eta=10.0, engine_diversity_rounds=0, instance_diversity_rounds=0),
        previous_ranks={candidates[2].candidate_id: 5},
        current_ranks={candidates[2].candidate_id: 3},
    )
    assert candidates[2].candidate_id in result.kept_ids
    assert "scaling_improved" in result.reasons[candidates[2].candidate_id]


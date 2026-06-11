from benchmarking.profiling.guarded_sha.analysis import make_recommendations, oracle_map
from benchmarking.profiling.guarded_sha.config import (
    CandidateConfig,
    Observation,
    ObjectiveWeights,
)
from benchmarking.profiling.guarded_sha.objective import rank_observations, score_observations


def _obs(candidate_id, runtime, cost, valid=True):
    task, _workload, engine, ad_mode, instance = candidate_id.split("|")
    return Observation(
        candidate_id=candidate_id,
        task=task,
        engine=engine,
        ad_mode=ad_mode,
        instance_type=instance,
        budget_M=100_000,
        stage="full_grid",
        valid=valid,
        runtime_ms=runtime,
        cost_per_run=cost,
    )


def _candidate(candidate_id):
    task, workload, engine, ad_mode, instance = candidate_id.split("|")
    return CandidateConfig(
        workload_type=workload,
        task=task,
        engine=engine,
        ad_mode=ad_mode,
        instance_type=instance,
        hourly_rate=0.1,
    )


def test_weighted_objective_normalises_runtime_and_cost():
    weights = ObjectiveWeights("balanced", 0.5, 0.5)
    rows = [
        _obs("price_only|european_local_vol|cpu|none|e2-standard-4", 100.0, 0.04),
        _obs("price_only|european_local_vol|rust|none|n2-standard-4", 50.0, 0.08),
    ]
    scores = score_observations(rows, weights)
    assert scores[rows[0].candidate_id] == 1.5
    assert scores[rows[1].candidate_id] == 1.5


def test_invalid_candidates_are_not_scored():
    weights = ObjectiveWeights("balanced", 0.5, 0.5)
    rows = [
        _obs("price_only|european_local_vol|cpu|none|e2-standard-4", 100.0, 0.04),
        _obs("price_only|european_local_vol|rust|none|n2-standard-4", 50.0, 0.08, valid=False),
    ]
    ranked = rank_observations(rows, weights)
    assert len(ranked) == 1
    assert ranked[0][0].engine == "cpu"


def test_regret_calculation_against_oracle():
    weights = [ObjectiveWeights("balanced", 0.5, 0.5)]
    ids = [
        "price_only|european_local_vol|cpu|none|e2-standard-4",
        "price_only|european_local_vol|rust|none|n2-standard-4",
    ]
    rows = [_obs(ids[0], 100.0, 0.04), _obs(ids[1], 50.0, 0.08)]
    oracle = oracle_map(weights, rows, 100_000)
    recs = make_recommendations(
        method="plain_sha",
        objectives=weights,
        final_observations=[rows[0]],
        candidates=[_candidate(candidate_id) for candidate_id in ids],
        full_budget=100_000,
        full_grid_candidate_count=2,
        full_runs_used=1,
        oracle_by_task_objective=oracle,
    )
    assert recs[0].full_runs_saved == 1
    assert recs[0].objective_regret_pct == 0.0


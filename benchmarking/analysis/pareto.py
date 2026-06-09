"""
Pareto frontier utilities for benchmark result analysis.

compute_pareto_frontier(rows, x_key, y_key)
-------------------------------------------
Filters a list of result dicts to the non-dominated subset on two objectives
(minimise both x and y).  Rows with missing, NaN, infinite, or negative
values in either key are silently excluded.

A row A dominates row B if:
    A[x_key] <= B[x_key]  AND  A[y_key] <= B[y_key]
    AND at least one inequality is strict.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List


def _valid(val: Any) -> bool:
    try:
        f = float(val)
        return math.isfinite(f) and f >= 0.0
    except (TypeError, ValueError):
        return False


def dominates(
    a: Dict[str, Any],
    b: Dict[str, Any],
    x_key: str = "mean_runtime_ms",
    y_key: str = "cost_per_run",
) -> bool:
    """Return True if row *a* dominates row *b* on both objectives."""
    ax, ay = float(a[x_key]), float(a[y_key])
    bx, by = float(b[x_key]), float(b[y_key])
    return (ax <= bx and ay <= by) and (ax < bx or ay < by)


def compute_pareto_frontier(
    rows: List[Dict[str, Any]],
    x_key: str = "mean_runtime_ms",
    y_key: str = "cost_per_run",
) -> List[Dict[str, Any]]:
    """
    Return the non-dominated subset of *rows* sorted by x_key then y_key.

    Rows with missing/invalid x or y values are excluded before the
    dominance test.
    """
    valid = [r for r in rows if _valid(r.get(x_key)) and _valid(r.get(y_key))]
    frontier = []
    for candidate in valid:
        dominated_by_any = any(
            dominates(other, candidate, x_key, y_key)
            for other in valid
            if other is not candidate
        )
        if not dominated_by_any:
            frontier.append(candidate)
    frontier.sort(key=lambda r: (float(r[x_key]), float(r[y_key])))
    return frontier


def ci_overlap_select(
    rows: List[Dict[str, Any]],
    n_runs: int,
    key: str = "mean_runtime_ms",
    std_key: str = "std_runtime_ms",
    z: float = 1.96,
) -> List[Dict[str, Any]]:
    """
    Select configs whose 95% CI overlaps with the best config's CI.

    A config *survives* if its CI lower bound is less than the best
    config's CI upper bound — i.e., it could plausibly be as fast as the
    best config given measurement noise.

    This replaces a hard score-margin constant (e.g. 1.5×) with a
    statistically principled criterion that adapts to measurement noise:
    more runs → tighter CIs → more aggressive pruning.

    Parameters
    ----------
    rows    : list of result dicts with `key` and `std_key` fields
    n_runs  : number of timed repetitions used to compute mean/std
    key     : column for the mean runtime (lower = better)
    std_key : column for the std runtime
    z       : z-score for the CI (default 1.96 → 95%)

    Returns the subset of rows that survive the overlap test, sorted by key.
    Rows with missing/invalid key or std_key are excluded.
    """
    import math as _math

    def _half_width(row: Dict[str, Any]) -> float:
        try:
            s = float(row.get(std_key) or 0.0)
            return z * s / _math.sqrt(max(n_runs, 1))
        except (TypeError, ValueError):
            return 0.0

    valid = [
        r for r in rows
        if _valid(r.get(key)) and r.get(std_key) is not None
    ]
    if not valid:
        return list(rows)  # fallback: keep all

    best = min(valid, key=lambda r: float(r[key]))
    best_upper = float(best[key]) + _half_width(best)

    survivors = [
        r for r in valid
        if (float(r[key]) - _half_width(r)) <= best_upper
    ]
    survivors.sort(key=lambda r: float(r[key]))
    return survivors

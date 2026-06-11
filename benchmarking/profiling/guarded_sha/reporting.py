from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List

from benchmarking.profiling.guarded_sha.config import Observation, Recommendation, SHADecision


def _write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    output_dir: Path,
    observations: Iterable[Observation],
    decisions: Iterable[SHADecision],
    recommendations: Iterable[Recommendation],
    rejected_candidates: dict[str, str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    observation_rows = [obs.to_dict() for obs in observations]
    decision_rows = [decision.to_dict() for decision in decisions]
    recommendation_rows = [rec.to_dict() for rec in recommendations]

    _write_csv(output_dir / "candidate_observations.csv", observation_rows)
    _write_csv(output_dir / "sha_rounds.csv", decision_rows)
    _write_csv(output_dir / "recommendations.csv", recommendation_rows)
    _write_csv(output_dir / "oracle_comparison.csv", recommendation_rows)

    (output_dir / "recommendations.json").write_text(
        json.dumps(recommendation_rows, indent=2, sort_keys=True)
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "num_observations": len(observation_rows),
                "num_decisions": len(decision_rows),
                "num_recommendations": len(recommendation_rows),
                "rejected_candidates": rejected_candidates,
            },
            indent=2,
            sort_keys=True,
        )
    )
    _write_markdown(output_dir / "guarded_sha_report.md", recommendation_rows, rejected_candidates)


def read_observations(paths: Iterable[Path]) -> List[Observation]:
    rows: List[Observation] = []
    for path in paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows.extend(Observation.from_dict(row) for row in reader)
    return rows


def _write_markdown(path: Path, recommendations: List[dict], rejected_candidates: dict[str, str]) -> None:
    lines = [
        "# Guarded SHA Local-Volatility Profiler",
        "",
        "## Experiment Setup",
        "",
        "The profiler optimises local-volatility Monte Carlo deployment choices over engine, AD mode, and cloud instance type.",
        "Lower objective scores are better and combine normalised runtime with normalised cost per run.",
        "",
        "## Recommendations",
        "",
    ]
    if recommendations:
        lines.append("| Method | Task | Objective | Engine | AD | Instance | Runtime ms | Cost/run | Score | Full runs saved | Regret % |")
        lines.append("|---|---|---|---|---|---|---:|---:|---:|---:|---:|")
        for rec in recommendations:
            lines.append(
                "| {method} | {task} | {objective} | {engine} | {ad_mode} | "
                "{instance_type} | {runtime_ms:.3f} | {cost_per_run:.6g} | "
                "{objective_score:.4f} | {full_runs_saved} | {objective_regret_pct} |".format(
                    **{
                        **rec,
                        "runtime_ms": rec.get("runtime_ms") or 0.0,
                        "cost_per_run": rec.get("cost_per_run") or 0.0,
                        "objective_score": rec.get("objective_score") or 0.0,
                        "objective_regret_pct": (
                            "n/a"
                            if rec.get("objective_regret_pct") is None
                            else f"{rec.get('objective_regret_pct'):.2f}"
                        ),
                    }
                )
            )
    else:
        lines.append("No valid recommendations were produced.")

    lines.extend([
        "",
        "## Guard Rules",
        "",
        "- Near-tie guard retains candidates close to the cutoff score.",
        "- Engine-diversity guard keeps at least one surviving candidate per engine in early rounds.",
        "- Instance-diversity guard keeps broad hardware families represented in early rounds.",
        "- Scaling guard retains candidates whose rank improves between probe budgets.",
        "- Correctness/AD guard excludes failed or unsupported configurations.",
        "",
        "## Rejected Candidate Reasons",
        "",
    ])
    if rejected_candidates:
        for key, reason in sorted(rejected_candidates.items()):
            lines.append(f"- `{key}`: {reason}")
    else:
        lines.append("- None.")
    path.write_text("\n".join(lines) + "\n")


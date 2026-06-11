# Guarded SHA Local-Volatility Profiler

## Experiment Setup

The profiler optimises local-volatility Monte Carlo deployment choices over engine, AD mode, and cloud instance type.
Lower objective scores are better and combine normalised runtime with normalised cost per run.

## Recommendations

| Method | Task | Objective | Engine | AD | Instance | Runtime ms | Cost/run | Score | Full runs saved | Regret % |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| full_grid_oracle | price_only | speed_sensitive | rust | none | t2d-standard-4 | 445.098 | 2.10284e-05 | 1.0476 | 0 | 0.00 |
| full_grid_oracle | price_only | balanced | rust | none | t2d-standard-4 | 445.098 | 2.10284e-05 | 1.0298 | 0 | 0.00 |
| full_grid_oracle | price_only | cost_sensitive | rust | none | t2d-standard-4 | 445.098 | 2.10284e-05 | 1.0119 | 0 | 0.00 |
| full_grid_oracle | ad_required | speed_sensitive | jax | reverse | t2d-standard-4 | 1183.007 | 5.58905e-05 | 1.0395 | 0 | 0.00 |
| full_grid_oracle | ad_required | balanced | jax | reverse | t2d-standard-4 | 1183.007 | 5.58905e-05 | 1.0247 | 0 | 0.00 |
| full_grid_oracle | ad_required | cost_sensitive | jax | reverse | t2d-standard-4 | 1183.007 | 5.58905e-05 | 1.0099 | 0 | 0.00 |
| plain_sha | ad_required | speed_sensitive | jax | reverse | t2d-standard-4 | 1183.007 | 5.58905e-05 | 1.0395 | 23 | 0.00 |
| plain_sha | ad_required | balanced | jax | reverse | t2d-standard-4 | 1183.007 | 5.58905e-05 | 1.0247 | 23 | 0.00 |
| plain_sha | ad_required | cost_sensitive | jax | reverse | t2d-standard-4 | 1183.007 | 5.58905e-05 | 1.0099 | 23 | 0.00 |
| plain_sha | price_only | speed_sensitive | rust | none | t2d-standard-4 | 445.098 | 2.10284e-05 | 1.0476 | 23 | 0.00 |
| plain_sha | price_only | balanced | rust | none | t2d-standard-4 | 445.098 | 2.10284e-05 | 1.0298 | 23 | 0.00 |
| plain_sha | price_only | cost_sensitive | rust | none | t2d-standard-4 | 445.098 | 2.10284e-05 | 1.0119 | 23 | 0.00 |
| guarded_sha | price_only | speed_sensitive | rust | none | t2d-standard-4 | 445.098 | 2.10284e-05 | 1.0476 | 13 | 0.00 |
| guarded_sha | price_only | balanced | rust | none | t2d-standard-4 | 445.098 | 2.10284e-05 | 1.0298 | 13 | 0.00 |
| guarded_sha | price_only | cost_sensitive | rust | none | t2d-standard-4 | 445.098 | 2.10284e-05 | 1.0119 | 13 | 0.00 |
| guarded_sha | ad_required | speed_sensitive | jax | reverse | t2d-standard-4 | 1183.007 | 5.58905e-05 | 1.0395 | 13 | 0.00 |
| guarded_sha | ad_required | balanced | jax | reverse | t2d-standard-4 | 1183.007 | 5.58905e-05 | 1.0247 | 13 | 0.00 |
| guarded_sha | ad_required | cost_sensitive | jax | reverse | t2d-standard-4 | 1183.007 | 5.58905e-05 | 1.0099 | 13 | 0.00 |

## Guard Rules

- Near-tie guard retains candidates close to the cutoff score.
- Engine-diversity guard keeps at least one surviving candidate per engine in early rounds.
- Instance-diversity guard keeps broad hardware families represented in early rounds.
- Scaling guard retains candidates whose rank improves between probe budgets.
- Correctness/AD guard excludes failed or unsupported configurations.

## Rejected Candidate Reasons

- `cpp|forward|*`: unsupported AD mode
- `cpp|reverse|*`: unsupported AD mode
- `cpu|forward|*`: unsupported AD mode
- `cpu|reverse|*`: unsupported AD mode
- `rust|forward|*`: unsupported AD mode
- `rust|reverse|*`: unsupported AD mode

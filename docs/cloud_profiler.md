# Cloud Profiler Experiment — Setup and Reproduction Guide

**Experiment:** Budget-Aware Profiler vs Naive Grid Search  
**Context:** Final-year project, Imperial College London  
**Goal:** Show that cheap probe runs can prune the hardware/software search space before expensive full benchmarking, and evaluate the quality of that pruning.

---

## What the experiment does

1. **Candidate grid** — all combinations of `(workload, engine, ad_mode)` across the configured machine types.
2. **Probe phase** — each candidate is run at a small `M` (e.g. 1 000 paths) to estimate relative performance cheaply.  A JIT-warmup correction is applied to JAX probes so the one-off compile cost does not unfairly discard JAX.
3. **Selection** — candidates are scored and filtered:
   - Pareto-non-dominated on `(mean_runtime_ms, cost_per_run)` (synthetic cost proxy used locally)
   - Top-K from the non-dominated set
   - Safety margin: configs within `score_margin × best_score` are kept even outside top-K
   - Diversity guard: at least one config per available engine
   - Safety pin: the globally fastest probe runtime is always included
4. **Full-grid phase** — naive sweep over all candidates × all M values (ground truth).
5. **Evaluation** — compares profiler selection against the full-grid Pareto frontier:
   - Runs saved / percentage reduction
   - Pareto recovery %
   - Runtime regret vs full-grid best
   - Cost-performance regret
   - Spearman ρ between probe ranking and full-run ranking
   - Selected configurations with reasons
   - Missed Pareto configurations

All runs are stored in the SQLite database with phase labels, profiler decisions, reasons, cloud metadata, and git commit hash.

---

## How it fits the existing architecture

```
WorkloadConfig  →  BenchmarkRunner  →  BenchmarkResult
                        ↓
                  BenchmarkDB (SQLite)
                        ↓
          run_profiler_vs_grid.py  ←  cloud metadata via CLI args
                        ↓
          results/profiler_summary.txt
          results/profiler_vs_grid.csv
          results/cloud_cost_analysis.csv
          results/pareto_frontier.csv
          results/report_tables/
```

The profiler calls the existing `BenchmarkRunner` and `BenchmarkDB` interfaces directly.  No new storage system or separate benchmark path is introduced.

---

## Prerequisites

### Local
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### GCP
- `gcloud` CLI installed: https://cloud.google.com/sdk/docs/install
- Authenticated: `gcloud auth login`
- Project: `project-fba1f920-ee91-46d1-99c`
- Compute Engine API enabled (the script enables it automatically)

---

## Account-specific values you must provide

| Value | Where to set |
|---|---|
| GCP project ID | `--project project-fba1f920-ee91-46d1-99c` |
| Region/zone | `--zone europe-west2-b` (London — default in the script) |
| Machine types | `--machine-types "n2-standard-4 t2d-standard-4 c2-standard-8"` |
| Budget cap | £50 — cost estimate is ~£0.22 per full 3-machine run |
| `GCP_PRICING_API_KEY` | Optional; leave unset to use built-in static rates |

---

## Local dry-run (smoke test — no GCP needed)

Verifies the full pipeline: candidate generation, probe/full phase labelling, DB writes, exports, and profiler metrics.  Uses tiny M values and a temp DB.

```bash
python experiments/run_profiler_vs_grid.py --dry-run
```

Expected output:
```
[dry-run] Using temp DB: /tmp/tmpXXXXXX.db
...
[dry-run] Pipeline completed successfully.
[dry-run] Cleaning up temp DB: /tmp/tmpXXXXXX.db
```

---

## Local full experiment (no GCP)

```bash
python experiments/run_profiler_vs_grid.py \
    --experiment-id local_test_v1 \
    --workloads european european_local_vol asian \
    --engines cpu jax \
    --m-probe 1000 \
    --m-values 10000 50000 100000 \
    --runs 5 --warmup 2 \
    --runs-probe 2 --warmup-probe 1 \
    --top-k 3 \
    --write-db results/benchmarks.db \
    --export results/
```

---

## GCP full experiment

### Step 1 — Authenticate
```bash
gcloud auth login
gcloud config set project project-fba1f920-ee91-46d1-99c
```

### Step 2 — Smoke test (probe-only, single cheap machine)
```bash
./scripts/run_cloud_profiler.sh \
    --project project-fba1f920-ee91-46d1-99c \
    --machine-types n2-standard-4 \
    --probe-only \
    --experiment-id smoke_test_v1
```

### Step 3 — Full experiment (3 machine types in parallel)
```bash
./scripts/run_cloud_profiler.sh \
    --project project-fba1f920-ee91-46d1-99c \
    --experiment-id final_cloud_profiler_v1
```

This will:
1. Enable the Compute Engine API
2. Create 3 VMs (`n2-standard-4`, `t2d-standard-4`, `c2-standard-8`) in `europe-west2-b`
3. Install deps, clone the repo, run the profiler experiment on each VM in parallel
4. Copy `results/benchmarks.db` from each VM back locally
5. Delete all 3 VMs
6. Merge the 3 VM databases into `results/benchmarks.db`
7. Export `results/profiler_vs_grid.csv`, `cloud_cost_analysis.csv`, `pareto_frontier.csv`

**Estimated runtime:** ~30 minutes (dominated by the longest VM)  
**Estimated cost:** ~£0.22 total

---

## Monitoring VMs during the run

```bash
# List running VMs
gcloud compute instances list --project project-fba1f920-ee91-46d1-99c

# SSH into a VM to check progress
gcloud compute ssh bench-n2-standard-4-TIMESTAMP \
    --project project-fba1f920-ee91-46d1-99c \
    --zone europe-west2-b \
    --command "tail -f ~/benchmark_repo/results/profiler_summary.txt"
```

---

## Copying results back manually

If a VM is still running or `--no-delete-vms` was used:

```bash
gcloud compute scp \
    bench-n2-standard-4-TIMESTAMP:~/benchmark_repo/results/benchmarks.db \
    results/benchmarks_n2-standard-4_manual.db \
    --project project-fba1f920-ee91-46d1-99c \
    --zone europe-west2-b
```

---

## Merging VM databases

```bash
# Merge all timestamped VM DBs into the main DB
python scripts/merge_sqlite_results.py \
    --source results/benchmarks_n2-standard-4_*.db \
              results/benchmarks_t2d-standard-4_*.db \
              results/benchmarks_c2-standard-8_*.db \
    --target results/benchmarks.db

# Dry-run first to see what would be inserted
python scripts/merge_sqlite_results.py \
    --source results/benchmarks_*.db \
    --target results/benchmarks.db \
    --dry-run
```

---

## Regenerating exports

```bash
# Re-export all CSVs from the merged DB
python experiments/run_profiler_vs_grid.py \
    --experiment-id final_cloud_profiler_v1 \
    --probe-only \
    --write-db results/benchmarks.db \
    --export results/
```

Or query directly:
```bash
sqlite3 results/benchmarks.db "
  SELECT instance_type, engine, workload_type, ad_mode, M,
         round(mean_runtime_ms, 2) AS ms,
         round(cost_per_run, 6)    AS cost,
         profiler_decision
  FROM runs
  WHERE experiment_id = 'final_cloud_profiler_v1'
    AND status = 'completed'
  ORDER BY instance_type, mean_runtime_ms
  LIMIT 30;"
```

---

## Files to commit

After the experiment, these files should be committed:

```
experiments/run_profiler_vs_grid.py      # hardened profiler
benchmarking/storage/database.py         # extended schema
scripts/merge_sqlite_results.py          # new merge script
scripts/run_cloud_profiler.sh            # new GCP orchestration
docs/cloud_profiler.md                   # this file
tests/test_system.py                     # test drift fix
results/benchmarks.db                    # merged benchmark results
results/profiler_summary.txt             # text summary
results/profiler_vs_grid.csv             # full run data
results/cloud_cost_analysis.csv          # cost-efficiency table
results/pareto_frontier.csv              # Pareto configs
results/report_tables/                   # per-table CSVs
```

## Files that must NEVER be committed

```
.env
venv/
*.db.bak
results/benchmarks_*_????????_??????.db  # per-VM temporary DBs
**/__pycache__/
*.pyc
*.key
*.json  # service account keys
gcloud-credentials*
```

---

## Summary of key CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--experiment-id` | auto UUID | Stable ID stored in DB and exports |
| `--dry-run` | off | Tiny M, temp DB, smoke-test mode |
| `--probe-only` | off | Stop after probe phase |
| `--top-k` | 3 | Max configs selected per workload |
| `--score-margin` | 1.5 | Safety margin multiplier |
| `--m-probe` | 1000 | Paths for probe runs |
| `--m-values` | 10k 50k 100k | Full-run path counts |
| `--write-db` | results/benchmarks.db | Target SQLite DB |
| `--export` | (none) | Directory for CSV exports |
| `--instance-type` | (auto-detected) | GCP machine type for cost calculation |
| `--region` / `--zone` | (auto-detected) | GCP location metadata |

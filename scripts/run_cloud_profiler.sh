#!/usr/bin/env bash
# =============================================================================
# run_cloud_profiler.sh — Automated GCP multi-VM profiler experiment
#
# Creates one VM per machine type, runs the budget-aware profiler vs grid
# experiment on each, copies results back, merges DBs, and exports summaries.
# VMs are deleted after results are safely copied.
#
# Usage
# -----
#   # Full experiment: 3 machine types, London region
#   ./scripts/run_cloud_profiler.sh \
#     --project project-fba1f920-ee91-46d1-99c \
#     --experiment-id final_cloud_profiler_v1
#
#   # Quick test: single machine, probe-only
#   ./scripts/run_cloud_profiler.sh \
#     --project project-fba1f920-ee91-46d1-99c \
#     --machine-types n2-standard-4 \
#     --probe-only \
#     --experiment-id smoke_test_v1
#
#   # Skip VM creation (VMs already running)
#   ./scripts/run_cloud_profiler.sh \
#     --project project-fba1f920-ee91-46d1-99c \
#     --no-create-vms \
#     --experiment-id final_cloud_profiler_v1
#
# Prerequisites
# -------------
#   gcloud CLI installed and authenticated:   gcloud auth login
#   Project billing enabled.
#   Compute Engine API enabled:
#     gcloud services enable compute.googleapis.com --project PROJECT
#
# Cost estimate (europe-west2, on-demand)
# ---------------------------------------
#   n2-standard-4:  ~$0.19/hr  x ~25 min = ~$0.08 per run
#   t2d-standard-4: ~$0.17/hr  x ~30 min = ~$0.09 per run
#   c2-standard-8:  ~$0.42/hr  x ~15 min = ~$0.11 per run
#   Total for 3 machines: ~$0.28 per full experiment (~£0.22)
#   Budget headroom: £49.78 remaining of £50 cap
#
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults — override via CLI flags
# ---------------------------------------------------------------------------
PROJECT=""
ZONE="europe-west2-b"
REGION="europe-west2"
# 6 diverse machine types: AMD Milan, Intel Cascade Lake, Google cost-opt, AMD EPYC, AMD compute-opt
# n2-standard-4 uses europe-west1-b to avoid europe-west2 quota exhaustion
MACHINE_TYPES="e2-standard-4 e2-standard-8 t2d-standard-4 n2d-standard-4 c2d-standard-4"
N2_ZONE="europe-west1-b"    # Intel Cascade Lake — different zone for quota
EXPERIMENT_ID="sha_cloud_profiler_v2"
RUNS=5
WARMUP=2
M_VALUES="10000 50000 100000"
M_PROBE=1000
TOP_K=3
SCORE_MARGIN=1.5
# SHA parameters
SHA_M_LEVELS="1000 5000 25000"
SHA_ETA=2.0
SHA_RUNS_PER_LEVEL="3 5 7"
SHA_WARMUP_PER_LEVEL="1 2 2"
CREATE_VMS=true
DELETE_VMS_AFTER=true
PROBE_ONLY=false
GCP_API_KEY="${GCP_PRICING_API_KEY:-}"
REPO_URL="https://github.com/Vivian-Lopez/Monte-Carlo-and-Automatic-Differentiation-Systematic-Benchmarking-Framework.git"
REPO_DIR="~/benchmark_repo"
RESULTS_LOCAL_DIR="results"
DISK_SIZE="30GB"
IMAGE_FAMILY="debian-12"
IMAGE_PROJECT="debian-cloud"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
usage() {
    grep '^#' "$0" | head -50 | sed 's/^# \{0,1\}//'
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --project)        PROJECT="$2";              shift 2 ;;
        --zone)           ZONE="$2";                 shift 2 ;;
        --region)         REGION="$2";               shift 2 ;;
        --machine-types)  MACHINE_TYPES="$2";        shift 2 ;;
        --experiment-id)  EXPERIMENT_ID="$2";        shift 2 ;;
        --runs)           RUNS="$2";                 shift 2 ;;
        --warmup)         WARMUP="$2";               shift 2 ;;
        --runs-probe)     RUNS_PROBE="$2";           shift 2 ;;
        --warmup-probe)   WARMUP_PROBE="$2";         shift 2 ;;
        --m-values)       M_VALUES="$2";             shift 2 ;;
        --m-probe)        M_PROBE="$2";              shift 2 ;;
        --top-k)          TOP_K="$2";                shift 2 ;;
        --score-margin)   SCORE_MARGIN="$2";         shift 2 ;;
        --no-create-vms)          CREATE_VMS=false;              shift   ;;
        --no-delete-vms)          DELETE_VMS_AFTER=false;        shift   ;;
        --probe-only)             PROBE_ONLY=true;               shift   ;;
        --gcp-api-key)            GCP_API_KEY="$2";              shift 2 ;;
        --sha-m-levels)           SHA_M_LEVELS="$2";             shift 2 ;;
        --sha-eta)                SHA_ETA="$2";                  shift 2 ;;
        --sha-runs-per-level)     SHA_RUNS_PER_LEVEL="$2";       shift 2 ;;
        --sha-warmup-per-level)   SHA_WARMUP_PER_LEVEL="$2";    shift 2 ;;
        --help|-h)                usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if [[ -z "$PROJECT" ]]; then
    echo "ERROR: --project is required."
    echo "  Usage: $0 --project YOUR_PROJECT_ID"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$RESULTS_LOCAL_DIR"

echo "========================================================================"
echo "  GCP Cloud Profiler Experiment"
echo "========================================================================"
echo "  Project         : $PROJECT"
echo "  Region/zone     : $REGION / $ZONE"
echo "  Machine types   : $MACHINE_TYPES"
echo "  Experiment ID   : $EXPERIMENT_ID"
echo "  Runs/warmup     : $RUNS / $WARMUP"
echo "  M values        : $M_VALUES"
echo "  SHA M levels    : $SHA_M_LEVELS  eta=$SHA_ETA"
echo "  SHA runs/lvl    : $SHA_RUNS_PER_LEVEL  warmup=$SHA_WARMUP_PER_LEVEL"
echo "  Create VMs      : $CREATE_VMS"
echo "  Delete after    : $DELETE_VMS_AFTER"
echo "  Probe only      : $PROBE_ONLY"
echo "========================================================================"
echo ""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
vm_name_for() {
    # Sanitise machine type to a valid VM name (lowercase letters, digits, hyphens only)
    local mt="$1"
    # Strip timestamp to date-only (YYYYMMDD) to avoid underscore from HHMMSS separator
    local ts
    ts=$(echo "$TIMESTAMP" | cut -c1-8)
    echo "bench-${mt}-${ts}" | tr '_' '-' | tr '.' '-' | tr '[:upper:]' '[:lower:]' | cut -c1-63
}

vm_ssh() {
    local vm="$1"; shift
    gcloud compute ssh "$vm" \
        --project="$PROJECT" \
        --zone="$ZONE" \
        --command="$1" \
        --ssh-flag="-o StrictHostKeyChecking=no" \
        --ssh-flag="-o ConnectTimeout=30" \
        --quiet
}

wait_for_ssh() {
    local vm="$1"
    local max=24   # 4 minutes total
    local i=0
    echo "  Waiting for SSH on $vm ..."
    until vm_ssh "$vm" "echo ok" &>/dev/null; do
        i=$(( i + 1 ))
        if [[ $i -ge $max ]]; then
            echo "  ERROR: $vm did not become SSH-ready after $(( max * 10 ))s."
            return 1
        fi
        echo "    attempt $i/$max — retrying in 10s ..."
        sleep 10
    done
    echo "  SSH ready: $vm"
}

# ---------------------------------------------------------------------------
# Step 1 — Enable Compute Engine API (idempotent)
# ---------------------------------------------------------------------------
echo "[1/6] Enabling Compute Engine API ..."
gcloud services enable compute.googleapis.com \
    --project="$PROJECT" --quiet 2>/dev/null || true
echo "  OK"

# ---------------------------------------------------------------------------
# Step 2 — Create VMs in parallel
# ---------------------------------------------------------------------------
declare -A VM_NAMES=()
declare -A VM_DBS=()

for MT in $MACHINE_TYPES n2-standard-4; do
    VM="${VM_NAMES[$MT]:-$(vm_name_for "$MT")}"
    VM_NAMES[$MT]="$VM"
    VM_DBS[$MT]="$RESULTS_LOCAL_DIR/benchmarks_${MT}_${TIMESTAMP}.db"
done

if [[ "$CREATE_VMS" == true ]]; then
    echo ""
    echo "[2/6] Creating VMs ..."
    for MT in $MACHINE_TYPES; do
        VM="${VM_NAMES[$MT]}"
        echo "  Creating: $VM ($MT)"
        if gcloud compute instances create "$VM" \
            --project="$PROJECT" \
            --zone="$ZONE" \
            --machine-type="$MT" \
            --image-family="$IMAGE_FAMILY" \
            --image-project="$IMAGE_PROJECT" \
            --boot-disk-size="$DISK_SIZE" \
            --boot-disk-type="pd-ssd" \
            --scopes="cloud-platform" \
            --labels="project=mcad,experiment=cloud-profiler,instance-type=${MT}" \
            --metadata="enable-oslogin=true" \
            --quiet 2>&1; then
            echo "  Created: $VM"
        else
            echo "  [WARN] Failed to create $VM ($MT) — skipping this machine type"
            unset "VM_NAMES[$MT]"
        fi
    done
else
    echo ""
    echo "[2/6] Skipping VM creation (--no-create-vms)"
fi

# ---------------------------------------------------------------------------
# Step 3 — Wait for SSH, install deps, run experiment (per VM)
# ---------------------------------------------------------------------------
echo ""
echo "[3/6] Setting up and running experiments ..."

declare -A VM_PIDS=()

for MT in $MACHINE_TYPES n2-standard-4; do
    VM="${VM_NAMES[$MT]:-}"
    if [[ -z "$VM" ]]; then
        echo "  Skipping $MT (VM not created)"
        continue
    fi
    DB_LOCAL="${VM_DBS[$MT]}"

    # Zone for this machine type
    USE_ZONE="$ZONE"
    [[ "$MT" == "n2-standard-4" ]] && USE_ZONE="$N2_ZONE"

    # Build CLI args for the profiler
    M_CLI=$(echo "$M_VALUES" | tr ' ' '\n' | xargs printf " %s")
    SHA_M_CLI=$(echo "$SHA_M_LEVELS" | tr ' ' '\n' | xargs printf " %s")
    SHA_RUNS_CLI=$(echo "$SHA_RUNS_PER_LEVEL" | tr ' ' '\n' | xargs printf " %s")
    SHA_WARMUP_CLI=$(echo "$SHA_WARMUP_PER_LEVEL" | tr ' ' '\n' | xargs printf " %s")
    PROBE_ONLY_FLAG=""
    [[ "$PROBE_ONLY" == true ]] && PROBE_ONLY_FLAG="--probe-only"

    API_KEY_ARG=""
    [[ -n "$GCP_API_KEY" ]] && API_KEY_ARG="--gcp-api-key '${GCP_API_KEY}'"

    (
        echo "  [$VM] Waiting for SSH ..."
        wait_for_ssh "$VM" || exit 1

        echo "  [$VM] Installing system dependencies ..."
        vm_ssh "$VM" "
            sudo apt-get update -qq 2>/dev/null
            sudo apt-get install -y -qq python3-venv python3-pip git build-essential \
                libopenblas-dev pkg-config curl ca-certificates 2>/dev/null
        " || true

        echo "  [$VM] Cloning / updating repository ..."
        vm_ssh "$VM" "
            if [[ -d ${REPO_DIR} ]]; then
                cd ${REPO_DIR} && git pull --quiet
            else
                git clone --quiet '${REPO_URL}' ${REPO_DIR}
            fi
        "

        echo "  [$VM] Installing Python requirements ..."
        vm_ssh "$VM" "
            cd ${REPO_DIR}
            python3 -m venv venv
            source venv/bin/activate
            pip install --quiet --upgrade pip
            pip install --quiet -r requirements.txt
        "

        echo "  [$VM] Attempting optional C++ build ..."
        vm_ssh "$VM" "
            cd ${REPO_DIR}
            source venv/bin/activate
            pip install --quiet -e benchmarking/cpp/ 2>/dev/null && echo 'C++ OK' \
                || echo 'C++ skipped'
        " || true

        echo "  [$VM] Attempting Rust build (maturin) ..."
        vm_ssh "$VM" "
            cd ${REPO_DIR}
            source venv/bin/activate
            pip install --quiet maturin 2>/dev/null
            cd benchmarking/rust && maturin develop --release --quiet 2>/dev/null \
                && echo 'Rust OK' || echo 'Rust skipped'
        " || true

        echo "  [$VM] Running profiler experiment (SHA edition) ..."
        vm_ssh "$VM" "
            cd ${REPO_DIR}
            source venv/bin/activate
            python experiments/run_profiler_vs_grid.py \
                --experiment-id '${EXPERIMENT_ID}' \
                --workloads european european_local_vol asian \
                --engines cpu jax cpp rust \
                --m-probe ${M_PROBE} \
                --m-values ${M_CLI} \
                --runs ${RUNS} --warmup ${WARMUP} \
                --top-k ${TOP_K} \
                --score-margin ${SCORE_MARGIN} \
                --sha-m-levels ${SHA_M_CLI} \
                --sha-eta ${SHA_ETA} \
                --sha-runs-per-level ${SHA_RUNS_CLI} \
                --sha-warmup-per-level ${SHA_WARMUP_CLI} \
                --cloud-provider gcp \
                --region ${REGION} \
                --zone ${USE_ZONE} \
                --instance-type ${MT} \
                --write-db results/benchmarks.db \
                --export results/ \
                ${PROBE_ONLY_FLAG} \
                ${API_KEY_ARG}
        "

        echo "  [$VM] Copying results ..."
        mkdir -p "$RESULTS_LOCAL_DIR"
        gcloud compute scp \
            "${VM}:${REPO_DIR}/results/benchmarks.db" \
            "$DB_LOCAL" \
            --project="$PROJECT" \
            --zone="$ZONE" \
            --quiet
        echo "  [$VM] Results saved to: $DB_LOCAL"

        # Also copy the text summary and SHA progression CSV
        gcloud compute scp \
            "${VM}:${REPO_DIR}/results/profiler_summary.txt" \
            "$RESULTS_LOCAL_DIR/profiler_summary_${MT}_${TIMESTAMP}.txt" \
            --project="$PROJECT" \
            --zone="$USE_ZONE" \
            --quiet 2>/dev/null || true

        gcloud compute scp \
            "${VM}:${REPO_DIR}/results/sha_progression.csv" \
            "$RESULTS_LOCAL_DIR/sha_progression_${MT}_${TIMESTAMP}.csv" \
            --project="$PROJECT" \
            --zone="$USE_ZONE" \
            --quiet 2>/dev/null || true

    ) &

    VM_PIDS[$MT]=$!
    echo "  Launched background job for $MT (PID ${VM_PIDS[$MT]})"
done

# Wait for all VMs to finish
echo ""
echo "  Waiting for all VM jobs to complete ..."
ALL_OK=true
for MT in $MACHINE_TYPES n2-standard-4; do
    [[ -z "${VM_NAMES[$MT]:-}" ]] && continue
    [[ -z "${VM_PIDS[$MT]:-}" ]] && continue
    wait "${VM_PIDS[$MT]}" || { echo "  [WARN] VM job for $MT failed"; ALL_OK=false; }
done

if [[ "$ALL_OK" == false ]]; then
    echo ""
    echo "[WARN] One or more VM jobs failed. Check output above."
    echo "       Continuing with available results ..."
fi

# ---------------------------------------------------------------------------
# Step 4 — Delete VMs
# ---------------------------------------------------------------------------
if [[ "$DELETE_VMS_AFTER" == true ]]; then
    echo ""
    echo "[4/6] Deleting VMs ..."
    for MT in $MACHINE_TYPES; do
        VM="${VM_NAMES[$MT]:-}"
        [[ -z "$VM" ]] && continue
        echo "  Deleting: $VM"
        gcloud compute instances delete "$VM" \
            --project="$PROJECT" \
            --zone="$ZONE" \
            --quiet 2>/dev/null || echo "  [WARN] Could not delete $VM"
    done
    echo "  VMs deleted."
else
    echo ""
    echo "[4/6] Skipping VM deletion (--no-delete-vms)"
    echo "  Active VMs:"
    for MT in $MACHINE_TYPES n2-standard-4; do
        VM="${VM_NAMES[$MT]:-}"
        [[ -z "$VM" ]] && continue
        echo "    ${VM}"
    done
    echo "  To delete manually:"
    for MT in $MACHINE_TYPES n2-standard-4; do
        VM="${VM_NAMES[$MT]:-}"
        [[ -z "$VM" ]] && continue
        USE_ZONE="$ZONE"; [[ "$MT" == "n2-standard-4" ]] && USE_ZONE="$N2_ZONE"
        echo "    gcloud compute instances delete ${VM} --project=$PROJECT --zone=${USE_ZONE}"
    done
fi

# ---------------------------------------------------------------------------
# Step 5 — Merge all VM databases into main DB
# ---------------------------------------------------------------------------
echo ""
echo "[5/6] Merging VM databases into results/benchmarks.db ..."

MERGE_SOURCES=()
for MT in $MACHINE_TYPES n2-standard-4; do
    DB="${VM_DBS[$MT]:-}"
    [[ -z "$DB" ]] && continue
    if [[ -f "$DB" ]]; then
        MERGE_SOURCES+=("$DB")
        echo "  Found: $DB"
    else
        echo "  [WARN] Missing: $DB"
    fi
done

if [[ ${#MERGE_SOURCES[@]} -gt 0 ]]; then
    python scripts/merge_sqlite_results.py \
        --source "${MERGE_SOURCES[@]}" \
        --target results/benchmarks.db
else
    echo "  [WARN] No source DBs found to merge."
fi

# ---------------------------------------------------------------------------
# Step 6 — Regenerate exports from merged DB
# ---------------------------------------------------------------------------
echo ""
echo "[6/6] Regenerating exports from merged database ..."

# Re-run the profiler in export-only mode by querying the DB directly
python - <<'PYEOF'
import sys
sys.path.insert(0, ".")
from pathlib import Path
from benchmarking.storage.database import BenchmarkDB
import sqlite3, csv

db_path = Path("results/benchmarks.db")
if not db_path.exists():
    print("  [WARN] results/benchmarks.db not found — skipping exports")
    sys.exit(0)

conn = sqlite3.connect(str(db_path))
conn.row_factory = sqlite3.Row

export_dir = Path("results")
export_dir.mkdir(exist_ok=True)
report_dir = export_dir / "report_tables"
report_dir.mkdir(exist_ok=True)

def export_query(path, sql, params=()):
    rows = conn.execute(sql, params).fetchall()
    if not rows:
        print(f"  [SKIP] No rows for {path.name}")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(dict(r) for r in rows)
    print(f"  Exported: {path}  ({len(rows)} rows)")

export_query(
    export_dir / "profiler_vs_grid.csv",
    """SELECT id, experiment_id, experiment_type, workload_type, engine, ad_mode, M,
              mean_runtime_ms, std_runtime_ms, throughput_paths_per_sec,
              cost_per_run, paths_per_dollar,
              profiler_phase, profiler_decision, profiler_reason, dominated,
              sha_round, sha_eliminated,
              scaling_law_alpha, scaling_law_beta,
              extrapolated_runtime_ms, extrapolation_error_pct,
              ad_overhead_ratio,
              instance_type, region, zone, machine_family, vcpu_count,
              result_value, memory_peak_mb, git_commit_hash, created_at
       FROM runs WHERE status = 'completed'
       ORDER BY instance_type, workload_type, engine, ad_mode, M"""
)

export_query(
    export_dir / "sha_progression.csv",
    """SELECT workload_type, engine, ad_mode, M, instance_type,
              mean_runtime_ms, std_runtime_ms,
              sha_round, sha_eliminated,
              scaling_law_alpha, scaling_law_beta,
              extrapolated_runtime_ms, extrapolation_error_pct,
              ad_overhead_ratio, profiler_phase, profiler_decision
       FROM runs WHERE status = 'completed' AND sha_round IS NOT NULL
       ORDER BY instance_type, sha_round, workload_type, engine, ad_mode"""
)

export_query(
    export_dir / "cloud_cost_analysis.csv",
    """SELECT workload_type, engine, ad_mode, M, instance_type, machine_family, vcpu_count,
              mean_runtime_ms, throughput_paths_per_sec, cost_per_run, paths_per_dollar,
              region, zone
       FROM runs WHERE status = 'completed' AND cost_per_run IS NOT NULL AND cost_per_run > 0
       ORDER BY instance_type, paths_per_dollar DESC"""
)

export_query(
    export_dir / "pareto_frontier.csv",
    """SELECT workload_type, engine, ad_mode, M, instance_type,
              mean_runtime_ms, cost_per_run, paths_per_dollar,
              profiler_decision, profiler_reason, dominated
       FROM runs WHERE status = 'completed'
         AND experiment_type IN ('profiler_selected', 'grid_search_full')
       ORDER BY workload_type, mean_runtime_ms"""
)

export_query(
    report_dir / "probe_vs_full_ranking.csv",
    """SELECT workload_type, engine, ad_mode, M, instance_type,
              mean_runtime_ms, profiler_phase, profiler_decision
       FROM runs WHERE status = 'completed'
       ORDER BY workload_type, profiler_phase, mean_runtime_ms"""
)

conn.close()
print("  Exports complete.")
PYEOF

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "  EXPERIMENT COMPLETE"
echo "========================================================================"
echo "  Experiment ID : $EXPERIMENT_ID"
echo "  Main DB       : results/benchmarks.db"
echo "  Exports       : results/profiler_vs_grid.csv"
echo "                  results/cloud_cost_analysis.csv"
echo "                  results/pareto_frontier.csv"
echo "                  results/report_tables/probe_vs_full_ranking.csv"
echo ""
echo "  To preview results:"
echo "    python preview_results.py"
echo ""
echo "  To query the DB:"
echo "    sqlite3 results/benchmarks.db \\"
echo "      \"SELECT instance_type, engine, workload_type, ad_mode, M,"
echo "               round(mean_runtime_ms,2), round(cost_per_run,6)"
echo "        FROM runs WHERE experiment_id='$EXPERIMENT_ID'"
echo "        ORDER BY mean_runtime_ms LIMIT 20;\""
echo ""
echo "  To re-export:"
echo "    python experiments/run_profiler_vs_grid.py --dry-run  # test"
echo "========================================================================"

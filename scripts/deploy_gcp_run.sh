#!/usr/bin/env bash
# =============================================================================
# deploy_gcp_run.sh — GCP VM deployment script for the local vol benchmark
#
# Sets up a GCP Compute Engine VM, installs all dependencies, runs the local
# volatility cloud profile experiment, and retrieves the results.
#
# Usage
# -----
#   ./scripts/deploy_gcp_run.sh \
#     --project my-gcp-project \
#     --zone    us-central1-a \
#     --machine-type n2-standard-8 \
#     --vm-name bench-n2-standard-8 \
#     [--create-vm] \
#     [--delete-vm-after] \
#     [--runs 7] [--warmup 3] \
#     [--m-values "10000 50000 100000"] \
#     [--ad-modes "none forward reverse"] \
#     [--gcp-api-key KEY]
#
# Requirements
# ------------
#   - gcloud CLI installed and authenticated (gcloud auth login)
#   - Target project has Compute Engine API enabled
#   - Your gcloud account has roles/compute.instanceAdmin.v1
#
# Target machine types (add more to the KNOWN_RATES table in pricing.py)
# -----------------------------------------------------------------------
#   n2-standard-8   Intel Cascade Lake / Sapphire Rapids  ~$0.388/hr
#   t2d-standard-8  AMD EPYC Milan                        ~$0.340/hr
#   c2-standard-8   Intel compute-optimised               ~$0.419/hr
#
# GPU (future extension)
# ----------------------
#   Swap --image-family to a CUDA image, e.g. c0-deeplearning-common-cu122-*
#   and install jax[cuda12] instead of jax[cpu].
#   See scripts/README.md for details.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
PROJECT=""
ZONE="us-central1-a"
MACHINE_TYPE="n2-standard-8"
VM_NAME=""
CREATE_VM=false
DELETE_VM_AFTER=false
RUNS=7
WARMUP=3
M_VALUES="10000 50000 100000"
AD_MODES="none forward reverse"
GCP_API_KEY="${GCP_PRICING_API_KEY:-}"
REPO_URL="https://github.com/Vivian-Lopez/Monte-Carlo-and-Automatic-Differentiation-Systematic-Benchmarking-Framework.git"
REPO_DIR="~/benchmark_repo"
RESULTS_DIR="./results"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
usage() {
    grep '^#' "$0" | head -30 | sed 's/^# \{0,1\}//'
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --project)       PROJECT="$2";       shift 2 ;;
        --zone)          ZONE="$2";          shift 2 ;;
        --machine-type)  MACHINE_TYPE="$2";  shift 2 ;;
        --vm-name)       VM_NAME="$2";       shift 2 ;;
        --create-vm)     CREATE_VM=true;     shift   ;;
        --delete-vm-after) DELETE_VM_AFTER=true; shift ;;
        --runs)          RUNS="$2";          shift 2 ;;
        --warmup)        WARMUP="$2";        shift 2 ;;
        --m-values)      M_VALUES="$2";      shift 2 ;;
        --ad-modes)      AD_MODES="$2";      shift 2 ;;
        --gcp-api-key)   GCP_API_KEY="$2";   shift 2 ;;
        --help|-h)       usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if [[ -z "$PROJECT" ]]; then
    echo "ERROR: --project is required."
    exit 1
fi
if [[ -z "$VM_NAME" ]]; then
    VM_NAME="bench-${MACHINE_TYPE}-$(date +%Y%m%d%H%M%S)"
    echo "[info] --vm-name not set; using auto-generated name: $VM_NAME"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOCAL_DB="$RESULTS_DIR/benchmarks_${MACHINE_TYPE}_${TIMESTAMP}.db"

# ---------------------------------------------------------------------------
# Helper: run a command on the VM via gcloud compute ssh
# ---------------------------------------------------------------------------
vm_ssh() {
    gcloud compute ssh "$VM_NAME" \
        --project="$PROJECT" \
        --zone="$ZONE" \
        --command="$1" \
        --ssh-flag="-o StrictHostKeyChecking=no" \
        --ssh-flag="-o ConnectTimeout=30"
}

# ---------------------------------------------------------------------------
# Step 1 — Optionally create VM
# ---------------------------------------------------------------------------
if [[ "$CREATE_VM" == true ]]; then
    echo
    echo "==> [1/8] Creating VM: $VM_NAME ($MACHINE_TYPE, $ZONE)"
    gcloud compute instances create "$VM_NAME" \
        --project="$PROJECT" \
        --zone="$ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --image-family="debian-12" \
        --image-project="debian-cloud" \
        --boot-disk-size="20GB" \
        --boot-disk-type="pd-ssd" \
        --scopes="cloud-platform" \
        --metadata="enable-oslogin=true"
    echo "    VM created."
else
    echo
    echo "==> [1/8] Skipping VM creation (--create-vm not set)"
fi

# ---------------------------------------------------------------------------
# Step 2 — Wait for SSH readiness (up to 3 minutes)
# ---------------------------------------------------------------------------
echo
echo "==> [2/8] Waiting for SSH readiness ..."
MAX_ATTEMPTS=18
ATTEMPT=0
until vm_ssh "echo ready" 2>/dev/null; do
    ATTEMPT=$(( ATTEMPT + 1 ))
    if [[ $ATTEMPT -ge $MAX_ATTEMPTS ]]; then
        echo "ERROR: VM did not become SSH-ready after $(( MAX_ATTEMPTS * 10 )) seconds."
        exit 1
    fi
    echo "    Not ready yet (attempt $ATTEMPT/$MAX_ATTEMPTS) — waiting 10 s ..."
    sleep 10
done
echo "    SSH ready."

# ---------------------------------------------------------------------------
# Step 3 — Install system dependencies
# ---------------------------------------------------------------------------
echo
echo "==> [3/8] Installing system dependencies ..."
vm_ssh "sudo apt-get update -qq && sudo apt-get install -y -qq \
    python3-venv python3-pip git \
    build-essential libopenblas-dev pkg-config \
    curl ca-certificates"

# ---------------------------------------------------------------------------
# Step 4 — Clone repository
# ---------------------------------------------------------------------------
echo
echo "==> [4/8] Cloning repository ..."
vm_ssh "
    if [[ -d ${REPO_DIR} ]]; then
        echo 'Repo already exists — pulling latest';
        cd ${REPO_DIR} && git pull --quiet;
    else
        git clone --quiet '${REPO_URL}' ${REPO_DIR};
    fi
"

# ---------------------------------------------------------------------------
# Step 5 — Python venv + core requirements
# ---------------------------------------------------------------------------
echo
echo "==> [5/8] Setting up Python venv and installing requirements ..."
vm_ssh "
    cd ${REPO_DIR}
    python3 -m venv venv
    source venv/bin/activate
    pip install --quiet --upgrade pip
    pip install --quiet -r requirements.txt
"

# ---------------------------------------------------------------------------
# Step 6 — Optional C++ build
# ---------------------------------------------------------------------------
echo
echo "==> [6/8] Attempting C++ OpenMP build ..."
vm_ssh "
    cd ${REPO_DIR}
    source venv/bin/activate
    pip install --quiet -e benchmarking/cpp/ && echo 'C++ build: OK' \
        || echo 'C++ build: SKIPPED (pybind11 or compiler missing)'
" || true

# ---------------------------------------------------------------------------
# Step 7 — Optional Rust build
# ---------------------------------------------------------------------------
echo
echo "==> [7/8] Attempting Rust build ..."
vm_ssh "
    set +e
    # Install Rust toolchain if not present
    if ! command -v cargo &>/dev/null; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
            | sh -s -- -y --quiet
        source \"\$HOME/.cargo/env\"
    fi
    source \"\$HOME/.cargo/env\"
    cd ${REPO_DIR}
    source venv/bin/activate
    pip install --quiet maturin
    cd benchmarking/rust
    maturin develop --release --quiet && echo 'Rust build: OK' \
        || echo 'Rust build: SKIPPED'
    set -e
" || true

# ---------------------------------------------------------------------------
# Step 8 — Run the experiment
# ---------------------------------------------------------------------------
echo
echo "==> [8/8] Running local vol cloud profile ..."
echo "    Params: runs=$RUNS  warmup=$WARMUP  M='$M_VALUES'  AD='$AD_MODES'"

# Build optional api-key arg
API_KEY_ARG=""
if [[ -n "$GCP_API_KEY" ]]; then
    API_KEY_ARG="--gcp-api-key '${GCP_API_KEY}'"
fi

# Convert space-separated strings to CLI args
M_CLI=$(echo "$M_VALUES" | tr ' ' '\n' | xargs printf " %s")
AD_CLI=$(echo "$AD_MODES" | tr ' ' '\n' | xargs printf " %s")

vm_ssh "
    cd ${REPO_DIR}
    source venv/bin/activate
    python experiments/run_localvol_cloud.py \
        --runs ${RUNS} \
        --warmup ${WARMUP} \
        --m-values ${M_CLI} \
        --ad-modes ${AD_CLI} \
        --instance-type '${MACHINE_TYPE}' \
        --cloud-provider gcp \
        ${API_KEY_ARG}
"

# ---------------------------------------------------------------------------
# Retrieve results
# ---------------------------------------------------------------------------
echo
echo "==> Retrieving results database ..."
mkdir -p "$RESULTS_DIR"
gcloud compute scp \
    "${VM_NAME}:${REPO_DIR}/results/benchmarks.db" \
    "$LOCAL_DB" \
    --project="$PROJECT" \
    --zone="$ZONE"
echo "    Saved to: $LOCAL_DB"

# ---------------------------------------------------------------------------
# Optional VM deletion
# ---------------------------------------------------------------------------
if [[ "$DELETE_VM_AFTER" == true ]]; then
    echo
    echo "==> Deleting VM: $VM_NAME ..."
    gcloud compute instances delete "$VM_NAME" \
        --project="$PROJECT" \
        --zone="$ZONE" \
        --quiet
    echo "    VM deleted."
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo
echo "========================================================================"
echo "  Run complete."
echo "  Results DB  : $LOCAL_DB"
echo "  Experiment  : localvol_cloud_profile / $MACHINE_TYPE"
echo ""
echo "  To analyse:"
echo "    python experiments/run_cloud_cost_analysis.py --db-path $LOCAL_DB"
echo "========================================================================"
echo

#!/usr/bin/env bash
set -euo pipefail

# Guarded SHA local-volatility GCP runner.
#
# Creates one VM per requested machine type, runs the local-volatility
# observation pool on the matching VM, copies structured outputs back, and then
# performs a combined cross-instance guarded-SHA analysis locally.

PROJECT=""
REGION="europe-west1"
MACHINE_TYPES="e2-standard-4 n2-standard-4 n2d-standard-4 c2d-standard-4 t2d-standard-4"
VCPU_BUDGET=12
FULL_BUDGET=100000
PROBE_BUDGETS="1000,5000,25000"
REPEATS=3
WARMUP=1
OUTPUT_DIR="results/guarded_sha_local_vol_gcp"
REPO_URL="https://github.com/Vivian-Lopez/Monte-Carlo-and-Automatic-Differentiation-Systematic-Benchmarking-Framework.git"
REPO_BRANCH="guarded-sha-profiler"
REPO_DIR="~/benchmark_repo"
GCP_API_KEY="${GCP_PRICING_API_KEY:-}"
HOURLY_RATES_JSON=""
DELETE_VMS_AFTER=true
DRY_RUN=false

usage() {
    cat <<'EOF'
Usage:
  scripts/run_guarded_sha_gcp.sh --project PROJECT [options]

Options:
  --machine-types "e2-standard-4 n2-standard-4 ..."
  --full-budget 100000
  --probe-budgets 1000,5000,25000
  --repeats 3
  --warmup 1
  --output-dir results/guarded_sha_local_vol_gcp
  --gcp-api-key KEY
  --hourly-rates-json path/to/rates.json
  --repo-branch guarded-sha-profiler
  --no-delete-vms
  --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --project) PROJECT="$2"; shift 2 ;;
        --region) REGION="$2"; shift 2 ;;
        --machine-types) MACHINE_TYPES="$2"; shift 2 ;;
        --full-budget) FULL_BUDGET="$2"; shift 2 ;;
        --probe-budgets) PROBE_BUDGETS="$2"; shift 2 ;;
        --repeats) REPEATS="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --gcp-api-key) GCP_API_KEY="$2"; shift 2 ;;
        --hourly-rates-json) HOURLY_RATES_JSON="$2"; shift 2 ;;
        --repo-branch) REPO_BRANCH="$2"; shift 2 ;;
        --no-delete-vms) DELETE_VMS_AFTER=false; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown argument: $1"; usage; exit 1 ;;
    esac
done

if [[ -z "$PROJECT" ]]; then
    echo "ERROR: --project is required"
    exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOCAL_RUN_DIR="${OUTPUT_DIR}/${TIMESTAMP}"
mkdir -p "$LOCAL_RUN_DIR"

mt_zone() {
    case "$1" in
        e2-*) echo "europe-west2-b" ;;
        n2-*) echo "europe-west1-b" ;;
        n2d-*) echo "europe-west4-a" ;;
        c2d-*) echo "europe-west1-b" ;;
        t2d-*) echo "europe-west1-b" ;;
        *) echo "europe-west1-b" ;;
    esac
}

mt_region() {
    local zone
    zone="$(mt_zone "$1")"
    echo "${zone%-*}"
}

mt_vcpus() {
    echo "$1" | sed -E 's/.*-([0-9]+)$/\1/'
}

vm_name_for() {
    local mt="$1"
    local vm="guarded-sha-${mt//-/_}-${TIMESTAMP}"
    echo "${vm//_/-}"
}

vm_ssh() {
    local vm="$1"
    local zone="$2"
    local command="$3"
    gcloud compute ssh "$vm" --project="$PROJECT" --zone="$zone" --quiet --command "$command"
}

wait_for_ssh() {
    local vm="$1"
    local zone="$2"
    for _ in {1..30}; do
        if vm_ssh "$vm" "$zone" "echo ready" >/dev/null 2>&1; then
            return 0
        fi
        sleep 10
    done
    return 1
}

declare -a ALL_MTS
read -r -a ALL_MTS <<< "$MACHINE_TYPES"

echo "Guarded SHA GCP run: $TIMESTAMP"
echo "Project: $PROJECT"
echo "Machine types: ${ALL_MTS[*]}"
echo "Output: $LOCAL_RUN_DIR"
echo "Repo branch: $REPO_BRANCH"

gcloud services enable compute.googleapis.com --project "$PROJECT" >/dev/null

current_wave_vcpus=0
declare -a wave=()

run_wave() {
    if [[ ${#wave[@]} -eq 0 ]]; then
        return
    fi
    echo "Running wave: ${wave[*]}"
    local pids=()
    local pid_mts=()

    for mt in "${wave[@]}"; do
        zone="$(mt_zone "$mt")"
        vm="$(vm_name_for "$mt")"
        echo "Creating $vm ($mt in $zone)"
        gcloud compute instances create "$vm" \
            --project="$PROJECT" \
            --zone="$zone" \
            --machine-type="$mt" \
            --image-family=debian-12 \
            --image-project=debian-cloud \
            --boot-disk-size=30GB \
            --scopes=cloud-platform \
            --labels=project=mcad,experiment=guarded-sha \
            --quiet
    done

    for mt in "${wave[@]}"; do
        (
            vm="$(vm_name_for "$mt")"
            zone="$(mt_zone "$mt")"
            region="$(mt_region "$mt")"
            echo "[$vm] Waiting for SSH"
            wait_for_ssh "$vm" "$zone"
            echo "[$vm] Installing dependencies"
            vm_ssh "$vm" "$zone" "
                sudo apt-get update -qq
                sudo apt-get install -y -qq python3-venv python3-pip git build-essential libopenblas-dev pkg-config curl ca-certificates
            "
            echo "[$vm] Cloning repository"
            vm_ssh "$vm" "$zone" "
                if [[ -d ${REPO_DIR} ]]; then
                    cd ${REPO_DIR} && git fetch --quiet origin '${REPO_BRANCH}' && git checkout --quiet '${REPO_BRANCH}' && git pull --quiet origin '${REPO_BRANCH}'
                else
                    git clone --quiet --branch '${REPO_BRANCH}' '${REPO_URL}' ${REPO_DIR}
                fi
            "
            echo "[$vm] Installing Python dependencies"
            vm_ssh "$vm" "$zone" "
                cd ${REPO_DIR}
                python3 -m venv venv
                source venv/bin/activate
                pip install --quiet --upgrade pip
                pip install --quiet -r requirements.txt
                pip install --quiet wheel setuptools pybind11
                pip install --quiet --no-build-isolation benchmarking/cpp/ || true
                command -v cargo >/dev/null 2>&1 || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --quiet || true
                export PATH=\"\$HOME/.cargo/bin:\$PATH\"
                pip install --quiet maturin || true
                cd benchmarking/rust && maturin develop --release --quiet || true
            "

            api_arg=""
            [[ -n "$GCP_API_KEY" ]] && api_arg="--gcp-api-key '$GCP_API_KEY'"
            rates_arg=""
            if [[ -n "$HOURLY_RATES_JSON" ]]; then
                gcloud compute scp "$HOURLY_RATES_JSON" "${vm}:/tmp/hourly_rates.json" --project="$PROJECT" --zone="$zone" --quiet
                rates_arg="--hourly-rates-json /tmp/hourly_rates.json"
            fi
            dry_arg=""
            [[ "$DRY_RUN" == true ]] && dry_arg="--dry-run"

            echo "[$vm] Running guarded SHA local-vol observation pool"
            vm_ssh "$vm" "$zone" "
                cd ${REPO_DIR}
                source venv/bin/activate
                python experiments/run_guarded_sha_local_vol.py \
                    --run-full-grid --run-plain-sha --run-guarded-sha \
                    --instances '${mt}' \
                    --region '${region}' \
                    --full-budget '${FULL_BUDGET}' \
                    --probe-budgets '${PROBE_BUDGETS}' \
                    --repeats '${REPEATS}' \
                    --warmup '${WARMUP}' \
                    --output-dir results/guarded_sha_local_vol \
                    ${dry_arg} ${api_arg} ${rates_arg}
            "

            mkdir -p "${LOCAL_RUN_DIR}/${mt}"
            remote_csv="$(vm_ssh "$vm" "$zone" "ls -td ${REPO_DIR}/results/guarded_sha_local_vol/* | head -1")"
            gcloud compute scp --recurse "${vm}:${remote_csv}" "${LOCAL_RUN_DIR}/${mt}/" \
                --project="$PROJECT" --zone="$zone" --quiet
            echo "[$vm] Copied outputs"
        ) &
        pids+=("$!")
        pid_mts+=("$mt")
    done

    wave_ok=true
    for idx in "${!pids[@]}"; do
        mt="${pid_mts[$idx]}"
        if ! wait "${pids[$idx]}"; then
            echo "Wave job failed for $mt"
            wave_ok=false
        fi
    done

    if [[ "$DELETE_VMS_AFTER" == true ]]; then
        for mt in "${wave[@]}"; do
            gcloud compute instances delete "$(vm_name_for "$mt")" \
                --project="$PROJECT" --zone="$(mt_zone "$mt")" --quiet || true
        done
    fi

    if [[ "$wave_ok" != true ]]; then
        exit 1
    fi
}

for mt in "${ALL_MTS[@]}"; do
    vcpus="$(mt_vcpus "$mt")"
    if (( current_wave_vcpus + vcpus > VCPU_BUDGET )); then
        run_wave
        wave=()
        current_wave_vcpus=0
    fi
    wave+=("$mt")
    current_wave_vcpus=$((current_wave_vcpus + vcpus))
done
run_wave

echo "Combining observations locally"
OBS_FILES=()
while IFS= read -r obs_file; do
    OBS_FILES+=("$obs_file")
done < <(find "$LOCAL_RUN_DIR" -name candidate_observations.csv -print)
if [[ ${#OBS_FILES[@]} -eq 0 ]]; then
    echo "ERROR: no candidate_observations.csv files found under $LOCAL_RUN_DIR"
    exit 1
fi
python3 experiments/run_guarded_sha_local_vol.py \
    --input-observations "${OBS_FILES[@]}" \
    --run-full-grid --run-plain-sha --run-guarded-sha \
    --instances "$(IFS=,; echo "${ALL_MTS[*]}")" \
    --full-budget "$FULL_BUDGET" \
    --probe-budgets "$PROBE_BUDGETS" \
    --repeats "$REPEATS" \
    --output-dir "${LOCAL_RUN_DIR}/combined"

echo "Combined outputs written under ${LOCAL_RUN_DIR}/combined"

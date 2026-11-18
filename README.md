# Monte Carlo + Automatic Differentiation Systematic Benchmarking Framework

**Imperial College London · MEng Computing (AI/ML) final-year research project**

Industrial collaboration: **HSBC Quant Equity Derivatives** · Supervisor: William Knottenbelt

A research and software engineering project exploring **how to make Monte Carlo option pricing faster and more cost-effective**. The framework compares Python, JAX, C++ and Rust implementations, measuring runtime, numerical accuracy, automatic differentiation (AD) overhead and estimated cloud costs through repeatable experiments.

**[View the deployed research dashboard →](https://mcad-pareto-research.pages.dev/#summary)**

Explore the research through a frontend presenting historical benchmark results, engine comparisons, scaling experiments and cloud measurements.

## Project at a glance

**My contribution:** Designed the modular system architecture and shared benchmarking interfaces, guided framework development, and evaluated performance and cost optimisation strategies. The work spans numerical computing, native-language integration, cloud experimentation and data analysis.

**Key result:** Selected **24 instead of 48 full-scale benchmark evaluations (50% fewer)** in the primary Google Cloud experiment while retaining all observed optimal runtime/cost trade-offs. Results varied across other cloud configurations.

**Tech stack:** Python · C++ · Rust · JAX/XLA · OpenMP · Google Cloud · SQLite · Flask · React · TypeScript

**Explore:** [Live dashboard](https://mcad-pareto-research.pages.dev/#summary) · [Architecture](#architecture-and-technology-stack) · [Research findings](#selected-research-findings) · [Full research report](Final_Year_Project_Report/main.pdf)

## Implementation highlights

- Four CPU engines: NumPy, JAX/XLA, C++/OpenMP via pybind11, and Rust/Rayon via PyO3; three option-pricing workloads with explicit engine/AD capability checks.
- Forward- and reverse-mode JAX AD, with European prices and sensitivities (Greeks) checked against Black-Scholes; multi-step local volatility uses `jax.lax.scan`.
- A shared benchmark runner with warmup, repeated timings, AD/no-AD baselines, configuration fingerprints and environment metadata; SQLite storage and structured exports.
- Google Cloud experiment orchestration, runtime/cost Pareto analysis, successive halving (SHA), and a modular guarded-SHA profiler with recorded promotion/elimination reasons.
- Flask API and React/TypeScript dashboard for submitting runs, comparing results and exploring history; automated tests cover the core pipeline and profiler policies.

Native engines require optional builds. AD support varies by engine; Rust local-volatility prices require further validation. CUDA is an optional experimental path. See [engine support](#engines) and [limitations](#research-report-and-limitations).

## Selected research findings

These are measurements from the evaluated workloads and environments, not general backend or cloud guarantees. `M` denotes simulated paths and `N` time steps.

| Finding | Scope and evidence |
|---|---|
| SHA selected **24 rather than 48 full-scale benchmark cells (50% fewer)**, recovering 100% of the observed full-grid Pareto frontier with 0% runtime/cost regret. | `sha_cloud_profiler_v4`, `n2-standard-4`, 16 workload/engine/AD candidates at three full path counts. [Report comparison table](Final_Year_Project_Report/evaluation/tables/profiler_comparison.tex), [cross-instance CSV](results/profiler_cross_instance.csv). This counts selected full-scale cells, not timed repetitions or an equivalent reduction in cloud expenditure; the evaluation also ran probes and the reference grid. |
| JAX reverse-mode AD overhead was **2.06–2.35×**, versus **4.08–6.00×** for forward mode. | Local CPU/Codespace LV-2 experiment: scalar-output local-volatility pricing, `N=252`, `M=5,000–250,000`. Ratios use separately measured no-AD baselines. [Report, “Local-Volatility AD Scaling with Path Count”](Final_Year_Project_Report/evaluation/evaluation.tex); [experiment driver](experiments/run_localvol_suite.py). |
| The cross-instance audit exposed selection failures: SHA recovered only **33.3–66.7%** of the Pareto frontier on the other five evaluated instance types. | The same `sha_cloud_profiler_v4` audit, contrasting with 100% on `n2-standard-4`. This motivated safeguards for near ties, engine/hardware diversity and changing probe ranks. [Audit data](results/profiler_cross_instance.csv), [report table](Final_Year_Project_Report/evaluation/tables/profiler_cross_instance.tex). |

The later guarded local-volatility study separates price-only and AD-required tasks and evaluates weighted runtime/cost objectives against an observed full-grid oracle. Its [committed report](results/guarded_sha_local_vol_gcp/20260611_152107/combined_clean/20260611_153812/guarded_sha_report.md) retains Rust performance candidates; numerical validity must be assessed separately.

## Architecture and technology stack

Configuration → engine → benchmark runner → result → storage, with experiment drivers coordinating sweeps and the API serving the dashboard. Engine interfaces separate numerical kernels from timing and persistence; the guarded profiler separates candidates, objectives, guards, execution and reporting.

```text
benchmarking/
├── core/          # Registered workload dataclasses, engine interface, result model
├── workloads/     # NumPy/JAX engines, native wrappers, CUDA path, AD validation
├── cpp/           # C++17/OpenMP kernels + pybind11 bindings
├── rust/          # Rayon kernels + PyO3/maturin extension
├── runner/        # Warmup, repeated timings, baseline and environment capture
├── storage/       # SQLite persistence, WAL mode, additive schema migration
├── cloud/         # GCP metadata and hourly pricing / cost calculations
├── analysis/      # Runtime/cost Pareto analysis
├── profiling/guarded_sha/  # Modular local-volatility selection policies
└── api/           # Flask endpoints + sequential background worker
experiments/       # Baseline, AD, scalability, local-volatility and profiler CLIs
scripts/           # GCP VM orchestration and SQLite merging
frontend/          # React, TypeScript, Vite, Material UI, Recharts
results/           # Committed CSV/JSON/Markdown evidence; generated DBs
Final_Year_Project_Report/  # Report PDF, LaTeX sources, tables and figures
tests/             # System, AD and guarded-profiler tests
```

The existing [Mermaid class diagram source](mermaid_system_diagram.txt) illustrates the core data flow; it predates the Rust and profiler additions.

## Quick start

Use Python 3.10+ and run commands from the repository root. The frontend additionally requires Node.js/npm; native engines have separate toolchain requirements.

```bash
git clone https://github.com/Vivian-Lopez/Monte-Carlo-and-Automatic-Differentiation-Systematic-Benchmarking-Framework.git
cd Monte-Carlo-and-Automatic-Differentiation-Systematic-Benchmarking-Framework
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
python -m pip install -r requirements.txt
python experiments/run_european_baseline.py --runs 5 --warmup 2
```

Optional C++ build (requires a C++17 compiler supporting the configured `-fopenmp` flags; Apple Clang needs a separate OpenMP/toolchain setup):

```bash
python -m pip install setuptools wheel
python -m pip install -e benchmarking/cpp/ --no-build-isolation
```

Optional Rust build (requires Rust/Cargo and an active Python virtual environment):

```bash
python -m pip install maturin
(cd benchmarking/rust && maturin develop --release)
```

Missing native extensions do not prevent NumPy/JAX experiments. Availability handling differs by driver: some register a wrapper and report failed native runs when the compiled module is absent.

## Running experiments

```bash
# European GBM engine comparison, JAX AD/accuracy, and path-count scaling
python experiments/run_european_baseline.py --runs 10 --warmup 3
python experiments/run_european_ad_analysis.py --runs 10 --warmup 3
python experiments/run_european_scalability.py --runs 5 --warmup 2

# Local-volatility smoke test, engine scaling and AD scaling
python experiments/run_localvol_suite.py --experiments LV0 LV1 LV2

# Profiler versus full grid: tiny real-workload pipeline check
python experiments/run_profiler_vs_grid.py --dry-run

# Guarded local-volatility profiler: synthetic observations, no cloud deployment
python experiments/run_guarded_sha_local_vol.py --dry-run
```

The local-volatility suite covers LV0–LV10: engine/AD scaling, time steps, memory pressure, strong/weak thread scaling, surface presets, timing stability and stress limits. It uses per-experiment repetition counts; unlike the three European drivers, it does not accept `--runs` or `--warmup`. Large AD/stress cells can exhaust memory.

`run_profiler_vs_grid.py` compares a full grid, the original probe-based selector and SHA, with SQLite phase labels and optional CSV exports (`--export`). Its dry run executes small benchmarks and writes a summary; the guarded profiler's dry run uses synthetic timings and writes timestamped artefacts. The guarded study measures an observation pool and replays selection policies against it; selected-cell savings do not mean the evaluation itself avoided those measurements.

For cloud execution, `scripts/run_cloud_profiler.sh` orchestrates VMs, retrieves/merges databases and exports results; `scripts/run_guarded_sha_gcp.sh` gathers per-instance local-volatility observations and combines the analysis. Both require an authenticated `gcloud` setup, your project ID, billing and quota. Inspect their `--help` and defaults, including region, machine types and repository branch, before use. [Cloud reproduction notes](docs/cloud_profiler.md) describe the original workflow but contain historical account-specific values and estimates; use the scripts for current options.

## Flask API + frontend

```bash
# Terminal 1: active Python environment, repository root
python -m benchmarking.api.server        # http://localhost:5050

# Terminal 2: repository root
cd frontend
npm ci
npm run dev                             # http://localhost:5173
# npm run build                         # TypeScript check + Vite build
```

Vite proxies `/api` to port 5050. The dashboard provides simulation, comparison, history, summary and analysis pages. `POST /api/runs` queues a run; poll `GET /api/runs/<id>` for `pending → running → completed/failed`. Other endpoints expose workloads, engine capabilities, run lists, summaries and the comparison matrix.

The simulation form expects a `schema` for each workload, but `/api/workloads` currently returns only labels and identifiers; configurable form fields therefore remain incomplete. Frontend build/browser behaviour was not verified in this documentation update.

The API uses a single sequential daemon worker. Its engine registry includes NumPy/JAX, a C++ wrapper and CUDA when its startup probe succeeds; Rust is currently available through experiment drivers rather than the API. C++ wrapper registration alone does not prove that `cpp_mc` has been built.

## Querying results

The standard experiment drivers and API write to `results/benchmarks.db` (created on first use). Committed CSV exports are available in `results/`; the historical database is not checked in. The guarded local-volatility profiler writes separate CSV, JSON and Markdown artefacts.

```bash
sqlite3 results/benchmarks.db "
  SELECT engine, ad_mode, M,
         round(mean_runtime_ms, 3)        AS mean_ms,
         round(throughput_paths_per_sec)  AS paths_per_sec,
         round(rel_price_error, 6)        AS price_err
  FROM runs
  WHERE experiment_type = 'european_baseline'
  ORDER BY engine, M;"
```

### `runs` table — key columns

| Group | Columns |
|---|---|
| Identity | `id`, `experiment_id`, `experiment_type`, `config_hash` |
| Workload | `workload_type`, `M`, `N`, `seed`, `config_json` |
| Engine | `engine`, `language`, `backend` |
| Timing | `mean_runtime_ms`, `std_runtime_ms`, `min_runtime_ms`, `max_runtime_ms`, `throughput_paths_per_sec` |
| Price | `result_value`, `analytical_price`, `abs_price_error`, `rel_price_error` |
| Greeks | `greek_delta`, `greek_vega`, `greek_rho`, `analytical_delta`, `analytical_vega`, `analytical_rho`, `abs_delta_error`, `rel_delta_error`, … |
| AD | `ad_mode`, `ad_overhead_ratio`, `baseline_mean_ms` |
| Resources | `memory_peak_mb` |
| Environment | `cpu_model`, `cpu_count`, `python_version`, `numpy_version`, `jax_version` |
| Threads | `num_threads`, `requested_threads`, `observed_threads_max`, `env_omp_num_threads`, `env_xla_flags` |
| Cloud | `instance_type`, `region`, `zone`, `cost_per_run`, `paths_per_dollar` |
| Profiler | `profiler_phase`, `profiler_decision`, `profiler_reason`, `git_commit_hash` |

---

## Workloads

### 1. European option (GBM)

Plain vanilla European call or put under Geometric Brownian Motion. Serves as the reference workload for all engine and AD comparisons, with Black-Scholes providing an exact analytical benchmark.

$$S_T = S_0 \exp\!\left[\left(r - \tfrac{1}{2}\sigma^2\right)T + \sigma\sqrt{T}\,Z\right], \quad Z \sim \mathcal{N}(0,1)$$

Single-step exact simulation. `N` defaults to 1.

```python
from benchmarking.core.config import EuropeanOptionConfig

config = EuropeanOptionConfig(
    S0=100.0, K=100.0, r=0.05, sigma=0.20,
    T=1.0, option_type="call", M=10_000, seed=42,
)
```

### 2. European option — local volatility

European call or put under a 4-parameter parametric local volatility model, simulated with log-Euler discretisation over `N` time steps. The primary workload for AD benchmarking and Greek computation.

**Dynamics:**

$$dS_t = r\,S_t\,dt + \sigma(S_t, t;\theta)\,S_t\,dW_t$$

**Log-Euler step:**

$$S_{n+1} = S_n \exp\!\left[\left(r - \tfrac{1}{2}\sigma_n^2\right)\Delta t + \sigma_n\sqrt{\Delta t}\,Z_n\right]$$

**Local volatility surface:**

$$x_n = \ln(S_n/S_0), \qquad \text{raw}_n = a_0 + a_1 x_n + a_2 x_n^2 + b_1 t_n$$

$$\sigma_n = \sigma_{\min} + \mathrm{softplus}(\text{raw}_n), \qquad \mathrm{softplus}(z) = \max(z,0) + \ln(1+e^{-|z|})$$

```python
from benchmarking.core.config import EuropeanLocalVolConfig

config = EuropeanLocalVolConfig(
    S0=100.0, K=100.0, r=0.05, T=1.0,
    M=100_000, N=252,
    sigma_min=0.01,
    theta=[-1.564, -0.10, 0.20, 0.00],  # [a0, a1, a2, b1]
    option_type="call",
    seed=42,
)
```

The default intercept `a0` satisfies `sigma_min + softplus(a0) = 0.20` at the initial spot/time; the default nonzero slope and curvature produce a non-flat surface. Setting `a1 = a2 = b1 = 0` gives constant volatility and a Black-Scholes price reference (subject to Monte Carlo sampling error).

**AD targets (JAX engine):** Delta (∂P/∂S₀), and all four theta sensitivities (∂P/∂a₀, ∂P/∂a₁, ∂P/∂a₂, ∂P/∂b₁), and ∂P/∂σ_min.

---

### 3. Arithmetic-average Asian option (GBM)

`AsianOptionConfig` in `benchmarking/core/config.py` defines a call/put on the arithmetic average of simulated prices over `N` steps (defaults: `M=100_000`, `N=252`). NumPy and JAX implement this path-dependent workload; JAX also computes Delta, Vega and Rho. It has no Black-Scholes analytical reference in the framework.

---

## Engines

| Engine | Key | Workloads | AD modes | Execution / status |
|---|---|---|---|---|
| NumPy | `cpu` | European GBM, local volatility, Asian | `none` | Vectorised CPU reference |
| JAX/XLA | `jax` | European GBM, local volatility, Asian | `none`, `forward`, `reverse` | JIT kernels; `lax.scan` for multi-step simulation; float64 enabled |
| C++/OpenMP | `cpp` | European GBM, local volatility | `none` | Optional pybind11 extension; OpenMP path parallelism |
| Rust/Rayon | `rust` | European GBM, local volatility | `none` | Optional PyO3 extension; local-volatility pricing discrepancy remains unresolved |
| CUDA/PyCUDA | `cuda` | European GBM only | `none`, `forward` | Experimental NVIDIA path; forward path computes Delta only; requires CUDA/PyCUDA, outside the final cloud comparison |

### Greeks — JAX engine

European GBM and Asian options expose Delta (∂P/∂S₀), Vega (∂P/∂σ) and Rho (∂P/∂r). European values have Black-Scholes validation; Asian values do not have an analytical reference here.

Local volatility exposes Delta and sensitivities to `a0`, `a1`, `a2`, `b1` and `sigma_min`. Reverse mode uses one `jax.grad` call for all target arguments; forward mode uses one `jax.jvp` per input direction. The measured AD runtime includes pricing and sensitivity computation, with a separately timed no-AD baseline for the overhead ratio.

## Testing and reproducibility

```bash
python -m pytest --collect-only -q
python -m pytest -q
```

The current suite collects **64 tests across five test modules**. All 64 passed during README verification in a temporary Python 3.13 environment with NumPy 2.5.4 and JAX 0.12.0.1 (CPU), without native or CUDA builds. This does not verify those optional backends.

Tests cover configuration serialisation/hashing, European pricing and JAX Greeks, timing statistics, result serialisation, SQLite round-trips, API contracts, the end-to-end pipeline, guarded selection rules, objective/regret calculations and structured dry-run outputs. They do not establish correctness of every native/GPU workload.

- Warmup runs precede timing to amortise JAX compilation; reported results describe warm execution, not complete cold-start job latency.
- Fixed seeds support repeatability within a fixed backend/environment. Different RNGs and thread assignments prevent identical paths across engines; Rust/Rayon stream reproducibility needs further audit.
- `config_hash` is an eight-character SHA-256 prefix over the serialised workload. It excludes engine, AD mode and thread settings, so compare it alongside execution metadata.
- `BenchmarkResult` retains individual timings and can be saved as JSON. The SQLite schema stores timing summaries, configuration, Greeks and metadata; it does not store every raw timing sample.
- Cloud drivers record additional instance, pricing and git metadata. Estimated cost per run is runtime × hourly rate / 3,600,000 (runtime in ms); it excludes provisioning, idle time and other invoice charges. Local synthetic cost proxies are not cloud prices.
- Dependency minimums are specified, rather than a fully pinned Python environment. Record package versions, compiler/toolchain, hardware and thread settings when reproducing historical results. `memory_peak_mb` is sampled RSS growth above a baseline, not a precise allocator peak.

## Adding a workload or engine

Register a dataclass in `benchmarking/core/config.py` with `@workload("name")`, define fields/defaults, then implement it in the desired engines. API workload discovery follows the registry; configurable frontend fields require a separate schema integration. Update `supports()`/dispatch and declare supported AD modes. The generic runner and storage can then carry the new configuration without workload-specific changes.

For a new engine, implement `MonteCarloEngine.run(config, ad_mode) → (price, greeks_or_none)`, `supports()` and `supported_ad_modes()`, then add it to the relevant experiment/API registries. Add numerical checks appropriate to its workload; registration alone does not validate a pricing backend.

## Research report and limitations

Read the [final-year report (PDF)](Final_Year_Project_Report/main.pdf) for methodology and discussion; [evaluation sources](Final_Year_Project_Report/evaluation/evaluation.tex) and [committed results](results/) provide traceable evidence. The report's implementation chapter contains historical paths and a stale test count; the architecture above reflects the current source tree.

- Rust local-volatility prices disagree with an independent high-path-count NumPy reference beyond the reported sampling uncertainty. Its timings remain performance evidence; its pricing/RNG streams require investigation before treating it as a validated backend.
- Local-volatility Greeks have no closed-form validation in this study. Common-random-number finite differences remain a useful next check. Profiler validity checks reject execution failures, non-finite prices and missing required Greeks; they do not resolve the Rust discrepancy.
- Reverse-mode JAX encountered out-of-memory failures at large path/time-step combinations. XLA manages its own threading, so requested OpenMP/Rayon thread counts are not equivalent JAX controls.
- Measurements combine RNG, path evolution and reduction costs and depend on the tested environment. The cross-instance audit demonstrates that cheap-probe selection can miss full-scale winners; guarded selection is evaluated against a finite observed oracle.
- CUDA is implemented but not part of the final measured CPU/cloud comparison. Mojo, native-engine AD, mixed precision/operator-fusion studies and broader GPU validation remain future work. The API/dashboard is a research interface, with profiler configuration/report generation still driven by scripts.

# Monte Carlo + Automatic Differentiation Systematic Benchmarking Framework

**Imperial College London – Department of Computing · Final Year Project**  
**Supervisor:** William Knottenbelt · **Industrial Partner:** HSBC

A modular, extensible framework for systematically benchmarking Monte Carlo simulation engines with automatic differentiation (AD) support. The primary goal is rigorous, reproducible comparison of computational efficiency, scalability, and resource utilisation across hardware and software environments — targeting high-performance scientific computing for derivative pricing.

---

## Research objectives

- **Cross-engine comparison** — evaluate NumPy (CPU), JAX (XLA-compiled), and C++ (OpenMP) on identical workloads with identical seeds
- **AD analysis** — measure the overhead of forward vs. reverse-mode AD and validate computed Greeks against Black-Scholes analytical solutions
- **Scalability** — characterise throughput scaling as path count M grows from 1k to 100k
- **Reproducibility** — every result is fully traceable: config hash, seed, engine version, platform metadata, and all timing samples are stored in SQLite

Planned extensions: CPU architecture comparison (AMD vs Intel, SIMD backends), GPU/CUDA acceleration, Rust and Mojo implementations, cloud resource profiling on Google Cloud, and mixed-precision / operator-fusion techniques drawn from the ML community.

---

## Setup

```bash
git clone https://github.com/Vivian-Lopez/Monte-Carlo-and-Automatic-Differentiation-Systematic-Benchmarking-Framework.git
cd Monte-Carlo-and-Automatic-Differentiation-Systematic-Benchmarking-Framework

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Optional — build the C++ OpenMP engine:**
```bash
pip install -e benchmarking/cpp/
```
The C++ engine is silently skipped in all scripts if not built.

---

## Running experiments

Three ready-made experiment scripts cover the current research surface:

```bash
# Engine comparison: NumPy vs JAX vs C++ at M=10k, no AD
python experiments/run_european_baseline.py

# AD analysis: JAX forward vs reverse overhead, Greek accuracy, M sweep
python experiments/run_european_ad_analysis.py

# Scalability: all engines × M = 1k, 5k, 10k, 50k, 100k
python experiments/run_european_scalability.py
```

All scripts accept `--runs N --warmup N`:
```bash
python experiments/run_european_baseline.py --runs 10 --warmup 3
```

Results are printed to the terminal and written to `results/benchmarks.db`.

---

## Querying results

SQLite is the single source of truth. All results are queryable immediately after a run.

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

---

## Project structure

```
benchmarking/
├── core/
│   ├── config.py       ← WorkloadConfig base class, @workload decorator, WORKLOAD_REGISTRY
│   ├── engine.py       ← MonteCarloEngine abstract base class
│   └── result.py       ← BenchmarkResult dataclass (timings, Greeks, metadata)
├── workloads/
│   ├── mc_cpu.py       ← NumPy engine + Black-Scholes analytical functions
│   ├── mc_jax.py       ← JAX JIT engine + forward/reverse AD
│   └── mc_cpp.py       ← C++ OpenMP engine (via pybind11)
├── runner/
│   └── runner.py       ← BenchmarkRunner: warmup, timed loop, env capture
├── storage/
│   └── database.py     ← BenchmarkDB: SQLite, WAL mode, auto-migration
└── api/
    └── server.py       ← Flask REST API

experiments/
├── run_european_baseline.py     ← start here; template for new scripts
├── run_european_ad_analysis.py
└── run_european_scalability.py

tests/
├── test_system.py          ← end-to-end: config → engine → runner → DB → API
└── test_ad_framework.py    ← AD correctness and Greek accuracy

frontend/                   ← React + TypeScript result dashboard
results/
└── benchmarks.db           ← SQLite result store (created on first run)
```

---

## Current workload: European option

The sole active workload is a plain vanilla European call or put under Geometric Brownian Motion (GBM). It serves as the reference workload for all engine and AD comparisons, with Black-Scholes providing an exact analytical benchmark.

### Pricing model

$$S_T = S_0 \exp\!\left[\left(r - \tfrac{1}{2}\sigma^2\right)T + \sigma\sqrt{T}\,Z\right], \quad Z \sim \mathcal{N}(0,1)$$

$$P = e^{-rT}\,\mathbb{E}\!\left[\max(S_T - K,\, 0)\right]$$

Single-step exact simulation (no Euler discretisation). `N` time steps is exposed as a config field for future path-dependent extensions; it defaults to 1.

### Configuration

```python
from benchmarking.core.config import EuropeanOptionConfig

config = EuropeanOptionConfig(
    S0=100.0,        # initial spot price
    K=100.0,         # strike
    r=0.05,          # continuously-compounded risk-free rate
    sigma=0.20,      # annualised volatility
    T=1.0,           # time to maturity in years
    option_type="call",  # "call" or "put"
    M=10000,         # number of Monte Carlo paths
    seed=42,         # RNG seed — fixes random paths for reproducibility
)
```

All fields have defaults; `EuropeanOptionConfig()` with no arguments is valid.

### Engines

| Engine | Key | AD modes | Notes |
|---|---|---|---|
| NumPy (CPU) | `cpu` | `none` | Reference; uses `np.random.default_rng(seed)` |
| JAX (XLA) | `jax` | `none`, `forward`, `reverse` | Module-level `@jax.jit` kernel; warmup excludes XLA compile time |
| C++ OpenMP | `cpp` | `none` | pybind11 binding; build separately |

### Greeks (JAX engine)

The JAX engine computes Delta (∂P/∂S₀), Vega (∂P/∂σ), and Rho (∂P/∂r):

- **Reverse mode** — one `jax.grad(argnums=(0,1,2))` call; single backward pass returning all three partials simultaneously
- **Forward mode** — three `jax.jvp` calls, one per parameter

Both are validated against Black-Scholes analytical Greeks. Relative errors are stored in the DB for every run.

---

## Adding a new workload

Two files. Nothing else changes.

**1. `benchmarking/core/config.py`** — add a decorated class:

```python
@workload("digital")
class DigitalOptionConfig(WorkloadConfig):
    S0:      float = 100.0
    barrier: float = 110.0
    M:       int   = 10000
    seed:    int   = 42
```

`@workload("digital")` applies `@dataclass`, sets `workload_type`, and registers the class — no other boilerplate. Override `validate()` only if you need cross-field constraints (e.g. `barrier > S0`).

**2. `benchmarking/workloads/mc_cpu.py` and/or `mc_jax.py`** — add the type to `SUPPORTED`, a branch in `run()`, and a pricing method.

The runner, storage, and API pick up the new workload automatically.

---

## Design principles

- **Warmup before timing** — JIT compilation (JAX/XLA: ~115 ms cold, ~0.6 ms warm at M=10k) is always excluded from measurements
- **Fixed seeds** — same `seed` in config produces bit-identical prices and paths across runs and engines
- **`config_hash`** — SHA-256 fingerprint of the sorted serialised config; stable across machines; groups repeated runs of the same experiment
- **Single backward pass** — reverse-mode AD uses `jax.grad(argnums=(0,1,2))`, not three separate calls; overhead ratio is measured against an identical no-AD baseline
- **Auto-migration** — new DB columns are added with `ALTER TABLE` at startup; existing databases are never invalidated

---

## Running tests

```bash
pytest -q
```

55 tests covering config serialisation, engine correctness, runner statistics, DB round-trips, and all API endpoints.

---

## Flask API + frontend

```bash
# Terminal 1
python -m benchmarking.api.server        # listens on http://localhost:5050

# Terminal 2
cd frontend && npm install && npm run dev  # opens http://localhost:5173
```

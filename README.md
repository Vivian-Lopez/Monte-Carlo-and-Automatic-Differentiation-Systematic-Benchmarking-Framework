# Systematic Monte Carlo Benchmarking Engine

**Imperial College London – Department of Computing · Final Year Project**  
**Supervisor:** William Knottenbelt · **Industrial Partner:** HSBC

A modular framework for benchmarking Monte Carlo option pricing engines (NumPy, JAX, C++) with automatic differentiation support. Results are stored in SQLite with full reproducibility metadata.

---

## Setup (from clone)

```bash
git clone https://github.com/Vivian-Lopez/Systematic_Monte_Carlo_Benchmarking_Engine.git
cd Systematic_Monte_Carlo_Benchmarking_Engine

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**Optional — build the C++ OpenMP engine:**
```bash
pip install -e benchmarking/cpp/
```
If not built, the C++ engine is silently skipped in all experiment scripts.

---

## Running experiments

```bash
# Compare NumPy, JAX, C++ on European call (ad_mode=none)
python experiments/run_european_baseline.py

# JAX AD sweep: none / forward / reverse over M = 1k–100k paths
python experiments/run_european_ad_analysis.py

# Scalability: all engines × M = 1k–100k paths
python experiments/run_european_scalability.py
```

All scripts accept `--runs N --warmup N` to override the default repetition counts:
```bash
python experiments/run_european_baseline.py --runs 5 --warmup 2
```

Results print to the terminal and are written to `results/benchmarks.db`.

---

## Querying results

SQLite is the single source of truth.

```bash
sqlite3 results/benchmarks.db "
  SELECT engine, ad_mode, M,
         mean_runtime_ms, result_value, rel_price_error,
         throughput_paths_per_sec
  FROM runs
  WHERE experiment_type = 'european_baseline'
  ORDER BY engine;"
```

Key columns in the `runs` table:

| Group | Columns |
|---|---|
| Identity | `id`, `experiment_id`, `experiment_type`, `config_hash` |
| Config | `workload_type`, `M`, `N`, `seed`, `config_json` |
| Engine | `engine`, `language`, `backend` |
| Timing | `mean_runtime_ms`, `std_runtime_ms`, `min_runtime_ms`, `max_runtime_ms`, `throughput_paths_per_sec` |
| Price | `result_value`, `analytical_price`, `abs_price_error`, `rel_price_error` |
| Greeks | `greek_delta`, `greek_vega`, `greek_rho`, `analytical_delta/vega/rho`, `abs/rel_*_error` |
| AD | `ad_mode`, `ad_overhead_ratio`, `baseline_mean_ms` |
| Resources | `memory_peak_mb` |
| Environment | `cpu_model`, `cpu_count`, `python_version`, `numpy_version`, `jax_version` |

---

## Project structure

```
benchmarking/
├── core/
│   ├── config.py       ← WorkloadConfig base + all 4 option config classes + WORKLOAD_REGISTRY
│   ├── engine.py       ← MonteCarloEngine abstract base class
│   └── result.py       ← BenchmarkResult dataclass
├── workloads/
│   ├── mc_cpu.py       ← NumPy engine + Black-Scholes analytical functions
│   └── mc_jax.py       ← JAX JIT engine + forward/reverse AD
├── cpp/                ← C++ OpenMP engine (pybind11); build with pip install -e benchmarking/cpp/
├── runner/
│   └── runner.py       ← BenchmarkRunner: warmup, timed loop, memory, env capture
├── storage/
│   └── database.py     ← BenchmarkDB: SQLite, 58-column schema, auto-migration
└── api/
    └── server.py       ← Flask REST API (used by frontend)

experiments/
├── run_european_baseline.py     ← reference script — read this first
├── run_european_ad_analysis.py
└── run_european_scalability.py

frontend/               ← React + TypeScript dashboard (reads from Flask API)
results/
└── benchmarks.db       ← SQLite result store (created on first run)
tests/
├── test_system.py
└── test_ad_framework.py
```

---

## Experiment scripts

| Script | Purpose | Reference for |
|---|---|---|
| `run_european_baseline.py` | Engine comparison at fixed M=10k, ad_mode=none | **Start here.** Template for any new experiment script. |
| `run_european_ad_analysis.py` | JAX AD overhead and Greek accuracy over M sweep | AD overhead ratios, forward vs reverse comparison |
| `run_european_scalability.py` | All engines × 5 path counts (1k–100k) | Throughput and scaling behaviour |

---

## Adding a new workload

Three files to touch — nothing else changes.

**1. `benchmarking/core/config.py`** — add a `@dataclass` subclassing `WorkloadConfig` (`workload_type` property, `SCHEMA`, `validate()`, `to_dict()`, `from_dict()`), then add one line to `WORKLOAD_REGISTRY` at the bottom.

**2. `benchmarking/workloads/mc_cpu.py`** — add the type string to `SUPPORTED`, a branch in `run()`, and a `_price_<type>()` method.

**3. `benchmarking/workloads/mc_jax.py`** — same pattern: `SUPPORTED`, `run()` branch, `_run_<type>()` method with a `price_fn(S0, r, sigma)` closure passed to `_compute_greeks()` for AD.

The runner, storage, API, and frontend pick up new workloads automatically.

---

## Running the Flask API + frontend

```bash
# Terminal 1 — API server
python -m benchmarking.api.server

# Terminal 2 — frontend dev server
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The API listens on `http://localhost:5050`.

---

## Running tests

```bash
pytest -q
```

---

## Active scope

| Workload | Status | Engines |
|---|---|---|
| European (call/put) | **Active** | NumPy, JAX, C++ |

---

## Design principles

- **Warmup before timing** — JIT compilation (JAX/XLA: ~115 ms cold vs ~0.6 ms warm at M=10k) is excluded from all measurements
- **Fixed seeds** — same config always produces bit-identical prices
- **`config_hash`** — SHA-256 fingerprint of sorted JSON config; stable across machines; used to group repeated runs
- **Single backward pass** — reverse-mode AD uses `jax.grad(argnums=(0,1,2))`, not three separate calls
- **No JSON middleman** — `store_run_full()` writes all 58 fields atomically to SQLite in one shot

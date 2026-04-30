# Systematic Monte Carlo Benchmarking Engine

**Imperial College London – Department of Computing**  
**Final Year Project**  
**Supervisor:** William Knottenbelt  
**Industrial Partner:** HSBC

---

## MVP Scope (current)

This project is currently stabilised around a **European option only** pipeline.  
The goal is correctness, reproducibility, and a queryable SQLite result store before
reintroducing additional workloads and hardware targets.

**Active workloads:** European option (plain vanilla call/put, GBM, single-step exact)  
**Dormant workloads:** Asian, Barrier, Basket — code preserved, excluded from default experiments  
**Active engines:** NumPy/CPU · JAX (XLA) · C++ OpenMP (if built)  
**AD modes:** none · forward (`jax.jvp`) · reverse (`jax.grad`, single backward pass)  
**Storage:** SQLite is canonical — no JSON middleman  
**Analysis:** CLI scripts produce terminal tables and store to SQLite

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run European baseline (NumPy, JAX, C++ vs European call)
python experiments/run_european_baseline.py

# 3. Run AD analysis (JAX none / forward / reverse, M = 1k–100k)
python experiments/run_european_ad_analysis.py

# 4. Run scalability sweep (M = 1k–100k, all active engines)
python experiments/run_european_scalability.py
```

All results are written to `results/benchmarks.db` (SQLite).

---

## Result Storage

SQLite is the **single source of truth**.  
The `runs` table contains:
- workload/engine/config identity and `config_hash`
- runtime statistics (mean, std, min, max, throughput paths/s)
- computed price and Greeks (delta, vega, rho)
- analytical Black-Scholes reference (price, delta, vega, rho)
- absolute and relative errors for price and each Greek
- peak memory usage (`memory_peak_mb`)
- environment metadata (CPU, Python/NumPy/JAX versions, …)
- AD overhead ratio and baseline runtime
- nullable cloud fields for future Google Cloud experiments

Query the database directly:
```bash
sqlite3 results/benchmarks.db \
  "SELECT engine, ad_mode, mean_runtime_ms, result_value, rel_price_error \
   FROM runs WHERE experiment_type='european_baseline';"
```

---

## Architecture

```
EuropeanOptionConfig
  → BenchmarkRunner (warmup + timed repetitions)
      → Engine.run(config, ad_mode)  [NumPy | JAX | C++]
  → BenchmarkResult (price, Greeks, runtimes, memory)
  → BenchmarkDB.store_run_full(...)  [SQLite]
  → CLI terminal table / Flask API / React UI (all read-only)
```

**Key design principles:**
- Python coordinates; engines compute
- Warmup runs exclude JIT compilation from timing
- Fixed seeds → deterministic reproducibility
- `config_hash` fingerprints every run for integrity checking
- `experiment_id` groups related runs for analysis

---

## Project Structure

```
benchmarking/
├── api/             # Flask REST API (read-only UI backend)
├── core/            # Config, engine abstraction, BenchmarkResult
├── cpp/             # C++ OpenMP backend (pybind11)
├── runner/          # BenchmarkRunner (timing, memory, env capture)
├── storage/         # SQLite BenchmarkDB
└── workloads/       # Engine implementations (cpu, jax, cpp, cuda*)

experiments/
├── run_european_baseline.py      ← START HERE
├── run_european_ad_analysis.py
├── run_european_scalability.py
└── (legacy scripts preserved but not part of default run)

results/
└── benchmarks.db    ← SQLite result store

frontend/            # React dashboard (reads from API, not JSON files)
```

---

## Dormant Workloads

The following workloads are implemented in the engine code but **excluded from the
default experiment matrices** until the European pipeline is fully validated:

| Workload | Status | How to re-enable |
|---|---|---|
| Asian option (arithmetic/geometric) | Dormant | Add `AsianOptionConfig` to experiment scripts |
| Barrier option (knock-in/knock-out) | Dormant | Add `BarrierOptionConfig` to experiment scripts |
| Basket option (correlated GBM) | Dormant | Add `BasketOptionConfig` to experiment scripts |
| CUDA engine | Dormant | GPU hardware + CUDA build required |
| Cloud experiments | Future | Google Cloud credentials + cost tagging |

No code was deleted. All engine methods remain intact.

---

## Documentation

| File | Purpose |
|---|---|
| [METHODOLOGY.md](METHODOLOGY.md) | Timing/AD/error metric methodology |
| [SUITE.md](SUITE.md) | Full intended benchmark suite with active/dormant labels |

---

## Building the C++ Engine (optional)

```bash
cd benchmarking/cpp
pip install -e .
```

If not built, the C++ engine is silently skipped in experiment scripts.

---

## Overview

A modular, extensible benchmarking framework for evaluating Monte Carlo (MC) +
Automatic Differentiation (AD) simulation engines across implementations, languages,
and hardware targets.

**Key insight:** This project is a **benchmarking system for computational finance
infrastructure**, designed to answer:

> *How do different implementations perform under identical workloads?*

---

## License

MIT License

---

## Acknowledgements

Imperial College London · Department of Computing · HSBC Quantitative Equities Department

The system is designed with **clean separation between orchestration and compute**, enabling fair, reproducible comparisons between:

- Python (NumPy baseline)
- C++ (OpenMP CPU backend)
- JAX (vectorised engine with AD support for European workloads)
- Future backends such as CUDA and Rust

The framework supports multiple option pricing workloads and is built to scale toward **quantitative research and production benchmarking environments**.

---

## Core Features

- **Workload abstraction**
  - European, Asian, Barrier, Basket options
  - Strong validation + schema-driven configs

- **Pluggable engine architecture**
  - Swap compute backends without changing orchestration
  - Python, JAX, and C++ engines currently supported

- **Benchmark runner**
  - Warmup + repeated runs
  - Runtime statistics (mean, std, min, max)
  - Deterministic reproducibility via seeds + config hashing

- **API + Frontend**
  - Flask backend for execution
  - React (Vite + MUI) dashboard for interaction and visualization

- **Structured output**
  - JSON results with full metadata
  - SQLite-backed run history for API-driven workflows
  - Designed for later extension to Parquet / distributed storage

---

## Contact

**Author:** Vivian Lopez  
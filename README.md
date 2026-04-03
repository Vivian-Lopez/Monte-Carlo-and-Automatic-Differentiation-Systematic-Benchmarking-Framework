# Systematic Monte Carlo Benchmarking Engine

**Imperial College London – Department of Computing**  
**Final Year Project**  
**Supervisor:** William Knottenbelt  
**Industrial Partner:** HSBC  

---

## Overview

A modular, extensible benchmarking framework for evaluating Monte Carlo (MC) + Automatic Differentiation (AD) simulation engines across different implementations, languages, and hardware targets.

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

## Architecture

```
Python Layer (Orchestration)
│
├── WorkloadConfig (typed configs + validation)
├── BenchmarkRunner (execution + timing)
├── API (Flask)
│
├── Engines (pluggable)
│   ├── CPU (NumPy)
│   ├── JAX
│   ├── C++ (OpenMP via pybind11)
│   └── (future: CUDA, Rust)
│
└── Frontend (React dashboard)
```

**Key design principle:**  
> Python coordinates experiments. Engines do computation.

---

## Project Structure

```
benchmarking/
├── api/             # Flask server + background execution worker
├── core/            # Config system, engine abstractions, benchmark results
├── cpp/             # C++ backend source + pybind11 build
├── runner/          # Benchmark execution and timing logic
├── storage/         # SQLite persistence layer
└── workloads/       # CPU, JAX, C++ engines + AD validation helpers

frontend/
├── src/
│   ├── api/         # Frontend API client
│   ├── components/  # Dashboard UI components
│   └── pages/       # Simulate, history, and summary views
└── ...              # Vite / TypeScript app config

experiments/
├── generate_runs.py
├── run_ad_experiment.py
└── run_experiment.py

examples/
└── engine_abstraction.py

results/             # JSON outputs + SQLite benchmark database

tests/               # Unit tests
```

---

## Engines

### Python (NumPy)
- Reference implementation
- Deterministic, easy to debug
- Baseline for correctness

### C++ (OpenMP)
- Multi-threaded CPU implementation
- Integrated via `pybind11`
- Currently supports European workloads

### JAX
- Vectorised Monte Carlo engine
- Supports the same workload registry as the CPU engine
- Forward / reverse AD currently wired for European workloads

### Planned
- CUDA kernels
- Rust backend

---

## Workloads

Each workload is defined as a `WorkloadConfig`:

| Workload | Description |
|--------|------------|
| European | Closed-form comparable (Black-Scholes) |
| Asian | Path-dependent averaging |
| Barrier | Knock-in / knock-out |
| Basket | Multi-asset correlated simulation |

Adding new workloads requires:
1. New config subclass
2. Engine implementation
3. Registry entry

---

## API

Flask server exposes endpoints for:

- Listing available workloads and their schemas
- Listing available engines and supported workloads
- Submitting benchmark runs
- Polling run status and history
- Returning aggregate summary metrics

Runs are queued through a lightweight background worker and persisted in SQLite under `results/benchmarks.db`.

Default:
```
http://localhost:5050
```

---

## Frontend

- Built with **React + Vite + Material UI**
- Provides:
  - Parameter input forms (driven by config schema)
  - Engine selection
  - Run submission and live backend connectivity status
  - Run history view
  - Summary view over recorded benchmarks

Default:
```
http://localhost:5173
```

---

## Design Principles

- **Separation of concerns**
  - Python = orchestration
  - Engines = compute

- **Extensibility**
  - New workloads and engines require minimal changes

- **Reproducibility**
  - Config hashing
  - Seed control
  - Structured outputs

- **Comparability**
  - Identical configs across engines
  - Standardised metrics

---

## Current Status

### Completed
- Multi-workload config system
- Python CPU engine (NumPy)
- C++ CPU engine (OpenMP)
- JAX engine
- Benchmark runner with metrics
- Flask API
- React frontend (MUI dashboard)
- SQLite-backed run storage
- JSON result pipeline
- AD validation utilities for European options

### In Progress / Planned
- Broader automatic differentiation benchmarking
- GPU backends (CUDA and expanded JAX acceleration)
- Cross-language benchmarking expansion
- Cloud/distributed execution
- Advanced profiling (hardware counters)

---

## Usage

See **Quick Start Guide** for setup and running instructions.

---

## Example Use Case

Benchmarking performance vs accuracy across engines:

- Compare Python vs C++ runtime scaling with paths (M)
- Validate Monte Carlo vs Black-Scholes
- Evaluate multi-threading speedup
- Extend to GPU acceleration

---

## Target Users

- Quantitative developers
- Performance engineers
- Research teams benchmarking simulation engines
- Financial institutions (e.g. equities / derivatives desks)

---

## Key Insight

This project is not just an MC pricer.

It is a **benchmarking system for computational finance infrastructure**, designed to answer:

> *How do different implementations perform under identical workloads?*

---

## License

MIT License

---

## Acknowledgements

- Imperial College London  
- Department of Computing  
- HSBC Quantitative Equities Department

---

## Contact

**Author:** Vivian Lopez  
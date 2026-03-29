# Monte Carlo Benchmarking Framework

**Imperial College London – Department of Computing**  
**Final Year Project | Supervisor: William Knottenbelt**  
**Industrial Partner: HSBC**

Modular, reproducible benchmarking framework for Monte Carlo simulation engines. Phase 1: CPU reference implementation with Streamlit dashboard.

## What's Included

✓ Configuration with validation & reproducibility hashing  
✓ BenchmarkRunner with warmup + repeated timed runs  
✓ European call option pricing workload (pure Python)  
✓ Black-Scholes validation  
✓ Streamlit dashboard for visualization & comparison  
✓ JSON result storage with full metadata  

**Not in Phase 1:** JAX/NumPy AD, GPU, C++/Rust, advanced profiling, distributed execution

## Getting Started (5 minutes)

```bash
# 1. Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Run benchmark
python experiments/run_experiment.py

# 3. View results
streamlit run benchmarking/frontend/app.py
```

For dashboard comparison, first generate multiple runs:
```bash
python experiments/generate_runs.py
```

Then click **Load Results** in sidebar (set directory to `results`).

---

## API Essentials

### MCConfig – Define Parameters

```python
from fyp.core.config import MCConfig

config = MCConfig(
    S0=100.0,          # Spot
    K=100.0,           # Strike
    r=0.05,            # Risk-free rate
    sigma=0.2,         # Volatility
    T=1.0,             # Time to maturity
    N=1,               # Time steps
    M=10000,           # MC paths
    seed=42            # Reproducibility
)

config.validate()                    # Check params
hash_val = config.config_hash()     # 8-char hash for tracking
```

### BenchmarkRunner – Execute & Measure

```python
from fyp.runner.runner import BenchmarkRunner
from fyp.workloads.mc_cpu import monte_carlo_european_call

runner = BenchmarkRunner(monte_carlo_european_call)
result = runner.run(config, num_warmup=1, num_runs=5)

# Access results
print(f"Price: ${result.result:.6f}")
print(f"Time: {result.mean_runtime*1000:.2f} ms")

# Save
runner.save_results(result, "output.json")
```

---

## Project Structure

```
benchmarking/         Core framework
├── core/            Configuration & result aggregation
├── workloads/       MC engine implementations
├── runner/          Benchmark harness
└── frontend/        Streamlit dashboard

results/            JSON benchmark outputs

experiments/        Script entry points
├── run_experiment.py        Single benchmark
└── generate_runs.py         Multi-run test data
```

---

## Key Design

| Aspect | Current | Future |
|---|---|---|
| **Hardware** | Single-machine CPU | GPU, distributed |
| **Languages** | Python | C++, JAX, Rust |
| **AD Support** | Framework only | Forward/reverse mode |
| **Output** | JSON | HDF5, Parquet |
| **Profiling** | Basic timing | LIKWID, VTune |

---

## Extending

**Add new workload:**
```python
# fyp/workloads/new_workload.py
def new_workload(config: MCConfig, ad_mode: str = "none") -> float:
    return result
```

**Use it:**
```python
runner = BenchmarkRunner(new_workload)
result = runner.run(config)
```

---

## Contact

**Supervisor:** William Knottenbelt (Imperial College)  
**Industrial Partner:** HSBC (Zouhair Rajehi)

**Status:** Phase 1 (CPU + dashboard)  
**Last Updated:** 29 March 2026

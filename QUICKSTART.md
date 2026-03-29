# Quick Start Guide

## 1-Minute Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run a Single Benchmark

```bash
python experiments/run_experiment.py
```

**Output:** JSON file in `results/` folder with estimated price, runtime statistics, and metadata.

**Example output:**
```
Price: $10.3095
Mean runtime: 5.26 ms
Throughput: 1.90M paths/sec
```

## View Results in Dashboard

```bash
# Generate 4 test runs with different path counts
python experiments/generate_runs.py

# Launch Streamlit app
streamlit run benchmarking/frontend/app.py
```

**Dashboard URL:** `http://localhost:8501`

**In the app:**
1. Click "Load Results" in sidebar (set directory to `results`)
2. Explore **Overview** tab for summary metrics
3. Go to **Compare Runs** tab to see multi-run analysis
4. Select a run in **Detailed View** for full inspection

---

## Customize a Benchmark

```python
from benchmarking.core.config import MCConfig
from benchmarking.runner.runner import BenchmarkRunner
from benchmarking.workloads.mc_cpu import monte_carlo_european_call

# Define parameters
config = MCConfig(
    S0=100.0,   # Spot
    K=105.0,    # OTM call
    r=0.03,     # Lower rate
    sigma=0.25, # Higher vol
    T=0.25,     # Quarterly
    N=1,
    M=50000,    # More paths
    seed=123
)

# Run benchmark
runner = BenchmarkRunner(monte_carlo_european_call)
result = runner.run(config, num_warmup=2, num_runs=10)

# Save results
runner.save_results(result, "my_benchmark.json")

print(f"Price: ${result.result:.6f}")
print(f"Mean time: {result.mean_runtime*1000:.2f} ms")
```

---

## Add Custom Workload

Create `benchmarking/workloads/my_option.py`:

```python
from benchmarking.core.config import MCConfig
import math
import random

def my_asian_call(config: MCConfig, ad_mode: str = "none") -> float:
    """Asian call option pricing."""
    random.seed(config.seed)
    
    payoffs = []
    for _ in range(config.M):
        # Simulate path and compute average
        S = config.S0
        path_sum = 0
        for step in range(config.N):
            Z = random.gauss(0, 1)
            S *= math.exp((config.r - 0.5*config.sigma**2) * (config.T/config.N) + 
                         config.sigma * math.sqrt(config.T/config.N) * Z)
            path_sum += S / config.N
        
        payoff = max(path_sum - config.K, 0)
        payoffs.append(payoff)
    
    price = math.exp(-config.r * config.T) * sum(payoffs) / config.M
    return price
```

Then run:
```python
from benchmarking.runner.runner import BenchmarkRunner
from benchmarking.workloads.my_option import my_asian_call

runner = BenchmarkRunner(my_asian_call)
result = runner.run(config)
```

---

## Files Generated

When you run benchmarks, JSON files are created:

```json
{
  "config": {
    "S0": 100.0,
    "K": 100.0,
    ...,
    "seed": 42
  },
  "result": 10.309491,
  "runtimes": [0.00499, 0.00501, ...],
  "statistics": {
    "mean_runtime": 0.005259,
    "std_runtime": 0.000488,
    "min_runtime": 0.004988,
    "max_runtime": 0.006126
  },
  "config_hash": "49c27a6a",
  "ad_mode": "none",
  "metadata": {
    "timestamp": "2026-03-29T15:02:33",
    "python_version": "3.12.3",
    "platform": "Linux-5.15.0-x86_64"
  }
}
```

Load multiple files in the dashboard to compare.

---

## Troubleshooting

**"ModuleNotFoundError: No module named 'fyp'"**
- Make sure you're in the project root directory
- Streamlit should auto-handle path; if not, it's handled in `fyp/frontend/app.py`

**Dashboard not loading results**
- Click "Load Results" button in sidebar
- Set directory to `.` (current directory)
- Ensure `.json` files exist in the directory

**Results show as "N/A"**
- Check that JSON is properly formatted (use generated files as template)
- Ensure `config` and `result` keys exist

---

**For more details, see [README.md](README.md)**
# Engine Abstraction Refactoring

This document outlines the structural refactoring to introduce a proper `MonteCarloEngine` abstraction while maintaining full backwards compatibility.

## What Changed

### 1. New Abstract Base Class: `MonteCarloEngine`

**File:** `benchmarking/core/engine.py`

```python
from abc import ABC, abstractmethod
from benchmarking.core.config import MCConfig

class MonteCarloEngine(ABC):
    """Abstract base class for Monte Carlo simulation engines."""
    
    @abstractmethod
    def run(self, config: MCConfig, ad_mode: str = "none") -> float:
        """Execute a Monte Carlo simulation."""
        pass
```

**Purpose:**
- Defines clear interface for all MC engines
- Enables swapping implementations (CPU → GPU → JAX)
- Makes AD support explicit in interface

---

### 2. CPU Engine Implementation

**File:** `benchmarking/workloads/mc_cpu.py` (new class added)

```python
from benchmarking.core.engine import MonteCarloEngine

class CPUMonteCarloEngine(MonteCarloEngine):
    """Pure Python CPU implementation of MC simulator."""
    
    def run(self, config: MCConfig, ad_mode: str = "none") -> float:
        """Execute geometric Brownian motion simulation."""
        random.seed(config.seed)
        payoff_sum = 0.0
        
        for _ in range(config.M):
            Z = random.gauss(0, 1)
            S_T = config.S0 * math.exp(...)
            payoff = max(S_T - config.K, 0)
            payoff_sum += payoff
        
        price = math.exp(-config.r * config.T) * payoff_sum / config.M
        return price
```

**Legacy Function Preserved:**
```python
# Original function still exists for backwards compatibility
def monte_carlo_european_call(config: MCConfig, ad_mode: str = "none") -> float:
    """Legacy function interface."""
    # ... same implementation
```

---

### 3. Updated Runner

**File:** `benchmarking/runner/runner.py`

Before:
```python
class BenchmarkRunner:
    def __init__(self, workload_func: Callable[[MCConfig, str], float]):
        self.workload_func = workload_func
    
    def run(self, config: MCConfig, ...):
        for _ in range(num_warmup):
            self.workload_func(config, ad_mode)
        
        for _ in range(num_runs):
            start = time.perf_counter()
            res = self.workload_func(config, ad_mode)
            end = time.perf_counter()
```

After:
```python
from benchmarking.core.engine import MonteCarloEngine

class BenchmarkRunner:
    def __init__(self, engine: Union[MonteCarloEngine, Callable]):
        self.engine = engine  # Accepts both types
    
    def run(self, config: MCConfig, ...):
        for _ in range(num_warmup):
            if isinstance(self.engine, MonteCarloEngine):
                self.engine.run(config, ad_mode)
            else:
                self.engine(config, ad_mode)  # Legacy
        
        for _ in range(num_runs):
            start = time.perf_counter()
            if isinstance(self.engine, MonteCarloEngine):
                res = self.engine.run(config, ad_mode)
            else:
                res = self.engine(config, ad_mode)  # Legacy
            end = time.perf_counter()
```

---

## Backwards Compatibility

**No experiments needed to change.**

### Pattern 1: Legacy (Still Works)

```python
from benchmarking.runner.runner import BenchmarkRunner
from benchmarking.workloads.mc_cpu import monte_carlo_european_call

runner = BenchmarkRunner(monte_carlo_european_call)  # Pass function
result = runner.run(config)
```

### Pattern 2: New (Recommended)

```python
from benchmarking.runner.runner import BenchmarkRunner
from benchmarking.workloads.mc_cpu import CPUMonteCarloEngine

engine = CPUMonteCarloEngine()
runner = BenchmarkRunner(engine)  # Pass engine instance
result = runner.run(config)
```

---

## What Did NOT Change

- ✓ `MCConfig` (identical)
- ✓ `BenchmarkResult` (identical)
- ✓ JSON result format (identical)
- ✓ All utility functions: `black_scholes_call()`, `european_call_delta()`
- ✓ Experiment scripts: `experiments/run_experiment.py`, `experiments/generate_runs.py`
- ✓ Frontend dashboard
- ✓ Statistics computation

---

## Migration Path

**Immediate:** No action needed. Legacy code works as-is.

**Future:** Gradually update scripts to use new pattern:
```python
# Old
runner = BenchmarkRunner(monte_carlo_european_call)

# New
runner = BenchmarkRunner(CPUMonteCarloEngine())
```

---

## Benefits

1. **Extensibility**: New engines (JAX, GPU, etc.) just inherit `MonteCarloEngine`
2. **Type Safety**: Runner now knows engine interface at compile time
3. **Introspection**: Can query `isinstance(engine, MonteCarloEngine)`
4. **AD Ready**: Engine interface explicitly supports `ad_mode` parameter
5. **No Breaking Changes**: Legacy code continues to work

---

## Example: Adding a JAX Engine (Future)

```python
# benchmarking/workloads/mc_jax.py

from benchmarking.core.engine import MonteCarloEngine
import jax.numpy as jnp
import jax

class JAXMonteCarloEngine(MonteCarloEngine):
    """GPU-compatible MC simulator using JAX."""
    
    def run(self, config: MCConfig, ad_mode: str = "none") -> float:
        # JAX implementation with automatic differentiation
        # Works with same runner and config, no changes needed
        pass

# Use immediately:
runner = BenchmarkRunner(JAXMonteCarloEngine())
result = runner.run(config)
```

---

## Verification

All existing functionality preserved:

```bash
✓ python experiments/run_experiment.py      # Legacy pattern
✓ python experiments/generate_runs.py       # Legacy pattern
✓ python examples/engine_abstraction.py     # New pattern
✓ streamlit run benchmarking/frontend/app.py # Unchanged
```

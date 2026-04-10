"""
Workload configuration hierarchy.

WorkloadConfig is the abstract base for all option pricing workloads.
Each subclass owns its own parameters and validation logic.
The runner and storage layers only know about the base class, so new
workloads can be added without touching those layers.

Workloads currently supported:
  european  – plain vanilla European call/put (GBM, single step)
  asian     – arithmetic/geometric averaging Asian call/put (multi-step path)
  barrier   – knock-in / knock-out up/down barrier call/put (multi-step path)
  basket    – multi-asset basket call/put (correlated GBM, multi-step path)

To add a new workload:
  1. Subclass WorkloadConfig, define your parameters and SCHEMA.
  2. Register it in WORKLOAD_REGISTRY at the bottom of this file.
  3. Implement the corresponding engine method in each engine class.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, ClassVar, Dict, List, Type


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class WorkloadConfig(ABC):
    """
    Abstract base for all Monte Carlo workload configurations.

    Subclasses must define:
      workload_type : str  – unique identifier used by engines and the registry
      SCHEMA        : list[dict]  – JSON-serialisable field descriptions for
                                    dynamic form generation in the frontend
      validate()    – raise ValueError on invalid params
      to_dict()     – full serialisation (used by storage / API)
    """

    @property
    @abstractmethod
    def workload_type(self) -> str:
        ...

    @property
    @abstractmethod
    def M(self) -> int:
        """Number of Monte Carlo paths (common to every workload)."""
        ...

    @property
    @abstractmethod
    def seed(self) -> int:
        """Random seed (common to every workload)."""
        ...

    @abstractmethod
    def validate(self) -> None:
        ...

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        ...

    def config_hash(self) -> str:
        """SHA-256 fingerprint of the serialised config (first 8 hex chars)."""
        s = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(s.encode()).hexdigest()[:8]

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkloadConfig":
        ...


# ---------------------------------------------------------------------------
# European option
# ---------------------------------------------------------------------------

@dataclass
class EuropeanOptionConfig(WorkloadConfig):
    """Plain vanilla European call/put under GBM (closed-form benchmark available)."""

    SCHEMA: ClassVar[List[Dict[str, Any]]] = [
        {"key": "S0",          "label": "Initial Price (S₀)",  "type": "number", "default": 100.0, "min": 0.01},
        {"key": "K",           "label": "Strike (K)",           "type": "number", "default": 100.0, "min": 0.01},
        {"key": "r",           "label": "Risk-free Rate (r)",   "type": "number", "default": 0.05,  "min": 0.0, "max": 1.0},
        {"key": "sigma",       "label": "Volatility (σ)",       "type": "number", "default": 0.20,  "min": 0.0, "max": 1.0},
        {"key": "T",           "label": "Maturity (T, years)",  "type": "number", "default": 1.0,   "min": 0.01},
        {"key": "option_type", "label": "Option Type",          "type": "select", "default": "call", "options": ["call", "put"]},
        {"key": "M",           "label": "Paths (M)",            "type": "integer", "default": 10000, "min": 100},
        {"key": "seed",        "label": "Random Seed",          "type": "integer", "default": 42,    "min": 0},
    ]

    S0: float = 100.0
    K: float = 100.0
    r: float = 0.05
    sigma: float = 0.20
    T: float = 1.0
    option_type: str = "call"
    N: int = 1          # single-step; kept for runner compatibility
    M: int = 10000
    seed: int = 42

    @property
    def workload_type(self) -> str:
        return "european"

    def validate(self) -> None:
        if self.S0 <= 0:
            raise ValueError(f"S0 must be positive, got {self.S0}")
        if self.K <= 0:
            raise ValueError(f"K must be positive, got {self.K}")
        if self.T <= 0:
            raise ValueError(f"T must be positive, got {self.T}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        if self.M <= 0:
            raise ValueError(f"M must be positive, got {self.M}")
        if self.option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got {self.option_type!r}")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["workload_type"] = self.workload_type
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EuropeanOptionConfig":
        keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in keys})


# ---------------------------------------------------------------------------
# Asian option
# ---------------------------------------------------------------------------

@dataclass
class AsianOptionConfig(WorkloadConfig):
    """Arithmetic or geometric averaging Asian call/put (path-dependent, multi-step)."""

    SCHEMA: ClassVar[List[Dict[str, Any]]] = [
        {"key": "S0",          "label": "Initial Price (S₀)",   "type": "number",  "default": 100.0, "min": 0.01},
        {"key": "K",           "label": "Strike (K)",            "type": "number",  "default": 100.0, "min": 0.01},
        {"key": "r",           "label": "Risk-free Rate (r)",    "type": "number",  "default": 0.05,  "min": 0.0, "max": 1.0},
        {"key": "sigma",       "label": "Volatility (σ)",        "type": "number",  "default": 0.20,  "min": 0.0, "max": 1.0},
        {"key": "T",           "label": "Maturity (T, years)",   "type": "number",  "default": 1.0,   "min": 0.01},
        {"key": "option_type", "label": "Option Type",           "type": "select",  "default": "call", "options": ["call", "put"]},
        {"key": "averaging",   "label": "Averaging Method",      "type": "select",  "default": "arithmetic", "options": ["arithmetic", "geometric"]},
        {"key": "N",           "label": "Time Steps (N)",        "type": "integer", "default": 252,   "min": 2},
        {"key": "M",           "label": "Paths (M)",             "type": "integer", "default": 10000, "min": 100},
        {"key": "seed",        "label": "Random Seed",           "type": "integer", "default": 42,    "min": 0},
    ]

    S0: float = 100.0
    K: float = 100.0
    r: float = 0.05
    sigma: float = 0.20
    T: float = 1.0
    option_type: str = "call"
    averaging: str = "arithmetic"
    N: int = 252
    M: int = 10000
    seed: int = 42

    @property
    def workload_type(self) -> str:
        return "asian"

    def validate(self) -> None:
        if self.S0 <= 0 or self.K <= 0 or self.T <= 0 or self.sigma <= 0:
            raise ValueError("S0, K, T, sigma must be positive")
        if self.N < 2:
            raise ValueError("N must be at least 2 for path-dependent options")
        if self.M <= 0:
            raise ValueError("M must be positive")
        if self.option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put'")
        if self.averaging not in ("arithmetic", "geometric"):
            raise ValueError("averaging must be 'arithmetic' or 'geometric'")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["workload_type"] = self.workload_type
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AsianOptionConfig":
        keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in keys})


# ---------------------------------------------------------------------------
# Barrier option
# ---------------------------------------------------------------------------

@dataclass
class BarrierOptionConfig(WorkloadConfig):
    """Knock-in / knock-out up/down barrier option (path-dependent, multi-step)."""

    SCHEMA: ClassVar[List[Dict[str, Any]]] = [
        {"key": "S0",           "label": "Initial Price (S₀)",  "type": "number",  "default": 100.0, "min": 0.01},
        {"key": "K",            "label": "Strike (K)",           "type": "number",  "default": 100.0, "min": 0.01},
        {"key": "B",            "label": "Barrier Level (B)",    "type": "number",  "default": 120.0, "min": 0.01},
        {"key": "r",            "label": "Risk-free Rate (r)",   "type": "number",  "default": 0.05,  "min": 0.0, "max": 1.0},
        {"key": "sigma",        "label": "Volatility (σ)",       "type": "number",  "default": 0.20,  "min": 0.0, "max": 1.0},
        {"key": "T",            "label": "Maturity (T, years)",  "type": "number",  "default": 1.0,   "min": 0.01},
        {"key": "option_type",  "label": "Option Type",          "type": "select",  "default": "call", "options": ["call", "put"]},
        {"key": "barrier_type", "label": "Barrier Type",         "type": "select",  "default": "knock_out", "options": ["knock_out", "knock_in"]},
        {"key": "barrier_side", "label": "Barrier Side",         "type": "select",  "default": "up",   "options": ["up", "down"]},
        {"key": "N",            "label": "Time Steps (N)",       "type": "integer", "default": 252,   "min": 2},
        {"key": "M",            "label": "Paths (M)",            "type": "integer", "default": 10000, "min": 100},
        {"key": "seed",         "label": "Random Seed",          "type": "integer", "default": 42,    "min": 0},
    ]

    S0: float = 100.0
    K: float = 100.0
    B: float = 120.0
    r: float = 0.05
    sigma: float = 0.20
    T: float = 1.0
    option_type: str = "call"
    barrier_type: str = "knock_out"
    barrier_side: str = "up"
    N: int = 252
    M: int = 10000
    seed: int = 42

    @property
    def workload_type(self) -> str:
        return "barrier"

    def validate(self) -> None:
        if self.S0 <= 0 or self.K <= 0 or self.B <= 0 or self.T <= 0 or self.sigma <= 0:
            raise ValueError("S0, K, B, T, sigma must be positive")
        if self.N < 2:
            raise ValueError("N must be at least 2 for path-dependent options")
        if self.M <= 0:
            raise ValueError("M must be positive")
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")
        if self.barrier_type not in ("knock_out", "knock_in"):
            raise ValueError("barrier_type must be 'knock_out' or 'knock_in'")
        if self.barrier_side not in ("up", "down"):
            raise ValueError("barrier_side must be 'up' or 'down'")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["workload_type"] = self.workload_type
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BarrierOptionConfig":
        keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in keys})


# ---------------------------------------------------------------------------
# Basket option
# ---------------------------------------------------------------------------

@dataclass
class BasketOptionConfig(WorkloadConfig):
    """Equal-weighted basket of correlated assets (multi-asset, multi-step)."""

    SCHEMA: ClassVar[List[Dict[str, Any]]] = [
        {"key": "n_assets",    "label": "Number of Assets",     "type": "integer", "default": 3,     "min": 2, "max": 10},
        {"key": "S0",          "label": "Initial Prices (S₀)",  "type": "number",  "default": 100.0, "min": 0.01, "note": "uniform across assets"},
        {"key": "K",           "label": "Strike (K)",           "type": "number",  "default": 100.0, "min": 0.01},
        {"key": "r",           "label": "Risk-free Rate (r)",   "type": "number",  "default": 0.05,  "min": 0.0, "max": 1.0},
        {"key": "sigma",       "label": "Volatility (σ)",       "type": "number",  "default": 0.20,  "min": 0.0, "max": 1.0, "note": "uniform across assets"},
        {"key": "rho",         "label": "Correlation (ρ)",      "type": "number",  "default": 0.50,  "min": -1.0, "max": 1.0, "note": "pairwise correlation"},
        {"key": "T",           "label": "Maturity (T, years)",  "type": "number",  "default": 1.0,   "min": 0.01},
        {"key": "option_type", "label": "Option Type",          "type": "select",  "default": "call", "options": ["call", "put"]},
        {"key": "N",           "label": "Time Steps (N)",       "type": "integer", "default": 52,    "min": 1},
        {"key": "M",           "label": "Paths (M)",            "type": "integer", "default": 10000, "min": 100},
        {"key": "seed",        "label": "Random Seed",          "type": "integer", "default": 42,    "min": 0},
    ]

    n_assets: int = 3       # Number of assets in the basket
    S0: float = 100.0       # Uniform initial price for all assets
    K: float = 100.0
    r: float = 0.05
    sigma: float = 0.20     # Uniform vol for all assets
    rho: float = 0.50       # Pairwise correlation between all asset pairs
    T: float = 1.0
    option_type: str = "call"
    N: int = 52
    M: int = 10000
    seed: int = 42

    @property
    def workload_type(self) -> str:
        return "basket"

    def validate(self) -> None:
        if self.n_assets < 2:
            raise ValueError("n_assets must be at least 2")
        if self.S0 <= 0 or self.K <= 0 or self.sigma <= 0 or self.T <= 0:
            raise ValueError("S0, K, sigma, T must be positive")
        if not (-1.0 <= self.rho <= 1.0):
            raise ValueError("rho must be in [-1, 1]")
        if self.M <= 0:
            raise ValueError("M must be positive")
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["workload_type"] = self.workload_type
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BasketOptionConfig":
        keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in keys})


# ---------------------------------------------------------------------------
# Registry  (add new workloads here — nothing else needs to change)
# ---------------------------------------------------------------------------

WORKLOAD_REGISTRY: Dict[str, Type[WorkloadConfig]] = {
    "european": EuropeanOptionConfig,
    "asian":    AsianOptionConfig,
    "barrier":  BarrierOptionConfig,
    "basket":   BasketOptionConfig,
}


def config_from_dict(data: Dict[str, Any]) -> WorkloadConfig:
    """Deserialise any config dict to the correct WorkloadConfig subclass."""
    wtype = data.get("workload_type", "european")
    cls = WORKLOAD_REGISTRY.get(wtype)
    if cls is None:
        raise ValueError(f"Unknown workload_type {wtype!r}. "
                         f"Registered: {list(WORKLOAD_REGISTRY)}")
    return cls.from_dict(data)


# ---------------------------------------------------------------------------
# Backward-compatibility alias (existing code using MCConfig keeps working)
# ---------------------------------------------------------------------------

MCConfig = EuropeanOptionConfig
"""
Workload configuration hierarchy.

WorkloadConfig is the abstract base for all option pricing workloads.
Each subclass owns its own parameters and validation logic.
The runner and storage layers only know about the base class, so new
workloads can be added without touching those layers.

Workloads currently supported:
  european  – plain vanilla European call/put (GBM, single step)

To add a new workload:
  1. Subclass WorkloadConfig, define your parameters and SCHEMA.
  2. Register it in WORKLOAD_REGISTRY at the bottom of this file.
  3. Implement the corresponding engine method in each engine class.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
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
# Registry  (add new workloads here — nothing else needs to change)
# ---------------------------------------------------------------------------

WORKLOAD_REGISTRY: Dict[str, Type[WorkloadConfig]] = {
    "european": EuropeanOptionConfig,
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
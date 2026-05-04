"""
Workload configuration hierarchy.

To add a new workload:
  1. Create a class subclassing WorkloadConfig decorated with @workload("name").
  2. Define fields as plain class attributes with default values.
  3. That's it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Any, ClassVar, Dict, Type


WORKLOAD_REGISTRY: Dict[str, Type["WorkloadConfig"]] = {}


def workload(name: str):
    """Register a WorkloadConfig subclass under *name* and apply @dataclass.

    This is the only decorator a new workload needs::

        @workload("digital")
        class DigitalOptionConfig(WorkloadConfig):
            S0:      float = 100.0
            barrier: float = 110.0

    That's it.  No ``@dataclass``, no ``workload_type`` property, no registry
    edit required.
    """
    def decorator(cls):
        cls = dataclass(cls)
        cls.workload_type = name
        WORKLOAD_REGISTRY[name] = cls
        return cls
    return decorator


class WorkloadConfig:
    """
    Base class for Monte Carlo workload configurations.

    Use ``@workload("name")`` on subclasses — it handles @dataclass,
    workload_type, and registration automatically.

    to_dict(), from_dict(), config_hash(), and validate() are all inherited.
    Override validate() only when you need cross-field logic (e.g. barrier > S0).
    """

    workload_type: ClassVar[str]  # set automatically by @workload(...)

    def validate(self) -> None:
        """Override to add cross-field validation.  Call super().validate() first."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["workload_type"] = self.workload_type
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkloadConfig":
        keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in keys})

    def config_hash(self) -> str:
        """8-char SHA-256 fingerprint of the config, stable across machines."""
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True).encode()
        ).hexdigest()[:8]



@workload(name="european")
class EuropeanOptionConfig(WorkloadConfig):
    """Plain vanilla European call/put under GBM (closed-form benchmark available)."""

    S0:          float = 100.0
    K:           float = 100.0
    r:           float = 0.05
    sigma:       float = 0.20
    T:           float = 1.0
    option_type: str   = "call"
    N:           int   = 1
    M:           int   = 10000
    seed:        int   = 42


def config_from_dict(data: Dict[str, Any]) -> WorkloadConfig:
    """Deserialise a config dict to the correct WorkloadConfig subclass."""
    wtype = data.get("workload_type", "european")
    cls = WORKLOAD_REGISTRY.get(wtype)
    if cls is None:
        raise ValueError(f"Unknown workload_type {wtype!r}. Registered: {list(WORKLOAD_REGISTRY)}")
    return cls.from_dict(data)


"""
GCP instance hourly pricing and per-run cost calculation.

Pricing resolution order
------------------------
1. GCP Cloud Billing Catalog REST API (requires GCP_PRICING_API_KEY env var
   or the api_key argument).  Fetches live on-demand US prices.
2. KNOWN_RATES fallback dict — covers the three target machine families for
   us-central1 on-demand pricing (May 2026 rates).  Used when no API key is
   available or when the API call fails.

Usage
-----
    from benchmarking.cloud.pricing import get_hourly_rate, compute_cost_per_run

    rate = get_hourly_rate("n2-standard-8", region="us-central1")
    cost = compute_cost_per_run(mean_runtime_ms=45.3, hourly_rate=rate)
"""

from __future__ import annotations

import logging
import os
import re
from typing import Dict, Optional

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static fallback rates: us-central1, on-demand, USD/hr (May 2026)
# Source: https://cloud.google.com/compute/vm-instance-pricing
# ---------------------------------------------------------------------------
KNOWN_RATES: Dict[str, float] = {
    # Intel Cascade Lake / Sapphire Rapids (n2-standard)
    "n2-standard-2":   0.097_05,
    "n2-standard-4":   0.194_10,
    "n2-standard-8":   0.388_19,
    "n2-standard-16":  0.776_38,
    "n2-standard-32":  1.552_77,
    # AMD EPYC (t2d-standard)
    "t2d-standard-1":  0.042_52,
    "t2d-standard-2":  0.085_04,
    "t2d-standard-4":  0.170_08,
    "t2d-standard-8":  0.340_16,
    "t2d-standard-16": 0.680_32,
    "t2d-standard-32": 1.360_64,
    # Compute-optimised (c2-standard)
    "c2-standard-4":   0.209_52,
    "c2-standard-8":   0.419_04,
    "c2-standard-16":  0.838_08,
    "c2-standard-30":  1.571_40,
    "c2-standard-60":  3.142_80,
}

# GCP Billing Catalog service ID for Compute Engine
_COMPUTE_ENGINE_SERVICE = "6F81-5844-456A"
_BILLING_BASE = "https://cloudbilling.googleapis.com/v1"

# Per-vCPU rates (USD/hr) by machine family, used to scale KNOWN_RATES for
# sizes not in the table.
_VCPU_RATES: Dict[str, float] = {
    "n2":  0.048_524,
    "t2d": 0.042_52 / 1,   # 1 vCPU baseline
    "c2":  0.052_38,
}

# vCPU count extracted from machine type string, e.g. "n2-standard-8" → 8
_VCPU_RE = re.compile(r"-(\d+)$")


def _vcpu_count(machine_type: str) -> Optional[int]:
    m = _VCPU_RE.search(machine_type)
    return int(m.group(1)) if m else None


def _family(machine_type: str) -> str:
    """Return the machine family prefix, e.g. 'n2-standard-8' → 'n2'."""
    return machine_type.split("-")[0]


def _lookup_known_rates(machine_type: str) -> Optional[float]:
    """Exact match first; then scale from a known sibling by vCPU count."""
    if machine_type in KNOWN_RATES:
        return KNOWN_RATES[machine_type]

    fam = _family(machine_type)
    vcpus = _vcpu_count(machine_type)
    if vcpus is None:
        return None

    # Find a known sibling in the same family
    siblings = {k: v for k, v in KNOWN_RATES.items() if k.startswith(fam + "-")}
    if not siblings:
        return None

    # Pick the sibling closest in size; scale by vCPU ratio
    sib_type = min(siblings, key=lambda k: abs((_vcpu_count(k) or 0) - vcpus))
    sib_vcpus = _vcpu_count(sib_type)
    if not sib_vcpus:
        return None
    return siblings[sib_type] * vcpus / sib_vcpus


def _fetch_from_billing_api(
    machine_type: str,
    region: str,
    api_key: str,
) -> Optional[float]:
    """
    Query the GCP Cloud Billing Catalog API to find the on-demand hourly rate
    for a given machine type.

    This works by:
    1. Listing all SKUs for the Compute Engine service.
    2. Filtering for vCPU and memory SKUs matching the machine family and region.
    3. Computing total_cost = (vcpus × core_rate_per_hr) + (memory_gb × ram_rate_per_hr).

    Returns None on any error so callers can fall back to KNOWN_RATES.
    """
    try:
        import requests
    except ImportError:
        _LOG.warning("requests library not available; falling back to KNOWN_RATES")
        return None

    fam    = _family(machine_type)
    vcpus  = _vcpu_count(machine_type)
    if vcpus is None:
        return None

    # GCP machine families map to SKU description prefixes
    _FAMILY_TO_SKU_PREFIX: Dict[str, str] = {
        "n2":  "N2 Instance",
        "t2d": "T2D AMD Instance",
        "c2":  "Compute optimized",
        "e2":  "E2 Instance",
        "n1":  "N1 Predefined Instance",
    }
    sku_prefix = _FAMILY_TO_SKU_PREFIX.get(fam)
    if sku_prefix is None:
        _LOG.debug("No SKU prefix mapping for family %r; using KNOWN_RATES", fam)
        return None

    # Fetch SKUs (paginated)
    url = f"{_BILLING_BASE}/services/{_COMPUTE_ENGINE_SERVICE}/skus"
    params: dict = {"key": api_key, "currencyCode": "USD", "pageSize": 5000}
    core_rate: Optional[float] = None
    ram_rate: Optional[float] = None

    region_norm = region.lower()

    try:
        while True:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            for sku in data.get("skus", []):
                desc = sku.get("description", "")
                if not desc.startswith(sku_prefix):
                    continue
                # Check region applicability
                regions = [
                    sr.get("region", "") for sr in
                    sku.get("serviceRegions", [])
                ]
                if region_norm not in [r.lower() for r in regions]:
                    continue
                # Extract the per-unit nanos price
                pricing = sku.get("pricingInfo", [])
                if not pricing:
                    continue
                tiered = (pricing[0]
                          .get("pricingExpression", {})
                          .get("tieredRates", []))
                if not tiered:
                    continue
                # First tier (index 0 = usage up to 0 units = base rate)
                unit_price = tiered[-1].get("unitPrice", {})
                nanos = (int(unit_price.get("nanos", 0)) +
                         int(unit_price.get("units", 0)) * 1_000_000_000)
                rate_per_unit = nanos / 1_000_000_000  # USD per unit-hour

                if "Core" in desc and core_rate is None:
                    core_rate = rate_per_unit
                elif "Ram" in desc and ram_rate is None:
                    ram_rate = rate_per_unit

                if core_rate is not None and ram_rate is not None:
                    break

            next_page = data.get("nextPageToken")
            if not next_page or (core_rate is not None and ram_rate is not None):
                break
            params["pageToken"] = next_page

    except Exception as exc:
        _LOG.warning("GCP Billing API request failed: %s", exc)
        return None

    if core_rate is None or ram_rate is None:
        _LOG.debug(
            "Could not find both core+RAM SKUs for %r in %r (core=%s, ram=%s)",
            machine_type, region, core_rate, ram_rate,
        )
        return None

    # Approximate memory per vCPU: n2/c2 → 4 GB/vCPU, t2d → 4 GB/vCPU (default)
    gb_per_vcpu = 4.0
    memory_gb = vcpus * gb_per_vcpu
    hourly = vcpus * core_rate + memory_gb * ram_rate
    return round(hourly, 6)


def get_hourly_rate(
    machine_type: str,
    region: str = "us-central1",
    api_key: Optional[str] = None,
) -> Optional[float]:
    """
    Return the on-demand hourly rate (USD) for *machine_type* in *region*.

    Resolution order:
      1. GCP Billing Catalog API (if api_key or GCP_PRICING_API_KEY env var)
      2. KNOWN_RATES static dict (exact match or vCPU-scaled sibling)
      3. Returns None if neither source can provide a rate.

    Parameters
    ----------
    machine_type : str
        GCP machine type, e.g. "n2-standard-8".
    region : str
        GCP region, e.g. "us-central1" (default).
    api_key : str, optional
        GCP API key.  If not provided, read from GCP_PRICING_API_KEY env var.
    """
    key = api_key or os.environ.get("GCP_PRICING_API_KEY")

    if key:
        rate = _fetch_from_billing_api(machine_type, region, key)
        if rate is not None:
            _LOG.debug("GCP API rate for %s in %s: $%.6f/hr", machine_type, region, rate)
            return rate
        _LOG.debug(
            "GCP API lookup failed for %s; falling back to KNOWN_RATES", machine_type
        )

    rate = _lookup_known_rates(machine_type)
    if rate is not None:
        _LOG.debug("KNOWN_RATES fallback for %s: $%.6f/hr", machine_type, rate)
    else:
        _LOG.warning(
            "No hourly rate found for machine type %r. "
            "Pass --hourly-rate or set GCP_PRICING_API_KEY.", machine_type
        )
    return rate


def compute_cost_per_run(mean_runtime_ms: float, hourly_rate: float) -> float:
    """
    Compute the cost of a single benchmark run.

    cost_per_run = mean_runtime_ms / 3_600_000 * hourly_rate

    Parameters
    ----------
    mean_runtime_ms : float
        Mean runtime of a single run in milliseconds.
    hourly_rate : float
        Instance hourly rate in USD.

    Returns
    -------
    float
        Cost in USD.
    """
    return mean_runtime_ms / 3_600_000.0 * hourly_rate

"""
GCP instance metadata client.

Queries the GCP metadata server (available only on GCP VMs).  On any other
machine the functions return None values without raising.

Usage
-----
    from benchmarking.cloud.metadata import get_instance_metadata

    meta = get_instance_metadata()
    # {"cloud_provider": "gcp", "instance_type": "n2-standard-8",
    #  "zone": "us-central1-a"}
    # or {"cloud_provider": None, "instance_type": None, "zone": None}
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

_LOG = logging.getLogger(__name__)

_METADATA_BASE = "http://metadata.google.internal/computeMetadata/v1"
_HEADERS = {"Metadata-Flavor": "Google"}
_TIMEOUT = 1.5  # seconds — fast fail if not on GCP


def _fetch(path: str) -> Optional[str]:
    """GET a single metadata path and return the response text, or None."""
    try:
        import requests
        r = requests.get(
            f"{_METADATA_BASE}/{path}",
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        return r.text.strip()
    except Exception as exc:
        _LOG.debug("GCP metadata fetch failed (%s): %s", path, exc)
        return None


def get_instance_metadata() -> Dict[str, Optional[str]]:
    """
    Return GCP instance metadata as a dict.

    Keys
    ----
    cloud_provider : "gcp" if on a GCP VM, else None
    instance_type  : e.g. "n2-standard-8"  (last component of the
                     full resource path returned by the metadata server)
    zone           : e.g. "us-central1-a"  (last component of zone path)

    Always returns a dict with all three keys; values are None when not on GCP
    or when the metadata server is unreachable.
    """
    machine_type_raw = _fetch("instance/machine-type")
    if machine_type_raw is None:
        return {"cloud_provider": None, "instance_type": None, "zone": None}

    # Response is a full resource path, e.g.
    # "projects/123456789/machineTypes/n2-standard-8"
    instance_type = machine_type_raw.rsplit("/", 1)[-1]

    zone_raw = _fetch("instance/zone")
    zone = zone_raw.rsplit("/", 1)[-1] if zone_raw else None

    return {
        "cloud_provider": "gcp",
        "instance_type": instance_type,
        "zone": zone,
    }

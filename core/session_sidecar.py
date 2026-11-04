"""
session_sidecar.py

Small companion files that travel next to a detection CSV and record
the parameters and analysis window that produced it. Enables reloading
a CSV into a fresh session and continuing where the user left off,
without having to re-run detection.

Format
------
A JSON file next to the CSV with the same base name and a
``.meta.json`` suffix:

    ripples_ch18.csv      ->  ripples_ch18.meta.json
    theta_cycles.csv      ->  theta_cycles.meta.json

Contents (version 1):

    {
      "kind": "ripples" | "theta_epochs" | "theta_cycles",
      "version": 1,
      "created": "2026-10-16T14:23:01",
      "sample_rate": 20000.0,
      "analysis_start_sample": 0,
      "analysis_end_sample": 24000000,
      "channels": [18],
      "params": { ... },             # kind-specific, matches the dataclass
      "landmark_names": [ ... ]      # OPTIONAL: 6 strings, theta_cycles only
    }

``landmark_names`` is only written by theta-cycle saves. It gives the
exact landmark identity of each ref_1..ref_6 CSV column for the
scheme that produced the file, so a reader of the sidecar can
interpret the position-only CSV without knowing the scheme rules.
Absent for other kinds and for any file saved before this field
existed; in that case the reader falls back to the rising_zc layout.

No GUI dependency.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


SIDECAR_SUFFIX = ".meta.json"
CURRENT_VERSION = 2


def sidecar_path_for(csv_path: str | Path) -> Path:
    """Return the sidecar path corresponding to a given CSV path.

    ``foo.csv`` -> ``foo.meta.json``
    ``foo``     -> ``foo.meta.json``
    """
    csv_path = Path(csv_path)
    stem = csv_path.stem if csv_path.suffix.lower() == ".csv" else csv_path.name
    return csv_path.with_name(stem + SIDECAR_SUFFIX)


def _params_to_dict(params: Any) -> dict:
    """Convert a params dataclass (or plain dict) into a JSON-safe dict."""
    if is_dataclass(params) and not isinstance(params, type):
        return asdict(params)
    if isinstance(params, dict):
        return dict(params)
    raise TypeError(
        f"params must be a dataclass instance or dict, got {type(params).__name__}"
    )


def save_sidecar(
    csv_path: str | Path,
    kind: str,
    sample_rate: float,
    analysis_start_sample: int,
    analysis_end_sample: int,
    channels: list[int],
    params: Any,
    landmark_names: list[str] | None = None,
) -> Path:
    """Write the sidecar next to the CSV. Returns the sidecar path.

    ``landmark_names``, when provided, is stored under a top-level
    key and gives the exact meaning of each ref_N column for the
    scheme that produced the file. Only theta_cycles saves currently
    pass it.
    """
    sidecar = sidecar_path_for(csv_path)
    payload = {
        "kind": str(kind),
        "version": CURRENT_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "sample_rate": float(sample_rate),
        "analysis_start_sample": int(analysis_start_sample),
        "analysis_end_sample": int(analysis_end_sample),
        "channels": [int(c) for c in channels],
        "params": _params_to_dict(params),
    }
    if landmark_names is not None:
        payload["landmark_names"] = [str(n) for n in landmark_names]
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with open(sidecar, "w") as f:
        json.dump(payload, f, indent=2)
    return sidecar


def load_sidecar(csv_path: str | Path) -> dict | None:
    """Return the parsed sidecar for a CSV, or None if absent.

    Raises on malformed JSON or an unsupported version so the caller
    can decide what to do; a missing sidecar is not an error.
    """
    sidecar = sidecar_path_for(csv_path)
    if not sidecar.exists():
        return None
    with open(sidecar, "r") as f:
        data = json.load(f)

    version = int(data.get("version", 0))
    if version > CURRENT_VERSION:
        raise ValueError(
            f"Sidecar version {version} is newer than this app supports "
            f"(version {CURRENT_VERSION}). Update the app to read it."
        )
    return data
"""
theta_epoch_export.py

CSV export for detected/edited theta epochs.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

from core.theta_epoch_detector import ThetaEpoch


CSV_FIELDNAMES = [
    "channel",
    "start_sample",
    "end_sample",
    "peak_sample",
    "mean_theta_power",
    "mean_delta_power",
    "mean_ratio",
    "peak_ratio",
    "duration_ms",
    "manual",
]


def export_theta_epochs_to_csv(
    epochs: Sequence[ThetaEpoch],
    sample_rate: float,
    sample_offset: int,
    output_path: str | Path,
) -> Path:
    """
    Write one row per theta epoch to a CSV file.
    
    Parameters
    ----------
    epochs : sequence of ThetaEpoch
    sample_rate : Hz
    sample_offset : samples to add to relative sample indices
    output_path : where to write the CSV
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    rows = []
    for epoch in epochs:
        rows.append({
            "channel": epoch.channel,
            "start_sample": epoch.start_sample + sample_offset,
            "end_sample": epoch.end_sample + sample_offset,
            "peak_sample": epoch.peak_sample + sample_offset,
            "mean_theta_power": round(epoch.mean_theta_power, 6),
            "mean_delta_power": round(epoch.mean_delta_power, 6),
            "mean_ratio": round(epoch.mean_ratio, 4),
            "peak_ratio": round(epoch.peak_ratio, 4),
            "duration_ms": round(epoch.duration_ms, 3),
            "manual": epoch.manual,
        })
    
    rows.sort(key=lambda r: (r["channel"], r["start_sample"]))
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    
    return output_path
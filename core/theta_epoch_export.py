"""
theta_epoch_export.py

CSV export for detected/edited theta epochs.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

from core.theta_epoch_detector import ThetaEpoch, ThetaEpochParams


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


def import_theta_epochs_from_csv(
    input_path: str | Path,
    sample_offset: int = 0,
) -> list[ThetaEpoch]:
    """
    Read a previously-exported theta-epoch CSV back into ThetaEpoch
    objects, mirroring core/ripple_export.py's read_ripples_from_csv.

    The CSV (see export_theta_epochs_to_csv) stores ABSOLUTE sample
    positions (recording-global, with the detection-time sample_offset
    already added in). Since a re-loaded epoch is displayed against
    whatever detection window/offset is active in the dialog at load
    time -- not necessarily the same offset used when it was exported --
    `sample_offset` here is SUBTRACTED from each row's absolute sample
    values to convert them back to the window-relative basis
    ThetaEpochTraceViewWidget/ThetaEpochDialog expect (the same
    convention as detection results, which are relative to
    ThetaEpochDialog._sample_offset). Pass the dialog's current
    _sample_offset (or 0 if the epochs should be interpreted as already
    relative to the recording start).

    Rows are not otherwise validated against the currently-loaded
    recording (e.g. channel existing, samples in range) -- out-of-range
    epochs simply won't render until the view/channels/time window make
    them visible, same as a detection run against a wider time range
    than currently displayed.
    """
    input_path = Path(input_path)
    epochs: list[ThetaEpoch] = []
    with open(input_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(ThetaEpoch(
                channel=int(row["channel"]),
                start_sample=int(row["start_sample"]) - sample_offset,
                end_sample=int(row["end_sample"]) - sample_offset,
                peak_sample=int(row["peak_sample"]) - sample_offset,
                mean_theta_power=float(row["mean_theta_power"]),
                mean_delta_power=float(row["mean_delta_power"]),
                mean_ratio=float(row["mean_ratio"]),
                peak_ratio=float(row["peak_ratio"]),
                duration_ms=float(row["duration_ms"]),
                manual=row["manual"].strip().lower() in ("true", "1", "yes"),
            ))
    epochs.sort(key=lambda e: (e.channel, e.start_sample))
    return epochs
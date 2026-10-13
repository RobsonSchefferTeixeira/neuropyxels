"""
theta_epoch_export.py

CSV export / import for detected or edited theta epochs.

Column conventions match theta_cycle_export.py:

  * ``*_smp`` columns are ABSOLUTE sample indices into the ``.dat``
    file. Epochs are stored slice-relative in memory (the detector
    returns indices relative to the analysis window), so export adds
    ``sample_offset`` before writing, and import subtracts it.
  * ``*_s`` columns are computed at EXPORT time from the engine's
    current time base and are informational only.
  * ``duration_s`` is ``duration_smp / engine.sr``, always.

Epoch duration uses the INCLUSIVE convention: an epoch spans from
``start_sample`` through ``end_sample``, both included, so
``duration_smp = end_smp - start_smp + 1``. This is different from
cycles (which use a half-open span) and reflects the semantics of
the two detections.

No GUI dependency.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

from core.theta_epoch_detector import ThetaEpoch


CSV_FIELDNAMES = [
    "channel",
    "start_smp",
    "end_smp",
    "duration_smp",
    "start_s",
    "end_s",
    "duration_s",
    "mean_ratio",
    "peak_ratio",
    "mean_theta_power",
    "mean_delta_power",
]


def _to_row(epoch: ThetaEpoch, sample_offset: int, engine) -> dict:
    """Build one CSV row. ``sample_offset`` converts the epoch's
    in-memory (slice-relative) sample indices to absolute dat-file
    indices. ``engine`` supplies the current time base for ``*_s``."""
    start_smp = int(epoch.start_sample) + int(sample_offset)
    end_smp = int(epoch.end_sample) + int(sample_offset)
    # Inclusive span: both endpoints are part of the epoch.
    duration_smp = end_smp - start_smp + 1

    return {
        "channel": int(epoch.channel),
        "start_smp": start_smp,
        "end_smp": end_smp,
        "duration_smp": duration_smp,
        "start_s": round(float(engine.sample_to_time(start_smp)), 6),
        "end_s": round(float(engine.sample_to_time(end_smp)), 6),
        "duration_s": round(duration_smp / float(engine.sr), 6),
        "mean_ratio": round(float(epoch.mean_ratio), 6),
        "peak_ratio": round(float(epoch.peak_ratio), 6),
        "mean_theta_power": round(float(epoch.mean_theta_power), 8),
        "mean_delta_power": round(float(epoch.mean_delta_power), 8),
    }


def export_theta_epochs_to_csv(
    epochs: Sequence[ThetaEpoch],
    sample_offset: int,
    engine,
    output_path: str | Path,
) -> Path:
    """Write one row per epoch, sorted by (channel, start_smp)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [_to_row(e, sample_offset, engine) for e in epochs]
    rows.sort(key=lambda r: (r["channel"], r["start_smp"]))

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def export_theta_epochs_per_channel(
    epochs_by_channel: dict[int, Sequence[ThetaEpoch]],
    sample_offset: int,
    engine,
    output_dir: str | Path,
    filename_template: str = "theta_epochs_channel{channel}.csv",
) -> list[Path]:
    """One CSV per channel, header-only file for empty channels."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for channel, epochs in epochs_by_channel.items():
        path = output_dir / filename_template.format(channel=channel)
        export_theta_epochs_to_csv(epochs, sample_offset, engine, path)
        written.append(path)
    return written


def import_theta_epochs_from_csv(
    input_path: str | Path,
    sample_offset: int = 0,
) -> list[ThetaEpoch]:
    """Read an epoch CSV back into ThetaEpoch objects.

    ``sample_offset`` is SUBTRACTED from every ``*_smp`` value so the
    returned epochs are slice-relative (the convention the trace view
    and the dialog expect in memory). Pass the dialog's current
    ``_sample_offset``, or 0 to keep the loaded values as-is.

    The ``*_s`` columns are ignored: times are recomputed from the
    current engine at display time. ``peak_sample`` and ``manual``
    are not stored in the CSV; reloaded epochs get ``peak_sample = 0``
    and ``manual = False``.
    """
    input_path = Path(input_path)
    epochs: list[ThetaEpoch] = []
    with open(input_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            start_abs = int(row["start_smp"])
            end_abs = int(row["end_smp"])
            epochs.append(ThetaEpoch(
                channel=int(row["channel"]),
                start_sample=start_abs - int(sample_offset),
                end_sample=end_abs - int(sample_offset),
                peak_sample=0,
                mean_theta_power=float(row["mean_theta_power"]),
                mean_delta_power=float(row["mean_delta_power"]),
                mean_ratio=float(row["mean_ratio"]),
                peak_ratio=float(row["peak_ratio"]),
                duration_ms=0.0,
                manual=False,
            ))
    epochs.sort(key=lambda e: (e.channel, e.start_sample))
    return epochs
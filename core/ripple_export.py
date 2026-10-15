"""
ripple_export.py

CSV export for detected/edited ripple events.

Column conventions match theta_epoch_export.py and theta_cycle_export.py:

  * ``*_smp`` columns are ABSOLUTE sample indices into the ``.dat`` file.
    Ripple events are stored slice-relative in memory (the detector
    returns indices relative to the analysis window), so export adds
    ``sample_offset`` before writing and import subtracts it.
  * ``*_s`` columns are computed at EXPORT time from the engine's
    current time base (timestamps.npy if loaded, else sample / sr) and
    are informational only. On import they're ignored and recomputed
    from the samples against the current engine.
  * ``duration_s`` is ``duration_smp / engine.sr``. ``duration_smp``
    is half-open (end - start), matching the cycle convention; ripple
    end is defined by the boundary-threshold walk, so the last sample
    is the last one above the boundary, and the next sample is where
    the walk stopped.

No GUI dependency.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

from core.ripple_detector import RippleEvent


CSV_FIELDNAMES = [
    "channel",
    "start_smp",
    "end_smp",
    "peak_smp",
    "trough_smp",
    "duration_smp",
    "start_s",
    "end_s",
    "peak_s",
    "trough_s",
    "duration_s",
    "peak_amplitude",
    "trough_amplitude",
]


def _to_row(event: RippleEvent, sample_offset: int, engine) -> dict:
    """Build one CSV row. ``sample_offset`` converts the event's
    in-memory slice-relative indices to absolute dat-file indices.
    ``engine`` supplies the time base for the ``*_s`` columns."""
    start_smp = int(event.start_sample) + int(sample_offset)
    end_smp = int(event.end_sample) + int(sample_offset)
    peak_smp = int(event.peak_sample) + int(sample_offset)
    trough_smp = int(event.trough_sample) + int(sample_offset)
    # Half-open: a ripple spans from its first above-boundary sample up
    # to (but not including) the sample where the walk stopped.
    duration_smp = end_smp - start_smp

    return {
        "channel": int(event.channel),
        "start_smp": start_smp,
        "end_smp": end_smp,
        "peak_smp": peak_smp,
        "trough_smp": trough_smp,
        "duration_smp": duration_smp,
        "start_s": round(float(engine.sample_to_time(start_smp)), 6),
        "end_s": round(float(engine.sample_to_time(end_smp)), 6),
        "peak_s": round(float(engine.sample_to_time(peak_smp)), 6),
        "trough_s": round(float(engine.sample_to_time(trough_smp)), 6),
        "duration_s": round(duration_smp / float(engine.sr), 6),
        "peak_amplitude": round(float(event.peak_amplitude), 6),
        "trough_amplitude": round(float(event.trough_amplitude), 6),
    }


def export_ripples_to_csv(
    events: Sequence[RippleEvent],
    sample_offset: int,
    engine,
    output_path: str | Path,
) -> Path:
    """Write one row per ripple event, sorted by (channel, start_smp)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [_to_row(ev, sample_offset, engine) for ev in events]
    rows.sort(key=lambda r: (r["channel"], r["start_smp"]))

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def export_ripples_per_channel(
    events_by_channel: dict[int, Sequence[RippleEvent]],
    sample_offset: int,
    engine,
    output_dir: str | Path,
    filename_template: str = "ripples_channel{channel}.csv",
) -> list[Path]:
    """One CSV per channel, header-only file for empty channels."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for channel, events in events_by_channel.items():
        path = output_dir / filename_template.format(channel=channel)
        export_ripples_to_csv(events, sample_offset, engine, path)
        written.append(path)
    return written


def read_ripples_from_csv(path: str | Path, sample_offset: int = 0) -> list[RippleEvent]:
    """Read a ripple CSV back into RippleEvent objects.

    ``sample_offset`` is SUBTRACTED from every ``*_smp`` value so the
    returned events are slice-relative (the convention the trace view
    and the dialog expect in memory). The ``*_s`` columns are ignored;
    times are recomputed from the samples against the current engine at
    display time.
    """
    path = Path(path)
    events: list[RippleEvent] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append(RippleEvent(
                channel=int(row["channel"]),
                start_sample=int(row["start_smp"]) - int(sample_offset),
                end_sample=int(row["end_smp"]) - int(sample_offset),
                peak_sample=int(row["peak_smp"]) - int(sample_offset),
                peak_amplitude=float(row["peak_amplitude"]),
                trough_sample=int(row["trough_smp"]) - int(sample_offset),
                trough_amplitude=float(row["trough_amplitude"]),
                manual=False,
            ))
    events.sort(key=lambda e: (e.channel, e.start_sample))
    return events
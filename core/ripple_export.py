"""
ripple_export.py

CSV export for detected/edited ripple events.

Design note on sample offsets
------------------------------
RippleDetector.detect() returns RippleEvent objects whose sample indices
(start_sample, end_sample, peak_sample, trough_sample) are relative to
whatever signal segment was passed in -- e.g. if detection was run on
data starting at t=30s, sample 0 in the event is actually sample
30s * sample_rate in the full recording, not sample 0 of the recording.

Exporting relative sample indices without accounting for this would
silently produce a CSV that looks correct but points to the wrong place
in the recording the moment detection wasn't run from t=0. To avoid that,
every export function here takes an explicit sample_offset (in samples,
not seconds) and adds it before writing, so the CSV always contains
absolute sample positions in the original continuous.dat.

No GUI dependency.

Usage
-----
    from core.ripple_export import export_ripples_to_csv, RippleExportRow

    export_ripples_to_csv(
        events, sample_rate=2500, sample_offset=start_sample,
        output_path="ripples_channel12.csv",
    )

    # Or combine multiple channels' events into one file:
    export_ripples_to_csv(
        all_events, sample_rate=2500, sample_offset=start_sample,
        output_path="ripples_all_channels.csv",
    )
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from core.ripple_detector import RippleEvent


CSV_FIELDNAMES = [
    "channel",
    "start_sample",
    "end_sample",
    "peak_sample",
    "peak_amplitude",
    "trough_sample",
    "trough_amplitude",
    "duration_ms",
    "manual",
]


def _to_row(event: RippleEvent, sample_rate: float, sample_offset: int) -> dict:
    duration_ms = (event.end_sample - event.start_sample) / sample_rate * 1000
    return {
        "channel": event.channel,
        "start_sample": event.start_sample + sample_offset,
        "end_sample": event.end_sample + sample_offset,
        "peak_sample": event.peak_sample + sample_offset,
        "peak_amplitude": event.peak_amplitude,
        "trough_sample": event.trough_sample + sample_offset,
        "trough_amplitude": event.trough_amplitude,
        "duration_ms": round(duration_ms, 3),
        "manual": event.manual,
    }


def export_ripples_to_csv(
    events: Sequence[RippleEvent],
    sample_rate: float,
    sample_offset: int,
    output_path: str | Path,
) -> Path:
    """
    Write one row per ripple event to a CSV file, sorted by
    (channel, start_sample) so multi-channel exports read in a sane order.

    Parameters
    ----------
    events : sequence of RippleEvent (can span multiple channels)
    sample_rate : Hz, needed to compute duration_ms
    sample_offset : samples to add to every event's relative sample
        indices so the CSV reflects absolute position in the original
        recording (see module docstring)
    output_path : where to write the CSV

    Returns
    -------
    Path to the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [_to_row(ev, sample_rate, sample_offset) for ev in events]
    rows.sort(key=lambda r: (r["channel"], r["start_sample"]))

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def export_ripples_per_channel(
    events_by_channel: dict[int, Sequence[RippleEvent]],
    sample_rate: float,
    sample_offset: int,
    output_dir: str | Path,
    filename_template: str = "ripples_channel{channel}.csv",
) -> list[Path]:
    """
    Convenience wrapper: write one CSV file per channel, for the common
    "inspect each channel separately" workflow. All channels share the
    same sample_offset (i.e. detection was run on the same time window
    across channels) -- pass per-channel offsets by calling
    export_ripples_to_csv directly if that's not the case.

    Returns list of written file paths, one per channel (channels with
    zero events still get an empty CSV with just the header, so it's
    clear detection ran and found nothing, rather than silently skipping
    the file).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for channel, events in events_by_channel.items():
        path = output_dir / filename_template.format(channel=channel)
        export_ripples_to_csv(events, sample_rate, sample_offset, path)
        written.append(path)
    return written


def read_ripples_from_csv(path: str | Path) -> list[RippleEvent]:
    """
    Read a previously-exported CSV back into RippleEvent objects, with
    sample indices left as-is (absolute, as written) -- callers that need
    them relative to a particular segment must subtract their own offset.
    Useful for reloading a saved/edited ripple list in a later session.
    """
    path = Path(path)
    events: list[RippleEvent] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append(RippleEvent(
                channel=int(row["channel"]),
                start_sample=int(row["start_sample"]),
                end_sample=int(row["end_sample"]),
                peak_sample=int(row["peak_sample"]),
                peak_amplitude=float(row["peak_amplitude"]),
                trough_sample=int(row["trough_sample"]),
                trough_amplitude=float(row["trough_amplitude"]),
                manual=row["manual"].strip().lower() in ("true", "1", "yes"),
            ))
    return events

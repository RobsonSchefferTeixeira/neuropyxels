"""
theta_cycle_export.py

CSV export / import for detected or edited theta cycles.

Column conventions
------------------
Every ``*_smp`` column is an ABSOLUTE sample index into the original
``.dat`` file. Cycles are stored in memory with absolute sample
indices (the detection thread converts from slice-relative to absolute
before returning), so no offset is applied at export time.

Every ``*_s`` column is a convenience value derived at EXPORT time
from the engine's current time base:

  * if a timestamps.npy is loaded, the value is the timestamp at that
    sample;
  * otherwise, the value is ``sample / engine.sr`` (elapsed seconds).

The CSV file is therefore a self-consistent snapshot, but the ``_s``
columns are informational only. On import, ``_s`` values are
recomputed from the samples against the current engine, so a cycle
loaded after a timestamps file is attached will display correctly.

``duration_s`` is always ``duration_smp / engine.sr`` -- sample-count
duration is a property of the recording, not of the timestamps array,
and a delta of samples divided by the sampling rate gives the right
answer even when timestamps are non-uniform.

No GUI dependency.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

from core.theta_cycle_detector import ThetaCycle


# Snake_case on disk. The in-app table uses human-readable headers
# (see gui/theta_cycle_dialog.py) but the CSV is designed for scripts
# and spreadsheets, where a single canonical spelling matters more
# than brevity.
CSV_FIELDNAMES = [
    "channel",
    "zc_start_smp",
    "peak1_smp",
    "zc_mid_smp",
    "valley_smp",
    "zc_end_smp",
    "peak2_smp",
    "duration_smp",
    "zc_start_s",
    "peak1_s",
    "zc_mid_s",
    "valley_s",
    "zc_end_s",
    "peak2_s",
    "duration_s",
    "amp",
    "theta_delta_ratio",
    "fast_theta_ratio",
]


def _to_row(cycle: ThetaCycle, engine) -> dict:
    """Build one CSV row from a ThetaCycle. ``engine`` supplies the
    time base: ``engine.sample_to_time(sample)`` returns the timestamp
    if a timestamps.npy is loaded, else ``sample / engine.sr``."""
    zc_start_smp = int(cycle.zero_crossing_start)
    peak1_smp = int(cycle.peak1)
    zc_mid_smp = int(cycle.zero_crossing_mid)
    valley_smp = int(cycle.valley)
    zc_end_smp = int(cycle.zero_crossing_end)
    peak2_smp = int(cycle.peak2)
    # Half-open span: a cycle runs from its start crossing up to (but
    # not including) the next cycle's start crossing. Consecutive
    # cycle durations therefore sum exactly to the interval between
    # consecutive starts.
    duration_smp = zc_end_smp - zc_start_smp

    return {
        "channel": int(cycle.channel),
        "zc_start_smp": zc_start_smp,
        "peak1_smp": peak1_smp,
        "zc_mid_smp": zc_mid_smp,
        "valley_smp": valley_smp,
        "zc_end_smp": zc_end_smp,
        "peak2_smp": peak2_smp,
        "duration_smp": duration_smp,
        "zc_start_s": round(float(engine.sample_to_time(zc_start_smp)), 6),
        "peak1_s": round(float(engine.sample_to_time(peak1_smp)), 6),
        "zc_mid_s": round(float(engine.sample_to_time(zc_mid_smp)), 6),
        "valley_s": round(float(engine.sample_to_time(valley_smp)), 6),
        "zc_end_s": round(float(engine.sample_to_time(zc_end_smp)), 6),
        "peak2_s": round(float(engine.sample_to_time(peak2_smp)), 6),
        "duration_s": round(duration_smp / float(engine.sr), 6),
        "amp": round(float(cycle.amplitude), 6),
        "theta_delta_ratio": (
            "" if cycle.theta_delta_ratio is None
            else round(float(cycle.theta_delta_ratio), 4)
        ),
        "fast_theta_ratio": (
            "" if cycle.fast_theta_ratio is None
            else round(float(cycle.fast_theta_ratio), 4)
        ),
    }


def export_cycles_to_csv(
    cycles: Sequence[ThetaCycle],
    engine,
    output_path: str | Path,
) -> Path:
    """Write one row per cycle, sorted by (channel, zc_start_smp).

    ``engine`` is a TraceEngine; it supplies both ``sr`` and the
    current time base (timestamps-aware).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [_to_row(c, engine) for c in cycles]
    rows.sort(key=lambda r: (r["channel"], r["zc_start_smp"]))

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def export_cycles_per_channel(
    cycles_by_channel: dict[int, Sequence[ThetaCycle]],
    engine,
    output_dir: str | Path,
    filename_template: str = "theta_cycles_channel{channel}.csv",
) -> list[Path]:
    """One CSV per channel. Channels with zero cycles still get a
    header-only file, so it's clear detection ran and found nothing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for channel, cycles in cycles_by_channel.items():
        path = output_dir / filename_template.format(channel=channel)
        export_cycles_to_csv(cycles, engine, path)
        written.append(path)
    return written


def read_cycles_from_csv(path: str | Path) -> list[ThetaCycle]:
    """Read a cycle CSV.

    Only the ``*_smp`` columns are read. The ``*_s`` columns are
    ignored (they were computed against whatever time base was active
    when the file was saved, which may not match the current session).
    Every returned ThetaCycle holds absolute sample indices; callers
    that want to display times compute them from the current engine.

    ``duration_ms`` is populated from the half-open sample span and
    ``engine.sr`` is not available here, so we recompute it later at
    display time. For now store the sample-count duration scaled by
    a placeholder sr of 1.0 and rely on display to recompute.
    Actually: store ``duration_ms`` as 0 and let display recompute;
    see gui/theta_cycle_dialog._populate_table.
    """
    path = Path(path)
    cycles: list[ThetaCycle] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tdr = row.get("theta_delta_ratio", "").strip()
            ftr = row.get("fast_theta_ratio", "").strip()
            zc_start = int(row["zc_start_smp"])
            zc_end = int(row["zc_end_smp"])
            cycles.append(ThetaCycle(
                channel=int(row["channel"]),
                zero_crossing_start=zc_start,
                peak1=int(row["peak1_smp"]),
                zero_crossing_mid=int(row["zc_mid_smp"]),
                valley=int(row["valley_smp"]),
                zero_crossing_end=zc_end,
                peak2=int(row["peak2_smp"]),
                amplitude=float(row["amp"]),
                # duration_ms is informational; the dialog recomputes
                # from samples. Leaving 0.0 here keeps the dataclass
                # valid while avoiding the illusion of a real value.
                duration_ms=0.0,
                theta_delta_ratio=(float(tdr) if tdr else None),
                fast_theta_ratio=(float(ftr) if ftr else None),
                manual=False,
            ))
    cycles.sort(key=lambda c: (c.channel, c.zero_crossing_start))
    return cycles
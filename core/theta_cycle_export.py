"""
theta_cycle_export.py

CSV export / import for detected or edited theta cycles. All sample
fields in the CSV are RAW sample indices into the original recording,
matching the convention used by ripple_export.py and
theta_epoch_export.py.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

from core.theta_cycle_detector import ThetaCycle


CSV_FIELDNAMES = [
    "channel",
    "zero_crossing_start",
    "peak1",
    "zero_crossing_mid",
    "valley",
    "zero_crossing_end",
    "peak2",
    "amplitude",
    "duration_ms",
    "theta_delta_ratio",
    "fast_theta_ratio",
    "manual",
]


def _to_row(cycle: ThetaCycle, sample_offset: int) -> dict:
    """Every sample field is offset into raw recording coordinates.
    theta_delta_ratio and fast_theta_ratio are written as empty strings
    when None, so a correction that was off doesn't look like a real
    zero."""
    return {
        "channel": cycle.channel,
        "zero_crossing_start": cycle.zero_crossing_start + sample_offset,
        "peak1": cycle.peak1 + sample_offset,
        "zero_crossing_mid": cycle.zero_crossing_mid + sample_offset,
        "valley": cycle.valley + sample_offset,
        "zero_crossing_end": cycle.zero_crossing_end + sample_offset,
        "peak2": cycle.peak2 + sample_offset,
        "amplitude": round(cycle.amplitude, 6),
        "duration_ms": round(cycle.duration_ms, 3),
        "theta_delta_ratio": (
            "" if cycle.theta_delta_ratio is None
            else round(cycle.theta_delta_ratio, 4)
        ),
        "fast_theta_ratio": (
            "" if cycle.fast_theta_ratio is None
            else round(cycle.fast_theta_ratio, 4)
        ),
        "manual": cycle.manual,
    }


def export_cycles_to_csv(
    cycles: Sequence[ThetaCycle],
    sample_offset: int,
    output_path: str | Path,
) -> Path:
    """Write one row per cycle, sorted by (channel, zero_crossing_start)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [_to_row(c, sample_offset) for c in cycles]
    rows.sort(key=lambda r: (r["channel"], r["zero_crossing_start"]))

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def export_cycles_per_channel(
    cycles_by_channel: dict[int, Sequence[ThetaCycle]],
    sample_offset: int,
    output_dir: str | Path,
    filename_template: str = "theta_cycles_channel{channel}.csv",
) -> list[Path]:
    """One CSV per channel, matching ripple_export.export_ripples_per_channel.
    Channels with zero cycles still get a header-only file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for channel, cycles in cycles_by_channel.items():
        path = output_dir / filename_template.format(channel=channel)
        export_cycles_to_csv(cycles, sample_offset, path)
        written.append(path)
    return written


def read_cycles_from_csv(path: str | Path) -> list[ThetaCycle]:
    """Read a previously-exported cycle CSV. Sample indices in the file
    are absolute (raw recording coordinates); they're returned as-is."""
    path = Path(path)
    cycles: list[ThetaCycle] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tdr = row.get("theta_delta_ratio", "").strip()
            ftr = row.get("fast_theta_ratio", "").strip()
            cycles.append(ThetaCycle(
                channel=int(row["channel"]),
                zero_crossing_start=int(row["zero_crossing_start"]),
                peak1=int(row["peak1"]),
                zero_crossing_mid=int(row["zero_crossing_mid"]),
                valley=int(row["valley"]),
                zero_crossing_end=int(row["zero_crossing_end"]),
                peak2=int(row["peak2"]),
                amplitude=float(row["amplitude"]),
                duration_ms=float(row["duration_ms"]),
                theta_delta_ratio=(float(tdr) if tdr else None),
                fast_theta_ratio=(float(ftr) if ftr else None),
                manual=row["manual"].strip().lower() in ("true", "1", "yes"),
            ))
    cycles.sort(key=lambda c: (c.channel, c.zero_crossing_start))
    return cycles
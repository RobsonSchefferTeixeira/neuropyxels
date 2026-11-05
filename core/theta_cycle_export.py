"""
theta_cycle_export.py

CSV export / import for detected or edited theta cycles.

Column conventions
------------------
The six landmark columns are POSITIONAL: ref_1_smp .. ref_6_smp. What
each position means depends on the file's cycle_start_edge, which is
stored in the sidecar (params.cycle_start_edge and landmark_names),
NOT in the CSV. The CSV alone is not self-describing: without the
sidecar, there is no way to know whether ref_2 is a peak, a valley, or
a zero crossing.

  rising_zc : zc_rising_1, peak1, zc_falling_1, valley_1, zc_rising_2, peak2
  falling_zc: zc_falling_1, valley_1, zc_rising_2, peak2, zc_falling_2, valley_2
  peak      : peak1, zc_falling_1, valley_1, zc_rising_2, peak2, zc_falling_2
  valley    : valley_1, zc_rising_2, peak2, zc_falling_2, valley_2, zc_rising_3

Legacy CSVs written before the ref_N convention (with zc_start_smp,
peak1_smp, ... columns) are still readable; those were always
rising_zc, so read_cycles_from_csv reports that scheme regardless of
any external input.

Every *_smp column is an ABSOLUTE sample index into the original
.dat file. Every *_s column is derived at export time from the
engine's current time base (timestamps if loaded, else sample/sr).
duration_smp is (ref_5 - ref_1), one full oscillation.

No GUI dependency.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

from core.theta_cycle_detector import (
    ThetaCycle, LANDMARK_NAMES_BY_MODE, VALID_CYCLE_START_EDGES,
)


# Columns shared by every scheme.
_STATIC_HEAD = ["channel"]
_STATIC_TAIL = [
    "duration_smp",
    "duration_s",
    "amp",
    "theta_delta_ratio",
    "fast_theta_ratio",
]


def _ref_columns() -> list[str]:
    cols: list[str] = []
    cols.extend(f"ref_{i}_smp" for i in range(1, 7))
    cols.extend(f"ref_{i}_s"   for i in range(1, 7))
    return cols


CSV_FIELDNAMES = _STATIC_HEAD + _ref_columns() + _STATIC_TAIL


def _landmark_names_for(mode: str) -> list[str]:
    return list(LANDMARK_NAMES_BY_MODE.get(mode, []))


def _to_row(cycle: ThetaCycle, engine) -> dict:
    landmarks = cycle.landmarks()
    ref_smp = [int(x) for x in landmarks]
    duration_smp = ref_smp[4] - ref_smp[0]

    row = {
        "channel": int(cycle.channel),
    }
    for i, smp in enumerate(ref_smp, start=1):
        row[f"ref_{i}_smp"] = smp
    for i, smp in enumerate(ref_smp, start=1):
        row[f"ref_{i}_s"] = round(
            float(engine.sample_to_time(smp)), 6
        )
    row["duration_smp"] = duration_smp
    row["duration_s"] = round(duration_smp / float(engine.sr), 6)
    row["amp"] = round(float(cycle.amplitude), 6)
    row["theta_delta_ratio"] = (
        "" if cycle.theta_delta_ratio is None
        else round(float(cycle.theta_delta_ratio), 4)
    )
    row["fast_theta_ratio"] = (
        "" if cycle.fast_theta_ratio is None
        else round(float(cycle.fast_theta_ratio), 4)
    )
    return row


def export_cycles_to_csv(
    cycles: Sequence[ThetaCycle],
    engine,
    output_path: str | Path,
) -> Path:
    """Write one row per cycle. The meaning of the six ref_N columns
    (which landmark is ref_1, ref_2, ...) is NOT carried in the CSV;
    it lives in the sidecar's params.cycle_start_edge / landmark_names.
    A CSV without its sidecar has no way to interpret ref_N.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [_to_row(c, engine) for c in cycles]
    rows.sort(key=lambda r: (r["channel"], r["ref_1_smp"]))

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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for channel, cycles in cycles_by_channel.items():
        path = output_dir / filename_template.format(channel=channel)
        export_cycles_to_csv(cycles, engine, path)
        written.append(path)
    return written


def _read_old_scheme_row(row: dict) -> dict | None:
    """Translate an old (zc_*_smp) row into the six-landmark tuple
    used by read_cycles_from_csv. Returns None if required columns
    are missing. Old files were always rising_zc."""
    try:
        ref_smp = [
            int(row["zc_start_smp"]),
            int(row["peak1_smp"]),
            int(row["zc_mid_smp"]),
            int(row["valley_smp"]),
            int(row["zc_end_smp"]),
            int(row["peak2_smp"]),
        ]
    except (KeyError, ValueError, TypeError):
        return None
    return {"ref_smp": ref_smp, "cycle_start_edge": "rising_zc", "row": row}


def read_cycles_from_csv(
    path: str | Path,
    cycle_start_edge: str | None = None,
) -> tuple[list[ThetaCycle], str | None, list[str] | None]:
    """Read a cycle CSV. Returns (cycles, cycle_start_edge, landmark_names).

    cycle_start_edge must be supplied by the caller from the sidecar
    (sidecar["params"]["cycle_start_edge"]); the CSV itself no longer
    carries it. If not supplied, the reader returns cycles with the
    scheme set to None, which means the caller cannot interpret the
    ref_N columns correctly -- callers that need the scheme should pass
    it explicitly.

    Legacy zc_*_smp-column files (written before the ref_N convention)
    are still readable; those were always rising_zc, and the function
    reports that as the scheme regardless of the cycle_start_edge
    argument.

    landmark_names is always None from the CSV alone -- the dialog
    fills it from the sidecar.
    """
    path = Path(path)
    cycles: list[ThetaCycle] = []
    file_scheme: str | None = cycle_start_edge

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        is_legacy = "zc_start_smp" in header

        for row in reader:
            if is_legacy:
                translated = _read_old_scheme_row(row)
                if translated is None:
                    continue
                ref_smp = translated["ref_smp"]
                file_scheme = translated["cycle_start_edge"]
            else:
                try:
                    ref_smp = [int(row[f"ref_{i}_smp"]) for i in range(1, 7)]
                except (KeyError, ValueError, TypeError):
                    continue

            tdr = (row.get("theta_delta_ratio") or "").strip()
            ftr = (row.get("fast_theta_ratio") or "").strip()

            cycles.append(ThetaCycle(
                channel=int(row["channel"]),
                zero_crossing_start=ref_smp[0],
                peak1=ref_smp[1],
                zero_crossing_mid=ref_smp[2],
                valley=ref_smp[3],
                zero_crossing_end=ref_smp[4],
                peak2=ref_smp[5],
                amplitude=float(row["amp"]),
                duration_ms=0.0,
                theta_delta_ratio=(float(tdr) if tdr else None),
                fast_theta_ratio=(float(ftr) if ftr else None),
                manual=False,
            ))

    cycles.sort(key=lambda c: (c.channel, c.zero_crossing_start))
    return cycles, file_scheme, None
"""
phy_loader.py

Reads Kilosort4 / Phy2 output folders. Supports the common file layout:

Required:
    spike_times.npy     -- (n_spikes,) int64, spike sample indices
    spike_clusters.npy  -- (n_spikes,) int32, cluster id per spike

Optional but used if present:
    cluster_info.tsv    -- per-unit metadata (Phy or Kilosort column layout)
    cluster_group.tsv   -- legacy Phy quality labels ('good'/'mua'/'noise')
    channel_map.npy     -- (n_channels,) int, mapping row-index -> hardware ch

App-specific:
    cluster_info_reclassified.tsv
        Written by the app when the user reclassifies units. Same layout
        as cluster_info.tsv, with an extra trailing column
        'reclassified_group' holding the app's own classification. Loaded
        in preference to cluster_info.tsv when present.

No GUI dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


RECLASSIFIED_FILENAME = "cluster_info_reclassified.tsv"
RECLASSIFIED_COLUMN = "reclassified_group"

# Canonical class values, in descending-quality order. Anything outside
# this set is preserved verbatim (we never remap unknown labels), it just
# won't appear in the filter dropdown's preset options.
KNOWN_CLASSES = ("excellent", "good", "poor", "unsorted", "mua", "noise")


@dataclass
class Unit:
    """One sorted unit (cluster)."""
    cluster_id: int
    channel: Optional[int] = None
    depth: Optional[float] = None
    quality: Optional[str] = None        # app's classification (from reclassified file if present, else from group/kslabel)
    kslabel: Optional[str] = None
    group: Optional[str] = None          # Phy's 'group' column, untouched
    n_spikes: int = 0
    firing_rate: Optional[float] = None
    amplitude: Optional[float] = None
    contamination: Optional[float] = None
    shank: Optional[int] = None
    # True if this unit's `quality` value came from the reclassified file
    # rather than from cluster_info.tsv/group. Used by the panel to show
    # which rows have already been reclassified in a previous session.
    reclassified: bool = False

    @property
    def display_name(self) -> str:
        bits = [f"#{self.cluster_id}"]
        if self.channel is not None:
            bits.append(f"ch{self.channel}")
        if self.quality:
            bits.append(self.quality)
        return "  ".join(bits)


@dataclass
class PhyData:
    """Everything the loader managed to read from a Phy folder."""
    folder: Path
    spike_times: np.ndarray
    spike_clusters: np.ndarray
    units: dict[int, Unit] = field(default_factory=dict)
    # Raw header + rows of the source cluster_info file, kept so the
    # save routine can write back an exact copy with one added column.
    info_header: list[str] = field(default_factory=list)
    info_rows: list[list[str]] = field(default_factory=list)
    # Path of the source info file actually used at load time.
    info_source: Optional[Path] = None

    def spikes_for_unit(self, cluster_id: int) -> np.ndarray:
        mask = self.spike_clusters == cluster_id
        return np.sort(self.spike_times[mask])

    def spikes_for_unit_in_window(
        self, cluster_id: int, start_sample: int, end_sample: int
    ) -> np.ndarray:
        spikes = self.spikes_for_unit(cluster_id)
        if spikes.size == 0:
            return spikes
        lo = int(np.searchsorted(spikes, start_sample, side="left"))
        hi = int(np.searchsorted(spikes, end_sample, side="left"))
        return spikes[lo:hi]


class PhyLoadError(RuntimeError):
    pass


def _read_tsv_columns(path: Path) -> tuple[list[str], list[list[str]]]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip() and not ln.startswith("#")]
    if not lines:
        return [], []
    header = lines[0].split("\t")
    rows = [ln.split("\t") for ln in lines[1:]]
    return header, rows


def _find_column(header: list[str], *candidates: str) -> Optional[int]:
    lowered = [h.strip().lower() for h in header]
    for cand in candidates:
        cand_lower = cand.strip().lower()
        if cand_lower in lowered:
            return lowered.index(cand_lower)
    return None


def _safe_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _safe_int(s: str) -> Optional[int]:
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None


def load_phy_folder(folder: str | Path) -> PhyData:
    folder = Path(folder)
    if not folder.is_dir():
        raise PhyLoadError(f"Not a folder: {folder}")

    spike_times_path = folder / "spike_times.npy"
    spike_clusters_path = folder / "spike_clusters.npy"

    if not spike_times_path.exists():
        raise PhyLoadError(
            f"spike_times.npy not found in {folder}. Is this really a "
            f"Kilosort/Phy output folder?"
        )
    if not spike_clusters_path.exists():
        raise PhyLoadError(
            f"spike_clusters.npy not found in {folder}. It is created by "
            f"Kilosort4 during the sorting step; without it we can't map "
            f"spikes to units."
        )

    try:
        spike_times = np.load(spike_times_path).flatten().astype(np.int64)
        spike_clusters = np.load(spike_clusters_path).flatten().astype(np.int64)
    except Exception as exc:
        raise PhyLoadError(f"Failed to read spike arrays: {exc}") from exc

    if spike_times.shape != spike_clusters.shape:
        raise PhyLoadError(
            f"spike_times has {spike_times.size} entries but "
            f"spike_clusters has {spike_clusters.size}; they must match 1:1."
        )

    data = PhyData(
        folder=folder,
        spike_times=spike_times,
        spike_clusters=spike_clusters,
    )

    # ---- Choose source: reclassified file wins if present ----
    reclassified_path = folder / RECLASSIFIED_FILENAME
    plain_info_path = folder / "cluster_info.tsv"

    if reclassified_path.exists():
        source_path = reclassified_path
        data.info_source = source_path
    elif plain_info_path.exists():
        source_path = plain_info_path
        data.info_source = source_path
    else:
        source_path = None

    if source_path is not None:
        _parse_cluster_info(source_path, data, is_reclassified=(source_path == reclassified_path))

    # ---- cluster_group.tsv (legacy Phy; group column only) ----
    group_path = folder / "cluster_group.tsv"
    if group_path.exists() and data.info_source is None:
        # Only apply if we didn't already read a richer source.
        _parse_cluster_group(group_path, data)

    # ---- Ensure every cluster has a Unit ----
    for cid in np.unique(spike_clusters):
        cid = int(cid)
        if cid not in data.units:
            data.units[cid] = Unit(
                cluster_id=cid,
                n_spikes=int(np.sum(spike_clusters == cid)),
            )

    # ---- Fill n_spikes from the raw spike arrays if not present ----
    for cid, unit in data.units.items():
        if unit.n_spikes == 0:
            unit.n_spikes = int(np.sum(spike_clusters == cid))

    return data


def _parse_cluster_info(path: Path, data: PhyData, is_reclassified: bool) -> None:
    """Populate data.units from a cluster_info-style TSV. Reads the
    app's 'reclassified_group' column if present; otherwise leaves
    Unit.reclassified as False and Unit.quality sourced from Phy's
    'group' / 'KSLabel'."""
    header, rows = _read_tsv_columns(path)
    if not header:
        return

    data.info_header = header
    data.info_rows = rows

    idx_cluster = _find_column(header, "cluster_id", "id", "cluster")
    idx_ch = _find_column(header, "ch", "channel", "best_channel")
    idx_depth = _find_column(header, "depth")
    idx_ks = _find_column(header, "KSLabel", "kslabel", "ks_label")
    idx_group = _find_column(header, "group")
    idx_fr = _find_column(header, "fr", "firing_rate", "fire_rate")
    idx_n = _find_column(header, "n_spikes", "num_spikes", "spikes")
    idx_amp = _find_column(header, "Amplitude", "amplitude", "amp")
    idx_cont = _find_column(header, "ContamPct", "contamination", "contam")
    idx_sh = _find_column(header, "sh", "shank", "shank_id")
    idx_reclass = _find_column(header, RECLASSIFIED_COLUMN)

    if idx_cluster is None:
        return

    for row in rows:
        if idx_cluster >= len(row):
            continue
        cid = _safe_int(row[idx_cluster])
        if cid is None:
            continue

        unit = data.units.get(cid) or Unit(cluster_id=cid)

        if idx_ch is not None and idx_ch < len(row):
            unit.channel = _safe_int(row[idx_ch])
        if idx_depth is not None and idx_depth < len(row):
            unit.depth = _safe_float(row[idx_depth])
        if idx_ks is not None and idx_ks < len(row):
            unit.kslabel = row[idx_ks].strip() or None
        if idx_group is not None and idx_group < len(row):
            unit.group = row[idx_group].strip() or None
        if idx_fr is not None and idx_fr < len(row):
            unit.firing_rate = _safe_float(row[idx_fr])
        if idx_n is not None and idx_n < len(row):
            n = _safe_int(row[idx_n])
            if n is not None:
                unit.n_spikes = n
        if idx_amp is not None and idx_amp < len(row):
            unit.amplitude = _safe_float(row[idx_amp])
        if idx_cont is not None and idx_cont < len(row):
            unit.contamination = _safe_float(row[idx_cont])
        if idx_sh is not None and idx_sh < len(row):
            unit.shank = _safe_int(row[idx_sh])

        # Classification: prefer the app's reclassified column when the
        # source file has it and the value isn't empty; otherwise fall
        # back to Phy's own labels.
        reclass_val = None
        if idx_reclass is not None and idx_reclass < len(row):
            reclass_val = row[idx_reclass].strip()

        if reclass_val:
            unit.quality = reclass_val
            unit.reclassified = True
        else:
            unit.quality = unit.group or unit.kslabel
            unit.reclassified = False

        data.units[cid] = unit


def _parse_cluster_group(path: Path, data: PhyData) -> None:
    header, rows = _read_tsv_columns(path)
    if not header:
        return
    idx_cluster = _find_column(header, "cluster_id", "id", "cluster")
    idx_group = _find_column(header, "group", "KSLabel", "kslabel")
    if idx_cluster is None or idx_group is None:
        return
    for row in rows:
        if idx_cluster >= len(row) or idx_group >= len(row):
            continue
        cid = _safe_int(row[idx_cluster])
        if cid is None:
            continue
        group = row[idx_group].strip() or None
        unit = data.units.get(cid) or Unit(cluster_id=cid)
        unit.group = group
        if not unit.reclassified:
            unit.quality = group or unit.quality
        data.units[cid] = unit


def save_reclassified(data: PhyData, units: dict[int, Unit]) -> Path:
    """Write cluster_info_reclassified.tsv into data.folder.

    Preserves every column and row order from the source info file, and
    appends one column ('reclassified_group') holding the app's current
    classification for each unit. Units the user has never touched get
    an empty value in that column.

    If no source info file was loaded (rare; the folder has spike files
    but no cluster_info.tsv), a minimal two-column file is written
    instead: cluster_id, reclassified_group.
    """
    folder = Path(data.folder)
    out_path = folder / RECLASSIFIED_FILENAME

    if data.info_header:
        header = list(data.info_header)
        rows = [list(r) for r in data.info_rows]
        # Locate cluster_id column to map rows to units.
        idx_cluster = _find_column(header, "cluster_id", "id", "cluster")
        if idx_cluster is None:
            raise PhyLoadError(
                "Source info file has no cluster_id column; cannot save."
            )
        # Append reclassified_group if not already present.
        if RECLASSIFIED_COLUMN not in [h.strip().lower() for h in header]:
            header.append(RECLASSIFIED_COLUMN)
        idx_reclass = _find_column(header, RECLASSIFIED_COLUMN)

        # Pad every row to the new column count with empty strings.
        target_len = len(header)
        for row in rows:
            if len(row) < target_len:
                row.extend([""] * (target_len - len(row)))
            cid = _safe_int(row[idx_cluster])
            if cid is None:
                continue
            unit = units.get(cid)
            if unit is None:
                continue
            # Empty for never-touched units; the app's class otherwise.
            row[idx_reclass] = unit.quality if unit.reclassified else ""

        with open(out_path, "w", encoding="utf-8", newline="") as f:
            f.write("\t".join(header) + "\n")
            for row in rows:
                f.write("\t".join(row) + "\n")
    else:
        # Fallback: minimal file.
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            f.write(f"cluster_id\t{RECLASSIFIED_COLUMN}\n")
            for cid in sorted(units.keys()):
                unit = units[cid]
                value = unit.quality if unit.reclassified else ""
                f.write(f"{cid}\t{value or ''}\n")

    return out_path
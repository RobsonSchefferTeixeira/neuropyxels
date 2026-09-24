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

The class_info / cluster_info layouts from Phy have changed across
versions. Rather than hard-coding column names, we look them up
case-insensitively and fall back to scanning for a column whose name
starts with the same prefix (e.g. 'KSLabel' vs 'kslabel' vs 'group').
Whatever isn't found is left as None in the Unit object, so the UI can
degrade gracefully rather than crash.

No GUI dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class Unit:
    """One sorted unit (cluster). All fields past `cluster_id` are
    optional -- what's actually populated depends on which files were
    present in the Phy folder."""
    cluster_id: int
    channel: Optional[int] = None
    depth: Optional[float] = None
    quality: Optional[str] = None        # 'good' / 'mua' / 'noise' / None
    kslabel: Optional[str] = None        # Kilosort's own quality label
    group: Optional[str] = None          # Phy's 'group' column, if present
    n_spikes: int = 0
    firing_rate: Optional[float] = None
    amplitude: Optional[float] = None
    contamination: Optional[float] = None
    shank: Optional[int] = None

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
    spike_times: np.ndarray              # (n_spikes,) int64
    spike_clusters: np.ndarray           # (n_spikes,) int32
    units: dict[int, Unit] = field(default_factory=dict)

    def spikes_for_unit(self, cluster_id: int) -> np.ndarray:
        """Return spike sample indices for one unit, sorted ascending."""
        mask = self.spike_clusters == cluster_id
        return np.sort(self.spike_times[mask])

    def spikes_for_unit_in_window(
        self, cluster_id: int, start_sample: int, end_sample: int
    ) -> np.ndarray:
        """Return spike sample indices for one unit in
        [start_sample, end_sample). Uses searchsorted on the sorted
        per-unit spike times rather than a full-array boolean mask,
        so it's O(log n + k) instead of O(n)."""
        spikes = self.spikes_for_unit(cluster_id)
        if spikes.size == 0:
            return spikes
        lo = int(np.searchsorted(spikes, start_sample, side="left"))
        hi = int(np.searchsorted(spikes, end_sample, side="left"))
        return spikes[lo:hi]


class PhyLoadError(RuntimeError):
    pass


def _read_tsv_columns(path: Path) -> tuple[list[str], list[list[str]]]:
    """Return (header, rows) from a tab-separated file, skipping any
    comment lines that start with '#'."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip() and not ln.startswith("#")]
    if not lines:
        return [], []
    header = lines[0].split("\t")
    rows = [ln.split("\t") for ln in lines[1:]]
    return header, rows


def _find_column(header: list[str], *candidates: str) -> Optional[int]:
    """Case-insensitive lookup of a column name. Returns its index in
    `header`, or None if no candidate matches."""
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
    """
    Read a Phy / Kilosort output folder. Raises PhyLoadError if the
    required files (spike_times.npy, spike_clusters.npy) can't be found
    or read.
    """
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
            f"spike_clusters has {spike_clusters.size}; they must match "
            f"1:1."
        )

    data = PhyData(
        folder=folder,
        spike_times=spike_times,
        spike_clusters=spike_clusters,
    )

    # ---- cluster_info.tsv (optional but common) ----
    info_path = folder / "cluster_info.tsv"
    if info_path.exists():
        _parse_cluster_info(info_path, data)

    # ---- cluster_group.tsv (legacy Phy; overrides quality if present) ----
    group_path = folder / "cluster_group.tsv"
    if group_path.exists():
        _parse_cluster_group(group_path, data)

    # ---- Ensure every cluster in spike_clusters has a Unit entry ----
    for cid in np.unique(spike_clusters):
        cid = int(cid)
        if cid not in data.units:
            data.units[cid] = Unit(cluster_id=cid, n_spikes=int(np.sum(spike_clusters == cid)))

    return data


def _parse_cluster_info(path: Path, data: PhyData) -> None:
    """Populate data.units from cluster_info.tsv. Column names vary
    across Phy versions, so we look them up by name with fallbacks."""
    header, rows = _read_tsv_columns(path)
    if not header:
        return

    # Column indices. The Phy2 layout the user showed uses 'cluster_id'
    # but Kilosort4 native uses 'cluster_id' too; 'id' is a Phy1 legacy
    # name. We accept any of these.
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

        # Prefer group over kslabel for the "quality" display field --
        # Phy's group column reflects manual curation; kslabel is
        # Kilosort's automated guess.
        unit.quality = unit.group or unit.kslabel

        # Fill in n_spikes from the raw spike array if the tsv didn't
        # provide it.
        if unit.n_spikes == 0:
            unit.n_spikes = int(np.sum(data.spike_clusters == cid))

        data.units[cid] = unit


def _parse_cluster_group(path: Path, data: PhyData) -> None:
    """cluster_group.tsv is a two-column file: cluster_id, group.
    When present it's the most authoritative source for quality
    (Phy writes it whenever the user saves curation state)."""
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
        unit.quality = group or unit.quality
        data.units[cid] = unit
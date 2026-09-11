"""
neuropixels_probe_extractor.py

Extract probe geometry from Open Ephys settings.xml files and generate
Kilosort4-compatible probe JSON files for ALL streams (AP and LFP).

Kilosort4 (Python) expects a probe dict with:
    chanMap  : (n_chan,) int array, 0-indexed channel indices
    xc       : (n_chan,) float array, electrode x-positions in µm
    yc       : (n_chan,) float array, electrode y-positions in µm
    kcoords  : (n_chan,) int array, shank id per channel (0-indexed)
    n_chan   : int, total number of channels

Usage:
    from neuropixels_probe_extractor import (
        extract_probes_from_settings,
        save_kilosort_probe,
        plot_probe_layout,
        plot_all_probes,
        print_probe_summary,
        process_settings_file,
    )

    probes = extract_probes_from_settings('path/to/settings.xml')
    print_probe_summary(probes)
    save_kilosort_probe(probes, output_dir='./probe_maps')
"""

import json
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# EXTRACTION FUNCTIONS
# =============================================================================

def extract_probes_from_settings(xml_file_path: str) -> Dict:
    """
    Extract ALL enabled Neuropixels streams (AP and LFP) from settings.xml.

    LFP streams share the same electrode coordinates as their AP counterpart.
    If a stream lacks coordinates, they are copied from a sibling stream with
    the same device_name.

    Args:
        xml_file_path: Path to the settings.xml file.

    Returns:
        dict with keys:
            'record_path' : str  — recordPath from the CONTROLPANEL element
            'probes'      : dict — {probe_key: probe_data, ...}
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    control_panel = root.find(".//CONTROLPANEL")
    record_path = control_panel.get("recordPath", "") if control_panel is not None else ""

    result: Dict = {"record_path": record_path, "probes": {}}

    neuropix_processors = root.findall(".//PROCESSOR[@pluginName='Neuropix-PXI']")

    for processor in neuropix_processors:
        node_id = processor.get("nodeId")

        # ------------------------------------------------------------------
        # 1. Collect metadata for ALL streams (no AP/LFP filtering).
        # ------------------------------------------------------------------
        stream_info: Dict[str, Dict] = {}
        for stream in processor.findall(".//STREAM"):
            stream_name = stream.get("name", "")
            stream_info[stream_name] = {
                "stream_name": stream_name,
                "sample_rate": float(stream.get("sample_rate", "0")),
                "channel_count": int(stream.get("channel_count", "0")),
                "device_name": stream.get("device_name", ""),
                "coordinates": {"x": [], "y": [], "channels": []},
                "shanks": {"ids": [], "mapping": {}},
                "num_shanks": 1,
                "electrode_config": "NONE",
                "serial_number": "",
                "enabled": True,
                "node_id": node_id,
            }

        # ------------------------------------------------------------------
        # 2. Extract electrode coordinates from NP_PROBE elements.
        # ------------------------------------------------------------------
        for np_probe in processor.findall(".//NP_PROBE"):
            if np_probe.get("isEnabled", "1") != "1":
                continue

            probe_name = np_probe.get("probe_name", "")
            serial_number = np_probe.get("probe_serial_number", "")
            electrode_config = np_probe.get("electrodeConfigurationPreset", "NONE")

            coords, shank_info = _extract_electrode_coordinates(np_probe)

            # Find ALL streams that belong to this physical probe (by device_name).
            matching = [
                s for s, d in stream_info.items() if d["device_name"] == probe_name
            ]
            if not matching:
                # Fallback: match by probe family name.
                for family in ("Neuropixels 2.0", "Neuropixels 1.0"):
                    if family in probe_name:
                        matching = [
                            s for s, d in stream_info.items() if family in d["device_name"]
                        ]
                        break

            for stream_name in matching:
                stream_info[stream_name].update(
                    {
                        "serial_number": serial_number,
                        "coordinates": coords,
                        "shanks": shank_info,
                        "num_shanks": len(set(shank_info["ids"])) if shank_info["ids"] else 1,
                        "electrode_config": electrode_config,
                        "device_name": probe_name,
                    }
                )

        # ------------------------------------------------------------------
        # 3. Copy coordinates from sibling streams for any stream that lacks them.
        # ------------------------------------------------------------------
        for stream_name, stream_data in stream_info.items():
            if not stream_data["coordinates"].get("x"):
                # Look for another stream from the same device that has coordinates.
                for other_name, other_data in stream_info.items():
                    if (
                        other_data["device_name"] == stream_data["device_name"]
                        and other_data["coordinates"].get("x")
                    ):
                        stream_data["coordinates"] = other_data["coordinates"].copy()
                        stream_data["shanks"] = other_data["shanks"].copy()
                        stream_data["num_shanks"] = other_data["num_shanks"]
                        stream_data["electrode_config"] = other_data["electrode_config"]
                        stream_data["serial_number"] = other_data["serial_number"]
                        break

        # ------------------------------------------------------------------
        # 4. Store results; warn about streams that still lack coordinates.
        # ------------------------------------------------------------------
        for stream_name, stream_data in stream_info.items():
            if not stream_data["coordinates"].get("x"):
                warnings.warn(
                    f"Stream '{stream_name}' (node {node_id}) has no electrode "
                    "coordinates and no sibling to copy from — skipping.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            probe_key = f"{stream_name}_node{node_id}"
            result["probes"][probe_key] = stream_data

    return result


def _extract_electrode_coordinates(np_probe_element) -> Tuple[Dict, Dict]:
    """
    Extract electrode x/y coordinates and shank IDs from an NP_PROBE element.

    The CHANNELS attribute stores per-channel values in the format
    "electrode:shank" (e.g. "48:1" → electrode 48, shank 1).
    """
    coordinates: Dict = {"x": [], "y": [], "channels": []}
    shank_info: Dict = {"ids": [], "mapping": {}}

    xpos_elem = np_probe_element.find(".//ELECTRODE_XPOS")
    ypos_elem = np_probe_element.find(".//ELECTRODE_YPOS")
    channels_elem = np_probe_element.find(".//CHANNELS")

    if xpos_elem is not None and ypos_elem is not None:
        x_dict = {
            int(k[2:]): float(v)
            for k, v in xpos_elem.attrib.items()
            if k.startswith("CH")
        }
        y_dict = {
            int(k[2:]): float(v)
            for k, v in ypos_elem.attrib.items()
            if k.startswith("CH")
        }

        all_channels = sorted(x_dict.keys() & y_dict.keys())
        coordinates["channels"] = all_channels
        coordinates["x"] = [x_dict[ch] for ch in all_channels]
        coordinates["y"] = [y_dict[ch] for ch in all_channels]

    if channels_elem is not None:
        for key, value in channels_elem.attrib.items():
            if not key.startswith("CH"):
                continue
            try:
                ch_num = int(key[2:])
                shank_info["mapping"][ch_num] = (
                    int(value.split(":")[1]) if ":" in value else 0
                )
            except (ValueError, IndexError):
                shank_info["mapping"][int(key[2:])] = 0

    shank_info["ids"] = [
        shank_info["mapping"].get(ch, 0) for ch in coordinates["channels"]
    ]

    return coordinates, shank_info


# =============================================================================
# KILOSORT PROBE DICTIONARY
# =============================================================================

def create_kilosort_probe_dict(probe_data: Dict) -> Dict:
    """
    Build a Kilosort4-compatible probe dictionary from extracted probe data.

    Channels are sorted by shank index first, then by depth (y-coordinate),
    which is the ordering Kilosort4 expects for multi-shank probes.

    Kilosort4 key reference
    -----------------------
    chanMap   : 0-indexed channel indices (hardware channel numbers)
    xc        : electrode x-positions in µm
    yc        : electrode y-positions in µm
    kcoords   : shank id per channel, 0-indexed
    n_chan    : number of recording channels passed to Kilosort

    Returns:
        dict with numpy arrays ready for JSON serialisation.

    Raises:
        ValueError: if the probe_data has no coordinate information.
    """
    coords = probe_data["coordinates"]
    shanks = probe_data.get("shanks", {})

    if not coords.get("x") or not coords.get("y"):
        raise ValueError("No coordinate information for probe.")

    channels = np.array(coords["channels"], dtype=np.int64)
    xcoords = np.array(coords["x"], dtype=np.float64)
    ycoords = np.array(coords["y"], dtype=np.float64)
    shank_ids = (
        np.array(shanks["ids"], dtype=np.int64)
        if shanks.get("ids")
        else np.zeros(len(channels), dtype=np.int64)
    )

    # Sort: primary key = shank, secondary key = depth (ascending y).
    sort_idx = np.lexsort((ycoords, shank_ids))
    channels = channels[sort_idx]
    xcoords = xcoords[sort_idx]
    ycoords = ycoords[sort_idx]
    shank_ids = shank_ids[sort_idx]

    return {
        "chanMap": channels,
        "xc": xcoords,
        "yc": ycoords,
        "kcoords": shank_ids,
        "n_chan": int(len(channels)),
    }


# =============================================================================
# SAVE FUNCTION (.json only)
# =============================================================================

def save_kilosort_probe(
    probes: Dict,
    output_dir: Union[str, Path] = None,
    prefix: str = "probe",
) -> List[str]:
    """
    Save Kilosort4 probe dictionaries as .json files for all streams.

    Args:
        probes     : dict returned by extract_probes_from_settings.
        output_dir : directory to write files into (created if absent).
        prefix     : filename prefix, e.g. "probe" → "probe_<key>.json".

    Returns:
        List of paths to the saved files.
    """
    saved_files: List[str] = []

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    for probe_key, probe_data in probes["probes"].items():
        if not probe_data["coordinates"].get("x"):
            print(f"  Skipping '{probe_key}': no electrode coordinates.")
            continue

        try:
            probe_dict = create_kilosort_probe_dict(probe_data)
        except ValueError as exc:
            print(f"  ERROR for '{probe_key}': {exc}")
            continue

        identifier = probe_key.replace("-", "_")
        filename = f"{prefix}_{identifier}.json"
        json_path = (output_dir / filename) if output_dir else Path(filename)

        _save_probe_as_json(probe_dict, json_path)
        saved_files.append(str(json_path))
        print(f"  Saved: {json_path}")

    return saved_files


def _save_probe_as_json(probe_dict: Dict, output_path: Union[str, Path]) -> None:
    """
    Serialise a Kilosort4 probe dict to JSON.

    Numpy arrays are converted to plain Python lists; bool arrays become
    lists of Python booleans (not numpy.bool_) so json.dump handles them.
    """
    json_compatible = {
        "chanMap": probe_dict["chanMap"].tolist(),
        "xc": probe_dict["xc"].tolist(),
        "yc": probe_dict["yc"].tolist(),
        "kcoords": probe_dict["kcoords"].tolist(),
        "n_chan": int(probe_dict["n_chan"]),
    }
    with open(output_path, "w") as f:
        json.dump(json_compatible, f, indent=2)


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_probe_layout(
    probe_data: Dict,
    show_channel_labels: bool = False,
    figsize: Tuple[int, int] = (10, 8),
    ax: Optional[plt.Axes] = None,
    max_labels_per_column: int = 10,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the electrode layout of a single probe."""
    coords = probe_data["coordinates"]
    shanks = probe_data.get("shanks", {})

    if not coords.get("x") or not coords.get("y"):
        raise ValueError(
            f"No coordinate information for probe {probe_data.get('stream_name', 'unknown')}"
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure

    x_arr = np.array(coords["x"])
    y_arr = np.array(coords["y"])
    ch_arr = np.array(coords["channels"])

    has_multiple_shanks = shanks.get("ids") and len(set(shanks["ids"])) > 1

    if has_multiple_shanks:
        unique_shanks = sorted(set(shanks["ids"]))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_shanks)))
        shank_to_color = dict(zip(unique_shanks, colors))

        for shank_id in unique_shanks:
            mask = np.array(shanks["ids"]) == shank_id
            ax.scatter(
                x_arr[mask], y_arr[mask],
                c=[shank_to_color[shank_id]],
                s=20, alpha=0.7, label=f"Shank {shank_id}",
            )
    else:
        ax.scatter(x_arr, y_arr, c="steelblue", s=20, alpha=0.7)

    if show_channel_labels:
        _annotate_channel_labels(ax, x_arr, y_arr, ch_arr, max_labels_per_column, fontsize=6)

    ax.set_xlabel("X Coordinate (µm)", fontsize=12)
    _set_x_ticks_from_electrodes(ax, x_arr)
    _set_x_limits_with_padding(ax, x_arr)
    ax.set_ylabel("Depth (µm) — 0 at tip, increasing toward brain surface", fontsize=12)
    _set_y_ticks_from_electrodes(ax, y_arr)

    title_parts = [
        probe_data.get("stream_name", "Probe"),
        probe_data.get("device_name", ""),
    ]
    if probe_data.get("serial_number"):
        title_parts.append(f"Serial: {probe_data['serial_number']}")
    ax.set_title("\n".join(filter(None, title_parts)), fontsize=13)

    _add_legend(ax, shanks, has_multiple_shanks, n_electrodes=len(coords["channels"]))
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_all_probes(
    probes: Dict,
    show_channel_labels: bool = False,
    figsize: Tuple[int, int] = (6, 10),
    max_labels_per_column: int = 10,
) -> None:
    """Plot all streams in a single figure with subplots."""
    active_probes = {
        k: v for k, v in probes["probes"].items() if v["coordinates"].get("x")
    }

    if not active_probes:
        print("No probes with coordinates to plot.")
        return

    n_probes = len(active_probes)
    n_cols = min(2, n_probes)
    n_rows = (n_probes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize[0] * n_cols, figsize[1] * n_rows),
        constrained_layout=True,
    )

    axes_flat = [axes] if n_probes == 1 else np.array(axes).flatten().tolist()

    for ax, (key, probe) in zip(axes_flat, active_probes.items()):
        try:
            plot_probe_layout(
                probe,
                show_channel_labels=show_channel_labels,
                max_labels_per_column=max_labels_per_column,
                ax=ax,
            )
            # Compact title for the multi-probe overview
            device = probe.get("device_name", "").replace("Neuropixels ", "NP")
            ax.set_title(f"{key}\n{device}", fontsize=9)
        except ValueError as exc:
            ax.text(0.5, 0.5, str(exc), ha="center", va="center", transform=ax.transAxes)
            ax.set_title(key)

    # Hide unused subplots
    for ax in axes_flat[n_probes:]:
        ax.set_visible(False)

    record_label = Path(probes.get("record_path", "")).name
    fig.suptitle(
        f"Probe Layouts — {record_label}" if record_label else "Probe Layouts",
        fontsize=14,
        fontweight="bold",
    )
    plt.show()


# =============================================================================
# PLOTTING HELPERS
# =============================================================================

def _annotate_channel_labels(
    ax: plt.Axes,
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    ch_arr: np.ndarray,
    max_labels_per_column: int,
    fontsize: int = 6,
) -> None:
    """Annotate a thinned subset of channel numbers on the plot."""
    for x_val in np.unique(x_arr):
        mask = x_arr == x_val
        y_col = y_arr[mask]
        ch_col = ch_arr[mask]

        order = np.argsort(y_col)
        y_col = y_col[order]
        ch_col = ch_col[order]

        n = len(ch_col)
        if n <= max_labels_per_column:
            indices = list(range(n))
        else:
            step = n / (max_labels_per_column - 1)
            indices = [0] + [int(i * step) for i in range(1, max_labels_per_column - 1)] + [n - 1]

        for idx in indices:
            ax.annotate(
                str(ch_col[idx]),
                (x_val, y_col[idx]),
                xytext=(3, 0),
                textcoords="offset points",
                fontsize=fontsize,
                alpha=0.85,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"),
            )


def _set_x_ticks_from_electrodes(ax: plt.Axes, x_arr: np.ndarray) -> None:
    unique_x = np.unique(x_arr)
    ax.set_xticks(unique_x)
    ax.set_xticklabels([f"{x:.0f}" for x in unique_x], rotation=45, ha="right", fontsize=8)


def _set_x_limits_with_padding(ax: plt.Axes, x_arr: np.ndarray, padding: float = 100) -> None:
    ax.set_xlim(x_arr.min() - padding, x_arr.max() + padding)


def _set_y_ticks_from_electrodes(ax: plt.Axes, y_arr: np.ndarray, max_ticks: int = 12) -> None:
    unique_y = np.unique(y_arr)
    if len(unique_y) > max_ticks:
        step = max(1, len(unique_y) // (max_ticks - 1))
        unique_y = unique_y[::step]
    ax.set_yticks(unique_y)
    ax.set_yticklabels([f"{y:.0f}" for y in unique_y], fontsize=8)


def _add_legend(
    ax: plt.Axes, shanks: Dict, has_multiple_shanks: bool, n_electrodes: Optional[int] = None
) -> None:
    if has_multiple_shanks:
        ax.legend(loc="best", fontsize=10)
    else:
        label = f"Single shank ({n_electrodes} electrodes)" if n_electrodes else "Single shank"
        ax.scatter([], [], c="steelblue", s=20, alpha=0.7, label=label)
        ax.legend(loc="best", fontsize=10)


# =============================================================================
# SUMMARY
# =============================================================================

def print_probe_summary(probes: Dict) -> None:
    """Print a formatted summary of all extracted streams."""
    sep = "=" * 72
    print(f"\n{sep}")
    print("NEUROPIXELS STREAM EXTRACTION SUMMARY")
    print(sep)
    print(f"Record path : {probes['record_path']}")
    print(f"Streams     : {len(probes['probes'])}")
    print(f"\n{'-' * 72}")

    for key, probe in probes["probes"].items():
        coords = probe["coordinates"]
        x_range = (
            f"{min(coords['x']):.0f} – {max(coords['x']):.0f} µm"
            if coords.get("x") else "N/A"
        )
        y_range = (
            f"{min(coords['y']):.0f} – {max(coords['y']):.0f} µm"
            if coords.get("y") else "N/A"
        )
        print(f"\n  Key            : {key}")
        print(f"  Device         : {probe.get('device_name', 'N/A')}")
        print(f"  Serial         : {probe.get('serial_number', 'N/A')}")
        print(f"  Sample rate    : {probe.get('sample_rate', 0):.0f} Hz")
        print(f"  Channel count  : {probe.get('channel_count', 0)}")
        print(f"  Electrodes     : {len(coords.get('x', []))}")
        print(f"  Shanks         : {probe.get('num_shanks', 1)}")
        print(f"  Electrode cfg  : {probe.get('electrode_config', 'NONE')}")
        print(f"  X-range        : {x_range}")
        print(f"  Y-range        : {y_range}")

    print(f"\n{sep}\n")


# =============================================================================
# CONVENIENCE WRAPPER
# =============================================================================

def process_settings_file(
    xml_file_path: str,
    output_dir: Optional[str] = None,
    plot: bool = True,
    save_json: bool = True,
) -> Dict:
    """
    One-shot helper: parse settings.xml, print summary, optionally save and plot.

    Args:
        xml_file_path : path to settings.xml.
        output_dir    : directory to write probe JSON files into.
        plot          : if True, show probe layout plots.
        save_json     : if True, write .json files for Kilosort.

    Returns:
        Probe dictionary from extract_probes_from_settings.
    """
    print(f"Loading settings from: {xml_file_path}")
    probes = extract_probes_from_settings(xml_file_path)

    if not probes["probes"]:
        warnings.warn("No enabled streams found in settings file.", RuntimeWarning)
        return probes

    print_probe_summary(probes)

    if save_json and output_dir:
        saved = save_kilosort_probe(probes, output_dir=output_dir)
        print(f"\nSaved {len(saved)} JSON file(s).")

    if plot:
        if len(probes["probes"]) == 1:
            probe_data = next(iter(probes["probes"].values()))
            fig, _ = plot_probe_layout(probe_data)
            plt.show()
        else:
            plot_all_probes(probes)

    return probes


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        process_settings_file(
            xml_file_path=sys.argv[1],
            output_dir=sys.argv[2] if len(sys.argv) > 2 else "./probe_maps",
        )
    else:
        print("Usage : python neuropixels_probe_extractor.py <settings.xml> [output_dir]")
        print("Example: python neuropixels_probe_extractor.py settings.xml ./probe_maps")
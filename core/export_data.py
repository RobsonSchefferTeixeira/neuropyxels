"""
export_data.py

Export a filtered + downsampled subset of channels to disk. Independent
of the trace view's own filter pipeline: this module computes its own
filter/downsample so the export is safe and correct even if the trace
view is showing something different.

Design notes
------------
- Filtering is Butterworth only (order 2-8, default 4), zero-phase via
  sosfiltfilt. Butterworth is the de facto standard for LFP/EEG
  preprocessing: maximally flat passband, no ripple, monotonic roll-off.
- Downsampling uses scipy.signal.decimate(..., ftype='fir') which
  applies a proper anti-aliasing FIR filter before subsampling. This is
  slower than naive slicing but avoids aliasing artifacts.
- Padding: for a custom [start, end] window, the filter sees
  [start - pad_before, end + pad_after] and the padded region is
  trimmed after filtering. This prevents sosfiltfilt's startup/endup
  transients from contaminating the exported signal. Defaults are
  1 s before and 2 s after (sosfiltfilt transients are asymmetric for
  IIR filters: the tail is longer than the head).
- For whole-recording exports, there is nothing to pad against, so
  filtering runs on the raw window and the transients live in the
  first/last few ms of the file. The metadata records this via
  'padded': False so downstream users can trim if they care.

No GUI dependency. Safe to call from a script.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, decimate

try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False


SUPPORTED_FORMATS = ("npz", "mat", "h5")
SUPPORTED_FILTER_ORDERS = (2, 3, 4, 5, 6, 7, 8)


@dataclass
class ExportParams:
    """All parameters for one export run."""
    channels: list[int]                              # dat-file channel indices
    start_sample: int                                # absolute, inclusive
    end_sample: int                                  # absolute, exclusive
    output_sample_rate: float                        # Hz
    filter_low: float                                # Hz, 0 = no low cutoff
    filter_high: float                               # Hz, 0 = no high cutoff
    filter_order: int = 4
    notch_freq: float | None = 50.0                  # None disables
    notch_q: float = 30.0
    detrend: bool = False
    pad_before_seconds: float = 1.0
    pad_after_seconds: float = 2.0
    target_format: str = "npz"                       # one of SUPPORTED_FORMATS
    output_path: Path = field(default_factory=lambda: Path("export.npz"))
    per_channel: bool = False                        # True = one file per channel
    dtype: str = "float32"
    # Probe geometry for the mapping (channel -> shank, x, y). Required
    # for the channel_map metadata. Can be None when exporting from a
    # script that doesn't care about it.
    channel_geometry: dict[int, dict] | None = None


@dataclass
class ExportResult:
    """What was written."""
    written_paths: list[Path]
    n_channels: int
    n_samples: int
    output_sample_rate: float
    channels: list[int]


# ----------------------------------------------------------------------
# Filter helpers
# ----------------------------------------------------------------------

def _build_butterworth_sos(sr: float, low: float, high: float, order: int):
    """Butterworth bandpass/lowpass/highpass SOS. Same 0-sentinel
    convention as the trace view: low=0 -> lowpass, high=0 -> highpass,
    both 0 -> no filtering (returns None)."""
    nyq = 0.5 * sr
    safe_high = 0.99 * nyq  # leave a hair for numerical stability

    low = max(0.0, min(float(low), safe_high))
    high = max(0.0, min(float(high), safe_high))

    if low == 0.0 and high == 0.0:
        return None
    if low == 0.0:
        return butter(order, high / nyq, btype="low", output="sos")
    if high == 0.0:
        return butter(order, low / nyq, btype="high", output="sos")
    if low >= high:
        raise ValueError(
            f"Invalid band: low ({low}) must be below high ({high})"
        )
    return butter(order, [low / nyq, high / nyq], btype="band", output="sos")


def _apply_notch(signal: np.ndarray, sr: float, notch_freq: float, q: float) -> np.ndarray:
    nyq = 0.5 * sr
    if notch_freq is None or notch_freq <= 0 or notch_freq >= nyq:
        return signal
    b, a = iirnotch(notch_freq, q, fs=sr)
    from scipy.signal import filtfilt
    return filtfilt(b, a, signal)


def _filter_one_channel(
    raw: np.ndarray,
    sr: float,
    params: ExportParams,
) -> np.ndarray:
    """Filter one channel's raw samples. Returns float64, same length
    as input."""
    data = np.asarray(raw, dtype=np.float64)

    if params.detrend:
        from scipy.signal import detrend as scipy_detrend
        data = scipy_detrend(data, type="linear")

    sos = _build_butterworth_sos(
        sr, params.filter_low, params.filter_high, params.filter_order,
    )
    if sos is not None:
        data = sosfiltfilt(sos, data, axis=-1)

    if params.notch_freq is not None and params.notch_freq > 0:
        data = _apply_notch(data, sr, params.notch_freq, params.notch_q)

    return data


def _downsample(raw: np.ndarray, sr: float, target_sr: float) -> tuple[np.ndarray, int]:
    """Integer-factor decimation with proper FIR anti-aliasing. Returns
    (downsampled, decimation_factor). If target_sr >= sr, factor is 1
    and no decimation happens."""
    if target_sr >= sr:
        return raw, 1
    factor = int(round(sr / target_sr))
    if factor < 2:
        return raw, 1
    return decimate(raw, factor, ftype="fir", zero_phase=True), factor


# ----------------------------------------------------------------------
# Metadata builder
# ----------------------------------------------------------------------

def _build_metadata(
    params: ExportParams,
    engine,
    output_sr: float,
    n_samples_out: int,
    padded: bool,
) -> dict:
    """Everything the downstream user needs to interpret the file."""
    # Channel ordering: depth-sorted within shank, shanks left-to-right.
    # Same order the probe map displays.
    geometry = params.channel_geometry or {}
    ordered = []
    for ch in params.channels:
        info = geometry.get(int(ch), {})
        ordered.append({
            "channel": int(ch),
            "shank": int(info.get("shank", 0)),
            "x": float(info.get("x", 0.0)),
            "y": float(info.get("y", 0.0)),
        })
    ordered.sort(key=lambda d: (d["shank"], -d["y"]))

    channels_sorted = [d["channel"] for d in ordered]
    shanks_sorted = [d["shank"] for d in ordered]
    xs_sorted = [d["x"] for d in ordered]
    ys_sorted = [d["y"] for d in ordered]

    indices_0based = list(range(len(channels_sorted)))
    indices_1based = [i + 1 for i in indices_0based]

    # Time base
    start_time = float(engine.sample_to_time(params.start_sample))
    end_time = float(engine.sample_to_time(params.end_sample))

    return {
        "channels": channels_sorted,
        "channel_indices": indices_0based,
        "channel_indices_matlab": indices_1based,
        "shanks": shanks_sorted,
        "xcoords": xs_sorted,
        "ycoords": ys_sorted,
        "sample_rate": float(engine.sr),
        "output_sample_rate": float(output_sr),
        "start_sample": int(params.start_sample),
        "end_sample": int(params.end_sample),
        "start_time": start_time,
        "end_time": end_time,
        "n_samples": int(n_samples_out),
        "n_channels": int(len(channels_sorted)),
        "filter_low": float(params.filter_low),
        "filter_high": float(params.filter_high),
        "filter_order": int(params.filter_order),
        "filter_type": "butterworth",
        "filter_zero_phase": True,
        "notch_freq": float(params.notch_freq) if params.notch_freq else None,
        "notch_q": float(params.notch_q) if params.notch_freq else None,
        "detrend": bool(params.detrend),
        "padded": bool(padded),
        "pad_before_seconds": float(params.pad_before_seconds) if padded else 0.0,
        "pad_after_seconds": float(params.pad_after_seconds) if padded else 0.0,
        "dtype": str(params.dtype),
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
    }


# ----------------------------------------------------------------------
# Format writers
# ----------------------------------------------------------------------

def _write_npz(path: Path, data: np.ndarray, metadata: dict) -> None:
    """np.savez_compressed with data as 'data' and everything else as
    parallel arrays/scalars under matching keys."""
    payload = {"data": data}
    payload.update(metadata)
    np.savez_compressed(path, **payload)


def _write_h5(path: Path, data: np.ndarray, metadata: dict) -> None:
    """HDF5 with 'data' as a dataset and metadata as root-group attributes.
    Attributes must be scalar/array/string; nested dicts are flattened
    to JSON strings."""
    if not _HAS_H5PY:
        raise RuntimeError("h5py is required for .h5 and .mat (v7.3) export")
    import json
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=data, compression="gzip", compression_opts=4)
        for key, value in metadata.items():
            if value is None:
                # HDF5 has no native null; store the string "null".
                f.attrs[key] = "null"
            elif isinstance(value, (list, tuple)):
                # Try to store as an array; fall back to JSON for
                # heterogeneous lists (unlikely here).
                try:
                    f.attrs[key] = np.asarray(value)
                except (TypeError, ValueError):
                    f.attrs[key] = json.dumps(value)
            elif isinstance(value, dict):
                f.attrs[key] = json.dumps(value)
            elif isinstance(value, bool):
                f.attrs[key] = int(value)  # HDF5 has no native bool
            else:
                f.attrs[key] = value


def _write_mat_v73(path: Path, data: np.ndarray, metadata: dict) -> None:
    """MATLAB v7.3 is HDF5 under the hood. To make the file load cleanly
    in MATLAB, we write the data and metadata under a top-level group
    named after the file stem. MATLAB's `load` then returns a struct
    with one field."""
    if not _HAS_H5PY:
        raise RuntimeError("h5py is required for .mat v7.3 export")
    import json
    stem = path.stem
    with h5py.File(path, "w") as f:
        group = f.create_group(stem)
        group.create_dataset("data", data=data, compression="gzip", compression_opts=4)
        # MATLAB-friendly channel_indices (1-based). 'channel_indices'
        # (0-based) is kept for cross-language consistency, but the
        # *_matlab field is the one MATLAB users should use.
        for key, value in metadata.items():
            if value is None:
                group.attrs[key] = "null"
            elif isinstance(value, (list, tuple)):
                try:
                    group.attrs[key] = np.asarray(value)
                except (TypeError, ValueError):
                    group.attrs[key] = json.dumps(value)
            elif isinstance(value, dict):
                group.attrs[key] = json.dumps(value)
            elif isinstance(value, bool):
                group.attrs[key] = int(value)
            else:
                group.attrs[key] = value


_WRITERS = {
    "npz": _write_npz,
    "h5": _write_h5,
    "mat": _write_mat_v73,
}


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------

def export_channels(
    engine,
    params: ExportParams,
    progress_cb=None,
) -> ExportResult:
    """Filter, downsample, and write the requested channels.

    Parameters
    ----------
    engine : TraceEngine
        The loaded recording. Only engine.data, engine.sr, engine.sr,
        and engine.sample_to_time are used.
    params : ExportParams
    progress_cb : callable | None
        Called as progress_cb(done, total) after each channel finishes.

    Returns
    -------
    ExportResult
    """
    if not engine.data_loaded:
        raise ValueError("No data loaded.")
    if params.target_format not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format {params.target_format!r}; "
            f"expected one of {SUPPORTED_FORMATS}"
        )
    if params.filter_order not in SUPPORTED_FILTER_ORDERS:
        raise ValueError(
            f"Filter order must be one of {SUPPORTED_FILTER_ORDERS}"
        )
    if not params.channels:
        raise ValueError("No channels selected.")
    if params.start_sample >= params.end_sample:
        raise ValueError(
            f"start_sample ({params.start_sample}) must be < "
            f"end_sample ({params.end_sample})"
        )

    total_samples = int(engine.data.shape[0])
    start = max(0, int(params.start_sample))
    end = min(total_samples, int(params.end_sample))
    if start >= end:
        raise ValueError(
            f"Requested window [{start}, {end}) is empty after clipping "
            f"to recording length {total_samples}"
        )

    padded = (start > 0 and end < total_samples)
    pad_before_smp = int(round(params.pad_before_seconds * engine.sr)) if padded else 0
    pad_after_smp = int(round(params.pad_after_seconds * engine.sr)) if padded else 0
    read_start = max(0, start - pad_before_smp)
    read_end = min(total_samples, end + pad_after_smp)
    trim_lo = start - read_start
    trim_hi = trim_lo + (end - start)

    # Pre-compute the output length so we can allocate the final array.
    # Decimation drops floor(n / factor) samples.
    if params.output_sample_rate < engine.sr:
        factor = int(round(engine.sr / params.output_sample_rate))
    else:
        factor = 1
    in_len = trim_hi - trim_lo
    out_len = len(range(0, in_len, factor))

    n_ch = len(params.channels)
    dtype = np.dtype(params.dtype)
    out = np.empty((n_ch, out_len), dtype=dtype)

    for i, ch in enumerate(params.channels):
        try:
            raw = engine.data[read_start:read_end, int(ch)]
        except Exception as exc:
            raise RuntimeError(f"Failed to read channel {ch}: {exc}") from exc

        filtered = _filter_one_channel(raw, engine.sr, params)

        # Trim padding back off.
        if padded:
            filtered = filtered[trim_lo:trim_hi]

        downsampled, _ = _downsample(filtered, engine.sr, params.output_sample_rate)

        # _downsample uses decimate, which returns floor(n / factor)
        # samples. Match that to out_len.
        if downsampled.size != out_len:
            # Guard against off-by-one from decimate's internal handling.
            if downsampled.size > out_len:
                downsampled = downsampled[:out_len]
            else:
                pad = np.zeros(out_len - downsampled.size, dtype=downsampled.dtype)
                downsampled = np.concatenate([downsampled, pad])

        out[i, :] = downsampled.astype(dtype, copy=False)

        if progress_cb is not None:
            try:
                progress_cb(i + 1, n_ch)
            except Exception:
                pass

    # Sort channels into display order for writing.
    geometry = params.channel_geometry or {}
    order_keys = []
    for ch in params.channels:
        info = geometry.get(int(ch), {})
        order_keys.append((
            int(info.get("shank", 0)),
            -float(info.get("y", 0.0)),
            int(ch),
        ))
    sort_idx = sorted(range(n_ch), key=lambda i: order_keys[i])
    out = out[sort_idx, :]
    sorted_channels = [params.channels[i] for i in sort_idx]
    sorted_geometry = {
        ch: geometry.get(ch, {}) for ch in sorted_channels
    }

    # Build metadata against the sorted channel list.
    sorted_params = ExportParams(
        channels=sorted_channels,
        start_sample=params.start_sample,
        end_sample=params.end_sample,
        output_sample_rate=params.output_sample_rate,
        filter_low=params.filter_low,
        filter_high=params.filter_high,
        filter_order=params.filter_order,
        notch_freq=params.notch_freq,
        notch_q=params.notch_q,
        detrend=params.detrend,
        pad_before_seconds=params.pad_before_seconds,
        pad_after_seconds=params.pad_after_seconds,
        target_format=params.target_format,
        output_path=params.output_path,
        per_channel=params.per_channel,
        dtype=params.dtype,
        channel_geometry=sorted_geometry,
    )
    metadata = _build_metadata(
        sorted_params, engine, params.output_sample_rate,
        out_len, padded=padded,
    )

    writer = _WRITERS[params.target_format]
    written: list[Path] = []

    if params.per_channel:
        out_dir = Path(params.output_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        ext = params.target_format
        for i, ch in enumerate(sorted_channels):
            ch_geometry = {ch: sorted_geometry.get(ch, {})}
            ch_params = ExportParams(
                channels=[ch],
                start_sample=params.start_sample,
                end_sample=params.end_sample,
                output_sample_rate=params.output_sample_rate,
                filter_low=params.filter_low,
                filter_high=params.filter_high,
                filter_order=params.filter_order,
                notch_freq=params.notch_freq,
                notch_q=params.notch_q,
                detrend=params.detrend,
                pad_before_seconds=params.pad_before_seconds,
                pad_after_seconds=params.pad_after_seconds,
                target_format=params.target_format,
                output_path=out_dir / f"filtered_ch{ch}.{ext}",
                per_channel=True,
                dtype=params.dtype,
                channel_geometry=ch_geometry,
            )
            ch_metadata = _build_metadata(
                ch_params, engine, params.output_sample_rate,
                out_len, padded=padded,
            )
            path = out_dir / f"filtered_ch{ch}.{ext}"
            writer(path, out[i:i + 1, :], ch_metadata)
            written.append(path)
    else:
        path = Path(params.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        writer(path, out, metadata)
        written.append(path)

    return ExportResult(
        written_paths=written,
        n_channels=n_ch,
        n_samples=out_len,
        output_sample_rate=params.output_sample_rate,
        channels=sorted_channels,
    )
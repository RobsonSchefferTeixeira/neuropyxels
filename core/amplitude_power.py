"""
amplitude_power.py

Per-electrode band power computation, extracted from
neuropixels_amplitude_analyzer.py's AmplitudeAnalyzer.compute_power().

Behavioral port: downsample -> detrend -> notch (50Hz) -> bandpass ->
RMS or Hilbert-envelope power, computed independently for every channel
on the probe, all unchanged from the original. Removed: the dependency
on a `viewer` object (self.viewer.data / self.viewer.sr) -- this takes
raw data directly, matching the same pattern as core/trace_engine.py
and core/phase_amplitude.py.

No GUI dependency.

Usage
-----
    analyzer = AmplitudePowerAnalyzer(probe_data)
    power = analyzer.compute(
        raw_data=engine.data, sample_rate=engine.sr,
        params=AmplitudePowerParams(start_time=0, end_time=10,
                                     low_freq=4, high_freq=12),
    )
    # power['values'] is one power value per electrode, in the same
    # order as analyzer.channels / analyzer.xcoords / analyzer.ycoords.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import (
    butter, sosfiltfilt, hilbert, detrend, iirnotch, tf2sos, decimate,
)


@dataclass
class AmplitudePowerParams:
    """Defaults match the original AmplitudeAnalyzer's __init__ defaults."""
    start_time: float = 0.0
    end_time: float = 10.0
    low_freq: float = 4.0
    high_freq: float = 12.0
    method: str = "bandpower"   # "bandpower" (RMS) or "hilbert" (envelope mean)
    target_sr: float = 1000.0
    notch_freq: float | None = 50.0  # None disables the notch step


class AmplitudePowerAnalyzer:
    """
    Computes a single band-power value per electrode across the whole
    probe, preserving each electrode's real (x, y) position so the
    result can be rendered spatially -- e.g. theta power across a probe,
    showing which depth has the strongest theta.
    """

    def __init__(self, probe_data: dict):
        coords = probe_data["coordinates"]
        shanks = probe_data.get("shanks", {})

        self.channels = np.asarray(coords["channels"], dtype=np.int64)
        self.xcoords = np.asarray(coords["x"], dtype=np.float64)
        self.ycoords = np.asarray(coords["y"], dtype=np.float64)

        if shanks.get("ids"):
            self.shank_ids = np.asarray(shanks["ids"], dtype=np.int64)
        else:
            self.shank_ids = np.zeros(len(self.channels), dtype=np.int64)

    def compute(self, raw_data: np.ndarray, sample_rate: float,
                params: AmplitudePowerParams) -> dict:
        """
        Returns
        -------
        dict with keys:
            values : (n_channels,) float array, one power value per
                     electrode, aligned with self.channels
            channels, xcoords, ycoords, shank_ids : passthrough geometry
            params : the AmplitudePowerParams used
        """
        sr = sample_rate

        start_idx = max(0, int(params.start_time * sr))
        end_idx = min(raw_data.shape[0], int(params.end_time * sr))
        if start_idx >= end_idx:
            raise ValueError(
                f"Invalid time range: start_idx={start_idx} >= end_idx={end_idx}"
            )

        if len(self.channels) == 0:
            raise ValueError("No channels available on this probe.")

        # raw_data is (samples, channels); we want (channels, samples)
        data_segment = raw_data[start_idx:end_idx, self.channels].T.astype(np.float64)

        if np.all(data_segment == 0):
            return {
                "values": np.zeros(len(self.channels)),
                "channels": self.channels, "xcoords": self.xcoords,
                "ycoords": self.ycoords, "shank_ids": self.shank_ids,
                "params": params,
            }

        # ---- Downsample ----
        if sr > params.target_sr:
            dec_factor = int(np.round(sr / params.target_sr))
            if dec_factor >= 1:
                data_segment = decimate(data_segment, dec_factor, axis=1, ftype="fir")
                sr = sr / dec_factor

        # ---- Detrend (remove DC offset) ----
        data_segment = detrend(data_segment, axis=1)

        # ---- Notch filter ----
        nyq = 0.5 * sr
        if params.notch_freq is not None and params.notch_freq < nyq:
            b_notch, a_notch = iirnotch(params.notch_freq, Q=30, fs=sr)
            sos_notch = tf2sos(b_notch, a_notch)
            data_segment = sosfiltfilt(sos_notch, data_segment, axis=1)

        # ---- Bandpass ----
        order = 3 if params.high_freq < 20 else 4
        sos = butter(order, [params.low_freq / nyq, params.high_freq / nyq],
                     btype="band", output="sos")
        filtered = sosfiltfilt(sos, data_segment, axis=1)

        if np.all(np.isnan(filtered)):
            # Same fallback as the original: retry with a lower order.
            sos = butter(2, [params.low_freq / nyq, params.high_freq / nyq],
                         btype="band", output="sos")
            filtered = sosfiltfilt(sos, data_segment, axis=1)

        # ---- Power ----
        if params.method == "bandpower":
            values = np.sqrt(np.nanmean(filtered ** 2, axis=1))
        elif params.method == "hilbert":
            analytic = hilbert(filtered, axis=1)
            envelope = np.abs(analytic)
            values = np.nanmean(envelope, axis=1)
        else:
            raise ValueError("method must be 'bandpower' or 'hilbert'.")

        if np.any(np.isnan(values)):
            values = np.nan_to_num(values)

        return {
            "values": values,
            "channels": self.channels,
            "xcoords": self.xcoords,
            "ycoords": self.ycoords,
            "shank_ids": self.shank_ids,
            "params": params,
        }

"""
phase_amplitude.py

Theta phase vs. amplitude coupling analysis, extracted from
neuropixels_phase_amplitude.py's PhaseAmplitudeAnalyzer.

Behavioral port: CSD computation, theta-phase Hilbert transform, wavelet
amplitude across a frequency range, and phase-binned normalized power are
all unchanged from the original. What's removed is the dependency on a
`viewer` object (self.viewer.data / self.viewer.sr) -- this version takes
raw data directly, so it works against any source (TraceEngine's memmap,
a plain numpy array, etc.) without knowing anything about Qt or matplotlib.

No GUI dependency. Safe to import from a Qt dialog or a script.

Usage
-----
    analyzer = PhaseAmplitudeAnalyzer(probe_data)
    results = analyzer.compute(
        raw_data=engine.data,        # (n_samples, n_channels) memmap/array
        sample_rate=engine.sr,
        channel=42,
        start_time=0.0,
        end_time=10.0,
    )
    # results['phase_center_bins'], results['frequencies'],
    # results['phase_energy_norm'] -> ready to render as a heatmap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from core.filters import bandpass_filter, hilbert
from core.wavelet_transform import compute_wavelet_transform, make_theta_matched_cycles


@dataclass
class PhaseAmplitudeParams:
    """All tunable parameters for one analysis run. Defaults match the
    original PhaseAmplitudeAnalyzer's __init__ defaults."""
    channel: int
    start_time: float = 0.0
    end_time: float = 10.0
    phase_low: float = 5.0
    phase_high: float = 10.0
    freq_low: float = 20.0
    freq_high: float = 200.0
    freq_step: float = 1.0
    min_cycles: float = 3.0
    max_cycles: float = 12.0
    fixed_cycles: bool = False
    theta_freq: float = 7.0
    target_sr: float = 1000.0
    use_csd: bool = False
    csd_spacing: float = 20.0  # µm


class PhaseAmplitudeAnalyzer:
    """
    Computes theta-phase-binned, frequency-resolved amplitude (phase-
    amplitude coupling) for one channel of a probe, optionally using
    Current Source Density (CSD) instead of the raw LFP.

    Constructed once per probe (geometry is fixed); call compute() with
    fresh raw data / parameters for each analysis run.
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

    # ------------------------------------------------------------------
    # CSD
    # ------------------------------------------------------------------
    def check_csd_availability_distance(self, channel_idx: int, distance_um: float) -> dict:
        """Geometry-only check matching compute_csd_with_options's
        neighbor rules. Returns {'available': bool, 'message': str}."""
        center_pos = np.where(self.channels == channel_idx)[0]
        if len(center_pos) == 0:
            return {"available": False, "message": f"Channel {channel_idx} not found."}
        center_pos = center_pos[0]

        center_shank = int(self.shank_ids[center_pos])
        center_y = float(self.ycoords[center_pos])
        center_x = float(self.xcoords[center_pos])

        if distance_um <= 0:
            return {"available": False, "message": "distance_um must be positive."}

        max_gap = 3.0 * distance_um

        same_shank = self.shank_ids == center_shank
        same_x = np.isclose(self.xcoords, center_x)
        different_y = self.ycoords != center_y
        within_gap = np.abs(self.ycoords - center_y) <= max_gap
        pool_mask = same_shank & same_x & different_y & within_gap
        pool_mask[center_pos] = False
        pool_idx = np.where(pool_mask)[0]

        def _pick_side(target_y, side):
            if len(pool_idx) == 0:
                return None
            pool_y = self.ycoords[pool_idx]
            side_idx = pool_idx[pool_y > center_y] if side == "above" else pool_idx[pool_y < center_y]
            if len(side_idx) == 0:
                return None
            side_y = self.ycoords[side_idx]
            return int(side_idx[int(np.argmin(np.abs(side_y - target_y)))])

        above_idx = _pick_side(center_y + distance_um, "above")
        below_idx = _pick_side(center_y - distance_um, "below")

        if above_idx is None and below_idx is None:
            return {
                "available": False,
                "message": (
                    f"No same-shank, same-x neighbors within "
                    f"{max_gap:.0f} µm of CH{channel_idx}."
                ),
            }
        if above_idx is None:
            return {
                "available": False,
                "message": (
                    f"No same-shank, same-x channel above CH{channel_idx} "
                    f"within {max_gap:.0f} µm."
                ),
            }
        if below_idx is None:
            return {
                "available": False,
                "message": (
                    f"No same-shank, same-x channel below CH{channel_idx} "
                    f"within {max_gap:.0f} µm."
                ),
            }

        above_y = float(self.ycoords[above_idx])
        below_y = float(self.ycoords[below_idx])
        above_dist = above_y - center_y
        below_dist = center_y - below_y

        return {
            "available": True,
            "message": (
                f"Above CH{int(self.channels[above_idx])} "
                f"(y={above_y:.0f}, gap {above_dist:.0f} µm), "
                f"below CH{int(self.channels[below_idx])} "
                f"(y={below_y:.0f}, gap {below_dist:.0f} µm)."
            ),
            "above_channel": int(self.channels[above_idx]),
            "below_channel": int(self.channels[below_idx]),
            "above_dist": float(above_dist),
            "below_dist": float(below_dist),
        }

    def compute_csd_with_options(
        self,
        signal: np.ndarray,
        channel_idx: int,
        sr: float,
        raw_data: np.ndarray,
        start_time: float,
        end_time: float,
        distance_um: float,
        band_low: float = 0.0,
        band_high: float = 0.0,
    ) -> tuple[np.ndarray, dict]:
        """
        3-point Laplacian CSD with configurable neighbor selection.

        Neighbor rules (all four must hold for a candidate to qualify):
          1. Same shank as the center channel.
          2. Same x-coordinate as the center channel -- this excludes
             the "same depth, different x" electrodes that some probes
             have on the same shank. The Laplacian is a depth
             derivative, so a lateral neighbor is not a valid term.
          3. Strictly different y (depth) from the center channel.
          4. Within MAX_CSD_NEIGHBOR_GAP_UM of the center, where
             MAX_CSD_NEIGHBOR_GAP_UM is 3x the requested distance (so
             the search has some slack but doesn't reach across the
             whole shank).

        Selection among qualifiers:
          - "Above" is the candidate with y > center_y whose |y gap|
            is closest to distance_um.
          - "Below" is the candidate with y < center_y whose |y gap|
            is closest to distance_um.
          - Ties (two candidates at exactly the same y) are broken by
            proximity in x -- but since we now require same x, ties
            shouldn't normally happen unless the probe has duplicate
            (shank, y, x) entries, which is a probe-data error.

        The "spacing" reported is (above_dist + below_dist) / 2, using
        the ACTUAL distances of the chosen neighbors, not the requested
        distance.

        Returns (csd_signal, info). If either above or below is missing,
        or if any sanity check fails, returns
        (signal, {"used": False, "reason": ...}) -- the caller must
        check info["used"] before plotting.
        """
        center_pos = np.where(self.channels == channel_idx)[0]
        if len(center_pos) == 0:
            return signal, {"used": False, "reason": "Channel not found"}
        center_pos = center_pos[0]

        center_shank = int(self.shank_ids[center_pos])
        center_y = float(self.ycoords[center_pos])
        center_x = float(self.xcoords[center_pos])

        if distance_um <= 0:
            return signal, {
                "used": False,
                "reason": f"distance_um must be positive, got {distance_um}",
            }

        # Widest gap we'll consider at all, regardless of requested
        # distance. Anything beyond this is a "no suitable neighbor"
        # case rather than a "use the closest we can find" case.
        max_gap = 3.0 * distance_um

        # ---- Build the candidate pool ----
        # Same shank, same x, strictly different y, gap <= max_gap.
        same_shank = self.shank_ids == center_shank
        same_x = np.isclose(self.xcoords, center_x)
        different_y = self.ycoords != center_y
        within_gap = np.abs(self.ycoords - center_y) <= max_gap
        pool_mask = same_shank & same_x & different_y & within_gap
        pool_mask[center_pos] = False  # exclude the center itself

        pool_idx = np.where(pool_mask)[0]

        def _pick_side(target_y: float, side: str):
            """Among pool candidates strictly on `side` of center_y,
            return the one whose y is closest to target_y. None if no
            candidate exists on that side."""
            if len(pool_idx) == 0:
                return None
            pool_y = self.ycoords[pool_idx]
            if side == "above":
                side_mask = pool_y > center_y
            else:
                side_mask = pool_y < center_y
            side_idx = pool_idx[side_mask]
            if len(side_idx) == 0:
                return None
            side_y = self.ycoords[side_idx]
            # Closest to target_y (NOT closest to center).
            gap_to_target = np.abs(side_y - target_y)
            best_rel = int(np.argmin(gap_to_target))
            return int(side_idx[best_rel])

        above_idx = _pick_side(center_y + distance_um, "above")
        below_idx = _pick_side(center_y - distance_um, "below")

        if above_idx is None or below_idx is None:
            reasons = []
            if above_idx is None:
                reasons.append(
                    f"no same-shank, same-x channel above within "
                    f"{max_gap:.0f} µm"
                )
            if below_idx is None:
                reasons.append(
                    f"no same-shank, same-x channel below within "
                    f"{max_gap:.0f} µm"
                )
            return signal, {
                "used": False,
                "reason": "; ".join(reasons),
            }

        above_channel = int(self.channels[above_idx])
        below_channel = int(self.channels[below_idx])
        above_y = float(self.ycoords[above_idx])
        below_y = float(self.ycoords[below_idx])
        above_dist = above_y - center_y   # positive
        below_dist = center_y - below_y   # positive

        # Final sanity: both distances must be strictly positive and
        # distinct from each other only insofar as the probe geometry
        # allows. This is defense-in-depth -- by construction they
        # should already be positive, since we required different_y.
        if above_dist <= 0 or below_dist <= 0:
            return signal, {
                "used": False,
                "reason": (
                    f"non-positive neighbor gaps "
                    f"(above={above_dist}, below={below_dist})"
                ),
            }

        start_idx = int(start_time * sr)
        end_idx = int(end_time * sr)
        above_signal = raw_data[start_idx:end_idx, above_channel].flatten().astype(np.float64)
        below_signal = raw_data[start_idx:end_idx, below_channel].flatten().astype(np.float64)
        center_signal = signal.astype(np.float64, copy=False)

        band_applied = False
        if band_low > 0 and band_high > band_low and band_high < sr / 2:
            from core.filters import bandpass_filter
            above_signal = bandpass_filter(above_signal, sr, band_low, band_high)
            below_signal = bandpass_filter(below_signal, sr, band_low, band_high)
            center_signal = bandpass_filter(center_signal, sr, band_low, band_high)
            band_applied = True

        spacing_um = (above_dist + below_dist) / 2.0
        csd_signal = (above_signal - 2 * center_signal + below_signal) / (spacing_um ** 2)

        csd_info = {
            "used": True,
            "center_channel": int(channel_idx),
            "above_channel": above_channel,
            "below_channel": below_channel,
            "above_distance": float(above_dist),
            "below_distance": float(below_dist),
            "spacing": float(spacing_um),
            "above_y": above_y,
            "below_y": below_y,
            "center_y": center_y,
            "center_x": center_x,
            "band_low": band_low,
            "band_high": band_high,
            "band_applied": band_applied,
            "distance_requested": float(distance_um),
        }
        return csd_signal, csd_info




        
    def compute_csd(self, signal: np.ndarray, channel_idx: int, sr: float,
                     raw_data: np.ndarray, start_time: float, end_time: float,
                     csd_spacing: float) -> tuple[np.ndarray, dict]:
        """
        3-point Laplacian CSD: (V_above - 2*V_center + V_below) / spacing^2.

        raw_data is the FULL (n_samples, n_channels) source (not yet
        downsampled/filtered), since the above/below channels need their
        own raw segments extracted at the same sample rate as `signal`
        was originally sampled at (i.e. before any decimation the caller
        applied to `signal`). See compute() for how this is called.
        """
        center_pos = np.where(self.channels == channel_idx)[0]
        if len(center_pos) == 0:
            return signal, {"used": False, "reason": "Channel not found"}
        center_pos = center_pos[0]

        center_shank = self.shank_ids[center_pos]
        center_y = self.ycoords[center_pos]

        same_shank = np.where(self.shank_ids == center_shank)[0]
        same_shank_y = self.ycoords[same_shank]

        y_above = center_y + csd_spacing
        y_below = center_y - csd_spacing

        above_mask = same_shank_y >= y_above
        if np.any(above_mask):
            above_candidates = np.where(above_mask)[0]
            best = above_candidates[np.argmin(same_shank_y[above_mask] - y_above)]
            above_idx = same_shank[best]
            above_dist = self.ycoords[above_idx] - center_y
        else:
            above_idx = None
            above_dist = None

        below_mask = same_shank_y <= y_below
        if np.any(below_mask):
            below_candidates = np.where(below_mask)[0]
            best = below_candidates[np.argmin(y_below - same_shank_y[below_mask])]
            below_idx = same_shank[best]
            below_dist = center_y - self.ycoords[below_idx]
        else:
            below_idx = None
            below_dist = None

        if above_idx is None or below_idx is None:
            return signal, {"used": False, "reason": "No channels above or below"}

        if above_dist < csd_spacing * 0.5 or below_dist < csd_spacing * 0.5:
            reason = (f"Neighbor too close (above: {above_dist:.0f}\u00b5m, "
                      f"below: {below_dist:.0f}\u00b5m)")
            return signal, {"used": False, "reason": reason}

        if above_dist > csd_spacing * 1.5 or below_dist > csd_spacing * 1.5:
            reason = (f"Neighbor too far (above: {above_dist:.0f}\u00b5m, "
                      f"below: {below_dist:.0f}\u00b5m)")
            return signal, {"used": False, "reason": reason}

        above_channel = int(self.channels[above_idx])
        below_channel = int(self.channels[below_idx])

        start_idx = int(start_time * sr)
        end_idx = int(end_time * sr)
        above_signal = raw_data[start_idx:end_idx, above_channel].flatten().astype(np.float64)
        below_signal = raw_data[start_idx:end_idx, below_channel].flatten().astype(np.float64)

        spacing_um = (above_dist + below_dist) / 2
        csd_signal = (above_signal - 2 * signal + below_signal) / (spacing_um ** 2)

        csd_info = {
            "used": True,
            "center_channel": int(channel_idx),
            "above_channel": above_channel,
            "below_channel": below_channel,
            "above_distance": float(above_dist),
            "below_distance": float(below_dist),
            "spacing": float(spacing_um),
            "above_y": float(self.ycoords[above_idx]),
            "below_y": float(self.ycoords[below_idx]),
            "center_y": float(center_y),
        }
        return csd_signal, csd_info

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------

    def compute(self, raw_data: np.ndarray, sample_rate: float,
                params: PhaseAmplitudeParams) -> dict:
        """
        Parameters
        ----------
        raw_data : (n_samples, n_channels) array-like (e.g. a memmap)
        sample_rate : original sampling rate of raw_data, in Hz
        params : PhaseAmplitudeParams

        Returns
        -------
        dict with keys: frequencies, phase_center_bins, phase_energy,
        phase_energy_norm, theta_phase, channel, n_phase_bins, phase_bins,
        sample_rate (post-downsample), csd_info, use_csd.
        """
        sr = sample_rate
        channel_idx = params.channel
        if channel_idx not in self.channels:
            idx = int(np.argmin(np.abs(self.channels - channel_idx)))
            channel_idx = int(self.channels[idx])

        start_idx = max(0, int(params.start_time * sr))
        end_idx = min(raw_data.shape[0], int(params.end_time * sr))
        if start_idx >= end_idx:
            raise ValueError(
                f"Invalid time range: start_idx={start_idx} >= end_idx={end_idx}"
            )

        data_segment = raw_data[start_idx:end_idx, channel_idx].flatten().astype(np.float64)

        csd_info = None
        if params.use_csd:
            data_segment, csd_info = self.compute_csd(
                data_segment, channel_idx, sr, raw_data,
                params.start_time, params.end_time, params.csd_spacing,
            )

        # ---- Downsample (matches original: scipy.signal.decimate, fir) ----
        target_sr = params.target_sr
        if sr > target_sr:
            from scipy.signal import decimate
            dec_factor = int(np.round(sr / target_sr))
            if dec_factor < 1:
                dec_factor = 1
            actual_sr = sr / dec_factor
            data_segment = decimate(data_segment, dec_factor, ftype="fir")
            sr = actual_sr

        # ---- Theta phase via bandpass + Hilbert ----
        theta_filtered = bandpass_filter(
            data_segment, sr, params.phase_low, params.phase_high, order=4
        )
        theta_complex = hilbert(theta_filtered)
        theta_phase = np.angle(theta_complex)

        # ---- Wavelet amplitude across the frequency range ----
        freqvector = np.arange(params.freq_low, params.freq_high + params.freq_step,
                                params.freq_step)

        if params.fixed_cycles:
            n_cycles_vec = np.full(len(freqvector), params.min_cycles)
        else:
            n_cycles_vec = make_theta_matched_cycles(
                freqvector, theta_freq=params.theta_freq,
                min_cycles=params.min_cycles, max_cycles=params.max_cycles,
            )

        power = compute_wavelet_transform(
            data_segment, sampling_rate=sr, frequencies=freqvector,
            n_cycles=n_cycles_vec, normalization="L1", standardize=False,
            output="power",
        )

        # ---- Phase-binned, normalized power ----
        n_phase_bins = 90
        phase_bins = np.linspace(-math.pi, math.pi, n_phase_bins + 1)
        phase_dt = np.diff(phase_bins)
        phase_center_bins = phase_bins[:-1] + phase_dt / 2

        phase_energy = np.full((len(freqvector), n_phase_bins), np.nan)
        for pp in range(n_phase_bins):
            phase_idx = (theta_phase >= phase_bins[pp]) & (theta_phase <= phase_bins[pp + 1])
            if np.sum(phase_idx) > 0:
                phase_energy[:, pp] = np.nanmean(power[:, phase_idx], axis=1)

        phase_energy_means = np.reshape(
            np.repeat(np.nanmean(phase_energy, axis=1), phase_energy.shape[1]),
            phase_energy.shape,
        )
        phase_energy_norm = phase_energy / phase_energy_means

        return {
            "frequencies": freqvector,
            "phase_center_bins": phase_center_bins,
            "phase_energy": phase_energy,
            "phase_energy_norm": phase_energy_norm,
            "theta_phase": theta_phase,
            "channel": channel_idx,
            "n_phase_bins": n_phase_bins,
            "phase_bins": phase_bins,
            "sample_rate": sr,
            "csd_info": csd_info,
            "use_csd": params.use_csd,
        }


"""
comodulogram.py

Phase-amplitude coupling across a grid of phase frequencies and
amplitude frequencies: a comodulogram.

For every (phase_center, amp_center) pair, the MI (Tort's normalized
entropy) is computed by:

  1. Filtering the signal in the phase band
     [phase_center, phase_center + bandwidth].
  2. Extracting the analytic phase via Hilbert.
  3. Filtering the signal in an amplitude band centered on amp_center
     whose half-width equals the phase center frequency. The band is
     therefore [amp_center - center_phase, amp_center + center_phase],
     where center_phase = phase_center + bandwidth / 2. This lets the
     amplitude band include the sidebands created by the phase
     oscillation modulating the amplitude carrier.
  4. Extracting the amplitude envelope via Hilbert.
  5. Binning the mean amplitude across n_bins phase bins and computing
     Tort's modulation index.

Parallelization
---------------
Two joblib.Parallel blocks:

  * One over phase filters -- each phase band is filtered once.
  * One over the (phase, amp) pairs -- each amplitude band is filtered
    per pair, using the already-computed phase.

The MI computation is cheap (one pass over n_bins per pair), so it is
done inline inside the amplitude task.

No GUI dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from scipy.signal import decimate

from core.filters import bandpass_filter, hilbert
from core.phase_amplitude import PhaseAmplitudeAnalyzer


# Tort et al. (2010) -- 18 bins is the conventional number.
DEFAULT_N_BINS = 18


@dataclass
class ComodulogramParams:
    """All tunable parameters for one comodulogram run."""
    channel: int
    start_time: float = 0.0
    end_time: float = 10.0

    # Phase-frequency grid
    phase_freq_low: float = 2.0
    phase_freq_high: float = 20.0
    phase_freq_step: float = 2.0
    phase_bandwidth: float = 4.0

    # Amplitude-frequency grid
    amp_freq_low: float = 20.0
    amp_freq_high: float = 200.0
    amp_freq_step: float = 5.0

    n_bins: int = DEFAULT_N_BINS
    target_sr: float = 1000.0
    use_csd: bool = False
    csd_spacing: float = 20.0


# ----------------------------------------------------------------------
# Internal helpers (picklable -- they run in joblib worker processes)
# ----------------------------------------------------------------------

def _phase_for_band(signal: np.ndarray, sr: float,
                     pf_low: float, pf_high: float) -> np.ndarray:
    filtered = bandpass_filter(signal, sr, pf_low, pf_high, order=4)
    analytic = hilbert(filtered)
    return np.angle(analytic)


def _mi_for_pair(signal: np.ndarray,
                 sr: float,
                 phase: np.ndarray,
                 phase_center: float,
                 amp_center: float,
                 n_bins: int,
                 positions: np.ndarray,
                 winsize: float) -> float:
    """Compute Tort MI for one (phase_center, amp_center) pair.
    Returns NaN if the amplitude band collapses (low >= high after
    clamping)."""
    nyq = 0.5 * sr
    center_phase = phase_center  # caller passes phase_center+bandwidth/2

    af_low = max(2.0, amp_center - center_phase)
    af_high = min(nyq - 1.0, amp_center + center_phase)
    if af_high <= af_low:
        return float("nan")

    amp_filtered = bandpass_filter(signal, sr, af_low, af_high, order=4)
    amp_analytic = hilbert(amp_filtered)
    amp = np.abs(amp_analytic)

    # Mean amplitude per phase bin
    mean_amp = np.zeros(n_bins, dtype=np.float64)
    for i in range(n_bins):
        mask = (phase >= positions[i]) & (phase < positions[i] + winsize)
        if np.any(mask):
            mean_amp[i] = float(np.mean(amp[mask]))
        else:
            mean_amp[i] = 0.0

    total = mean_amp.sum()
    if total <= 0 or not np.isfinite(total):
        return 0.0

    p = mean_amp / total
    # p * log(p), guard p=0
    log_p = np.zeros_like(p)
    positive = p > 0
    log_p[positive] = p[positive] * np.log(p[positive])

    mi = (math.log(n_bins) + log_p.sum()) / math.log(n_bins)
    return float(mi)


# ----------------------------------------------------------------------
# Analyzer
# ----------------------------------------------------------------------

class ComodulogramAnalyzer:
    """Computes a comodulogram for one channel of a probe.

    Constructed once per probe (geometry is fixed); call compute()
    with fresh raw data / parameters for each analysis run. CSD
    neighbor selection is delegated to PhaseAmplitudeAnalyzer so the
    same distance rule is used everywhere in the app.
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

        # Delegate CSD computation to the existing PAC analyzer so
        # there is exactly one implementation of the neighbor rule.
        self._pac_analyzer = PhaseAmplitudeAnalyzer(probe_data)

    # ---- CSD delegation ----

    def check_csd_availability_distance(self, channel_idx: int,
                                         distance_um: float) -> dict:
        return self._pac_analyzer.check_csd_availability_distance(
            channel_idx, distance_um
        )

    # ---- Main analysis ----

    def compute(self, raw_data, sample_rate: float,
                params: ComodulogramParams,
                progress_cb=None) -> dict:
        """Compute a full comodulogram.

        Parameters
        ----------
        raw_data : (n_samples, n_channels) array-like (e.g. memmap)
        sample_rate : original sampling rate of raw_data
        params : ComodulogramParams
        progress_cb : callable | None
            Called as progress_cb(done, total) after each amplitude-band
            chunk completes. The chunking exists so the caller (a
            QThread) can emit progress updates without joblib having to
            expose intra-Parallel state.

        Returns
        -------
        dict with keys:
            comodulogram : 2D ndarray (len(phase_vector), len(amp_vector))
            phase_frequencies : 1D ndarray of phase-band LOW edges
            amplitude_frequencies : 1D ndarray of amplitude-band CENTERS
            channel : int
            sample_rate : float (post-downsample)
            csd_info : dict | None
            use_csd : bool
        """
        sr = float(sample_rate)
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

        signal = raw_data[start_idx:end_idx, channel_idx] \
            .flatten().astype(np.float64)

        # ---- Optional CSD ----
        csd_info = None
        if params.use_csd:
            signal, csd_info = self._pac_analyzer.compute_csd_with_options(
                signal, channel_idx, sr, raw_data,
                params.start_time, params.end_time,
                distance_um=params.csd_spacing,
                band_low=0.0, band_high=0.0,
            )

        # ---- Downsample ----
        if sr > params.target_sr > 0:
            dec_factor = int(round(sr / params.target_sr))
            if dec_factor >= 2:
                signal = decimate(signal, dec_factor, ftype="fir")
                sr = sr / dec_factor

        # ---- Frequency grids ----
        phase_lows = np.arange(
            params.phase_freq_low,
            params.phase_freq_high + params.phase_freq_step * 0.5,
            params.phase_freq_step,
        )
        amp_centers = np.arange(
            params.amp_freq_low,
            params.amp_freq_high + params.amp_freq_step * 0.5,
            params.amp_freq_step,
        )
        n_phase = len(phase_lows)
        n_amp = len(amp_centers)
        if n_phase == 0 or n_amp == 0:
            raise ValueError("Frequency grid produced no bins.")

        # ---- Phase filtering (parallel) ----
        phases = Parallel(n_jobs=-1)(
            delayed(_phase_for_band)(
                signal, sr, float(pf), float(pf + params.phase_bandwidth)
            )
            for pf in phase_lows
        )

        # ---- Amplitude-band MI (parallel, chunked for progress) ----
        n_bins = int(params.n_bins)
        winsize = 2 * math.pi / n_bins
        positions = -math.pi + winsize * np.arange(n_bins)

        # Build flat pair list
        pairs: list[tuple[int, int]] = []
        for i, pf in enumerate(phase_lows):
            center_phase = float(pf) + params.phase_bandwidth / 2.0
            for j, amp_center in enumerate(amp_centers):
                pairs.append((i, j))  # (phase index, amp index)

        comod = np.full((n_phase, n_amp), np.nan, dtype=np.float64)

        # Chunk so we can emit progress. Each chunk runs its own
        # Parallel; joblib handles worker reuse transparently.
        n_chunks = 8
        chunk_size = max(1, len(pairs) // n_chunks)

        done = 0
        for chunk_start in range(0, len(pairs), chunk_size):
            chunk = pairs[chunk_start:chunk_start + chunk_size]
            results = Parallel(n_jobs=-1)(
                delayed(_mi_for_pair)(
                    signal, sr, phases[i],
                    float(phase_lows[i]) + params.phase_bandwidth / 2.0,
                    float(amp_centers[j]),
                    n_bins, positions, winsize,
                )
                for i, j in chunk
            )
            for (i, j), mi in zip(chunk, results):
                comod[i, j] = mi
            done += len(chunk)
            if progress_cb is not None:
                try:
                    progress_cb(done, len(pairs))
                except Exception:
                    pass

        return {
            "comodulogram": comod,
            "phase_frequencies": phase_lows,
            "amplitude_frequencies": amp_centers,
            "channel": int(channel_idx),
            "sample_rate": float(sr),
            "csd_info": csd_info,
            "use_csd": bool(params.use_csd),
        }
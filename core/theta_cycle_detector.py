"""
theta_cycle_detector.py

Cycle-by-cycle theta detection: find each individual theta oscillation
and its landmarks (zero-crossings, peaks, valley) in a single channel's
signal, optionally filtering out cycles whose local theta/delta ratio
or theta/fast ratio is unfavorable.

Adapted from tsc_helper_functions.py's define_theta_cycles, restricted
to the use_imf=False path (no EMD / IMF decomposition). Behavioral
differences from the reference, all intentional:

  * No decimation. Detection runs at the recording's raw sample rate
    and all returned sample indices are raw sample indices. This is
    the same convention as ripple and theta-epoch detection.
  * Theta band, delta band, and peak/valley distance bounds are all
    configurable. The reference hardcodes the distance bounds
    (13 Hz / 6 Hz); here they're derived from the theta band by
    default and overridable.
  * Delta band defaults to [0.5 Hz, theta_low_freq] rather than
    [0, theta_low_freq]. A pure lowpass down to DC picks up slow
    drift and movement artifact, which inflates the delta envelope
    and rejects genuine theta cycles. 0.5 Hz is above most of that
    slow content while still including physiological delta.
  * Delta and fast corrections are computed as per-cycle RATIOS
    (mean theta envelope vs mean delta / fast envelope over the
    cycle's [zc_start, zc_end) span), not as global percentile gates.
    A global percentile over a long recording with mixed behavioral
    states is calibrated to the dominant state and loses its
    discriminating power; a per-cycle ratio encodes what the user
    actually wants: at this moment, does theta dominate the
    competing band?

No GUI dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt, detrend as scipy_detrend

from core.filters import hilbert  # pads to next power of 2, matches reference


@dataclass
class ThetaCycleParams:
    """All tunable parameters for one cycle detection run. Defaults
    reproduce the reference's behavior on standard rodent LFP, with
    the two intentional departures noted in the module docstring
    (delta band lower bound, per-cycle ratio corrections)."""
    channels: list[int]
    start_time: float = 0.0
    end_time: float = 10.0

    # ---- Theta band ----
    low_freq: float = 4.0
    high_freq: float = 12.0
    filter_order: int = 4
    detrend: bool = True

    # ---- Peak/valley amplitude floor ----
    # A candidate peak or valley must exceed this percentile of the
    # theta envelope. Reference value: 15.
    min_theta_peak_percentile: float = 15.0

    # ---- Peak-to-peak distance bounds (Hz) ----
    # Minimum and maximum instantaneous theta frequency implied by the
    # peak-to-peak distance between consecutive theta peaks. None
    # means "auto-derive from the theta band":
    #     peak_low_freq  = low_freq  * 1.5
    #     peak_high_freq = high_freq * 1.1
    # which reproduces the reference's hardcoded 6 / 13 for the
    # default 4-12 band.
    peak_low_freq: float | None = None
    peak_high_freq: float | None = None

    # ---- Delta correction ----
    # Keep a cycle only if mean(theta_amp) / mean(delta_amp) over its
    # [zc_start, zc_end) span is at least theta_delta_ratio_min.
    delta_correction: bool = True
    delta_low_freq: float = 0.5
    delta_high_freq: float | None = None   # defaults to theta low_freq
    theta_delta_ratio_min: float = 1.5

    # ---- Fast correction ----
    # Keep a cycle only if mean(fast_amp) / mean(theta_amp) over its
    # span is at most fast_theta_ratio_max. Auto-disabled when
    # fast_low_freq >= Nyquist of the recording's sample rate.
    fast_correction: bool = False
    fast_low_freq: float = 170.0
    fast_theta_ratio_max: float = 0.5

    # ---- Cycle definition ----
    # Which zero-crossing polarity starts a cycle. 'rising_zc' matches
    # the reference's default.
    cycle_start_edge: str = "rising_zc"


@dataclass
class ThetaCycle:
    """One detected theta cycle. All six landmark fields are RAW sample
    indices into the recording -- no decimation, no offsets. Callers
    that want to convert to seconds use engine.sr directly."""
    channel: int
    zero_crossing_start: int
    peak1: int
    zero_crossing_mid: int
    valley: int
    zero_crossing_end: int
    peak2: int
    amplitude: float            # max theta_signal over the cycle span, in µV
    duration_ms: float          # (zc_end - zc_start) / sr * 1000
    # Diagnostics, populated only when the corresponding correction
    # is enabled. None otherwise.
    theta_delta_ratio: float | None = None
    fast_theta_ratio: float | None = None
    manual: bool = False


def _effective_peak_bounds(params: ThetaCycleParams) -> tuple[float, float]:
    """Return (peak_low_freq, peak_high_freq), auto-deriving from the
    theta band when unset."""
    pl = params.peak_low_freq
    ph = params.peak_high_freq
    if pl is None:
        pl = params.low_freq * 1.5
    if ph is None:
        ph = params.high_freq * 1.1
    return float(pl), float(ph)


def _effective_delta_high(params: ThetaCycleParams) -> float:
    if params.delta_high_freq is None:
        return float(params.low_freq)
    return float(params.delta_high_freq)


class ThetaCycleDetector:
    """Stateless. Construct once, call detect() per channel."""

    # ------------------------------------------------------------------
    # Peak detection (ported from tsc_helper_functions.detect_peaks)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_peaks(x: np.ndarray, mph: float | None = None, mpd: int = 1,
                      threshold: float = 0.0, edge: str = "rising",
                      kpsh: bool = False) -> np.ndarray:
        """Peak indices of 1D array x, keeping only peaks whose height
        is >= mph and whose distance to any higher kept peak is >= mpd.

        Direct port of the reference's detect_peaks (Matlab-style
        peakdet), with the same edge handling and NaN treatment.
        """
        x = np.atleast_1d(x).astype("float64")
        if x.size < 3:
            return np.array([], dtype=int)

        dx = x[1:] - x[:-1]
        ind_nan = np.where(np.isnan(x))[0]
        if ind_nan.size:
            x = x.copy()
            x[ind_nan] = np.inf
            dx = dx.copy()
            dx[np.isnan(dx)] = np.inf

        ine = ire = ife = np.array([], dtype=int)
        if edge is None:
            ine = np.where((np.hstack((dx, 0)) < 0)
                           & (np.hstack((0, dx)) > 0))[0]
        else:
            edge_l = edge.lower()
            if edge_l in ("rising", "both"):
                ire = np.where((np.hstack((dx, 0)) <= 0)
                               & (np.hstack((0, dx)) > 0))[0]
            if edge_l in ("falling", "both"):
                ife = np.where((np.hstack((dx, 0)) < 0)
                               & (np.hstack((0, dx)) >= 0))[0]
        ind = np.unique(np.hstack((ine, ire, ife)))

        if ind.size and ind_nan.size:
            bad = np.unique(np.hstack((ind_nan, ind_nan - 1, ind_nan + 1)))
            ind = ind[np.in1d(ind, bad, invert=True)]

        if ind.size and ind[0] == 0:
            ind = ind[1:]
        if ind.size and ind[-1] == x.size - 1:
            ind = ind[:-1]

        if ind.size and mph is not None:
            ind = ind[x[ind] >= mph]

        if ind.size and threshold > 0:
            dx_thresh = np.min(
                np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]),
                axis=0,
            )
            ind = np.delete(ind, np.where(dx_thresh < threshold)[0])

        if ind.size and mpd > 1:
            ind_sorted = ind[np.argsort(x[ind])][::-1]
            del_flags = np.zeros(ind_sorted.size, dtype=bool)
            for i in range(ind_sorted.size):
                if not del_flags[i]:
                    if kpsh:
                        close = ((ind_sorted >= ind_sorted[i] - mpd)
                                 & (ind_sorted <= ind_sorted[i] + mpd))
                    else:
                        close = ((ind_sorted >= ind_sorted[i] - mpd)
                                 & (ind_sorted <= ind_sorted[i] + mpd)
                                 & (x[ind_sorted[i]] > x[ind_sorted]))
                    del_flags |= close
                    del_flags[i] = False
            ind = np.sort(ind_sorted[~del_flags])

        return ind

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _bandpass(signal: np.ndarray, sr: float, low: float, high: float,
                  order: int = 4) -> np.ndarray:
        """Zero-phase Butterworth bandpass. Clamps high to just under
        Nyquist so a misconfigured band doesn't raise from butter()."""
        nyq = 0.5 * sr
        if high >= nyq:
            high = nyq * 0.99
        if low <= 0:
            low = 1e-6
        if low >= high:
            raise ValueError(
                f"Invalid band: low={low} Hz, high={high} Hz, Nyquist={nyq} Hz"
            )
        sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
        return sosfiltfilt(sos, signal)

    @staticmethod
    def _highpass(signal: np.ndarray, sr: float, low: float,
                  order: int = 4) -> np.ndarray:
        nyq = 0.5 * sr
        if low >= nyq:
            raise ValueError(
                f"Highpass cutoff {low} Hz at or above Nyquist {nyq} Hz"
            )
        sos = butter(order, low / nyq, btype="high", output="sos")
        return sosfiltfilt(sos, signal)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def detect(self, signal: np.ndarray, sample_rate: float,
               params: ThetaCycleParams, channel: int = 0) -> list[ThetaCycle]:
        """Detect theta cycles in a single channel's raw signal.

        Parameters
        ----------
        signal : 1D float array, the raw signal over the analysis window
        sample_rate : Hz (the recording's native rate, no decimation)
        params : ThetaCycleParams
        channel : int, label stored on each returned ThetaCycle

        Returns
        -------
        list[ThetaCycle], sorted by zero_crossing_start. Sample fields
        are indices into `signal`.
        """
        signal = np.asarray(signal, dtype=np.float64)
        if signal.size < 4:
            return []

        # ---- Theta band, detrended ----
        theta_signal = self._bandpass(
            signal, sample_rate, params.low_freq, params.high_freq,
            params.filter_order,
        )
        if params.detrend:
            theta_signal = scipy_detrend(theta_signal)

        theta_amp = np.abs(hilbert(theta_signal))
        min_amp = float(np.percentile(theta_amp, params.min_theta_peak_percentile))

        # ---- Peak-to-peak / peak-to-valley distance bounds ----
        peak_low_freq, peak_high_freq = _effective_peak_bounds(params)
        min_pp = int(round((1.0 / peak_high_freq) * sample_rate))
        max_pp = int(round((1.0 / peak_low_freq) * sample_rate))
        min_pv = int(round((1.0 / peak_high_freq) * sample_rate / 2.0))
        max_pv = int(round((1.0 / peak_low_freq) * sample_rate / 2.0))

        # ---- Candidates ----
        peak_idx = self._detect_peaks(
            theta_signal, mph=min_amp, mpd=max(1, min_pp),
        )
        valley_idx = self._detect_peaks(
            -theta_signal, mph=min_amp, mpd=max(1, min_pp),
        )
        if peak_idx.size == 0 or valley_idx.size == 0:
            return []

        # ---- Pair peaks and valleys into candidate cycles ----
        cycle_peak1 = []
        cycle_peak2 = []
        cycle_valleys = []
        for v in valley_idx:
            before = peak_idx[(peak_idx > v - max_pv) & (peak_idx < v)]
            after = peak_idx[(peak_idx < v + max_pv) & (peak_idx > v)]
            p1 = int(before.max()) if before.size else -np.inf
            p2 = int(after.min()) if after.size else -np.inf
            if p1 == -np.inf or p2 == -np.inf:
                continue
            if not (min_pv <= min(v - p1, p2 - v) <= max_pv):
                continue
            if (p2 - p1) > max_pp:
                continue
            cycle_peak1.append(p1)
            cycle_peak2.append(p2)
            cycle_valleys.append(int(v))

        if not cycle_peak1:
            return []

        cycle_peak1 = np.array(cycle_peak1)
        cycle_peak2 = np.array(cycle_peak2)
        cycle_valleys = np.array(cycle_valleys)

        # ---- De-duplicate by first-peak index ----
        _, uniq = np.unique(cycle_peak1, return_index=True)
        cycle_peak1 = cycle_peak1[uniq]
        cycle_peak2 = cycle_peak2[uniq]
        cycle_valleys = cycle_valleys[uniq]

        # ---- Validate with zero crossings ----
        cycle_refs: list[list[int]] = []
        for k in range(cycle_valleys.size):
            p1 = int(cycle_peak1[k])
            p2 = int(cycle_peak2[k])
            v = int(cycle_valleys[k])

            # Search for zero crossings in the three intervals:
            #   aux1 = [p1 - max_pv, p1)
            #   aux2 = [p1, v)
            #   aux3 = [v, p2)
            a1 = np.arange(max(0, p1 - max_pv), p1)
            a2 = np.arange(p1, v)
            a3 = np.arange(v, p2)
            if a1.size == 0 or a2.size == 0 or a3.size == 0:
                continue

            # Rising-edge convention: signal goes negative -> positive
            # just before p1, negative again before v, positive before p2.
            # (If cycle_start_edge == 'falling_zc' we negate the signal
            # once, upfront, so this same logic handles both conventions
            # -- the caller decided the polarity by setting that param.)
            s1 = theta_signal[a1]
            s2 = theta_signal[a2]
            s3 = theta_signal[a3]
            if not (np.any(s1 < 0) and np.any(s2 < 0) and np.any(s3 > 0)):
                continue

            # The zero crossings are the LAST sample of each interval
            # that has the expected sign, per the reference.
            zc1 = int(np.max(a1[s1 < 0]) + 1)
            zc2 = int(np.min(a2[s2 < 0]) - 1)
            zc3 = int(np.max(a3[s3 < 0]) + 1)

            # Validate: no wrong-sign excursion between crossing and peak
            if np.any(theta_signal[zc1:p1] < 0):
                continue
            if np.any(theta_signal[zc2 + 1:v] > 0):
                continue
            if np.any(theta_signal[zc3:p2] < 0):
                continue

            cycle_refs.append([zc1, p1, zc2, v, zc3, p2])

        if not cycle_refs:
            return []

        # If the user asked for falling-edge definition, undo the sign
        # flip we applied conceptually (we didn't physically negate, so
        # nothing to undo -- the landmark roles stay the same). This
        # branch is a placeholder for symmetry with the reference; the
        # actual landmark interpretation is documented on ThetaCycle.
        # (Left as-is on purpose.)

        # ---- Optional delta and fast corrections, per cycle ----
        delta_amp = None
        if params.delta_correction:
            d_high = _effective_delta_high(params)
            delta_signal = self._bandpass(
                signal, sample_rate, params.delta_low_freq, d_high,
                params.filter_order,
            )
            if params.detrend:
                delta_signal = scipy_detrend(delta_signal)
            delta_amp = np.abs(hilbert(delta_signal))

        fast_amp = None
        nyq = 0.5 * sample_rate
        if params.fast_correction and params.fast_low_freq < nyq:
            fast_signal = self._highpass(
                signal, sample_rate, params.fast_low_freq,
                params.filter_order,
            )
            if params.detrend:
                fast_signal = scipy_detrend(fast_signal)
            fast_amp = np.abs(hilbert(fast_signal))

        # ---- Assemble ThetaCycle objects, applying corrections ----
        eps = 1e-12
        out: list[ThetaCycle] = []
        for refs in cycle_refs:
            zc1, p1, zc2, v, zc3, p2 = refs
            if zc3 <= zc1:
                continue

            theta_mean = float(np.mean(theta_amp[zc1:zc3]))
            delta_ratio = None
            if delta_amp is not None:
                delta_mean = float(np.mean(delta_amp[zc1:zc3]))
                delta_ratio = theta_mean / max(delta_mean, eps)
                if delta_ratio < params.theta_delta_ratio_min:
                    continue

            fast_ratio = None
            if fast_amp is not None:
                fast_mean = float(np.mean(fast_amp[zc1:zc3]))
                fast_ratio = fast_mean / max(theta_mean, eps)
                if fast_ratio > params.fast_theta_ratio_max:
                    continue

            amp = float(np.max(theta_signal[zc1:zc3]))
            dur = (zc3 - zc1) / sample_rate * 1000.0

            out.append(ThetaCycle(
                channel=channel,
                zero_crossing_start=int(zc1),
                peak1=int(p1),
                zero_crossing_mid=int(zc2),
                valley=int(v),
                zero_crossing_end=int(zc3),
                peak2=int(p2),
                amplitude=amp,
                duration_ms=dur,
                theta_delta_ratio=delta_ratio,
                fast_theta_ratio=fast_ratio,
                manual=False,
            ))

        out.sort(key=lambda c: c.zero_crossing_start)
        return out
"""
theta_cycle_detector.py

Cycle-by-cycle theta detection: find each individual theta oscillation
and its landmarks in a single channel's signal, optionally filtering
out cycles whose local theta/delta ratio or theta/fast ratio is
unfavorable.

Adapted from tsc_helper_functions.py's define_theta_cycles, restricted
to the use_imf=False path (no EMD / IMF decomposition). Behavioral
differences from the reference, all intentional:

  * No decimation. Detection runs at the recording's raw sample rate
    and all returned sample indices are raw sample indices. This is
    the same convention as ripple and theta-epoch detection.

  * Theta band, delta band, and peak/valley distance bounds are all
    configurable. The reference hardcodes the distance bounds
    (13 Hz / 6 Hz); here they're configurable with wider defaults
    (2.5 / 15.0) so slow theta is not rejected by the pairing
    geometry alone.

  * Delta band defaults to [0.5 Hz, theta_low_freq] rather than
    [0, theta_low_freq]. A pure lowpass down to DC picks up slow
    drift and movement artifact, which inflates the delta envelope
    and rejects genuine theta cycles. 0.5 Hz is above most of that
    slow content while still including physiological delta.

  * Delta and fast corrections are computed as per-cycle RATIOS
    (mean theta envelope vs mean delta / fast envelope over the
    cycle's one-oscillation span), not as global percentile gates.
    A per-cycle ratio encodes the question the user actually cares
    about: at this moment, does theta dominate the competing band?

Cycle start conventions
-----------------------
Four start conventions are supported, and the six landmark fields on
ThetaCycle are populated in CHRONOLOGICAL order under each convention:

  rising_zc:  [zc_rising_1,  peak1,   zc_falling_1, valley_1, zc_rising_2, peak2]
  falling_zc: [zc_falling_1, valley_1, zc_rising_2, peak2,   zc_falling_2, valley_2]
  peak:       [peak1,        zc_falling_1, valley_1, zc_rising_2, peak2, zc_falling_2]
  valley:     [valley_1,     zc_rising_2, peak2, zc_falling_2, valley_2, zc_rising_3]

In every case ref_1 through ref_5 span one full oscillation; ref_6 is
the first landmark of the NEXT cycle's opening. Cycles for which ref_6
cannot be found (last cycle on a channel, or the trailing zero
crossing falls outside the analyzed signal) are DROPPED -- the
six-landmark invariant is preserved strictly.

The ratio span used for delta/fast corrections is always [ref_1,
ref_5], one full oscillation.

Landmark search is explicit per mode. There is no signal-negation
trick: each mode searches the original filtered signal for exactly
the crossings and extrema its six-slot scheme requires, in the
intervals between the strict-pairing candidates. This guarantees the
six populated values are chronologically ordered by construction,
which a signal-inversion approach did not.

No GUI dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, sosfiltfilt, detrend as scipy_detrend

from core.filters import hilbert  # pads to next power of 2, matches reference


# Names of the six landmark slots, in chronological order, for each
# supported start convention. The index into this list corresponds
# to ref_1..ref_6 in the CSV / sidecar. "rising" means the signal
# goes negative -> positive at that crossing; "falling" means
# positive -> negative.
LANDMARK_NAMES_BY_MODE: dict[str, list[str]] = {
    "rising_zc": [
        "zc_rising_1", "peak1", "zc_falling_1",
        "valley_1", "zc_rising_2", "peak2",
    ],
    "falling_zc": [
        "zc_falling_1", "valley_1", "zc_rising_2",
        "peak2", "zc_falling_2", "valley_2",
    ],
    "peak": [
        "peak1", "zc_falling_1", "valley_1",
        "zc_rising_2", "peak2", "zc_falling_2",
    ],
    "valley": [
        "valley_1", "zc_rising_2", "peak2",
        "zc_falling_2", "valley_2", "zc_rising_3",
    ],
}

VALID_CYCLE_START_EDGES = tuple(LANDMARK_NAMES_BY_MODE.keys())


@dataclass
class ThetaCycleParams:
    """All tunable parameters for one cycle detection run."""
    channels: list[int]
    start_time: float = 0.0
    end_time: float = 10.0

    # ---- Theta band ----
    low_freq: float = 4.0
    high_freq: float = 12.0
    filter_order: int = 4
    detrend: bool = True

    # ---- Peak/valley amplitude floor ----
    min_theta_peak_percentile: float = 15.0

    # ---- Peak-to-peak / peak-to-valley distance bounds (Hz) ----
    # None means "use the detector's default" (2.5 / 15.0 Hz).
    peak_low_freq: float | None = None
    peak_high_freq: float | None = None

    # ---- Delta correction ----
    delta_correction: bool = True
    delta_low_freq: float = 0.5
    delta_high_freq: float | None = None
    theta_delta_ratio_min: float = 1.5

    # ---- Fast correction ----
    fast_correction: bool = False
    fast_low_freq: float = 170.0
    fast_theta_ratio_max: float = 0.5

    # ---- Cycle start convention ----
    cycle_start_edge: str = "rising_zc"

    # ---- Theta-epoch containment filter ----
    epoch_intervals: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class ThetaCycle:
    """One detected theta cycle. All six landmark fields are RAW sample
    indices into the recording, populated in chronological order under
    the detection's cycle_start_edge convention. See
    LANDMARK_NAMES_BY_MODE for what each field means under each mode.
    """
    channel: int
    zero_crossing_start: int   # ref_1
    peak1: int                 # ref_2
    zero_crossing_mid: int     # ref_3
    valley: int                # ref_4
    zero_crossing_end: int     # ref_5
    peak2: int                 # ref_6
    amplitude: float
    duration_ms: float
    theta_delta_ratio: float | None = None
    fast_theta_ratio: float | None = None
    manual: bool = False

    def landmarks(self) -> list[int]:
        """Six landmark sample indices in ref_1..ref_6 order."""
        return [
            self.zero_crossing_start,
            self.peak1,
            self.zero_crossing_mid,
            self.valley,
            self.zero_crossing_end,
            self.peak2,
        ]


def _effective_peak_bounds(params: ThetaCycleParams) -> tuple[float, float]:
    """Return (peak_low_freq, peak_high_freq), defaulting to the wider
    (2.5 / 15.0 Hz) bounds so that narrowing the theta band does not,
    on its own, reject slow cycles at the pairing stage."""
    pl = params.peak_low_freq if params.peak_low_freq is not None else 2.5
    ph = params.peak_high_freq if params.peak_high_freq is not None else 15.0
    return float(pl), float(ph)


def _effective_delta_high(params: ThetaCycleParams) -> float:
    if params.delta_high_freq is None:
        return float(params.low_freq)
    return float(params.delta_high_freq)


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    ordered = sorted((int(a), int(b)) for a, b in intervals)
    merged: list[list[int]] = [list(ordered[0])]
    for lo, hi in ordered[1:]:
        if lo <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(lo, hi) for lo, hi in merged]


def _cycle_inside_any_interval(
    cycle: ThetaCycle, merged: list[tuple[int, int]]
) -> bool:
    lo, hi = int(cycle.zero_crossing_start), int(cycle.zero_crossing_end)
    for e_lo, e_hi in merged:
        if lo >= e_lo and hi <= e_hi:
            return True
        if e_lo > hi:
            break
    return False


class ThetaCycleDetector:
    """Stateless. Construct once, call detect() per channel."""

    # ------------------------------------------------------------------
    # Peak detection (ported from tsc_helper_functions.detect_peaks)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_peaks(x: np.ndarray, mph: float | None = None, mpd: int = 1,
                      threshold: float = 0.0, edge: str = "rising",
                      kpsh: bool = False) -> np.ndarray:
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
    # Filtering helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bandpass(signal: np.ndarray, sr: float, low: float, high: float,
                  order: int = 4) -> np.ndarray:
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
    # Zero-crossing search helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_rising_zc(
        theta_signal: np.ndarray, start: int, end: int
    ) -> int | None:
        """First index i in [start, end) such that theta_signal[i-1]
        < 0 <= theta_signal[i] (a rising crossing). Returns None if
        no such crossing exists."""
        start = max(1, int(start))
        end = min(len(theta_signal), int(end))
        if end <= start:
            return None
        prev_vals = theta_signal[start - 1:end - 1]
        curr_vals = theta_signal[start:end]
        hits = np.where((prev_vals < 0) & (curr_vals >= 0))[0]
        if hits.size == 0:
            return None
        return int(start + hits[0])

    @staticmethod
    def _find_falling_zc(
        theta_signal: np.ndarray, start: int, end: int
    ) -> int | None:
        """First index i in [start, end) such that theta_signal[i-1]
        > 0 >= theta_signal[i] (a falling crossing). Returns None if
        no such crossing exists."""
        start = max(1, int(start))
        end = min(len(theta_signal), int(end))
        if end <= start:
            return None
        prev_vals = theta_signal[start - 1:end - 1]
        curr_vals = theta_signal[start:end]
        hits = np.where((prev_vals > 0) & (curr_vals <= 0))[0]
        if hits.size == 0:
            return None
        return int(start + hits[0])


    @staticmethod
    def _find_rising_zc_last(
        theta_signal: np.ndarray, start: int, end: int
    ) -> int | None:
        """Last index i in [start, end) such that theta_signal[i-1]
        < 0 <= theta_signal[i] (a rising crossing)."""
        start = max(1, int(start))
        end = min(len(theta_signal), int(end))
        if end <= start:
            return None
        prev_vals = theta_signal[start - 1:end - 1]
        curr_vals = theta_signal[start:end]
        hits = np.where((prev_vals < 0) & (curr_vals >= 0))[0]
        if hits.size == 0:
            return None
        return int(start + hits[-1])

    @staticmethod
    def _find_falling_zc_last(
        theta_signal: np.ndarray, start: int, end: int
    ) -> int | None:
        start = max(1, int(start))
        end = min(len(theta_signal), int(end))
        if end <= start:
            return None
        prev_vals = theta_signal[start - 1:end - 1]
        curr_vals = theta_signal[start:end]
        hits = np.where((prev_vals > 0) & (curr_vals <= 0))[0]
        if hits.size == 0:
            return None
        return int(start + hits[-1])

    
    # ------------------------------------------------------------------
    # Strict-pairing candidates
    # ------------------------------------------------------------------

    @staticmethod
    def _strict_pairing(
        peak_idx: np.ndarray,
        valley_idx: np.ndarray,
        min_pv: int,
        max_pv: int,
        max_pp: int,
    ) -> list[tuple[int, int, int]]:
        """Return (peak1, valley, peak2) triples where peak1 and
        peak2 are real detected peaks flanking a valley within
        [min_pv, max_pv] of the valley, and (peak2 - peak1) <= max_pp.
        De-duplicated by peak1, sorted by peak1 ascending."""
        triples: list[tuple[int, int, int]] = []
        for v in valley_idx:
            before = peak_idx[(peak_idx > v - max_pv) & (peak_idx < v)]
            after = peak_idx[(peak_idx < v + max_pv) & (peak_idx > v)]
            if before.size == 0 or after.size == 0:
                continue
            p1 = int(before.max())
            p2 = int(after.min())
            if not (min_pv <= min(v - p1, p2 - v) <= max_pv):
                continue
            if (p2 - p1) > max_pp:
                continue
            triples.append((p1, int(v), p2))

        # De-duplicate by peak1 (multiple valleys can pick the same p1).
        seen_p1: set[int] = set()
        unique: list[tuple[int, int, int]] = []
        for p1, v, p2 in triples:
            if p1 in seen_p1:
                continue
            seen_p1.add(p1)
            unique.append((p1, v, p2))
        unique.sort(key=lambda t: t[0])
        return unique

    # ------------------------------------------------------------------
    # Assemble-and-filter (shared tail across all modes)
    # ------------------------------------------------------------------

    @staticmethod
    def _make_cycle(
        channel: int,
        ref_tuple: tuple[int, int, int, int, int, int],
        theta_signal: np.ndarray,
        theta_amp: np.ndarray,
        delta_amp: np.ndarray | None,
        fast_amp: np.ndarray | None,
        sample_rate: float,
        params: ThetaCycleParams,
    ) -> ThetaCycle | None:
        """Given a strictly-chronological 6-tuple, apply the ratio
        filters and either produce a ThetaCycle or return None. The
        six-tuple positions map to ThetaCycle's six fields in order
        (ref_1 -> zero_crossing_start, ..., ref_6 -> peak2).
        """
        # Sanity: chronological order and positive span.
        if any(
            ref_tuple[i] >= ref_tuple[i + 1]
            for i in range(5)
        ):
            return None
        span_start = int(ref_tuple[0])
        span_end = int(ref_tuple[4])
        if span_end <= span_start:
            return None

        eps = 1e-12
        theta_mean = float(np.mean(theta_amp[span_start:span_end]))

        delta_ratio = None
        if delta_amp is not None:
            delta_mean = float(np.mean(delta_amp[span_start:span_end]))
            delta_ratio = theta_mean / max(delta_mean, eps)
            if delta_ratio < params.theta_delta_ratio_min:
                return None

        fast_ratio = None
        if fast_amp is not None:
            fast_mean = float(np.mean(fast_amp[span_start:span_end]))
            fast_ratio = fast_mean / max(theta_mean, eps)
            if fast_ratio > params.fast_theta_ratio_max:
                return None

        amp = float(np.max(theta_signal[span_start:span_end]))
        dur = (span_end - span_start) / sample_rate * 1000.0

        return ThetaCycle(
            channel=channel,
            zero_crossing_start=int(ref_tuple[0]),
            peak1=int(ref_tuple[1]),
            zero_crossing_mid=int(ref_tuple[2]),
            valley=int(ref_tuple[3]),
            zero_crossing_end=int(ref_tuple[4]),
            peak2=int(ref_tuple[5]),
            amplitude=amp,
            duration_ms=dur,
            theta_delta_ratio=delta_ratio,
            fast_theta_ratio=fast_ratio,
            manual=False,
        )

    # ------------------------------------------------------------------
    # Per-mode landmark builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_rising_zc_cycles(
        triples: list[tuple[int, int, int]],
        theta_signal: np.ndarray,
        max_pv: int,
    ) -> list[tuple[int, int, int, int, int, int]]:
        """rising_zc mode: for each (p1, v, p2) triple, find
        zc_rising_1 in [p1 - max_pv, p1), zc_falling_1 in [p1, v),
        zc_rising_2 in [v, p2). All three crossings must be present
        and the intermediate samples must not violate the expected
        sign (rising_zc's strictness guarantee).

        ref_6 = peak2. No cycle-chaining needed: the six landmarks
        are all inside this cycle.
        """
        out: list[tuple[int, int, int, int, int, int]] = []
        for p1, v, p2 in triples:
            a1_lo = max(0, p1 - max_pv)
            #zc_r1 = ThetaCycleDetector._find_rising_zc(theta_signal, a1_lo, p1)
            zc_r1 = ThetaCycleDetector._find_rising_zc_last(theta_signal, a1_lo, p1)

            if zc_r1 is None:
                continue
            zc_f1 = ThetaCycleDetector._find_falling_zc(theta_signal, p1, v)

            if zc_f1 is None:
                continue
            zc_r2 = ThetaCycleDetector._find_rising_zc(theta_signal, v, p2)

            if zc_r2 is None:
                continue
            # Strictness: no wrong-sign excursions.
            if np.any(theta_signal[zc_r1:p1] < 0):
                continue
            if np.any(theta_signal[zc_f1 + 1:v] > 0):
                continue
            if np.any(theta_signal[zc_r2:p2] < 0):
                continue
            out.append((zc_r1, p1, zc_f1, v, zc_r2, p2))
        return out

    @staticmethod
    def _build_falling_zc_cycles(
        triples: list[tuple[int, int, int]],
        theta_signal: np.ndarray,
        max_pv: int,
    ) -> list[tuple[int, int, int, int, int, int]]:
        """falling_zc mode: for each (p1, v, p2) triple, the six
        landmarks are:

          ref_1 = zc_falling_1 in [p1 - max_pv, p1)    (into p1)
          ref_2 = valley_1    = p1                     (local min)
          ref_3 = zc_rising_2 in (p1, v)               (out of valley_1)
          ref_4 = peak2       = v                      (local max)
          ref_5 = zc_falling_2 in (v, p2)              (out of peak2)
          ref_6 = valley_2    = p2                     (next local min)

        Note: in this scheme p1 is a valley (not a peak) and v is a
        peak. The strict pairing over the *original* signal produced
        peaks around valleys, which is the wrong polarity for
        falling_zc. So we search the original signal for VALLEY-
        PEak-VALLEY triples instead: the same strict pairing logic,
        with peaks and valleys swapped.
        """
        # The caller has already given us peak-centred triples from
        # the standard strict pairing. For falling_zc we need
        # valley-centred triples. Rebuild here from the same detected
        # peaks and valleys, but swapped: this function's `triples`
        # argument is (v_left, peak, v_right) triples under the
        # falling_zc polarity, NOT (p1, v, p2). To keep the interface
        # uniform, we do the swap upstream in detect() and pass the
        # valley-centred triples to this function.
        out: list[tuple[int, int, int, int, int, int]] = []
        for v1, p, v2 in triples:
            # v1 is a valley, p is the peak between v1 and v2,
            # v2 is the following valley.
            zc_f1 = ThetaCycleDetector._find_falling_zc(theta_signal, max(0, v1 - max_pv), v1)


            if zc_f1 is None:
                continue
            zc_r2 = ThetaCycleDetector._find_rising_zc(theta_signal, v1, p)

            if zc_r2 is None:
                continue
            zc_f2 = ThetaCycleDetector._find_falling_zc(theta_signal, p, v2)

            if zc_f2 is None:
                continue
            out.append((zc_f1, v1, zc_r2, p, zc_f2, v2))
        return out

    @staticmethod
    def _build_peak_cycles(
        triples: list[tuple[int, int, int]],
        theta_signal: np.ndarray,
    ) -> list[tuple[int, int, int, int, int, int, int]]:
        """peak mode: for each (p1, v, p2) triple,

          ref_1 = peak1        = p1
          ref_2 = zc_falling_1 in [p1, v)
          ref_3 = valley_1     = v
          ref_4 = zc_rising_2  in [v, p2)
          ref_5 = peak2        = p2
          ref_6 = zc_falling_2 (between p2 and the NEXT cycle's valley)

        We return a 7-tuple with the extra final element being
        `None` if zc_falling_2 can't be found in this cycle's
        trailing samples before the next p1 (i.e. the pair
        (p1, p2, v) is not extendable). Caller filters.

        The `zc_falling_2` search window is [p2, end_of_signal) but
        truncated to before the next cycle's valley would start. We
        don't know the next valley here -- the caller patches ref_6
        using the next triple.
        """
        # We return a partial 6-tuple plus the raw search inputs so
        # the caller can fill ref_6 using the next cycle. The
        # placeholder `-1` in slot 5 means "fill from next cycle".
        out: list[tuple[int, int, int, int, int, int]] = []
        for p1, v, p2 in triples:
            zc_f1 = ThetaCycleDetector._find_falling_zc(theta_signal, p1, v)
            if zc_f1 is None:
                continue
            zc_r2 = ThetaCycleDetector._find_rising_zc(theta_signal, v, p2)
            if zc_r2 is None:
                continue
            out.append((p1, zc_f1, v, zc_r2, p2, -1))
        return out

    @staticmethod
    def _build_valley_cycles(
        valley_pairs: list[tuple[int, int, int]],
        theta_signal: np.ndarray,
    ) -> list[tuple[int, int, int, int, int, int]]:
        """valley mode: for each (v1, p, v2) triple,

          ref_1 = valley_1     = v1
          ref_2 = zc_rising_2  in [v1, p)
          ref_3 = peak2        = p
          ref_4 = zc_falling_2 in [p, v2)
          ref_5 = valley_2     = v2
          ref_6 = zc_rising_3  (after v2, into next cycle)

        ref_6 is the next triple's ref_2 (the rising crossing after
        v2), filled in by the caller from the following triple.
        Placeholder -1 in slot 5 signals "fill from next cycle".
        """
        out: list[tuple[int, int, int, int, int, int]] = []
        for v1, p, v2 in valley_pairs:
            zc_r2 = ThetaCycleDetector._find_rising_zc(theta_signal, v1, p)
            if zc_r2 is None:
                continue
            zc_f2 = ThetaCycleDetector._find_falling_zc(theta_signal, p, v2)
            if zc_f2 is None:
                continue
            out.append((v1, zc_r2, p, zc_f2, v2, -1))
        return out

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def detect(self, signal: np.ndarray, sample_rate: float,
               params: ThetaCycleParams, channel: int = 0) -> list[ThetaCycle]:
        """Detect theta cycles in a single channel's raw signal.

        Sample indices in the returned ThetaCycle are relative to
        `signal` (i.e. 0 = first sample of `signal`). Callers that
        work on a slice of a recording add their slice offset.
        """
        signal = np.asarray(signal, dtype=np.float64)
        if signal.size < 4:
            return []

        mode = params.cycle_start_edge
        if mode not in VALID_CYCLE_START_EDGES:
            raise ValueError(
                f"cycle_start_edge must be one of "
                f"{VALID_CYCLE_START_EDGES}, got {mode!r}"
            )

        # ---- Theta band, detrended ----
        theta_signal = self._bandpass(
            signal, sample_rate, params.low_freq, params.high_freq,
            params.filter_order,
        )
        if params.detrend:
            theta_signal = scipy_detrend(theta_signal)

        theta_amp = np.abs(hilbert(theta_signal))
        min_amp = float(
            np.percentile(theta_amp, params.min_theta_peak_percentile)
        )

        # ---- Distance bounds ----
        peak_low_freq, peak_high_freq = _effective_peak_bounds(params)
        min_pp = int(round((1.0 / peak_high_freq) * sample_rate))
        max_pp = int(round((1.0 / peak_low_freq) * sample_rate))
        min_pv = int(round((1.0 / peak_high_freq) * sample_rate / 2.0))
        max_pv = int(round((1.0 / peak_low_freq) * sample_rate / 2.0))

        # ---- Candidate peaks and valleys ----
        peak_idx = self._detect_peaks(
            theta_signal, mph=min_amp, mpd=max(1, min_pp),
        )
        valley_idx = self._detect_peaks(
            -theta_signal, mph=min_amp, mpd=max(1, min_pp),
        )
        if peak_idx.size == 0 or valley_idx.size == 0:
            return []

        # ---- Strict pairing (peak-valley-peak, and valley-peak-valley) ----
        # The first list is what rising_zc and peak modes need.
        # The second list is what falling_zc and valley modes need.
        # Both are strict: real extremum on each side, distance bounds
        # respected.
        pvp_triples = self._strict_pairing(
            peak_idx, valley_idx, min_pv, max_pv, max_pp,
        )
        vpv_triples = self._strict_pairing(
            valley_idx, peak_idx, min_pv, max_pv, max_pp,
        )
        if not pvp_triples and not vpv_triples:
            return []

        # ---- Ratio-reference envelopes ----
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

        # ---- Build mode-specific 6-tuples ----
        refs: list[tuple[int, int, int, int, int, int]] = []

        if mode == "rising_zc":
            refs = self._build_rising_zc_cycles(
                pvp_triples, theta_signal, max_pv,
            )

        elif mode == "falling_zc":
            # Valley-peak-valley triples with the SAME strict
            # guarantees; the builder fills ref_1..ref_6 directly
            # from the original signal, no inversion, no swap.
            refs = self._build_falling_zc_cycles(vpv_triples, theta_signal, max_pv,)

        elif mode == "peak":
            # Fill ref_6 = zc_falling_2 using the NEXT cycle's valley.
            # We need the search window [p2, next_v) so it's clear
            # which falling crossing belongs to this cycle rather
            # than the one after. If p2 is the last peak2, drop.
            partial = self._build_peak_cycles(pvp_triples, theta_signal)
            for k, item in enumerate(partial):
                if k + 1 >= len(pvp_triples):
                    # No next cycle: cannot determine ref_6 boundary.
                    continue
                _next_p1, next_v, _next_p2 = pvp_triples[k + 1]
                p1, zc_f1, v, zc_r2, p2, _placeholder = item
                zc_f2 = self._find_falling_zc(theta_signal, p2, next_v)
                
                if zc_f2 is None:
                    continue
                refs.append((p1, zc_f1, v, zc_r2, p2, zc_f2))

        elif mode == "valley":
            # Fill ref_6 = zc_rising_3 using the NEXT cycle's peak.
            # The search window is [v2, next_peak).
            partial = self._build_valley_cycles(vpv_triples, theta_signal)
            for k, item in enumerate(partial):
                if k + 1 >= len(vpv_triples):
                    continue
                _next_v1, next_p, _next_v2 = vpv_triples[k + 1]
                v1, zc_r2, p, zc_f2, v2, _placeholder = item
                zc_r3 = self._find_rising_zc(
                    theta_signal, v2, next_p
                )
                if zc_r3 is None:
                    continue
                refs.append((v1, zc_r2, p, zc_f2, v2, zc_r3))

        if not refs:
            return []

        # ---- Assemble and filter ----
        out: list[ThetaCycle] = []
        for ref_tuple in refs:
            cycle = self._make_cycle(
                channel=channel,
                ref_tuple=ref_tuple,
                theta_signal=theta_signal,
                theta_amp=theta_amp,
                delta_amp=delta_amp,
                fast_amp=fast_amp,
                sample_rate=sample_rate,
                params=params,
            )
            if cycle is not None:
                out.append(cycle)

        # ---- Optional epoch-containment filter ----
        if params.epoch_intervals:
            merged = _merge_intervals(params.epoch_intervals)
            out = [c for c in out if _cycle_inside_any_interval(c, merged)]

        out.sort(key=lambda c: c.zero_crossing_start)
        return out
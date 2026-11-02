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

  * Theta band, delta band, and half-cycle distance bounds are all
    configurable. The reference hardcodes peak/valley bounds; here
    they're configurable with wider defaults (2.5 / 15.0 Hz half-cycle
    rates) so slow theta is not rejected by geometry alone.

  * Delta band defaults to [0.5 Hz, theta_low_freq] rather than
    [0, theta_low_freq]. A pure lowpass down to DC picks up slow
    drift and movement artifact, which inflates the delta envelope
    and rejects genuine theta cycles.

  * Delta and fast corrections are computed as per-cycle RATIOS, not
    as global percentile gates.

Half-cycle semantics
--------------------
`half_cycle_low_freq` / `half_cycle_high_freq` bound the DURATION of
each of the two half-cycles that make up a candidate theta cycle. In
every cycle-start convention the six landmark slots are ordered so
that ref_1, ref_3, ref_5 are the "boundary" landmarks (crossings or
extrema) between halves:

    half-cycle 1 = [ref_1, ref_3]
    half-cycle 2 = [ref_3, ref_5]

So one pair of bounds (min_hc, max_hc) covers both halves in every
mode:

    min_hc = round(sr / half_cycle_high_freq)   # fastest accepted half
    max_hc = round(sr / half_cycle_low_freq)    # slowest accepted half

A candidate is rejected if EITHER half is shorter than min_hc or
longer than max_hc. This catches "one fast half + one slow half"
deformations whose total duration happens to fit the theta band but
whose halves are individually outside the accepted rate.

The pairing stage still uses min_pv / max_pv bounds derived from the
same two rates, without the /2 factor that the old peak-to-valley
semantics used -- under the new semantics the two rates bound half-
cycle durations directly.

Cycle start conventions
-----------------------
Four start conventions are supported, and the six landmark fields on
ThetaCycle are populated in CHRONOLOGICAL order under each convention:

  rising_zc:  [zc_rising_1, peak1, zc_falling_1, valley_1, zc_rising_2, peak2]
  falling_zc: [zc_falling_1, valley_1, zc_rising_2, peak2, zc_falling_2, valley_2]
  peak:       [peak1, zc_falling_1, valley_1, zc_rising_2, peak2, zc_falling_2]
  valley:     [valley_1, zc_rising_2, peak2, zc_falling_2, valley_2, zc_rising_3]

In every case ref_1 through ref_5 span one full oscillation; ref_6 is
the first landmark of the NEXT cycle's opening. Cycles for which ref_6
cannot be found are DROPPED.

The ratio span used for delta/fast corrections is always [ref_1,
ref_5], one full oscillation.

No GUI dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, sosfiltfilt, detrend as scipy_detrend

from core.filters import hilbert  # pads to next power of 2, matches reference


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

LANDMARK_DESCRIPTIONS: dict[str, str] = {
    "zc_rising_1":  "rising zero crossing entering the first peak",
    "peak1":        "first peak of the cycle",
    "zc_falling_1": "falling zero crossing between peak1 and valley_1",
    "valley_1":     "trough of the cycle",
    "zc_rising_2":  "rising zero crossing leaving valley_1",
    "peak2":        "second peak of the cycle",
    "zc_falling_2": "falling zero crossing between peak2 and valley_2",
    "valley_2":     "trough of the next cycle",
    "zc_rising_3":  "rising zero crossing leaving valley_2",
}

VALID_CYCLE_START_EDGES = tuple(LANDMARK_NAMES_BY_MODE.keys())


@dataclass
class ThetaCycleParams:
    """All tunable parameters for one cycle detection run.

    Note on backward compatibility
    ------------------------------
    Older sessions stored ``peak_low_freq`` / ``peak_high_freq``. The
    __init__ override below accepts either spelling and maps the old
    names to the new fields, so old sidecars load cleanly. Writes
    (asdict) always use the new names, so a save-then-load round-trip
    upgrades the file's schema on disk.
    """
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

    # ---- Half-cycle rate bounds (Hz) ----
    # Bounds the duration of EACH of the two half-cycles per candidate
    # cycle. See module docstring for the full semantics. None means
    # "use the detector's default" (2.5 / 15.0 Hz).
    half_cycle_low_freq: float | None = None
    half_cycle_high_freq: float | None = None

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

    def __init__(
        self,
        channels: list[int],
        start_time: float = 0.0,
        end_time: float = 10.0,
        low_freq: float = 4.0,
        high_freq: float = 12.0,
        filter_order: int = 4,
        detrend: bool = True,
        min_theta_peak_percentile: float = 15.0,
        half_cycle_low_freq: float | None = None,
        half_cycle_high_freq: float | None = None,
        delta_correction: bool = True,
        delta_low_freq: float = 0.5,
        delta_high_freq: float | None = None,
        theta_delta_ratio_min: float = 1.5,
        fast_correction: bool = False,
        fast_low_freq: float = 170.0,
        fast_theta_ratio_max: float = 0.5,
        cycle_start_edge: str = "rising_zc",
        epoch_intervals: list[tuple[int, int]] | None = None,
        *,
        # Legacy keyword aliases. Accepted for sidecars written before
        # the rename; ignored if the corresponding new field is also
        # provided. Not carried as dataclass fields (see dataclass
        # body above) so asdict() never emits them.
        peak_low_freq: float | None = None,
        peak_high_freq: float | None = None,
    ):
        self.channels = channels
        self.start_time = start_time
        self.end_time = end_time
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.filter_order = filter_order
        self.detrend = detrend
        self.min_theta_peak_percentile = min_theta_peak_percentile

        # New field wins if both old and new are provided.
        if half_cycle_low_freq is None and peak_low_freq is not None:
            half_cycle_low_freq = peak_low_freq
        if half_cycle_high_freq is None and peak_high_freq is not None:
            half_cycle_high_freq = peak_high_freq
        self.half_cycle_low_freq = half_cycle_low_freq
        self.half_cycle_high_freq = half_cycle_high_freq

        self.delta_correction = delta_correction
        self.delta_low_freq = delta_low_freq
        self.delta_high_freq = delta_high_freq
        self.theta_delta_ratio_min = theta_delta_ratio_min
        self.fast_correction = fast_correction
        self.fast_low_freq = fast_low_freq
        self.fast_theta_ratio_max = fast_theta_ratio_max
        self.cycle_start_edge = cycle_start_edge
        self.epoch_intervals = list(epoch_intervals or [])


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
        return [
            self.zero_crossing_start,
            self.peak1,
            self.zero_crossing_mid,
            self.valley,
            self.zero_crossing_end,
            self.peak2,
        ]


def _effective_half_cycle_bounds(
    params: ThetaCycleParams,
) -> tuple[float, float]:
    """Return (half_cycle_low_freq, half_cycle_high_freq), defaulting
    to (2.5, 15.0)."""
    lo = (
        params.half_cycle_low_freq
        if params.half_cycle_low_freq is not None
        else 2.5
    )
    hi = (
        params.half_cycle_high_freq
        if params.half_cycle_high_freq is not None
        else 15.0
    )
    return float(lo), float(hi)


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
        """Last rising crossing in [start, end). See _build_rising_zc_cycles
        for why the pre-crossing search must pick the LAST crossing,
        not the first: at very permissive half_cycle_low_freq the
        window [p1 - max_pv, p1) can contain several crossings, and
        only the last one is the immediate pre-p1 crossing that the
        strictness check (no negative sample between zc and p1)
        requires."""
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
        min_hc: int,
        max_hc: int,
    ) -> ThetaCycle | None:
        """Apply the half-cycle shape filter, the chronological-order
        guard, the ratio filters, and build a ThetaCycle. Positions
        map to ThetaCycle's six fields in order (ref_1 ->
        zero_crossing_start, ..., ref_6 -> peak2)."""
        if any(
            ref_tuple[i] >= ref_tuple[i + 1]
            for i in range(5)
        ):
            return None

        span_start = int(ref_tuple[0])
        span_end = int(ref_tuple[4])
        if span_end <= span_start:
            return None

        # ---- Half-cycle shape filter ----
        # The two halves are [ref_1, ref_3] and [ref_3, ref_5] in every
        # cycle-start mode (see module docstring). Reject if either is
        # outside [min_hc, max_hc].
        hc1 = int(ref_tuple[2]) - int(ref_tuple[0])
        hc2 = int(ref_tuple[4]) - int(ref_tuple[2])
        if hc1 < min_hc or hc1 > max_hc:
            return None
        if hc2 < min_hc or hc2 > max_hc:
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
        out: list[tuple[int, int, int, int, int, int]] = []
        for p1, v, p2 in triples:
            a1_lo = max(0, p1 - max_pv)
            zc_r1 = ThetaCycleDetector._find_rising_zc_last(
                theta_signal, a1_lo, p1
            )
            if zc_r1 is None:
                continue
            zc_f1 = ThetaCycleDetector._find_falling_zc(
                theta_signal, p1, v
            )
            if zc_f1 is None:
                continue
            zc_r2 = ThetaCycleDetector._find_rising_zc(
                theta_signal, v, p2
            )
            if zc_r2 is None:
                continue
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
        out: list[tuple[int, int, int, int, int, int]] = []
        for v1, p, v2 in triples:
            zc_f1 = ThetaCycleDetector._find_falling_zc_last(
                theta_signal, max(0, v1 - max_pv), v1
            )
            if zc_f1 is None:
                continue
            zc_r2 = ThetaCycleDetector._find_rising_zc(
                theta_signal, v1, p
            )
            if zc_r2 is None:
                continue
            zc_f2 = ThetaCycleDetector._find_falling_zc(
                theta_signal, p, v2
            )
            if zc_f2 is None:
                continue
            out.append((zc_f1, v1, zc_r2, p, zc_f2, v2))
        return out

    @staticmethod
    def _build_peak_cycles(
        triples: list[tuple[int, int, int]],
        theta_signal: np.ndarray,
    ) -> list[tuple[int, int, int, int, int, int]]:
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
        theta_signal = self._bandpass(signal, sample_rate, params.low_freq, params.high_freq,params.filter_order,)
        
        if params.detrend:
            theta_signal = scipy_detrend(theta_signal)

        theta_amp = np.abs(hilbert(theta_signal))
        min_amp = float(np.percentile(theta_amp, params.min_theta_peak_percentile))

        # ---- Half-cycle rate bounds -> sample-duration bounds ----
        hc_low_freq, hc_high_freq = _effective_half_cycle_bounds(params)
        min_hc = int(round((1.0 / hc_high_freq) * sample_rate))
        max_hc = int(round((1.0 / hc_low_freq) * sample_rate))

        # Pairing stage: min_pv/max_pv bound the peak-to-valley search
        # radius. The distance is at most one half-cycle (peak to
        # valley), so min_pv/max_pv take the same duration bounds as
        # min_hc/max_hc. max_pp bounds the full cycle: at most two
        # half-cycles.
        min_pv = min_hc
        max_pv = max_hc
        max_pp = 2 * max_hc
        min_pp = max(1, min_hc)

        # ---- Candidate peaks and valleys ----
        peak_idx = self._detect_peaks(
            theta_signal, mph=min_amp, mpd=max(1, min_pp),
        )
        valley_idx = self._detect_peaks(
            -theta_signal, mph=min_amp, mpd=max(1, min_pp),
        )
        if peak_idx.size == 0 or valley_idx.size == 0:
            return []

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
            refs = self._build_falling_zc_cycles(
                vpv_triples, theta_signal, max_pv,
            )
        elif mode == "peak":
            partial = self._build_peak_cycles(pvp_triples, theta_signal)
            for k, item in enumerate(partial):
                if k + 1 >= len(pvp_triples):
                    continue
                _next_p1, next_v, _next_p2 = pvp_triples[k + 1]
                p1, zc_f1, v, zc_r2, p2, _placeholder = item
                zc_f2 = self._find_falling_zc(theta_signal, p2, next_v)
                if zc_f2 is None:
                    continue
                refs.append((p1, zc_f1, v, zc_r2, p2, zc_f2))
        elif mode == "valley":
            partial = self._build_valley_cycles(vpv_triples, theta_signal)
            for k, item in enumerate(partial):
                if k + 1 >= len(vpv_triples):
                    continue
                _next_v1, next_p, _next_v2 = vpv_triples[k + 1]
                v1, zc_r2, p, zc_f2, v2, _placeholder = item
                zc_r3 = self._find_rising_zc(theta_signal, v2, next_p)
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
                min_hc=min_hc,
                max_hc=max_hc,
            )
            if cycle is not None:
                out.append(cycle)

        if params.epoch_intervals:
            merged = _merge_intervals(params.epoch_intervals)
            out = [c for c in out if _cycle_inside_any_interval(c, merged)]

        out.sort(key=lambda c: c.zero_crossing_start)
        return out
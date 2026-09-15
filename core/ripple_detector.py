"""
ripple_detector.py

Sharp-wave ripple (SWR) detection: bandpass filter -> envelope (Hilbert or
RMS) -> dual-threshold crossing detection (peak threshold + boundary
threshold), independently per channel.

Design notes
------------
- Takes a single already-extracted 1D signal (raw LFP or CSD) rather than
  raw multi-channel data + probe geometry. CSD computation stays in
  core/phase_amplitude.py's PhaseAmplitudeAnalyzer (already implemented
  and tested there) -- callers needing CSD call that first, then pass the
  resulting signal in here. This keeps CSD logic in exactly one place
  and makes RippleDetector trivially testable with synthetic signals.
- CAVEAT: peak/boundary thresholds are computed from the mean and SD of
  the same signal being analyzed (matching the spec: "4 std" / "1 std").
  This means a very long or high-amplitude artifact can inflate the SD
  enough to suppress detection of other genuine ripples in the same
  window -- a known property of self-referential z-score thresholding,
  not a bug. If this becomes a problem in practice, computing mean/SD
  from a separate quiet baseline period instead of the analyzed window
  is the standard fix, but isn't implemented here since it wasn't asked
  for.
- Detection is per-channel by design: each channel gets an independent
  RippleEvent list, so channels/shanks can be compared to find which one
  has the strongest ripple power.
- Events are plain, mutable dataclass instances held in a list -- the
  caller (dialog) can append/remove entries directly to support manual
  add/remove during inspection, without needing detector methods for
  every possible edit.

No GUI dependency.

Usage
-----
    detector = RippleDetector()
    params = RippleParams(low_freq=120, high_freq=250, peak_threshold_sd=4.0,
                           boundary_threshold_sd=1.0)
    events = detector.detect(signal, sample_rate=2500, params=params)
    # events: list[RippleEvent], each with start/end/peak/trough sample
    # indices and amplitudes in the original signal's units.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert


@dataclass
class RippleParams:
    low_freq: float = 120.0
    high_freq: float = 250.0
    filter_order: int = 4
    envelope_method: str = "hilbert"   # "hilbert" or "rms"
    rms_window_ms: float = 10.0        # only used when envelope_method == "rms"
    peak_threshold_sd: float = 4.0     # candidate must have a sample above mean + N*SD
    boundary_threshold_sd: float = 1.0  # start/end where envelope crosses mean + N*SD
    min_duration_ms: float | None = 15.0
    max_duration_ms: float | None = 250.0


@dataclass
class RippleEvent:
    """One detected (or manually added) ripple. Sample indices are
    relative to the signal passed into detect() -- callers combine with
    whatever start_time/offset they used to extract that segment."""
    channel: int
    start_sample: int
    end_sample: int
    peak_sample: int
    peak_amplitude: float       # envelope value at peak_sample
    trough_sample: int          # sample of the most negative raw deflection within the event
    trough_amplitude: float     # raw signal value at trough_sample
    manual: bool = False        # True if added/edited by hand rather than detected


class RippleDetector:
    """Stateless detection logic; safe to reuse across channels/calls."""

    def apply_ripple_filter(self, signal: np.ndarray, sample_rate: float,
                             params: RippleParams) -> np.ndarray:
        """Exposed separately from detect() so callers that need to
        *display* the exact filtered signal used for detection (e.g. an
        inspector view showing raw vs. filtered) get precisely the same
        result, rather than risking a second, possibly-drifting
        reimplementation of the same filter elsewhere."""
        nyq = 0.5 * sample_rate
        if params.low_freq >= params.high_freq:
            raise ValueError("low_freq must be below high_freq")
        if params.high_freq >= nyq:
            raise ValueError(
                f"high_freq ({params.high_freq} Hz) must be below Nyquist ({nyq} Hz)"
            )
        sos = butter(params.filter_order, [params.low_freq / nyq, params.high_freq / nyq],
                     btype="band", output="sos")
        return sosfiltfilt(sos, np.asarray(signal, dtype=np.float64))

    def compute_envelope(self, filtered: np.ndarray, sample_rate: float,
                          params: RippleParams) -> np.ndarray:
        if params.envelope_method == "hilbert":
            return np.abs(hilbert(filtered))
        elif params.envelope_method == "rms":
            window_samples = max(1, int(params.rms_window_ms / 1000 * sample_rate))
            kernel = np.ones(window_samples) / window_samples
            squared = filtered ** 2
            # 'same' mode centered RMS via convolution, matching the
            # window semantics used elsewhere in this codebase (sliding
            # window, edge-padded by convolution's own boundary handling)
            mean_sq = np.convolve(squared, kernel, mode="same")
            return np.sqrt(mean_sq)
        else:
            raise ValueError("envelope_method must be 'hilbert' or 'rms'")

    def detect(self, signal: np.ndarray, sample_rate: float,
               params: RippleParams, channel: int = 0,
               exclude_mask: np.ndarray | None = None) -> list[RippleEvent]:
        """
        Parameters
        ----------
        signal : 1D array, raw (or CSD) LFP for one channel
        sample_rate : Hz
        params : RippleParams
        channel : channel identifier stored on each returned RippleEvent
                  (purely a label, doesn't affect detection)
        exclude_mask : optional 1D boolean array, same length as signal.
                  True marks samples to exclude from BOTH (a) the
                  mean/SD baseline used to set peak/boundary thresholds,
                  and (b) candidate peak selection -- a candidate whose
                  peak sample falls in an excluded region is dropped
                  entirely, same as if it were never detected. This is
                  the mechanism behind "detect ripples only outside
                  theta epochs": passing a mask of theta-epoch samples
                  means high-frequency contamination during theta can't
                  inflate the threshold used for the rest of the signal,
                  which a purely post-hoc peak-location filter (compute
                  threshold from everything, then throw away theta-time
                  detections) would not fix.

        Returns
        -------
        list[RippleEvent], sorted by start_sample ascending.
        """
        signal = np.asarray(signal, dtype=np.float64)
        filtered = self.apply_ripple_filter(signal, sample_rate, params)

        envelope = self.compute_envelope(filtered, sample_rate, params)

        if exclude_mask is not None:
            exclude_mask = np.asarray(exclude_mask, dtype=bool)
            if exclude_mask.shape != envelope.shape:
                raise ValueError(
                    f"exclude_mask shape {exclude_mask.shape} must match "
                    f"envelope shape {envelope.shape}"
                )
            baseline_samples = envelope[~exclude_mask]
            if baseline_samples.size == 0:
                # Everything excluded -- nothing to compute a baseline
                # from, so nothing can be detected either.
                return []
        else:
            baseline_samples = envelope

        mean_env = np.mean(baseline_samples)
        sd_env = np.std(baseline_samples)
        peak_threshold = mean_env + params.peak_threshold_sd * sd_env
        boundary_threshold = mean_env + params.boundary_threshold_sd * sd_env

        # Candidate peak samples: local points above the peak threshold.
        # Excluded samples never qualify as a candidate peak, on top of
        # not contributing to the threshold above.
        above_peak = envelope > peak_threshold
        if exclude_mask is not None:
            above_peak = above_peak & ~exclude_mask
        if not np.any(above_peak):
            return []

        # Group contiguous above-peak-threshold runs; take the argmax
        # within each run as the candidate peak (there can be multiple
        # separate excursions above peak_threshold within a longer
        # above-boundary-threshold segment -- each is its own candidate,
        # they get merged below only if their boundary-threshold windows
        # overlap, otherwise "stay separate" per spec).
        above_peak_idx = np.where(above_peak)[0]
        gaps = np.where(np.diff(above_peak_idx) > 1)[0]
        run_starts = np.concatenate(([0], gaps + 1))
        run_ends = np.concatenate((gaps, [len(above_peak_idx) - 1]))

        events: list[RippleEvent] = []
        n = len(envelope)

        for rs, re in zip(run_starts, run_ends):
            run_indices = above_peak_idx[rs:re + 1]
            # Peak = highest envelope sample within this above-peak-threshold run.
            peak_sample = int(run_indices[np.argmax(envelope[run_indices])])

            # Walk left from peak until envelope drops below boundary_threshold
            start_sample = peak_sample
            while start_sample > 0 and envelope[start_sample] > boundary_threshold:
                start_sample -= 1
            # Walk right from peak until envelope drops below boundary_threshold
            end_sample = peak_sample
            while end_sample < n - 1 and envelope[end_sample] > boundary_threshold:
                end_sample += 1

            duration_ms = (end_sample - start_sample) / sample_rate * 1000
            if params.min_duration_ms is not None and duration_ms < params.min_duration_ms:
                continue
            if params.max_duration_ms is not None and duration_ms > params.max_duration_ms:
                continue

            window = filtered[start_sample:end_sample + 1]
            if len(window) == 0:
                continue
            trough_rel = int(np.argmin(window))
            trough_sample = start_sample + trough_rel

            events.append(RippleEvent(
                channel=channel,
                start_sample=start_sample,
                end_sample=end_sample,
                peak_sample=peak_sample,
                peak_amplitude=float(envelope[peak_sample]),
                trough_sample=trough_sample,
                trough_amplitude=float(filtered[trough_sample]),
                manual=False,
            ))

        # Deduplicate: multiple above-peak-threshold runs can expand (via
        # the boundary-threshold walk) into overlapping or identical
        # [start_sample, end_sample] windows when they're close together.
        # Per spec, ripples "stay separate" rather than merge -- but two
        # runs walking to the exact same boundary-crossing window is not
        # "separate events", it's the same event detected twice, so exact
        # duplicates are collapsed (keeping the one with the higher peak).
        deduped: dict[tuple[int, int], RippleEvent] = {}
        for ev in events:
            key = (ev.start_sample, ev.end_sample)
            if key not in deduped or ev.peak_amplitude > deduped[key].peak_amplitude:
                deduped[key] = ev

        result = sorted(deduped.values(), key=lambda e: e.start_sample)
        return result

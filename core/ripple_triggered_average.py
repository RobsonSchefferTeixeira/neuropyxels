"""
ripple_triggered_average.py

Ripple-triggered average (RTA): extract a fixed time window around each
detected ripple event's alignment point, for one or more channels, and
average across events -- the standard "spike/event-triggered average"
technique applied to ripples instead of spikes.

No GUI dependency, mirroring core/amplitude_power.py and
core/phase_amplitude.py's separation from their dialogs.

Design notes
------------
- Alignment point per event is configurable, matching the three
  reference points a person might reasonably want to align on:
    'envelope_peak'  -- RippleEvent.peak_sample (already computed at
                         detection time: the envelope's maximum, i.e.
                         "ripple max amplitude").
    'filtered_min'   -- RippleEvent.trough_sample (already computed at
                         detection time: the filtered signal's most
                         negative deflection within the event).
    'filtered_max'   -- the filtered signal's most POSITIVE deflection
                         within the event. NOT stored on RippleEvent
                         (only the trough/minimum is), so this is
                         computed here on demand from a re-filtered
                         window around the event.
  'filtered_min'/'filtered_max' together are what the person means by
  "ripple max/min cycle" -- the peak and trough of the oscillation
  itself, as opposed to its envelope.
- Averaging is always done on RAW (or CSD, if that's what was detected
  on) broadband LFP, pulled directly from engine.data at absolute
  sample positions -- NOT the filtered/ripple-band signal -- since the
  point of an RTA is usually to see the LFP waveform (sharp wave +
  ripple, slow components, etc.) around the event, not just the
  band-limited oscillation. The alignment SAMPLE can come from the
  filtered signal (see above); the WAVEFORM that gets averaged is raw.
- Works across channels that never had their own detection run: only
  the triggering events' (channel, sample) positions matter for where
  to center each window, and the averaged channel is independent of
  which channel(s) the events were detected on. This lets you ask "what
  does channel 12's LFP look like around ripples detected on channel 5"
  directly.
- Samples are always indexed via plain sample/sample_rate math, never
  engine.timestamps -- consistent with the rest of ripple/theta
  detection (see ripple_dialog.py's _DetectThread and
  ThetaEpochTraceViewWidget for the same convention).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.ripple_detector import RippleEvent, RippleParams, RippleDetector


ALIGNMENT_MODES = ("envelope_peak", "filtered_max", "filtered_min")


@dataclass
class RippleTriggeredAverageParams:
    channels: list[int]
    window_before_ms: float = 100.0
    window_after_ms: float = 100.0
    alignment: str = "envelope_peak"  # one of ALIGNMENT_MODES

    def __post_init__(self):
        if self.alignment not in ALIGNMENT_MODES:
            raise ValueError(f"alignment must be one of {ALIGNMENT_MODES}, got {self.alignment!r}")
        if self.window_before_ms < 0 or self.window_after_ms < 0:
            raise ValueError("window_before_ms/window_after_ms must be >= 0")
        if self.window_before_ms + self.window_after_ms <= 0:
            raise ValueError("Total window (before + after) must be > 0")


class RippleTriggeredAverageAnalyzer:
    """Stateless-ish computation: holds nothing but the ripple filter
    params needed to re-derive filtered_max/filtered_min alignment
    points on demand; every compute() call is independent."""

    def __init__(self, ripple_params: RippleParams):
        self.ripple_params = ripple_params
        self._detector = RippleDetector()

    def _alignment_sample(self, event: RippleEvent, raw_data: np.ndarray,
                           sample_rate: float, alignment: str) -> int | None:
        """
        Return the GLOBAL (recording-absolute) sample to center this
        event's window on, or None if it can't be determined (e.g. the
        event's own channel data is unavailable).

        event.start_sample/end_sample/peak_sample/trough_sample are all
        relative to whatever sample_offset was in effect at detection
        time -- but RippleEvent itself doesn't carry that offset (see
        module docstring in ripple_detector.py: "callers combine with
        whatever start_time/offset they used"). Callers of THIS module
        are expected to have already re-based event sample fields to
        GLOBAL/absolute terms before calling compute() (see
        RippleTriggeredAverageAnalyzer.compute's docstring) -- this
        keeps this module's internals in one consistent sample basis
        throughout, rather than threading a second offset parameter
        through every method.
        """
        if alignment == "envelope_peak":
            return int(event.peak_sample)
        if alignment == "filtered_min":
            return int(event.trough_sample)

        # filtered_max: not precomputed on RippleEvent, so re-filter
        # just the event's own [start_sample, end_sample] window (plus
        # a little padding for filter edge effects) and find the peak.
        if event.channel >= raw_data.shape[1]:
            return None
        pad = max(1, int(round(0.01 * sample_rate)))  # 10ms padding
        lo = max(0, event.start_sample - pad)
        hi = min(raw_data.shape[0] - 1, event.end_sample + pad)
        if hi <= lo:
            return None
        segment = raw_data[lo:hi + 1, event.channel].astype(np.float64)
        try:
            filtered = self._detector.apply_ripple_filter(segment, sample_rate, self.ripple_params)
        except ValueError:
            return None
        # Restrict the argmax search back to the true event window
        # (excluding the filter-edge padding) so a large edge artifact
        # just outside the event can't be picked as "the peak".
        event_lo_rel = event.start_sample - lo
        event_hi_rel = event.end_sample - lo
        event_lo_rel = max(0, event_lo_rel)
        event_hi_rel = min(len(filtered) - 1, event_hi_rel)
        if event_hi_rel < event_lo_rel:
            return None
        window = filtered[event_lo_rel:event_hi_rel + 1]
        peak_rel = int(np.argmax(window))
        return lo + event_lo_rel + peak_rel

    def compute(self, raw_data: np.ndarray, sample_rate: float,
                events: list[RippleEvent], params: RippleTriggeredAverageParams) -> dict:
        """
        Parameters
        ----------
        raw_data : (n_samples, n_channels) array-like, e.g. engine.data
        sample_rate : Hz (engine.sr)
        events : list[RippleEvent] with sample fields ALREADY RE-BASED
            to global/absolute recording-sample terms (i.e. the
            caller has added each event's detection-time sample_offset
            in before calling this -- see ripple_dialog.py's
            _rta_events_in_global_samples helper). Events may span
            multiple source channels; only the ones actually used for
            alignment matter, each event contributes to every requested
            output channel's average using that SAME event's alignment
            sample (i.e. "ripples detected on channel 5" can still be
            used to average channel 12's LFP around those same moments).
        params : RippleTriggeredAverageParams

        Returns
        -------
        dict: {
            'time_axis': (n_window,) ndarray, seconds, 0 = alignment sample,
            'channels': {
                channel: {
                    'mean': (n_window,) ndarray,
                    'sem': (n_window,) ndarray,
                    'n_events': int,
                    'per_event': (n_events_used, n_window) ndarray,
                },
                ...
            },
            'n_events_total': int,       # events with a usable alignment sample
            'n_events_skipped': int,     # events too close to a recording edge, or filter failure
            'params': params,
        }
        """
        if not events:
            raise ValueError("No ripple events provided.")
        if not params.channels:
            raise ValueError("No channels selected for the ripple-triggered average.")

        n_before = int(round(params.window_before_ms / 1000.0 * sample_rate))
        n_after = int(round(params.window_after_ms / 1000.0 * sample_rate))
        n_window = n_before + n_after + 1
        time_axis = (np.arange(n_window) - n_before) / sample_rate

        n_total_samples = raw_data.shape[0]
        n_total_channels = raw_data.shape[1]

        alignment_samples: list[int] = []
        skipped = 0
        for event in events:
            a_sample = self._alignment_sample(event, raw_data, sample_rate, params.alignment)
            if a_sample is None:
                skipped += 1
                continue
            lo = a_sample - n_before
            hi = a_sample + n_after
            if lo < 0 or hi >= n_total_samples:
                # Too close to a recording edge for a full window --
                # skipped rather than zero-padded, so the average isn't
                # biased by partially-fabricated edge data.
                skipped += 1
                continue
            alignment_samples.append(a_sample)

        if not alignment_samples:
            raise ValueError(
                "No ripple events had a usable alignment window (all were "
                "too close to a recording edge, or filtering failed)."
            )

        channels_out = {}
        for ch in params.channels:
            if ch < 0 or ch >= n_total_channels:
                continue
            per_event = np.empty((len(alignment_samples), n_window), dtype=np.float64)
            for i, a_sample in enumerate(alignment_samples):
                lo = a_sample - n_before
                hi = a_sample + n_after
                per_event[i] = raw_data[lo:hi + 1, ch].astype(np.float64)

            mean = np.mean(per_event, axis=0)
            sem = (
                np.std(per_event, axis=0, ddof=1) / np.sqrt(per_event.shape[0])
                if per_event.shape[0] > 1
                else np.zeros(n_window)
            )
            channels_out[ch] = {
                "mean": mean,
                "sem": sem,
                "n_events": per_event.shape[0],
                "per_event": per_event,
            }

        return {
            "time_axis": time_axis,
            "channels": channels_out,
            "n_events_total": len(alignment_samples),
            "n_events_skipped": skipped,
            "params": params,
        }

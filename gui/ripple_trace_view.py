"""
ripple_trace_view.py

TraceViewWidget extension that draws and edits ripple events directly on
the main trace view, mirroring theta_epoch_trace_view.py's architecture
(ThetaEpochTraceViewWidget) so ripple detection has the same interaction
model as theta epoch detection: click-select, drag-to-resize boundaries,
delete, merge-on-drag for overlapping/adjacent events, split-by-
double-click, and arm-then-drag creation of new manual events.

Design notes
------------
- RippleEvent (core/ripple_detector.py) is the source-of-truth data
  object, analogous to ThetaEpoch -- this widget owns interactive state
  (selection, drag) plus purely-rendering-only per-channel context
  (envelope, raw/CSD signal, sample_offset/sample_rate, thresholds) that
  isn't part of a "ripple event" the way epoch boundaries are. That
  context is set separately via set_ripple_render_context() so
  RippleEvent objects themselves stay lean and mergeable exactly like
  ThetaEpoch.
- Only same-channel events participate in merge-candidate detection,
  same as theta.
- INHERITS DIRECTLY FROM ThetaEpochTraceViewWidget (not from
  TraceViewWidget, and not combined with it via multiple inheritance).
  An earlier version of this file used
  class NeuralTraceViewWidget(RippleTraceViewWidget, ThetaEpochTraceViewWidget)
  -- two independent QWidget subclasses combined via Python multiple
  inheritance. That pattern is NOT supported by PyQt6's meta-object
  system: pyqtSignal attributes declared on whichever base class ends
  up second in the MRO fail to bind at connect() time. Single, linear
  inheritance (this class -> ThetaEpochTraceViewWidget ->
  TraceViewWidget) is the only reliable way to combine both overlays.

  Every overridden Qt event handler calls super().<handler>(event)
  either at the top (paintEvent) or as its final fallthrough branch
  (mouse/key handlers), so a mouse click first gets a chance to hit a
  ripple handle/event; if it misses, it falls through to
  ThetaEpochTraceViewWidget's own handle/epoch hit-test; if that also
  misses, it finally reaches TraceViewWidget's base pan/zoom/cursor
  logic.
"""

from __future__ import annotations

import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal, QRectF
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QCursor, QPainterPath

from core.ripple_detector import RippleEvent
from gui.theta_epoch_trace_view import ThetaEpochTraceViewWidget


class RippleTraceViewWidget(ThetaEpochTraceViewWidget):
    """TraceViewWidget with an interactive ripple-event overlay."""

    rippleEventsChanged = pyqtSignal(object)  # list[RippleEvent]
    rippleEventSelected = pyqtSignal(int)

    def __init__(self, engine, parent=None):
        super().__init__(engine, parent)

        self.ripple_events: list[RippleEvent] = []

        # Per-channel rendering context, NOT part of RippleEvent itself:
        # {channel: {'envelope': ndarray, 'signal': ndarray,
        #            'sample_offset': int, 'sample_rate': float,
        #            'peak_threshold_sd': float, 'boundary_threshold_sd': float}}
        self._ripple_render_context: dict[int, dict] = {}

        self.show_ripple_envelope = True
        self.show_ripple_thresholds = True
        self.show_ripple_events = True
        self.show_ripple_filtered = False

        self._ripple_drag_event: int | None = None
        self._ripple_drag_side: str | None = None
        self._ripple_merge_candidate: int | None = None
        self._ripple_merge_candidates: set[int] = set()
        self._ripple_selected_event: int | None = None
        self.ripple_min_merge_gap_ms = 0.0  # 0 = no auto-merge unless events actually overlap
        self.ripple_min_merge_gap_samples = 0
        self._ripple_drag_original_start = 0
        self._ripple_drag_original_end = 0

        # Drag-handle HIT-TEST radius (invisible at both boundaries),
        # so resizing works from either edge without a visible circle
        # cluttering the display.
        self._ripple_handle_hit_radius = 12.0
        # Visible peak-marker radius.
        self._ripple_peak_marker_radius = 4.5

        # Minimum duration (in samples) for a ripple created by hand
        # (arm-then-drag) or by splitting an existing one. Anything
        # shorter is discarded, so a stray click can't create a
        # degenerate zero-width event.
        self._ripple_min_duration_samples = max(1, int(round(0.005 * self.engine.sr)))

        # ---- Arm-and-drag creation state ----
        # Set by arm_ripple_creation(); the NEXT left-press inside a
        # channel lane starts a new manual ripple whose width is
        # defined by the drag. Same UX as ThetaEpochTraceViewWidget.
        self._ripple_creating_armed = False
        self._ripple_creating_channel: int | None = None
        self._ripple_creating_start_sample: int | None = None
        self._ripple_creating_end_sample: int | None = None

    # ------------------------------------------------------------------
    # Ripple event API
    # ------------------------------------------------------------------

    def set_ripple_events(self, events: list[RippleEvent] | None):
        """Replace the current set of ripple events (source of truth for
        rendering/interaction). Does not touch render context -- call
        set_ripple_render_context() separately (typically once, right
        after detection)."""
        self.ripple_events = list(events or [])
        self._ripple_drag_event = None
        self._ripple_drag_side = None
        self._ripple_merge_candidate = None
        self._ripple_merge_candidates.clear()
        self.update()

    def set_ripple_render_context(self, context: dict[int, dict] | None):
        """Set (or replace) per-channel envelope/signal/threshold data
        used purely for rendering. `context` is
        {channel: {'envelope', 'signal', 'sample_offset', 'sample_rate',
                    'peak_threshold_sd', 'boundary_threshold_sd'}}."""
        self._ripple_render_context = dict(context or {})
        self.update()

    def set_ripple_overlay_visibility(self, show_envelope: bool | None = None,
                                       show_thresholds: bool | None = None,
                                       show_events: bool | None = None,
                                       show_filtered: bool | None = None):
        if show_envelope is not None:
            self.show_ripple_envelope = show_envelope
        if show_thresholds is not None:
            self.show_ripple_thresholds = show_thresholds
        if show_events is not None:
            self.show_ripple_events = show_events
        if show_filtered is not None:
            self.show_ripple_filtered = show_filtered
        self.update()

    def set_ripple_merge_gap(self, gap_ms: float, update: bool = True):
        self.ripple_min_merge_gap_ms = max(0.0, float(gap_ms))
        self.ripple_min_merge_gap_samples = int(round(self.ripple_min_merge_gap_ms / 1000.0 * self.engine.sr))
        if update:
            self.update()

    def set_ripple_selected_event(self, index: int):
        """Select a ripple event by its current table/overlay index."""
        if 0 <= int(index) < len(self.ripple_events):
            self._ripple_selected_event = int(index)
            self.update()

    def delete_selected_ripple_event(self):
        i = self._ripple_selected_event
        if i is None or not (0 <= i < len(self.ripple_events)):
            return False
        self.ripple_events.pop(i)
        self._ripple_selected_event = None
        self._ripple_drag_event = None
        self._ripple_drag_side = None
        self._ripple_merge_candidate = None
        self._ripple_merge_candidates.clear()
        self.rippleEventsChanged.emit(self.ripple_events)
        self.update()
        return True

    def clear_ripple_overlay(self):
        """Clear both events and render context -- name kept for parity
        with the old TraceViewWidget.clear_ripple_overlay() API that
        RippleDialog already calls."""
        self.ripple_events = []
        self._ripple_render_context = {}
        self._ripple_drag_event = None
        self._ripple_drag_side = None
        self._ripple_merge_candidate = None
        self._ripple_merge_candidates.clear()
        self._ripple_selected_event = None
        self._ripple_creating_armed = False
        self._ripple_creating_channel = None
        self._ripple_creating_start_sample = None
        self._ripple_creating_end_sample = None
        self.update()

    # ------------------------------------------------------------------
    # Creation / split API
    # ------------------------------------------------------------------

    def arm_ripple_creation(self):
        """Arm click-drag creation of a new manual ripple event: the
        NEXT left-button press-and-drag inside a channel lane creates
        a new event there (start = press position, end = release
        position). Disarms itself automatically once the drag
        completes, or if the press lands outside any channel lane.
        Same UX as ThetaEpochTraceViewWidget.arm_theta_creation()."""
        self._ripple_creating_armed = True
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))

    def cancel_ripple_creation(self):
        """Disarm creation mode without creating anything."""
        self._ripple_creating_armed = False
        self._ripple_creating_channel = None
        self._ripple_creating_start_sample = None
        self._ripple_creating_end_sample = None
        self.unsetCursor()

    def split_selected_ripple_at(self, split_sample: int) -> bool:
        """Split the currently-selected ripple event at `split_sample`
        (a sample index relative to that event's own analysis window).

        Both halves are marked manual=True and duplicate the original's
        peak_amplitude and trough_sample/amplitude unchanged --
        recomputing them properly would require re-running the ripple
        filter on each half, which the trace view doesn't have the
        full processing context for. Duplicating is the same pragmatic
        choice the theta merge logic makes when it averages stats
        rather than re-detecting.

        Returns True if a split happened, False if it was rejected
        (no selection, or split_sample too close to an edge to leave
        both halves above the minimum duration)."""
        i = self._ripple_selected_event
        if i is None or not (0 <= i < len(self.ripple_events)):
            return False

        event = self.ripple_events[i]
        min_w = self._ripple_min_duration_samples
        if split_sample - event.start_sample < min_w or event.end_sample - split_sample < min_w:
            return False

        first = RippleEvent(
            channel=event.channel,
            start_sample=event.start_sample,
            end_sample=split_sample,
            peak_sample=min(event.peak_sample, split_sample),
            peak_amplitude=event.peak_amplitude,
            trough_sample=min(event.trough_sample, split_sample),
            trough_amplitude=event.trough_amplitude,
            manual=True,
        )
        second = RippleEvent(
            channel=event.channel,
            start_sample=split_sample,
            end_sample=event.end_sample,
            peak_sample=max(event.peak_sample, split_sample),
            peak_amplitude=event.peak_amplitude,
            trough_sample=max(event.trough_sample, split_sample),
            trough_amplitude=event.trough_amplitude,
            manual=True,
        )

        self.ripple_events.pop(i)
        self.ripple_events.append(first)
        self.ripple_events.append(second)
        self.ripple_events.sort(key=lambda e: (e.channel, e.start_sample))

        self._ripple_selected_event = next(j for j, e in enumerate(self.ripple_events) if e is first)
        self._ripple_drag_event = None
        self._ripple_drag_side = None
        self._ripple_merge_candidate = None
        self._ripple_merge_candidates.clear()

        self.rippleEventsChanged.emit(self.ripple_events)
        self.rippleEventSelected.emit(self._ripple_selected_event)
        self.update()
        return True

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _ripple_context_for(self, channel: int) -> dict | None:
        return self._ripple_render_context.get(channel)

    def _ripple_event_time(self, event: RippleEvent, sample: int) -> float:
        """Convert a relative-to-analysis-window sample to recording time.
        Always pure sample/sample_rate math -- never engine.timestamps --
        so ripple identity and on-screen position are identical whether
        or not the user has loaded a timestamps.npy file."""
        ctx = self._ripple_context_for(event.channel)
        sample_offset = int(ctx.get("sample_offset", 0)) if ctx else 0
        sample_rate = float(ctx.get("sample_rate", self.engine.sr)) if ctx else self.engine.sr
        global_sample = sample_offset + int(sample)
        return global_sample / float(sample_rate)

    def _ripple_time_to_x(self, time: float) -> float:
        plot_left, plot_right, _ = self._get_plot_bounds()
        if self.window_duration <= 0:
            return plot_left
        return plot_left + ((time - self.start_time) / self.window_duration) * (plot_right - plot_left)

    def _ripple_channel_geometry(self, channel: int):
        if channel not in self._sorted_channels:
            return None

        rect = self.rect()
        if hasattr(self, "scrollbar"):
            rect.setBottom(rect.bottom() - self.scrollbar.height())

        _, _, plot_bottom = self._get_plot_bounds()
        plot_top = rect.top()
        n_channels = len(self._sorted_channels)
        if n_channels <= 0:
            return None

        channel_height = (plot_bottom - plot_top) / n_channels
        idx = self._sorted_channels.index(channel)
        y_center = plot_bottom - (idx + 0.5) * channel_height + self._channel_offset * channel_height

        return plot_top, plot_bottom, channel_height, y_center

    def _ripple_handle_position(self, event: RippleEvent, side: str):
        """side is 'start', 'end' (drag handles, hit-test only), or
        'peak' (the visible peak-amplitude marker's x position)."""
        geometry = self._ripple_channel_geometry(event.channel)
        if geometry is None:
            return None

        _, _, _, y_center = geometry
        if side == "start":
            sample = event.start_sample
        elif side == "end":
            sample = event.end_sample
        else:  # "peak"
            sample = event.peak_sample
        time = self._ripple_event_time(event, sample)
        return self._ripple_time_to_x(time), y_center

    def _ripple_handle_at(self, pos):
        """Return (event_index, 'start'/'end') for a nearby handle."""
        best = None
        best_dist2 = self._ripple_handle_hit_radius ** 2

        for i, event in enumerate(self.ripple_events):
            if event.channel not in self._sorted_channels:
                continue

            for side in ("start", "end"):
                point = self._ripple_handle_position(event, side)
                if point is None:
                    continue

                dx = pos.x() - point[0]
                dy = pos.y() - point[1]
                dist2 = dx * dx + dy * dy
                if dist2 <= best_dist2:
                    best = (i, side)
                    best_dist2 = dist2

        return best

    def _ripple_x_to_sample(self, x: float, channel: int) -> int:
        """Convert a screen x-coordinate to a sample index relative to
        the analysis window for the given channel's render context.
        Always plain sample/sample_rate math (see _ripple_event_time)."""
        plot_left, plot_right, _ = self._get_plot_bounds()
        ratio = (x - plot_left) / max(1.0, plot_right - plot_left)
        time = self.start_time + ratio * self.window_duration

        ctx = self._ripple_context_for(channel)
        sample_offset = int(ctx.get("sample_offset", 0)) if ctx else 0
        sample_rate = float(ctx.get("sample_rate", self.engine.sr)) if ctx else self.engine.sr

        global_sample = int(round(time * sample_rate))
        return int(global_sample - sample_offset)

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        super().paintEvent(event)

        if not self._sorted_channels:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._ripple_render_context:
            self._draw_ripple_render_context(painter)
        if self.ripple_events:
            self._draw_ripple_events(painter)
        if self._ripple_creating_channel is not None and self._ripple_creating_start_sample is not None:
            self._draw_ripple_creation_preview(painter)

        painter.end()

    def _ripple_context_stats(self, channel: int, ctx: dict, key: str = "envelope"):
        """Return (data_min, data_max, mean, sd) for ctx[key], computed
        ONCE per distinct array and cached on the context dict itself --
        these are whole-signal statistics used only for display
        scaling / threshold lines, which don't depend on the current
        pan/zoom window. Recomputing them on every paintEvent is pure
        waste: a full-recording detection run's envelope can be
        hundreds of thousands of samples, and every pan/zoom/scroll
        triggers a repaint."""
        cache_key = f"_stats_cache_{key}"
        cache = ctx.get(cache_key)
        if cache is not None:
            return cache

        data = ctx.get(key)
        if data is None:
            return None
        data = np.asarray(data, dtype=np.float64)
        if data.size == 0:
            return None

        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return None

        stats = (float(np.min(finite)), float(np.max(finite)), float(np.mean(finite)), float(np.std(finite)))
        ctx[cache_key] = stats
        return stats

    def _draw_ripple_signal_curve(self, painter: QPainter, array: np.ndarray,
                                   sample_offset: int, sample_rate: float,
                                   start_time: float, end_time: float,
                                   plot_left: float, plot_right: float,
                                   plot_top: float, plot_bottom: float,
                                   plot_width: float, max_points: int,
                                   to_y, color: QColor, width: float = 1.0):
        """Shared windowed/downsampled path-drawing routine used by both
        the envelope curve and the filtered-signal curve -- resolves the
        visible sample range directly (no full-length times/mask array),
        downsamples to ~max_points before building the QPainterPath."""
        n_total = len(array)
        first_idx = int(np.ceil((start_time - sample_offset / sample_rate) * sample_rate))
        last_idx = int(np.floor((end_time - sample_offset / sample_rate) * sample_rate))
        first_idx = max(0, first_idx)
        last_idx = min(n_total - 1, last_idx)
        if first_idx > last_idx:
            return

        data_slice = np.asarray(array[first_idx:last_idx + 1], dtype=np.float64)
        n_visible = len(data_slice)

        if n_visible > max_points:
            step = int(np.ceil(n_visible / max_points))
            data_slice = data_slice[::step]
            sample_indices = np.arange(first_idx, last_idx + 1, step, dtype=np.float64)
        else:
            sample_indices = np.arange(first_idx, last_idx + 1, dtype=np.float64)

        times_plot = (sample_offset + sample_indices) / sample_rate
        x = plot_left + ((times_plot - start_time) / (end_time - start_time)) * plot_width
        y = to_y(data_slice)
        y = np.clip(y, plot_top, plot_bottom)
        finite = np.isfinite(data_slice) & np.isfinite(y)
        if not np.any(finite):
            return

        path = QPainterPath()
        started = False
        for xi, yi, ok in zip(x, y, finite):
            if not ok:
                started = False
                continue
            if not started:
                path.moveTo(float(xi), float(yi))
                started = True
            else:
                path.lineTo(float(xi), float(yi))

        painter.save()
        painter.setPen(QPen(color, width))
        painter.drawPath(path)
        painter.restore()

    def _draw_ripple_render_context(self, painter: QPainter):
        """Draw envelope + filtered-signal + threshold lines per
        channel (purely visual, not interactive). Only the samples
        inside the current visible time window are ever touched,
        downsampled to roughly one point per horizontal pixel."""
        rect = self.rect()
        if hasattr(self, "scrollbar"):
            rect.setBottom(rect.bottom() - self.scrollbar.height())
        plot_left, plot_right, plot_bottom = self._get_plot_bounds()
        plot_top = rect.top()
        plot_width = plot_right - plot_left
        plot_height = plot_bottom - plot_top
        if plot_width <= 0 or plot_height <= 0:
            return

        n_channels = len(self._sorted_channels)
        if n_channels <= 0:
            return
        channel_height = plot_height / n_channels

        start_time = float(self.start_time)
        end_time = start_time + float(self.window_duration)
        if end_time <= start_time:
            return

        max_points = max(200, int(plot_width * 2))

        for idx, channel in enumerate(self._sorted_channels):
            ctx = self._ripple_context_for(channel)
            if ctx is None:
                continue

            y_center = plot_bottom - (idx + 0.5) * channel_height + self._channel_offset * channel_height

            envelope = ctx.get("envelope")
            sample_offset = int(ctx.get("sample_offset", 0))
            sample_rate = float(ctx.get("sample_rate", self.engine.sr))
            if sample_rate <= 0:
                continue

            env_stats = self._ripple_context_stats(channel, ctx, key="envelope")

            if self.show_ripple_envelope and envelope is not None and env_stats is not None:
                env_min, env_max, mean_env, sd_env = env_stats
                env_range = env_max - env_min
                if env_range <= 0:
                    env_range = 1.0
                env_scale = (channel_height * 0.30) / env_range

                def env_to_y(v, y_center=y_center, env_min=env_min, env_max=env_max, env_scale=env_scale):
                    return y_center - (v - (env_min + env_max) / 2.0) * env_scale

                self._draw_ripple_signal_curve(
                    painter, np.asarray(envelope, dtype=np.float64), sample_offset, sample_rate,
                    start_time, end_time, plot_left, plot_right, plot_top, plot_bottom,
                    plot_width, max_points, env_to_y, QColor("#ffcc00"), width=1.0,
                )

                if self.show_ripple_thresholds:
                    peak_sd = ctx.get("peak_threshold_sd")
                    boundary_sd = ctx.get("boundary_threshold_sd")
                    if peak_sd is not None or boundary_sd is not None:
                        painter.save()
                        painter.setPen(QPen(QColor("#ff6666"), 1.0, Qt.PenStyle.DashLine))
                        for sd in (peak_sd, boundary_sd):
                            if sd is None:
                                continue
                            y = env_to_y(mean_env + float(sd) * sd_env)
                            if plot_top <= y <= plot_bottom:
                                painter.drawLine(int(plot_left), int(y), int(plot_right), int(y))
                        painter.restore()

            filtered = ctx.get("filtered")
            filt_stats = self._ripple_context_stats(channel, ctx, key="filtered")

            if self.show_ripple_filtered and filtered is not None and filt_stats is not None:
                filt_min, filt_max, _mean_filt, _sd_filt = filt_stats
                filt_extent = max(abs(filt_min), abs(filt_max))
                if filt_extent <= 0:
                    filt_extent = 1.0
                filt_scale = (channel_height * 0.30) / filt_extent

                def filt_to_y(v, y_center=y_center, filt_scale=filt_scale):
                    return y_center - v * filt_scale

                self._draw_ripple_signal_curve(
                    painter, np.asarray(filtered, dtype=np.float64), sample_offset, sample_rate,
                    start_time, end_time, plot_left, plot_right, plot_top, plot_bottom,
                    plot_width, max_points, filt_to_y, QColor("#7fd4ff"), width=1.0,
                )

    def _draw_ripple_creation_preview(self, painter: QPainter):
        """Live preview rectangle shown while dragging out a new ripple
        (armed via arm_ripple_creation). Same visual language as a
        selected event so it's obvious this is about to become a real
        one."""
        geometry = self._ripple_channel_geometry(self._ripple_creating_channel)
        if geometry is None:
            return
        _, _, channel_height, y_center = geometry

        plot_left, plot_right, _ = self._get_plot_bounds()

        start_sample = self._ripple_creating_start_sample
        end_sample = self._ripple_creating_end_sample if self._ripple_creating_end_sample is not None else start_sample
        lo, hi = min(start_sample, end_sample), max(start_sample, end_sample)

        x1 = self._ripple_time_to_x(self._ripple_event_time_for_channel(self._ripple_creating_channel, lo))
        x2 = self._ripple_time_to_x(self._ripple_event_time_for_channel(self._ripple_creating_channel, hi))
        left = max(plot_left, min(x1, x2))
        right = min(plot_right, max(x1, x2))
        if right < plot_left or left > plot_right:
            return

        band_half_h = channel_height * 0.40

        painter.setBrush(QBrush(QColor(70, 200, 255, 55)))
        painter.setPen(QPen(QColor(60, 190, 255, 235), 1.5, Qt.PenStyle.DashLine))
        painter.drawRect(
            QRectF(
                left,
                y_center - band_half_h,
                max(1.0, right - left),
                2 * band_half_h,
            )
        )

    def _ripple_event_time_for_channel(self, channel: int, sample: int) -> float:
        """Same as _ripple_event_time but takes a raw channel id rather
        than a RippleEvent, so the creation-preview can compute times
        for a not-yet-existing event."""
        ctx = self._ripple_context_for(channel)
        sample_offset = int(ctx.get("sample_offset", 0)) if ctx else 0
        sample_rate = float(ctx.get("sample_rate", self.engine.sr)) if ctx else self.engine.sr
        global_sample = sample_offset + int(sample)
        return global_sample / float(sample_rate)

    def _draw_ripple_events(self, painter: QPainter):
        """Draw detected/edited ripple events directly OVER the trace
        lane, so the box is visually anchored to the signal it refers
        to rather than floating in a strip above it.

        Layout per event:
          - A translucent filled rectangle spanning the event's time
            range and the trace lane's own vertical extent, so the
            underlying signal stays visible through it.
          - A bright border around that rectangle (color coded by
            state: default, selected, merge candidate).
          - A single small circle marker at the envelope's peak_sample,
            positioned at the trace's vertical center.
          - Small start/end tick marks, drawn only for the currently
            selected event, as a visible affordance for resizing.

        Everything is drawn AFTER the trace and the envelope (see
        paintEvent), so it's unambiguously on top.
        """
        if not self.show_ripple_events:
            return

        plot_left, plot_right, plot_bottom = self._get_plot_bounds()
        rect = self.rect()
        if hasattr(self, "scrollbar"):
            rect.setBottom(rect.bottom() - self.scrollbar.height())
        plot_top = rect.top()

        n_channels = len(self._sorted_channels)
        if n_channels <= 0:
            return
        channel_height = (plot_bottom - plot_top) / n_channels

        # The trace itself is drawn in _draw_traces with y_values
        # centered at y_center and a half-extent of channel_height*0.40.
        # Use the same extents here so the ripple band exactly overlays
        # the trace's own lane.
        band_half_h = channel_height * 0.40

        for i, event in enumerate(self.ripple_events):
            geometry = self._ripple_channel_geometry(event.channel)
            if geometry is None:
                continue

            _, _, _, y_center = geometry

            start_time = self._ripple_event_time(event, event.start_sample)
            end_time = self._ripple_event_time(event, event.end_sample)

            if end_time < self.start_time or start_time > self.start_time + self.window_duration:
                continue

            x1 = self._ripple_time_to_x(start_time)
            x2 = self._ripple_time_to_x(end_time)

            left = max(plot_left, min(x1, x2))
            right = min(plot_right, max(x1, x2))
            if right < plot_left or left > plot_right:
                continue

            merge_candidate = (
                self._ripple_drag_event is not None
                and (i == self._ripple_drag_event or i in self._ripple_merge_candidates)
            )
            selected = i == self._ripple_selected_event

            if merge_candidate:
                fill = QColor(255, 70, 70, 55)
                border = QColor(255, 60, 60, 235)
                peak_color = QColor(255, 60, 60)
            elif selected:
                fill = QColor(70, 255, 70, 55)
                border = QColor(60, 255, 60, 235)
                peak_color = QColor(60, 255, 60)
            else:
                fill = QColor(255, 100, 100, 35)
                border = QColor(255, 120, 120, 200)
                peak_color = QColor(255, 140, 60)

            band_top = y_center - band_half_h
            band_bottom = y_center + band_half_h

            painter.setBrush(QBrush(fill))
            painter.setPen(QPen(border, 1.5))
            painter.drawRect(QRectF(left, band_top, max(1.0, right - left), band_bottom - band_top))

            if selected and not merge_candidate:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor(255, 255, 255, 220), 2.0))
                painter.drawRect(QRectF(left, band_top, max(1.0, right - left), band_bottom - band_top))

            # Peak marker -- small filled circle, centered on the
            # trace's y_center at the envelope's peak_sample time.
            peak_point = self._ripple_handle_position(event, "peak")
            if peak_point is not None:
                px, _ = peak_point
                r = self._ripple_peak_marker_radius
                painter.setBrush(QBrush(peak_color))
                painter.setPen(QPen(QColor(20, 20, 20), 1.2))
                painter.drawEllipse(QRectF(px - r, y_center - r, 2 * r, 2 * r))

            # Start/end tick marks, drawn only for the selected event.
            # Hit-testing still works for all events (see
            # _ripple_handle_hit_radius); this is just a visible hint
            # for the one currently selected.
            if selected:
                tick_len = max(4.0, band_half_h * 0.25)
                painter.setPen(QPen(QColor(255, 255, 255, 200), 1.5))
                painter.drawLine(int(left), int(y_center - tick_len), int(left), int(y_center + tick_len))
                painter.drawLine(int(right), int(y_center - tick_len), int(right), int(y_center + tick_len))

    # ------------------------------------------------------------------
    # Merge logic
    # ------------------------------------------------------------------

    def _ripple_find_merge_candidates(self, event_index: int, side: str, sample: int) -> set[int]:
        """Directional merge search, identical logic to theta's
        _theta_find_merge_candidates but operating on RippleEvent."""
        if not (0 <= event_index < len(self.ripple_events)):
            return set()

        event = self.ripple_events[event_index]
        candidates: set[int] = set()
        sample = int(sample)

        original_start = int(self._ripple_drag_original_start)
        original_end = int(self._ripple_drag_original_end)
        gap = int(self.ripple_min_merge_gap_samples)

        for i, other in enumerate(self.ripple_events):
            if i == event_index or other.channel != event.channel:
                continue

            if side == "end":
                other_start = int(other.start_sample)
                if other_start >= original_end and other_start - sample <= gap:
                    candidates.add(i)
            else:
                other_end = int(other.end_sample)
                if other_end <= original_start and sample - other_end <= gap:
                    candidates.add(i)

        return candidates

    def _ripple_merge_events(self, indices):
        indices = sorted(set(int(i) for i in indices if 0 <= int(i) < len(self.ripple_events)))
        if len(indices) < 2:
            return

        selected = [self.ripple_events[i] for i in indices]
        channels = {e.channel for e in selected}
        if len(channels) != 1:
            return

        start = min(e.start_sample for e in selected)
        end = max(e.end_sample for e in selected)
        peak_event = max(selected, key=lambda e: e.peak_amplitude)
        trough_event = min(selected, key=lambda e: e.trough_amplitude)

        merged = RippleEvent(
            channel=selected[0].channel,
            start_sample=start,
            end_sample=end,
            peak_sample=peak_event.peak_sample,
            peak_amplitude=float(peak_event.peak_amplitude),
            trough_sample=trough_event.trough_sample,
            trough_amplitude=float(trough_event.trough_amplitude),
            manual=True,
        )

        for i in sorted(indices, reverse=True):
            self.ripple_events.pop(i)
        self.ripple_events.append(merged)
        self.ripple_events.sort(key=lambda e: (e.channel, e.start_sample))

        self._ripple_drag_event = None
        self._ripple_drag_side = None
        self._ripple_merge_candidate = None
        self._ripple_merge_candidates.clear()
        self._ripple_selected_event = min(
            range(len(self.ripple_events)),
            key=lambda i: abs(self.ripple_events[i].start_sample - start) if self.ripple_events[i].channel == merged.channel else 10**18,
        )
        self.rippleEventsChanged.emit(self.ripple_events)
        self.rippleEventSelected.emit(self._ripple_selected_event)
        self.unsetCursor()
        self.update()

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        # ---- Arm-and-drag new-ripple creation ----
        if self._ripple_creating_armed and event.button() == Qt.MouseButton.LeftButton:
            channel = self._theta_channel_at_y(event.position().y())
            if channel is not None:
                sample = self._ripple_x_to_sample(event.position().x(), channel)
                self._ripple_creating_channel = channel
                self._ripple_creating_start_sample = sample
                event.accept()
                return
            self.cancel_ripple_creation()

        # ---- Hit a resize handle ----
        if event.button() == Qt.MouseButton.LeftButton and self.ripple_events:
            handle = self._ripple_handle_at(event.position())
            if handle is not None:
                index, side = handle
                self._ripple_drag_event = index
                self._ripple_drag_side = side
                self._ripple_selected_event = index
                self.rippleEventSelected.emit(index)
                ev_obj = self.ripple_events[index]
                self._ripple_drag_original_start = ev_obj.start_sample
                self._ripple_drag_original_end = ev_obj.end_sample
                self._ripple_merge_candidate = None
                self._ripple_merge_candidates.clear()
                self.setCursor(QCursor(Qt.CursorShape.SizeHorCursor))
                self.update()
                event.accept()
                return

        # ---- Click inside an event to select it ----
        if event.button() == Qt.MouseButton.LeftButton and self.ripple_events:
            selected = self._ripple_event_at(event.position())
            if selected is not None:
                self._ripple_selected_event = selected
                self.rippleEventSelected.emit(selected)
                self.update()
                event.accept()
                return

        super().mousePressEvent(event)

    def _ripple_event_at(self, pos):
        plot_left, plot_right, _ = self._get_plot_bounds()
        for i, event in enumerate(self.ripple_events):
            geometry = self._ripple_channel_geometry(event.channel)
            if geometry is None:
                continue
            _, _, channel_height, y_center = geometry
            x1 = self._ripple_time_to_x(self._ripple_event_time(event, event.start_sample))
            x2 = self._ripple_time_to_x(self._ripple_event_time(event, event.end_sample))
            left = max(plot_left, min(x1, x2))
            right = min(plot_right, max(x1, x2))
            y1 = y_center - channel_height * 0.45
            y2 = y_center + channel_height * 0.45
            if left <= pos.x() <= right and y1 <= pos.y() <= y2:
                return i
        return None

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete and self.delete_selected_ripple_event():
            event.accept()
            return
        super().keyPressEvent(event)

    def mouseMoveEvent(self, event):
        # ---- Live preview while dragging out a new ripple ----
        if self._ripple_creating_channel is not None and self._ripple_creating_start_sample is not None:
            self._ripple_creating_end_sample = self._ripple_x_to_sample(
                event.position().x(), self._ripple_creating_channel
            )
            self.update()
            event.accept()
            return

        if self._ripple_drag_event is not None and self._ripple_drag_side is not None:
            index = self._ripple_drag_event
            if not (0 <= index < len(self.ripple_events)):
                self._ripple_cancel_drag()
                return

            ev_obj = self.ripple_events[index]
            sample = self._ripple_x_to_sample(event.position().x(), ev_obj.channel)

            ctx = self._ripple_context_for(ev_obj.channel)
            sample_rate = float(ctx.get("sample_rate", self.engine.sr)) if ctx else self.engine.sr
            sample_offset = int(ctx.get("sample_offset", 0)) if ctx else 0
            max_global_sample = max(0, int(round(self.engine.total_duration * sample_rate)))
            min_relative = -sample_offset
            max_relative = max_global_sample - sample_offset
            sample = max(min_relative, min(sample, max_relative))

            if self._ripple_drag_side == "start":
                ev_obj.start_sample = min(sample, ev_obj.end_sample - 1)
            else:
                ev_obj.end_sample = max(sample, ev_obj.start_sample + 1)
            ev_obj.manual = True

            self._ripple_merge_candidates = self._ripple_find_merge_candidates(index, self._ripple_drag_side, sample)
            self._ripple_merge_candidate = min(self._ripple_merge_candidates) if self._ripple_merge_candidates else None

            self.update()
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        # ---- Commit a new manual ripple, if we were dragging one out ----
        if self._ripple_creating_channel is not None and self._ripple_creating_start_sample is not None:
            channel = self._ripple_creating_channel
            start_sample = self._ripple_creating_start_sample
            end_sample = self._ripple_creating_end_sample if self._ripple_creating_end_sample is not None else start_sample
            lo, hi = min(start_sample, end_sample), max(start_sample, end_sample)

            if hi - lo >= self._ripple_min_duration_samples:
                new_event = RippleEvent(
                    channel=channel,
                    start_sample=lo,
                    end_sample=hi,
                    peak_sample=(lo + hi) // 2,
                    peak_amplitude=0.0,
                    trough_sample=(lo + hi) // 2,
                    trough_amplitude=0.0,
                    manual=True,
                )
                self.ripple_events.append(new_event)
                self.ripple_events.sort(key=lambda e: (e.channel, e.start_sample))
                self._ripple_selected_event = next(j for j, e in enumerate(self.ripple_events) if e is new_event)
                self.rippleEventsChanged.emit(self.ripple_events)
                self.rippleEventSelected.emit(self._ripple_selected_event)

            self._ripple_creating_armed = False
            self._ripple_creating_channel = None
            self._ripple_creating_start_sample = None
            self._ripple_creating_end_sample = None
            self.unsetCursor()
            self.update()
            event.accept()
            return

        if self._ripple_drag_event is not None:
            index = self._ripple_drag_event
            candidates = set(self._ripple_merge_candidates)
            if candidates and 0 <= index < len(self.ripple_events):
                candidates.add(index)
                self._ripple_merge_events(candidates)
                self.unsetCursor()
                event.accept()
                return

            self._ripple_cancel_drag()
            self.rippleEventsChanged.emit(self.ripple_events)
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        """Double-click INSIDE the currently-selected ripple's band
        splits it at the click position. Double-click anywhere else
        falls through to the base class's behavior."""
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._ripple_selected_event is not None
            and 0 <= self._ripple_selected_event < len(self.ripple_events)
        ):
            i = self._ripple_selected_event
            event_obj = self.ripple_events[i]
            geometry = self._ripple_channel_geometry(event_obj.channel)
            if geometry is not None:
                _, _, channel_height, y_center = geometry
                plot_left, plot_right, _ = self._get_plot_bounds()
                x1 = self._ripple_time_to_x(self._ripple_event_time(event_obj, event_obj.start_sample))
                x2 = self._ripple_time_to_x(self._ripple_event_time(event_obj, event_obj.end_sample))
                left = max(plot_left, min(x1, x2))
                right = min(plot_right, max(x1, x2))
                band_half_h = channel_height * 0.40
                y1 = y_center - band_half_h
                y2 = y_center + band_half_h

                pos = event.position()
                if left <= pos.x() <= right and y1 <= pos.y() <= y2:
                    split_sample = self._ripple_x_to_sample(pos.x(), event_obj.channel)
                    if self.split_selected_ripple_at(split_sample):
                        event.accept()
                        return

        super().mouseDoubleClickEvent(event)

    def _ripple_cancel_drag(self):
        self._ripple_drag_event = None
        self._ripple_drag_side = None
        self._ripple_merge_candidate = None
        self._ripple_merge_candidates.clear()
        self.unsetCursor()
        self.update()

    def leaveEvent(self, event):
        if self._ripple_drag_event is None and not self._ripple_creating_armed:
            self.unsetCursor()
        super().leaveEvent(event)
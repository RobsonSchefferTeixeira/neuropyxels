"""
theta_cycle_trace_view.py

Trace view extension that draws theta cycles detected by
core/theta_cycle_detector.py as an overlay: the theta-band filtered
signal in soft orange, with one small circle per landmark sample
(6 per cycle) sitting on that trace. Selected cycles turn green.

Interaction
-----------
  * Click a landmark circle -> select that cycle (deselects others).
  * Click empty space      -> deselect.
  * Delete key             -> delete the selected cycle.
  * Ctrl+Z                 -> undo the last cycle deletion.

Unlike theta epochs and ripples, cycles have no drag-to-resize,
split, merge, or manual creation -- a cycle is defined by its six
landmark samples, and editing is limited to delete + undo.

Class chain
-----------
Inherits from ThetaEpochTraceViewWidget so the paint and event chains
unwind in the right order:

    NeuralTraceViewWidget
      -> RippleTraceViewWidget
        -> ThetaCycleTraceViewWidget   (this class)
          -> ThetaEpochTraceViewWidget
            -> TraceViewWidget
              -> QWidget

Each widget's paintEvent calls super().paintEvent() first, so layers
draw bottom-up: traces, then epochs, then cycles, then ripples.
"""

from __future__ import annotations

import copy

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, QRectF
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor

from core.filters import bandpass_filter
from core.theta_cycle_detector import ThetaCycle
from gui.theta_epoch_trace_view import ThetaEpochTraceViewWidget


# Soft orange, distinct from the ripple overlay's cyan-blue filtered
# trace and its yellow envelope.
CYCLE_TRACE_COLOR = QColor("#ffa94d")
CYCLE_SELECTED_COLOR = QColor("#42d77d")
CYCLE_MARKER_BORDER = QColor("#1e1e1e")


class ThetaCycleTraceViewWidget(ThetaEpochTraceViewWidget):
    """Trace view with an interactive theta-cycle overlay."""

    thetaCyclesChanged = pyqtSignal(object)   # list[ThetaCycle]
    thetaCycleSelected = pyqtSignal(int)      # index into the flat list

    def __init__(self, engine, parent=None):
        super().__init__(engine, parent)

        self.theta_cycles: list[ThetaCycle] = []
        self.show_theta_cycles = True
        self.theta_cycle_trace_width = 1.0

        # The band used to compute the overlay trace. Kept in sync with
        # the dialog's parameters via set_theta_cycle_band(). Defaults
        # match ThetaCycleParams's defaults.
        self._theta_cycle_low_freq = 4.0
        self._theta_cycle_high_freq = 12.0

        # Cached filtered trace per channel, invalidated on window
        # change or band change. Filtering at 30 kHz over a 10-second
        # window is not free; caching makes repaints cheap.
        self._cycle_trace_cache: dict[int, np.ndarray] = {}
        self._cycle_trace_cache_key = None

        self._theta_cycle_selected: int | None = None
        self._cycle_marker_radius = 4.0
        self._cycle_marker_hit_radius = 8.0

        # Undo stack for deletions.
        self._theta_cycle_undo_stack: list[list[ThetaCycle]] = []
        self._theta_cycle_undo_max = 50

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_theta_cycles(self, cycles: list[ThetaCycle] | None):
        self.theta_cycles = list(cycles or [])
        self._theta_cycle_selected = None
        self._invalidate_cycle_trace_cache()
        self.update()

    def clear_theta_cycles(self):
        self.theta_cycles = []
        self._theta_cycle_selected = None
        self._theta_cycle_undo_stack.clear()
        self._invalidate_cycle_trace_cache()
        self.update()

    def set_theta_cycle_band(self, low_freq: float, high_freq: float):
        """Called by the dialog whenever its theta band changes, so the
        overlay trace matches the band the cycles were detected with."""
        self._theta_cycle_low_freq = float(low_freq)
        self._theta_cycle_high_freq = float(high_freq)
        self._invalidate_cycle_trace_cache()
        self.update()

    def set_theta_cycle_selected(self, index: int):
        if 0 <= int(index) < len(self.theta_cycles):
            self._theta_cycle_selected = int(index)
            self.update()

    def set_theta_cycle_trace_width(self, width: float):
        self.theta_cycle_trace_width = max(0.5, min(float(width), 5.0))
        self.update()

    def set_theta_cycles_visible(self, visible: bool):
        self.show_theta_cycles = bool(visible)
        self.update()

    def delete_selected_theta_cycle(self) -> bool:
        i = self._theta_cycle_selected
        if i is None or not (0 <= i < len(self.theta_cycles)):
            return False
        self._cycle_push_undo()
        self.theta_cycles.pop(i)
        self._theta_cycle_selected = None
        self.thetaCyclesChanged.emit(self.theta_cycles)
        self.update()
        return True

    # ------------------------------------------------------------------
    # Undo
    # ------------------------------------------------------------------

    def _cycle_push_undo(self):
        snapshot = [copy.deepcopy(c) for c in self.theta_cycles]
        self._theta_cycle_undo_stack.append(snapshot)
        if len(self._theta_cycle_undo_stack) > self._theta_cycle_undo_max:
            self._theta_cycle_undo_stack.pop(0)

    def _cycle_undo(self) -> bool:
        if not self._theta_cycle_undo_stack:
            return False
        self.theta_cycles = self._theta_cycle_undo_stack.pop()
        self._theta_cycle_selected = None
        self.thetaCyclesChanged.emit(self.theta_cycles)
        self.update()
        return True

    # ------------------------------------------------------------------
    # Filtered-trace cache
    # ------------------------------------------------------------------

    def _invalidate_cycle_trace_cache(self):
        self._cycle_trace_cache.clear()
        self._cycle_trace_cache_key = None

    def _current_cycle_trace_cache_key(self):
        return (
            round(float(self.start_time), 6),
            round(float(self.window_duration), 6),
            round(float(self._theta_cycle_low_freq), 3),
            round(float(self._theta_cycle_high_freq), 3),
            float(self.engine.sr),
        )

    def _get_cycle_filtered_trace(self, channel: int) -> np.ndarray | None:
        """Theta-band filtered signal for `channel` over the current
        visible window, or None if the channel has no cycles."""
        if channel not in self.channels:
            return None
        # Only channels that actually have cycles get a filtered trace
        # computed -- filtering every channel would be wasteful.
        if not any(c.channel == channel for c in self.theta_cycles):
            return None

        key = self._current_cycle_trace_cache_key()
        if key != self._cycle_trace_cache_key:
            self._invalidate_cycle_trace_cache()
            self._cycle_trace_cache_key = key

        cached = self._cycle_trace_cache.get(channel)
        if cached is not None:
            return cached

        end_time = self.start_time + self.window_duration
        try:
            raw = self.engine.get_channel_data(channel, self.start_time, end_time)
        except Exception:
            return None
        if raw is None or len(raw) == 0:
            return None

        try:
            filtered = bandpass_filter(
                np.asarray(raw, dtype=np.float64),
                float(self.engine.sr),
                self._theta_cycle_low_freq,
                self._theta_cycle_high_freq,
                order=4,
            )
        except Exception:
            return None

        self._cycle_trace_cache[channel] = filtered
        return filtered

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _cycle_channel_geometry(self, channel: int):
        """Returns (plot_top, plot_bottom, channel_height, y_center) for
        the channel's lane, or None if the channel isn't visible."""
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
        y_center = (plot_bottom
                    - (idx + 0.5) * channel_height
                    + self._channel_offset * channel_height)
        return plot_top, plot_bottom, channel_height, y_center

    def _sample_to_x(self, sample: int) -> float:
        """Convert a raw sample index to a screen x coordinate, using
        the same time base the trace view uses everywhere."""
        time = float(sample) / float(self.engine.sr)
        plot_left, plot_right, _ = self._get_plot_bounds()
        if self.window_duration <= 0:
            return plot_left
        return plot_left + (
            (time - self.start_time) / self.window_duration
        ) * (plot_right - plot_left)

    def _x_to_sample(self, x: float) -> int:
        plot_left, plot_right, _ = self._get_plot_bounds()
        ratio = (x - plot_left) / max(1.0, plot_right - plot_left)
        time = self.start_time + ratio * self.window_duration
        return int(round(time * float(self.engine.sr)))

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        super().paintEvent(event)

        if not self.show_theta_cycles or not self.theta_cycles:
            return
        if not self._sorted_channels:
            return

        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            self._draw_theta_cycles(painter)
        finally:
            painter.end()

    def _draw_theta_cycles(self, painter: QPainter):
        plot_left, plot_right, _ = self._get_plot_bounds()
        rect = self.rect()
        if hasattr(self, "scrollbar"):
            rect.setBottom(rect.bottom() - self.scrollbar.height())
        plot_top = rect.top()
        plot_bottom = self._get_plot_bounds()[2]

        # Group cycles by channel so the filtered trace is fetched and
        # drawn once per channel, not once per cycle.
        by_channel: dict[int, list[tuple[int, ThetaCycle]]] = {}
        for i, c in enumerate(self.theta_cycles):
            by_channel.setdefault(c.channel, []).append((i, c))

        for channel, indexed_cycles in by_channel.items():
            geom = self._cycle_channel_geometry(channel)
            if geom is None:
                continue
            _, _, channel_height, y_center = geom

            filtered = self._get_cycle_filtered_trace(channel)
            if filtered is None or filtered.size == 0:
                continue

            # Scale the filtered trace to fit the channel's lane, same
            # way the ripple overlay scales its filtered signal.
            finite = np.isfinite(filtered)
            if not np.any(finite):
                continue
            peak = float(np.max(np.abs(filtered[finite])))
            if peak <= 0:
                peak = 1.0
            scale = (channel_height * 0.40) / peak

            # x-pixel positions for every sample in the window.
            n = filtered.size
            start_idx, end_idx = self.engine.get_time_window_sample_range(
                self.start_time, self.start_time + self.window_duration
            )
            # Guard against a length mismatch (e.g. the cache was built
            # for a slightly different window); if they differ, fall
            # back to a uniform linspace over the visible samples.
            if n == (end_idx - start_idx) and n > 1:
                sample_indices = np.arange(start_idx, end_idx)
            else:
                sample_indices = np.linspace(
                    start_idx, max(start_idx, end_idx - 1), n
                )

            plot_width = plot_right - plot_left
            if plot_width <= 0 or n < 2:
                continue

            times = sample_indices / float(self.engine.sr)
            x = (plot_left
                 + ((times - self.start_time) / self.window_duration)
                 * plot_width)
            y = y_center - filtered * scale

            # Clip y to the channel's vertical extent so a filtering
            # transient at the edge of the window doesn't bleed into
            # neighbouring lanes.
            y = np.clip(y, plot_top, plot_bottom)

            painter.save()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            pen = QPen(CYCLE_TRACE_COLOR, self.theta_cycle_trace_width)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            from PyQt6.QtGui import QPainterPath
            path = QPainterPath()
            finite_mask = np.isfinite(y)
            started = False
            for xi, yi, ok in zip(x, y, finite_mask):
                if not ok:
                    started = False
                    continue
                if not started:
                    path.moveTo(float(xi), float(yi))
                    started = True
                else:
                    path.lineTo(float(xi), float(yi))
            painter.drawPath(path)


            # Draw a green highlight over the selected cycle's span, on top
            # of the full filtered trace. This is the visual equivalent of the
            # epoch overlay's "selected epoch" outline -- a local highlight,
            # not a global recolor, so the rest of the trace stays soft orange
            # and the user's eye goes straight to the highlighted cycle.
            if self._theta_cycle_selected is not None:
                sel = self.theta_cycles[self._theta_cycle_selected]
                if sel.channel == channel:
                    sel_lo = max(int(sel.zero_crossing_start), int(start_idx))
                    sel_hi = min(int(sel.peak2), int(end_idx))
                    if sel_hi > sel_lo:
                        # Slice the filtered trace and the x pixel positions
                        # over exactly the selected cycle's span.
                        lo_local = sel_lo - int(start_idx)
                        hi_local = sel_hi - int(start_idx)
                        if 0 <= lo_local < hi_local <= filtered.size:
                            xs_sel = x[lo_local:hi_local]
                            ys_sel = y[lo_local:hi_local]
                            finite_sel = finite_mask[lo_local:hi_local]

                            painter.save()
                            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                            hi_pen = QPen(CYCLE_SELECTED_COLOR,
                                        self.theta_cycle_trace_width + 1.0)
                            hi_pen.setCosmetic(True)
                            painter.setPen(hi_pen)
                            painter.setBrush(Qt.BrushStyle.NoBrush)

                            sel_path = QPainterPath()
                            started_sel = False
                            for xi, yi, ok in zip(xs_sel, ys_sel, finite_sel):
                                if not ok:
                                    started_sel = False
                                    continue
                                if not started_sel:
                                    sel_path.moveTo(float(xi), float(yi))
                                    started_sel = True
                                else:
                                    sel_path.lineTo(float(xi), float(yi))
                            painter.drawPath(sel_path)
                            painter.restore()




            # Now the landmark circles. For each cycle, draw 6 circles
            # at the (x, y) that the filtered trace passes through at
            # each landmark sample.
            #
            # Cycles are drawn unselected-first, selected-last, so a
            # selected cycle's markers land on top of any markers it
            # shares with an adjacent cycle (zc_end/peak2 of cycle N
            # coincide with zc_start/peak1 of cycle N+1). Without this,
            # the seam markers would be drawn in the following cycle's
            # color even while the earlier cycle is selected, and the
            # selected cycle would appear to be missing its last two
            # markers.
            sorted_cycles = sorted(
                indexed_cycles,
                key=lambda pair: (pair[0] == self._theta_cycle_selected),
            )
            for global_idx, cycle in sorted_cycles:
                selected = (global_idx == self._theta_cycle_selected)
                color = CYCLE_SELECTED_COLOR if selected else CYCLE_TRACE_COLOR
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(CYCLE_MARKER_BORDER, 1.0))
                r = self._cycle_marker_radius


                # All 6 landmarks are drawn for every cycle, even though
                # zc_end/peak2 of cycle N coincide with zc_start/peak1
                # of cycle N+1. The overlap is intentional: when the
                # user selects a cycle, all 6 of its markers turn green,
                # including the two that sit on top of the next cycle's
                # opening markers. The click-resolution rule (see
                # _cycle_marker_at) decides which cycle a click selects
                # at those overlaps -- always the later one, so the
                # highlighted 6 stay consistent with the selected cycle.
                for sample in (
                    cycle.zero_crossing_start,
                    cycle.peak1,
                    cycle.zero_crossing_mid,
                    cycle.valley,
                    cycle.zero_crossing_end,
                    cycle.peak2,
                ):
                    if sample < start_idx or sample >= end_idx:
                        continue

                    try:
                        local = int(sample) - int(start_idx)
                        if local < 0 or local >= filtered.size:
                            continue
                        value = float(filtered[local])
                    except Exception:
                        continue
                    if not np.isfinite(value):
                        continue
                    xi = self._sample_to_x(int(sample))
                    yi = y_center - value * scale
                    yi = max(plot_top, min(plot_bottom, yi))
                    if xi < plot_left - 5 or xi > plot_right + 5:
                        continue
                    painter.drawEllipse(
                        QRectF(xi - r, yi - r, 2 * r, 2 * r)
                    )

            painter.restore()

    # ------------------------------------------------------------------
    # Mouse / key handling
    # ------------------------------------------------------------------
    
    def mousePressEvent(self, event):
        if (
            self.show_theta_cycles
            and event.button() == Qt.MouseButton.LeftButton
            and self.theta_cycles
        ):
            hit = self._cycle_marker_at(event.position())
            if hit is not None:
                self._theta_cycle_selected = hit
                self.thetaCycleSelected.emit(hit)
                self.update()
                event.accept()
                return
            # Click on empty space within the trace view deselects, but
            # only if a cycle was selected -- otherwise let the base
            # class handle the click (pan, cursor placement, etc.).
            if self._theta_cycle_selected is not None:
                self._theta_cycle_selected = None
                self.update()
                event.accept()
                return

        super().mousePressEvent(event)

    def _cycle_marker_at(self, pos) -> int | None:
        """Return the index of the cycle whose nearest landmark circle
        contains `pos`, or None.

        Iterates cycles in REVERSE order so that, when a marker position
        is shared between cycle N (as zc_end/peak2) and cycle N+1 (as
        zc_start/peak1), the LATER cycle wins. This is the "never go
        backwards" selection rule: clicking a shared marker selects the
        cycle that starts there, not the one that ends there.
        """
        hit_radius2 = self._cycle_marker_hit_radius ** 2

        for i in range(len(self.theta_cycles) - 1, -1, -1):
            cycle = self.theta_cycles[i]
            geom = self._cycle_channel_geometry(cycle.channel)
            if geom is None:
                continue
            _, _, channel_height, y_center = geom

            filtered = self._get_cycle_filtered_trace(cycle.channel)
            if filtered is None or filtered.size == 0:
                continue
            finite = np.isfinite(filtered)
            if not np.any(finite):
                continue
            peak = float(np.max(np.abs(filtered[finite]))) or 1.0
            scale = (channel_height * 0.40) / peak

            start_idx, end_idx = self.engine.get_time_window_sample_range(
                self.start_time, self.start_time + self.window_duration
            )

            for sample in (
                cycle.zero_crossing_start,
                cycle.peak1,
                cycle.zero_crossing_mid,
                cycle.valley,
                cycle.zero_crossing_end,
                cycle.peak2,
            ):
                if sample < start_idx or sample >= end_idx:
                    continue
                local = int(sample) - int(start_idx)
                if local < 0 or local >= filtered.size:
                    continue
                value = float(filtered[local])
                if not np.isfinite(value):
                    continue
                xi = self._sample_to_x(int(sample))
                yi = y_center - value * scale

                dx = pos.x() - xi
                dy = pos.y() - yi
                if dx * dx + dy * dy <= hit_radius2:
                    return i

        return None

    def keyPressEvent(self, event):
        # Ctrl+Z: undo the last cycle deletion. Handled here (before
        # the epoch / ripple handlers above in the MRO can claim it) so
        # Ctrl+Z picks the most-recently-touched overlay. If this
        # overlay has nothing to undo, fall through so the epoch and
        # ripple handlers get a chance.
        if (
            event.key() == Qt.Key.Key_Z
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            if self._cycle_undo():
                event.accept()
                return

        if (event.key() == Qt.Key.Key_Delete
                and self.delete_selected_theta_cycle()):
            event.accept()
            return

        super().keyPressEvent(event)

    # Invalidate cached trace whenever the view scrolls/zooms. The
    # parent's `_invalidate_cache` is the right hook because it's called
    # on every window change; we just piggyback.
    def _invalidate_cache(self):
        super()._invalidate_cache()
        self._invalidate_cycle_trace_cache()
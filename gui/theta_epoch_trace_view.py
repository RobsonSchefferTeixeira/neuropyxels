"""
theta_epoch_trace_view.py

TraceViewWidget extension that draws and edits theta epochs directly on the
main trace view. The original trace/spectrogram implementation is inherited
unchanged.
"""

from __future__ import annotations

import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal, QRectF
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QCursor

from core.theta_epoch_detector import ThetaEpoch
from gui.trace_view import TraceViewWidget


class ThetaEpochTraceViewWidget(TraceViewWidget):
    """TraceViewWidget with an interactive theta-epoch overlay."""

    thetaEpochsChanged = pyqtSignal(object)  # list[ThetaEpoch]
    thetaEpochSelected = pyqtSignal(int)

    def __init__(self, engine, parent=None):
        super().__init__(engine, parent)

        self.theta_epochs: list[ThetaEpoch] = []
        self.theta_detection_start_time = 0.0
        self.theta_detection_sample_offset = 0

        self._theta_drag_epoch: int | None = None
        self._theta_drag_side: str | None = None
        self._theta_merge_candidate: int | None = None
        self._theta_merge_candidates: set[int] = set()
        self._theta_selected_epoch: int | None = None
        self.theta_min_merge_gap_ms = 500.0
        self.theta_min_merge_gap_samples = int(round(0.5 * self.engine.sr))
        self._theta_drag_original_start = 0
        self._theta_drag_original_end = 0

        self._theta_handle_radius = 7.0
        self._theta_handle_hit_radius = 12.0

        # "Add new window" (click-drag creation) state. Armed via
        # arm_theta_creation() (called by ThetaEpochDialog's "Add Theta
        # Epoch" button); the NEXT left-button press/drag on a channel
        # lane creates a new manual epoch there instead of the usual
        # pan/select behavior, then disarms itself automatically.
        self._theta_creating_armed = False
        self._theta_creating_channel: int | None = None
        self._theta_creating_start_sample: int | None = None
        self._theta_creating_end_sample: int | None = None
        # Minimum epoch width, in samples, below which a click-drag
        # creation (or a split) is discarded as accidental/too-short
        # rather than producing a degenerate zero/near-zero-width
        # epoch. 5ms is short enough to never reject an intentional
        # drag, long enough to filter out an accidental single click.
        self._theta_min_epoch_samples = max(1, int(round(0.005 * self.engine.sr)))

    # ------------------------------------------------------------------
    # Theta epoch API
    # ------------------------------------------------------------------

    def set_theta_epochs(
        self,
        epochs: list[ThetaEpoch] | None,
        detection_start_time: float = 0.0,
        detection_sample_offset: int | None = None,
        min_merge_gap_ms: float | None = None,
    ):
        self.theta_epochs = list(epochs or [])
        self.theta_detection_start_time = float(detection_start_time)
        if detection_sample_offset is None:
            detection_sample_offset = int(
                round(self.theta_detection_start_time * self.engine.sr)
            )
        self.theta_detection_sample_offset = int(detection_sample_offset)
        if min_merge_gap_ms is not None:
            self.set_theta_merge_gap(min_merge_gap_ms, update=False)
        self._theta_drag_epoch = None
        self._theta_drag_side = None
        self._theta_merge_candidate = None
        self._theta_merge_candidates.clear()
        self.update()

    def set_theta_merge_gap(self, gap_ms: float, update: bool = True):
        self.theta_min_merge_gap_ms = max(0.0, float(gap_ms))
        self.theta_min_merge_gap_samples = int(
            round(self.theta_min_merge_gap_ms / 1000.0 * self.engine.sr)
        )
        if update:
            self.update()

    def set_theta_selected_epoch(self, index: int):
        """Select an epoch by its current table/overlay index."""
        if 0 <= int(index) < len(self.theta_epochs):
            self._theta_selected_epoch = int(index)
            self.update()

    def delete_selected_theta_epoch(self):
        """Delete the currently selected theta epoch."""
        i = self._theta_selected_epoch
        if i is None or not (0 <= i < len(self.theta_epochs)):
            return False
        self.theta_epochs.pop(i)
        self._theta_selected_epoch = None
        self._theta_drag_epoch = None
        self._theta_drag_side = None
        self._theta_merge_candidate = None
        self._theta_merge_candidates.clear()
        self.thetaEpochsChanged.emit(self.theta_epochs)
        self.update()
        return True

    def arm_theta_creation(self):
        """
        Arm click-drag creation of a new manual theta epoch: the NEXT
        left-button press+drag on a channel lane creates a new epoch
        there (start = press position, end = release position),
        instead of the usual pan/select behavior. Disarms itself
        automatically once the drag completes (or is cancelled by
        releasing without having moved past _theta_min_epoch_samples).

        Called by ThetaEpochDialog's "Add Theta Epoch" button -- the
        button-then-drag flow (rather than inserting a fixed-duration
        epoch immediately) mirrors how resizing an existing epoch's
        boundary already works, so creating one uses the same gesture
        instead of introducing a second, different interaction model.
        """
        self._theta_creating_armed = True
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))

    def cancel_theta_creation(self):
        """Disarm creation mode without creating anything -- e.g. if
        the dialog's button is clicked again to toggle it off, or the
        dialog closes while armed."""
        self._theta_creating_armed = False
        self._theta_creating_channel = None
        self._theta_creating_start_sample = None
        self.unsetCursor()

    def _theta_channel_at_y(self, y: float) -> int | None:
        """Reverse of _theta_channel_geometry: given a y-pixel
        position, return which channel's lane it falls in (or None if
        outside the plot area / no channels)."""
        rect = self.rect()
        if hasattr(self, "scrollbar"):
            rect.setBottom(rect.bottom() - self.scrollbar.height())

        _, _, plot_bottom = self._get_plot_bounds()
        plot_top = rect.top()
        n_channels = len(self._sorted_channels)
        if n_channels <= 0:
            return None

        channel_height = (plot_bottom - plot_top) / n_channels
        # Inverse of the y_center formula in _theta_channel_geometry:
        #   y_center = plot_bottom - (idx + 0.5) * channel_height + channel_offset * channel_height
        # Solve for idx given y.
        idx_f = (plot_bottom - y) / channel_height - 0.5 + self._channel_offset
        idx = int(round(idx_f))
        if 0 <= idx < n_channels:
            return self._sorted_channels[idx]
        return None

    def _theta_split_epoch(self, index: int, split_sample: int):
        """
        Split theta_epochs[index] into two epochs at split_sample (a
        sample position relative to the SAME detection-window basis as
        epoch.start_sample/end_sample). Both halves are marked
        manual=True and duplicate the original epoch's aggregate stats
        (mean_ratio, peak_ratio, mean_theta_power, mean_delta_power)
        unchanged -- recomputing them properly would require re-running
        detection math on each half's raw signal, which this dialog
        doesn't have access to after the fact; duplicating is the same
        pragmatic choice _theta_merge_epochs already makes by averaging
        rather than re-detecting.

        No-ops (returns False) if split_sample doesn't leave both
        halves at least _theta_min_epoch_samples wide, so a double-click
        very close to either edge can't produce a degenerate sliver.
        """
        if not (0 <= index < len(self.theta_epochs)):
            return False
        epoch = self.theta_epochs[index]
        min_w = self._theta_min_epoch_samples

        if split_sample - epoch.start_sample < min_w or epoch.end_sample - split_sample < min_w:
            return False

        first = ThetaEpoch(
            channel=epoch.channel,
            start_sample=epoch.start_sample,
            end_sample=split_sample,
            peak_sample=min(epoch.peak_sample, split_sample),
            mean_theta_power=epoch.mean_theta_power,
            mean_delta_power=epoch.mean_delta_power,
            mean_ratio=epoch.mean_ratio,
            peak_ratio=epoch.peak_ratio,
            duration_ms=(split_sample - epoch.start_sample) / float(self.engine.sr) * 1000.0,
            manual=True,
        )
        second = ThetaEpoch(
            channel=epoch.channel,
            start_sample=split_sample,
            end_sample=epoch.end_sample,
            peak_sample=max(epoch.peak_sample, split_sample),
            mean_theta_power=epoch.mean_theta_power,
            mean_delta_power=epoch.mean_delta_power,
            mean_ratio=epoch.mean_ratio,
            peak_ratio=epoch.peak_ratio,
            duration_ms=(epoch.end_sample - split_sample) / float(self.engine.sr) * 1000.0,
            manual=True,
        )

        self.theta_epochs.pop(index)
        self.theta_epochs.append(first)
        self.theta_epochs.append(second)
        self.theta_epochs.sort(key=lambda e: (e.channel, e.start_sample))

        # Select the first half (whichever new index it landed at after
        # sorting), mirroring _theta_merge_epochs's own post-op
        # selection pattern.
        self._theta_selected_epoch = next(
            i for i, e in enumerate(self.theta_epochs)
            if e is first
        )
        self._theta_drag_epoch = None
        self._theta_drag_side = None
        self._theta_merge_candidate = None
        self._theta_merge_candidates.clear()
        self.thetaEpochsChanged.emit(self.theta_epochs)
        self.thetaEpochSelected.emit(self._theta_selected_epoch)
        self.update()
        return True

    def clear_theta_epochs(self):
        self.theta_epochs.clear()
        self._theta_drag_epoch = None
        self._theta_drag_side = None
        self._theta_merge_candidate = None
        self._theta_merge_candidates.clear()
        self._theta_selected_epoch = None
        self.update()

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _theta_epoch_time(self, sample: int) -> float:
        """Convert a relative detection-window sample to recording time.

        Always pure sample/sr math -- never engine.timestamps -- so
        epoch identity and on-screen position are identical whether or
        not the user has loaded a timestamps.npy file.
        """
        return self.theta_detection_start_time + int(sample) / float(self.engine.sr)

    def _theta_time_to_x(self, time: float) -> float:
        plot_left, plot_right, _ = self._get_plot_bounds()
        if self.window_duration <= 0:
            return plot_left
        return plot_left + (
            (time - self.start_time) / self.window_duration
        ) * (plot_right - plot_left)

    def _theta_channel_geometry(self, channel: int):
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
        y_center = (
            plot_bottom
            - (idx + 0.5) * channel_height
            + self._channel_offset * channel_height
        )

        return plot_top, plot_bottom, channel_height, y_center

    def _theta_handle_position(self, epoch: ThetaEpoch, side: str):
        geometry = self._theta_channel_geometry(epoch.channel)
        if geometry is None:
            return None

        _, _, _, y_center = geometry
        sample = epoch.start_sample if side == "start" else epoch.end_sample
        time = self._theta_epoch_time(sample)
        return self._theta_time_to_x(time), y_center

    def _theta_handle_at(self, pos):
        """Return (epoch_index, 'start'/'end') for a nearby handle."""
        best = None
        best_dist2 = self._theta_handle_hit_radius ** 2

        for i, epoch in enumerate(self.theta_epochs):
            if epoch.channel not in self._sorted_channels:
                continue

            for side in ("start", "end"):
                point = self._theta_handle_position(epoch, side)
                if point is None:
                    continue

                dx = pos.x() - point[0]
                dy = pos.y() - point[1]
                dist2 = dx * dx + dy * dy
                if dist2 <= best_dist2:
                    best = (i, side)
                    best_dist2 = dist2

        return best

    def _theta_x_to_sample(self, x: float) -> int:
        """Convert a screen x to a sample index relative to the
        detection window, always via plain sample/sr math (see
        _theta_epoch_time)."""
        plot_left, plot_right, _ = self._get_plot_bounds()
        ratio = (x - plot_left) / max(1.0, plot_right - plot_left)
        time = self.start_time + ratio * self.window_duration

        global_sample = int(round(time * self.engine.sr))
        return int(global_sample - self.theta_detection_sample_offset)

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        super().paintEvent(event)

        if not self._sorted_channels:
            return

        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            if self.theta_epochs:
                self._draw_theta_epochs(painter)
            if self._theta_creating_channel is not None and self._theta_creating_start_sample is not None:
                self._draw_theta_creation_preview(painter)
        finally:
            painter.end()

    def _draw_theta_creation_preview(self, painter: QPainter):
        """Live rectangle shown while dragging out a new epoch (armed
        via arm_theta_creation(), see mousePressEvent/mouseMoveEvent).
        Same visual language as a selected epoch, so it's obvious this
        is "about to become" a real epoch rather than an existing one."""
        geometry = self._theta_channel_geometry(self._theta_creating_channel)
        if geometry is None:
            return
        _, _, channel_height, y_center = geometry
        plot_left, plot_right, _ = self._get_plot_bounds()

        start_sample = self._theta_creating_start_sample
        end_sample = self._theta_creating_end_sample if self._theta_creating_end_sample is not None else start_sample
        lo, hi = min(start_sample, end_sample), max(start_sample, end_sample)

        x1 = self._theta_time_to_x(self._theta_epoch_time(lo))
        x2 = self._theta_time_to_x(self._theta_epoch_time(hi))
        left = max(plot_left, min(x1, x2))
        right = min(plot_right, max(x1, x2))
        if right < plot_left or left > plot_right:
            return

        painter.setBrush(QBrush(QColor(70, 200, 255, 55)))
        painter.setPen(QPen(QColor(60, 190, 255, 235), 1.5, Qt.PenStyle.DashLine))
        painter.drawRect(
            QRectF(
                left,
                y_center - channel_height * 0.43,
                max(1.0, right - left),
                channel_height * 0.86,
            )
        )

    def _draw_theta_epochs(self, painter: QPainter):
        _, _, plot_bottom = self._get_plot_bounds()
        rect = self.rect()
        if hasattr(self, "scrollbar"):
            rect.setBottom(rect.bottom() - self.scrollbar.height())

        plot_top = rect.top()
        plot_left, plot_right, _ = self._get_plot_bounds()

        for i, epoch in enumerate(self.theta_epochs):
            geometry = self._theta_channel_geometry(epoch.channel)
            if geometry is None:
                continue

            _, _, channel_height, y_center = geometry

            start_time = self._theta_epoch_time(epoch.start_sample)
            end_time = self._theta_epoch_time(epoch.end_sample)

            if end_time < self.start_time or start_time > self.start_time + self.window_duration:
                continue

            x1 = self._theta_time_to_x(start_time)
            x2 = self._theta_time_to_x(end_time)

            left = max(plot_left, min(x1, x2))
            right = min(plot_right, max(x1, x2))
            if right < plot_left or left > plot_right:
                continue

            merge_candidate = (
                self._theta_drag_epoch is not None
                and (i == self._theta_drag_epoch or i in self._theta_merge_candidates)
            )
            selected = i == self._theta_selected_epoch

            # Normal = yellow, selected = green, merge candidate pair = red.
            if merge_candidate:
                fill = QColor(255, 70, 70, 55)
                border = QColor(255, 60, 60, 235)
            elif selected:
                fill = QColor(70, 255, 70, 55)
                border = QColor(60, 255, 60, 235)
            else:
                fill = QColor(255, 220, 80, 38)
                border = QColor(255, 220, 80, 190)

            painter.setBrush(QBrush(fill))
            painter.setPen(QPen(border, 1.5))
            painter.drawRect(
                QRectF(
                    left,
                    y_center - channel_height * 0.43,
                    max(1.0, right - left),
                    channel_height * 0.86,
                )
            )
            if selected and not merge_candidate:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor(255, 255, 255, 220), 2.0))
                painter.drawRect(
                    QRectF(
                        left,
                        y_center - channel_height * 0.43,
                        max(1.0, right - left),
                        channel_height * 0.86,
                    )
                )

            # Both circles become red when this epoch participates in a
            # merge candidate. This is deliberately symmetric: the user
            # can see immediately which two epochs will be merged.
            for side in ("start", "end"):
                point = self._theta_handle_position(epoch, side)
                if point is None:
                    continue

                if merge_candidate:
                    handle_color = QColor(255, 60, 60)
                elif selected:
                    handle_color = QColor(60, 255, 60)
                else:
                    handle_color = QColor(255, 220, 80)

                painter.setBrush(QBrush(handle_color))
                painter.setPen(QPen(QColor(20, 20, 20), 1.2))
                r = self._theta_handle_radius
                painter.drawEllipse(
                    QRectF(point[0] - r, point[1] - r, 2 * r, 2 * r)
                )

    # ------------------------------------------------------------------
    # Merge logic
    # ------------------------------------------------------------------

    def _theta_find_merge_candidates(self, epoch_index: int, side: str, sample: int) -> set[int]:
        """Return every same-channel epoch touched by the dragged boundary.

        The search is directional, so if a boundary is dragged through several
        epochs, every intervening epoch becomes part of the merge candidate.
        The configured minimum merge gap is also respected.
        """
        if not (0 <= epoch_index < len(self.theta_epochs)):
            return set()

        epoch = self.theta_epochs[epoch_index]
        candidates: set[int] = set()
        sample = int(sample)

        original_start = int(self._theta_drag_original_start)
        original_end = int(self._theta_drag_original_end)
        gap = int(self.theta_min_merge_gap_samples)

        for i, other in enumerate(self.theta_epochs):
            if i == epoch_index or other.channel != epoch.channel:
                continue

            if side == "end":
                # The end handle moves to the right. Only epochs that were
                # originally to the right can participate. Once the dragged
                # boundary is within the configured gap of an epoch's start
                # (or has crossed it), that epoch becomes a candidate.
                other_start = int(other.start_sample)
                if other_start >= original_end and other_start - sample <= gap:
                    candidates.add(i)
            else:
                # The start handle moves to the left. Only epochs that were
                # originally to the left can participate. This is the exact
                # mirror of the end-handle logic above.
                other_end = int(other.end_sample)
                if other_end <= original_start and sample - other_end <= gap:
                    candidates.add(i)

        return candidates

    def _theta_merge_epochs(self, indices):
        indices = sorted(set(int(i) for i in indices if 0 <= int(i) < len(self.theta_epochs)))
        if len(indices) < 2:
            return

        selected = [self.theta_epochs[i] for i in indices]
        channels = {e.channel for e in selected}
        if len(channels) != 1:
            return

        start = min(e.start_sample for e in selected)
        end = max(e.end_sample for e in selected)
        duration_ms = (end - start) / float(self.engine.sr) * 1000.0

        weights = np.array([max(1, e.end_sample - e.start_sample + 1) for e in selected], dtype=float)
        theta_power = float(np.average([e.mean_theta_power for e in selected], weights=weights))
        delta_power = float(np.average([e.mean_delta_power for e in selected], weights=weights))
        mean_ratio = float(np.average([e.mean_ratio for e in selected], weights=weights))
        peak_epoch = max(selected, key=lambda e: e.peak_ratio)

        merged = ThetaEpoch(
            channel=selected[0].channel,
            start_sample=start,
            end_sample=end,
            peak_sample=peak_epoch.peak_sample,
            mean_theta_power=theta_power,
            mean_delta_power=delta_power,
            mean_ratio=mean_ratio,
            peak_ratio=float(peak_epoch.peak_ratio),
            duration_ms=duration_ms,
            manual=True,
        )

        for i in sorted(indices, reverse=True):
            self.theta_epochs.pop(i)
        self.theta_epochs.append(merged)
        self.theta_epochs.sort(key=lambda e: (e.channel, e.start_sample))

        self._theta_drag_epoch = None
        self._theta_drag_side = None
        self._theta_merge_candidate = None
        self._theta_merge_candidates.clear()
        self._theta_selected_epoch = min(
            range(len(self.theta_epochs)),
            key=lambda i: abs(self.theta_epochs[i].start_sample - start) if self.theta_epochs[i].channel == merged.channel else 10**18,
        )
        self.thetaEpochsChanged.emit(self.theta_epochs)
        self.thetaEpochSelected.emit(self._theta_selected_epoch)
        self.unsetCursor()
        self.update()


    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        if self._theta_creating_armed and event.button() == Qt.MouseButton.LeftButton:
            channel = self._theta_channel_at_y(event.position().y())
            if channel is not None:
                sample = self._theta_x_to_sample(event.position().x())
                self._theta_creating_channel = channel
                self._theta_creating_start_sample = sample
                event.accept()
                return
            # Clicked outside any channel lane while armed -- disarm
            # rather than leaving creation mode stuck active with
            # nothing to anchor it to.
            self.cancel_theta_creation()

        if event.button() == Qt.MouseButton.LeftButton and self.theta_epochs:
            handle = self._theta_handle_at(event.position())
            if handle is not None:
                index, side = handle
                self._theta_drag_epoch = index
                self._theta_drag_side = side
                self._theta_selected_epoch = index
                self.thetaEpochSelected.emit(index)
                epoch = self.theta_epochs[index]
                self._theta_drag_original_start = epoch.start_sample
                self._theta_drag_original_end = epoch.end_sample
                self._theta_merge_candidate = None
                self._theta_merge_candidates.clear()
                self.setCursor(QCursor(Qt.CursorShape.SizeHorCursor))
                self.update()
                event.accept()
                return

        if event.button() == Qt.MouseButton.LeftButton and self.theta_epochs:
            selected = self._theta_epoch_at(event.position())
            if selected is not None:
                self._theta_selected_epoch = selected
                self.thetaEpochSelected.emit(selected)
                self.update()
                event.accept()
                return

        super().mousePressEvent(event)

    def _theta_epoch_at(self, pos):
        """Return the epoch index whose visible rectangle contains pos."""
        plot_left, plot_right, _ = self._get_plot_bounds()
        for i, epoch in enumerate(self.theta_epochs):
            geometry = self._theta_channel_geometry(epoch.channel)
            if geometry is None:
                continue
            _, _, channel_height, y_center = geometry
            x1 = self._theta_time_to_x(self._theta_epoch_time(epoch.start_sample))
            x2 = self._theta_time_to_x(self._theta_epoch_time(epoch.end_sample))
            left = max(plot_left, min(x1, x2))
            right = min(plot_right, max(x1, x2))
            y1 = y_center - channel_height * 0.43
            y2 = y_center + channel_height * 0.43
            if left <= pos.x() <= right and y1 <= pos.y() <= y2:
                return i
        return None

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete and self.delete_selected_theta_epoch():
            event.accept()
            return
        super().keyPressEvent(event)

    def mouseMoveEvent(self, event):
        if self._theta_creating_channel is not None and self._theta_creating_start_sample is not None:
            # Live end position while dragging out a new epoch --
            # actual creation (with min-width validation) happens on
            # release; this just tracks the current extent for the
            # preview rectangle drawn by _draw_theta_epochs.
            self._theta_creating_end_sample = self._theta_x_to_sample(event.position().x())
            self.update()
            event.accept()
            return

        if self._theta_drag_epoch is not None and self._theta_drag_side is not None:
            index = self._theta_drag_epoch
            if not (0 <= index < len(self.theta_epochs)):
                self._theta_cancel_drag()
                return

            epoch = self.theta_epochs[index]
            sample = self._theta_x_to_sample(event.position().x())
            # Clamp using sample/sr math only -- see _theta_epoch_time.
            max_global_sample = max(0, int(round(self.engine.total_duration * self.engine.sr)))
            min_relative = -self.theta_detection_sample_offset
            max_relative = max_global_sample - self.theta_detection_sample_offset
            sample = max(min_relative, min(sample, max_relative))

            if self._theta_drag_side == "start":
                # Don't let the endpoint destroy its own epoch while dragging.
                epoch.start_sample = min(sample, epoch.end_sample - 1)
            else:
                epoch.end_sample = max(sample, epoch.start_sample + 1)

            epoch.duration_ms = (
                epoch.end_sample - epoch.start_sample
            ) / float(self.engine.sr) * 1000.0
            epoch.manual = True

            self._theta_merge_candidates = self._theta_find_merge_candidates(
                index,
                self._theta_drag_side,
                sample,
            )
            self._theta_merge_candidate = (
                min(self._theta_merge_candidates)
                if self._theta_merge_candidates else None
            )

            self.update()
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._theta_creating_channel is not None and self._theta_creating_start_sample is not None:
            channel = self._theta_creating_channel
            start_sample = self._theta_creating_start_sample
            end_sample = self._theta_creating_end_sample if self._theta_creating_end_sample is not None else start_sample

            lo, hi = min(start_sample, end_sample), max(start_sample, end_sample)
            if hi - lo >= self._theta_min_epoch_samples:
                new_epoch = ThetaEpoch(
                    channel=channel,
                    start_sample=lo,
                    end_sample=hi,
                    peak_sample=(lo + hi) // 2,
                    mean_theta_power=0.0,
                    mean_delta_power=0.0,
                    mean_ratio=0.0,
                    peak_ratio=0.0,
                    duration_ms=(hi - lo) / float(self.engine.sr) * 1000.0,
                    manual=True,
                )
                self.theta_epochs.append(new_epoch)
                self.theta_epochs.sort(key=lambda e: (e.channel, e.start_sample))
                self._theta_selected_epoch = next(
                    i for i, e in enumerate(self.theta_epochs) if e is new_epoch
                )
                self.thetaEpochsChanged.emit(self.theta_epochs)
                self.thetaEpochSelected.emit(self._theta_selected_epoch)
            # Too short (e.g. a stray click with no real drag) -- silently
            # discarded rather than creating a degenerate epoch.

            self._theta_creating_armed = False
            self._theta_creating_channel = None
            self._theta_creating_start_sample = None
            self._theta_creating_end_sample = None
            self.unsetCursor()
            self.update()
            event.accept()
            return

        if self._theta_drag_epoch is not None:
            index = self._theta_drag_epoch
            candidates = set(self._theta_merge_candidates)
            if candidates and 0 <= index < len(self.theta_epochs):
                candidates.add(index)
                self._theta_merge_epochs(candidates)
                self.unsetCursor()
                event.accept()
                return

            self._theta_cancel_drag()
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        """Double-click INSIDE the currently selected epoch's rectangle
        splits it at the click position. Anywhere else, falls through
        to the base class's double-click-to-reset-view behavior --
        double-click is only intercepted when it can unambiguously be
        interpreted as "split this epoch", not generally."""
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._theta_selected_epoch is not None
            and 0 <= self._theta_selected_epoch < len(self.theta_epochs)
        ):
            index = self._theta_selected_epoch
            epoch = self.theta_epochs[index]
            geometry = self._theta_channel_geometry(epoch.channel)
            if geometry is not None:
                _, _, channel_height, y_center = geometry
                plot_left, plot_right, _ = self._get_plot_bounds()
                x1 = self._theta_time_to_x(self._theta_epoch_time(epoch.start_sample))
                x2 = self._theta_time_to_x(self._theta_epoch_time(epoch.end_sample))
                left = max(plot_left, min(x1, x2))
                right = min(plot_right, max(x1, x2))
                y1 = y_center - channel_height * 0.43
                y2 = y_center + channel_height * 0.43
                pos = event.position()
                if left <= pos.x() <= right and y1 <= pos.y() <= y2:
                    split_sample = self._theta_x_to_sample(pos.x())
                    if self._theta_split_epoch(index, split_sample):
                        event.accept()
                        return
                    # Split rejected (too close to an edge) -- fall
                    # through to the base reset-view behavior rather
                    # than silently doing nothing on the double-click.

        super().mouseDoubleClickEvent(event)

    def _theta_cancel_drag(self):
        self._theta_drag_epoch = None
        self._theta_drag_side = None
        self._theta_merge_candidate = None
        self._theta_merge_candidates.clear()
        self.unsetCursor()
        self.update()

    def leaveEvent(self, event):
        if self._theta_drag_epoch is None:
            self.unsetCursor()
        super().leaveEvent(event)
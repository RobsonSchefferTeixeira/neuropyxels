"""
ripple_triggered_average_dialog.py

Non-modal QDialog for ripple-triggered average (RTA) analysis: pick
channels, a time window around each ripple's alignment point, and an
alignment reference (envelope peak, or the filtered signal's own
max/min cycle), then compute and display the averaged raw LFP trace
per channel across every currently-detected ripple event.

Opened from RippleDialog via a "Ripple-Triggered Average..." button,
using whatever events are in RippleDialog.events_by_channel at the time
(across all channels that have been detected/loaded, not just one).
Follows the same non-modal QDialog + QThread-computation pattern as
amplitude_power_dialog.py / phase_amplitude_dialog.py; the math lives
in core/ripple_triggered_average.py.

Usage
-----
    dialog = RippleTriggeredAverageDialog(engine, events_by_channel,
                                           sample_offsets_by_channel,
                                           ripple_params, available_channels,
                                           parent=ripple_dialog)
    dialog.show()   # non-modal
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QDoubleSpinBox,
    QCheckBox, QPushButton, QComboBox, QMessageBox, QWidget, QListWidget,
    QListWidgetItem, QAbstractItemView,
)
from PyQt6.QtGui import QColor

from core.ripple_detector import RippleEvent, RippleParams
from core.ripple_triggered_average import (
    RippleTriggeredAverageAnalyzer, RippleTriggeredAverageParams, ALIGNMENT_MODES,
)
from core.trace_engine import TraceEngine


BG_COLOR = "#1e1e1e"

ALIGNMENT_LABELS = {
    "envelope_peak": "Ripple max amplitude (envelope peak)",
    "filtered_max": "Ripple max cycle (filtered signal peak)",
    "filtered_min": "Ripple min cycle (filtered signal trough)",
}
ALIGNMENT_LABELS_REVERSE = {v: k for k, v in ALIGNMENT_LABELS.items()}

# Distinct, readable line colors cycled across channels -- avoids
# pyqtgraph's default palette repeating too quickly for probes with
# many selected channels.
CHANNEL_COLORS = [
    "#4a9eff", "#ff6b6b", "#42d77d", "#ffd23f", "#c77dff",
    "#ff9f1c", "#4cc9f0", "#f72585", "#94d2bd", "#e9c46a",
]


class _ComputeThread(QThread):
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, analyzer: RippleTriggeredAverageAnalyzer, raw_data,
                 sample_rate: float, events: list[RippleEvent],
                 params: RippleTriggeredAverageParams, parent=None):
        super().__init__(parent)
        self.analyzer = analyzer
        self.raw_data = raw_data
        self.sample_rate = sample_rate
        self.events = events
        self.params = params

    def run(self):
        try:
            result = self.analyzer.compute(self.raw_data, self.sample_rate, self.events, self.params)
            self.finished_ok.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class RippleTriggeredAverageDialog(QDialog):
    """
    Non-modal dialog: averaged LFP traces around ripple events, for a
    chosen set of channels, window, and alignment reference.
    """

    def __init__(self, engine: TraceEngine, events_by_channel: dict[int, list[RippleEvent]],
                 sample_offset_by_channel: dict[int, int], ripple_params: RippleParams,
                 available_channels: list[int], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ripple-Triggered Average")
        self.setModal(False)
        self.resize(900, 700)

        self.engine = engine
        # events_by_channel's sample fields are relative to whatever
        # sample_offset was active for THAT channel's detection run
        # (see RippleDialog's own _sample_offset bookkeeping) -- re-base
        # every event to absolute/global samples ONCE here, so
        # core.ripple_triggered_average only ever deals in one
        # consistent sample basis (its documented convention).
        self.events_global: list[RippleEvent] = self._rebase_events_to_global(events_by_channel, sample_offset_by_channel)
        self.available_channels = list(available_channels)
        self.analyzer = RippleTriggeredAverageAnalyzer(ripple_params)
        self._compute_thread: _ComputeThread | None = None
        self._last_result: dict | None = None

        self._build_ui()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @staticmethod
    def _rebase_events_to_global(events_by_channel: dict[int, list[RippleEvent]],
                                  sample_offset_by_channel: dict[int, int]) -> list[RippleEvent]:
        """Return a flat list of RippleEvent copies with every sample
        field shifted into absolute/global recording-sample terms."""
        import dataclasses
        rebased = []
        for ch, events in events_by_channel.items():
            offset = int(sample_offset_by_channel.get(ch, 0))
            if offset == 0:
                rebased.extend(events)
                continue
            for ev in events:
                rebased.append(dataclasses.replace(
                    ev,
                    start_sample=ev.start_sample + offset,
                    end_sample=ev.end_sample + offset,
                    peak_sample=ev.peak_sample + offset,
                    trough_sample=ev.trough_sample + offset,
                ))
        return rebased

    def _build_ui(self):
        pg.setConfigOption("background", BG_COLOR)
        pg.setConfigOption("foreground", "#d0d0d0")

        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        layout.addLayout(self._build_param_row())

        btn_row = QHBoxLayout()
        self.compute_btn = QPushButton("Compute Average")
        self.compute_btn.clicked.connect(self._on_compute_clicked)
        btn_row.addWidget(self.compute_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel("bottom", "Time from alignment point", units="s")
        self.plot_widget.setLabel("left", "Amplitude")
        self.plot_widget.setMenuEnabled(False)
        self.plot_widget.addLegend(offset=(10, 10))
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        layout.addWidget(self.plot_widget, stretch=1)

        self.status_label = QLabel("Select channels and click Compute Average.")
        self.status_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self.status_label)

    def _build_param_row(self) -> QHBoxLayout:
        row = QHBoxLayout()

        # ---- Channel picker (multi-select list) ----
        chan_col = QVBoxLayout()
        chan_col.addWidget(QLabel("Channels:"))
        self.channel_list = QListWidget()
        self.channel_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.channel_list.setMaximumWidth(120)
        for ch in self.available_channels:
            item = QListWidgetItem(f"CH{ch}")
            item.setData(Qt.ItemDataRole.UserRole, ch)
            self.channel_list.addItem(item)
            item.setSelected(True)  # default: all channels selected
        chan_col.addWidget(self.channel_list)
        row.addLayout(chan_col)

        # ---- Params grid ----
        grid = QGridLayout()

        grid.addWidget(QLabel("Window before (ms):"), 0, 0)
        self.window_before_spin = QDoubleSpinBox()
        self.window_before_spin.setRange(1.0, 5000.0)
        self.window_before_spin.setValue(100.0)
        grid.addWidget(self.window_before_spin, 0, 1)

        grid.addWidget(QLabel("Window after (ms):"), 0, 2)
        self.window_after_spin = QDoubleSpinBox()
        self.window_after_spin.setRange(1.0, 5000.0)
        self.window_after_spin.setValue(100.0)
        grid.addWidget(self.window_after_spin, 0, 3)

        grid.addWidget(QLabel("Align on:"), 1, 0)
        self.alignment_combo = QComboBox()
        self.alignment_combo.addItems([ALIGNMENT_LABELS[m] for m in ALIGNMENT_MODES])
        grid.addWidget(self.alignment_combo, 1, 1, 1, 3)

        self.show_sem_check = QCheckBox("Show SEM shading")
        self.show_sem_check.setChecked(True)
        self.show_sem_check.toggled.connect(self._on_display_options_changed)
        grid.addWidget(self.show_sem_check, 2, 0, 1, 2)

        self.normalize_check = QCheckBox("Offset channels vertically")
        self.normalize_check.setChecked(True)
        self.normalize_check.setToolTip(
            "Stack channels with a vertical offset (like the main trace "
            "view) instead of overlaying them at their native scale."
        )
        self.normalize_check.toggled.connect(self._on_display_options_changed)
        grid.addWidget(self.normalize_check, 2, 2, 1, 2)

        row.addLayout(grid, stretch=1)
        return row

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------

    def _selected_channels(self) -> list[int]:
        return [item.data(Qt.ItemDataRole.UserRole) for item in self.channel_list.selectedItems()]

    def _collect_params(self) -> RippleTriggeredAverageParams:
        alignment = ALIGNMENT_LABELS_REVERSE[self.alignment_combo.currentText()]
        return RippleTriggeredAverageParams(
            channels=self._selected_channels(),
            window_before_ms=self.window_before_spin.value(),
            window_after_ms=self.window_after_spin.value(),
            alignment=alignment,
        )

    def _on_compute_clicked(self):
        if not self.engine.data_loaded:
            QMessageBox.warning(self, "No data", "No data is currently loaded.")
            return
        if not self.events_global:
            QMessageBox.warning(
                self, "No ripple events",
                "No ripple events are available. Run detection (or load a "
                "ripple CSV) in the Ripple Detection dialog first."
            )
            return
        if not self._selected_channels():
            QMessageBox.warning(self, "No channels", "Select at least one channel.")
            return

        try:
            params = self._collect_params()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid parameters", str(exc))
            return

        self.compute_btn.setEnabled(False)
        self.compute_btn.setText("Computing...")
        self.status_label.setText(
            f"Averaging {len(self.events_global)} ripple event(s) across "
            f"{len(params.channels)} channel(s), please wait..."
        )

        self._compute_thread = _ComputeThread(
            self.analyzer, self.engine.data, self.engine.sr,
            self.events_global, params, parent=self,
        )
        self._compute_thread.finished_ok.connect(self._on_compute_finished)
        self._compute_thread.failed.connect(self._on_compute_failed)
        self._compute_thread.start()

    def _on_compute_failed(self, message: str):
        self.compute_btn.setEnabled(True)
        self.compute_btn.setText("Compute Average")
        self.status_label.setText(f"Error: {message}")
        QMessageBox.critical(self, "Computation failed", message)

    def _on_compute_finished(self, result: dict):
        self.compute_btn.setEnabled(True)
        self.compute_btn.setText("Compute Average")
        self._last_result = result
        self._render(result)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _on_display_options_changed(self, _checked: bool):
        if self._last_result is not None:
            self._render(self._last_result)

    def _render(self, result: dict):
        self.plot_widget.clear()
        # addLegend() only needs to be called once per PlotWidget, but
        # .clear() above wipes the plot items, not the legend -- however
        # re-adding a second legend on subsequent renders would stack
        # duplicate legend entries, so remove and rebuild it explicitly.
        if self.plot_widget.plotItem.legend is not None:
            self.plot_widget.plotItem.legend.clear()

        t = result["time_axis"]
        show_sem = self.show_sem_check.isChecked()
        offset_channels = self.normalize_check.isChecked()

        channels = sorted(result["channels"].keys())
        if not channels:
            self.status_label.setText("No channels in the result to display.")
            return

        # Vertical offset step: if stacking, space channels apart by a
        # multiple of the largest single channel's peak-to-peak mean
        # amplitude, so traces don't overlap regardless of relative
        # scale across channels.
        offset_step = 0.0
        if offset_channels:
            max_ptp = max(float(np.ptp(result["channels"][ch]["mean"])) for ch in channels)
            offset_step = max_ptp * 1.5 if max_ptp > 0 else 1.0

        for i, ch in enumerate(channels):
            data = result["channels"][ch]
            mean = data["mean"]
            sem = data["sem"]
            color = CHANNEL_COLORS[i % len(CHANNEL_COLORS)]

            y_offset = -i * offset_step if offset_channels else 0.0

            if show_sem and data["n_events"] > 1:
                upper = mean + sem + y_offset
                lower = mean - sem + y_offset
                fill_color = QColor(color)
                fill_color.setAlpha(40)
                fill = pg.FillBetweenItem(
                    pg.PlotDataItem(t, upper),
                    pg.PlotDataItem(t, lower),
                    brush=pg.mkBrush(fill_color),
                )
                self.plot_widget.addItem(fill)

            self.plot_widget.plot(
                t, mean + y_offset,
                pen=pg.mkPen(color, width=2),
                name=f"CH{ch} (n={data['n_events']})",
            )

        # Vertical marker at the alignment point (t=0).
        vline = pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen("#ffffff", width=1, style=Qt.PenStyle.DashLine))
        self.plot_widget.addItem(vline)

        n_total = result["n_events_total"]
        n_skipped = result["n_events_skipped"]
        alignment_label = ALIGNMENT_LABELS[result["params"].alignment]
        status = f"{n_total} event(s) averaged"
        if n_skipped:
            status += f" ({n_skipped} skipped: too close to a recording edge, or filtering failed)"
        status += f"  |  Aligned on: {alignment_label}"
        self.status_label.setText(status)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if self._compute_thread is not None and self._compute_thread.isRunning():
            self._compute_thread.wait(2000)
        super().closeEvent(event)

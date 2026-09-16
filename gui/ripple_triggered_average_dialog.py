"""
ripple_triggered_average_dialog.py

Non-modal QDialog for ripple-triggered average (RTA) analysis: pick
channels, a time window around each ripple's alignment point, and an
alignment reference (envelope peak, or the filtered signal's own
max/min cycle), then compute and display the averaged raw LFP trace per
channel across every currently-detected ripple event.

Opened from RippleDialog via a "Ripple-Triggered Average..." button,
using whatever events are in RippleDialog.events_by_channel at the time
(across all channels that have been detected/loaded, not just one).
Follows the same non-modal QDialog + QThread-computation pattern as
amplitude_power_dialog.py / phase_amplitude_dialog.py; the math lives
in core/ripple_triggered_average.py.

Channel selection and display
------------------------------
Starting channel selection is whatever's currently ACTIVE in the main
trace view (RippleDialog passes this in), not every channel in the
recording -- the person is expected to already be looking at the
channels they care about. A "Select channels from Probe Map..." button
opens a SEPARATE, scoped ProbeMapWidget instance (own popup dialog, not
shared with the main window) pre-seeded with the current selection, for
picking a different/wider/narrower set spatially.

Results render as small trace glyphs positioned at each selected
channel's real (x, y) probe position -- phy2's waveform-view idiom,
implemented in gui/rta_probe_trace_widget.py's RTAProbeTraceWidget
(custom QPainter, not pyqtgraph, for performance at high channel
counts). Channels group into columns by shank, ordered left-to-right by
real physical x-position, with a vertical divider between distinct
shank/probe columns -- multiple probes/shanks appear side by side
rather than interleaved.

Usage
-----
    dialog = RippleTriggeredAverageDialog(engine, probe_data, events_by_channel,
                                           sample_offsets_by_channel, ripple_params,
                                           initial_channels, parent=ripple_dialog)
    dialog.show()   # non-modal
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QDoubleSpinBox,
    QCheckBox, QPushButton, QComboBox, QMessageBox, QWidget, QListWidget,
    QListWidgetItem, QAbstractItemView,
)

from core.ripple_detector import RippleEvent, RippleParams
from core.ripple_triggered_average import (
    RippleTriggeredAverageAnalyzer, RippleTriggeredAverageParams, ALIGNMENT_MODES,
)
from core.trace_engine import TraceEngine
from gui.probe_map_widget import ProbeMapWidget
from gui.rta_probe_trace_widget import RTAProbeTraceWidget


ALIGNMENT_LABELS = {
    "envelope_peak": "Ripple max amplitude (envelope peak)",
    "filtered_max": "Ripple max cycle (filtered signal peak)",
    "filtered_min": "Ripple min cycle (filtered signal trough)",
}
ALIGNMENT_LABELS_REVERSE = {v: k for k, v in ALIGNMENT_LABELS.items()}


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


class _ProbePickerDialog(QDialog):
    """
    Small popup hosting a SCOPED ProbeMapWidget instance for selecting
    which channels feed the ripple-triggered average -- deliberately
    separate from the main window's own probe map (different selection
    state, different purpose), reusing the same widget class since the
    interaction (click electrodes to select, drag/zoom, double-click to
    clear) is exactly what's wanted here too.
    """

    def __init__(self, probe_data: dict, initial_channels: list[int], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Channels")
        self.setModal(True)
        self.resize(500, 650)

        layout = QVBoxLayout(self)
        self.probe_map = ProbeMapWidget(probe_data)
        layout.addWidget(self.probe_map, stretch=1)

        btn_row = QHBoxLayout()
        done_btn = QPushButton("Done")
        done_btn.setAutoDefault(False)
        done_btn.setDefault(False)
        done_btn.clicked.connect(self.accept)
        btn_row.addStretch(1)
        btn_row.addWidget(done_btn)
        layout.addLayout(btn_row)

        # Seed with whatever's already selected -- set AFTER the widget
        # is fully constructed and in the layout so its internal scatter
        # item/state arrays exist.
        self.probe_map.set_selected_channels(initial_channels)

    def selected_channels(self) -> list[int]:
        return self.probe_map.get_selected_channels()


class RippleTriggeredAverageDialog(QDialog):
    """
    Non-modal dialog: averaged LFP traces around ripple events, for a
    chosen set of channels, window, and alignment reference, displayed
    as small glyphs positioned by real probe geometry.
    """

    def __init__(self, engine: TraceEngine, probe_data: dict,
                 events_by_channel: dict[int, list[RippleEvent]],
                 sample_offset_by_channel: dict[int, int], ripple_params: RippleParams,
                 initial_channels: list[int], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ripple-Triggered Average")
        self.setModal(False)
        self.resize(1000, 750)

        self.engine = engine
        self.probe_data = probe_data
        # events_by_channel's sample fields are relative to whatever
        # sample_offset was active for THAT channel's detection run
        # (see RippleDialog's own _sample_offset bookkeeping) -- re-base
        # every event to absolute/global samples ONCE here, so
        # core.ripple_triggered_average only ever deals in one
        # consistent sample basis (its documented convention).
        self.events_global: list[RippleEvent] = self._rebase_events_to_global(
            events_by_channel, sample_offset_by_channel
        )
        # Starting selection = whatever's currently active in the main
        # trace view, NOT every channel in the recording -- the person
        # is expected to already be looking at the channels they care
        # about; the probe-picker popup covers picking something else.
        self.selected_channels: list[int] = list(initial_channels)
        self.analyzer = RippleTriggeredAverageAnalyzer(ripple_params)
        self._compute_thread: _ComputeThread | None = None
        self._last_result: dict | None = None

        self._build_ui()
        self.probe_trace_widget.set_geometry_from_probe(self.probe_data, self.selected_channels)

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
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        layout.addLayout(self._build_param_row())

        btn_row = QHBoxLayout()
        self.compute_btn = QPushButton("Compute Average")
        self.compute_btn.setAutoDefault(False)
        self.compute_btn.setDefault(False)
        self.compute_btn.clicked.connect(self._on_compute_clicked)
        btn_row.addWidget(self.compute_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self.probe_trace_widget = RTAProbeTraceWidget()
        layout.addWidget(self.probe_trace_widget, stretch=1)

        self.status_label = QLabel("Select channels and click Compute Average.")
        self.status_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self.status_label)

    def _build_param_row(self) -> QHBoxLayout:
        row = QHBoxLayout()

        # ---- Channel picker: current selection list + probe-map button ----
        chan_col = QVBoxLayout()
        chan_col.addWidget(QLabel("Channels:"))
        self.channel_list = QListWidget()
        self.channel_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.channel_list.setMaximumWidth(120)
        self._populate_channel_list()
        chan_col.addWidget(self.channel_list)

        self.pick_from_probe_btn = QPushButton("Select from Probe Map...")
        self.pick_from_probe_btn.setAutoDefault(False)
        self.pick_from_probe_btn.setDefault(False)
        self.pick_from_probe_btn.setToolTip(
            "Open a probe map to select channels spatially, instead of "
            "picking from the plain list above."
        )
        self.pick_from_probe_btn.clicked.connect(self._on_pick_from_probe_clicked)
        chan_col.addWidget(self.pick_from_probe_btn)

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

        row.addLayout(grid, stretch=1)
        return row

    def _populate_channel_list(self):
        self.channel_list.clear()
        for ch in self.selected_channels:
            item = QListWidgetItem(f"CH{ch}")
            item.setData(Qt.ItemDataRole.UserRole, ch)
            self.channel_list.addItem(item)
            item.setSelected(True)

    def _on_pick_from_probe_clicked(self):
        picker = _ProbePickerDialog(self.probe_data, self.selected_channels, parent=self)
        if picker.exec() == QDialog.DialogCode.Accepted:
            new_selection = picker.selected_channels()
            if new_selection:
                self.selected_channels = new_selection
                self._populate_channel_list()
                self.probe_trace_widget.set_geometry_from_probe(self.probe_data, self.selected_channels)

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

        # Channel selection may have changed (via the list or the probe
        # picker) since the geometry was last set -- keep the probe
        # trace widget's layout in sync with what's about to be computed.
        if sorted(params.channels) != sorted(self.selected_channels):
            self.selected_channels = list(params.channels)
            self.probe_trace_widget.set_geometry_from_probe(self.probe_data, self.selected_channels)

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
        channels = sorted(result["channels"].keys())
        if not channels:
            self.status_label.setText("No channels in the result to display.")
            return

        self.probe_trace_widget.set_result(
            result["time_axis"], result["channels"],
            show_sem=self.show_sem_check.isChecked(),
        )

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


"""
theta_cycle_dialog.py

Non-modal dialog for theta cycle detection. Detection runs on a
QThread per requested channel; results are collected into
`cycles_by_channel`, which drives both the dialog's table and the
trace view's overlay.

One table per channel, selected via a dropdown. The channels field at
the top accepts a comma-separated list; detection runs on each
independently, matching ripple detection's model.

Save writes a single CSV (all channels, one table) plus a small
.meta.json sidecar into a `neuropyxels/` folder next to the loaded
.dat. Load reads both back from the same folder, restoring the
detection parameters so the filtered overlay and RTA-style analyses
work without re-running detection.

Editing is minimal: click a cycle's landmark circle on the trace view
to select it, Delete to remove, Ctrl+Z to undo. Cycles have no drag,
split, or merge.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QDoubleSpinBox,
    QSpinBox, QCheckBox, QPushButton, QComboBox, QMessageBox, QWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QLineEdit, QFileDialog,
)
from core.analysis_folder import (
    analysis_folder_for, ensure_analysis_folder,
)
from core.session_sidecar import save_sidecar, load_sidecar
from core.theta_cycle_detector import (
    ThetaCycleDetector, ThetaCycleParams, ThetaCycle,
)
from core.theta_cycle_export import (
    export_cycles_to_csv, read_cycles_from_csv,
)
from core.trace_engine import TraceEngine
from gui.theta_cycle_trace_view import ThetaCycleTraceViewWidget


class _DetectThread(QThread):
    """Runs cycle detection across the requested channels off the UI
    thread. Emits results as {channel: list[ThetaCycle]}."""

    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, int)

    def __init__(self, channels, raw_data, sample_rate, start_idx, end_idx, params, parent=None):
        super().__init__(parent)
        self.channels = channels
        self.raw_data = raw_data
        self.sample_rate = float(sample_rate)
        # Sample indices, already resolved by the dialog against the
        # engine's current time base. See theta_epoch_dialog for the
        # same pattern.
        self.start_idx = int(start_idx)
        self.end_idx = int(end_idx)
        self.params = params
        self.detector = ThetaCycleDetector()

    def run(self):
        try:
            start_idx = max(0, self.start_idx)
            end_idx = min(self.raw_data.shape[0], self.end_idx)
            if start_idx >= end_idx:
                raise ValueError("Invalid time range for detection.")

            results: dict[int, list[ThetaCycle]] = {}
            for i, ch in enumerate(self.channels):
                if ch < 0 or ch >= self.raw_data.shape[1]:
                    raise ValueError(
                        f"Channel {ch} is outside the loaded data."
                    )
                signal = self.raw_data[start_idx:end_idx, ch] \
                    .flatten().astype(np.float64)
                cycles = self.detector.detect(
                    signal, self.sample_rate, self.params, channel=ch,
                )
                # Detector returns sample indices relative to `signal`
                # (i.e. 0 = start_idx). Convert to raw recording samples
                # before handing off.
                for c in cycles:
                    c.zero_crossing_start += start_idx
                    c.peak1 += start_idx
                    c.zero_crossing_mid += start_idx
                    c.valley += start_idx
                    c.zero_crossing_end += start_idx
                    c.peak2 += start_idx
                results[ch] = cycles
                self.progress.emit(i + 1, len(self.channels))

            self.finished_ok.emit(results)
        except Exception as exc:
            self.failed.emit(str(exc))


class ThetaCycleDialog(QDialog):
    """Non-modal theta cycle detection dialog."""

    cyclesChanged = pyqtSignal(object)   # list[ThetaCycle] (flat)
    cycleSelected = pyqtSignal(int)
    eventFocused = pyqtSignal(int)   # channel id; the user clicked a row

    def __init__(
        self,
        probe_data: dict,
        engine: TraceEngine,
        trace_view: ThetaCycleTraceViewWidget,
        initial_channels: list[int] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Theta Cycle Detection")
        self.setModal(False)
        self.resize(900, 640)

        self.engine = engine
        self.trace_view = trace_view
        self.probe_data = probe_data
        self._detect_thread: _DetectThread | None = None

        # Source of truth: cycles grouped by channel.
        self.cycles_by_channel: dict[int, list[ThetaCycle]] = {}
        self._selected_channel: int | None = None
        self._selected_index: int | None = None

        self._detect_n_samples = 0
        self._last_params_for_sidecar = None

        # ---- Autosave ----
        self._autosave_armed = False
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(600)
        self._autosave_timer.timeout.connect(self._autosave_now)

        self._build_ui()
        self._sync_time_range_from_engine()
        self._refresh_export_buttons()

        # Push the spinboxes' initial values to the trace view so the
        # two start in agreement even if a previous dialog session left
        # the trace view at a different gain / width.
        if self.trace_view is not None:
            self.trace_view.set_theta_cycle_gain(
                float(self.trace_gain_spin.value())
            )
            self.trace_view.set_theta_cycle_trace_width(
                float(self.trace_width_spin.value())
            )

        if initial_channels:
            self.channel_edit.setText(
                ",".join(str(c) for c in initial_channels)
            )

        # Keep the dialog's copy in sync with edits made on the trace
        # view (deletions via Delete key or Ctrl+Z).
        self.trace_view.thetaCyclesChanged.connect(self.set_cycles_from_trace)
        self.trace_view.thetaCycleSelected.connect(self.set_selected_cycle)

    # ------------------------------------------------------------------
    # UI scaffolding
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        layout.addLayout(self._build_param_grid())

        btn_row = QHBoxLayout()
        self.detect_btn = QPushButton("Detect Theta Cycles")
        self.detect_btn.clicked.connect(self._on_detect_clicked)
        btn_row.addWidget(self.detect_btn)

        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color: #888; font-size: 10px;")
        btn_row.addWidget(self.progress_label, stretch=1)
        layout.addLayout(btn_row)

        self.table = QTableWidget(0, 18)
        self.table.setHorizontalHeaderLabels([
            "Channel",
            "zc_start_smp", "peak1_smp", "zc_mid_smp", "valley_smp",
            "zc_end_smp", "peak2_smp", "duration_smp",
            "zc_start_s", "peak1_s", "zc_mid_s", "valley_s", "zc_end_s",
            "peak2_s", "duration_s",
            "amp", "θ/δ", "fast/θ",
        ])

        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.table.itemSelectionChanged.connect(
            self._on_table_selection_changed
        )
        layout.addWidget(self.table, stretch=1)

        # ---- Navigation ----
        nav_row = QHBoxLayout()
        self.prev_btn = QPushButton("◄ Prev Cycle")
        self.prev_btn.setAutoDefault(False)
        self.prev_btn.clicked.connect(lambda: self._step_cycle(-1))
        nav_row.addWidget(self.prev_btn)

        self.next_btn = QPushButton("Next Cycle ►")
        self.next_btn.setAutoDefault(False)
        self.next_btn.clicked.connect(lambda: self._step_cycle(+1))
        nav_row.addWidget(self.next_btn)

        nav_row.addWidget(QLabel("Window (s):"))
        self.nav_window_spin = QDoubleSpinBox()
        self.nav_window_spin.setDecimals(2)
        self.nav_window_spin.setRange(0.05, 30.0)
        self.nav_window_spin.setValue(2.0)
        nav_row.addWidget(self.nav_window_spin)
        nav_row.addStretch(1)
        layout.addLayout(nav_row)

        # ---- Export / load ----
        # One export button, matching the theta epoch dialog: everything
        # goes into a single CSV, with a channel column inside. No
        # per-channel split, no channel suffix in the filename.
        io_row = QHBoxLayout()
        self.export_btn = QPushButton("Save")
        self.export_btn.setAutoDefault(False)
        self.export_btn.clicked.connect(self._on_save_clicked)
        io_row.addWidget(self.export_btn)

        self.load_btn = QPushButton("Load")
        self.load_btn.setAutoDefault(False)
        self.load_btn.clicked.connect(self._on_load_clicked)
        io_row.addWidget(self.load_btn)
        io_row.addStretch(1)
        layout.addLayout(io_row)

        self.status_label = QLabel(
            "Select channels and click Detect Theta Cycles."
        )
        self.status_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self.status_label)

    def _build_param_grid(self) -> QGridLayout:
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        pairs = []

        self.channel_edit = QLineEdit()
        self.channel_edit.setPlaceholderText("e.g. 12,13,14")
        pairs.append(("Channels", self.channel_edit))

        self.start_spin = QDoubleSpinBox()
        self.start_spin.setDecimals(3)
        self.start_spin.setRange(0.0, 1e9)
        pairs.append(("Start (s)", self.start_spin))

        self.end_spin = QDoubleSpinBox()
        self.end_spin.setDecimals(3)
        self.end_spin.setRange(0.01, 1e9)
        self.end_spin.setValue(10.0)
        pairs.append(("End (s)", self.end_spin))

        self.low_spin = QDoubleSpinBox()
        self.low_spin.setRange(0.1, 100.0)
        self.low_spin.setValue(4.0)
        pairs.append(("Theta low (Hz)", self.low_spin))

        self.high_spin = QDoubleSpinBox()
        self.high_spin.setRange(0.2, 100.0)
        self.high_spin.setValue(12.0)
        pairs.append(("Theta high (Hz)", self.high_spin))

        self.min_amp_spin = QDoubleSpinBox()
        self.min_amp_spin.setRange(0.0, 99.0)
        self.min_amp_spin.setValue(15.0)
        self.min_amp_spin.setSuffix(" %")
        self.min_amp_spin.setToolTip(
            "A candidate peak or valley must exceed this percentile of "
            "the theta envelope."
        )
        pairs.append(("Min amp percentile", self.min_amp_spin))

        self.delta_check = QCheckBox("Delta correction")
        self.delta_check.setChecked(True)
        pairs.append((None, self.delta_check))

        self.delta_low_spin = QDoubleSpinBox()
        self.delta_low_spin.setRange(0.0, 20.0)
        self.delta_low_spin.setValue(0.5)
        pairs.append(("δ low (Hz)", self.delta_low_spin))

        self.delta_high_spin = QDoubleSpinBox()
        self.delta_high_spin.setRange(0.1, 20.0)
        self.delta_high_spin.setValue(4.0)
        pairs.append(("δ high (Hz)", self.delta_high_spin))

        self.delta_ratio_spin = QDoubleSpinBox()
        self.delta_ratio_spin.setRange(0.0, 20.0)
        self.delta_ratio_spin.setSingleStep(0.1)
        self.delta_ratio_spin.setValue(1.5)
        self.delta_ratio_spin.setToolTip(
            "Minimum mean(theta)/mean(delta) ratio over each cycle's "
            "span. Cycles below this are discarded."
        )
        pairs.append(("θ/δ ratio min", self.delta_ratio_spin))

        self.fast_check = QCheckBox("Fast correction")
        self.fast_check.setChecked(False)
        pairs.append((None, self.fast_check))

        self.fast_low_spin = QDoubleSpinBox()
        self.fast_low_spin.setRange(20.0, 5000.0)
        self.fast_low_spin.setValue(170.0)
        pairs.append(("Fast low (Hz)", self.fast_low_spin))

        self.fast_ratio_spin = QDoubleSpinBox()
        self.fast_ratio_spin.setRange(0.0, 20.0)
        self.fast_ratio_spin.setSingleStep(0.1)
        self.fast_ratio_spin.setValue(0.5)
        self.fast_ratio_spin.setToolTip(
            "Maximum mean(fast)/mean(theta) ratio over each cycle's "
            "span. Cycles above this are discarded."
        )
        pairs.append(("fast/θ ratio max", self.fast_ratio_spin))

        self.edge_combo = QComboBox()
        self.edge_combo.addItems(["rising_zc", "falling_zc"])
        pairs.append(("Cycle start", self.edge_combo))

        self.trace_gain_spin = QDoubleSpinBox()
        self.trace_gain_spin.setRange(0.05, 100.0)
        self.trace_gain_spin.setSingleStep(0.1)
        self.trace_gain_spin.setDecimals(2)
        self.trace_gain_spin.setValue(1.0)
        self.trace_gain_spin.setSuffix(" ×")
        self.trace_gain_spin.setToolTip(
            "Multiplier on the fixed reference scale of the theta-\n"
            "filtered overlay. The reference is measured once per\n"
            "channel from the first window it is drawn in, so\n"
            "scrolling does not change the trace height; only this\n"
            "spinbox does."
        )
        self.trace_gain_spin.valueChanged.connect(self._on_trace_gain_changed)
        pairs.append(("Trace gain", self.trace_gain_spin))

        self.trace_width_spin = QDoubleSpinBox()
        self.trace_width_spin.setRange(0.5, 5.0)
        self.trace_width_spin.setSingleStep(0.1)
        self.trace_width_spin.setDecimals(2)
        self.trace_width_spin.setValue(1.0)
        self.trace_width_spin.setSuffix(" px")
        self.trace_width_spin.setToolTip(
            "Stroke width of the theta-filtered overlay trace. The\n"
            "selected-cycle highlight is always one pixel wider."
        )
        self.trace_width_spin.valueChanged.connect(
            self._on_trace_width_changed
        )
        pairs.append(("Trace width", self.trace_width_spin))


        n_cols = 4
        row = 0
        col = 0
        for label_text, widget in pairs:
            cell = QWidget()
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(2)
            if label_text is not None:
                lbl = QLabel(label_text)
                lbl.setStyleSheet("color: #aaa; font-size: 10px;")
                cell_layout.addWidget(lbl)
            else:
                cell_layout.addStretch(0)
            cell_layout.addWidget(widget)
            grid.addWidget(cell, row, col)
            col += 1
            if col >= n_cols:
                col = 0
                row += 1

        return grid

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _sync_time_range_from_engine(self):
        """Spinbox bounds follow the engine's current time base: clock
        values when a timestamps.npy is loaded, elapsed seconds
        otherwise. Called from __init__ and by MainWindow whenever the
        timestamps state changes after the dialog was opened."""
        if not self.engine.data_loaded:
            return
        t_min, t_max = self.engine.get_time_range()
        self.start_spin.blockSignals(True)
        self.end_spin.blockSignals(True)
        self.start_spin.setRange(t_min, t_max)
        self.end_spin.setRange(t_min + 1e-3, t_max)
        span = t_max - t_min
        self.start_spin.setValue(t_min)
        self.end_spin.setValue(t_min + min(10.0, max(0.0, span)))
        self.start_spin.blockSignals(False)
        self.end_spin.blockSignals(False)

    def on_engine_time_base_changed(self):
        """Called by MainWindow when a timestamps.npy is loaded after
        the dialog opened. Re-syncs spinbox bounds and repopulates the
        table so *_s columns reflect the new time base."""
        self._sync_time_range_from_engine()
        self._populate_table()
        self._refresh_export_buttons()
    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _parse_channels(self) -> list[int]:
        text = self.channel_edit.text().strip()
        if not text:
            return []
        out = []
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                out.append(int(part))
            except ValueError:
                raise ValueError(f"'{part}' is not a valid channel number.")
        return out
    
    def _on_trace_gain_changed(self, value: float):
        if self.trace_view is not None:
            self.trace_view.set_theta_cycle_gain(float(value))

    def _on_trace_width_changed(self, value: float):
        if self.trace_view is not None:
            self.trace_view.set_theta_cycle_trace_width(float(value))

    def _collect_params(self) -> ThetaCycleParams:
        return ThetaCycleParams(
            channels=[],  # unused by the detector; the thread iterates
            start_time=self.start_spin.value(),
            end_time=self.end_spin.value(),
            low_freq=self.low_spin.value(),
            high_freq=self.high_spin.value(),
            min_theta_peak_percentile=self.min_amp_spin.value(),
            delta_correction=self.delta_check.isChecked(),
            delta_low_freq=self.delta_low_spin.value(),
            delta_high_freq=self.delta_high_spin.value(),
            theta_delta_ratio_min=self.delta_ratio_spin.value(),
            fast_correction=self.fast_check.isChecked(),
            fast_low_freq=self.fast_low_spin.value(),
            fast_theta_ratio_max=self.fast_ratio_spin.value(),
            cycle_start_edge=self.edge_combo.currentText(),
        )

    def _on_detect_clicked(self):
        if not self.engine.data_loaded:
            QMessageBox.warning(self, "No data", "No data is currently loaded.")
            return

        try:
            channels = self._parse_channels()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid channels", str(exc))
            return
        if not channels:
            QMessageBox.warning(self, "No channels",
                                "Enter at least one channel.")
            return

        if self.start_spin.value() >= self.end_spin.value():
            QMessageBox.warning(self, "Invalid range",
                                "Start time must be before end time.")
            return
        if self.low_spin.value() >= self.high_spin.value():
            QMessageBox.warning(self, "Invalid theta band",
                                "Theta low must be below theta high.")
            return

        params = self._collect_params()

        # Tell the overlay which band we're using, so the filtered
        # trace it draws matches the one the detector saw.
        self.trace_view.set_theta_cycle_band(
            params.low_freq, params.high_freq,
        )

        self.detect_btn.setEnabled(False)
        self.detect_btn.setText("Detecting...")
        self.progress_label.setText("")
        self.status_label.setText("Running detection...")


        start_idx = self.engine.timestamp_to_sample(self.start_spin.value())
        end_idx = self.engine.timestamp_to_sample(self.end_spin.value())

        self._sample_offset = int(start_idx)
        self._detect_n_samples = max(0, int(end_idx) - int(start_idx))
        self._last_params_for_sidecar = params

        self._detect_thread = _DetectThread(
            channels, self.engine.data, self.engine.sr,
            start_idx, end_idx, params, parent=self,
        )

        self._detect_thread.progress.connect(self._on_progress)
        self._detect_thread.finished_ok.connect(self._on_detect_finished)
        self._detect_thread.failed.connect(self._on_detect_failed)
        self._detect_thread.start()

    def _on_progress(self, done: int, total: int):
        self.progress_label.setText(f"{done}/{total} channels")

    def _on_detect_failed(self, message: str):
        self.detect_btn.setEnabled(True)
        self.detect_btn.setText("Detect Theta Cycles")
        self.progress_label.setText("")
        self.status_label.setText(f"Error: {message}")
        QMessageBox.critical(self, "Detection failed", message)

    def _on_detect_finished(self, results: dict):
        self.detect_btn.setEnabled(True)
        self.detect_btn.setText("Detect Theta Cycles")
        self.progress_label.setText("")
        self.cycles_by_channel = results
        self._populate_table()
        self._push_flat_cycles_to_trace()
        self._refresh_export_buttons()
        total = sum(len(v) for v in results.values())
        n_ch = sum(1 for v in results.values() if v)
        self.status_label.setText(
            f"Detected {total} cycle(s) across {n_ch} channel(s)."
        )
        self._autosave_armed = True
        self._schedule_autosave()

    # ------------------------------------------------------------------
    # Channel dropdown + table
    # ------------------------------------------------------------------


    def _populate_table(self):
        """Populate the pooled table. Time columns are derived from
        the current engine's time base every time this runs, so a
        timestamps file loaded after detection (or after a CSV load)
        is reflected on the next repopulation without any explicit
        'recompute' step."""
        flat = self._flat_cycles()
        self._selected_index = None
        self.table.blockSignals(True)
        try:
            self.table.setRowCount(len(flat))
            for row, c in enumerate(flat):
                zc_start = int(c.zero_crossing_start)
                peak1 = int(c.peak1)
                zc_mid = int(c.zero_crossing_mid)
                valley = int(c.valley)
                zc_end = int(c.zero_crossing_end)
                peak2 = int(c.peak2)
                dur_smp = zc_end - zc_start

                def t(smp):
                    return float(self.engine.sample_to_time(smp))

                vals = [
                    str(c.channel),
                    str(zc_start),
                    str(peak1),
                    str(zc_mid),
                    str(valley),
                    str(zc_end),
                    str(peak2),
                    str(dur_smp),
                    f"{t(zc_start):.3f}",
                    f"{t(peak1):.3f}",
                    f"{t(zc_mid):.3f}",
                    f"{t(valley):.3f}",
                    f"{t(zc_end):.3f}",
                    f"{t(peak2):.3f}",
                    f"{dur_smp / float(self.engine.sr):.3f}",
                    f"{c.amplitude:.3g}",
                    "" if c.theta_delta_ratio is None
                        else f"{c.theta_delta_ratio:.3f}",
                    "" if c.fast_theta_ratio is None
                        else f"{c.fast_theta_ratio:.3f}",
                ]
                for col, v in enumerate(vals):
                    item = QTableWidgetItem(v)
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignCenter
                        | Qt.AlignmentFlag.AlignVCenter
                    )
                    self.table.setItem(row, col, item)
            self.table.clearSelection()
        finally:
            self.table.blockSignals(False)

    def _flat_cycles(self) -> list[ThetaCycle]:
        flat: list[ThetaCycle] = []
        for ch in sorted(self.cycles_by_channel.keys()):
            flat.extend(self.cycles_by_channel[ch])
        flat.sort(key=lambda c: (c.channel, c.zero_crossing_start))
        return flat

    def _push_flat_cycles_to_trace(self):
        self.trace_view.set_theta_cycles(self._flat_cycles())
        self.cyclesChanged.emit(self._flat_cycles())

    def _on_table_selection_changed(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            self._selected_index = None
            return
        row = rows[0].row()
        self._selected_index = row

        flat = self._flat_cycles()
        if not (0 <= row < len(flat)):
            return
        target = flat[row]

        self.trace_view.set_theta_cycle_selected(row)
        self.cycleSelected.emit(row)

        self._ensure_channel_visible(target.channel)
        self._pan_to_event(target)
        # Ask MainWindow to highlight this channel on the probe map.
        self.eventFocused.emit(int(target.channel))

    def _pan_to_event(self, cycle: ThetaCycle):
        """Center the trace view on this cycle's peak1 time."""
        try:
            peak_time = float(self.engine.sample_to_time(int(cycle.peak1)))
        except Exception:
            return
        window_duration = max(0.05, self.nav_window_spin.value())
        t_min, _ = self.engine.get_time_range()
        new_start = max(t_min, peak_time - window_duration / 2)

        self.trace_view.window_duration = window_duration
        self.trace_view.start_time = new_start
        self.trace_view.control_panel.set_duration(window_duration)
        self.trace_view._update_time_labels()
        self.trace_view._update_scrollbar_position()
        self.trace_view._invalidate_cache()
        self.trace_view.update()

    def _ensure_channel_visible(self, channel: int):
        if channel not in self.trace_view.channels:
            chans = list(self.trace_view.channels)
            chans.append(channel)
            self.trace_view.set_channels(chans)
        else:
            self.trace_view.update()

    def set_selected_cycle(self, flat_index: int):
        """Slot for trace_view.thetaCycleSelected -- a cycle was clicked on
        the trace view. Row index in the table equals flat_index, so
        selection is a direct lookup."""
        flat = self._flat_cycles()
        if not (0 <= flat_index < len(flat)):
            return
        self.table.blockSignals(True)
        self.table.selectRow(flat_index)
        self.table.blockSignals(False)
        self._selected_index = flat_index

    def set_cycles_from_trace(self, cycles: list):
        """Slot for trace_view.thetaCyclesChanged -- a cycle was deleted
        or an undo happened on the trace view. Regroup into
        cycles_by_channel and refresh the table."""
        regrouped: dict[int, list[ThetaCycle]] = {}
        for c in cycles:
            regrouped.setdefault(c.channel, []).append(c)
        for ch in list(self.cycles_by_channel.keys()):
            regrouped.setdefault(ch, [])
        for ch in regrouped:
            regrouped[ch].sort(key=lambda c: c.zero_crossing_start)
        self.cycles_by_channel = regrouped
        self._populate_table()
        self._refresh_export_buttons()
        self._schedule_autosave()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _step_cycle(self, direction: int):
        """Step Prev/Next through the flat, (channel, start_sample)-
        sorted list of every cycle across every detected channel --
        the same order the table shows. Crossing from the last cycle
        of one channel into the first of the next is intentional: the
        button means 'next row in the table', not 'next cycle on this
        channel'.

        If nothing is selected, direction > 0 starts at row 0 and
        direction < 0 starts at the last row. When the end (or start)
        is reached, the selection stays put -- no wrap.
        """
        flat = self._flat_cycles()
        if not flat:
            return

        rows = self.table.selectionModel().selectedRows()
        if rows:
            current_row = int(rows[0].row())
        else:
            current_row = -1 if direction > 0 else len(flat)

        new_row = current_row + direction
        new_row = max(0, min(new_row, len(flat) - 1))

        target = flat[new_row]

        self._ensure_channel_visible(int(target.channel))

        # Center the view on the target's peak1.
        peak_time = float(self.engine.sample_to_time(int(target.peak1)))
        window = max(0.05, self.nav_window_spin.value())
        t_min, _ = self.engine.get_time_range()
        new_start = max(t_min, peak_time - window / 2)

        self.trace_view.window_duration = window
        self.trace_view.start_time = new_start
        self.trace_view.control_panel.set_duration(window)
        self.trace_view._update_time_labels()
        self.trace_view._invalidate_cache()
        self.trace_view.update()

        self.table.blockSignals(True)
        self.table.selectRow(new_row)
        self.table.blockSignals(False)
        self.trace_view.set_theta_cycle_selected(new_row)
        self._selected_index = new_row

        self.status_label.setText(
            f"Cycle {new_row + 1} of {len(flat)} on CH{target.channel} "
            f"({'(next)' if direction > 0 else '(prev)'})"
        )
    # ------------------------------------------------------------------
    # Export / import
    # ------------------------------------------------------------------
    def _on_save_clicked(self):
        """Write every cycle from every channel into a single CSV in
        the neuropyxels/ folder next to the loaded .dat, plus a
        sidecar with the parameters that produced them."""
        if not self.cycles_by_channel or not any(self.cycles_by_channel.values()):
            QMessageBox.information(
                self, "Nothing to save",
                "Run detection (or load a CSV) first."
            )
            return

        folder = ensure_analysis_folder(self.engine)
        if folder is None:
            QMessageBox.warning(
                self, "Cannot save",
                "No data file is loaded, so there is no folder to save "
                "the analysis into."
            )
            return

        csv_path = folder / "theta_cycles.csv"
        flat = self._flat_cycles()
        try:
            export_cycles_to_csv(flat, self.engine, csv_path)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return

        n = int(getattr(self, "_detect_n_samples", 0))
        if n > 0:
            analysis_end_sample = int(self._sample_offset) + n
            try:
                save_sidecar(
                    csv_path=csv_path,
                    kind="theta_cycles",
                    sample_rate=float(self.engine.sr),
                    analysis_start_sample=int(self._sample_offset),
                    analysis_end_sample=analysis_end_sample,
                    channels=sorted(self.cycles_by_channel.keys()),
                    params=self._last_params_for_sidecar,
                )
            except Exception as exc:
                print(f"[theta-cycle-save] sidecar failed: {exc}")

        self.status_label.setText(
            f"Saved {len(flat)} cycle(s) to {folder}"
        )


    def _on_load_clicked(self):
        if not self.engine.data_loaded:
            QMessageBox.warning(self, "No data", "No data is currently loaded.")
            return

        folder = analysis_folder_for(self.engine)
        if folder is None:
            QMessageBox.warning(
                self, "Cannot load",
                "No data file is loaded, so there is no folder to load "
                "the analysis from."
            )
            return

        csv_path = folder / "theta_cycles.csv"
        if not csv_path.exists():
            QMessageBox.information(
                self, "Nothing to load",
                f"No saved theta cycles were found in:\n\n{folder}\n\n"
                "Detect theta cycles and click Save first."
            )
            return

        sidecar = None
        try:
            sidecar = load_sidecar(csv_path)
        except Exception as exc:
            QMessageBox.warning(
                self, "Sidecar unreadable",
                f"Could not read the sidecar for {csv_path.name}:\n\n{exc}\n\n"
                "The cycles will load, but saved parameters will not be "
                "available."
            )
            sidecar = None

        try:
            cycles = read_cycles_from_csv(csv_path)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to load",
                                 f"Could not read {csv_path}:\n\n{exc}")
            return

        if not cycles:
            QMessageBox.information(self, "No cycles found",
                                    f"{csv_path} contains no cycles.")
            return

        # Replace the in-memory cycle set with what was loaded. Loading
        # the same file twice produces the same state, not a doubled
        # list. This also discards any in-memory edits that weren't
        # saved -- which is the expected behavior for "Load" (as
        # opposed to "Import" or "Merge").
        regrouped: dict[int, list[ThetaCycle]] = {}
        for c in cycles:
            regrouped.setdefault(c.channel, []).append(c)
        for ch in regrouped:
            regrouped[ch].sort(key=lambda c: c.zero_crossing_start)
        self.cycles_by_channel = regrouped

        if sidecar is not None:
            self._apply_sidecar(sidecar)

        self._populate_table()
        self._push_flat_cycles_to_trace()
        self._refresh_export_buttons()
        status = f"Loaded {len(cycles)} cycle(s) from {folder}."
        if sidecar is not None:
            status += " Parameters restored."
        self.status_label.setText(status)
        self._autosave_armed = True

    def _apply_sidecar(self, sidecar: dict):
        """Restore detection params and the theta band from a sidecar.
        The band is pushed to the trace view so the filtered overlay
        matches the band the cycles were detected with."""
        params_dict = sidecar.get("params") or {}
        try:
            params = ThetaCycleParams(**params_dict)
        except TypeError as exc:
            print(f"[theta-cycle-load] could not build params: {exc}")
            return

        self._last_params_for_sidecar = params

        for spin, value in (
            (self.low_spin, params.low_freq),
            (self.high_spin, params.high_freq),
            (self.min_amp_spin, params.min_theta_peak_percentile),
            (self.delta_low_spin, params.delta_low_freq),
            (self.delta_ratio_spin, params.theta_delta_ratio_min),
            (self.fast_low_spin, params.fast_low_freq),
            (self.fast_ratio_spin, params.fast_theta_ratio_max),
        ):
            spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(False)

        if params.delta_high_freq is not None:
            self.delta_high_spin.blockSignals(True)
            self.delta_high_spin.setValue(float(params.delta_high_freq))
            self.delta_high_spin.blockSignals(False)

        self.delta_check.setChecked(bool(params.delta_correction))
        self.fast_check.setChecked(bool(params.fast_correction))
        self.edge_combo.setCurrentText(str(params.cycle_start_edge))

        # Restore the overlay's band so the filtered trace it draws
        # matches what the cycles were detected against.
        if self.trace_view is not None:
            self.trace_view.set_theta_cycle_band(
                params.low_freq, params.high_freq,
            )


    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if self._detect_thread is not None and self._detect_thread.isRunning():
            self._detect_thread.wait(2000)

        if self._autosave_armed and self._autosave_timer.isActive():
            self._autosave_timer.stop()
            self._autosave_now()

        try:
            self.trace_view.thetaCyclesChanged.disconnect(self.set_cycles_from_trace)
        except (TypeError, RuntimeError):
            pass
        try:
            self.trace_view.thetaCycleSelected.disconnect(self.set_selected_cycle)
        except (TypeError, RuntimeError):
            pass
        self.trace_view.clear_theta_cycles()
        super().closeEvent(event)



    def _refresh_export_buttons(self):
        """Enable Save only when there are cycles to write and the
        engine has a .dat to anchor the analysis folder to. Load stays
        enabled as long as there's a .dat, so the user can still try to
        load a saved session even with nothing detected yet (the
        handler shows a helpful message if the folder is empty)."""
        has_cycles = bool(
            self.cycles_by_channel
            and any(self.cycles_by_channel.values())
        )
        has_file = getattr(self.engine, "data_file", None) is not None

        self.export_btn.setEnabled(has_cycles and has_file)

        if has_file:
            folder = analysis_folder_for(self.engine)
            self.export_btn.setToolTip(
                f"Write all theta cycles and their parameters to:\n{folder}"
            )
            self.load_btn.setToolTip(
                f"Load theta cycles and their parameters from:\n{folder}"
            )
        else:
            self.export_btn.setToolTip(
                "No data file is loaded. Load a continuous.dat before "
                "saving or loading an analysis."
            )
            self.load_btn.setToolTip(
                "No data file is loaded. Load a continuous.dat before "
                "saving or loading an analysis."
            )

        # Load is never disabled by 'no cycles' -- the user may want to
        # load a saved session before running detection.
        self.load_btn.setEnabled(has_file)

    def _schedule_autosave(self):
        if not self._autosave_armed:
            return
        self._autosave_timer.start()

    def _autosave_now(self):
        if not self._autosave_armed:
            return
        flat = self._flat_cycles()
        if not flat:
            return
        folder = ensure_analysis_folder(self.engine)
        if folder is None:
            return

        csv_path = folder / "theta_cycles.csv"
        try:
            export_cycles_to_csv(flat, self.engine, csv_path)
        except Exception as exc:
            print(f"[theta-cycle-autosave] CSV write failed: {exc}")
            return

        n = int(getattr(self, "_detect_n_samples", 0))
        if n > 0 and self._last_params_for_sidecar is not None:
            try:
                save_sidecar(
                    csv_path=csv_path,
                    kind="theta_cycles",
                    sample_rate=float(self.engine.sr),
                    analysis_start_sample=int(self._sample_offset),
                    analysis_end_sample=int(self._sample_offset) + n,
                    channels=sorted(self.cycles_by_channel.keys()),
                    params=self._last_params_for_sidecar,
                )
            except Exception as exc:
                print(f"[theta-cycle-autosave] sidecar write failed: {exc}")

        self.status_label.setText(
            f"Auto-saved {len(flat)} cycle(s)."
        )
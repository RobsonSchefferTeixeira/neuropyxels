"""
ripple_dialog.py

Non-modal QDialog for sharp-wave ripple (SWR) detection: channel
selection, detection parameters, CSD toggle, and a per-channel summary
table (sorted by mean peak amplitude, so the strongest-ripple
channel/depth is immediately visible).

Redesigned from the original two-panel (controls + own pyqtgraph
inspector plot) layout: this dialog is now controls-only. Results are
rendered as an overlay directly on the main TraceViewWidget instead of a
second, independent plot -- for two reasons:
  1. Performance: the old inspector duplicated windowing/filtering/
     painting that TraceViewWidget already does, so every inspection
     paid for that pipeline twice.
  2. Correctness: TraceViewWidget owns the one time <-> pixel mapping
     that the vertical cursor line and the trace both agree on (after
     a couple of rounds fixing drift bugs there). A second, independent
     plot risks silently re-diverging from that mapping. Feeding
     results into TraceViewWidget's own paint pipeline means there's
     exactly one place where "where is time on screen" is decided.

Detection math lives in core/ripple_detector.py; CSD (optional) reuses
core/phase_amplitude.py's PhaseAmplitudeAnalyzer.compute_csd, the same
CSD implementation already used and tested there, rather than
duplicating it.

Selecting a row in the summary table (or Prev/Next Ripple) selects that
channel in the main trace view and pans/zooms it to center the ripple,
rather than redrawing a separate inspector.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QDoubleSpinBox,
    QSpinBox, QCheckBox, QPushButton, QComboBox, QMessageBox, QWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QLineEdit, QFileDialog, QSplitter,
)

from core.ripple_detector import RippleDetector, RippleParams, RippleEvent
from core.phase_amplitude import PhaseAmplitudeAnalyzer
from core.ripple_export import export_ripples_to_csv, export_ripples_per_channel
from core.trace_engine import TraceEngine
from gui.trace_view import TraceViewWidget


class _DetectThread(QThread):
    """Runs detection across all requested channels off the UI thread."""

    finished_ok = pyqtSignal(dict, dict)   # ({channel: list[RippleEvent]}, {channel: signal_used})
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, int)  # (channels_done, channels_total)

    def __init__(self, channels: list[int], raw_data, sample_rate: float,
                 start_time: float, end_time: float, params: RippleParams,
                 use_csd: bool, csd_spacing: float,
                 pac_analyzer: PhaseAmplitudeAnalyzer, parent=None):
        super().__init__(parent)
        self.channels = channels
        self.raw_data = raw_data
        self.sample_rate = sample_rate
        self.start_time = start_time
        self.end_time = end_time
        self.params = params
        self.use_csd = use_csd
        self.csd_spacing = csd_spacing
        self.pac_analyzer = pac_analyzer
        self.detector = RippleDetector()

    def run(self):
        try:
            # NOTE: this is plain elapsed-seconds indexing (int(time * sr)),
            # NOT timestamp-aware like TraceEngine.get_time_window_sample_range.
            # Detection results will be offset if run against a recording
            # where timestamps.npy introduces non-uniform sample spacing.
            # Flagged as a known limitation, not fixed here -- fixing it
            # means threading a TraceEngine reference (or its
            # get_time_window_sample_range) into this thread instead of
            # raw_data/sample_rate, which is a bigger change than this pass.
            start_idx = max(0, int(self.start_time * self.sample_rate))
            end_idx = min(self.raw_data.shape[0], int(self.end_time * self.sample_rate))
            if start_idx >= end_idx:
                raise ValueError("Invalid time range for detection.")

            results: dict[int, list[RippleEvent]] = {}
            signals_used: dict[int, np.ndarray] = {}
            for i, ch in enumerate(self.channels):
                signal = self.raw_data[start_idx:end_idx, ch].flatten().astype(np.float64)

                if self.use_csd:
                    signal, csd_info = self.pac_analyzer.compute_csd(
                        signal, ch, self.sample_rate, self.raw_data,
                        self.start_time, self.end_time, self.csd_spacing,
                    )
                    if not csd_info.get("used", False):
                        # CSD unavailable for this channel (edge channel,
                        # no valid neighbors) -- fall back to raw signal
                        # rather than silently dropping the channel.
                        signal = self.raw_data[start_idx:end_idx, ch].flatten().astype(np.float64)

                events = self.detector.detect(
                    signal, self.sample_rate, self.params, channel=ch
                )
                results[ch] = events
                signals_used[ch] = signal  # kept so the overlay can compute the filtered envelope
                self.progress.emit(i + 1, len(self.channels))

            self.finished_ok.emit(results, signals_used)
        except Exception as exc:
            self.failed.emit(str(exc))


class RippleDialog(QDialog):
    """
    Non-modal, controls-only ripple detection dialog. Detection results
    are pushed to a TraceViewWidget as an overlay rather than rendered
    in this dialog.
    """

    def __init__(self, probe_data: dict, engine: TraceEngine,
                 trace_view: TraceViewWidget,
                 initial_channels: list[int] | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ripple Detection")
        self.setModal(False)
        self.resize(560, 520)

        self.engine = engine
        self.trace_view = trace_view
        self.pac_analyzer = PhaseAmplitudeAnalyzer(probe_data)  # reused for CSD only
        self._detect_thread: _DetectThread | None = None
        self.detector = RippleDetector()

        # channel -> list[RippleEvent]; source of truth, also what gets
        # exported to CSV.
        self.events_by_channel: dict[int, list[RippleEvent]] = {}
        self._signal_by_channel: dict[int, np.ndarray] = {}  # raw or CSD signal actually analyzed
        self._envelope_by_channel: dict[int, np.ndarray] = {}
        self._sample_offset = 0  # samples (engine.sr basis), set at detect time
        self._selected_channel: int | None = None
        self._params: RippleParams | None = None  # params used for the CURRENT detection run

        self._build_ui()
        self._sync_time_range_from_engine()

        if initial_channels:
            self.channel_edit.setText(",".join(str(c) for c in initial_channels))

    # ------------------------------------------------------------------
    # UI scaffolding
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        layout.addLayout(self._build_param_grid())

        btn_row = QHBoxLayout()
        self.detect_btn = QPushButton("Detect Ripples")
        self.detect_btn.clicked.connect(self._on_detect_clicked)
        btn_row.addWidget(self.detect_btn)
        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color: #888; font-size: 10px;")
        btn_row.addWidget(self.progress_label, stretch=1)
        layout.addLayout(btn_row)

        layout.addWidget(self._build_summary_table(), stretch=1)

        layout.addLayout(self._build_overlay_controls())
        layout.addLayout(self._build_nav_row())
        layout.addLayout(self._build_export_row())

        self.status_label = QLabel(
            "Select channels and click Detect Ripples. Click a row to jump the main "
            "trace view to that channel."
        )
        self.status_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self.status_label)

    def _build_param_grid(self) -> QGridLayout:
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        col_pairs = []

        self.channel_edit = QLineEdit()
        self.channel_edit.setPlaceholderText("e.g. 12,13,14,15")
        col_pairs.append(("Channels (comma-sep)", self.channel_edit))

        self.start_spin = QDoubleSpinBox()
        self.start_spin.setDecimals(2)
        self.start_spin.setRange(0.0, 1e9)
        col_pairs.append(("Start (s)", self.start_spin))

        self.end_spin = QDoubleSpinBox()
        self.end_spin.setDecimals(2)
        self.end_spin.setRange(0.01, 1e9)
        self.end_spin.setValue(10.0)
        col_pairs.append(("End (s)", self.end_spin))

        self.low_freq_spin = QDoubleSpinBox()
        self.low_freq_spin.setRange(1.0, 2000.0)
        self.low_freq_spin.setValue(120.0)
        col_pairs.append(("Band low (Hz)", self.low_freq_spin))

        self.high_freq_spin = QDoubleSpinBox()
        self.high_freq_spin.setRange(2.0, 2000.0)
        self.high_freq_spin.setValue(250.0)
        col_pairs.append(("Band high (Hz)", self.high_freq_spin))

        self.envelope_combo = QComboBox()
        self.envelope_combo.addItems(["hilbert", "rms"])
        col_pairs.append(("Envelope", self.envelope_combo))

        self.peak_sd_spin = QDoubleSpinBox()
        self.peak_sd_spin.setRange(0.5, 20.0)
        self.peak_sd_spin.setValue(4.0)
        col_pairs.append(("Peak threshold (SD)", self.peak_sd_spin))

        self.boundary_sd_spin = QDoubleSpinBox()
        self.boundary_sd_spin.setRange(0.1, 20.0)
        self.boundary_sd_spin.setValue(1.0)
        col_pairs.append(("Boundary threshold (SD)", self.boundary_sd_spin))

        self.min_dur_spin = QDoubleSpinBox()
        self.min_dur_spin.setRange(0.0, 5000.0)
        self.min_dur_spin.setValue(15.0)
        self.min_dur_spin.setSpecialValueText("off")
        col_pairs.append(("Min duration (ms)", self.min_dur_spin))

        self.max_dur_spin = QDoubleSpinBox()
        self.max_dur_spin.setRange(0.0, 5000.0)
        self.max_dur_spin.setValue(250.0)
        self.max_dur_spin.setSpecialValueText("off")
        col_pairs.append(("Max duration (ms)", self.max_dur_spin))

        self.csd_check = QCheckBox("Use CSD")
        col_pairs.append((None, self.csd_check))

        self.csd_spacing_spin = QDoubleSpinBox()
        self.csd_spacing_spin.setRange(1.0, 500.0)
        self.csd_spacing_spin.setValue(20.0)
        col_pairs.append(("CSD spacing (\u00b5m)", self.csd_spacing_spin))

        n_cols = 4
        row = 0
        col = 0
        for label_text, widget in col_pairs:
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

    def _build_summary_table(self) -> QTableWidget:
        self.summary_table = QTableWidget(0, 4)
        self.summary_table.setHorizontalHeaderLabels(
            ["Channel", "N Ripples", "Mean Peak Amp.", "Max Peak Amp."]
        )
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.summary_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.summary_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.summary_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.summary_table.itemSelectionChanged.connect(self._on_table_selection_changed)
        return self.summary_table

    def _build_overlay_controls(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(QLabel("Overlay on main trace view:"))

        self.show_envelope_check = QCheckBox("Envelope")
        self.show_envelope_check.setChecked(True)
        self.show_envelope_check.toggled.connect(self._on_overlay_visibility_changed)
        row.addWidget(self.show_envelope_check)

        self.show_thresholds_check = QCheckBox("Thresholds")
        self.show_thresholds_check.setChecked(True)
        self.show_thresholds_check.toggled.connect(self._on_overlay_visibility_changed)
        row.addWidget(self.show_thresholds_check)

        self.show_events_check = QCheckBox("Events")
        self.show_events_check.setChecked(True)
        self.show_events_check.toggled.connect(self._on_overlay_visibility_changed)
        row.addWidget(self.show_events_check)

        row.addStretch(1)
        return row

    def _build_nav_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.prev_ripple_btn = QPushButton("\u25c4 Prev Ripple")
        self.prev_ripple_btn.clicked.connect(self._go_to_prev_ripple)
        row.addWidget(self.prev_ripple_btn)

        self.next_ripple_btn = QPushButton("Next Ripple \u25ba")
        self.next_ripple_btn.clicked.connect(self._go_to_next_ripple)
        row.addWidget(self.next_ripple_btn)

        row.addWidget(QLabel("Window (s):"))
        self.nav_window_spin = QDoubleSpinBox()
        self.nav_window_spin.setDecimals(2)
        self.nav_window_spin.setRange(0.05, 30.0)
        self.nav_window_spin.setValue(2.0)
        row.addWidget(self.nav_window_spin)

        row.addStretch(1)
        return row

    def _build_export_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.export_selected_btn = QPushButton("Export Selected Channel CSV...")
        self.export_selected_btn.clicked.connect(self._on_export_selected_clicked)
        row.addWidget(self.export_selected_btn)

        self.export_all_btn = QPushButton("Export All Channels...")
        self.export_all_btn.clicked.connect(self._on_export_all_clicked)
        row.addWidget(self.export_all_btn)

        row.addStretch(1)
        return row

    def _sync_time_range_from_engine(self):
        if self.engine.data_loaded:
            self.start_spin.setRange(0.0, self.engine.total_duration)
            self.end_spin.setRange(0.01, self.engine.total_duration)
            self.end_spin.setValue(min(10.0, self.engine.total_duration))

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _parse_channels(self) -> list[int]:
        text = self.channel_edit.text().strip()
        if not text:
            return []
        channels = []
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                channels.append(int(part))
            except ValueError:
                raise ValueError(f"'{part}' is not a valid channel number.")
        return channels

    def _collect_params(self) -> RippleParams:
        return RippleParams(
            low_freq=self.low_freq_spin.value(),
            high_freq=self.high_freq_spin.value(),
            envelope_method=self.envelope_combo.currentText(),
            peak_threshold_sd=self.peak_sd_spin.value(),
            boundary_threshold_sd=self.boundary_sd_spin.value(),
            min_duration_ms=None if self.min_dur_spin.value() == 0.0 else self.min_dur_spin.value(),
            max_duration_ms=None if self.max_dur_spin.value() == 0.0 else self.max_dur_spin.value(),
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
            QMessageBox.warning(self, "No channels", "Enter at least one channel to detect on.")
            return

        if self.start_spin.value() >= self.end_spin.value():
            QMessageBox.warning(self, "Invalid range", "Start time must be before end time.")
            return

        if self.low_freq_spin.value() >= self.high_freq_spin.value():
            QMessageBox.warning(self, "Invalid band", "Band low must be below band high.")
            return

        params = self._collect_params()

        self.detect_btn.setEnabled(False)
        self.detect_btn.setText("Detecting...")
        self.status_label.setText("Running detection, please wait...")

        # Sample offset for absolute-position export/overlay: samples from
        # the start of the recording to the start of the analyzed window.
        self._sample_offset = int(self.start_spin.value() * self.engine.sr)

        self._detect_thread = _DetectThread(
            channels, self.engine.data, self.engine.sr,
            self.start_spin.value(), self.end_spin.value(), params,
            self.csd_check.isChecked(), self.csd_spacing_spin.value(),
            self.pac_analyzer, parent=self,
        )
        self._detect_thread.progress.connect(self._on_detect_progress)
        self._detect_thread.finished_ok.connect(self._on_detect_finished)
        self._detect_thread.failed.connect(self._on_detect_failed)
        self._detect_thread.start()

    def _on_detect_progress(self, done: int, total: int):
        self.progress_label.setText(f"{done}/{total} channels")

    def _on_detect_failed(self, message: str):
        self.detect_btn.setEnabled(True)
        self.detect_btn.setText("Detect Ripples")
        self.progress_label.setText("")
        self.status_label.setText(f"Error: {message}")
        QMessageBox.critical(self, "Detection failed", message)

    def _on_detect_finished(self, results: dict, signals_used: dict):
        self.detect_btn.setEnabled(True)
        self.detect_btn.setText("Detect Ripples")
        self.progress_label.setText("")
        self.events_by_channel = results
        self._signal_by_channel = signals_used
        self._params = self._collect_params()

        # Compute the filtered envelope for EVERY detected channel up
        # front (not lazily on row-select like the old inspector did) --
        # the overlay can show several channels at once since it rides
        # on whatever's currently visible in the main trace view, not on
        # a single selected channel.
        self._envelope_by_channel = {}
        for ch, signal in signals_used.items():
            filtered = self.detector.apply_ripple_filter(signal, self.engine.sr, self._params)
            self._envelope_by_channel[ch] = self.detector.compute_envelope(
                filtered, self.engine.sr, self._params
            )

        self._populate_summary_table()
        self._push_overlay_to_trace_view()

        total_events = sum(len(v) for v in results.values())
        self.status_label.setText(
            f"Detected {total_events} ripple(s) across {len(results)} channel(s). "
            "Click a row to jump the main trace view to that channel."
        )

    # ------------------------------------------------------------------
    # Overlay: pushing results to the main TraceViewWidget
    # ------------------------------------------------------------------

    def _push_overlay_to_trace_view(self):
        if not self.events_by_channel:
            self.trace_view.clear_ripple_overlay()
            return

        channels_overlay = {}
        for ch, events in self.events_by_channel.items():
            envelope = self._envelope_by_channel.get(ch)
            signal = self._signal_by_channel.get(ch)
            if envelope is None or signal is None:
                continue
            channels_overlay[ch] = {
                'envelope': envelope,
                'signal': signal,
                'sample_offset': self._sample_offset,
                'sample_rate': self.engine.sr,
                'events': events,
                'peak_threshold_sd': self._params.peak_threshold_sd if self._params else None,
                'boundary_threshold_sd': self._params.boundary_threshold_sd if self._params else None,
                # Ripple-band edges used for THIS detection run. Exposed so
                # TraceViewWidget can tell whether the user's own live
                # bandpass filter on the main trace matches the band the
                # ripples were detected in -- if so, it recomputes the
                # envelope from the currently-filtered visible trace
                # instead of this stored, detection-time-only envelope, so
                # what's drawn actually tracks the filtered signal on
                # screen rather than a frozen snapshot from detect time.
                'low_freq': self._params.low_freq if self._params else None,
                'high_freq': self._params.high_freq if self._params else None,
                'envelope_method': self._params.envelope_method if self._params else None,
            }

        self.trace_view.set_ripple_overlay(
            {'channels': channels_overlay},
            show_envelope=self.show_envelope_check.isChecked(),
            show_thresholds=self.show_thresholds_check.isChecked(),
            show_events=self.show_events_check.isChecked(),
        )

    def _on_overlay_visibility_changed(self, _checked: bool):
        self.trace_view.set_ripple_overlay_visibility(
            show_envelope=self.show_envelope_check.isChecked(),
            show_thresholds=self.show_thresholds_check.isChecked(),
            show_events=self.show_events_check.isChecked(),
        )

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------

    def _populate_summary_table(self):
        # Sort channels by mean peak amplitude descending, so the
        # strongest-ripple channel/depth is immediately visible at the top.
        def mean_peak(events: list[RippleEvent]) -> float:
            if not events:
                return -1.0  # channels with zero events sort to the bottom
            return float(np.mean([e.peak_amplitude for e in events]))

        ordered_channels = sorted(
            self.events_by_channel.keys(),
            key=lambda ch: mean_peak(self.events_by_channel[ch]),
            reverse=True,
        )

        self.summary_table.setRowCount(len(ordered_channels))
        for row, ch in enumerate(ordered_channels):
            events = self.events_by_channel[ch]
            n = len(events)
            mean_amp = np.mean([e.peak_amplitude for e in events]) if n > 0 else 0.0
            max_amp = np.max([e.peak_amplitude for e in events]) if n > 0 else 0.0

            self.summary_table.setItem(row, 0, QTableWidgetItem(str(ch)))
            self.summary_table.setItem(row, 1, QTableWidgetItem(str(n)))
            self.summary_table.setItem(row, 2, QTableWidgetItem(f"{mean_amp:.3g}"))
            self.summary_table.setItem(row, 3, QTableWidgetItem(f"{max_amp:.3g}"))
            # Stash the channel number on the row so selection lookup
            # doesn't depend on re-parsing the displayed text.
            self.summary_table.item(row, 0).setData(Qt.ItemDataRole.UserRole, ch)

    def _on_table_selection_changed(self):
        selected = self.summary_table.selectedItems()
        if not selected:
            self._selected_channel = None
            return
        row = selected[0].row()
        channel_item = self.summary_table.item(row, 0)
        self._selected_channel = channel_item.data(Qt.ItemDataRole.UserRole)
        n_events = len(self.events_by_channel.get(self._selected_channel, []))
        self.status_label.setText(
            f"Channel {self._selected_channel} selected ({n_events} ripples)."
        )
        self._show_channel_in_trace_view(self._selected_channel)

    # ------------------------------------------------------------------
    # Driving the main trace view
    # ------------------------------------------------------------------

    def _show_channel_in_trace_view(self, channel: int):
        """Make sure the selected channel is visible in the main trace
        view (so the overlay actually renders for it), without
        disturbing the current time window unless the channel has no
        events -- jumping straight to a ripple is handled separately by
        _go_to_prev_ripple/_go_to_next_ripple."""
        if channel not in self.trace_view.channels:
            channels = list(self.trace_view.channels)
            channels.append(channel)
            self.trace_view.set_channels(channels)
        else:
            self.trace_view.update()

    def _go_to_prev_ripple(self):
        self._step_ripple(direction=-1)

    def _go_to_next_ripple(self):
        self._step_ripple(direction=1)

    def _step_ripple(self, direction: int):
        if self._selected_channel is None:
            QMessageBox.information(
                self, "No channel selected",
                "Select a channel row in the table first."
            )
            return
        events = self.events_by_channel.get(self._selected_channel, [])
        if not events:
            return

        self._show_channel_in_trace_view(self._selected_channel)

        sr = self.engine.sr
        offset_time = self._sample_offset / sr if sr else 0.0
        current_center_time = self.trace_view.start_time + self.trace_view.window_duration / 2
        # Events' peak_sample is relative to the analyzed segment
        # (offset by self._sample_offset), so compare in absolute time.
        current_center_sample = (current_center_time - offset_time) * sr

        if direction > 0:
            candidates = [e for e in events if e.peak_sample > current_center_sample + 1]
            target = candidates[0] if candidates else events[-1]
        else:
            candidates = [e for e in events if e.peak_sample < current_center_sample - 1]
            target = candidates[-1] if candidates else events[0]

        window_duration = max(0.05, self.nav_window_spin.value())
        target_time = offset_time + target.peak_sample / sr
        new_start = max(0.0, target_time - window_duration / 2)

        self.trace_view.window_duration = window_duration
        self.trace_view.start_time = new_start
        self.trace_view.control_panel.set_duration(window_duration)
        self.trace_view._update_time_labels()
        self.trace_view._invalidate_cache()
        self.trace_view.update()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _on_export_selected_clicked(self):
        if self._selected_channel is None:
            QMessageBox.information(
                self, "No channel selected",
                "Select a channel row in the table first."
            )
            return
        events = self.events_by_channel.get(self._selected_channel, [])
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Export Ripples to CSV",
            f"ripples_channel{self._selected_channel}.csv",
            "CSV files (*.csv);;All files (*)"
        )
        if not path_str:
            return
        try:
            export_ripples_to_csv(events, self.engine.sr, self._sample_offset, path_str)
            self.status_label.setText(f"Exported {len(events)} event(s) to {path_str}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    def _on_export_all_clicked(self):
        if not self.events_by_channel:
            QMessageBox.information(self, "Nothing to export", "Run detection first.")
            return
        dir_str = QFileDialog.getExistingDirectory(self, "Export All Channels To Folder")
        if not dir_str:
            return
        try:
            written = export_ripples_per_channel(
                self.events_by_channel, self.engine.sr, self._sample_offset, dir_str
            )
            self.status_label.setText(f"Exported {len(written)} file(s) to {dir_str}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if self._detect_thread is not None and self._detect_thread.isRunning():
            self._detect_thread.wait(2000)
        self.trace_view.clear_ripple_overlay()
        super().closeEvent(event)
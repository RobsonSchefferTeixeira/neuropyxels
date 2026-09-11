"""
theta_epoch_dialog.py

Non-modal QDialog for theta epoch detection. The signal itself is displayed
only in the main TraceView; this dialog contains analysis parameters and the
results table.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QDoubleSpinBox,
    QPushButton, QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QLineEdit, QFileDialog, QProgressBar,
)

from core.theta_epoch_detector import ThetaEpochDetector, ThetaEpochParams, ThetaEpoch
from core.trace_engine import TraceEngine
from core.filters import bandpass_filter, notch_filter


class _DetectThread(QThread):
    """Runs theta epoch detection off the UI thread."""

    finished_ok = pyqtSignal(list, dict)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, int)

    def __init__(
        self,
        channels: list[int],
        raw_data,
        sample_rate: float,
        start_time: float,
        end_time: float,
        params: ThetaEpochParams,
        timestamps=None,
        filter_settings=None,
        parent=None,
    ):
        super().__init__(parent)
        self.channels = channels
        self.raw_data = raw_data
        self.sample_rate = float(sample_rate)
        self.start_time = float(start_time)
        self.end_time = float(end_time)
        self.params = params
        self.timestamps = None if timestamps is None else np.asarray(timestamps)
        self.filter_settings = dict(filter_settings or {})
        self.detector = ThetaEpochDetector()
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True

    @property
    def cancel_requested(self):
        return self._cancel_requested

    def _time_to_indices(self):
        if self.timestamps is not None and len(self.timestamps):
            start_idx = int(np.searchsorted(self.timestamps, self.start_time, side="left"))
            end_idx = int(np.searchsorted(self.timestamps, self.end_time, side="right"))
            start_idx = max(0, min(start_idx, len(self.timestamps) - 1))
            end_idx = max(start_idx + 1, min(end_idx, len(self.timestamps)))
            actual_start = float(self.timestamps[start_idx])
            actual_end = float(self.timestamps[min(end_idx - 1, len(self.timestamps) - 1)])
            return start_idx, end_idx, actual_start, actual_end

        start_idx = max(0, int(np.floor(self.start_time * self.sample_rate)))
        end_idx = min(self.raw_data.shape[0], int(np.ceil(self.end_time * self.sample_rate)))
        return start_idx, end_idx, start_idx / self.sample_rate, max(start_idx, end_idx - 1) / self.sample_rate

    def _apply_trace_view_filters(self, signal: np.ndarray) -> np.ndarray:
        """Apply the same trace-view preprocessing that was active when detection started."""
        settings = self.filter_settings
        if not settings.get("bandpass_enabled", False) and not settings.get("notch_enabled", False) and not settings.get("detrend", False):
            return signal

        data = np.asarray(signal, dtype=np.float64).copy()

        if settings.get("detrend", False):
            from scipy.signal import detrend as scipy_detrend
            data = scipy_detrend(data, axis=0)

        if settings.get("bandpass_enabled", False):
            nyquist = self.sample_rate / 2.0
            low = max(0.0, min(float(settings.get("low_freq", 1.0)), nyquist - 1.0))
            high = max(0.0, min(float(settings.get("high_freq", 300.0)), nyquist - 1.0))
            if low > 0 and high > 0:
                if low >= high:
                    low, high = min(low, high), max(low, high)
                data = bandpass_filter(data, self.sample_rate, low, high, order=4)
            elif high > 0:
                data = bandpass_filter(data, self.sample_rate, 0.0, high, order=4)
            elif low > 0:
                data = bandpass_filter(data, self.sample_rate, low, 0.0, order=4)

        if settings.get("notch_enabled", False):
            data = notch_filter(
                data, self.sample_rate, float(settings.get("notch_freq", 50.0)), quality_factor=30.0
            )

        return data

    def run(self):
        try:
            start_idx, end_idx, actual_start, actual_end = self._time_to_indices()
            if start_idx >= end_idx:
                raise ValueError("Invalid time range for detection.")

            all_epochs = []
            for i, ch in enumerate(self.channels):
                if self.cancel_requested:
                    self.finished_ok.emit([], {"cancelled": True})
                    return
                if ch < 0 or ch >= self.raw_data.shape[1]:
                    raise ValueError(f"Channel {ch} is outside the loaded data.")
                signal = self.raw_data[start_idx:end_idx, ch].flatten().astype(np.float64)
                signal = self._apply_trace_view_filters(signal)
                epochs = self.detector.detect(
                    signal, self.sample_rate, self.params, channel=ch
                )
                all_epochs.extend(epochs)
                self.progress.emit(i + 1, len(self.channels))

            if self.cancel_requested:
                self.finished_ok.emit([], {"cancelled": True})
                return

            self.finished_ok.emit(
                all_epochs,
                {
                    "sample_offset": start_idx,
                    "start_time": actual_start,
                    "end_time": actual_end,
                },
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class ThetaEpochDialog(QDialog):
    epochsChanged = pyqtSignal(object)
    mergeGapChanged = pyqtSignal(float)
    thetaEpochSelected = pyqtSignal(int)

    def __init__(
        self,
        probe_data: dict,
        engine: TraceEngine,
        initial_channels: list[int] | None = None,
        trace_view=None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Theta Epoch Detection")
        self.setModal(False)
        self.resize(900, 620)

        self.engine = engine
        self.trace_view = trace_view
        self._detect_thread: _DetectThread | None = None
        self.epochs: list[ThetaEpoch] = []
        self._detection_start_time = 0.0
        self._sample_offset = 0
        self._available_min_time, self._available_max_time = self._get_available_time_range()

        self._build_ui()

        if initial_channels:
            self.channel_edit.setText(",".join(str(c) for c in initial_channels))

        self._set_full_available_range()

    def _get_available_time_range(self):
        if (
            getattr(self.engine, "timestamps_loaded", False)
            and getattr(self.engine, "timestamps", None) is not None
            and len(self.engine.timestamps)
        ):
            return float(self.engine.timestamps[0]), float(self.engine.timestamps[-1])
        return 0.0, float(getattr(self.engine, "total_duration", 0.0))

    def _set_full_available_range(self):
        self.start_spin.setValue(self._available_min_time)
        self.end_spin.setValue(self._available_max_time)
        self.range_info_label.setText(
            f"Available: {self._available_min_time:.3f} – {self._available_max_time:.3f} s"
        )

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.addLayout(self._build_param_grid())

        btn_row = QHBoxLayout()
        self.detect_btn = QPushButton("Detect Theta Epochs")
        self.detect_btn.clicked.connect(self._on_detect_clicked)
        btn_row.addWidget(self.detect_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel_clicked)
        btn_row.addWidget(self.cancel_btn)

        self.export_btn = QPushButton("Export CSV")
        self.export_btn.clicked.connect(self._on_export_clicked)
        self.export_btn.setEnabled(False)
        btn_row.addWidget(self.export_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels([
            "Channel", "Start (s)", "End (s)", "Duration (ms)",
            "Mean Ratio", "Peak Ratio", "Theta Power"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self.table.cellDoubleClicked.connect(lambda row, _col: self.thetaEpochSelected.emit(row))
        layout.addWidget(self.table, stretch=1)

        self.status_label = QLabel("Select channels and click Detect.")
        self.status_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        layout.addWidget(self.progress_bar)

    def _build_param_grid(self) -> QGridLayout:
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)

        self.channel_edit = QLineEdit()
        self.channel_edit.setPlaceholderText("e.g. 12,13,14")
        grid.addWidget(QLabel("Channels:"), 0, 0)
        grid.addWidget(self.channel_edit, 0, 1)

        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(-1e12, 1e12)
        self.start_spin.setDecimals(3)
        grid.addWidget(QLabel("Start (s):"), 0, 2)
        grid.addWidget(self.start_spin, 0, 3)

        self.end_spin = QDoubleSpinBox()
        self.end_spin.setRange(-1e12, 1e12)
        self.end_spin.setDecimals(3)
        grid.addWidget(QLabel("End (s):"), 0, 4)
        grid.addWidget(self.end_spin, 0, 5)

        self.full_range_btn = QPushButton("Full available range")
        self.full_range_btn.setToolTip(
            "Set the analysis interval to the minimum and maximum available time."
        )
        self.full_range_btn.clicked.connect(self._set_full_available_range)
        grid.addWidget(self.full_range_btn, 1, 4, 1, 2)

        self.range_info_label = QLabel("")
        self.range_info_label.setStyleSheet("color: #888; font-size: 9px;")
        grid.addWidget(self.range_info_label, 1, 0, 1, 4)

        self.theta_low_spin = QDoubleSpinBox(); self.theta_low_spin.setRange(1.0, 50.0); self.theta_low_spin.setValue(4.0)
        grid.addWidget(QLabel("Theta low (Hz):"), 2, 0); grid.addWidget(self.theta_low_spin, 2, 1)
        self.theta_high_spin = QDoubleSpinBox(); self.theta_high_spin.setRange(2.0, 50.0); self.theta_high_spin.setValue(12.0)
        grid.addWidget(QLabel("Theta high (Hz):"), 2, 2); grid.addWidget(self.theta_high_spin, 2, 3)
        self.delta_low_spin = QDoubleSpinBox(); self.delta_low_spin.setRange(0.1, 10.0); self.delta_low_spin.setValue(1.0)
        grid.addWidget(QLabel("Delta low (Hz):"), 2, 4); grid.addWidget(self.delta_low_spin, 2, 5)
        self.delta_high_spin = QDoubleSpinBox(); self.delta_high_spin.setRange(0.5, 20.0); self.delta_high_spin.setValue(4.0)
        grid.addWidget(QLabel("Delta high (Hz):"), 3, 0); grid.addWidget(self.delta_high_spin, 3, 1)
        self.ratio_spin = QDoubleSpinBox(); self.ratio_spin.setRange(0.1, 20.0); self.ratio_spin.setValue(1.5)
        grid.addWidget(QLabel("Ratio threshold:"), 3, 2); grid.addWidget(self.ratio_spin, 3, 3)
        self.min_dur_spin = QDoubleSpinBox(); self.min_dur_spin.setRange(10.0, 5000.0); self.min_dur_spin.setValue(500.0); self.min_dur_spin.setSuffix(" ms")
        grid.addWidget(QLabel("Min duration:"), 3, 4); grid.addWidget(self.min_dur_spin, 3, 5)

        self.merge_gap_spin = QDoubleSpinBox(); self.merge_gap_spin.setRange(0.0, 5000.0); self.merge_gap_spin.setValue(500.0); self.merge_gap_spin.setSuffix(" ms")
        self.merge_gap_spin.setToolTip("Epochs separated by this gap or less are merged automatically and during manual editing.")
        self.merge_gap_spin.valueChanged.connect(self.mergeGapChanged.emit)
        grid.addWidget(QLabel("Minimum merge gap:"), 4, 0); grid.addWidget(self.merge_gap_spin, 4, 1)

        self.power_window_spin = QDoubleSpinBox(); self.power_window_spin.setRange(50.0, 10000.0); self.power_window_spin.setSingleStep(50.0); self.power_window_spin.setValue(1000.0); self.power_window_spin.setSuffix(" ms")
        self.power_window_spin.setToolTip("Window used to average instantaneous theta/delta power.")
        grid.addWidget(QLabel("Power window:"), 4, 2); grid.addWidget(self.power_window_spin, 4, 3)

        self.power_overlap_spin = QDoubleSpinBox(); self.power_overlap_spin.setRange(0.0, 95.0); self.power_overlap_spin.setSingleStep(5.0); self.power_overlap_spin.setValue(50.0); self.power_overlap_spin.setSuffix(" %")
        self.power_overlap_spin.setToolTip("Overlap between consecutive power-analysis windows.")
        grid.addWidget(QLabel("Power overlap:"), 4, 4); grid.addWidget(self.power_overlap_spin, 4, 5)

        self.delete_btn = QPushButton("Delete Selected Epoch")
        self.delete_btn.setEnabled(False)
        self.delete_btn.clicked.connect(self._delete_selected_epoch)
        grid.addWidget(self.delete_btn, 5, 0, 1, 2)
        return grid

    def _on_detect_clicked(self):
        if not self.engine.data_loaded:
            QMessageBox.warning(self, "No data", "No data loaded.")
            return
        try:
            channels = [int(c.strip()) for c in self.channel_edit.text().split(",") if c.strip()]
        except ValueError:
            QMessageBox.warning(self, "Invalid channels", "Enter valid channel numbers.")
            return
        if not channels:
            QMessageBox.warning(self, "No channels", "Enter at least one channel.")
            return
        if self.end_spin.value() <= self.start_spin.value():
            QMessageBox.warning(self, "Invalid time range", "End time must be greater than start time.")
            return

        params = ThetaEpochParams(
            theta_low=self.theta_low_spin.value(),
            theta_high=self.theta_high_spin.value(),
            delta_low=self.delta_low_spin.value(),
            delta_high=self.delta_high_spin.value(),
            ratio_threshold=self.ratio_spin.value(),
            min_duration_ms=self.min_dur_spin.value(),
            merge_gap_ms=self.merge_gap_spin.value(),
            power_window_ms=self.power_window_spin.value(),
            power_overlap=self.power_overlap_spin.value() / 100.0,
        )

        timestamps = None
        if getattr(self.engine, "timestamps_loaded", False) and getattr(self.engine, "timestamps", None) is not None:
            timestamps = np.asarray(self.engine.timestamps)

        self.detect_btn.setEnabled(False)
        self.detect_btn.setText("Detecting...")
        self.cancel_btn.setEnabled(True)
        self.full_range_btn.setEnabled(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label.setText("Running detection...")
        filter_settings = {}
        if self.trace_view is not None:
            filter_settings = {
                "bandpass_enabled": bool(getattr(self.trace_view, "filter_enabled", False)),
                "low_freq": float(getattr(self.trace_view, "filter_low_freq", 1.0)),
                "high_freq": float(getattr(self.trace_view, "filter_high_freq", 300.0)),
                "notch_enabled": bool(getattr(self.trace_view, "notch_enabled", False)),
                "notch_freq": float(getattr(self.trace_view, "notch_freq", 50.0)),
                "detrend": bool(getattr(self.trace_view, "detrend_enabled", False)),
            }

        self._detect_thread = _DetectThread(
            channels, self.engine.data, self.engine.sr,
            self.start_spin.value(), self.end_spin.value(), params,
            timestamps=timestamps, filter_settings=filter_settings, parent=self,
        )
        self._detect_thread.progress.connect(self._on_detection_progress)
        self._detect_thread.finished_ok.connect(self._on_detect_finished)
        self._detect_thread.failed.connect(self._on_detect_failed)
        self._detect_thread.start()

    def _on_detection_progress(self, completed: int, total: int):
        percent = int(round(100.0 * completed / max(1, total)))
        self.progress_bar.setValue(percent)
        self.status_label.setText(f"Running detection... channel {completed}/{total}")

    def _on_cancel_clicked(self):
        if self._detect_thread is not None and self._detect_thread.isRunning():
            self.cancel_btn.setEnabled(False)
            self.status_label.setText("Cancelling detection...")
            self._detect_thread.request_cancel()

    def _reset_detection_controls(self):
        self.detect_btn.setEnabled(True)
        self.detect_btn.setText("Detect Theta Epochs")
        self.cancel_btn.setEnabled(False)
        self.full_range_btn.setEnabled(True)

    def _on_detect_failed(self, message: str):
        self._reset_detection_controls()
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Error: {message}")

    def _on_detect_finished(self, epochs: list, info: dict):
        cancelled = bool(info.get("cancelled", False))
        self._reset_detection_controls()
        if cancelled:
            self.progress_bar.setValue(0)
            self.status_label.setText("Detection cancelled.")
            return
        self.progress_bar.setValue(100)
        self.epochs = sorted(list(epochs), key=lambda e: (e.channel, e.start_sample))
        self._detection_start_time = float(info.get("start_time", self.start_spin.value()))
        self._sample_offset = int(info.get("sample_offset", 0))
        self.start_spin.setValue(self._detection_start_time)
        self._populate_table()
        self.export_btn.setEnabled(bool(self.epochs))
        self.epochsChanged.emit(self.epochs)
        self.status_label.setText(f"Found {len(self.epochs)} theta epochs.")

    def set_epochs_from_trace(self, epochs: list[ThetaEpoch]):
        self.epochs = sorted(list(epochs), key=lambda e: (e.channel, e.start_sample))
        self._populate_table()
        self.export_btn.setEnabled(bool(self.epochs))
        self.status_label.setText(f"{len(self.epochs)} theta epochs.")

    def set_selected_epoch(self, index: int):
        if 0 <= index < self.table.rowCount():
            self.table.selectRow(index)

    def _on_table_selection_changed(self):
        row = self.table.currentRow()
        self.delete_btn.setEnabled(row >= 0)
        if row >= 0:
            self.thetaEpochSelected.emit(row)

    def _delete_selected_epoch(self):
        row = self.table.currentRow()
        if not (0 <= row < len(self.epochs)):
            return
        self.epochs.pop(row)
        self._populate_table()
        self.export_btn.setEnabled(bool(self.epochs))
        self.epochsChanged.emit(self.epochs)
        self.status_label.setText(f"{len(self.epochs)} theta epochs.")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete and self.delete_btn.isEnabled():
            self._delete_selected_epoch()
            event.accept()
            return
        super().keyPressEvent(event)

    def _sample_to_display_time(self, sample: int) -> float:
        global_sample = self._sample_offset + int(sample)
        if getattr(self.engine, "timestamps_loaded", False) and getattr(self.engine, "timestamps", None) is not None:
            ts = self.engine.timestamps
            global_sample = max(0, min(global_sample, len(ts) - 1))
            return float(ts[global_sample])
        return self._detection_start_time + int(sample) / float(self.engine.sr)

    def _populate_table(self):
        self.table.blockSignals(True)
        try:
            self.table.setRowCount(len(self.epochs))
            self.table.clearSelection()
            self.delete_btn.setEnabled(False)
            for row, epoch in enumerate(self.epochs):
                start_time = self._sample_to_display_time(epoch.start_sample)
                end_time = self._sample_to_display_time(epoch.end_sample)
                values = [
                    str(epoch.channel), f"{start_time:.3f}", f"{end_time:.3f}",
                    f"{epoch.duration_ms:.1f}", f"{epoch.mean_ratio:.3f}",
                    f"{epoch.peak_ratio:.3f}", f"{epoch.mean_theta_power:.4f}",
                ]
                for col, value in enumerate(values):
                    self.table.setItem(row, col, QTableWidgetItem(value))
        finally:
            self.table.blockSignals(False)

    def _on_export_clicked(self):
        if not self.epochs:
            return
        path_str, _ = QFileDialog.getSaveFileName(self, "Export Theta Epochs", "", "CSV files (*.csv);;All files (*)")
        if not path_str:
            return
        from core.theta_epoch_export import export_theta_epochs_to_csv
        export_theta_epochs_to_csv(self.epochs, self.engine.sr, self._sample_offset, path_str)
        self.status_label.setText(f"Exported to {path_str}")

    def closeEvent(self, event):
        if self._detect_thread is not None and self._detect_thread.isRunning():
            self._detect_thread.request_cancel()
            self._detect_thread.wait(2000)
        super().closeEvent(event)

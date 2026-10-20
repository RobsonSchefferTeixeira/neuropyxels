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
from core.analysis_folder import (
    analysis_folder_for, ensure_analysis_folder,
)
from core.session_sidecar import save_sidecar, load_sidecar

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
        start_idx: int,
        end_idx: int,
        params: ThetaEpochParams,
        filter_settings=None,
        parent=None,
    ):
        super().__init__(parent)
        self.channels = channels
        self.raw_data = raw_data
        self.sample_rate = float(sample_rate)
        # Absolute sample indices, not times. The dialog converts the
        # spinbox's time-base values to samples (timestamp-aware) before
        # spawning the thread, so the thread never has to guess which
        # time basis the caller is working in.
        self.start_idx = int(start_idx)
        self.end_idx = int(end_idx)
        self.params = params
        self.filter_settings = dict(filter_settings or {})
        self.detector = ThetaEpochDetector()
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True

    @property
    def cancel_requested(self):
        return self._cancel_requested



    def _time_to_indices(self):
        """The dialog has already resolved the analysis window to
        absolute sample indices. Clip to the recording and return them
        along with their elapsed-time equivalents (used only for the
        info dict's start_time/end_time fields, which the dialog
        consumes for display)."""
        start_idx = max(0, self.start_idx)
        end_idx = min(self.raw_data.shape[0], self.end_idx)
        return (
            start_idx,
            end_idx,
            start_idx / self.sample_rate,
            max(start_idx, end_idx - 1) / self.sample_rate,
        )
    
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
                    "n_samples": int(end_idx - start_idx),
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
    eventFocused = pyqtSignal(int)   # channel id; the user clicked a row
    
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
        self._available_min_time = 0.0
        self._available_max_time = 0.0
        self._detect_n_samples = 0
        self._suppress_pan_on_select = False

        self._build_ui()
        self._refresh_export_buttons()

        if initial_channels:
            self.channel_edit.setText(",".join(str(c) for c in initial_channels))

        self._sync_time_range_from_engine()
        self._set_default_range()

    def _get_available_time_range(self):
        """Return (t_min, t_max) in the current time base. When a
        timestamps.npy is loaded this is [first_timestamp, last_timestamp];
        otherwise it is [0, total_duration]. The spinboxes and the
        range-info label use this so they stay consistent with the
        trace view's axis."""
        if not self.engine.data_loaded:
            return 0.0, 0.0
        return self.engine.get_time_range()
    
    def _set_default_range(self):
        """Initial analysis window: 10 seconds from the start of the
        available range (clamped for short recordings). The spinbox
        min/max have already been set by _sync_time_range_from_engine,
        so here we only need to pick reasonable values inside them."""
        span = self._available_max_time - self._available_min_time
        default_end = self._available_min_time + min(10.0, max(0.0, span))
        self.start_spin.setValue(self._available_min_time)
        self.end_spin.setValue(default_end)
        self.range_info_label.setText(
            f"Available: {self._available_min_time:.3f} \u2013 "
            f"{self._available_max_time:.3f} s"
        )

    def _sync_time_range_from_engine(self):
        """Refresh the spinbox min/max and the available-range label
        from the engine's current time base. Called from __init__ and
        again by MainWindow whenever a timestamps.npy is loaded after
        the dialog was opened."""
        self._available_min_time, self._available_max_time = \
            self._get_available_time_range()
        if self._available_max_time <= self._available_min_time:
            return
        self.start_spin.blockSignals(True)
        self.end_spin.blockSignals(True)
        self.start_spin.setRange(self._available_min_time, self._available_max_time)
        self.end_spin.setRange(
            self._available_min_time + 1e-3, self._available_max_time
        )
        self.start_spin.blockSignals(False)
        self.end_spin.blockSignals(False)
        self.range_info_label.setText(
            f"Available: {self._available_min_time:.3f} \u2013 "
            f"{self._available_max_time:.3f} s"
        )

    def on_engine_time_base_changed(self):
        """Called by MainWindow when a timestamps.npy is loaded (or
        replaced) after the dialog opened. Re-syncs the spinbox bounds
        to the new time base and repopulates the table so *_s columns
        reflect the new values. Sample columns are untouched."""
        self._sync_time_range_from_engine()
        self._populate_table()
        
    def _set_full_available_range(self):
        self._available_min_time, self._available_max_time = self._get_available_time_range()
        self.start_spin.setValue(self._available_min_time)
        self.end_spin.setValue(self._available_max_time)


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
        self.cancel_btn.setAutoDefault(False)
        self.cancel_btn.setDefault(False)
        self.cancel_btn.clicked.connect(self._on_cancel_clicked)
        btn_row.addWidget(self.cancel_btn)

        self.export_btn = QPushButton("Save")
        self.export_btn.setAutoDefault(False)
        self.export_btn.setDefault(False)
        self.export_btn.clicked.connect(self._on_save_clicked)
        self.export_btn.setEnabled(False)
        btn_row.addWidget(self.export_btn)

        self.load_btn = QPushButton("Load")
        self.load_btn.setAutoDefault(False)
        self.load_btn.setDefault(False)
        self.load_btn.clicked.connect(self._on_load_clicked)
        btn_row.addWidget(self.load_btn)

        self.prev_epoch_btn = QPushButton("\u25c4 Prev Epoch")
        self.prev_epoch_btn.setAutoDefault(False)
        self.prev_epoch_btn.setDefault(False)
        self.prev_epoch_btn.clicked.connect(lambda: self._step_epoch(-1))
        btn_row.addWidget(self.prev_epoch_btn)

        self.next_epoch_btn = QPushButton("Next Epoch \u25ba")
        self.next_epoch_btn.setAutoDefault(False)
        self.next_epoch_btn.setDefault(False)
        self.next_epoch_btn.clicked.connect(lambda: self._step_epoch(+1))
        btn_row.addWidget(self.next_epoch_btn)

        btn_row.addWidget(QLabel("Window (s):"))
        self.nav_window_spin = QDoubleSpinBox()
        self.nav_window_spin.setDecimals(2)
        self.nav_window_spin.setRange(0.05, 60.0)
        self.nav_window_spin.setValue(5.0)
        btn_row.addWidget(self.nav_window_spin)


        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.table = QTableWidget(0, 11)
        self.table.setHorizontalHeaderLabels([
            "Channel",
            "start_smp", "end_smp", "duration_smp",
            "start_s", "end_s", "duration_s",
            "mean_ratio", "peak_ratio",
            "mean_theta_power", "mean_delta_power",
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
        self.full_range_btn.setAutoDefault(False)
        self.full_range_btn.setDefault(False)
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
        self.delete_btn.setAutoDefault(False)
        self.delete_btn.setDefault(False)
        self.delete_btn.clicked.connect(self._delete_selected_epoch)
        grid.addWidget(self.delete_btn, 5, 0, 1, 2)

        self.add_epoch_btn = QPushButton("Add Theta Epoch")
        self.add_epoch_btn.setCheckable(True)
        self.add_epoch_btn.setAutoDefault(False)
        self.add_epoch_btn.setDefault(False)
        self.add_epoch_btn.setToolTip(
            "Click, then click-drag on a channel's lane in the main "
            "trace view to draw a new manual theta epoch there. "
            "Click again to cancel without creating one."
        )
        self.add_epoch_btn.setEnabled(self.trace_view is not None)
        self.add_epoch_btn.toggled.connect(self._on_add_epoch_toggled)
        grid.addWidget(self.add_epoch_btn, 5, 2, 1, 2)
        return grid
    
    def _refresh_export_buttons(self):
        """Enable Save only when there are epochs to write. Load stays
        enabled so the user can still try to load a saved session even
        with nothing detected yet (the handler shows a helpful message
        if the folder is empty)."""
        has_events = bool(self.epochs)
        has_file = getattr(self.engine, "data_file", None) is not None
        self.export_btn.setEnabled(has_events and has_file)

        if has_file:
            folder = analysis_folder_for(self.engine)
            self.export_btn.setToolTip(
                f"Write all theta epochs and their parameters to:\n{folder}"
            )
            self.load_btn.setToolTip(
                f"Load theta epochs and their parameters from:\n{folder}"
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

        has_epochs = bool(self.epochs)
        self.prev_epoch_btn.setEnabled(has_epochs)
        self.next_epoch_btn.setEnabled(has_epochs)

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

        self._last_params_for_sidecar = params
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

        # Convert the spinbox values (which are in the engine's current
        # time base -- clock if timestamps are loaded, elapsed
        # otherwise) to absolute sample indices here, once, so both the
        # thread's slicing and the offset we record for later use agree.
        start_idx = self.engine.timestamp_to_sample(self.start_spin.value())
        end_idx = self.engine.timestamp_to_sample(self.end_spin.value())
        # Record the offset up-front so a mid-detection dialog close
        # still leaves the value in a sane state for any pending
        # export / display.
        self._sample_offset = int(start_idx)


        self._detect_thread = _DetectThread(
            channels, self.engine.data, self.engine.sr,
            start_idx, end_idx, params,
            filter_settings=filter_settings, parent=self,
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
        self._detect_n_samples = int(info.get("n_samples", 0))
        self.start_spin.setValue(self._detection_start_time)
        self._populate_table()
        # self.export_btn.setEnabled(bool(self.epochs))
        self._refresh_export_buttons()
        self.epochsChanged.emit(self.epochs)
        self.status_label.setText(f"Found {len(self.epochs)} theta epochs.")

    def set_epochs_from_trace(self, epochs: list[ThetaEpoch]):
        self.epochs = sorted(list(epochs), key=lambda e: (e.channel, e.start_sample))
        self._populate_table()
        # self.export_btn.setEnabled(bool(self.epochs))
        self._refresh_export_buttons()
        self.status_label.setText(f"{len(self.epochs)} theta epochs.")
        # A successful creation (or split, or drag/merge) coming back
        # from the trace view means creation mode -- if it was armed --
        # has already completed and disarmed itself there; keep the
        # button's checked state in sync rather than leaving it stuck
        # checked after the epoch was actually created.
        if self.add_epoch_btn.isChecked():
            self.add_epoch_btn.blockSignals(True)
            self.add_epoch_btn.setChecked(False)
            self.add_epoch_btn.blockSignals(False)

    def _on_add_epoch_toggled(self, checked: bool):
        if self.trace_view is None:
            return
        if checked:
            self.trace_view.arm_theta_creation()
            self.status_label.setText(
                "Click-drag on a channel's lane in the main trace view "
                "to draw a new theta epoch."
            )
        else:
            self.trace_view.cancel_theta_creation()
            self.status_label.setText(f"{len(self.epochs)} theta epochs.")

    def set_selected_epoch(self, index: int):
        """Slot for trace_view.thetaEpochSelected -- an epoch was clicked
        on the trace view. Highlight it in the table WITHOUT panning the
        trace view: the user is already looking at the epoch (they just
        clicked it), and re-centering on click makes it impossible to
        resize a boundary handle near the edge of the window.

        Navigation that SHOULD center -- clicking a table row, or the
        Prev/Next Epoch buttons -- goes through _on_table_selection_changed
        / _step_epoch instead, which do not set this flag.
        """
        if not (0 <= index < self.table.rowCount()):
            return
        self._suppress_pan_on_select = True
        try:
            self.table.blockSignals(True)
            self.table.selectRow(int(index))
            self.table.blockSignals(False)
        finally:
            self._suppress_pan_on_select = False
        # Keep the row visible in the table without touching the trace view.
        self.table.scrollToItem(self.table.item(int(index), 0))

    def _on_table_selection_changed(self):
        row = self.table.currentRow()
        self.delete_btn.setEnabled(row >= 0)
        if row < 0:
            return

        # Always tell the trace view which epoch is selected, so it can
        # draw the selection highlight.
        self.thetaEpochSelected.emit(row)

        if not (0 <= row < len(self.epochs)):
            return

        # Click-to-select on the trace view must NOT recenter: that would
        # move the epoch under the user's cursor every time they click it
        # to grab a resize handle. Only table-row clicks and the Prev/Next
        # buttons recenter.
        if getattr(self, "_suppress_pan_on_select", False):
            return

        target = self.epochs[row]
        self._show_channel_in_trace_view(int(target.channel))
        self._pan_to_event(target)
        self.eventFocused.emit(int(target.channel))

    def _show_channel_in_trace_view(self, channel: int):
        """Append the channel to the trace view if not already there.
        Mirrors RippleDialog._show_channel_in_trace_view."""
        if self.trace_view is None:
            return
        if channel not in self.trace_view.channels:
            chans = list(self.trace_view.channels)
            chans.append(channel)
            self.trace_view.set_channels(chans)
        else:
            self.trace_view.update()

    def _pan_to_event(self, epoch):
        """Center the trace view on this epoch's midpoint time.

        The midpoint of [start_sample, end_sample] is used rather than
        peak_sample because peak_sample is not part of the epoch CSV
        schema -- every loaded epoch has peak_sample = 0, and using it
        would pan the view to time 0 for every event. The midpoint is
        close enough to the true peak for a "center the event" gesture
        and is stable across detection-then-load cycles.
        """
        if self.trace_view is None:
            return
        try:
            start_abs = int(epoch.start_sample) + int(self._sample_offset)
            end_abs = int(epoch.end_sample) + int(self._sample_offset)
            midpoint_sample = (start_abs + end_abs) // 2
            midpoint_time = float(self.engine.sample_to_time(midpoint_sample))
        except Exception:
            return
        window_duration = max(0.05, self.nav_window_spin.value())
        t_min, _ = self.engine.get_time_range()
        new_start = max(t_min, midpoint_time - window_duration / 2)

        self.trace_view.window_duration = window_duration
        self.trace_view.start_time = new_start
        self.trace_view.control_panel.set_duration(window_duration)
        self.trace_view._update_time_labels()
        self.trace_view._update_scrollbar_position()
        self.trace_view._invalidate_cache()
        self.trace_view.update()

    def _step_epoch(self, direction: int):
        """Step Prev/Next through the flat, (channel, start_sample)-
        sorted list of every epoch across every detected channel --
        the same order the table shows. Crossing from the last epoch
        of one channel into the first of the next is intentional.

        The anchor is the table's own selection; if nothing is
        selected, direction > 0 starts at row 0 and direction < 0
        starts at the last row. When the end (or start) is reached,
        the selection stays put -- no wrap.
        """
        if not self.epochs:
            return

        rows = self.table.selectionModel().selectedRows()
        if rows:
            current_row = int(rows[0].row())
        else:
            current_row = -1 if direction > 0 else len(self.epochs)

        new_row = current_row + direction
        new_row = max(0, min(new_row, len(self.epochs) - 1))

        target = self.epochs[new_row]
        self._show_channel_in_trace_view(int(target.channel))
        self._pan_to_event(target)

        # Programmatic selection; block signals so
        # _on_table_selection_changed doesn't fire mid-way.
        self.table.blockSignals(True)
        self.table.selectRow(new_row)
        self.table.blockSignals(False)

        # Emit selection so the trace view highlights this epoch.
        self.thetaEpochSelected.emit(new_row)

        # Tell MainWindow to merge this channel into the probe map's
        # selection.
        self.eventFocused.emit(int(target.channel))

        self.status_label.setText(
            f"Epoch {new_row + 1} of {len(self.epochs)} on CH{target.channel} "
            f"({'(next)' if direction > 0 else '(prev)'})"
        )

    def _delete_selected_epoch(self):
        row = self.table.currentRow()
        if not (0 <= row < len(self.epochs)):
            return
        self.epochs.pop(row)
        self._populate_table()
        #self.export_btn.setEnabled(bool(self.epochs))
        self._refresh_export_buttons()
        self.epochsChanged.emit(self.epochs)
        self.status_label.setText(f"{len(self.epochs)} theta epochs.")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete and self.delete_btn.isEnabled():
            self._delete_selected_epoch()
            event.accept()
            return
        super().keyPressEvent(event)


    def _populate_table(self):
        """Populate the table with the pooled epoch list. Sample
        columns show in-memory (slice-relative) values offset into
        absolute dat-file indices; time columns are recomputed from
        the current engine every time this runs, so a timestamps file
        loaded after detection is reflected on repopulation."""
        self.table.blockSignals(True)
        try:
            self.table.setRowCount(len(self.epochs))
            self.table.clearSelection()
            self.delete_btn.setEnabled(False)

            def t_abs(smp):
                return float(self.engine.sample_to_time(smp))

            for row, epoch in enumerate(self.epochs):
                start_smp = int(epoch.start_sample) + int(self._sample_offset)
                end_smp = int(epoch.end_sample) + int(self._sample_offset)
                dur_smp = end_smp - start_smp + 1

                values = [
                    str(epoch.channel),
                    str(start_smp),
                    str(end_smp),
                    str(dur_smp),
                    f"{t_abs(start_smp):.3f}",
                    f"{t_abs(end_smp):.3f}",
                    f"{dur_smp / float(self.engine.sr):.3f}",
                    f"{epoch.mean_ratio:.3f}",
                    f"{epoch.peak_ratio:.3f}",
                    f"{epoch.mean_theta_power:.4f}",
                    f"{epoch.mean_delta_power:.4f}",
                ]
                for col, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignCenter
                        | Qt.AlignmentFlag.AlignVCenter
                    )
                    self.table.setItem(row, col, item)
        finally:
            self.table.blockSignals(False)

    def _on_save_clicked(self):
        if not self.epochs:
            QMessageBox.information(
                self, "Nothing to save",
                "Detect (or load) theta epochs first."
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

        csv_path = folder / "theta_epochs.csv"

        from core.theta_epoch_export import export_theta_epochs_to_csv
        try:
            export_theta_epochs_to_csv(
                self.epochs, self._sample_offset, self.engine, csv_path,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return

        # Sidecar: only when the current session actually ran a
        # detection (so we have a valid analysis window and params).
        # A session that only loaded epochs from disk doesn't have a
        # _detect_n_samples value; in that case skip the sidecar
        # rather than overwrite the one that produced the file.
        n = int(getattr(self, "_detect_n_samples", 0))
        if n > 0:
            analysis_end_sample = int(self._sample_offset) + n
            try:
                save_sidecar(
                    csv_path=csv_path,
                    kind="theta_epochs",
                    sample_rate=float(self.engine.sr),
                    analysis_start_sample=int(self._sample_offset),
                    analysis_end_sample=analysis_end_sample,
                    channels=sorted({e.channel for e in self.epochs}),
                    params=self._last_detection_params(),
                )
            except Exception as exc:
                print(f"[theta-epoch-save] sidecar failed: {exc}")

        self.status_label.setText(
            f"Saved {len(self.epochs)} epoch(s) to {folder}"
        )

    def _last_detection_params(self):
        """Return the params used by the last detection, or reconstruct
        them from the current spinbox values if no detection has run
        yet (so the sidecar still has a sensible payload)."""
        params = getattr(self, "_last_params_for_sidecar", None)
        if params is not None:
            return params
        return ThetaEpochParams(
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

    def _on_load_clicked(self):
        if not self.engine.data_loaded:
            QMessageBox.warning(self, "No data loaded", "Load continuous.dat first.")
            return

        folder = analysis_folder_for(self.engine)
        if folder is None:
            QMessageBox.warning(
                self, "Cannot load",
                "No data file is loaded, so there is no folder to load "
                "the analysis from."
            )
            return

        csv_path = folder / "theta_epochs.csv"
        if not csv_path.exists():
            QMessageBox.information(
                self, "Nothing to load",
                f"No saved theta epochs were found in:\n\n{folder}\n\n"
                "Detect theta epochs and click Save first."
            )
            return

        # Read the sidecar first: if it exists, its sample_offset is
        # authoritative and its params should be restored on the dialog.
        sidecar = None
        try:
            sidecar = load_sidecar(csv_path)
        except Exception as exc:
            QMessageBox.warning(
                self, "Sidecar unreadable",
                f"Could not read the sidecar for {csv_path.name}:\n\n{exc}\n\n"
                "The epochs will load, but saved parameters will not be "
                "available."
            )
            sidecar = None

        sample_offset_to_use = 0
        if sidecar is not None:
            sample_offset_to_use = int(sidecar.get("analysis_start_sample", 0))

        from core.theta_epoch_export import import_theta_epochs_from_csv
        try:
            epochs = import_theta_epochs_from_csv(
                csv_path, sample_offset=sample_offset_to_use,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Failed to load", f"Could not read {csv_path}:\n\n{exc}")
            return

        if not epochs:
            QMessageBox.information(self, "No epochs found", f"{csv_path} contains no theta epochs.")
            return

        self.epochs = epochs
        self._sample_offset = int(sample_offset_to_use)
        self._detection_start_time = float(self.engine.sample_to_time(self._sample_offset))
        self._detect_n_samples = 0  # not known; only set on fresh detection

        if sidecar is not None:
            self._apply_sidecar(sidecar)

        self._populate_table()
        self._refresh_export_buttons()
        self.epochsChanged.emit(self.epochs)
        status = f"Loaded {len(self.epochs)} theta epoch(s) from {folder}."
        if sidecar is not None:
            status += " Parameters restored."
        self.status_label.setText(status)

    def closeEvent(self, event):
        if self._detect_thread is not None and self._detect_thread.isRunning():
            self._detect_thread.request_cancel()
            self._detect_thread.wait(2000)
        if self.trace_view is not None and self.add_epoch_btn.isChecked():
            self.trace_view.cancel_theta_creation()
        super().closeEvent(event)


    def _apply_sidecar(self, sidecar: dict):
        """Restore detection params from a loaded sidecar into the
        spinboxes. The epochs themselves are already loaded; this just
        makes the UI reflect the parameters that produced them, so a
        subsequent re-detection uses the same settings."""
        params_dict = sidecar.get("params") or {}
        try:
            params = ThetaEpochParams(**params_dict)
        except TypeError as exc:
            print(f"[theta-epoch-load] could not build params: {exc}")
            return

        self._last_params_for_sidecar = params

        # Push into the spinboxes. Each blockSignals so the
        # valueChanged handlers don't fire and clear state.
        for spin, value in (
            (self.theta_low_spin, params.theta_low),
            (self.theta_high_spin, params.theta_high),
            (self.delta_low_spin, params.delta_low),
            (self.delta_high_spin, params.delta_high),
            (self.ratio_spin, params.ratio_threshold),
            (self.min_dur_spin, params.min_duration_ms),
            (self.merge_gap_spin, params.merge_gap_ms),
            (self.power_window_spin, params.power_window_ms),
        ):
            spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(False)

        self.power_overlap_spin.blockSignals(True)
        self.power_overlap_spin.setValue(float(params.power_overlap) * 100.0)
        self.power_overlap_spin.blockSignals(False)

        self._refresh_export_buttons()
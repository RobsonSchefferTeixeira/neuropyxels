"""
export_panel.py

Dock panel for exporting a filtered + downsampled subset of channels.
Independent of the trace view: has its own filter/downsample controls,
its own time range, and its own channel selection (via a scoped probe
map that inherits rejected channels from the main map).

Computation runs on a QThread so the UI stays responsive on long
recordings with many channels. Progress is reported per-channel.

See core/export_data.py for the computation itself.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel,
    QLineEdit, QPushButton, QComboBox, QDoubleSpinBox, QSpinBox,
    QCheckBox, QRadioButton, QButtonGroup, QFileDialog, QMessageBox,
    QProgressBar, QDialog, QDialogButtonBox,
)

from core.analysis_folder import ensure_analysis_folder
from core.export_data import (
    ExportParams, export_channels, SUPPORTED_FORMATS, SUPPORTED_FILTER_ORDERS,
)
from gui.probe_map_widget import ProbeMapWidget


class _ExportThread(QThread):
    finished_ok = pyqtSignal(object)   # ExportResult
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, int)    # (done, total)

    def __init__(self, engine, params: ExportParams, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.params = params

    def run(self):
        try:
            result = export_channels(
                self.engine, self.params,
                progress_cb=self._progress,
            )
            self.finished_ok.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))

    def _progress(self, done: int, total: int):
        self.progress.emit(int(done), int(total))


class _ScopedProbeMapDialog(QDialog):
    """Modal dialog hosting its own ProbeMapWidget, seeded with a
    starting selection and inheriting rejected channels from the main
    map. Returns the selected channels on accept."""

    def __init__(
        self,
        probe_data: dict,
        initial_selection: list[int],
        rejected_channels: list[int],
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Select Channels for Export")
        self.setModal(True)
        self.resize(560, 800)

        layout = QVBoxLayout(self)

        self.probe_map = ProbeMapWidget(probe_data)
        layout.addWidget(self.probe_map, stretch=1)

        if rejected_channels:
            self.probe_map.set_rejected(rejected_channels)
        if initial_selection:
            self.probe_map.set_selected_channels(initial_selection)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_channels(self) -> list[int]:
        return self.probe_map.get_selected_channels()


class ExportDataPanel(QWidget):
    """Dock panel: filter + downsample + export channels to disk."""

    def __init__(self, engine, probe_data: dict, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.probe_data = probe_data or {}
        self._selected_channels: list[int] = []
        self._rejected_channels: set[int] = set()
        self._export_thread: _ExportThread | None = None
        self._suggest_timer = QTimer(self)
        self._suggest_timer.setSingleShot(True)
        self._suggest_timer.setInterval(250)
        self._suggest_timer.timeout.connect(self._apply_suggestion)

        self._build_ui()
        self._sync_time_range_from_engine()
        self._refresh_selection_label()
        self._refresh_target_sr_hint()

    # ------------------------------------------------------------------
    # Public API (called by MainWindow)
    # ------------------------------------------------------------------

    def update_engine(self, engine, probe_data: dict | None = None):
        self.engine = engine
        if probe_data is not None:
            self.probe_data = probe_data
        self._sync_time_range_from_engine()

    def set_selected_channels(self, channels: list[int]):
        self._selected_channels = list(channels)
        self._refresh_selection_label()

    def set_rejected_channels(self, channels: list[int]):
        self._rejected_channels = set(int(c) for c in channels)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ---- Time range ----
        time_group = QGroupBox("Time range")
        time_layout = QGridLayout(time_group)

        self.range_whole_radio = QRadioButton("Whole recording")
        self.range_custom_radio = QRadioButton("Custom range")
        self.range_whole_radio.setChecked(True)
        self._range_group = QButtonGroup(self)
        self._range_group.addButton(self.range_whole_radio)
        self._range_group.addButton(self.range_custom_radio)
        self.range_whole_radio.toggled.connect(self._on_range_toggled)

        time_layout.addWidget(self.range_whole_radio, 0, 0, 1, 4)
        time_layout.addWidget(self.range_custom_radio, 1, 0, 1, 4)

        time_layout.addWidget(QLabel("Start (s):"), 2, 0)
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setDecimals(3)
        self.start_spin.setRange(-1e12, 1e12)
        self.start_spin.setEnabled(False)
        time_layout.addWidget(self.start_spin, 2, 1)

        time_layout.addWidget(QLabel("End (s):"), 2, 2)
        self.end_spin = QDoubleSpinBox()
        self.end_spin.setDecimals(3)
        self.end_spin.setRange(-1e12, 1e12)
        self.end_spin.setEnabled(False)
        time_layout.addWidget(self.end_spin, 2, 3)

        self.range_info_label = QLabel("")
        self.range_info_label.setStyleSheet("color: #888; font-size: 10px;")
        time_layout.addWidget(self.range_info_label, 3, 0, 1, 4)

        layout.addWidget(time_group)

        # ---- Channels ----
        chan_group = QGroupBox("Channels")
        chan_layout = QVBoxLayout(chan_group)
        self.selection_label = QLabel("No channels selected.")
        self.selection_label.setStyleSheet("color: #ccc; font-size: 11px;")
        chan_layout.addWidget(self.selection_label)
        self.pick_channels_btn = QPushButton("Select channels from probe map…")
        self.pick_channels_btn.setAutoDefault(False)
        self.pick_channels_btn.clicked.connect(self._on_pick_channels_clicked)
        chan_layout.addWidget(self.pick_channels_btn)
        layout.addWidget(chan_group)

        # ---- Preprocessing ----
        filt_group = QGroupBox("Filter (Butterworth, zero-phase)")
        filt_layout = QGridLayout(filt_group)

        filt_layout.addWidget(QLabel("Low (Hz):"), 0, 0)
        self.low_spin = QDoubleSpinBox()
        self.low_spin.setRange(0.0, 20000.0)
        self.low_spin.setSingleStep(0.5)
        self.low_spin.setValue(0.5)
        self.low_spin.setKeyboardTracking(False)
        self.low_spin.editingFinished.connect(self._refresh_target_sr_hint)
        filt_layout.addWidget(self.low_spin, 0, 1)

        filt_layout.addWidget(QLabel("High (Hz):"), 0, 2)
        self.high_spin = QDoubleSpinBox()
        self.high_spin.setRange(0.0, 20000.0)
        self.high_spin.setSingleStep(10.0)
        self.high_spin.setValue(450.0)
        self.high_spin.setKeyboardTracking(False)
        self.high_spin.editingFinished.connect(self._refresh_target_sr_hint)
        filt_layout.addWidget(self.high_spin, 0, 3)

        filt_layout.addWidget(QLabel("Order:"), 1, 0)
        self.order_spin = QSpinBox()
        self.order_spin.setRange(2, 8)
        self.order_spin.setValue(4)
        filt_layout.addWidget(self.order_spin, 1, 1)

        filt_layout.addWidget(QLabel("Notch (Hz):"), 1, 2)
        self.notch_spin = QDoubleSpinBox()
        self.notch_spin.setRange(0.0, 20000.0)
        self.notch_spin.setSingleStep(10.0)
        self.notch_spin.setValue(50.0)
        self.notch_spin.setToolTip("0 disables the notch filter")
        filt_layout.addWidget(self.notch_spin, 1, 3)

        self.detrend_check = QCheckBox("Detrend (remove DC offset)")
        filt_layout.addWidget(self.detrend_check, 2, 0, 1, 4)

        self.suggest_btn = QPushButton("Suggest filter for downsample target")
        self.suggest_btn.setAutoDefault(False)
        self.suggest_btn.setToolTip(
            "Set the filter band to a sensible default for the chosen "
            "output sample rate: high cutoff at 0.45 x target_sr "
            "(leaves transition-band room below the new Nyquist), "
            "low cutoff at 0.5 Hz to remove DC."
        )
        self.suggest_btn.clicked.connect(self._apply_suggestion)
        filt_layout.addWidget(self.suggest_btn, 3, 0, 1, 4)

        layout.addWidget(filt_group)

        # ---- Downsample ----
        ds_group = QGroupBox("Downsample")
        ds_layout = QGridLayout(ds_group)

        ds_layout.addWidget(QLabel("Target sample rate (Hz):"), 0, 0)
        self.target_sr_spin = QDoubleSpinBox()
        self.target_sr_spin.setRange(1.0, 100000.0)
        self.target_sr_spin.setDecimals(1)
        self.target_sr_spin.setValue(1000.0)
        self.target_sr_spin.setKeyboardTracking(False)
        self.target_sr_spin.editingFinished.connect(self._refresh_target_sr_hint)
        ds_layout.addWidget(self.target_sr_spin, 0, 1)

        self.target_sr_hint = QLabel("")
        self.target_sr_hint.setStyleSheet("color: #888; font-size: 10px;")
        ds_layout.addWidget(self.target_sr_hint, 1, 0, 1, 2)

        layout.addWidget(ds_group)

        # ---- Output ----
        out_group = QGroupBox("Output")
        out_layout = QGridLayout(out_group)

        out_layout.addWidget(QLabel("Format:"), 0, 0)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["npz", "mat", "h5"])
        self.format_combo.setToolTip(
            "npz: NumPy, load with np.load\n"
            "mat: MATLAB v7.3 (HDF5-based), load with scipy/h5py/MATLAB\n"
            "h5:  HDF5, load with h5py/MATLAB/Julia/R"
        )
        out_layout.addWidget(self.format_combo, 0, 1)

        out_layout.addWidget(QLabel("Per-channel files:"), 0, 2)
        self.per_channel_check = QCheckBox("One file per channel")
        self.per_channel_check.setToolTip(
            "On: one file per channel (filtered_ch<channel>.<ext>).\n"
            "Off: one pooled file with all channels sorted by probe "
            "geometry (filtered.<ext>)."
        )
        out_layout.addWidget(self.per_channel_check, 0, 3)

        out_layout.addWidget(QLabel("Folder:"), 1, 0)
        self.folder_edit = QLineEdit()
        self.folder_edit.setReadOnly(True)
        out_layout.addWidget(self.folder_edit, 1, 1, 1, 2)
        self.browse_btn = QPushButton("Browse…")
        self.browse_btn.setAutoDefault(False)
        self.browse_btn.clicked.connect(self._on_browse_clicked)
        out_layout.addWidget(self.browse_btn, 1, 3)

        layout.addWidget(out_group)

        # ---- Run ----
        run_row = QHBoxLayout()
        self.export_btn = QPushButton("Export")
        self.export_btn.setAutoDefault(False)
        self.export_btn.clicked.connect(self._on_export_clicked)
        run_row.addWidget(self.export_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        run_row.addWidget(self.progress_bar, stretch=1)

        layout.addLayout(run_row)

        self.status_label = QLabel("Ready.")
        self.status_label.setStyleSheet("color: #888; font-size: 10px;")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addStretch(1)

    # ------------------------------------------------------------------
    # Setup / refresh
    # ------------------------------------------------------------------

    def _sync_time_range_from_engine(self):
        if not self.engine.data_loaded:
            self.start_spin.setRange(0.0, 0.0)
            self.end_spin.setRange(0.0, 0.0)
            self.range_info_label.setText("No data loaded.")
            return

        t_min, t_max = self.engine.get_time_range()
        self.start_spin.blockSignals(True)
        self.end_spin.blockSignals(True)
        self.start_spin.setRange(t_min, t_max)
        self.end_spin.setRange(t_min, t_max)
        self.start_spin.setValue(t_min)
        self.end_spin.setValue(t_max)
        self.start_spin.blockSignals(False)
        self.end_spin.blockSignals(False)
        self.range_info_label.setText(
            f"Available: {t_min:.3f} \u2013 {t_max:.3f} s"
        )
        self._refresh_default_output_folder()

    def _refresh_default_output_folder(self):
        folder = ensure_analysis_folder(self.engine) if self.engine.data_loaded else None
        if folder is not None:
            self.folder_edit.setText(str(folder))

    def _refresh_selection_label(self):
        n = len(self._selected_channels)
        if n == 0:
            self.selection_label.setText("No channels selected.")
        else:
            self.selection_label.setText(
                f"{n} channel(s) selected: "
                f"{', '.join(str(c) for c in self._selected_channels[:8])}"
                + ("…" if n > 8 else "")
            )

    def _refresh_target_sr_hint(self):
        if not self.engine.data_loaded:
            self.target_sr_hint.setText("")
            return
        sr = float(self.engine.sr)
        target = float(self.target_sr_spin.value())
        if target >= sr:
            self.target_sr_hint.setText(
                f"No downsampling: source is {sr:.0f} Hz."
            )
            return
        factor = int(round(sr / target))
        actual = sr / factor
        self.target_sr_hint.setText(
            f"Decimation factor: {factor}  →  actual output {actual:.2f} Hz"
        )

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def _on_range_toggled(self, whole_checked: bool):
        self.start_spin.setEnabled(not whole_checked)
        self.end_spin.setEnabled(not whole_checked)

    def _on_pick_channels_clicked(self):
        if not self.probe_data:
            QMessageBox.warning(
                self, "No probe loaded",
                "Load a probe first so channels can be selected on a map."
            )
            return
        dialog = _ScopedProbeMapDialog(
            self.probe_data,
            self._selected_channels,
            sorted(self._rejected_channels),
            parent=self,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_selection = dialog.selected_channels()
            self._selected_channels = new_selection
            self._refresh_selection_label()

    def _on_browse_clicked(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Choose output folder",
            self.folder_edit.text() or "",
        )
        if folder:
            self.folder_edit.setText(folder)

    def _apply_suggestion(self):
        """Fill low/high with values chosen from the current target SR:
        high = 0.45 * target_sr, low = 0.5 Hz."""
        target = float(self.target_sr_spin.value())
        if target <= 0:
            return
        suggested_high = max(1.0, 0.45 * target)
        self.low_spin.blockSignals(True)
        self.high_spin.blockSignals(True)
        self.low_spin.setValue(0.5)
        self.high_spin.setValue(suggested_high)
        self.low_spin.blockSignals(False)
        self.high_spin.blockSignals(False)
        self.status_label.setText(
            f"Suggested filter: 0.5\u2013{suggested_high:.1f} Hz "
            f"for {target:.0f} Hz output."
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _collect_params(self) -> ExportParams:
        if not self.engine.data_loaded:
            raise ValueError("No data loaded.")
        if not self._selected_channels:
            raise ValueError("No channels selected.")

        if self.range_whole_radio.isChecked():
            t_min, t_max = self.engine.get_time_range()
            start_time, end_time = t_min, t_max
        else:
            start_time = float(self.start_spin.value())
            end_time = float(self.end_spin.value())
            if end_time <= start_time:
                raise ValueError("End time must be after start time.")

        start_sample = int(self.engine.timestamp_to_sample(start_time))
        end_sample = int(self.engine.timestamp_to_sample(end_time))

        fmt = self.format_combo.currentText()
        ext = fmt
        folder = Path(self.folder_edit.text())
        if self.per_channel_check.isChecked():
            out_path = folder / f"filtered.{ext}"  # parent of per-channel files
        else:
            out_path = folder / f"filtered.{ext}"

        order = int(self.order_spin.value())
        if order not in SUPPORTED_FILTER_ORDERS:
            raise ValueError(f"Filter order must be 2-8, got {order}.")

        notch = float(self.notch_spin.value())
        notch_val = notch if notch > 0 else None

        geometry = self._channel_geometry_map()

        return ExportParams(
            channels=list(self._selected_channels),
            start_sample=start_sample,
            end_sample=end_sample,
            output_sample_rate=float(self.target_sr_spin.value()),
            filter_low=float(self.low_spin.value()),
            filter_high=float(self.high_spin.value()),
            filter_order=order,
            notch_freq=notch_val,
            detrend=bool(self.detrend_check.isChecked()),
            target_format=fmt,
            output_path=out_path,
            per_channel=self.per_channel_check.isChecked(),
            channel_geometry=geometry,
        )

    def _channel_geometry_map(self) -> dict[int, dict] | None:
        if not self.probe_data:
            return None
        coords = self.probe_data.get("coordinates", {})
        channels = coords.get("channels", [])
        xs = coords.get("x", [])
        ys = coords.get("y", [])
        shanks = self.probe_data.get("shanks", {}).get("ids") or [0] * len(channels)
        if not channels:
            return None
        return {
            int(ch): {
                "shank": int(sh),
                "x": float(x),
                "y": float(y),
            }
            for ch, x, y, sh in zip(channels, xs, ys, shanks)
        }

    def _on_export_clicked(self):
        try:
            params = self._collect_params()
        except Exception as exc:
            QMessageBox.warning(self, "Cannot export", str(exc))
            return

        self.export_btn.setEnabled(False)
        self.export_btn.setText("Exporting…")
        self.progress_bar.setValue(0)
        self.status_label.setText("Filtering, downsampling, writing…")

        self._export_thread = _ExportThread(self.engine, params, parent=self)
        self._export_thread.progress.connect(self._on_progress)
        self._export_thread.finished_ok.connect(self._on_finished)
        self._export_thread.failed.connect(self._on_failed)
        self._export_thread.start()

    def _on_progress(self, done: int, total: int):
        pct = int(round(100 * done / max(1, total)))
        self.progress_bar.setValue(pct)
        self.status_label.setText(f"Processed {done}/{total} channel(s)…")

    def _on_finished(self, result):
        self.export_btn.setEnabled(True)
        self.export_btn.setText("Export")
        self.progress_bar.setValue(100)
        n_files = len(result.written_paths)
        first = result.written_paths[0] if result.written_paths else "?"
        self.status_label.setText(
            f"Exported {result.n_channels} channel(s) × "
            f"{result.n_samples} samples @ {result.output_sample_rate:.0f} Hz "
            f"→ {n_files} file(s) in {first.parent}"
        )
        self._export_thread = None

    def _on_failed(self, message: str):
        self.export_btn.setEnabled(True)
        self.export_btn.setText("Export")
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Export failed: {message}")
        QMessageBox.critical(self, "Export failed", message)
        self._export_thread = None
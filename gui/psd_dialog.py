"""
psd_dialog.py

Non-modal QDialog for power spectral density analysis of one channel.
Opened from the Analysis menu. Computation runs on a QThread so the UI
doesn't freeze on long segments. Rendering is a pyqtgraph PlotWidget
with optional log axes and user-controlled zoom.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QDoubleSpinBox,
    QSpinBox, QCheckBox, QPushButton, QComboBox, QMessageBox, QWidget,
    QGroupBox,
)

from core.psd_analyzer import PsdAnalyzer, PsdParams
from core.trace_engine import TraceEngine


BG_COLOR = "#1e1e1e"
CURVE_COLOR = "#5ad1ff"


class _ComputeThread(QThread):
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, analyzer, params, parent=None):
        super().__init__(parent)
        self.analyzer = analyzer
        self.params = params

    def run(self):
        try:
            result = self.analyzer.compute(self.params)
            self.finished_ok.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class PsdDialog(QDialog):
    """Non-modal PSD analysis dialog."""

    def __init__(self, engine: TraceEngine,
                 initial_channel: int | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PSD Analysis")
        self.setModal(False)
        self.resize(950, 720)

        self.engine = engine
        self.analyzer = PsdAnalyzer(engine)
        self._compute_thread: _ComputeThread | None = None
        self._last_result: dict | None = None

        # Set to True after the first render of each compute, so we only
        # apply the axis limits once per compute. This lets the user
        # zoom/pan freely afterward without the view being reset on
        # every repaint.
        self._view_initialized = False

        self._build_ui()
        self._sync_time_range_from_engine()

        if initial_channel is not None:
            self.channel_spin.setValue(int(initial_channel))

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        pg.setConfigOption("background", BG_COLOR)
        pg.setConfigOption("foreground", "#d0d0d0")

        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        layout.addWidget(self._build_window_group())
        layout.addWidget(self._build_method_group())
        layout.addWidget(self._build_axes_group())

        btn_row = QHBoxLayout()
        self.compute_btn = QPushButton("Compute PSD")
        self.compute_btn.clicked.connect(self._on_compute_clicked)
        btn_row.addWidget(self.compute_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        # ---- Plot ----
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setMenuEnabled(False)
        self.plot_widget.setLabel("bottom", "Frequency", units="Hz")
        self.plot_widget.setLabel("left", "PSD")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.15)
        # Explicitly enable interactive zoom/pan on both axes.
        self.plot_widget.setMouseEnabled(x=True, y=True)
        self.curve = self.plot_widget.plot([], [], pen=pg.mkPen(CURVE_COLOR, width=1.6))
        layout.addWidget(self.plot_widget, stretch=1)

        self.status_label = QLabel("Set parameters and click Compute PSD.")
        self.status_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self.status_label)

    def _build_window_group(self) -> QGroupBox:
        group = QGroupBox("Analysis window")
        grid = QGridLayout(group)

        grid.addWidget(QLabel("Channel:"), 0, 0)
        self.channel_spin = QSpinBox()
        self.channel_spin.setRange(0, 100000)
        grid.addWidget(self.channel_spin, 0, 1)

        grid.addWidget(QLabel("Start (s):"), 0, 2)
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setDecimals(3)
        self.start_spin.setRange(0.0, 1e9)
        grid.addWidget(self.start_spin, 0, 3)

        grid.addWidget(QLabel("End (s):"), 0, 4)
        self.end_spin = QDoubleSpinBox()
        self.end_spin.setDecimals(3)
        self.end_spin.setRange(0.0, 1e9)
        self.end_spin.setValue(10.0)
        grid.addWidget(self.end_spin, 0, 5)

        self.use_whole_checkbox = QCheckBox("Use whole recording")
        self.use_whole_checkbox.toggled.connect(self._on_use_whole_toggled)
        grid.addWidget(self.use_whole_checkbox, 1, 0, 1, 6)

        return group

    def _build_method_group(self) -> QGroupBox:
        group = QGroupBox("Method")
        grid = QGridLayout(group)

        grid.addWidget(QLabel("Method:"), 0, 0)
        self.method_combo = QComboBox()
        self.method_combo.addItems(["welch", "periodogram"])
        grid.addWidget(self.method_combo, 0, 1)

        grid.addWidget(QLabel("Window:"), 0, 2)
        self.window_combo = QComboBox()
        self.window_combo.addItems(["hann", "hamming", "blackman", "boxcar"])
        grid.addWidget(self.window_combo, 0, 3)

        grid.addWidget(QLabel("nperseg:"), 1, 0)
        self.nperseg_spin = QSpinBox()
        self.nperseg_spin.setRange(8, 10_000_000)
        self.nperseg_spin.setValue(4096)
        grid.addWidget(self.nperseg_spin, 1, 1)

        grid.addWidget(QLabel("noverlap:"), 1, 2)
        self.noverlap_spin = QSpinBox()
        self.noverlap_spin.setRange(0, 10_000_000)
        self.noverlap_spin.setValue(2048)
        grid.addWidget(self.noverlap_spin, 1, 3)

        grid.addWidget(QLabel("nfft:"), 1, 4)
        self.nfft_spin = QSpinBox()
        self.nfft_spin.setRange(8, 100_000_000)
        self.nfft_spin.setValue(8192)
        grid.addWidget(self.nfft_spin, 1, 5)

        grid.addWidget(QLabel("Target SR (Hz):"), 2, 0)
        self.target_sr_spin = QSpinBox()
        self.target_sr_spin.setRange(0, 30000)
        self.target_sr_spin.setValue(1000)
        self.target_sr_spin.setSpecialValueText("off")
        self.target_sr_spin.setToolTip(
            "Decimate the signal to this sample rate before computing "
            "the PSD. 0 disables downsampling."
        )
        grid.addWidget(self.target_sr_spin, 2, 1)

        return group

    def _build_axes_group(self) -> QGroupBox:
        group = QGroupBox("Axes")
        grid = QGridLayout(group)

        # ---- X axis ----
        self.x_log_checkbox = QCheckBox("X log")
        grid.addWidget(self.x_log_checkbox, 0, 0)

        self.x_auto_checkbox = QCheckBox("X auto")
        self.x_auto_checkbox.setChecked(True)
        self.x_auto_checkbox.toggled.connect(
            lambda checked: self._set_axis_limits_enabled("x", not checked)
        )
        grid.addWidget(self.x_auto_checkbox, 0, 1)

        grid.addWidget(QLabel("X min:"), 0, 2)
        self.x_min_spin = QDoubleSpinBox()
        self.x_min_spin.setRange(0.0, 1e6)
        self.x_min_spin.setValue(0.0)
        self.x_min_spin.setEnabled(False)
        grid.addWidget(self.x_min_spin, 0, 3)

        grid.addWidget(QLabel("X max:"), 0, 4)
        self.x_max_spin = QDoubleSpinBox()
        self.x_max_spin.setRange(0.0, 1e6)
        self.x_max_spin.setValue(100.0)
        self.x_max_spin.setEnabled(False)
        grid.addWidget(self.x_max_spin, 0, 5)

        # ---- Y axis ----
        self.y_log_checkbox = QCheckBox("Y log (dB)")
        self.y_log_checkbox.setChecked(True)
        grid.addWidget(self.y_log_checkbox, 1, 0)

        self.y_auto_checkbox = QCheckBox("Y auto")
        self.y_auto_checkbox.setChecked(True)
        self.y_auto_checkbox.toggled.connect(
            lambda checked: self._set_axis_limits_enabled("y", not checked)
        )
        grid.addWidget(self.y_auto_checkbox, 1, 1)

        grid.addWidget(QLabel("Y min:"), 1, 2)
        self.y_min_spin = QDoubleSpinBox()
        self.y_min_spin.setRange(-1e6, 1e6)
        self.y_min_spin.setValue(-100.0)
        self.y_min_spin.setEnabled(False)
        grid.addWidget(self.y_min_spin, 1, 3)

        grid.addWidget(QLabel("Y max:"), 1, 4)
        self.y_max_spin = QDoubleSpinBox()
        self.y_max_spin.setRange(-1e6, 1e6)
        self.y_max_spin.setValue(50.0)
        self.y_max_spin.setEnabled(False)
        grid.addWidget(self.y_max_spin, 1, 5)

        return group

    def _set_axis_limits_enabled(self, axis: str, enabled: bool):
        if axis == "x":
            self.x_min_spin.setEnabled(enabled)
            self.x_max_spin.setEnabled(enabled)
        else:
            self.y_min_spin.setEnabled(enabled)
            self.y_max_spin.setEnabled(enabled)

    def _on_use_whole_toggled(self, checked: bool):
        self.start_spin.setEnabled(not checked)
        self.end_spin.setEnabled(not checked)

    def _sync_time_range_from_engine(self):
        if self.engine.data_loaded:
            duration = float(self.engine.total_duration)
            self.start_spin.setRange(0.0, duration)
            self.end_spin.setRange(0.0, duration)
            self.end_spin.setValue(min(10.0, duration))

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------

    def _collect_params(self) -> PsdParams:
        target_sr_value = int(self.target_sr_spin.value())
        return PsdParams(
            channel=int(self.channel_spin.value()),
            start_time=float(self.start_spin.value()),
            end_time=float(self.end_spin.value()),
            use_whole_recording=bool(self.use_whole_checkbox.isChecked()),
            method=str(self.method_combo.currentText()),
            window=str(self.window_combo.currentText()),
            nperseg=int(self.nperseg_spin.value()),
            noverlap=int(self.noverlap_spin.value()),
            nfft=int(self.nfft_spin.value()),
            target_sr=(None if target_sr_value <= 0 else float(target_sr_value)),
        )

    def _on_compute_clicked(self):
        if not self.engine.data_loaded:
            QMessageBox.warning(self, "No data", "No data is loaded.")
            return

        params = self._collect_params()

        if not params.use_whole_recording and params.start_time >= params.end_time:
            QMessageBox.warning(self, "Invalid range",
                                "Start time must be before end time.")
            return

        self.compute_btn.setEnabled(False)
        self.compute_btn.setText("Computing...")
        self.status_label.setText("Computing PSD...")

        self._compute_thread = _ComputeThread(self.analyzer, params, parent=self)
        self._compute_thread.finished_ok.connect(self._on_compute_finished)
        self._compute_thread.failed.connect(self._on_compute_failed)
        self._compute_thread.start()

    def _on_compute_failed(self, message: str):
        self.compute_btn.setEnabled(True)
        self.compute_btn.setText("Compute PSD")
        self.status_label.setText(f"Error: {message}")
        QMessageBox.critical(self, "PSD computation failed", message)

    def _on_compute_finished(self, result: dict):
        self.compute_btn.setEnabled(True)
        self.compute_btn.setText("Compute PSD")
        self._last_result = result
        # Fresh result -> allow the view to be re-set on this render.
        self._view_initialized = False
        self._render(result)

    def mouseDoubleClickEvent(self, event):
        """Double-click anywhere in the dialog resets the plot view to
        the axis-limit settings the last Compute applied. If both axes
        are set to auto, this simply triggers a fresh auto-range."""
        if self._last_result is None:
            super().mouseDoubleClickEvent(event)
            return

        x_log = bool(self.x_log_checkbox.isChecked())

        if self.x_auto_checkbox.isChecked():
            self.plot_widget.enableAutoRange(axis="x")
        else:
            x_min = float(self.x_min_spin.value())
            x_max = float(self.x_max_spin.value())
            if x_max > x_min:
                if x_log:
                    safe_min = max(x_min, 1e-3)
                    self.plot_widget.setXRange(
                        np.log10(safe_min),
                        np.log10(x_max),
                        padding=0.0,
                    )
                else:
                    self.plot_widget.setXRange(x_min, x_max, padding=0.0)

        if self.y_auto_checkbox.isChecked():
            self.plot_widget.enableAutoRange(axis="y")
        else:
            y_min = float(self.y_min_spin.value())
            y_max = float(self.y_max_spin.value())
            if y_max > y_min:
                self.plot_widget.setYRange(y_min, y_max, padding=0.0)

        event.accept()
        
    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _render(self, result: dict):
        freqs = np.asarray(result["frequencies"], dtype=np.float64)
        psd = np.asarray(result["psd"], dtype=np.float64)

        x_log = bool(self.x_log_checkbox.isChecked())
        y_log = bool(self.y_log_checkbox.isChecked())

        # A DC bin (0 Hz) has no place on a log x axis; drop it only when
        # the x axis is log. On linear x, keep it.
        if x_log:
            keep = freqs > 0
            freqs = freqs[keep]
            psd = psd[keep]

        psd_safe = np.maximum(psd, np.finfo(float).tiny)

        if y_log:
            y = 10.0 * np.log10(psd_safe)
            ylabel = "PSD (dB)"
        else:
            y = psd_safe
            ylabel = "PSD"

        # ---- Plot ----
        self.plot_widget.clear()
        self.curve = self.plot_widget.plot(
            freqs, y,
            pen=pg.mkPen(CURVE_COLOR, width=1.6),
        )

        # ---- Axis scales ----
        # Only x uses pyqtgraph's log view mode. The y values already
        # carry their own log transform (10*log10(psd) when y_log is on),
        # so putting the y axis into log view mode as well would double
        # the transform. Keep y in linear view always and let the y
        # values be what they are (dB or raw PSD).
        vb = self.plot_widget.getViewBox()
        vb.setLogMode(x_log, False)
        self.plot_widget.setLabel("left", ylabel)

        # ---- Axis limits (only on first render of this result) ----
        if not self._view_initialized:
            if self.x_auto_checkbox.isChecked():
                self.plot_widget.enableAutoRange(axis="x")
            else:
                x_min = float(self.x_min_spin.value())
                x_max = float(self.x_max_spin.value())
                if x_max > x_min:
                    if x_log:
                        safe_min = max(x_min, 1e-3)
                        self.plot_widget.setXRange(
                            np.log10(safe_min),
                            np.log10(x_max),
                            padding=0.0,
                        )
                    else:
                        self.plot_widget.setXRange(x_min, x_max, padding=0.0)

            if self.y_auto_checkbox.isChecked():
                self.plot_widget.enableAutoRange(axis="y")
            else:
                y_min = float(self.y_min_spin.value())
                y_max = float(self.y_max_spin.value())
                if y_max > y_min:
                    self.plot_widget.setYRange(y_min, y_max, padding=0.0)

            self._view_initialized = True

        # ---- Status line ----
        params: PsdParams = result["params"]
        dur = result["end_time"] - result["start_time"]
        target_str = (
            f"target {params.target_sr:.0f} Hz"
            if params.target_sr is not None and params.target_sr > 0
            else "no downsample"
        )
        self.status_label.setText(
            f"CH{result['channel']}  |  "
            f"{params.method}  |  "
            f"window={params.window}  |  "
            f"nperseg={params.nperseg}  noverlap={params.noverlap}  nfft={params.nfft}  |  "
            f"{dur:.2f}s  ({result['n_samples']} samples @ "
            f"{result['sample_rate']:.0f} Hz, {target_str})"
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if self._compute_thread is not None and self._compute_thread.isRunning():
            self._compute_thread.wait(2000)
        super().closeEvent(event)
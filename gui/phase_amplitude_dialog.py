"""
phase_amplitude_dialog.py

Non-modal QDialog for theta phase-amplitude coupling analysis, replacing
neuropixels_phase_amplitude.py's tkinter/matplotlib PhaseAmplitudeAnalyzer
GUI. The math lives in core/phase_amplitude.py (PhaseAmplitudeAnalyzer);
this file is purely presentation + parameter controls.

Renders the same 2-theta-cycle phase-vs-frequency heatmap as the
original's contourf plot, but as a pyqtgraph ImageItem (GPU-backed,
much faster to redraw when parameters change). A cosine reference curve
is overlaid the same way the original did, to anchor the eye to the
theta cycle.

Computation runs on a QThread so the UI doesn't freeze during the
wavelet transform, which can take a second or more for wide frequency
ranges / long time windows.

Usage
-----
    dialog = PhaseAmplitudeDialog(probe_data, engine, parent=main_window)
    dialog.set_channel(current_channel)
    dialog.show()   # non-modal
"""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QDoubleSpinBox,
    QSpinBox, QCheckBox, QPushButton, QComboBox, QMessageBox, QWidget,
)

from core.phase_amplitude import PhaseAmplitudeAnalyzer, PhaseAmplitudeParams
from core.trace_engine import TraceEngine


CMAP_OPTIONS = [
    "viridis", "plasma", "inferno", "magma", "cividis",
    "CET-D1", "CET-D4",  # pyqtgraph's built-in diverging maps (RdBu-ish)
]


class _ComputeThread(QThread):
    """Runs PhaseAmplitudeAnalyzer.compute() off the UI thread."""

    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, analyzer: PhaseAmplitudeAnalyzer, raw_data,
                 sample_rate: float, params: PhaseAmplitudeParams, parent=None):
        super().__init__(parent)
        self.analyzer = analyzer
        self.raw_data = raw_data
        self.sample_rate = sample_rate
        self.params = params

    def run(self):
        try:
            results = self.analyzer.compute(self.raw_data, self.sample_rate, self.params)
            self.finished_ok.emit(results)
        except Exception as exc:
            self.failed.emit(str(exc))


class PhaseAmplitudeDialog(QDialog):
    """
    Non-modal analysis dialog. One instance analyzes one probe stream's
    data at a time; channel/params can be changed and recomputed freely
    without closing the dialog.
    """

    def __init__(self, probe_data: dict, engine: TraceEngine, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Phase-Amplitude Coupling Analysis")
        self.setModal(False)
        self.resize(950, 750)

        self.engine = engine
        self.analyzer = PhaseAmplitudeAnalyzer(probe_data)
        self._compute_thread: _ComputeThread | None = None
        self._last_results: dict | None = None

        self._build_ui()
        self._sync_time_range_from_engine()

    # ------------------------------------------------------------------
    # UI scaffolding
    # ------------------------------------------------------------------

    def _build_ui(self):
        pg.setConfigOption("background", "#1e1e1e")
        pg.setConfigOption("foreground", "#d0d0d0")

        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        layout.addLayout(self._build_param_grid())

        btn_row = QHBoxLayout()
        self.compute_btn = QPushButton("Compute")
        self.compute_btn.clicked.connect(self._on_compute_clicked)
        btn_row.addWidget(self.compute_btn)

        self.csd_status_label = QLabel("")
        self.csd_status_label.setStyleSheet("color: #999; font-size: 11px;")
        btn_row.addWidget(self.csd_status_label, stretch=1)
        layout.addLayout(btn_row)

        # ---- Heatmap ----
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel("bottom", "Theta Phase", units="\u00b0")
        self.plot_widget.setLabel("left", "Frequency", units="Hz")
        self.plot_widget.setMenuEnabled(False)

        self.image_item = pg.ImageItem()
        self.plot_widget.addItem(self.image_item)

        self.colorbar = pg.ColorBarItem(colorMap=pg.colormap.get("viridis"))
        self.colorbar.setImageItem(self.image_item, insert_in=self.plot_widget.plotItem)

        self.sine_curve = self.plot_widget.plot(
            pen=pg.mkPen((255, 255, 255, 180), width=2, style=Qt.PenStyle.DashLine)
        )

        layout.addWidget(self.plot_widget, stretch=1)

        self.status_label = QLabel("Set parameters and click Compute.")
        self.status_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self.status_label)

    def _build_param_grid(self) -> QGridLayout:
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        col_pairs = []  # (label, widget) laid out 4 per row

        self.channel_spin = QSpinBox()
        self.channel_spin.setRange(0, 100000)
        col_pairs.append(("Channel", self.channel_spin))

        self.start_spin = QDoubleSpinBox()
        self.start_spin.setDecimals(2)
        self.start_spin.setRange(0.0, 1e9)
        col_pairs.append(("Start (s)", self.start_spin))

        self.end_spin = QDoubleSpinBox()
        self.end_spin.setDecimals(2)
        self.end_spin.setRange(0.01, 1e9)
        self.end_spin.setValue(10.0)
        col_pairs.append(("End (s)", self.end_spin))

        self.phase_low_spin = QDoubleSpinBox()
        self.phase_low_spin.setRange(0.1, 50.0)
        self.phase_low_spin.setValue(5.0)
        col_pairs.append(("\u03b8 low (Hz)", self.phase_low_spin))

        self.phase_high_spin = QDoubleSpinBox()
        self.phase_high_spin.setRange(0.1, 50.0)
        self.phase_high_spin.setValue(10.0)
        col_pairs.append(("\u03b8 high (Hz)", self.phase_high_spin))

        self.freq_low_spin = QDoubleSpinBox()
        self.freq_low_spin.setRange(1.0, 5000.0)
        self.freq_low_spin.setValue(20.0)
        col_pairs.append(("Freq low (Hz)", self.freq_low_spin))

        self.freq_high_spin = QDoubleSpinBox()
        self.freq_high_spin.setRange(1.0, 5000.0)
        self.freq_high_spin.setValue(200.0)
        col_pairs.append(("Freq high (Hz)", self.freq_high_spin))

        self.freq_step_spin = QDoubleSpinBox()
        self.freq_step_spin.setRange(0.1, 50.0)
        self.freq_step_spin.setValue(1.0)
        col_pairs.append(("Freq step (Hz)", self.freq_step_spin))

        self.min_cycles_spin = QDoubleSpinBox()
        self.min_cycles_spin.setRange(1.0, 30.0)
        self.min_cycles_spin.setValue(3.0)
        col_pairs.append(("Min cycles", self.min_cycles_spin))

        self.max_cycles_spin = QDoubleSpinBox()
        self.max_cycles_spin.setRange(1.0, 30.0)
        self.max_cycles_spin.setValue(12.0)
        col_pairs.append(("Max cycles", self.max_cycles_spin))

        self.fixed_cycles_check = QCheckBox("Fixed cycles")
        col_pairs.append((None, self.fixed_cycles_check))

        self.theta_freq_spin = QDoubleSpinBox()
        self.theta_freq_spin.setRange(0.1, 30.0)
        self.theta_freq_spin.setValue(7.0)
        col_pairs.append(("\u03b8 center (Hz)", self.theta_freq_spin))

        self.target_sr_spin = QSpinBox()
        self.target_sr_spin.setRange(100, 30000)
        self.target_sr_spin.setValue(1000)
        col_pairs.append(("Target SR (Hz)", self.target_sr_spin))

        self.csd_check = QCheckBox("Use CSD")
        self.csd_check.stateChanged.connect(self._on_csd_toggled)
        col_pairs.append((None, self.csd_check))

        self.csd_spacing_spin = QDoubleSpinBox()
        self.csd_spacing_spin.setRange(1.0, 500.0)
        self.csd_spacing_spin.setValue(20.0)
        col_pairs.append(("CSD spacing (\u00b5m)", self.csd_spacing_spin))

        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(CMAP_OPTIONS)
        self.cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        col_pairs.append(("Colormap", self.cmap_combo))

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_channel(self, channel: int):
        self.channel_spin.setValue(int(channel))
        self._on_csd_toggled(self.csd_check.checkState())

    def _sync_time_range_from_engine(self):
        if self.engine.data_loaded:
            self.start_spin.setRange(0.0, self.engine.total_duration)
            self.end_spin.setRange(0.01, self.engine.total_duration)
            self.end_spin.setValue(min(10.0, self.engine.total_duration))

    # ------------------------------------------------------------------
    # CSD availability feedback
    # ------------------------------------------------------------------

    def _on_csd_toggled(self, _state):
        if not self.csd_check.isChecked():
            self.csd_status_label.setText("")
            return
        result = self.analyzer.check_csd_availability(
            self.channel_spin.value(), self.csd_spacing_spin.value()
        )
        color = "#4caf50" if result["available"] else "#e57373"
        self.csd_status_label.setText(result["message"])
        self.csd_status_label.setStyleSheet(f"color: {color}; font-size: 11px;")

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------

    def _collect_params(self) -> PhaseAmplitudeParams:
        return PhaseAmplitudeParams(
            channel=self.channel_spin.value(),
            start_time=self.start_spin.value(),
            end_time=self.end_spin.value(),
            phase_low=self.phase_low_spin.value(),
            phase_high=self.phase_high_spin.value(),
            freq_low=self.freq_low_spin.value(),
            freq_high=self.freq_high_spin.value(),
            freq_step=self.freq_step_spin.value(),
            min_cycles=self.min_cycles_spin.value(),
            max_cycles=self.max_cycles_spin.value(),
            fixed_cycles=self.fixed_cycles_check.isChecked(),
            theta_freq=self.theta_freq_spin.value(),
            target_sr=self.target_sr_spin.value(),
            use_csd=self.csd_check.isChecked(),
            csd_spacing=self.csd_spacing_spin.value(),
        )

    def _on_compute_clicked(self):
        if not self.engine.data_loaded:
            QMessageBox.warning(self, "No data", "No data is currently loaded.")
            return

        if self.start_spin.value() >= self.end_spin.value():
            QMessageBox.warning(self, "Invalid range", "Start time must be before end time.")
            return

        if self.phase_low_spin.value() >= self.phase_high_spin.value():
            QMessageBox.warning(self, "Invalid theta band", "\u03b8 low must be below \u03b8 high.")
            return

        if self.freq_low_spin.value() >= self.freq_high_spin.value():
            QMessageBox.warning(self, "Invalid frequency range", "Freq low must be below freq high.")
            return

        params = self._collect_params()

        self.compute_btn.setEnabled(False)
        self.compute_btn.setText("Computing...")
        self.status_label.setText("Computing wavelet transform, please wait...")

        self._compute_thread = _ComputeThread(
            self.analyzer, self.engine.data, self.engine.sr, params, parent=self
        )
        self._compute_thread.finished_ok.connect(self._on_compute_finished)
        self._compute_thread.failed.connect(self._on_compute_failed)
        self._compute_thread.start()

    def _on_compute_failed(self, message: str):
        self.compute_btn.setEnabled(True)
        self.compute_btn.setText("Compute")
        self.status_label.setText(f"Error: {message}")
        QMessageBox.critical(self, "Computation failed", message)

    def _on_compute_finished(self, results: dict):
        self.compute_btn.setEnabled(True)
        self.compute_btn.setText("Compute")
        self._last_results = results
        self._render(results)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self, results: dict):
        freqvector = results["frequencies"]
        phase_center_bins = results["phase_center_bins"]
        phase_energy_norm = results["phase_energy_norm"]

        # Same 2-theta-cycle wraparound display as the original: tile the
        # phase axis [-pi,pi] twice so a full theta cycle is visible with
        # room to see it repeat, which is the standard phase-amplitude
        # coupling display convention (avoids an artificial cut at +-180).
        phase_2cycles = np.hstack([phase_center_bins, phase_center_bins + 2 * math.pi]) + math.pi
        phase_deg = np.degrees(phase_2cycles)
        energy_2cycles = np.hstack([phase_energy_norm, phase_energy_norm])

        # pyqtgraph ImageItem expects (x, y) axis order in the array's
        # first two dims to match how setRect maps it -- transpose so
        # rows=phase (x), cols=freq (y) isn't required; instead we set
        # the image directly as (freq, phase) and use setRect to place
        # it in data coordinates, with axisOrder='row-major' handling
        # the rest.
        vmin = np.nanpercentile(energy_2cycles, 2)
        vmax = np.nanpercentile(energy_2cycles, 98)
        if vmin == vmax:
            vmin = np.nanmin(energy_2cycles)
            vmax = np.nanmax(energy_2cycles)

        self.image_item.setImage(energy_2cycles.T, autoLevels=False)
        # Place at true bin edges (not bin centers) so the image spans
        # exactly [0, 720] degrees and [freq_low, freq_high] Hz, rather
        # than being inset by half a bin width on each side.
        phase_bin_width_deg = np.degrees(results["phase_bins"][1] - results["phase_bins"][0])
        x0 = phase_deg[0] - phase_bin_width_deg / 2
        x1 = phase_deg[-1] + phase_bin_width_deg / 2
        freq_step = freqvector[1] - freqvector[0] if len(freqvector) > 1 else 1.0
        y0 = freqvector[0] - freq_step / 2
        y1 = freqvector[-1] + freq_step / 2
        self.image_item.setRect(x0, y0, x1 - x0, y1 - y0)

        self.colorbar.setLevels((vmin, vmax))
        self.image_item.setLevels((vmin, vmax))

        # Cosine reference curve, scaled into a thin band near the bottom
        # of the frequency axis, same purpose as the original's dashed
        # sine overlay: shows where the theta peak/trough falls.
        x_t = np.linspace(-math.pi, 3 * math.pi, 1000)
        sine_wave = np.cos(x_t)
        x_deg = np.degrees(x_t + math.pi)
        freq_range = freqvector[-1] - freqvector[0]
        sine_scale = freq_range * 0.05
        sine_offset = freqvector[0] + sine_scale * 2
        self.sine_curve.setData(x_deg, sine_scale * sine_wave + sine_offset)

        self.plot_widget.setXRange(0, 720, padding=0)
        self.plot_widget.setYRange(freqvector[0], freqvector[-1], padding=0)

        self._update_status_label(results)

    def _update_status_label(self, results: dict):
        depth = "unknown"
        for ch, y in zip(self.analyzer.channels, self.analyzer.ycoords):
            if ch == results["channel"]:
                depth = f"{y:.0f} \u00b5m"
                break

        csd_str = ""
        if results.get("use_csd") and results.get("csd_info") and results["csd_info"]["used"]:
            info = results["csd_info"]
            csd_str = (f" | CSD: ch{info['above_channel']}\u2191 ch{info['below_channel']}\u2193 "
                       f"(spacing {info['spacing']:.0f}\u00b5m)")

        self.status_label.setText(
            f"Channel {results['channel']} (depth {depth}){csd_str}  |  "
            f"\u03b8 {self.phase_low_spin.value():.0f}-{self.phase_high_spin.value():.0f} Hz  |  "
            f"Freq {results['frequencies'][0]:.0f}-{results['frequencies'][-1]:.0f} Hz  |  "
            f"SR after downsample: {results['sample_rate']:.0f} Hz"
        )

    def _on_cmap_changed(self, name: str):
        try:
            cmap = pg.colormap.get(name)
        except Exception:
            return
        self.colorbar.setColorMap(cmap)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if self._compute_thread is not None and self._compute_thread.isRunning():
            self._compute_thread.wait(2000)
        super().closeEvent(event)

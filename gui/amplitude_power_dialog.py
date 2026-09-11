"""
amplitude_power_dialog.py

Non-modal QDialog showing per-electrode band power across the full probe,
preserving electrode geometry so spatial patterns (e.g. "where is theta
strongest") are visible directly. Replaces
neuropixels_amplitude_analyzer.py's tkinter/matplotlib AmplitudeAnalyzer
GUI; the math lives in core/amplitude_power.py.

Rendering approach mirrors gui/probe_map_widget.py: a single batched
pyqtgraph ScatterPlotItem for all electrodes (one draw call), GPU pan/
zoom via ViewBox. Electrode color encodes power value via a colormap +
colorbar instead of the probe map's selection-state colors.

Computation runs on a QThread, same as the phase-amplitude dialog, since
filtering every channel on the probe can take a noticeable moment.

Usage
-----
    dialog = AmplitudePowerDialog(probe_data, engine, parent=main_window)
    dialog.show()   # non-modal
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QDoubleSpinBox,
    QSpinBox, QCheckBox, QPushButton, QComboBox, QMessageBox, QWidget,
)

from core.amplitude_power import AmplitudePowerAnalyzer, AmplitudePowerParams
from core.trace_engine import TraceEngine


CMAP_OPTIONS = [
    "viridis", "plasma", "inferno", "magma", "cividis",
    "CET-D1", "CET-D4", "CET-L17",
]

BG_COLOR = "#1e1e1e"
GRID_COLOR = (255, 255, 255, 15)

# Common named bands, purely a convenience preset -- picking one just
# fills in the low/high spinboxes, nothing more.
BAND_PRESETS = {
    "Custom": None,
    "Delta (1-4 Hz)": (1.0, 4.0),
    "Theta (4-12 Hz)": (4.0, 12.0),
    "Alpha (8-13 Hz)": (8.0, 13.0),
    "Beta (13-30 Hz)": (13.0, 30.0),
    "Low Gamma (30-60 Hz)": (30.0, 60.0),
    "High Gamma (60-150 Hz)": (60.0, 150.0),
}


class _ComputeThread(QThread):
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, analyzer: AmplitudePowerAnalyzer, raw_data,
                 sample_rate: float, params: AmplitudePowerParams, parent=None):
        super().__init__(parent)
        self.analyzer = analyzer
        self.raw_data = raw_data
        self.sample_rate = sample_rate
        self.params = params

    def run(self):
        try:
            result = self.analyzer.compute(self.raw_data, self.sample_rate, self.params)
            self.finished_ok.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class AmplitudePowerDialog(QDialog):
    """
    Non-modal dialog: spatial band-power map across the whole probe.
    """

    def __init__(self, probe_data: dict, engine: TraceEngine, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Spatial Power Map")
        self.setModal(False)
        self.resize(700, 900)

        self.engine = engine
        self.analyzer = AmplitudePowerAnalyzer(probe_data)
        self._compute_thread: _ComputeThread | None = None
        self._last_result: dict | None = None

        self._build_ui()
        self._sync_time_range_from_engine()
        self._draw_geometry_only()

    # ------------------------------------------------------------------
    # UI scaffolding
    # ------------------------------------------------------------------

    def _build_ui(self):
        pg.setConfigOption("background", BG_COLOR)
        pg.setConfigOption("foreground", "#d0d0d0")

        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        layout.addLayout(self._build_param_grid())

        btn_row = QHBoxLayout()
        self.compute_btn = QPushButton("Compute")
        self.compute_btn.clicked.connect(self._on_compute_clicked)
        btn_row.addWidget(self.compute_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        # ---- Probe map with power coloring ----
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.setMenuEnabled(False)
        self.plot_widget.hideButtons()
        self.plot_widget.setLabel("bottom", "X", units="\u00b5m")
        self.plot_widget.setLabel("left", "Depth", units="\u00b5m")
        layout.addWidget(self.plot_widget, stretch=1)

        self.scatter = pg.ScatterPlotItem(
            size=14, pxMode=True, symbol="s",
            pen=pg.mkPen((0, 0, 0, 120), width=0.5),
            hoverable=True,
        )
        self.scatter.sigHovered.connect(self._on_hover)
        self.plot_widget.addItem(self.scatter)

        self.colorbar = pg.ColorBarItem(colorMap=pg.colormap.get("viridis"), interactive=False)
        # Deliberately not bound via setImageItem: ColorBarItem's binding
        # path assumes an ImageItem (it calls img.setLevels(...) on
        # whatever's passed in), which ScatterPlotItem doesn't implement
        # and raises AttributeError. Used here purely as a standalone
        # legend; colors are computed and applied to the scatter directly
        # in _render().
        self.plot_widget.plotItem.layout.addItem(self.colorbar, 2, 5)

        self.hover_label = QLabel("")
        self.hover_label.setStyleSheet("color: #ccc; font-size: 11px;")
        layout.addWidget(self.hover_label)

        self.status_label = QLabel("Set parameters and click Compute.")
        self.status_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self.status_label)

    def _build_param_grid(self) -> QGridLayout:
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        col_pairs = []

        self.start_spin = QDoubleSpinBox()
        self.start_spin.setDecimals(2)
        self.start_spin.setRange(0.0, 1e9)
        col_pairs.append(("Start (s)", self.start_spin))

        self.end_spin = QDoubleSpinBox()
        self.end_spin.setDecimals(2)
        self.end_spin.setRange(0.01, 1e9)
        self.end_spin.setValue(10.0)
        col_pairs.append(("End (s)", self.end_spin))

        self.band_preset_combo = QComboBox()
        self.band_preset_combo.addItems(BAND_PRESETS.keys())
        self.band_preset_combo.setCurrentText("Theta (4-12 Hz)")
        self.band_preset_combo.currentTextChanged.connect(self._on_band_preset_changed)
        col_pairs.append(("Band preset", self.band_preset_combo))

        self.low_freq_spin = QDoubleSpinBox()
        self.low_freq_spin.setRange(0.1, 5000.0)
        self.low_freq_spin.setValue(4.0)
        col_pairs.append(("Low (Hz)", self.low_freq_spin))

        self.high_freq_spin = QDoubleSpinBox()
        self.high_freq_spin.setRange(0.2, 5000.0)
        self.high_freq_spin.setValue(12.0)
        col_pairs.append(("High (Hz)", self.high_freq_spin))

        self.method_combo = QComboBox()
        self.method_combo.addItems(["bandpower", "hilbert"])
        col_pairs.append(("Method", self.method_combo))

        self.target_sr_spin = QSpinBox()
        self.target_sr_spin.setRange(100, 30000)
        self.target_sr_spin.setValue(1000)
        col_pairs.append(("Target SR (Hz)", self.target_sr_spin))

        self.notch_check = QCheckBox("Notch 50Hz")
        self.notch_check.setChecked(True)
        col_pairs.append((None, self.notch_check))

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
    # Setup helpers
    # ------------------------------------------------------------------

    def _sync_time_range_from_engine(self):
        if self.engine.data_loaded:
            self.start_spin.setRange(0.0, self.engine.total_duration)
            self.end_spin.setRange(0.01, self.engine.total_duration)
            self.end_spin.setValue(min(10.0, self.engine.total_duration))

    def _draw_geometry_only(self):
        """Show electrode positions in neutral gray before any compute,
        same idea as the probe map's initial unselected state -- lets you
        see the layout immediately rather than a blank plot."""
        x = self.analyzer.xcoords
        y = self.analyzer.ycoords
        self.scatter.setData(
            x=x, y=y,
            brush=pg.mkBrush((100, 100, 100, 180)),
        )
        if len(x) > 0:
            pad_x = max((x.max() - x.min()) * 0.15, 30)
            pad_y = max((y.max() - y.min()) * 0.03, 30)
            self.plot_widget.getViewBox().setRange(
                xRange=(x.min() - pad_x, x.max() + pad_x),
                yRange=(y.min() - pad_y, y.max() + pad_y),
                padding=0,
            )

    def _on_band_preset_changed(self, name: str):
        band = BAND_PRESETS.get(name)
        if band is None:
            return
        low, high = band
        self.low_freq_spin.blockSignals(True)
        self.high_freq_spin.blockSignals(True)
        self.low_freq_spin.setValue(low)
        self.high_freq_spin.setValue(high)
        self.low_freq_spin.blockSignals(False)
        self.high_freq_spin.blockSignals(False)

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------

    def _collect_params(self) -> AmplitudePowerParams:
        return AmplitudePowerParams(
            start_time=self.start_spin.value(),
            end_time=self.end_spin.value(),
            low_freq=self.low_freq_spin.value(),
            high_freq=self.high_freq_spin.value(),
            method=self.method_combo.currentText(),
            target_sr=self.target_sr_spin.value(),
            notch_freq=50.0 if self.notch_check.isChecked() else None,
        )

    def _on_compute_clicked(self):
        if not self.engine.data_loaded:
            QMessageBox.warning(self, "No data", "No data is currently loaded.")
            return
        if self.start_spin.value() >= self.end_spin.value():
            QMessageBox.warning(self, "Invalid range", "Start time must be before end time.")
            return
        if self.low_freq_spin.value() >= self.high_freq_spin.value():
            QMessageBox.warning(self, "Invalid band", "Low frequency must be below high frequency.")
            return

        params = self._collect_params()

        self.compute_btn.setEnabled(False)
        self.compute_btn.setText("Computing...")
        self.status_label.setText("Computing power across all electrodes, please wait...")

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

    def _on_compute_finished(self, result: dict):
        self.compute_btn.setEnabled(True)
        self.compute_btn.setText("Compute")
        self._last_result = result
        self._render(result)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self, result: dict):
        values = result["values"]
        x = result["xcoords"]
        y = result["ycoords"]

        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = float(np.min(valid)), float(np.max(valid))
            if vmax - vmin < 1e-12:
                vmin -= 1
                vmax += 1

        cmap = pg.colormap.get(self.cmap_combo.currentText())
        norm = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
        colors = cmap.map(norm, mode="qcolor")
        brushes = [pg.mkBrush(c) for c in colors]

        self.scatter.setData(x=x, y=y, brush=brushes)
        self._hover_values = values  # for hover readout
        self._hover_channels = result["channels"]

        self.colorbar.setColorMap(cmap)
        self.colorbar.setLevels((vmin, vmax))

        params: AmplitudePowerParams = result["params"]
        self.status_label.setText(
            f"{len(values)} electrodes  |  "
            f"{params.low_freq:.1f}-{params.high_freq:.1f} Hz  |  "
            f"Method: {params.method}  |  "
            f"{params.start_time:.2f}s\u2013{params.end_time:.2f}s  |  "
            f"Range: {vmin:.3g} \u2013 {vmax:.3g}"
        )

    def _on_hover(self, plot_item, points, ev):
        if not points or self._last_result is None:
            self.hover_label.setText("")
            return
        idx = points[0].index()
        ch = self._hover_channels[idx]
        val = self._hover_values[idx]
        self.hover_label.setText(f"Channel {ch}: {val:.4g}")

    def _on_cmap_changed(self, name: str):
        if self._last_result is not None:
            self._render(self._last_result)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if self._compute_thread is not None and self._compute_thread.isRunning():
            self._compute_thread.wait(2000)
        super().closeEvent(event)

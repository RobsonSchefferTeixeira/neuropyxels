"""
comodulogram_dialog.py

Non-modal QDialog for phase-amplitude comodulogram analysis: sweeps a
grid of phase frequencies (with a fixed phase bandwidth per band) and
amplitude center frequencies, and renders the Tort modulation index as
a phase-frequency x amplitude-frequency heatmap.

Structure mirrors gui/phase_amplitude_dialog.py closely:
  - 4-column parameter grid at the top
  - Compute button + CSD status label
  - pyqtgraph PlotWidget with an ImageItem heatmap and a ColorBarItem
  - Status line

The math lives in core/comodulogram.py.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QDoubleSpinBox,
    QSpinBox, QCheckBox, QPushButton, QComboBox, QMessageBox, QWidget,
    QProgressBar, QSizePolicy,
)
from PyQt6.QtGui import QPainter

from core.comodulogram import ComodulogramAnalyzer, ComodulogramParams
from core.trace_engine import TraceEngine
from core.colormaps import COLORMAP_NAMES, resolve_name, get_pg_colormap
from scipy.ndimage import zoom as ndimage_zoom

def _pin_button(btn: QPushButton, min_width: int | None = None) -> QPushButton:
    """Keep a button at its natural width when the dialog resizes."""
    btn.setSizePolicy(
        QSizePolicy.Policy.Fixed, btn.sizePolicy().verticalPolicy()
    )
    if min_width is not None:
        btn.setMinimumWidth(int(min_width))
    return btn


class _ComputeThread(QThread):
    """Runs ComodulogramAnalyzer.compute() off the UI thread."""

    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, int)

    def __init__(self, analyzer: ComodulogramAnalyzer, raw_data,
                 sample_rate: float, params: ComodulogramParams,
                 parent=None):
        super().__init__(parent)
        self.analyzer = analyzer
        self.raw_data = raw_data
        self.sample_rate = sample_rate
        self.params = params

    def run(self):
        try:
            result = self.analyzer.compute(
                self.raw_data, self.sample_rate, self.params,
                progress_cb=self._on_progress,
            )
            self.finished_ok.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))

    def _on_progress(self, done: int, total: int):
        self.progress.emit(int(done), int(total))


class ComodulogramDialog(QDialog):
    """Non-modal comodulogram dialog."""

    def __init__(self, probe_data: dict, engine: TraceEngine, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Comodulogram Analysis")
        self.setModal(False)
        self.resize(1050, 800)

        self.engine = engine
        self.analyzer = ComodulogramAnalyzer(probe_data)
        self._compute_thread: _ComputeThread | None = None
        self._last_results: dict | None = None

        self._build_ui()
        self._sync_time_range_from_engine()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_channel(self, channel: int):
        self.channel_spin.setValue(int(channel))
        self._on_csd_toggled(self.csd_check.checkState())

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
        _pin_button(self.compute_btn, min_width=110)
        btn_row.addWidget(self.compute_btn)

        self.csd_status_label = QLabel("")
        self.csd_status_label.setStyleSheet("color: #999; font-size: 11px;")
        btn_row.addWidget(self.csd_status_label, stretch=1)
        layout.addLayout(btn_row)

        # ---- Progress bar ----
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(14)
        layout.addWidget(self.progress_bar)

        # ---- Heatmap ----
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel("bottom", "Phase frequency", units="Hz")
        self.plot_widget.setLabel("left", "Amplitude frequency", units="Hz")
        self.plot_widget.setMenuEnabled(False)

        self.plot_widget.getPlotItem().getViewBox().disableAutoRange()
        self.plot_widget.setAntialiasing(True)
        self.plot_widget.setRenderHint(
            QPainter.RenderHint.SmoothPixmapTransform, True
        )

        self.image_item = pg.ImageItem()
        self.plot_widget.addItem(self.image_item)

        self.colorbar = pg.ColorBarItem(
            colorMap=pg.colormap.get("viridis"),
            label="MI",
        )
        self.colorbar.setImageItem(
            self.image_item, insert_in=self.plot_widget.plotItem
        )

        layout.addWidget(self.plot_widget, stretch=1)

        self.status_label = QLabel("Set parameters and click Compute.")
        self.status_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self.status_label)

    def _build_param_grid(self) -> QGridLayout:
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(6)
        grid.setContentsMargins(0, 0, 0, 0)

        pairs = []

        # ---- Channel / Start / End / Full range ----
        self.channel_spin = QSpinBox()
        self.channel_spin.setRange(0, 100000)
        pairs.append(("Channel", self.channel_spin))

        self.start_spin = QDoubleSpinBox()
        self.start_spin.setDecimals(3)
        self.start_spin.setRange(0.0, 1e9)
        pairs.append(("Start (s)", self.start_spin))

        self.end_spin = QDoubleSpinBox()
        self.end_spin.setDecimals(3)
        self.end_spin.setRange(0.01, 1e9)
        self.end_spin.setValue(10.0)
        pairs.append(("End (s)", self.end_spin))

        self.full_range_btn = QPushButton("Full range")
        self.full_range_btn.setToolTip(
            "Set the analysis interval to the minimum and maximum "
            "available time."
        )
        self.full_range_btn.setAutoDefault(False)
        self.full_range_btn.setDefault(False)
        self.full_range_btn.clicked.connect(self._set_full_available_range)
        _pin_button(self.full_range_btn, min_width=90)
        pairs.append((None, self.full_range_btn))

        # ---- Phase-frequency grid ----
        self.phase_low_spin = QDoubleSpinBox()
        self.phase_low_spin.setRange(0.1, 200.0)
        self.phase_low_spin.setValue(2.0)
        pairs.append(("Phase low (Hz)", self.phase_low_spin))

        self.phase_high_spin = QDoubleSpinBox()
        self.phase_high_spin.setRange(0.2, 200.0)
        self.phase_high_spin.setValue(20.0)
        pairs.append(("Phase high (Hz)", self.phase_high_spin))

        self.phase_step_spin = QDoubleSpinBox()
        self.phase_step_spin.setRange(0.1, 50.0)
        self.phase_step_spin.setValue(2.0)
        pairs.append(("Phase step (Hz)", self.phase_step_spin))

        self.phase_bw_spin = QDoubleSpinBox()
        self.phase_bw_spin.setRange(0.5, 50.0)
        self.phase_bw_spin.setValue(4.0)
        self.phase_bw_spin.setToolTip(
            "Bandwidth of each phase filter. A phase center of 6 Hz "
            "with bandwidth 4 Hz filters the signal from 6 to 10 Hz."
        )
        pairs.append(("Phase bandwidth (Hz)", self.phase_bw_spin))

        # ---- Amplitude-frequency grid ----
        self.amp_low_spin = QDoubleSpinBox()
        self.amp_low_spin.setRange(1.0, 5000.0)
        self.amp_low_spin.setValue(20.0)
        pairs.append(("Amp low (Hz)", self.amp_low_spin))

        self.amp_high_spin = QDoubleSpinBox()
        self.amp_high_spin.setRange(2.0, 5000.0)
        self.amp_high_spin.setValue(200.0)
        pairs.append(("Amp high (Hz)", self.amp_high_spin))

        self.amp_step_spin = QDoubleSpinBox()
        self.amp_step_spin.setRange(0.5, 100.0)
        self.amp_step_spin.setValue(5.0)
        pairs.append(("Amp step (Hz)", self.amp_step_spin))

        # ---- Advanced / misc ----
        self.target_sr_spin = QSpinBox()
        self.target_sr_spin.setRange(100, 30000)
        self.target_sr_spin.setValue(1000)
        self.target_sr_spin.setToolTip(
            "The signal is decimated to this rate before analysis."
        )
        pairs.append(("Target SR (Hz)", self.target_sr_spin))

        self.n_bins_spin = QSpinBox()
        self.n_bins_spin.setRange(4, 72)
        self.n_bins_spin.setValue(18)
        self.n_bins_spin.setToolTip(
            "Number of phase bins for the Tort modulation index. "
            "18 is the conventional default."
        )
        pairs.append(("Phase bins", self.n_bins_spin))

        self.csd_check = QCheckBox("Use CSD")
        self.csd_check.stateChanged.connect(self._on_csd_toggled)
        pairs.append((None, self.csd_check))

        self.csd_spacing_spin = QDoubleSpinBox()
        self.csd_spacing_spin.setRange(1.0, 500.0)
        self.csd_spacing_spin.setValue(20.0)
        pairs.append(("CSD spacing (µm)", self.csd_spacing_spin))

        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(COLORMAP_NAMES)
        self.cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        pairs.append(("Colormap", self.cmap_combo))

        self.reverse_cmap_check = QCheckBox("Reverse colormap")
        self.reverse_cmap_check.toggled.connect(
            lambda _checked: self._on_cmap_changed(
                self.cmap_combo.currentText()
            )
        )
        pairs.append((None, self.reverse_cmap_check))

        # ---- Layout into a fixed-width grid ----
        n_cols = 4
        row = 0
        col = 0
        for label_text, widget in pairs:
            widget.setSizePolicy(
                QSizePolicy.Policy.Fixed, widget.sizePolicy().verticalPolicy()
            )
            cell = QWidget()
            cell.setSizePolicy(
                QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
            )
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(2)
            if label_text is not None:
                lbl = QLabel(label_text)
                lbl.setStyleSheet("color: #aaa; font-size: 10px;")
                cell_layout.addWidget(lbl)
            else:
                spacer = QLabel(" ")
                spacer.setStyleSheet(
                    "color: transparent; font-size: 10px;"
                )
                cell_layout.addWidget(spacer)
            cell_layout.addWidget(widget)
            grid.addWidget(cell, row, col)
            col += 1
            if col >= n_cols:
                col = 0
                row += 1

        grid.setColumnStretch(n_cols, 1)
        return grid

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _sync_time_range_from_engine(self):
        if self.engine.data_loaded:
            t_min, t_max = self.engine.get_time_range()
            self.start_spin.blockSignals(True)
            self.end_spin.blockSignals(True)
            self.start_spin.setRange(t_min, t_max)
            self.end_spin.setRange(t_min + 1e-3, t_max)
            self.start_spin.setValue(t_min)
            self.end_spin.setValue(t_min + min(10.0, t_max - t_min))
            self.start_spin.blockSignals(False)
            self.end_spin.blockSignals(False)

    def _set_full_available_range(self):
        if not self.engine.data_loaded:
            return
        t_min, t_max = self.engine.get_time_range()
        self.start_spin.setValue(t_min)
        self.end_spin.setValue(t_max)

    # ------------------------------------------------------------------
    # CSD feedback
    # ------------------------------------------------------------------

    def _on_csd_toggled(self, _state):
        if not self.csd_check.isChecked():
            self.csd_status_label.setText("")
            return
        result = self.analyzer.check_csd_availability_distance(
            self.channel_spin.value(),
            self.csd_spacing_spin.value(),
        )
        color = "#4caf50" if result.get("available") else "#e57373"
        self.csd_status_label.setText(result.get("message", ""))
        self.csd_status_label.setStyleSheet(
            f"color: {color}; font-size: 11px;"
        )

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------

    def _collect_params(self) -> ComodulogramParams:
        return ComodulogramParams(
            channel=int(self.channel_spin.value()),
            start_time=float(self.start_spin.value()),
            end_time=float(self.end_spin.value()),
            phase_freq_low=float(self.phase_low_spin.value()),
            phase_freq_high=float(self.phase_high_spin.value()),
            phase_freq_step=float(self.phase_step_spin.value()),
            phase_bandwidth=float(self.phase_bw_spin.value()),
            amp_freq_low=float(self.amp_low_spin.value()),
            amp_freq_high=float(self.amp_high_spin.value()),
            amp_freq_step=float(self.amp_step_spin.value()),
            n_bins=int(self.n_bins_spin.value()),
            target_sr=float(self.target_sr_spin.value()),
            use_csd=bool(self.csd_check.isChecked()),
            csd_spacing=float(self.csd_spacing_spin.value()),
        )

    def _on_compute_clicked(self):
        if not self.engine.data_loaded:
            QMessageBox.warning(
                self, "No data", "No data is currently loaded."
            )
            return

        if self.start_spin.value() >= self.end_spin.value():
            QMessageBox.warning(
                self, "Invalid range",
                "Start time must be before end time."
            )
            return

        if self.phase_low_spin.value() >= self.phase_high_spin.value():
            QMessageBox.warning(
                self, "Invalid phase range",
                "Phase low must be below phase high."
            )
            return

        if self.amp_low_spin.value() >= self.amp_high_spin.value():
            QMessageBox.warning(
                self, "Invalid amplitude range",
                "Amplitude low must be below amplitude high."
            )
            return

        if self.phase_step_spin.value() <= 0 or self.amp_step_spin.value() <= 0:
            QMessageBox.warning(
                self, "Invalid step",
                "Frequency step must be positive."
            )
            return

        params = self._collect_params()

        self.compute_btn.setEnabled(False)
        self.compute_btn.setText("Computing…")
        self.progress_bar.setValue(0)
        self.status_label.setText("Computing comodulogram…")

        self._compute_thread = _ComputeThread(
            self.analyzer, self.engine.data, self.engine.sr,
            params, parent=self,
        )
        self._compute_thread.progress.connect(self._on_progress)
        self._compute_thread.finished_ok.connect(self._on_compute_finished)
        self._compute_thread.failed.connect(self._on_compute_failed)
        self._compute_thread.start()

    def _on_progress(self, done: int, total: int):
        pct = int(round(100 * done / max(1, total)))
        self.progress_bar.setValue(pct)
        self.status_label.setText(
            f"Computing comodulogram… {done}/{total} pairs"
        )

    def _on_compute_failed(self, message: str):
        self.compute_btn.setEnabled(True)
        self.compute_btn.setText("Compute")
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Error: {message}")
        QMessageBox.critical(self, "Computation failed", message)

    def _on_compute_finished(self, results: dict):
        self.compute_btn.setEnabled(True)
        self.compute_btn.setText("Compute")
        self.progress_bar.setValue(100)
        self._last_results = results
        self._render(results)

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _render(self, results: dict):
        comod = results["comodulogram"]
        phase_freqs = results["phase_frequencies"]
        amp_freqs = results["amplitude_frequencies"]

        # Display limits: drop NaNs when computing percentiles.
        finite = comod[np.isfinite(comod)]
        if finite.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin = float(np.nanpercentile(finite, 2))
            vmax = float(np.nanpercentile(finite, 98))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmin = float(np.nanmin(finite))
                vmax = float(np.nanmax(finite))
                if vmax <= vmin:
                    vmax = vmin + 1.0

        # ---- Smooth the coarse MI grid before display ----
        # The comodulogram is computed on a coarse (n_phase, n_amp)
        # grid -- typically 10 x 37 for defaults. Rendering that grid
        # directly gives a pixelated heatmap; the original matplotlib
        # contourf view interpolates the same coarse grid into smooth
        # filled contours. We do the equivalent here: up-sample the
        # coarse matrix with bicubic interpolation so pyqtgraph draws
        # a continuous field, keeping the fast pyqtgraph interaction
        # (pan/zoom, hover) instead of embedding a matplotlib canvas.
        #
        # The interpolation is DISPLAY-ONLY -- the underlying MI values
        # are unchanged; only how they are shown between grid points
        # is smoothed. Any reader who wants per-cell values should
        # read the status line / export path (not yet implemented).
        coarse = comod.T  # transpose once for (rows=amp, cols=phase)
        rows, cols = coarse.shape

        # Upsample factors chosen so the displayed image is at least
        # ~300 px on each axis at typical dialog sizes. Cap at 16x so
        # a tiny grid doesn't produce a huge array.
        up_factor_phase = max(1, min(16, 300 // max(1, cols)))
        up_factor_amp = max(1, min(16, 400 // max(1, rows)))

        # NaN-safe: ndimage.zoom on NaN-propagating data would smear
        # the NaNs across the whole image. Interpolate on the finite
        # values, and mark interpolated cells that span a NaN region
        # as NaN so they stay transparent.
        if np.any(~np.isfinite(coarse)):
            # Replace NaN with nearest-neighbor fill BEFORE zoom so
            # zoom has clean data; then mask back the originally-NaN
            # cells and any upsampled cell that fell in a NaN gap.
            nan_mask = ~np.isfinite(coarse)
            filled = coarse.copy()
            if np.any(np.isfinite(coarse)):
                fill_value = float(np.nanmean(coarse))
                filled[nan_mask] = fill_value
            else:
                filled[:] = 0.0
            smooth = ndimage_zoom(
                filled,
                zoom=(up_factor_amp, up_factor_phase),
                order=3,   # bicubic
                mode="nearest",
            )
            # Mask: upsample the NaN mask with nearest-neighbour so
            # the final image has NaN where the coarse cell was NaN.
            smooth_mask = ndimage_zoom(
                nan_mask.astype(np.uint8),
                zoom=(up_factor_amp, up_factor_phase),
                order=0,   # nearest
                mode="nearest",
            ).astype(bool)
            smooth[smooth_mask] = np.nan
        else:
            smooth = ndimage_zoom(
                coarse,
                zoom=(up_factor_amp, up_factor_phase),
                order=3,
                mode="nearest",
            )

        # ImageItem with SmoothPixmapTransform gives a second layer of
        # blur when the display is larger than the array; combined with
        # the bicubic upsample above, the heatmap reads as a continuous
        # field rather than a set of cells.
        self.image_item.setImage(smooth, autoLevels=False)
        self.image_item.setOpts(
            autoDownsample=False,
            smooth=True,
        )
        self.image_item.setLevels((vmin, vmax))

        phase_step = (
            phase_freqs[1] - phase_freqs[0]
            if len(phase_freqs) > 1 else 1.0
        )
        amp_step = (
            amp_freqs[1] - amp_freqs[0]
            if len(amp_freqs) > 1 else 1.0
        )

        x0 = phase_freqs[0] - phase_step / 2.0
        x1 = phase_freqs[-1] + phase_step / 2.0
        y0 = amp_freqs[0] - amp_step / 2.0
        y1 = amp_freqs[-1] + amp_step / 2.0

        self.image_item.setRect(x0, y0, x1 - x0, y1 - y0)

        resolved = resolve_name(
            self.cmap_combo.currentText(),
            reverse=self.reverse_cmap_check.isChecked(),
        )
        try:
            self.image_item.setColorMap(get_pg_colormap(resolved))
        except Exception as exc:
            print(f"[comodulogram] colormap {resolved!r} unavailable: {exc}")

        self.plot_widget.setXRange(phase_freqs[0], phase_freqs[-1], padding=0)
        self.plot_widget.setYRange(amp_freqs[0], amp_freqs[-1], padding=0)


        vb = self.plot_widget.getPlotItem().getViewBox()
        vb.setAutoVisible(x=False, y=False)
        vb.disableAutoRange()

        vb.setLimits(
            xMin=phase_freqs[0] - phase_step / 2.0,
            xMax=phase_freqs[-1] + phase_step / 2.0,
            yMin=amp_freqs[0] - amp_step / 2.0,
            yMax=amp_freqs[-1] + amp_step / 2.0,
        )

        self._update_status_label(results)

    def _update_status_label(self, results: dict):
        depth = "unknown"
        for ch, y in zip(self.analyzer.channels, self.analyzer.ycoords):
            if ch == results["channel"]:
                depth = f"{y:.0f} µm"
                break

        csd_str = ""
        csd_info = results.get("csd_info")
        if results.get("use_csd") and csd_info and csd_info.get("used"):
            csd_str = (
                f" | CSD: ch{csd_info['above_channel']}↑ "
                f"ch{csd_info['below_channel']}↓ "
                f"(spacing {csd_info['spacing']:.0f}µm)"
            )

        comod = results["comodulogram"]
        n_pairs = int(np.isfinite(comod).sum())

        self.status_label.setText(
            f"Channel {results['channel']} (depth {depth}){csd_str}  |  "
            f"phase {results['phase_frequencies'][0]:.0f}–"
            f"{results['phase_frequencies'][-1]:.0f} Hz  |  "
            f"amp {results['amplitude_frequencies'][0]:.0f}–"
            f"{results['amplitude_frequencies'][-1]:.0f} Hz  |  "
            f"SR after downsample: {results['sample_rate']:.0f} Hz  |  "
            f"{n_pairs} usable (phase, amp) pairs"
        )

    def _on_cmap_changed(self, _name: str):
        if self._last_results is not None:
            self._render(self._last_results)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if self._compute_thread is not None and self._compute_thread.isRunning():
            self._compute_thread.wait(2000)
        super().closeEvent(event)
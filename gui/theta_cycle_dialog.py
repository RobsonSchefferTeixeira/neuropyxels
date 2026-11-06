"""
theta_cycle_dialog.py

Non-modal dialog for theta cycle detection.

Detection runs on a QThread per requested channel; results are
collected into `cycles_by_channel`, which drives both the dialog's
table and the trace view's overlay.

The six landmark columns are positional (ref_1 .. ref_6). What each
position means depends on the current cycle_start_edge scheme; the
sidecar records this in its `landmark_names` key on save, and the
dialog restores it on load. A label above the table shows the
current scheme so the reader knows what they're looking at without
having to scroll back to the controls.

Save writes a single CSV (all channels, one table) plus a small
.meta.json sidecar into a `neuropyxels/` folder next to the loaded
.dat. Load reads both back, restoring detection parameters, the
cycle_start_edge scheme, and (for new files) the landmark names.

Editing: click a cycle's landmark circle on the trace view to select
it, or click its row in the table. Delete removes the selected cycle
(button, Delete key, or the trace view's own Delete handler), Ctrl+Z
undoes. Prev/Next Cycle anchors on the table's current selection and
survives deletes, undos, and repopulations: after a delete, the next
Prev/Next click lands on what would have been the adjacent survivor
before the deletion.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QDoubleSpinBox,
    QSpinBox, QCheckBox, QPushButton, QComboBox, QMessageBox, QWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QLineEdit, QFileDialog, QToolButton, QSizePolicy,
)
from PyQt6.QtGui import QShortcut, QKeySequence

from core.analysis_folder import (
    analysis_folder_for, ensure_analysis_folder,
)
from core.session_sidecar import save_sidecar, load_sidecar
from core.theta_cycle_detector import (
    ThetaCycleDetector, ThetaCycleParams, ThetaCycle,
    LANDMARK_NAMES_BY_MODE, VALID_CYCLE_START_EDGES,
)
from core.theta_cycle_export import export_cycles_to_csv, read_cycles_from_csv

from core.trace_engine import TraceEngine
from gui.theta_cycle_trace_view import ThetaCycleTraceViewWidget


class _CollapsibleSection(QWidget):
    """A titled, collapsible container. Used for the 'Advanced' section
    so the dialog stays readable without hiding controls the user
    needs access to.

    Emits `toggled(bool)` on expand/collapse, matching
    QGroupBox.toggled, so downstream code doesn't need to special-case
    this widget type.
    """

    toggled = pyqtSignal(bool)

    def __init__(self, title: str, expanded: bool = False, parent=None):
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._toggle_btn = QToolButton()
        self._toggle_btn.setText(title)
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setChecked(bool(expanded))
        self._toggle_btn.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self._toggle_btn.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self._toggle_btn.setStyleSheet(
            "QToolButton {"
            "  border: none;"
            "  padding: 2px 4px;"
            "  font-weight: bold;"
            "  color: #ccc;"
            "}"
            "QToolButton:hover { color: #fff; }"
        )
        self._toggle_btn.toggled.connect(self._on_toggled)
        outer.addWidget(self._toggle_btn)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(12, 4, 4, 4)
        self._content_layout.setSpacing(4)
        self._content.setVisible(bool(expanded))
        outer.addWidget(self._content)

    def _on_toggled(self, checked: bool):
        self._toggle_btn.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )
        self._content.setVisible(checked)
        self.toggled.emit(checked)

    def content_layout(self) -> QVBoxLayout:
        """Layout to which the caller adds widgets."""
        return self._content_layout

    def is_expanded(self) -> bool:
        return self._toggle_btn.isChecked()

    def set_expanded(self, expanded: bool):
        self._toggle_btn.setChecked(bool(expanded))


class _DetectThread(QThread):
    """Runs cycle detection across the requested channels off the UI
    thread. Emits results as {channel: list[ThetaCycle]}."""

    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, int)

    def __init__(self, channels, raw_data, sample_rate, start_idx, end_idx,
                 params, epoch_intervals=None, parent=None):
        super().__init__(parent)
        self.channels = channels
        self.raw_data = raw_data
        self.sample_rate = float(sample_rate)
        self.start_idx = int(start_idx)
        self.end_idx = int(end_idx)
        self.params = params
        self.epoch_intervals = list(epoch_intervals or [])
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

                params_for_channel = dataclasses.replace(
                    self.params,
                    epoch_intervals=self.epoch_intervals,
                )

                cycles = self.detector.detect(
                    signal, self.sample_rate, params_for_channel,
                    channel=ch,
                )

                # Detector returns sample indices relative to `signal`
                # (i.e. 0 = start_idx). Convert to raw recording samples.
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

    cyclesChanged = pyqtSignal(object)
    cycleSelected = pyqtSignal(int)
    eventFocused = pyqtSignal(int)

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
        self.resize(1050, 680)

        self.engine = engine
        self.trace_view = trace_view
        self.probe_data = probe_data
        self._detect_thread: _DetectThread | None = None

        self.cycles_by_channel: dict[int, list[ThetaCycle]] = {}
        self._selected_channel: int | None = None
        self._selected_index: int | None = None

        self._post_delete_anchor: int | None = None

        self._sample_offset = 0
        self._detect_n_samples = 0
        self._last_params_for_sidecar = None

        self._loaded_cycle_start_edge: str | None = None
        self._loaded_landmark_names: list[str] | None = None

        # ---- Autosave ----
        self._autosave_armed = False
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(600)
        self._autosave_timer.timeout.connect(self._autosave_now)

        self._build_ui()
        self._sync_time_range_from_engine()
        self._refresh_export_buttons()
        self._refresh_landmark_scheme_label()

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

        self.trace_view.thetaCyclesChanged.connect(self.set_cycles_from_trace)
        self.trace_view.thetaCycleSelected.connect(self.set_selected_cycle)
        self.trace_view.thetaEpochsChanged.connect(self._on_theta_epochs_changed)

        self._refresh_theta_epoch_availability()

    # ------------------------------------------------------------------
    # UI scaffolding
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        layout.addLayout(self._build_param_grid())
        layout.addWidget(self._build_advanced_section())

        btn_row = QHBoxLayout()
        self.detect_btn = QPushButton("Detect Theta Cycles")
        self.detect_btn.clicked.connect(self._on_detect_clicked)
        btn_row.addWidget(self.detect_btn)

        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color: #888; font-size: 10px;")
        btn_row.addWidget(self.progress_label, stretch=1)
        layout.addLayout(btn_row)

        # ---- Landmark scheme label ----
        self.landmark_scheme_label = QLabel("")
        self.landmark_scheme_label.setStyleSheet(
            "color: #ccc; font-size: 10px; font-style: italic;"
        )
        layout.addWidget(self.landmark_scheme_label)

        # ---- Pooled table ----
        # Six ref_N landmark columns (smp + s), plus duration, amp,
        # ratios. Column count = 1 (channel) + 6 (ref_smp) + 1 (dur_smp)
        # + 6 (ref_s) + 1 (dur_s) + 3 (amp + 2 ratios) = 18, matching
        # the previous layout's column count.
        self.table = QTableWidget(0, 18)
        self.table.setHorizontalHeaderLabels([
            "Channel",
            "ref_1_smp", "ref_2_smp", "ref_3_smp",
            "ref_4_smp", "ref_5_smp", "ref_6_smp",
            "duration_smp",
            "ref_1_s", "ref_2_s", "ref_3_s",
            "ref_4_s", "ref_5_s", "ref_6_s",
            "duration_s",
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

        # ---- Save / Load / Delete ----
        io_row = QHBoxLayout()
        self.export_btn = QPushButton("Save")
        self.export_btn.setAutoDefault(False)
        self.export_btn.clicked.connect(self._on_save_clicked)
        io_row.addWidget(self.export_btn)

        self.load_btn = QPushButton("Load")
        self.load_btn.setAutoDefault(False)
        self.load_btn.clicked.connect(self._on_load_clicked)
        io_row.addWidget(self.load_btn)

        self.delete_cycle_btn = QPushButton("Delete Selected Cycle")
        self.delete_cycle_btn.setAutoDefault(False)
        self.delete_cycle_btn.setDefault(False)
        self.delete_cycle_btn.setToolTip(
            "Remove the currently-selected theta cycle. Also available "
            "via the Delete key when the table has focus, or when the "
            "trace view has focus with a cycle selected there.\n"
            "Ctrl+Z undoes it."
        )
        self.delete_cycle_btn.setEnabled(False)
        self.delete_cycle_btn.clicked.connect(self._on_delete_selected_cycle)
        io_row.addWidget(self.delete_cycle_btn)

        io_row.addStretch(1)
        layout.addLayout(io_row)

        # Keep the Save / Load / Delete buttons at their natural width
        # so they don't stretch when the dialog is widened, and so a
        # status-label text change doesn't push them around.
        for btn in (self.export_btn, self.load_btn, self.delete_cycle_btn):
            btn.setSizePolicy(
                QSizePolicy.Policy.Fixed, btn.sizePolicy().verticalPolicy()
            )

        self.status_label = QLabel(
            "Select channels and click Detect Theta Cycles."
        )
        self.status_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self.status_label)

        # Delete key is scoped to the table via WidgetWithChildrenShortcut:
        # it fires when the table (or a descendant) has focus, but NOT
        # while the user is typing in a spinbox elsewhere in the dialog.
        # The trace view's own Delete handler still works when a cycle
        # is selected there.
        self._delete_shortcut = QShortcut(
            QKeySequence(Qt.Key.Key_Delete), self.table
        )
        self._delete_shortcut.setContext(
            Qt.ShortcutContext.WidgetWithChildrenShortcut
        )
        self._delete_shortcut.activated.connect(self._on_delete_selected_cycle)

        # Ctrl+Z is ALSO scoped to the table, not the whole window. If
        # it were WindowShortcut, it would pre-empt the trace view's own
        # Ctrl+Z handler (shortcuts fire before the focused widget's
        # keyPressEvent), which would silently change the trace view's
        # undo behavior whenever this dialog is open. Scoping to the
        # table keeps both surfaces consistent: Ctrl+Z in the table
        # undoes from the dialog, Ctrl+Z in the trace view undoes from
        # the trace view, and neither steals from the other.
        self._undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self.table)
        self._undo_shortcut.setContext(
            Qt.ShortcutContext.WidgetWithChildrenShortcut
        )
        self._undo_shortcut.activated.connect(self._on_undo_cycle_deletion)

    def _build_param_grid(self) -> QGridLayout:
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(6)
        grid.setContentsMargins(0, 0, 0, 0)
        pairs = []

        self.channel_edit = QLineEdit()
        self.channel_edit.setPlaceholderText("e.g. 12,13,14")
        self.channel_edit.setFixedWidth(140)

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

        self.full_range_btn = QPushButton("Full range")
        self.full_range_btn.setToolTip("Set the analysis interval to the minimum and maximum ""available time.")
        self.full_range_btn.setAutoDefault(False)
        self.full_range_btn.setDefault(False)
        self.full_range_btn.clicked.connect(self._set_full_available_range)
        self.full_range_btn.setFixedWidth(70)
        pairs.append((None, self.full_range_btn))

        self.only_in_epochs_check = QCheckBox("Only inside theta epochs")
        self.only_in_epochs_check.setToolTip(
            "Keep only cycles whose span [ref_1, ref_5] lies inside a "
            "detected theta epoch. Requires theta epochs to be defined "
            "on the main trace view."
        )
        self.only_in_epochs_check.setEnabled(False)
        pairs.append((None, self.only_in_epochs_check))

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
            "Minimum mean(theta)/mean(delta) ratio over the cycle's "
            "[ref_1, ref_5] span. Cycles below this are discarded."
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
            "Maximum mean(fast)/mean(theta) ratio over the cycle's "
            "[ref_1, ref_5] span. Cycles above this are discarded."
        )
        pairs.append(("fast/θ ratio max", self.fast_ratio_spin))

        self.edge_combo = QComboBox()
        self.edge_combo.addItems(list(VALID_CYCLE_START_EDGES))
        self.edge_combo.setToolTip(
            "Which event opens a cycle. The six landmark slots "
            "(ref_1..ref_6) are populated chronologically under the "
            "chosen convention; see the Landmark scheme label above "
            "the table after the first run."
        )
        self.edge_combo.currentTextChanged.connect(
            self._refresh_landmark_scheme_label
        )
        pairs.append(("Cycle start", self.edge_combo))

        n_cols = 4
        row = 0
        col = 0
        for label_text, widget in pairs:
            widget.setSizePolicy(QSizePolicy.Policy.Fixed, widget.sizePolicy().verticalPolicy())
            cell = QWidget()
            cell.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(2)
            if label_text is not None:
                lbl = QLabel(label_text)
                lbl.setStyleSheet("color: #aaa; font-size: 10px;")
                cell_layout.addWidget(lbl)
            else:
                spacer_label = QLabel(" ")
                spacer_label.setStyleSheet("color: transparent; font-size: 10px;")
                cell_layout.addWidget(spacer_label)
            cell_layout.addWidget(widget)

            grid.addWidget(cell, row, col)
            col += 1
            if col >= n_cols:
                col = 0
                row += 1

        grid.setColumnStretch(n_cols, 1)

        return grid

    def _build_advanced_section(self) -> QWidget:
        """Collapsible 'Advanced' section: peak-distance bounds,
        filter order, and trace display controls. Placed here so the
        dialog's main grid stays compact."""

        self.advanced_section = _CollapsibleSection(
            "Advanced", expanded=False, parent=self,
        )
        content = self.advanced_section.content_layout()

        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(6)
        grid.setContentsMargins(0, 0, 0, 0)

        pairs = []

        self.half_cycle_low_spin = QDoubleSpinBox()
        self.half_cycle_low_spin.setRange(0.5, 10.0)
        self.half_cycle_low_spin.setSingleStep(0.1)
        self.half_cycle_low_spin.setDecimals(2)
        self.half_cycle_low_spin.setValue(2.5)
        self.half_cycle_low_spin.setSuffix(" Hz")
        self.half_cycle_low_spin.setToolTip(
            "Slowest accepted half-cycle rate. Each theta cycle has "
            "two half-cycles (see the Landmark scheme label); a "
            "candidate is rejected if either half is slower than "
            "this. Lower = permissive; raise to reject unusually "
            "slow halves."
        )
        pairs.append(("Min half-cycle rate (Hz)", self.half_cycle_low_spin))

        self.half_cycle_high_spin = QDoubleSpinBox()
        self.half_cycle_high_spin.setRange(5.0, 30.0)
        self.half_cycle_high_spin.setSingleStep(0.5)
        self.half_cycle_high_spin.setDecimals(2)
        self.half_cycle_high_spin.setValue(15.0)
        self.half_cycle_high_spin.setSuffix(" Hz")
        self.half_cycle_high_spin.setToolTip(
            "Fastest accepted half-cycle rate. A candidate is "
            "rejected if either half is faster than this. Raise to "
            "admit faster halves; lower to reject noisy short "
            "excursions."
        )
        pairs.append(("Max half-cycle rate (Hz)", self.half_cycle_high_spin))

        self.filter_order_spin = QSpinBox()
        self.filter_order_spin.setRange(2, 8)
        self.filter_order_spin.setValue(4)
        self.filter_order_spin.setToolTip(
            "Butterworth order for the theta, delta, and fast band "
            "filters. 4 is conventional; 2 widens the effective "
            "passband; higher orders sharpen the cutoff but slow "
            "detection on long windows."
        )
        pairs.append(("Filter order", self.filter_order_spin))

        self.trace_gain_spin = QDoubleSpinBox()
        self.trace_gain_spin.setRange(0.05, 100.0)
        self.trace_gain_spin.setSingleStep(0.1)
        self.trace_gain_spin.setDecimals(2)
        self.trace_gain_spin.setValue(1.0)
        self.trace_gain_spin.setSuffix(" ×")
        self.trace_gain_spin.setToolTip(
            "Multiplier on the fixed reference scale of the theta-"
            "filtered overlay."
        )
        self.trace_gain_spin.valueChanged.connect(
            self._on_trace_gain_changed
        )
        pairs.append(("Trace gain", self.trace_gain_spin))

        self.trace_width_spin = QDoubleSpinBox()
        self.trace_width_spin.setRange(0.5, 5.0)
        self.trace_width_spin.setSingleStep(0.1)
        self.trace_width_spin.setDecimals(2)
        self.trace_width_spin.setValue(1.0)
        self.trace_width_spin.setSuffix(" px")
        self.trace_width_spin.setToolTip(
            "Stroke width of the theta-filtered overlay trace."
        )
        self.trace_width_spin.valueChanged.connect(
            self._on_trace_width_changed
        )
        pairs.append(("Trace width", self.trace_width_spin))

        self.detrend_check = QCheckBox("Detrend")
        self.detrend_check.setChecked(True)
        self.detrend_check.setToolTip(
            "Remove a linear trend from the filtered signals before "
            "computing envelopes. Turn off if the signal is already "
            "detrended or if you suspect the linear fit is stripping "
            "slow theta itself."
        )
        pairs.append((None, self.detrend_check))

        n_cols = 3
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
                spacer_label = QLabel(" ")
                spacer_label.setStyleSheet("color: transparent; font-size: 10px;")
                cell_layout.addWidget(spacer_label)
            cell_layout.addWidget(widget)
            grid.addWidget(cell, row, col)
            col += 1
            if col >= n_cols:
                col = 0
                row += 1

        # Same left-hugging stretch column as the main grid.
        grid.setColumnStretch(n_cols, 1)

        content.addLayout(grid)
        return self.advanced_section

    # ------------------------------------------------------------------
    # Landmark scheme label
    # ------------------------------------------------------------------

    def _refresh_landmark_scheme_label(self, *_args):
        """Update the label above the table to describe the scheme
        currently attached to the displayed cycles. Preference order:
        loaded scheme (from Load) > current combo (what a fresh Detect
        would use)."""
        if self._loaded_cycle_start_edge is not None:
            scheme = self._loaded_cycle_start_edge
            source = "loaded"
        else:
            scheme = self.edge_combo.currentText()
            source = "current"
        names = LANDMARK_NAMES_BY_MODE.get(scheme, [])
        joined = ", ".join(
            f"ref_{i + 1}={name}" for i, name in enumerate(names)
        )
        self.landmark_scheme_label.setText(
            f"Landmark scheme ({source}): {scheme} — {joined}"
        )

    # ------------------------------------------------------------------
    # Theta-epoch availability
    # ------------------------------------------------------------------

    def _refresh_theta_epoch_availability(self):
        has_theta = bool(getattr(self.trace_view, "theta_epochs", None))
        self.only_in_epochs_check.setEnabled(has_theta)
        if not has_theta and self.only_in_epochs_check.isChecked():
            self.only_in_epochs_check.setChecked(False)

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_theta_epoch_availability()

    def _on_theta_epochs_changed(self, _epochs: list):
        self._refresh_theta_epoch_availability()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _set_full_available_range(self):
        if not self.engine.data_loaded:
            return
        t_min, t_max = self.engine.get_time_range()
        self.start_spin.setValue(t_min)
        self.end_spin.setValue(t_max)

    def _sync_time_range_from_engine(self):
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
                raise ValueError(
                    f"'{part}' is not a valid channel number."
                )
        return out

    def _on_trace_gain_changed(self, value: float):
        if self.trace_view is not None:
            self.trace_view.set_theta_cycle_gain(float(value))

    def _on_trace_width_changed(self, value: float):
        if self.trace_view is not None:
            self.trace_view.set_theta_cycle_trace_width(float(value))

    def _collect_params(self) -> ThetaCycleParams:
        return ThetaCycleParams(
            channels=[],
            start_time=self.start_spin.value(),
            end_time=self.end_spin.value(),
            low_freq=self.low_spin.value(),
            high_freq=self.high_spin.value(),
            filter_order=int(self.filter_order_spin.value()),
            detrend=bool(self.detrend_check.isChecked()),
            min_theta_peak_percentile=self.min_amp_spin.value(),
            half_cycle_low_freq=float(self.half_cycle_low_spin.value()),
            half_cycle_high_freq=float(self.half_cycle_high_spin.value()),
            delta_correction=self.delta_check.isChecked(),
            delta_low_freq=self.delta_low_spin.value(),
            delta_high_freq=self.delta_high_spin.value(),
            theta_delta_ratio_min=self.delta_ratio_spin.value(),
            fast_correction=self.fast_check.isChecked(),
            fast_low_freq=self.fast_low_spin.value(),
            fast_theta_ratio_max=self.fast_ratio_spin.value(),
            cycle_start_edge=self.edge_combo.currentText(),
            channels_text=self.channel_edit.text().strip(),
            trace_gain=float(self.trace_gain_spin.value()),
            trace_width=float(self.trace_width_spin.value()),
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

        # Half-cycle rate must straddle the theta band. If min half
        # rate exceeds the theta band's low edge, no real theta cycle
        # can pair; if max half rate falls below the theta band's high
        # edge, likewise.
        if self.half_cycle_low_spin.value() > self.low_spin.value():
            QMessageBox.warning(
                self, "Half-cycle band too narrow",
                f"Min half-cycle rate ({self.half_cycle_low_spin.value()} Hz) "
                f"must be at or below theta low ({self.low_spin.value()} Hz). "
                f"Otherwise a full theta cycle's halves are all faster than "
                f"the pairing will accept."
            )
            return
        if self.half_cycle_high_spin.value() < self.high_spin.value():
            QMessageBox.warning(
                self, "Half-cycle band too narrow",
                f"Max half-cycle rate ({self.half_cycle_high_spin.value()} Hz) "
                f"must be at or above theta high ({self.high_spin.value()} Hz). "
                f"Otherwise a full theta cycle's halves are all slower than "
                f"the pairing will accept."
            )
            return

        params = self._collect_params()

        self.trace_view.set_theta_cycle_band(
            params.low_freq, params.high_freq,
        )

        start_idx = self.engine.timestamp_to_sample(self.start_spin.value())
        end_idx = self.engine.timestamp_to_sample(self.end_spin.value())

        epoch_intervals = []
        if self.only_in_epochs_check.isChecked():
            epoch_intervals = self._theta_epoch_intervals_for_cycle_detection(
                start_idx
            )
            if not epoch_intervals:
                QMessageBox.information(
                    self, "No theta epochs",
                    "Only-inside-theta-epochs is checked, but no theta "
                    "epochs are currently defined on the main trace view."
                )
                return

        self.detect_btn.setEnabled(False)
        self.detect_btn.setText("Detecting...")
        self.progress_label.setText("")
        self.status_label.setText("Running detection...")

        self._sample_offset = int(start_idx)
        self._detect_n_samples = max(0, int(end_idx) - int(start_idx))
        self._last_params_for_sidecar = params

        # Fresh detection: any previously-loaded scheme/names are no
        # longer associated with the in-memory cycles, and any pending
        # post-delete anchor refers to cycles that are about to be
        # replaced.
        self._loaded_cycle_start_edge = None
        self._loaded_landmark_names = None
        self._post_delete_anchor = None
        self._refresh_landmark_scheme_label()

        self._detect_thread = _DetectThread(
            channels, self.engine.data, self.engine.sr,
            start_idx, end_idx, params, epoch_intervals, parent=self,
        )

        self._detect_thread.progress.connect(self._on_progress)
        self._detect_thread.finished_ok.connect(self._on_detect_finished)
        self._detect_thread.failed.connect(self._on_detect_failed)
        self._detect_thread.start()

    def _theta_epoch_intervals_for_cycle_detection(
        self, cycle_start_idx: int
    ) -> list[tuple[int, int]]:
        epochs = getattr(self.trace_view, "theta_epochs", None) or []
        if not epochs:
            return []
        epoch_offset = int(
            getattr(self.trace_view, "theta_detection_sample_offset", 0)
        )
        intervals: list[tuple[int, int]] = []
        for ep in epochs:
            lo_abs = int(ep.start_sample) + epoch_offset
            hi_abs = int(ep.end_sample) + epoch_offset
            intervals.append((
                lo_abs - int(cycle_start_idx),
                hi_abs - int(cycle_start_idx),
            ))
        return intervals

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
        # The filtered overlay is only shown for channels the dialog
        # just ran on. This means channels that returned zero cycles
        # are still covered (their filtered trace is the diagnostic
        # value), but channels that were never included in this
        # detection run are not.
        self.trace_view.set_theta_cycle_overlay_channels(
            sorted(self.cycles_by_channel.keys())
        )

        self._refresh_export_buttons()
        self._refresh_landmark_scheme_label()
        total = sum(len(v) for v in results.values())
        n_ch = sum(1 for v in results.values() if v)
        self.status_label.setText(
            f"Detected {total} cycle(s) across {n_ch} channel(s)."
        )
        self._autosave_armed = True
        self._schedule_autosave()

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------

    def _populate_table(self):
        flat = self._flat_cycles()
        self.table.blockSignals(True)
        try:
            self.table.setRowCount(len(flat))
            for row, c in enumerate(flat):
                landmarks = c.landmarks()
                ref_smp = [int(x) for x in landmarks]
                dur_smp = ref_smp[4] - ref_smp[0]

                def t(smp):
                    return float(self.engine.sample_to_time(smp))

                vals = [str(c.channel)]
                vals.extend(str(x) for x in ref_smp)
                vals.append(str(dur_smp))
                vals.extend(f"{t(x):.3f}" for x in ref_smp)
                vals.append(f"{dur_smp / float(self.engine.sr):.3f}")
                vals.append(f"{c.amplitude:.3g}")
                vals.append(
                    "" if c.theta_delta_ratio is None
                    else f"{c.theta_delta_ratio:.3f}"
                )
                vals.append(
                    "" if c.fast_theta_ratio is None
                    else f"{c.fast_theta_ratio:.3f}"
                )

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

        # Nothing is selected after a repopulation, so the Delete
        # button must be disabled until the user picks a row.
        self._selected_index = None
        self._selected_channel = None
        self.delete_cycle_btn.setEnabled(False)

    def _flat_cycles(self) -> list[ThetaCycle]:
        flat: list[ThetaCycle] = []
        for ch in sorted(self.cycles_by_channel.keys()):
            flat.extend(self.cycles_by_channel[ch])
        flat.sort(key=lambda c: (c.channel, c.zero_crossing_start))
        return flat

    def _push_flat_cycles_to_trace(self):
        flat = self._flat_cycles()
        self.trace_view.set_theta_cycles(flat)
        self.cyclesChanged.emit(flat)

    def _on_table_selection_changed(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            # Only drop the anchor if the table is genuinely empty. A
            # transient "no rows selected" during a repopulation must
            # not clear the selection the Prev/Next buttons rely on.
            if self.table.rowCount() == 0:
                self._selected_index = None
                self._selected_channel = None
                self.delete_cycle_btn.setEnabled(False)
            return

        row = rows[0].row()
        self._selected_index = row
        self.delete_cycle_btn.setEnabled(True)

        flat = self._flat_cycles()
        if not (0 <= row < len(flat)):
            return
        target = flat[row]
        self._selected_channel = int(target.channel)

        # Explicit user selection supersedes any pending post-delete
        # anchor: the next Prev/Next should step from THIS row, not
        # from where a deletion left off.
        self._post_delete_anchor = None

        self.trace_view.set_theta_cycle_selected(row)
        self.cycleSelected.emit(row)

        self._ensure_channel_visible(target.channel)
        self._pan_to_event(target)
        self.eventFocused.emit(int(target.channel))

    def _pan_to_event(self, cycle: ThetaCycle):
        """Center the trace view on the cycle's second landmark
        (ref_2), which is always the first peak or peak-equivalent
        landmark of the oscillation under every scheme."""
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
        flat = self._flat_cycles()
        if not (0 <= flat_index < len(flat)):
            return
        # Explicit trace-view selection supersedes any pending
        # post-delete anchor for the same reason as a table click.
        self._post_delete_anchor = None
        self.table.blockSignals(True)
        self.table.selectRow(flat_index)
        self.table.blockSignals(False)
        self._selected_index = flat_index
        self._selected_channel = int(flat[flat_index].channel)
        self.delete_cycle_btn.setEnabled(True)

    def set_cycles_from_trace(self, cycles: list):
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
        flat = self._flat_cycles()
        if not flat:
            return

        # Anchor priority:
        #   1. _post_delete_anchor: the row index of the cycle most
        #      recently deleted. Because the list shrank by one, this
        #      index now holds what was the deleted cycle's successor.
        #      Next -> land on that successor (or clamp to the new last
        #      row if the deleted cycle was last); Prev -> land on the
        #      predecessor one row before.
        #   2. The table's current selection.
        #   3. self._selected_index (survives transient clears).
        #   4. Nothing selected: start from row 0 (next) or last (prev).
        if self._post_delete_anchor is not None:
            anchor_row = int(self._post_delete_anchor)
            self._post_delete_anchor = None
            if direction > 0:
                new_row = min(anchor_row, len(flat) - 1)
            else:
                new_row = max(0, anchor_row - 1)
        else:
            rows = self.table.selectionModel().selectedRows()
            if rows:
                current_row = int(rows[0].row())
            elif (self._selected_index is not None
                  and 0 <= self._selected_index < len(flat)):
                current_row = int(self._selected_index)
            else:
                current_row = -1 if direction > 0 else len(flat)
            new_row = current_row + direction
            new_row = max(0, min(new_row, len(flat) - 1))

        target = flat[new_row]
        self._ensure_channel_visible(int(target.channel))

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
        self._selected_channel = int(target.channel)
        self.delete_cycle_btn.setEnabled(True)

        self.status_label.setText(
            f"Cycle {new_row + 1} of {len(flat)} on CH{target.channel} "
            f"({'(next)' if direction > 0 else '(prev)'})"
        )

    def _on_delete_selected_cycle(self):
        """Remove the currently-selected cycle from the in-memory list,
        rebuild the table, push the new flat list to the trace view,
        and schedule an autosave.

        Deletion is routed through the trace view's
        delete_selected_theta_cycle() when possible, so the undo stack,
        the trace-view highlight, and the thetaCyclesChanged signal
        stay consistent with how the trace view's own Delete key
        behaves. When the trace view does not have this cycle selected
        (e.g. the user selected it in the table but not on the trace),
        the dialog mutates its own state directly and mirrors the
        signal sequence the trace view would have emitted.
        """
        if self._selected_index is None:
            return
        flat = self._flat_cycles()
        if not (0 <= self._selected_index < len(flat)):
            return

        target = flat[self._selected_index]
        deleted_row = int(self._selected_index)

        # If the trace view's selected index matches ours, prefer its
        # delete path so undo works transparently.
        trace_selected = getattr(self.trace_view, "_theta_cycle_selected", None)
        if trace_selected == self._selected_index:
            # Set the anchor BEFORE the trace view emits
            # thetaCyclesChanged, since that signal triggers
            # set_cycles_from_trace which repopulates the table and
            # would otherwise wipe any selection we set afterwards.
            self._post_delete_anchor = deleted_row
            deleted = self.trace_view.delete_selected_theta_cycle()
            if deleted:
                self.status_label.setText(
                    f"Deleted cycle on CH{target.channel}. "
                    f"Ctrl+Z to undo."
                )
                return
            # The trace view refused the delete (its selection had
            # already changed). Clear the anchor we optimistically set
            # and fall through to the direct-mutation path below.
            self._post_delete_anchor = None

        # Fallback: mutate the dialog's own state. Push an undo
        # snapshot onto the trace view first so Ctrl+Z still reverts.
        try:
            self.trace_view._cycle_push_undo()
        except Exception:
            pass

        removed = False
        for cycles in self.cycles_by_channel.values():
            for i, c in enumerate(cycles):
                if c is target:
                    cycles.pop(i)
                    removed = True
                    break
            if removed:
                break
        if not removed:
            # The flat list and cycles_by_channel had diverged. Do not
            # claim a deletion that didn't happen.
            return

        self._post_delete_anchor = deleted_row
        self._populate_table()
        self._push_flat_cycles_to_trace()
        self._refresh_export_buttons()
        self._schedule_autosave()
        self.status_label.setText(
            f"Deleted cycle on CH{target.channel}. "
            f"Ctrl+Z to undo."
        )

    # ------------------------------------------------------------------
    # Export / import
    # ------------------------------------------------------------------

    def _current_export_scheme(self) -> str:
        """Scheme to write for the current cycle set: the loaded
        scheme if these cycles came from a Load, otherwise the current
        combo value (which is what a fresh Detect would have used)."""
        if self._loaded_cycle_start_edge is not None:
            return self._loaded_cycle_start_edge
        return self.edge_combo.currentText()

    def _current_export_landmark_names(self) -> list[str]:
        if self._loaded_landmark_names is not None:
            return list(self._loaded_landmark_names)
        scheme = self._current_export_scheme()
        return list(LANDMARK_NAMES_BY_MODE.get(scheme, []))

    def _on_save_clicked(self):
        if not self.cycles_by_channel or not any(
            self.cycles_by_channel.values()
        ):
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
            export_cycles_to_csv(flat, self.engine, csv_path,)
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
                    landmark_names=self._current_export_landmark_names(),
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

        default_folder = analysis_folder_for(self.engine)
        start_dir = str(default_folder) if default_folder is not None else ""

        csv_path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Load Theta Cycles",
            start_dir,
            "Theta-cycle CSV (*.csv);;All files (*)",
        )
        if not csv_path_str:
            return

        csv_path = Path(csv_path_str)
        if not csv_path.exists():
            QMessageBox.warning(
                self, "File not found",
                f"{csv_path} does not exist."
            )
            return

        # ---- 1. Read the sidecar FIRST, so its scheme can be handed to
        # the CSV reader. The CSV no longer carries the scheme itself, so
        # without the sidecar the reader has no way to know what ref_1
        # through ref_6 mean.
        sidecar = None
        try:
            sidecar = load_sidecar(csv_path)
        except Exception as exc:
            QMessageBox.warning(
                self, "Sidecar unreadable",
                f"Could not read the sidecar for {csv_path.name}:\n\n{exc}\n\n"
                "The cycle positions will load, but their landmark scheme "
                "cannot be recovered without the sidecar."
            )
            sidecar = None

        sidecar_scheme: str | None = None
        sidecar_landmarks: list[str] | None = None
        if sidecar is not None:
            sidecar_scheme = (
                (sidecar.get("params") or {}).get("cycle_start_edge")
            )
            lm = sidecar.get("landmark_names")
            if isinstance(lm, list) and len(lm) == 6:
                sidecar_landmarks = [str(x) for x in lm]

        # ---- 2. Read the CSV, passing the sidecar's scheme so the
        # returned cycles are labeled correctly.
        try:
            cycles, csv_scheme, _ = read_cycles_from_csv(
                csv_path, cycle_start_edge=sidecar_scheme,
            )
        except Exception as exc:
            QMessageBox.critical(
                self, "Failed to load",
                f"Could not read {csv_path}:\n\n{exc}"
            )
            return

        if not cycles:
            QMessageBox.information(
                self, "No cycles found",
                f"{csv_path} contains no cycles."
            )
            return

        regrouped: dict[int, list[ThetaCycle]] = {}
        for c in cycles:
            regrouped.setdefault(c.channel, []).append(c)
        for ch in regrouped:
            regrouped[ch].sort(key=lambda c: c.zero_crossing_start)
        self.cycles_by_channel = regrouped

        # ---- 3. Determine the scheme actually in use. Prefer the sidecar
        # (authoritative, has landmark_names). If neither the sidecar nor
        # a legacy `zc_*` CSV column supplies one, warn the user that the
        # ref_N columns can't be interpreted.
        if sidecar_scheme is not None:
            loaded_scheme = sidecar_scheme
        elif csv_scheme is not None:
            # Legacy `zc_*`-column file: always rising_zc.
            loaded_scheme = csv_scheme
        else:
            QMessageBox.warning(
                self, "Missing sidecar",
                f"{csv_path.name} has no readable .meta.json sidecar, so the "
                f"meaning of the ref_1..ref_6 landmark columns cannot be "
                f"determined. The cycle boundaries will be loaded and drawn "
                f"as numeric positions, but the per-landmark labels assume "
                f"'rising_zc'. Re-save the cycles from the original detection "
                f"session if you need the correct scheme."
            )
            loaded_scheme = "rising_zc"

        # ---- 4. Cross-check sidecar vs CSV if both claim a scheme
        # (only possible for legacy CSVs, whose CSV-side scheme is always
        # rising_zc). Prefer sidecar.
        if sidecar_scheme is not None and csv_scheme is not None \
                and sidecar_scheme != csv_scheme:
            print(
                f"[theta-cycle-load] scheme mismatch: sidecar="
                f"{sidecar_scheme!r} csv={csv_scheme!r}; using sidecar."
            )

        self._loaded_cycle_start_edge = loaded_scheme
        self._loaded_landmark_names = sidecar_landmarks

        # ---- 5. Apply the sidecar (panel params + analysis window). Do
        # this AFTER the CSV has been read, so the sidecar's own
        # sample_offset bookkeeping doesn't race the CSV read.
        if sidecar is not None:
            self._apply_sidecar(sidecar)
            self._sample_offset = int(sidecar.get("analysis_start_sample", 0))
            start_n = int(sidecar.get("analysis_start_sample", 0))
            end_n = int(sidecar.get("analysis_end_sample", 0))
            self._detect_n_samples = max(0, end_n - start_n)
        else:
            self._sample_offset = 0
            self._detect_n_samples = 0

        # ---- 6. A fresh load replaces the entire cycle set: any pending
        # post-delete anchor refers to cycles that no longer exist.
        self._post_delete_anchor = None

        # ---- 7. Reflect the loaded scheme in the combo so a user editing
        # afterwards sees the correct controls.
        if loaded_scheme is not None:
            self.edge_combo.blockSignals(True)
            self.edge_combo.setCurrentText(loaded_scheme)
            self.edge_combo.blockSignals(False)

        self._populate_table()
        self._push_flat_cycles_to_trace()
        self.trace_view.set_theta_cycle_overlay_channels(
            sorted(self.cycles_by_channel.keys())
        )
        self._refresh_export_buttons()
        self._refresh_landmark_scheme_label()

        status = f"Loaded {len(cycles)} cycle(s) from {csv_path.parent}."
        if sidecar is not None:
            status += " Parameters restored."
        self.status_label.setText(status)
        self._autosave_armed = True

        
    def _apply_sidecar(self, sidecar: dict):
        """Restore every detection parameter and every panel widget
        that carries user-configured state from a loaded sidecar.

        Field fallbacks for old (v1) sidecars: any key missing from
        sidecar["params"] gets its dataclass default, which is exactly
        the state a v1-era panel had.
        """
        params_dict = sidecar.get("params") or {}
        try:
            params = ThetaCycleParams(**params_dict)
        except TypeError as exc:
            print(f"[theta-cycle-load] could not build params: {exc}")
            return

        self._last_params_for_sidecar = params

        def _set_spin(spin, value):
            spin.blockSignals(True)
            try:
                spin.setValue(float(value))
            finally:
                spin.blockSignals(False)

        # ---- Main numeric params ----
        _set_spin(self.low_spin, params.low_freq)
        _set_spin(self.high_spin, params.high_freq)
        _set_spin(self.min_amp_spin, params.min_theta_peak_percentile)
        _set_spin(self.delta_low_spin, params.delta_low_freq)
        _set_spin(self.delta_ratio_spin, params.theta_delta_ratio_min)
        _set_spin(self.fast_low_spin, params.fast_low_freq)
        _set_spin(self.fast_ratio_spin, params.fast_theta_ratio_max)

        # ---- Half-cycle bounds (default 2.5 / 15.0 when None) ----
        hc_lo = params.half_cycle_low_freq if params.half_cycle_low_freq is not None else 2.5
        hc_hi = params.half_cycle_high_freq if params.half_cycle_high_freq is not None else 15.0
        _set_spin(self.half_cycle_low_spin, hc_lo)
        _set_spin(self.half_cycle_high_spin, hc_hi)

        # ---- Advanced params ----
        self.filter_order_spin.blockSignals(True)
        try:
            self.filter_order_spin.setValue(int(params.filter_order))
        finally:
            self.filter_order_spin.blockSignals(False)

        self.detrend_check.blockSignals(True)
        try:
            self.detrend_check.setChecked(bool(params.detrend))
        finally:
            self.detrend_check.blockSignals(False)

        if params.delta_high_freq is not None:
            _set_spin(self.delta_high_spin, params.delta_high_freq)

        self.delta_check.blockSignals(True)
        try:
            self.delta_check.setChecked(bool(params.delta_correction))
        finally:
            self.delta_check.blockSignals(False)

        self.fast_check.blockSignals(True)
        try:
            self.fast_check.setChecked(bool(params.fast_correction))
        finally:
            self.fast_check.blockSignals(False)

        self.edge_combo.blockSignals(True)
        try:
            self.edge_combo.setCurrentText(str(params.cycle_start_edge))
        finally:
            self.edge_combo.blockSignals(False)

        # ---- Trace-view display controls ----
        # setValue with signals UNBLOCKED here so the trace view picks
        # up the restored gain/width via the existing valueChanged ->
        # set_theta_cycle_gain / set_theta_cycle_trace_width wiring.
        self.trace_gain_spin.setValue(float(params.trace_gain))
        self.trace_width_spin.setValue(float(params.trace_width))

        # ---- Channel text ----
        channels_text = (params.channels_text or "").strip()
        if not channels_text:
            sidecar_channels = sidecar.get("channels") or []
            channels_text = ",".join(str(int(c)) for c in sidecar_channels)
        self.channel_edit.blockSignals(True)
        try:
            self.channel_edit.setText(channels_text)
        finally:
            self.channel_edit.blockSignals(False)

        # ---- Analysis window (Start / End spinboxes) ----
        analysis_start = int(sidecar.get("analysis_start_sample", 0))
        analysis_end = int(sidecar.get("analysis_end_sample", 0))

        if analysis_end > analysis_start:
            try:
                start_time = float(self.engine.sample_to_time(analysis_start))
                end_time = float(self.engine.sample_to_time(analysis_end - 1))
            except Exception:
                start_time = end_time = None
            if start_time is not None and end_time is not None:
                self.start_spin.blockSignals(True)
                self.end_spin.blockSignals(True)
                try:
                    self.start_spin.setValue(start_time)
                    self.end_spin.setValue(end_time)
                finally:
                    self.start_spin.blockSignals(False)
                    self.end_spin.blockSignals(False)

        # ---- Propagate theta band to the trace view overlay ----
        if self.trace_view is not None:
            self.trace_view.set_theta_cycle_band(
                params.low_freq, params.high_freq,
            )
    # ------------------------------------------------------------------
    # Export-button state / autosave
    # ------------------------------------------------------------------

    def _refresh_export_buttons(self):
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
            # Intentional: never write an empty file, so an accidental
            # delete-all-then-close does not wipe a good on-disk
            # result. The user must click Save manually to persist an
            # empty set.
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
                    landmark_names=self._current_export_landmark_names(),
                )
            except Exception as exc:
                print(f"[theta-cycle-autosave] sidecar write failed: {exc}")

        self.status_label.setText(
            f"Auto-saved {len(flat)} cycle(s)."
        )

    # ------------------------------------------------------------------
    # Undo
    # ------------------------------------------------------------------

    def _on_undo_cycle_deletion(self):
        # Undo restores whatever the trace view's stack last snapshotted
        # (which is also what a trace-view-side Ctrl+Z would restore).
        # Clear the post-delete anchor so the next Prev/Next does not
        # try to land on a slot that refers to a set no longer present.
        if self.trace_view._cycle_undo():
            self._post_delete_anchor = None
            self.status_label.setText("Undid cycle deletion.")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if hasattr(self, "_delete_shortcut"):
            self._delete_shortcut.setEnabled(False)
        if hasattr(self, "_undo_shortcut"):
            self._undo_shortcut.setEnabled(False)

        if self._detect_thread is not None and self._detect_thread.isRunning():
            self._detect_thread.wait(2000)

        if self._autosave_armed and self._autosave_timer.isActive():
            self._autosave_timer.stop()
            self._autosave_now()

        try:
            self.trace_view.thetaCyclesChanged.disconnect(
                self.set_cycles_from_trace
            )
        except (TypeError, RuntimeError):
            pass
        try:
            self.trace_view.thetaCycleSelected.disconnect(
                self.set_selected_cycle
            )
        except (TypeError, RuntimeError):
            pass
        try:
            self.trace_view.thetaEpochsChanged.disconnect(
                self._on_theta_epochs_changed
            )
        except (TypeError, RuntimeError):
            pass

        self.trace_view.clear_theta_cycles()
        super().closeEvent(event)
        self.finished.emit(self.result())
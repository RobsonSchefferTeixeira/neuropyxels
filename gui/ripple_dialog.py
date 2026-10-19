"""
ripple_dialog.py

Non-modal QDialog for sharp-wave ripple (SWR) detection: channel
selection, detection parameters, CSD toggle, and a per-channel summary
table (sorted by mean peak amplitude, so the strongest-ripple
channel/depth is immediately visible).

Redesigned from the original two-panel (controls + own pyqtgraph
inspector plot) layout: this dialog is now controls-only. Results are
rendered as an interactive overlay directly on the main trace view
(RippleTraceViewWidget, typically composed into a NeuralTraceViewWidget
alongside theta epochs -- see gui/neural_trace_view.py) instead of a
second, independent plot -- for two reasons:
  1. Performance: the old inspector duplicated windowing/filtering/
     painting that the trace view already does, so every inspection
     paid for that pipeline twice.
  2. Correctness: the trace view owns the one time <-> pixel mapping
     that the vertical cursor line and the trace both agree on. A
     second, independent plot risks silently re-diverging from that
     mapping. Feeding results into the trace view's own paint pipeline
     means there's exactly one place where "where is time on screen"
     is decided.

Ripple events have the same architectural parity with theta epochs:
the trace view owns a flat list[RippleEvent] plus click-select /
drag-to-resize boundaries / Delete-key removal / merge-on-drag /
split-by-double-click / arm-then-drag add. This dialog stays the
params/table window and the source of truth for events_by_channel
(summary table, CSV export); rippleEventsChanged / rippleEventSelected
keep it in sync with edits made directly on the trace view.

Detection math lives in core/ripple_detector.py; CSD (optional) reuses
core/phase_amplitude.py's PhaseAmplitudeAnalyzer.compute_csd, the same
CSD implementation already used and tested there.

Selecting a row in the summary table (or Prev/Next Ripple) selects that
channel in the main trace view and pans/zooms it to center the ripple,
rather than redrawing a separate inspector.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QDoubleSpinBox,
    QSpinBox, QCheckBox, QPushButton, QComboBox, QMessageBox, QWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QLineEdit, QFileDialog, QSplitter,
)

from core.session_sidecar import save_sidecar, load_sidecar
from core.ripple_detector import RippleDetector, RippleParams, RippleEvent
from core.phase_amplitude import PhaseAmplitudeAnalyzer
from core.ripple_export import export_ripples_to_csv, export_ripples_per_channel, read_ripples_from_csv
from core.trace_engine import TraceEngine
from gui.ripple_trace_view import RippleTraceViewWidget
from gui.ripple_triggered_average_dialog import RippleTriggeredAverageDialog
from core.analysis_folder import (analysis_folder_for, ensure_analysis_folder,)

class _DetectThread(QThread):
    """Runs detection across all requested channels off the UI thread."""

    finished_ok = pyqtSignal(dict, dict)   # ({channel: list[RippleEvent]}, {channel: signal_used})
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, int)  # (channels_done, channels_total)

    def __init__(self, channels: list[int], raw_data, sample_rate: float,
                 start_idx: int, end_idx: int, params: RippleParams,
                 use_csd: bool, csd_spacing: float,
                 pac_analyzer: PhaseAmplitudeAnalyzer,
                 engine=None,
                 exclude_intervals: list[tuple[int, int]] | None = None,
                 parent=None):
        super().__init__(parent)
        self.channels = channels
        self.raw_data = raw_data
        self.sample_rate = sample_rate
        # Absolute sample indices, resolved by the dialog against the
        # engine's current time base (timestamps-aware). The thread
        # never has to guess which basis it's in.
        self.start_idx = int(start_idx)
        self.end_idx = int(end_idx)
        self.params = params
        self.use_csd = use_csd
        self.csd_spacing = csd_spacing
        self.pac_analyzer = pac_analyzer
        # Needed so the CSD branch can convert start_idx/end_idx back
        # to the time values compute_csd_with_options expects. May be
        # None when the caller doesn't use CSD; the CSD branch is only
        # taken when use_csd is True, in which case the dialog always
        # passes it.
        self.engine = engine
        self.detector = RippleDetector()
        self.exclude_intervals = exclude_intervals or []

    def run(self):
        try:
            start_idx = max(0, self.start_idx)
            end_idx = min(self.raw_data.shape[0], self.end_idx)
            if start_idx >= end_idx:
                raise ValueError("Invalid time range for detection.")

            n_samples = end_idx - start_idx

            exclude_mask = None
            if self.exclude_intervals:
                exclude_mask = np.zeros(n_samples, dtype=bool)
                for lo, hi in self.exclude_intervals:
                    rel_lo = max(0, lo - start_idx)
                    rel_hi = min(n_samples - 1, hi - start_idx)
                    if rel_lo <= rel_hi:
                        exclude_mask[rel_lo:rel_hi + 1] = True
                if not np.any(exclude_mask):
                    exclude_mask = None

            results: dict[int, list[RippleEvent]] = {}
            signals_used: dict[int, np.ndarray] = {}
            for i, ch in enumerate(self.channels):
                signal = self.raw_data[start_idx:end_idx, ch].flatten().astype(np.float64)

                if self.use_csd:
                    # Use the same distance-rule CSD implementation as
                    # the trace view's per-channel CSD, so the envelope,
                    # threshold lines, and event peak positions are all
                    # computed against the SAME signal the trace view
                    # draws under the channel's "Show CSD" overlay. The
                    # older compute_csd() used a different neighbor rule
                    # (no same-x constraint, no strictly-different-y
                    # constraint), which could pick different neighbors
                    # and produce a CSD that didn't match the trace
                    # view's CSD for the same channel.
                    csd_signal, csd_info = self.pac_analyzer.compute_csd_with_options(
                        signal, ch, self.sample_rate, self.raw_data,
                        float(self.engine.sample_to_time(start_idx)),
                        float(self.engine.sample_to_time(end_idx)),
                        distance_um=self.csd_spacing,
                        band_low=0.0,
                        band_high=0.0,
                    )
                    if csd_info.get("used", False):
                        signal = csd_signal
                    else:
                        # CSD unavailable for this channel -- fall back
                        # to raw signal rather than silently dropping.
                        # The channel still gets a detection pass, just
                        # not on CSD.
                        print(f"[RIPPLE-DETECT] CH{ch}: CSD unavailable "
                              f"({csd_info.get('reason', 'unknown')}); "
                              f"falling back to raw signal.")
                        signal = self.raw_data[start_idx:end_idx, ch].flatten().astype(np.float64)

                events = self.detector.detect(
                    signal, self.sample_rate, self.params, channel=ch,
                    exclude_mask=exclude_mask,
                )

                results[ch] = events
                signals_used[ch] = signal
                self.progress.emit(i + 1, len(self.channels))

            self.finished_ok.emit(results, signals_used)
        except Exception as exc:
            self.failed.emit(str(exc))


class RippleDialog(QDialog):
    """
    Non-modal, controls-only ripple detection dialog. Detection results
    are pushed to a RippleTraceViewWidget (or a NeuralTraceViewWidget
    composing it) as an interactive overlay -- click-select, drag
    boundaries, delete, merge-on-drag, split, add -- rather than
    rendered in this dialog, mirroring how ThetaEpochDialog drives its
    trace view.
    """
    eventsChanged = pyqtSignal()
    
    def __init__(self, probe_data: dict, engine: TraceEngine,
                 trace_view: RippleTraceViewWidget,
                 initial_channels: list[int] | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ripple Detection")
        self.setModal(False)
        self.resize(560, 620)

        self.engine = engine
        self.trace_view = trace_view
        self.probe_data = probe_data
        self.pac_analyzer = PhaseAmplitudeAnalyzer(probe_data)  # reused for CSD only
        self._detect_thread: _DetectThread | None = None
        self.detector = RippleDetector()

        # channel -> list[RippleEvent]; source of truth, also what gets
        # exported to CSV.
        self.events_by_channel: dict[int, list[RippleEvent]] = {}
        self._signal_by_channel: dict[int, np.ndarray] = {}
        self._envelope_by_channel: dict[int, np.ndarray] = {}
        self._filtered_by_channel: dict[int, np.ndarray] = {}
        self._sample_offset = 0
        self._selected_channel: int | None = None
        self._selected_index: int | None = None
        self._params: RippleParams | None = None
        self._open_rta_dialogs: list[RippleTriggeredAverageDialog] = []
        self._detect_n_samples = 0

        self._open_inspector_dialogs: list = []

        self._build_ui()
        self._sync_time_range_from_engine()

        # Keep this dialog's events_by_channel / summary table in sync
        # with edits made directly on the trace view, and keep the
        # trace view's selection in sync with table row selection.
        self.trace_view.rippleEventsChanged.connect(self.set_events_from_trace)
        self.trace_view.rippleEventSelected.connect(self.set_selected_event)

        if initial_channels:
            self.channel_edit.setText(",".join(str(c) for c in initial_channels))

        self._refresh_theta_exclusion_availability()
        self._refresh_save_load_enabled()

    def showEvent(self, event):
        self._refresh_theta_exclusion_availability()
        super().showEvent(event)
        self._refresh_save_load_enabled()

    def _refresh_theta_exclusion_availability(self):
        has_theta = bool(getattr(self.trace_view, "theta_epochs", None))
        self.exclude_theta_check.setEnabled(has_theta)
        if not has_theta:
            self.exclude_theta_check.setChecked(False)
            self.exclude_theta_check.setToolTip(
                "No theta epochs are currently defined on the main trace "
                "view. Run theta epoch detection first to enable this."
            )
        else:
            self.exclude_theta_check.setToolTip(
                "Exclude every theta epoch (on any channel) from both "
                "the amplitude threshold baseline AND candidate "
                "detection -- prevents high-frequency activity during "
                "theta from inflating the ripple threshold."
            )

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

        self.table_summary_label = QLabel("No ripples detected yet.")
        self.table_summary_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self.table_summary_label)

        layout.addWidget(self._build_summary_table(), stretch=1)
        layout.addLayout(self._build_overlay_controls())
        layout.addLayout(self._build_nav_row())
        layout.addLayout(self._build_edit_row())
        layout.addLayout(self._build_export_row())
        layout.addLayout(self._build_analysis_row())

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

        self.full_range_btn = QPushButton("Full available range")
        self.full_range_btn.setToolTip(
            "Set the analysis interval to the minimum and maximum available time."
        )
        self.full_range_btn.setAutoDefault(False)
        self.full_range_btn.setDefault(False)
        self.full_range_btn.clicked.connect(self._set_full_available_range)
        col_pairs.append((None, self.full_range_btn))

        self.low_freq_spin = QDoubleSpinBox()
        self.low_freq_spin.setRange(1.0, 2000.0)
        self.low_freq_spin.setValue(130.0)
        col_pairs.append(("Band low (Hz)", self.low_freq_spin))

        self.high_freq_spin = QDoubleSpinBox()
        self.high_freq_spin.setRange(2.0, 2000.0)
        self.high_freq_spin.setValue(230.0)
        col_pairs.append(("Band high (Hz)", self.high_freq_spin))

        self.envelope_combo = QComboBox()
        self.envelope_combo.addItems(["hilbert", "rms"])
        col_pairs.append(("Envelope", self.envelope_combo))

        self.peak_sd_spin = QDoubleSpinBox()
        self.peak_sd_spin.setRange(0.1, 50.0)
        self.peak_sd_spin.setValue(4.0)
        col_pairs.append(("Peak threshold (SD)", self.peak_sd_spin))

        self.boundary_sd_spin = QDoubleSpinBox()
        self.boundary_sd_spin.setRange(0.1, 50.0)
        self.boundary_sd_spin.setValue(1.0)
        col_pairs.append(("Boundary threshold (SD)", self.boundary_sd_spin))

        self.min_dur_spin = QDoubleSpinBox()
        self.min_dur_spin.setRange(0.0, 5000.0)
        self.min_dur_spin.setValue(15.0)
        self.min_dur_spin.setSpecialValueText("off")
        col_pairs.append(("Min duration (ms)", self.min_dur_spin))

        self.max_dur_spin = QDoubleSpinBox()
        self.max_dur_spin.setRange(0.0, 5000.0)
        self.max_dur_spin.setValue(500.0)
        self.max_dur_spin.setSpecialValueText("off")
        col_pairs.append(("Max duration (ms)", self.max_dur_spin))

        self.csd_check = QCheckBox("Use CSD")
        col_pairs.append((None, self.csd_check))

        self.csd_spacing_spin = QDoubleSpinBox()
        self.csd_spacing_spin.setRange(1.0, 5000.0)
        self.csd_spacing_spin.setValue(30.0)
        col_pairs.append(("CSD spacing (\u00b5m)", self.csd_spacing_spin))

        self.exclude_theta_check = QCheckBox("Detect ripples outside theta only")
        self.exclude_theta_check.setToolTip(
            "Exclude every theta epoch (on any channel) from both the "
            "amplitude threshold baseline AND candidate detection."
        )
        self.exclude_theta_check.setEnabled(False)
        col_pairs.append((None, self.exclude_theta_check))

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
        """Per-event table, matching the theta epoch / cycle dialogs.
        Columns carry both sample and time for each landmark, so the
        table reads consistently with the other two event types. A
        compact summary line is shown in the header label above."""
        self.summary_table = QTableWidget(0, 13)
        self.summary_table.setHorizontalHeaderLabels([
            "Channel",
            "start_smp", "end_smp", "peak_smp", "trough_smp",
            "duration_smp",
            "start_s", "end_s", "peak_s", "trough_s",
            "duration_s",
            "peak_amp", "trough_amp",
        ])
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

        self.show_filtered_check = QCheckBox("Filtered signal")
        self.show_filtered_check.setChecked(False)
        self.show_filtered_check.setToolTip(
            "Draw the ripple-band bandpassed signal itself (the actual "
            "oscillation), not just its envelope. When 'Use CSD' is "
            "checked above, this is the bandpassed CSD; otherwise it's "
            "the bandpassed raw LFP."
        )
        self.show_filtered_check.toggled.connect(self._on_overlay_visibility_changed)
        row.addWidget(self.show_filtered_check)

        row.addStretch(1)
        return row

    def _build_nav_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.prev_ripple_btn = QPushButton("\u25c4 Prev Ripple")
        self.prev_ripple_btn.setAutoDefault(False)
        self.prev_ripple_btn.setDefault(False)
        self.prev_ripple_btn.clicked.connect(self._go_to_prev_ripple)
        row.addWidget(self.prev_ripple_btn)

        self.next_ripple_btn = QPushButton("Next Ripple \u25ba")
        self.next_ripple_btn.setAutoDefault(False)
        self.next_ripple_btn.setDefault(False)
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

    def _build_edit_row(self) -> QHBoxLayout:
        """Buttons that mirror the theta-epoch editor's manual-edit
        controls: arm-then-drag creation, and delete-selected. Split
        and resize are done directly on the trace view (double-click
        inside a selected ripple to split; drag its boundaries to
        resize), same as theta."""
        row = QHBoxLayout()

        self.add_ripple_btn = QPushButton("Add Ripple")
        self.add_ripple_btn.setCheckable(True)
        self.add_ripple_btn.setAutoDefault(False)
        self.add_ripple_btn.setDefault(False)
        self.add_ripple_btn.setToolTip(
            "Click, then click-drag on a channel's lane in the main "
            "trace view to draw a new manual ripple there. Click again "
            "to cancel without creating one."
        )
        self.add_ripple_btn.toggled.connect(self._on_add_ripple_toggled)
        row.addWidget(self.add_ripple_btn)

        self.delete_ripple_btn = QPushButton("Delete Selected Ripple")
        self.delete_ripple_btn.setAutoDefault(False)
        self.delete_ripple_btn.setDefault(False)
        self.delete_ripple_btn.setToolTip(
            "Delete the currently-selected ripple event (also "
            "available via the Delete key on the trace view)."
        )
        self.delete_ripple_btn.clicked.connect(self._on_delete_selected_ripple)
        row.addWidget(self.delete_ripple_btn)

        row.addStretch(1)
        return row
    
    def _refresh_save_load_enabled(self):
        """Enable Save/Load only when there is a data file to anchor
        the analysis folder to. Without one, both operations have
        nowhere to write or read from, so they're greyed out with a
        tooltip explaining why."""
        has_file = getattr(self.engine, "data_file", None) is not None
        self.export_all_btn.setEnabled(has_file)
        self.load_btn.setEnabled(has_file)
        if has_file:
            folder = analysis_folder_for(self.engine)
            self.export_all_btn.setToolTip(
                f"Write all detected ripples and their parameters to:\n"
                f"{folder}"
            )
            self.load_btn.setToolTip(
                f"Load ripples and their parameters from:\n"
                f"{folder}"
            )
        else:
            self.export_all_btn.setToolTip(
                "No data file is loaded. Load a continuous.dat before "
                "saving or loading an analysis."
            )
            self.load_btn.setToolTip(
                "No data file is loaded. Load a continuous.dat before "
                "saving or loading an analysis."
            )


    def _build_export_row(self) -> QHBoxLayout:
        row = QHBoxLayout()

        self.export_all_btn = QPushButton("Save")
        self.export_all_btn.setAutoDefault(False)
        self.export_all_btn.setDefault(False)
        self.export_all_btn.setToolTip(
            "Write one CSV per channel into a chosen folder, plus a "
            "small .meta.json sidecar next to each CSV recording the "
            "parameters and analysis window. Loading a CSV later "
            "restores those so Ripple-Triggered Average and Single "
            "Ripple View work without re-running detection."
        )
        self.export_all_btn.clicked.connect(self._on_export_all_clicked)
        row.addWidget(self.export_all_btn)


        self.load_btn = QPushButton("Load")
        self.load_btn.setAutoDefault(False)
        self.load_btn.setDefault(False)
        self.load_btn.setToolTip(
            "Load a previously-exported ripple CSV. Loaded events are "
            "added to the current set and behave exactly like freshly "
            "detected ones."
        )
        self.load_btn.clicked.connect(self._on_load_clicked)
        row.addWidget(self.load_btn)

        row.addStretch(1)
        return row

    def _build_analysis_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(QLabel("Further analysis:"))

        self.rta_btn = QPushButton("Ripple-Triggered Average...")
        self.rta_btn.setAutoDefault(False)
        self.rta_btn.setDefault(False)
        self.rta_btn.setToolTip(
            "Average the raw LFP around every detected ripple event, for "
            "a chosen set of channels, time window, and alignment "
            "reference."
        )
        self.rta_btn.clicked.connect(self._on_open_rta_clicked)
        row.addWidget(self.rta_btn)

        self.inspector_btn = QPushButton("Single Ripple View...")
        self.inspector_btn.setAutoDefault(False)
        self.inspector_btn.setDefault(False)
        self.inspector_btn.setToolTip(
            "Inspect one ripple at a time across multiple channels, "
            "with Prev / Next navigation and a Reject button. Uses the "
            "same probe-geometry rendering as the Ripple-Triggered "
            "Average, but without averaging."
        )
        self.inspector_btn.clicked.connect(self._on_open_inspector_clicked)
        row.addWidget(self.inspector_btn)

        row.addStretch(1)
        return row

    def _on_open_rta_clicked(self):
        if not self.events_by_channel or not any(self.events_by_channel.values()):
            QMessageBox.information(
                self, "No ripple events",
                "Run detection (or load a ripple CSV) first -- there are "
                "no ripple events to average yet."
            )
            return
        if self._params is None:
            QMessageBox.information(
                self, "No detection parameters",
                "Run detection at least once first -- the ripple-band "
                "filter settings are needed to compute alignment points "
                "for the 'max/min cycle' options."
            )
            return

        initial_channels = list(self.trace_view.channels) if self.trace_view.channels else []
        if not initial_channels:
            initial_channels = sorted(self.events_by_channel.keys())

        sample_offset_by_channel = {ch: self._sample_offset for ch in self.events_by_channel.keys()}

        dialog = RippleTriggeredAverageDialog(
            self.engine, self.probe_data, self.events_by_channel, sample_offset_by_channel,
            self._params, initial_channels, parent=self,
        )
        self._open_rta_dialogs.append(dialog)
        dialog.finished.connect(lambda _res, d=dialog: self._on_rta_dialog_closed(d))
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _on_rta_dialog_closed(self, dialog: RippleTriggeredAverageDialog):
        if dialog in self._open_rta_dialogs:
            self._open_rta_dialogs.remove(dialog)

    def _on_open_inspector_clicked(self):
        if not self.events_by_channel or not any(self.events_by_channel.values()):
            QMessageBox.information(
                self, "No ripple events",
                "Run detection (or load a ripple CSV) first -- there are "
                "no ripple events to inspect yet."
            )
            return
        if self._params is None:
            QMessageBox.information(
                self, "No detection parameters",
                "Run detection at least once first -- the ripple-band "
                "filter settings are needed to compute alignment points."
            )
            return

        from gui.ripple_inspector_dialog import RippleInspectorDialog

        # Pass the visible trace-view channels as the initial display
        # set; the inspector itself decides which channels to render,
        # and uses its own trigger-channel dropdown to pick which
        # ripple is being inspected.
        initial_channels = (
            list(self.trace_view.channels)
            if self.trace_view.channels
            else sorted(ch for ch, evs in self.events_by_channel.items() if evs)
        )

        dialog = RippleInspectorDialog(
            self.engine,
            self.probe_data,
            self.events_by_channel,
            self._sample_offset,
            self._params,
            initial_channels,
            parent=self,
        )
        self._open_inspector_dialogs.append(dialog)

        # Wire the two-way sync between the inspector and this dialog:
        #  - inspector.rejectRequested -> remove the event here
        #  - inspector.rippleSelected -> highlight the table row
        #  - this.eventsChanged -> refresh the inspector's caches
        dialog.rejectRequested.connect(self._on_inspector_reject)
        dialog.rippleSelected.connect(self._on_inspector_selected)
        self.eventsChanged.connect(dialog.on_events_changed)

        dialog.finished.connect(
            lambda _res, d=dialog: self._on_inspector_closed(d)
        )
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _on_inspector_closed(self, dialog):
        if dialog in self._open_inspector_dialogs:
            self._open_inspector_dialogs.remove(dialog)

    def _on_inspector_reject(self, channel: int, start_sample: int, end_sample: int):
        """Remove the identified ripple from events_by_channel, refresh
        the table, push to the trace view overlay, and tell subscribers
        (the inspector) that events changed.

        Undo: the removal is pushed onto the trace view's ripple undo
        stack BEFORE the mutation, so Ctrl+Z on the trace view
        restores it, and rippleEventsChanged flows back through
        set_events_from_trace, which re-notifies the inspector.
        """
        events = self.events_by_channel.get(int(channel), [])
        match_idx = None
        for i, ev in enumerate(events):
            if (int(ev.start_sample) == int(start_sample)
                    and int(ev.end_sample) == int(end_sample)):
                match_idx = i
                break
        if match_idx is None:
            return

        # Snapshot for undo, before mutation.
        try:
            self.trace_view._ripple_push_undo_snapshot()
        except Exception:
            pass

        events.pop(match_idx)
        self._populate_summary_table()
        self._push_overlay_to_trace_view()
        self.eventsChanged.emit()
        self.status_label.setText(
            f"Rejected ripple on CH{channel}. "
            f"Ctrl+Z on the trace view to undo."
        )

    def _on_inspector_selected(self, channel: int, start_sample: int):
        """Highlight the row in this dialog's table corresponding to
        the ripple the inspector is currently showing."""
        flat = self._flat_events()
        for i, ev in enumerate(flat):
            if int(ev.channel) == int(channel) and int(ev.start_sample) == int(start_sample):
                self.summary_table.blockSignals(True)
                self.summary_table.selectRow(i)
                self.summary_table.blockSignals(False)
                break

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
        self._populate_summary_table()

    def _set_full_available_range(self):
        if not self.engine.data_loaded:
            return
        t_min, t_max = self.engine.get_time_range()
        self.start_spin.setValue(t_min)
        self.end_spin.setValue(t_max)

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

    def _theta_global_exclusion_intervals(self) -> list[tuple[int, int]]:
        theta_epochs = getattr(self.trace_view, "theta_epochs", None)
        if not theta_epochs:
            return []

        theta_sample_offset = int(getattr(self.trace_view, "theta_detection_sample_offset", 0))

        intervals = sorted(
            (theta_sample_offset + int(e.start_sample), theta_sample_offset + int(e.end_sample))
            for e in theta_epochs
        )

        merged: list[tuple[int, int]] = []
        for lo, hi in intervals:
            if merged and lo <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))

        return merged

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

        exclude_intervals = []
        if self.exclude_theta_check.isChecked():
            exclude_intervals = self._theta_global_exclusion_intervals()

        self.detect_btn.setEnabled(False)
        self.detect_btn.setText("Detecting...")
        self.status_label.setText(
            "Running detection (excluding theta epochs), please wait..."
            if exclude_intervals else
            "Running detection, please wait..."
        )

        # Resolve the spinbox values to absolute sample indices using
        # the engine's current time base (timestamp-aware), then pass
        # those to the thread. This makes detection consistent whether
        # or not a timestamps file is loaded.
        start_idx = self.engine.timestamp_to_sample(self.start_spin.value())
        end_idx = self.engine.timestamp_to_sample(self.end_spin.value())
        self._detect_n_samples = int(end_idx) - int(start_idx)
        self._sample_offset = int(start_idx)
        self._detect_n_samples = int(end_idx - start_idx)

        self._detect_thread = _DetectThread(
            channels, self.engine.data, self.engine.sr,
            start_idx, end_idx, params,
            self.csd_check.isChecked(), self.csd_spacing_spin.value(),
            self.pac_analyzer,
            engine=self.engine,
            exclude_intervals=exclude_intervals,
            parent=self,
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

        # Compute the ripple-band filtered signal and its envelope for
        # every detected channel once. Both are used for the overlay
        # (envelope curve, threshold lines, filtered-signal curve), and
        # they persist across edits since only the events change, not
        # the underlying signal processing context.
        self._envelope_by_channel = {}
        self._filtered_by_channel = {}
        for ch, signal in signals_used.items():
            filtered = self.detector.apply_ripple_filter(signal, self.engine.sr, self._params)
            self._filtered_by_channel[ch] = filtered
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
    # Overlay: pushing results to the main trace view
    # ------------------------------------------------------------------

    def _push_overlay_to_trace_view(self):
        if not self.events_by_channel:
            self.trace_view.clear_ripple_overlay()
            return

        all_events: list[RippleEvent] = []
        render_context = {}
        for ch, events in self.events_by_channel.items():
            all_events.extend(events)

            envelope = self._envelope_by_channel.get(ch)
            signal = self._signal_by_channel.get(ch)
            filtered = self._filtered_by_channel.get(ch)
            if envelope is None or signal is None:
                continue
            render_context[ch] = {
                'envelope': envelope,
                'filtered': filtered,
                'signal': signal,
                'sample_offset': self._sample_offset,
                'sample_rate': self.engine.sr,
                'peak_threshold_sd': self._params.peak_threshold_sd if self._params else None,
                'boundary_threshold_sd': self._params.boundary_threshold_sd if self._params else None,
            }

        all_events.sort(key=lambda e: (e.channel, e.start_sample))

        self.trace_view.set_ripple_overlay_visibility(
            show_envelope=self.show_envelope_check.isChecked(),
            show_thresholds=self.show_thresholds_check.isChecked(),
            show_events=self.show_events_check.isChecked(),
            show_filtered=self.show_filtered_check.isChecked(),
        )
        self.trace_view.set_ripple_render_context(render_context)
        self.trace_view.set_ripple_events(all_events)

    def _on_overlay_visibility_changed(self, _checked: bool):
        self.trace_view.set_ripple_overlay_visibility(
            show_envelope=self.show_envelope_check.isChecked(),
            show_thresholds=self.show_thresholds_check.isChecked(),
            show_events=self.show_events_check.isChecked(),
            show_filtered=self.show_filtered_check.isChecked(),
        )

    def set_events_from_trace(self, events: list[RippleEvent]):
        """Slot for trace_view.rippleEventsChanged -- keeps
        events_by_channel (and therefore the summary table / CSV export)
        in sync after an in-place edit made directly on the trace view
        (drag boundary, delete, merge, split, add)."""
        regrouped: dict[int, list[RippleEvent]] = {}
        for ev in events:
            regrouped.setdefault(ev.channel, []).append(ev)
        for ch in self.events_by_channel.keys():
            regrouped.setdefault(ch, [])
        self.events_by_channel = regrouped
        self._populate_summary_table()
        self.eventsChanged.emit()
        # A successful edit (or add) coming back from the trace view
        # means creation mode -- if it was armed -- has already
        # disarmed itself there; keep the button's checked state in
        # sync rather than leaving it stuck checked.
        if self.add_ripple_btn.isChecked():
            self.add_ripple_btn.blockSignals(True)
            self.add_ripple_btn.setChecked(False)
            self.add_ripple_btn.blockSignals(False)

    # ------------------------------------------------------------------
    # Edit-mode handlers
    # ------------------------------------------------------------------

    def _on_add_ripple_toggled(self, checked: bool):
        if checked:
            self.trace_view.arm_ripple_creation()
            self.status_label.setText(
                "Click-drag on a channel's lane in the main trace view "
                "to draw a new ripple. Click 'Add Ripple' again to "
                "cancel without creating one."
            )
        else:
            self.trace_view.cancel_ripple_creation()
            self.status_label.setText(f"{len(self.events_by_channel)} channel(s) with ripple data.")

    def _on_delete_selected_ripple(self):
        deleted = self.trace_view.delete_selected_ripple_event()
        if not deleted:
            QMessageBox.information(
                self, "No ripple selected",
                "Click a ripple in the main trace view first, then "
                "press this button (or the Delete key)."
            )

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    def _flat_events(self) -> list[RippleEvent]:
        flat: list[RippleEvent] = []
        for ch in sorted(self.events_by_channel.keys()):
            flat.extend(self.events_by_channel[ch])
        flat.sort(key=lambda e: (e.channel, e.start_sample))
        return flat
    
    def _populate_summary_table(self):
        """Populate the per-event table with every ripple across every
        detected channel, sorted by (channel, start_sample). Time
        columns are recomputed from the engine on each repopulation, so
        a timestamps file loaded after detection is reflected without
        reopening the dialog."""
        flat = self._flat_events()
        # Null the anchor BEFORE touching the widget so the cache can
        # never disagree with the visible selection, even transiently.
        self._selected_index = None
        self.summary_table.blockSignals(True)
        try:
            self.summary_table.setRowCount(len(flat))

            def t(smp):
                # The in-memory event samples are slice-relative, so add
                # the detection-time offset before asking the engine for
                # the absolute time.
                return float(self.engine.sample_to_time(
                    int(smp) + int(self._sample_offset)
                ))

            for row, ev in enumerate(flat):
                start_smp = int(ev.start_sample) + int(self._sample_offset)
                end_smp = int(ev.end_sample) + int(self._sample_offset)
                peak_smp = int(ev.peak_sample) + int(self._sample_offset)
                trough_smp = int(ev.trough_sample) + int(self._sample_offset)
                dur_smp = end_smp - start_smp

                values = [
                    str(ev.channel),
                    str(start_smp),
                    str(end_smp),
                    str(peak_smp),
                    str(trough_smp),
                    str(dur_smp),
                    f"{t(ev.start_sample):.3f}",
                    f"{t(ev.end_sample):.3f}",
                    f"{t(ev.peak_sample):.3f}",
                    f"{t(ev.trough_sample):.3f}",
                    f"{dur_smp / float(self.engine.sr):.3f}",
                    f"{ev.peak_amplitude:.3g}",
                    f"{ev.trough_amplitude:.3g}",
                ]
                for col, v in enumerate(values):
                    item = QTableWidgetItem(v)
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignCenter
                        | Qt.AlignmentFlag.AlignVCenter
                    )
                    self.summary_table.setItem(row, col, item)
            self.summary_table.clearSelection()
            
        finally:
            self.summary_table.blockSignals(False)

        # Compact summary above the table.
        n_events = len(flat)
        n_channels = sum(1 for ch, evs in self.events_by_channel.items() if evs)
        if n_events == 0:
            self.table_summary_label.setText("No ripples detected yet.")
        else:
            self.table_summary_label.setText(
                f"{n_events} ripple(s) across {n_channels} channel(s)."
            )

    def _on_table_selection_changed(self):
        rows = self.summary_table.selectionModel().selectedRows()
        if not rows:
            self._selected_channel = None
            self._selected_index = None
            return
        row = rows[0].row()
        flat = self._flat_events()
        if not (0 <= row < len(flat)):
            return
        target = flat[row]
        self._selected_channel = int(target.channel)
        self._selected_index = int(row)

        self.trace_view.set_ripple_selected_event(row)
        self._show_channel_in_trace_view(self._selected_channel)
        self.status_label.setText(
            f"CH{self._selected_channel} ripple at "
            f"{self.engine.sample_to_time(int(target.start_sample) + int(self._sample_offset)):.3f} s"
        )

    def set_selected_event(self, index: int):
        """Slot for trace_view.rippleEventSelected -- index is into the
        trace view's flat, (channel, start_sample)-sorted event list,
        which matches the row order of our own per-event table."""
        if not (0 <= int(index) < self.summary_table.rowCount()):
            return
        self.summary_table.blockSignals(True)
        self.summary_table.selectRow(int(index))
        self.summary_table.blockSignals(False)

    # ------------------------------------------------------------------
    # Driving the main trace view
    # ------------------------------------------------------------------

    def _show_channel_in_trace_view(self, channel: int):
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
        """Step Prev/Next through the flat, (channel, start_sample)-
        sorted list of every ripple across every detected channel --
        the same order the table shows. Crossing from the last ripple
        of one channel into the first of the next is intentional: the
        button means 'next row in the table', not 'next ripple on this
        channel'.

        The anchor is the table's own selection, not a cached index.
        Whatever row is visibly highlighted is where the user is; if
        nothing is selected, direction > 0 starts at row 0 and
        direction < 0 starts at the last row. When the end (or start)
        is reached, the selection stays put -- no wrap.
        """
        flat = self._flat_events()
        if not flat:
            return

        rows = self.summary_table.selectionModel().selectedRows()
        if rows:
            current_row = int(rows[0].row())
        else:
            current_row = -1 if direction > 0 else len(flat)

        new_row = current_row + direction
        new_row = max(0, min(new_row, len(flat) - 1))

        target = flat[new_row]
        target_time = float(self.engine.sample_to_time(
            int(target.peak_sample) + int(self._sample_offset)
        ))

        # Pan the trace view to the target's peak time.
        window_duration = max(0.05, self.nav_window_spin.value())
        t_min, _ = self.engine.get_time_range()
        new_start = max(t_min, target_time - window_duration / 2)

        self.trace_view.window_duration = window_duration
        self.trace_view.start_time = new_start
        self.trace_view.control_panel.set_duration(window_duration)
        self.trace_view._update_time_labels()
        self.trace_view._update_scrollbar_position()
        self.trace_view._invalidate_cache()
        self.trace_view.update()

        # Ensure the target's channel is drawn (same effect as clicking
        # the corresponding table row).
        self._show_channel_in_trace_view(int(target.channel))
        self._selected_channel = int(target.channel)

        # Programmatic selection: block the table's signals so that
        # itemSelectionChanged doesn't fire mid-way through and read a
        # partially-updated selection. After this returns, the table
        # reflects the new state; _selected_index is set explicitly so
        # it can't disagree with what's visible.
        self.summary_table.blockSignals(True)
        self.summary_table.selectRow(new_row)
        self.summary_table.blockSignals(False)
        self._selected_index = new_row

        self.trace_view.set_ripple_selected_event(new_row)

        self.status_label.setText(
            f"Ripple {new_row + 1} of {len(flat)} on CH{target.channel} "
            f"({'(next)' if direction > 0 else '(prev)'})"
        )
    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def _on_export_all_clicked(self):
        if not self.events_by_channel or not any(self.events_by_channel.values()):
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

        csv_path = folder / "ripples.csv"
        meta_path = folder / "ripples.meta.json"

        flat = self._flat_events()
        try:
            export_ripples_to_csv(
                flat, self._sample_offset, self.engine, csv_path,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return

        if self._params is not None:
            n = int(getattr(self, "_detect_n_samples", 0))
            analysis_end_sample = int(self._sample_offset) + n
            if n > 0 and analysis_end_sample > int(self._sample_offset):
                try:
                    save_sidecar(
                        csv_path=csv_path,
                        kind="ripples",
                        sample_rate=float(self.engine.sr),
                        analysis_start_sample=int(self._sample_offset),
                        analysis_end_sample=analysis_end_sample,
                        channels=sorted(self.events_by_channel.keys()),
                        params=self._params,
                    )
                except Exception as exc:
                    print(f"[ripple-save] sidecar failed: {exc}")

        self.status_label.setText(
            f"Saved {len(flat)} event(s) to {folder}"
            + (" (with parameters)" if self._params is not None else "")
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

        csv_path = folder / "ripples.csv"
        if not csv_path.exists():
            QMessageBox.information(
                self, "Nothing to load",
                f"No saved ripples were found in:\n\n{folder}\n\n"
                "Detect ripples and click Save first."
            )
            return

        # Try to read the sidecar; if it exists, its sample offset is
        # authoritative for interpreting the CSV's absolute sample
        # indices. If not, fall back to the dialog's current offset.
        sidecar = None
        try:
            sidecar = load_sidecar(csv_path)
        except Exception as exc:
            QMessageBox.warning(
                self, "Sidecar unreadable",
                f"Could not read the sidecar for {csv_path.name}:\n\n{exc}\n\n"
                "The events will load, but saved parameters and signals "
                "will not be available."
            )
            sidecar = None

        sample_offset_to_use = self._sample_offset
        if sidecar is not None:
            sidecar_sr = float(sidecar.get("sample_rate", 0.0))
            if sidecar_sr > 0 and abs(sidecar_sr - float(self.engine.sr)) > 1e-6:
                proceed = QMessageBox.question(
                    self, "Sample rate mismatch",
                    f"Saved analysis was recorded at {sidecar_sr:.1f} Hz but "
                    f"the current engine reports {self.engine.sr:.1f} Hz.\n\n"
                    "Times derived from samples will be wrong. Load anyway?"
                )
                if proceed != QMessageBox.StandardButton.Yes:
                    return
            sample_offset_to_use = int(sidecar.get("analysis_start_sample", 0))

        try:
            events = read_ripples_from_csv(csv_path, sample_offset=sample_offset_to_use)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to load", f"Could not read {csv_path}:\n\n{exc}")
            return

        if not events:
            QMessageBox.information(self, "No events found", f"{csv_path} contains no ripple events.")
            return

        # Replace the in-memory event set with what was loaded. Loading
        # the same file twice produces the same state, not a doubled
        # list. This also discards any in-memory edits that weren't
        # saved -- which is the expected behavior for "Load" (as
        # opposed to "Import" or "Merge").
        regrouped: dict[int, list[RippleEvent]] = {}
        for ev in events:
            regrouped.setdefault(ev.channel, []).append(ev)
        for ch in regrouped:
            regrouped[ch].sort(key=lambda e: e.start_sample)
        self.events_by_channel = regrouped

        if sidecar is not None:
            self._apply_sidecar(sidecar, events)

        self._populate_summary_table()
        self._push_overlay_to_trace_view()

        status = f"Loaded {len(events)} ripple event(s) from {folder}."
        if sidecar is not None:
            status += " Parameters and signals restored."
        else:
            status += " No sidecar found; RTA and Single Ripple View will need fresh detection."
        self.status_label.setText(status)



    def _apply_sidecar(self, sidecar: dict, events: list[RippleEvent]):
        """Restore detection params, sample offset, and per-channel
        envelope / filtered signals from a loaded sidecar.

        The envelope and filtered signal are recomputed from
        engine.data over the recorded analysis window, matching the
        algorithm the detector used. Both are bit-identical to what
        the original detection produced because the filter is
        deterministic and the input slice is the same.
        """
        params_dict = sidecar.get("params") or {}
        try:
            self._params = RippleParams(**params_dict)
        except TypeError as exc:
            print(f"[ripple-load] could not construct RippleParams "
                  f"from sidecar: {exc}")
            return

        # Also restore the spinbox values so the UI shows the loaded
        # session's settings -- otherwise a subsequent detection would
        # silently use different parameters than what produced the CSV.
        self.low_freq_spin.setValue(float(self._params.low_freq))
        self.high_freq_spin.setValue(float(self._params.high_freq))
        self.envelope_combo.setCurrentText(str(self._params.envelope_method))
        self.peak_sd_spin.setValue(float(self._params.peak_threshold_sd))
        self.boundary_sd_spin.setValue(float(self._params.boundary_threshold_sd))
        self.min_dur_spin.setValue(
            float(self._params.min_duration_ms) if self._params.min_duration_ms else 0.0
        )
        self.max_dur_spin.setValue(
            float(self._params.max_duration_ms) if self._params.max_duration_ms else 0.0
        )

        self._sample_offset = int(sidecar.get("analysis_start_sample", 0))

        analysis_start = int(sidecar.get("analysis_start_sample", 0))
        analysis_end = int(sidecar.get("analysis_end_sample", 0))
        if analysis_end <= analysis_start:
            # Nothing to recompute; the events are still usable but
            # the overlay will be missing.
            return

        # Recompute the per-channel signals over the exact same window
        # the detector used. For each channel that appears in the
        # sidecar's channel list, slice engine.data and re-filter.
        for ch in sidecar.get("channels", []):
            ch = int(ch)
            try:
                signal = self.engine.data[analysis_start:analysis_end, ch] \
                    .flatten().astype(np.float64)
            except Exception as exc:
                print(f"[ripple-load] could not slice CH{ch}: {exc}")
                continue
            try:
                filtered = self.detector.apply_ripple_filter(
                    signal, self.engine.sr, self._params
                )
                envelope = self.detector.compute_envelope(
                    filtered, self.engine.sr, self._params
                )
            except Exception as exc:
                print(f"[ripple-load] could not filter CH{ch}: {exc}")
                continue
            self._signal_by_channel[ch] = signal
            self._filtered_by_channel[ch] = filtered
            self._envelope_by_channel[ch] = envelope
            
    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if self._detect_thread is not None and self._detect_thread.isRunning():
            self._detect_thread.wait(2000)
        try:
            self.trace_view.rippleEventsChanged.disconnect(self.set_events_from_trace)
        except (TypeError, RuntimeError):
            pass
        try:
            self.trace_view.rippleEventSelected.disconnect(self.set_selected_event)
        except (TypeError, RuntimeError):
            pass
        if self.add_ripple_btn.isChecked():
            self.trace_view.cancel_ripple_creation()
        self.trace_view.clear_ripple_overlay()
        for dialog in list(self._open_rta_dialogs):
            dialog.close()
        for dialog in list(self._open_inspector_dialogs):
            try:
                dialog.close()
            except Exception:
                pass
        super().closeEvent(event)
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
     that the vertical cursor line and the trace both agree on (after
     a couple of rounds fixing drift bugs there). A second, independent
     plot risks silently re-diverging from that mapping. Feeding
     results into the trace view's own paint pipeline means there's
     exactly one place where "where is time on screen" is decided.

As of this refactor, ripple events have the same architectural parity
with theta epochs: the trace view owns a flat list[RippleEvent] plus
click-select / drag-to-resize boundaries / Delete-key removal /
merge-on-drag, mirroring ThetaEpochTraceViewWidget exactly. This dialog
stays the params/table window and the source of truth for
events_by_channel (summary table, CSV export); rippleEventsChanged /
rippleEventSelected keep it in sync with edits made directly on the
trace view.

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
from gui.ripple_trace_view import RippleTraceViewWidget


class _DetectThread(QThread):
    """Runs detection across all requested channels off the UI thread."""

    finished_ok = pyqtSignal(dict, dict)   # ({channel: list[RippleEvent]}, {channel: signal_used})
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, int)  # (channels_done, channels_total)

    def __init__(self, channels: list[int], raw_data, sample_rate: float,
                 start_time: float, end_time: float, params: RippleParams,
                 use_csd: bool, csd_spacing: float,
                 pac_analyzer: PhaseAmplitudeAnalyzer,
                 exclude_intervals: list[tuple[int, int]] | None = None,
                 parent=None):
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
        # [(global_start_sample, global_end_sample), ...] -- applied
        # identically to every channel being detected (theta is treated
        # as network-wide, not per-channel; see
        # RippleDialog._theta_global_exclusion_intervals). Global-sample
        # terms, NOT relative to this thread's own start_idx.
        self.exclude_intervals = exclude_intervals or []

    def run(self):
        try:
            # Intentionally plain elapsed-seconds/sample indexing
            # (int(time * sr)), NEVER engine.timestamps-aware. Ripple
            # (and theta) detection must produce identical results
            # whether or not the user has loaded a timestamps.npy file --
            # timestamps are a display-only concern elsewhere in the app,
            # never load-bearing for detection/positioning here.
            start_idx = max(0, int(self.start_time * self.sample_rate))
            end_idx = min(self.raw_data.shape[0], int(self.end_time * self.sample_rate))
            if start_idx >= end_idx:
                raise ValueError("Invalid time range for detection.")

            n_samples = end_idx - start_idx

            # Build the exclude mask ONCE (same window for every
            # channel), in window-relative sample terms.
            exclude_mask = None
            if self.exclude_intervals:
                exclude_mask = np.zeros(n_samples, dtype=bool)
                for lo, hi in self.exclude_intervals:
                    # Clip each global interval to this detection
                    # window, then convert to window-relative indices.
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
                    signal, csd_info = self.pac_analyzer.compute_csd(
                        signal, ch, self.sample_rate, self.raw_data,
                        self.start_time, self.end_time, self.csd_spacing,
                    )
                    if not csd_info.get("used", False):
                        # CSD unavailable for this channel (edge channel,
                        # no valid neighbors) -- fall back to raw signal
                        # rather than silently dropping the channel.
                        signal = self.raw_data[start_idx:end_idx, ch].flatten().astype(np.float64)

                # Filtering (sosfiltfilt) still always runs on the FULL
                # signal -- it needs surrounding context to be accurate,
                # so masking the excluded regions out of the signal
                # itself before filtering would corrupt detection near
                # their edges. exclude_mask instead keeps theta-time
                # samples out of the post-filter envelope's mean/SD
                # baseline AND out of candidate-peak selection (see
                # RippleDetector.detect's exclude_mask parameter) --
                # this is the mechanism for "so we only use the ripple
                # amp thresholds during non-theta", not a purely
                # post-hoc peak-location filter.
                events = self.detector.detect(
                    signal, self.sample_rate, self.params, channel=ch,
                    exclude_mask=exclude_mask,
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
    are pushed to a RippleTraceViewWidget (or a NeuralTraceViewWidget
    composing it) as an interactive overlay -- click-select, drag
    boundaries, delete, merge-on-drag -- rather than rendered in this
    dialog, mirroring how ThetaEpochDialog drives its trace view.
    """

    def __init__(self, probe_data: dict, engine: TraceEngine,
                 trace_view: RippleTraceViewWidget,
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

        # Keep this dialog's events_by_channel / summary table in sync
        # with edits made directly on the trace view (drag boundary,
        # Delete key, merge-on-drag), and keep the trace view's selection
        # in sync with table row selection -- same two-way wiring
        # MainWindow sets up between ThetaEpochDialog and the trace view.
        self.trace_view.rippleEventsChanged.connect(self.set_events_from_trace)
        self.trace_view.rippleEventSelected.connect(self.set_selected_event)

        if initial_channels:
            self.channel_edit.setText(",".join(str(c) for c in initial_channels))

        self._refresh_theta_exclusion_availability()

    def showEvent(self, event):
        # Theta detection can happen (or epochs can be loaded/cleared)
        # while this dialog stays open, so re-check availability every
        # time the dialog becomes visible rather than only once at
        # construction.
        self._refresh_theta_exclusion_availability()
        super().showEvent(event)

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

        self.exclude_theta_check = QCheckBox("Detect ripples outside theta only")
        self.exclude_theta_check.setToolTip(
            "Exclude every theta epoch (on any channel) from both the "
            "amplitude threshold baseline AND candidate detection -- "
            "prevents high-frequency activity during theta from "
            "inflating the ripple threshold. Disabled if no theta "
            "epochs exist yet."
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

        self.load_btn = QPushButton("Load CSV...")
        self.load_btn.setToolTip(
            "Load a previously-exported ripple CSV. Loaded events are "
            "added to the current set and behave exactly like freshly "
            "detected ones (draggable, mergeable, deletable)."
        )
        self.load_btn.clicked.connect(self._on_load_clicked)
        row.addWidget(self.load_btn)

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

    def _theta_global_exclusion_intervals(self) -> list[tuple[int, int]]:
        """
        Build a flat, merged list of [(global_start_sample,
        global_end_sample), ...] from every theta epoch currently on
        self.trace_view, regardless of which channel each epoch was
        detected on.

        Theta is treated as a network/brain-state phenomenon, not a
        per-channel one: if theta was detected anywhere, that time
        window is excluded from ripple detection on EVERY channel being
        processed, not just the channel theta happened to be detected
        on. This mirrors how the person would reason about it manually
        (during a period the brain is in a theta state, don't count
        ripples on any channel).

        Returns [] if the trace view has no theta epochs (or doesn't
        support them at all -- e.g. a plain RippleTraceViewWidget not
        composed with theta).

        Both theta epochs and ripple detection index samples with plain
        sample/sample_rate math only (never engine.timestamps -- see
        _DetectThread.run() and ThetaEpochTraceViewWidget._theta_epoch_time),
        so adding each epoch's start/end sample to its own detection-time
        sample_offset gives a value directly comparable to ripple's
        start_idx + peak_sample used in _DetectThread, with no
        timestamp-related drift possible.
        """
        theta_epochs = getattr(self.trace_view, "theta_epochs", None)
        if not theta_epochs:
            return []

        theta_sample_offset = int(getattr(self.trace_view, "theta_detection_sample_offset", 0))

        intervals = sorted(
            (theta_sample_offset + int(e.start_sample), theta_sample_offset + int(e.end_sample))
            for e in theta_epochs
        )

        # Merge overlapping/adjacent intervals so downstream containment
        # checks (any(lo <= x <= hi ...)) and baseline-mask construction
        # don't have to reason about redundant overlapping ranges.
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
            # exclude_theta_check is only enabled when theta epochs
            # exist (see _refresh_theta_exclusion_availability), so
            # exclude_intervals should be non-empty here; an empty
            # result would only happen if theta epochs were cleared on
            # the trace view between the checkbox being enabled and
            # clicking Detect, which is harmless -- detection simply
            # proceeds without exclusion.

        self.detect_btn.setEnabled(False)
        self.detect_btn.setText("Detecting...")
        self.status_label.setText(
            "Running detection (excluding theta epochs), please wait..."
            if exclude_intervals else
            "Running detection, please wait..."
        )

        # Sample offset for absolute-position export/overlay: samples from
        # the start of the recording to the start of the analyzed window.
        self._sample_offset = int(self.start_spin.value() * self.engine.sr)

        self._detect_thread = _DetectThread(
            channels, self.engine.data, self.engine.sr,
            self.start_spin.value(), self.end_spin.value(), params,
            self.csd_check.isChecked(), self.csd_spacing_spin.value(),
            self.pac_analyzer,
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
    # Overlay: pushing results to the main trace view
    # ------------------------------------------------------------------
    #
    # The trace view (RippleTraceViewWidget, or a NeuralTraceViewWidget
    # composing it) owns interactive ripple state as a flat
    # list[RippleEvent] -- the same shape ThetaEpochTraceViewWidget uses
    # for theta epochs -- plus a separate per-channel render-context dict
    # for envelope/signal/threshold data that isn't part of a ripple
    # event's own identity. This dialog is the source of truth for
    # events_by_channel (used by the summary table and CSV export);
    # set_ripple_events()/set_ripple_render_context() push a fresh
    # flattened view of it whenever detection results or a manual
    # edit (drag/delete/merge, via rippleEventsChanged) change it.

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
            if envelope is None or signal is None:
                continue
            render_context[ch] = {
                'envelope': envelope,
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
        )
        self.trace_view.set_ripple_render_context(render_context)
        self.trace_view.set_ripple_events(all_events)

    def _on_overlay_visibility_changed(self, _checked: bool):
        self.trace_view.set_ripple_overlay_visibility(
            show_envelope=self.show_envelope_check.isChecked(),
            show_thresholds=self.show_thresholds_check.isChecked(),
            show_events=self.show_events_check.isChecked(),
        )

    def set_events_from_trace(self, events: list[RippleEvent]):
        """Slot for trace_view.rippleEventsChanged -- keeps
        events_by_channel (and therefore the summary table / CSV export)
        in sync after an in-place drag/delete/merge edit made directly
        on the trace view, mirroring ThetaEpochDialog.set_epochs_from_trace."""
        regrouped: dict[int, list[RippleEvent]] = {}
        for ev in events:
            regrouped.setdefault(ev.channel, []).append(ev)
        # Preserve channels that detection covered but that now have zero
        # events (e.g. the user deleted the only ripple on that channel)
        # so the summary table still shows a 0-ripple row instead of the
        # channel disappearing entirely.
        for ch in self.events_by_channel.keys():
            regrouped.setdefault(ch, [])
        self.events_by_channel = regrouped
        self._populate_summary_table()

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

    def set_selected_event(self, index: int):
        """Slot for trace_view.rippleEventSelected -- index is into the
        trace view's flat, (channel, start_sample)-sorted event list.
        Mirrors ThetaEpochDialog.set_selected_epoch, but this dialog's
        table is grouped by channel rather than listing individual
        events, so selecting an event just makes sure that event's
        channel is visible/selected in the summary table."""
        all_events = sorted(
            (ev for evs in self.events_by_channel.values() for ev in evs),
            key=lambda e: (e.channel, e.start_sample),
        )
        if not (0 <= index < len(all_events)):
            return
        channel = all_events[index].channel
        for row in range(self.summary_table.rowCount()):
            item = self.summary_table.item(row, 0)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == channel:
                self.summary_table.selectRow(row)
                break

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

    def _on_load_clicked(self):
        if not self.engine.data_loaded:
            QMessageBox.warning(self, "No data", "No data is currently loaded.")
            return
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Load Ripples", "", "CSV files (*.csv);;All files (*)"
        )
        if not path_str:
            return

        try:
            events = read_ripples_from_csv(path_str)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to load", f"Could not read {path_str}:\n\n{exc}")
            return

        if not events:
            QMessageBox.information(self, "No events found", f"{path_str} contains no ripple events.")
            return

        # read_ripples_from_csv returns ABSOLUTE (recording-global) sample
        # positions, unmodified -- unlike theta's importer this isn't
        # re-based to any particular sample_offset. Loaded events are
        # ADDED to whatever's already in events_by_channel (rather than
        # replacing it) so loading multiple exported files, or loading on
        # top of a fresh detection run, accumulates rather than clobbers.
        # This does mean a loaded channel's sample_offset must be treated
        # as 0 for rendering; channels that also have live detection
        # results keep their detection-time sample_offset/envelope --
        # mixing the two on the SAME channel in one session isn't
        # supported (the events would be interpreted against whichever
        # sample_offset that channel's render context currently uses).
        if self.events_by_channel and self._sample_offset != 0:
            proceed = QMessageBox.question(
                self, "Sample offset mismatch",
                "Currently detected events use a non-zero sample offset "
                f"({self._sample_offset} samples), but loaded events are "
                "always absolute (offset 0). Loaded events may not "
                "display in the correct position relative to existing "
                "ones on the same channel.\n\nLoad anyway?",
            )
            if proceed != QMessageBox.StandardButton.Yes:
                return

        for ev in events:
            self.events_by_channel.setdefault(ev.channel, []).append(ev)
        for ch in self.events_by_channel:
            self.events_by_channel[ch].sort(key=lambda e: e.start_sample)

        self._populate_summary_table()
        self._push_overlay_to_trace_view()
        self.status_label.setText(f"Loaded {len(events)} ripple event(s) from {path_str}.")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if self._detect_thread is not None and self._detect_thread.isRunning():
            self._detect_thread.wait(2000)
        # Disconnect before clearing so clear_ripple_overlay()'s
        # rippleEventsChanged-adjacent state resets don't loop back into
        # this (about to be destroyed) dialog's slots.
        try:
            self.trace_view.rippleEventsChanged.disconnect(self.set_events_from_trace)
        except TypeError:
            pass
        try:
            self.trace_view.rippleEventSelected.disconnect(self.set_selected_event)
        except TypeError:
            pass
        self.trace_view.clear_ripple_overlay()
        super().closeEvent(event)
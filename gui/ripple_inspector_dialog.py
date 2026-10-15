"""
ripple_inspector_dialog.py

Non-modal dialog for inspecting a single ripple at a time. Same
probe-geometry rendering as RippleTriggeredAverageDialog (via
RTAProbeTraceWidget), but for one event instead of an average, with
Prev / Next navigation and a Reject button.

Reject removes the ripple from the parent RippleDialog's
events_by_channel and pushes an undo snapshot onto the main trace
view's ripple undo stack, so Ctrl+Z on the trace view restores it.

The dialog listens to its parent's ``eventsChanged`` signal so that
undoes and CSV loads refresh the current view; it emits
``rippleSelected`` and ``rejectRequested`` so the parent can keep the
table and overlay in sync.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, QComboBox,
    QPushButton, QMessageBox, QListWidget, QListWidgetItem,
    QAbstractItemView, QWidget,
)

from gui.ripple_triggered_average_dialog import ProbePickerDialog

from core.ripple_detector import RippleEvent, RippleParams
from core.ripple_triggered_average import (
    RippleTriggeredAverageAnalyzer, RippleTriggeredAverageParams,
    ALIGNMENT_MODES,
)
from core.trace_engine import TraceEngine
from gui.rta_probe_trace_widget import RTAProbeTraceWidget


ALIGNMENT_LABELS = {
    "envelope_peak": "Ripple max amplitude (envelope peak)",
    "filtered_max": "Ripple max cycle (filtered signal peak)",
    "filtered_min": "Ripple min cycle (filtered signal trough)",
}
ALIGNMENT_LABELS_REVERSE = {v: k for k, v in ALIGNMENT_LABELS.items()}


class _ComputeThread(QThread):
    """Computes the single-event window off the UI thread. The compute
    itself is cheap (one extraction per channel), but reusing the same
    pattern as the RTA dialog keeps the two implementations parallel
    and avoids blocking if the display set is large."""

    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, analyzer, raw_data, sample_rate, events, params, parent=None):
        super().__init__(parent)
        self.analyzer = analyzer
        self.raw_data = raw_data
        self.sample_rate = sample_rate
        self.events = events
        self.params = params

    def run(self):
        try:
            result = self.analyzer.compute(
                self.raw_data, self.sample_rate, self.events, self.params
            )
            self.finished_ok.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class RippleInspectorDialog(QDialog):
    """Non-modal single-ripple inspector."""

    # Emitted when the user clicks Reject on the currently-displayed
    # ripple. The parent (RippleDialog) removes the event, updates the
    # table + trace overlay, and pushes an undo snapshot.
    rejectRequested = pyqtSignal(int, int, int)   # (channel, start_sample, end_sample)
    # Emitted whenever the currently-displayed ripple changes, so the
    # parent can highlight the corresponding table row.
    rippleSelected = pyqtSignal(int, int)         # (channel, start_sample)

    def __init__(
        self,
        engine: TraceEngine,
        probe_data: dict,
        events_by_channel: dict[int, list[RippleEvent]],
        sample_offset: int,
        ripple_params: RippleParams,
        initial_channels: list[int],
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Single Ripple View")
        self.setModal(False)
        self.resize(1000, 720)

        self.engine = engine
        self.probe_data = probe_data
        self._events_by_channel: dict[int, list[RippleEvent]] = {
            ch: list(evs) for ch, evs in (events_by_channel or {}).items()
        }
        self._sample_offset = int(sample_offset)
        self.ripple_params = ripple_params
        self.analyzer = RippleTriggeredAverageAnalyzer(ripple_params)

        # Channels currently plotted on the probe trace widget. Distinct
        # from the trigger channel (which is the ripple source).
        self.display_channels: list[int] = list(initial_channels)
        self._last_geometry_channels: list[int] | None = None
        
        # Index into the trigger channel's own flat event list. None
        # before the first load.
        self._trigger_channel: int | None = None
        self._trigger_events: list[RippleEvent] = []
        self._index: int | None = None

        self._compute_thread: _ComputeThread | None = None
        self._last_result: dict | None = None

        self._build_ui()
        self._refresh_trigger_channel_dropdown()
        # Pick the first channel with events as the initial trigger.
        if self._trigger_channel is not None:
            self._index = 0
            self._update_display()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        layout.addLayout(self._build_top_row())
        layout.addLayout(self._build_nav_row())

        self.probe_trace_widget = RTAProbeTraceWidget()
        layout.addWidget(self.probe_trace_widget, stretch=1)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self.status_label)

    def _build_top_row(self) -> QHBoxLayout:
        row = QHBoxLayout()

        row.addWidget(QLabel("Trigger channel:"))
        self.trigger_combo = QComboBox()
        self.trigger_combo.setMinimumWidth(160)
        self.trigger_combo.currentIndexChanged.connect(self._on_trigger_changed)
        row.addWidget(self.trigger_combo)

        # ---- Display channels: which channels to plot on the probe
        # geometry below. Distinct from the trigger channel dropdown:
        # the trigger picks which ripple is inspected, the display set
        # picks which channels appear in the plot. Defaults to whatever
        # the parent RippleDialog passed in (typically the currently
        # visible trace-view channels). ----
        self.pick_from_probe_btn = QPushButton("Display channels...")
        self.pick_from_probe_btn.setAutoDefault(False)
        self.pick_from_probe_btn.setDefault(False)
        self.pick_from_probe_btn.setToolTip(
            "Choose which channels to plot on the probe geometry. "
            "Opens a probe map seeded with the current display set; "
            "click electrodes to add or remove them."
        )
        self.pick_from_probe_btn.clicked.connect(self._on_pick_from_probe_clicked)
        row.addWidget(self.pick_from_probe_btn)

        row.addWidget(QLabel("Window before (ms):"))
        self.before_spin = QDoubleSpinBox()
        self.before_spin.setRange(1.0, 5000.0)
        self.before_spin.setValue(100.0)
        self.before_spin.valueChanged.connect(self._on_window_changed)
        row.addWidget(self.before_spin)

        row.addWidget(QLabel("after (ms):"))
        self.after_spin = QDoubleSpinBox()
        self.after_spin.setRange(1.0, 5000.0)
        self.after_spin.setValue(100.0)
        self.after_spin.valueChanged.connect(self._on_window_changed)
        row.addWidget(self.after_spin)

        row.addWidget(QLabel("Align:"))
        self.alignment_combo = QComboBox()
        self.alignment_combo.addItems(
            [ALIGNMENT_LABELS[m] for m in ALIGNMENT_MODES]
        )
        self.alignment_combo.currentIndexChanged.connect(
            self._on_alignment_changed
        )
        row.addWidget(self.alignment_combo)

        row.addStretch(1)
        return row

    def _build_nav_row(self) -> QHBoxLayout:
        row = QHBoxLayout()

        self.prev_btn = QPushButton("\u25c4 Prev Ripple")
        self.prev_btn.setAutoDefault(False)
        self.prev_btn.setDefault(False)
        self.prev_btn.clicked.connect(lambda: self._step(-1))
        row.addWidget(self.prev_btn)

        self.position_label = QLabel("")
        self.position_label.setStyleSheet("color: #ccc;")
        row.addWidget(self.position_label)

        self.next_btn = QPushButton("Next Ripple \u25ba")
        self.next_btn.setAutoDefault(False)
        self.next_btn.setDefault(False)
        self.next_btn.clicked.connect(lambda: self._step(+1))
        row.addWidget(self.next_btn)

        row.addSpacing(20)

        self.reject_btn = QPushButton("Reject Ripple")
        self.reject_btn.setAutoDefault(False)
        self.reject_btn.setDefault(False)
        self.reject_btn.setToolTip(
            "Remove the currently-displayed ripple from the parent "
            "dialog's event list. The row disappears from the ripple "
            "table and the marker disappears from the main trace view. "
            "Ctrl+Z on the trace view undoes it."
        )
        self.reject_btn.clicked.connect(self._on_reject_clicked)
        row.addWidget(self.reject_btn)

        row.addStretch(1)
        return row

    # ------------------------------------------------------------------
    # Trigger channel / events
    # ------------------------------------------------------------------

    def _refresh_trigger_channel_dropdown(self):
        current = self._trigger_channel
        self.trigger_combo.blockSignals(True)
        self.trigger_combo.clear()
        channels_with_events = sorted(
            ch for ch, evs in self._events_by_channel.items() if evs
        )
        for ch in channels_with_events:
            n = len(self._events_by_channel[ch])
            self.trigger_combo.addItem(f"CH {ch}  ({n} ripples)", ch)
        if current is not None:
            idx = self.trigger_combo.findData(current)
            if idx >= 0:
                self.trigger_combo.setCurrentIndex(idx)
        self.trigger_combo.blockSignals(False)

        if self.trigger_combo.count() == 0:
            self._trigger_channel = None
            self._trigger_events = []
            self._index = None
        else:
            if self.trigger_combo.currentIndex() < 0:
                self.trigger_combo.setCurrentIndex(0)
            self._trigger_channel = int(self.trigger_combo.currentData())
            self._refresh_trigger_events()

    def _refresh_trigger_events(self):
        if self._trigger_channel is None:
            self._trigger_events = []
            self._index = None
            return
        evs = self._events_by_channel.get(self._trigger_channel, [])
        self._trigger_events = sorted(evs, key=lambda e: e.start_sample)
        if not self._trigger_events:
            self._index = None
        elif self._index is None or self._index >= len(self._trigger_events):
            self._index = len(self._trigger_events) - 1

    def _current_event(self) -> RippleEvent | None:
        if self._index is None:
            return None
        if not (0 <= self._index < len(self._trigger_events)):
            return None
        return self._trigger_events[self._index]

    # ------------------------------------------------------------------
    # Slot: parent told us events changed (undo, CSV load, trace edit)
    # ------------------------------------------------------------------

    def on_events_changed(self):
        """Called when the parent's events_by_channel changes. Merge
        back in the current state (the parent owns the truth), then
        refresh the trigger channel and the display."""
        # We don't get the new dict directly -- the parent emits a
        # bare signal -- so pull it by re-reading through the parent's
        # public attribute. This is a bit under-coupled but avoids
        # adding a payload to the signal; the parent is a QDialog and
        # accessible via self.parent().
        parent = self.parent()
        if parent is not None and hasattr(parent, "events_by_channel"):
            self._events_by_channel = {
                ch: list(evs)
                for ch, evs in parent.events_by_channel.items()
            }

        # If the trigger channel disappeared entirely (all its events
        # were rejected), pick a new one.
        if (self._trigger_channel is not None
                and not self._events_by_channel.get(self._trigger_channel, [])):
            self._trigger_channel = None
            self._refresh_trigger_channel_dropdown()
        else:
            self._refresh_trigger_channel_dropdown()
            self._refresh_trigger_events()
        self._update_display()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _step(self, direction: int):
        if not self._trigger_events:
            return
        if self._index is None:
            self._index = 0
        else:
            self._index = max(0, min(self._index + direction,
                                     len(self._trigger_events) - 1))
        self._update_display()

    def _on_trigger_changed(self, _idx: int):
        self._trigger_channel = int(self.trigger_combo.currentData())
        self._index = 0
        self._refresh_trigger_events()
        self._update_display()

    def _on_pick_from_probe_clicked(self):
        """Open a scoped probe map seeded with the current display set;
        on accept, adopt the new selection and redraw."""
        picker = ProbePickerDialog(
            self.probe_data, self.display_channels, parent=self
        )
        if picker.exec() == QDialog.DialogCode.Accepted:
            new_selection = picker.selected_channels()
            if new_selection:
                self.display_channels = new_selection
                # Invalidate the geometry cache so the next _update_display
                # (called via _ensure_geometry_for) actually rebuilds the
                # layout -- the user asked for a different channel set.
                self._last_geometry_channels = None
                self._update_display()

    def _on_window_changed(self, _v: float):
        self._update_display()

    def _on_alignment_changed(self, _idx: int):
        self._update_display()

    # ------------------------------------------------------------------
    # Display / compute
    # ------------------------------------------------------------------

    def _update_display(self):
        event = self._current_event()
        if event is None:
            self.position_label.setText("(no ripples)")
            self.probe_trace_widget.set_result(
                np.array([0.0]), {}, show_sem=False
            )
            self.status_label.setText("No ripple available to display.")
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            self.reject_btn.setEnabled(False)
            return

        self.prev_btn.setEnabled(self._index is not None and self._index > 0)
        self.next_btn.setEnabled(
            self._index is not None
            and self._index < len(self._trigger_events) - 1
        )
        self.reject_btn.setEnabled(True)

        n = len(self._trigger_events)
        self.position_label.setText(
            f"Ripple {self._index + 1} of {n} on CH{self._trigger_channel}"
        )

        # Emit selection so the parent can highlight the matching row.
        self.rippleSelected.emit(
            int(event.channel), int(event.start_sample)
        )

        # Build the params for a single-event compute.
        alignment = ALIGNMENT_LABELS_REVERSE[self.alignment_combo.currentText()]
        # The display set is whatever the parent passed at open time;
        # the trigger channel is always included.
        channels = list(self.display_channels)
        if int(event.channel) not in channels:
            channels.append(int(event.channel))

        # Re-base the single event to absolute samples (the analyzer
        # expects global sample indices; see ripple_triggered_average
        # docstring).
        import dataclasses
        offset = int(self._sample_offset)
        event_global = dataclasses.replace(
            event,
            start_sample=event.start_sample + offset,
            end_sample=event.end_sample + offset,
            peak_sample=event.peak_sample + offset,
            trough_sample=event.trough_sample + offset,
        )

        params = RippleTriggeredAverageParams(
            channels=channels,
            window_before_ms=float(self.before_spin.value()),
            window_after_ms=float(self.after_spin.value()),
            alignment=alignment,
        )

        # Single event, computed synchronously -- it's one window
        # extraction per channel, cheap enough to do on the UI thread.
        try:
            result = self.analyzer.compute(
                self.engine.data, self.engine.sr,
                [event_global], params,
            )
        except Exception as exc:
            self.status_label.setText(f"Compute failed: {exc}")
            return

        # Only rebuild the layout if the channel set actually changed.
        # Otherwise the canvas's zoom / pan / gain get reset by the
        # _auto_range() call inside set_geometry_from_probe, and every
        # Prev/Next step would snap the view back to fit. The set only
        # changes when the user picks new display channels or the
        # trigger channel adds itself.
        self._ensure_geometry_for(channels)
        self.probe_trace_widget.set_result(
            result["time_axis"], result["channels"], show_sem=False,
        )

        alignment_label = ALIGNMENT_LABELS[alignment]
        self.status_label.setText(
            f"CH{event.channel} ripple  |  "
            f"start {self.engine.sample_to_time(int(event.start_sample) + offset):.3f}s  |  "
            f"aligned on: {alignment_label}"
        )

    def _ensure_geometry_for(self, channels: list[int]):
        """Rebuild the probe trace widget's channel layout ONLY when the
        display set has actually changed. Calling set_geometry_from_probe
        unconditionally resets the canvas's zoom, pan, and gain
        (_auto_range runs inside it), which we do not want to happen on
        every Prev/Next step -- the user's view state must survive
        navigating between ripples on the same display set."""
        if not channels:
            return
        new_set = list(channels)
        if getattr(self, "_last_geometry_channels", None) == new_set:
            return
        self._last_geometry_channels = new_set
        self.probe_trace_widget.set_geometry_from_probe(
            self.probe_data, new_set
        )


    # ------------------------------------------------------------------
    # Reject
    # ------------------------------------------------------------------

    def _on_reject_clicked(self):
        event = self._current_event()
        if event is None:
            return
        # Emit; the parent removes it and pushes an undo snapshot.
        self.rejectRequested.emit(
            int(event.channel),
            int(event.start_sample),
            int(event.end_sample),
        )
        # The parent will fire eventsChanged, which calls
        # on_events_changed and updates our view. The current index
        # stays the same unless the removed event was the last in the
        # list, in which case _refresh_trigger_events clamps it.

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if self._compute_thread is not None and self._compute_thread.isRunning():
            self._compute_thread.wait(2000)
        super().closeEvent(event)
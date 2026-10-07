"""
channel_options_panel.py

Floating popup panel for per-channel trace customization (Raw / Filtered /
CSD). One instance per channel; spawned by clicking the channel-number
button on the left of the trace view.

Not dockable. Closes when the user clicks outside it, presses Escape, or
clicks the channel button again. Opens with "Show Raw" pre-checked.

Design contract with TraceViewWidget:
  - TraceViewWidget owns the authoritative per-channel state (dict of
    ChannelOptions).
  - This panel reads that state on open and emits `optionsChanged` with
    the full updated ChannelOptions every time anything is toggled or
    edited.
  - TraceViewWidget writes the new state back and repaints. The panel
    does not decide what to draw -- it just reports user intent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QCheckBox,
    QDoubleSpinBox, QGroupBox, QSizePolicy,
)


@dataclass
class ChannelOptions:
    """Per-channel display + filter + per-mode gain options."""
    show_raw: bool = True
    raw_gain: float = 1.0

    show_filtered: bool = False
    filter_low: float = 1.0
    filter_high: float = 300.0
    filtered_gain: float = 1.0

    show_csd: bool = False
    csd_low: float = 0.0
    csd_high: float = 0.0
    csd_distance: float = 30.0
    csd_gain: float = 1.0

    def is_default(self) -> bool:
        """True if this is exactly the default state (raw on at gain 1,
        everything else off)."""
        return (
            self.show_raw
            and not self.show_filtered
            and not self.show_csd
            and self.raw_gain == 1.0
        )


class ChannelOptionsPanel(QWidget):
    """Floating panel for editing one channel's ChannelOptions."""

    optionsChanged = pyqtSignal(int, object)  # (channel, ChannelOptions)
    closed = pyqtSignal()

    def __init__(self, channel: int, options: ChannelOptions, parent=None):
        super().__init__(parent, Qt.WindowType.Tool)
        self.channel = int(channel)
        self.setWindowTitle(f"CH{self.channel}")
        self.setMinimumWidth(260)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        # Copy EVERY field. A hand-enumerated copy silently dropped the
        # gain fields, which then defaulted to 1.0 on every reopen.
        # dataclasses.replace() also means future fields added to
        # ChannelOptions are carried through automatically.
        import dataclasses
        self._options = dataclasses.replace(options)

        self._build_ui()

        # Load the saved values into the widgets FIRST. Any paired-range
        # priming must run AFTER this, because the range handlers end by
        # calling _on_any_changed(), which reads every widget back into
        # self._options -- running them before _sync_from_options would
        # clobber the saved state with the widgets' fresh-construction
        # defaults (which is exactly the "reopening looks like a fresh
        # opening" bug).
        self._sync_from_options()

        # Now safe to prime the paired-range handlers: widgets hold the
        # saved values, so _on_any_changed() inside them is a no-op on
        # every field that wasn't touched.
        self._on_filter_low_edited(self.filter_low_spin.value())
        self._on_csd_low_edited(self.csd_low_spin.value())

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        header = QLabel(f"Channel {self.channel}")
        header.setStyleSheet("font-weight: bold; color: #fff;")
        layout.addWidget(header)

        # ---- Raw ----
        raw_group = QGroupBox("Raw")
        raw_layout = QGridLayout(raw_group)
        self.raw_checkbox = QCheckBox("Show Raw")
        self.raw_checkbox.toggled.connect(self._on_any_changed)
        raw_layout.addWidget(self.raw_checkbox, 0, 0, 1, 2)
        raw_layout.addWidget(QLabel("Gain:"), 1, 0)
        self.raw_gain_spin = QDoubleSpinBox()
        self.raw_gain_spin.setRange(0.01, 1000.0)
        self.raw_gain_spin.setSingleStep(0.1)
        self.raw_gain_spin.setDecimals(3)
        self.raw_gain_spin.setValue(1.0)
        self.raw_gain_spin.valueChanged.connect(self._on_any_changed)
        raw_layout.addWidget(self.raw_gain_spin, 1, 1)
        layout.addWidget(raw_group)


        # ---- Filtered ----
        filt_group = QGroupBox("Filtered")
        filt_layout = QGridLayout(filt_group)
        self.filtered_checkbox = QCheckBox("Show Filtered")
        self.filtered_checkbox.toggled.connect(self._on_any_changed)
        filt_layout.addWidget(self.filtered_checkbox, 0, 0, 1, 2)


        filt_layout.addWidget(QLabel("Low (Hz):"), 1, 0)
        self.filter_low_spin = QDoubleSpinBox()
        self.filter_low_spin.setRange(0.0, 15000.0)
        self.filter_low_spin.setSingleStep(1.0)
        self.filter_low_spin.valueChanged.connect(self._on_any_changed)
        self.filter_low_spin.valueChanged.connect(self._on_filter_low_edited)
        filt_layout.addWidget(self.filter_low_spin, 1, 1)
                
        
        filt_layout.addWidget(QLabel("High (Hz):"), 2, 0)
        self.filter_high_spin = QDoubleSpinBox()
        self.filter_high_spin.setRange(0.0, 15000.0)
        self.filter_high_spin.setSingleStep(1.0)
        self.filter_high_spin.valueChanged.connect(self._on_any_changed)
        self.filter_high_spin.valueChanged.connect(self._on_filter_high_edited)
        filt_layout.addWidget(self.filter_high_spin, 2, 1)
        
        
        filt_layout.addWidget(QLabel("Gain:"), 3, 0)
        self.filtered_gain_spin = QDoubleSpinBox()
        self.filtered_gain_spin.setRange(0.01, 1000.0)
        self.filtered_gain_spin.setSingleStep(0.1)
        self.filtered_gain_spin.setDecimals(3)
        self.filtered_gain_spin.setValue(1.0)
        self.filtered_gain_spin.valueChanged.connect(self._on_any_changed)
        filt_layout.addWidget(self.filtered_gain_spin, 3, 1)
        layout.addWidget(filt_group)


        # ---- CSD ----
        csd_group = QGroupBox("CSD")
        csd_layout = QGridLayout(csd_group)
        self.csd_checkbox = QCheckBox("Show CSD")
        self.csd_checkbox.toggled.connect(self._on_any_changed)
        csd_layout.addWidget(self.csd_checkbox, 0, 0, 1, 2)
        csd_layout.addWidget(QLabel("Band low (Hz):"), 1, 0)
        self.csd_low_spin = QDoubleSpinBox()
        self.csd_low_spin.setRange(0.0, 15000.0)
        self.csd_low_spin.setSingleStep(1.0)
        self.csd_low_spin.setToolTip("0 = compute CSD directly from raw")


        self.csd_low_spin.valueChanged.connect(self._on_any_changed)
        self.csd_low_spin.valueChanged.connect(self._on_csd_low_edited)
        csd_layout.addWidget(self.csd_low_spin, 1, 1)
        csd_layout.addWidget(QLabel("Band high (Hz):"), 2, 0)
        self.csd_high_spin = QDoubleSpinBox()
        self.csd_high_spin.setRange(0.0, 15000.0)
        self.csd_high_spin.setSingleStep(1.0)
        self.csd_high_spin.setToolTip("0 = compute CSD directly from raw")
        self.csd_high_spin.valueChanged.connect(self._on_csd_high_edited)
        self.csd_high_spin.valueChanged.connect(self._on_any_changed)
        csd_layout.addWidget(self.csd_high_spin, 2, 1)



        csd_layout.addWidget(QLabel("Distance (µm):"), 3, 0)
        self.csd_distance_spin = QDoubleSpinBox()
        self.csd_distance_spin.setRange(1.0, 500.0)
        self.csd_distance_spin.setSingleStep(5.0)
        self.csd_distance_spin.setToolTip(
            "Minimum |y gap| to the neighbor channel used for the Laplacian. "
            "If two same-shank channels share a y, x is used to break ties."
        )
        self.csd_distance_spin.valueChanged.connect(self._on_any_changed)
        csd_layout.addWidget(self.csd_distance_spin, 3, 1)
        csd_layout.addWidget(QLabel("Gain:"), 4, 0)
        self.csd_gain_spin = QDoubleSpinBox()
        self.csd_gain_spin.setRange(0.01, 10000.0)
        self.csd_gain_spin.setSingleStep(0.1)
        self.csd_gain_spin.setDecimals(3)
        self.csd_gain_spin.setValue(1.0)
        self.csd_gain_spin.setToolTip(
            "CSD amplitude is much smaller than raw LFP (divided by "
            "spacing² in µm²). Increase gain to make it visible."
        )
        self.csd_gain_spin.valueChanged.connect(self._on_any_changed)
        csd_layout.addWidget(self.csd_gain_spin, 4, 1)
        self.csd_status_label = QLabel("")
        self.csd_status_label.setStyleSheet("color: #888; font-size: 10px;")
        self.csd_status_label.setWordWrap(True)
        csd_layout.addWidget(self.csd_status_label, 5, 0, 1, 2)
        layout.addWidget(csd_group)

        # 1. Paired-range priming: low/high spinboxes start consistent.
        #self._on_filter_low_edited(self.filter_low_spin.value())
        #if hasattr(self, "csd_low_spin"):
        #    self._on_csd_low_edited(self.csd_low_spin.value())

        # 2. Enabled-state sync: greys out sub-controls whose parent
        # checkbox is off (Filtered unchecked → its spinboxes disabled,
        # etc.).
        #self._update_enabled_states()




    def _on_filter_low_edited(self, value: float):
        step = self.filter_high_spin.singleStep()
        new_min = value + step
        high_max = self.filter_high_spin.maximum()
        if new_min > high_max:
            new_min = high_max
            value = high_max - step
            self.filter_low_spin.blockSignals(True)
            self.filter_low_spin.setValue(value)
            self.filter_low_spin.blockSignals(False)
        self.filter_high_spin.blockSignals(True)
        self.filter_high_spin.setMinimum(new_min)
        self.filter_high_spin.blockSignals(False)
        self._on_any_changed()

    def _on_filter_high_edited(self, value: float):
        step = self.filter_low_spin.singleStep()
        new_max = value - step
        low_min = self.filter_low_spin.minimum()
        if new_max < low_min:
            new_max = low_min
            value = low_min + step
            self.filter_high_spin.blockSignals(True)
            self.filter_high_spin.setValue(value)
            self.filter_high_spin.blockSignals(False)
        self.filter_low_spin.blockSignals(True)
        self.filter_low_spin.setMaximum(new_max)
        self.filter_low_spin.blockSignals(False)
        self._on_any_changed()

    def _on_csd_low_edited(self, value: float):
        """CSD band low. 0 is a valid sentinel meaning 'no low cutoff';
        the low < high ordering is only enforced when both sides are
        nonzero. Ranges are never mutated here -- only values are
        adjusted when they'd violate the ordering -- so both spinboxes
        stay able to reach 0 at any time, which is essential since 0
        disables the band (see
        PhaseAmplitudeAnalyzer.compute_csd_with_options)."""
        step = self.csd_low_spin.singleStep()
        high_value = self.csd_high_spin.value()

        if value > 0 and high_value > 0 and value >= high_value:
            new_value = max(0.0, high_value - step)
            self.csd_low_spin.blockSignals(True)
            self.csd_low_spin.setValue(new_value)
            self.csd_low_spin.blockSignals(False)

        self._on_any_changed()

    def _on_csd_high_edited(self, value: float):
        """CSD band high. Mirror of _on_csd_low_edited."""
        step = self.csd_high_spin.singleStep()
        low_value = self.csd_low_spin.value()

        if value > 0 and low_value > 0 and value <= low_value:
            new_value = low_value + step
            high_max = self.csd_high_spin.maximum()
            if new_value > high_max:
                # Can't push high above low's current value -- pull low
                # down instead. This never forces low to a nonzero value.
                new_value = high_max
                new_low = max(0.0, new_value - step)
                self.csd_low_spin.blockSignals(True)
                self.csd_low_spin.setValue(new_low)
                self.csd_low_spin.blockSignals(False)
            self.csd_high_spin.blockSignals(True)
            self.csd_high_spin.setValue(new_value)
            self.csd_high_spin.blockSignals(False)

        self._on_any_changed()
    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------
    def _sync_from_options(self):
        o = self._options
        self.raw_checkbox.blockSignals(True)
        self.raw_gain_spin.blockSignals(True)
        self.filtered_checkbox.blockSignals(True)
        self.csd_checkbox.blockSignals(True)
        self.filter_low_spin.blockSignals(True)
        self.filter_high_spin.blockSignals(True)
        self.filtered_gain_spin.blockSignals(True)
        self.csd_low_spin.blockSignals(True)
        self.csd_high_spin.blockSignals(True)
        self.csd_distance_spin.blockSignals(True)
        self.csd_gain_spin.blockSignals(True)

        self.raw_checkbox.setChecked(o.show_raw)
        self.raw_gain_spin.setValue(o.raw_gain)
        self.filtered_checkbox.setChecked(o.show_filtered)
        self.csd_checkbox.setChecked(o.show_csd)
        self.filter_low_spin.setValue(o.filter_low)
        self.filter_high_spin.setValue(o.filter_high)
        self.filtered_gain_spin.setValue(o.filtered_gain)
        self.csd_low_spin.setValue(o.csd_low)
        self.csd_high_spin.setValue(o.csd_high)
        self.csd_distance_spin.setValue(o.csd_distance)
        self.csd_gain_spin.setValue(o.csd_gain)

        self.raw_checkbox.blockSignals(False)
        self.raw_gain_spin.blockSignals(False)
        self.filtered_checkbox.blockSignals(False)
        self.csd_checkbox.blockSignals(False)
        self.filter_low_spin.blockSignals(False)
        self.filter_high_spin.blockSignals(False)
        self.filtered_gain_spin.blockSignals(False)
        self.csd_low_spin.blockSignals(False)
        self.csd_high_spin.blockSignals(False)
        self.csd_distance_spin.blockSignals(False)
        self.csd_gain_spin.blockSignals(False)

        self._update_enabled_states()

    def _update_enabled_states(self):
        raw_on = self.raw_checkbox.isChecked()
        self.raw_gain_spin.setEnabled(raw_on)
        filt_on = self.filtered_checkbox.isChecked()
        self.filter_low_spin.setEnabled(filt_on)
        self.filter_high_spin.setEnabled(filt_on)
        self.filtered_gain_spin.setEnabled(filt_on)
        csd_on = self.csd_checkbox.isChecked()
        self.csd_low_spin.setEnabled(csd_on)
        self.csd_high_spin.setEnabled(csd_on)
        self.csd_distance_spin.setEnabled(csd_on)
        self.csd_gain_spin.setEnabled(csd_on)

    def set_sample_rate(self, sample_rate: float):
        """Derive the per-channel filter and CSD spinbox ranges from the
        loaded recording's sample rate. Safe upper bound is
        0.95 * Nyquist, matching TraceControlPanel.set_sample_rate.

        Called right before the panel is shown, and again if the data
        source changes while the panel is open. Re-primes the paired-
        range guards against the new upper bound, then re-runs the
        checkbox-gating sync so each field's enabled state still
        follows its parent checkbox (Filtered, CSD).
        """
        if sample_rate is None or sample_rate <= 0:
            return
        nyq = 0.5 * float(sample_rate)
        safe_high = max(1.0, 0.95 * nyq)

        self.filter_low_spin.setRange(0.0, safe_high)
        self.filter_high_spin.setRange(0.0, safe_high)
        self.csd_low_spin.setRange(0.0, safe_high)
        self.csd_high_spin.setRange(0.0, safe_high)

        # Re-prime paired-range guards against the new bound. Runs
        # before _update_enabled_states so the ranges are already
        # correct by the time enabled-ness is applied.
        self._on_filter_low_edited(self.filter_low_spin.value())
        self._on_csd_low_edited(self.csd_low_spin.value())

        # Re-sync enabled states against the Filtered / CSD checkboxes.
        self._update_enabled_states()

        
    # ------------------------------------------------------------------
    # Change handling
    # ------------------------------------------------------------------

    def _on_any_changed(self, *args):
        self._options.show_raw = self.raw_checkbox.isChecked()
        self._options.raw_gain = float(self.raw_gain_spin.value())
        self._options.show_filtered = self.filtered_checkbox.isChecked()
        self._options.filter_low = float(self.filter_low_spin.value())
        self._options.filter_high = float(self.filter_high_spin.value())
        self._options.filtered_gain = float(self.filtered_gain_spin.value())
        self._options.show_csd = self.csd_checkbox.isChecked()
        self._options.csd_low = float(self.csd_low_spin.value())
        self._options.csd_high = float(self.csd_high_spin.value())
        self._options.csd_distance = float(self.csd_distance_spin.value())
        self._options.csd_gain = float(self.csd_gain_spin.value())
        self._update_enabled_states()
        self.optionsChanged.emit(self.channel, self._options)

    def set_csd_status(self, message: str, level: str = "info"):
        """Called by TraceViewWidget to report CSD availability for this
        channel. level is 'info', 'warning', or 'ok'."""
        colors = {"info": "#888", "warning": "#e57373", "ok": "#4caf50"}
        self.csd_status_label.setText(message)
        self.csd_status_label.setStyleSheet(
            f"color: {colors.get(level, '#888')}; font-size: 10px;"
        )


    def showEvent(self, event):
        """Install the app-level filter that closes this panel when the
        user clicks outside it. This replaces the click-outside behavior
        that Qt.WindowType.Popup gave us for free, which we lost by
        switching to a normal movable window so it can be dragged."""
        super().showEvent(event)
        from PyQt6.QtWidgets import QApplication
        QApplication.instance().installEventFilter(self)

    def hideEvent(self, event):
        """Remove the filter when hidden so we don't leak it across
        open/close cycles, and commit the panel's current state back to
        whoever owns it -- so nothing is lost if any individual edit
        path failed to emit optionsChanged (e.g. a value set
        programmatically during construction, or a future widget added
        without wiring up its signal)."""
        super().hideEvent(event)
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        # Re-read every widget into _options, then emit. This is the
        # same work _on_any_changed does; calling it here is cheap and
        # makes close a guaranteed save point.
        self._on_any_changed()
        self.closed.emit()

    def eventFilter(self, obj, event):
        """Close this panel when the user clicks a mouse button outside
        its geometry. The panel itself and its children are excluded so
        interacting with the panel's own widgets doesn't close it."""
        from PyQt6.QtCore import QEvent
        from PyQt6.QtWidgets import QApplication
        if event.type() == QEvent.Type.MouseButtonPress:
            if self.isVisible():
                try:
                    global_pos = event.globalPosition().toPoint()
                except AttributeError:
                    global_pos = event.globalPos()
                # If the click is outside this panel's frame, close.
                if not self.frameGeometry().contains(global_pos):
                    # Do not swallow the event -- let it propagate to
                    # whatever widget the user actually clicked on.
                    self.close()
        return super().eventFilter(obj, event)
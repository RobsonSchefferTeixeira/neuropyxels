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
    """Per-channel display + filter options."""
    show_raw: bool = True
    show_filtered: bool = False
    filter_low: float = 1.0
    filter_high: float = 300.0

    show_csd: bool = False
    csd_low: float = 0.0
    csd_high: float = 0.0
    csd_distance: float = 30.0

    def is_default(self) -> bool:
        """True if this is exactly the default state (raw on, everything
        else off). Used by the Reset All button to skip channels that
        have nothing to reset."""
        return (
            self.show_raw
            and not self.show_filtered
            and not self.show_csd
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

        self._options = ChannelOptions(
            show_raw=options.show_raw,
            show_filtered=options.show_filtered,
            filter_low=options.filter_low,
            filter_high=options.filter_high,
            show_csd=options.show_csd,
            csd_low=options.csd_low,
            csd_high=options.csd_high,
            csd_distance=options.csd_distance,
        )

        self._build_ui()
        self._sync_from_options()

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
        raw_layout = QVBoxLayout(raw_group)
        self.raw_checkbox = QCheckBox("Show Raw")
        self.raw_checkbox.toggled.connect(self._on_any_changed)
        raw_layout.addWidget(self.raw_checkbox)
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
        filt_layout.addWidget(self.filter_low_spin, 1, 1)
        filt_layout.addWidget(QLabel("High (Hz):"), 2, 0)
        self.filter_high_spin = QDoubleSpinBox()
        self.filter_high_spin.setRange(0.0, 15000.0)
        self.filter_high_spin.setSingleStep(1.0)
        self.filter_high_spin.valueChanged.connect(self._on_any_changed)
        filt_layout.addWidget(self.filter_high_spin, 2, 1)
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
        csd_layout.addWidget(self.csd_low_spin, 1, 1)
        csd_layout.addWidget(QLabel("Band high (Hz):"), 2, 0)
        self.csd_high_spin = QDoubleSpinBox()
        self.csd_high_spin.setRange(0.0, 15000.0)
        self.csd_high_spin.setSingleStep(1.0)
        self.csd_high_spin.setToolTip("0 = compute CSD directly from raw")
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

        self.csd_status_label = QLabel("")
        self.csd_status_label.setStyleSheet("color: #888; font-size: 10px;")
        self.csd_status_label.setWordWrap(True)
        csd_layout.addWidget(self.csd_status_label, 4, 0, 1, 2)

        layout.addWidget(csd_group)

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------

    def _sync_from_options(self):
        o = self._options
        self.raw_checkbox.blockSignals(True)
        self.filtered_checkbox.blockSignals(True)
        self.csd_checkbox.blockSignals(True)
        self.filter_low_spin.blockSignals(True)
        self.filter_high_spin.blockSignals(True)
        self.csd_low_spin.blockSignals(True)
        self.csd_high_spin.blockSignals(True)
        self.csd_distance_spin.blockSignals(True)

        self.raw_checkbox.setChecked(o.show_raw)
        self.filtered_checkbox.setChecked(o.show_filtered)
        self.csd_checkbox.setChecked(o.show_csd)
        self.filter_low_spin.setValue(o.filter_low)
        self.filter_high_spin.setValue(o.filter_high)
        self.csd_low_spin.setValue(o.csd_low)
        self.csd_high_spin.setValue(o.csd_high)
        self.csd_distance_spin.setValue(o.csd_distance)

        self.raw_checkbox.blockSignals(False)
        self.filtered_checkbox.blockSignals(False)
        self.csd_checkbox.blockSignals(False)
        self.filter_low_spin.blockSignals(False)
        self.filter_high_spin.blockSignals(False)
        self.csd_low_spin.blockSignals(False)
        self.csd_high_spin.blockSignals(False)
        self.csd_distance_spin.blockSignals(False)

        self._update_enabled_states()

    def _update_enabled_states(self):
        """Grey out parameters whose parent checkbox is off, so the
        panel reflects what's actually active at a glance."""
        filt_on = self.filtered_checkbox.isChecked()
        self.filter_low_spin.setEnabled(filt_on)
        self.filter_high_spin.setEnabled(filt_on)
        csd_on = self.csd_checkbox.isChecked()
        self.csd_low_spin.setEnabled(csd_on)
        self.csd_high_spin.setEnabled(csd_on)
        self.csd_distance_spin.setEnabled(csd_on)

    # ------------------------------------------------------------------
    # Change handling
    # ------------------------------------------------------------------

    def _on_any_changed(self, *args):
        self._options.show_raw = self.raw_checkbox.isChecked()
        self._options.show_filtered = self.filtered_checkbox.isChecked()
        self._options.filter_low = float(self.filter_low_spin.value())
        self._options.filter_high = float(self.filter_high_spin.value())
        self._options.show_csd = self.csd_checkbox.isChecked()
        self._options.csd_low = float(self.csd_low_spin.value())
        self._options.csd_high = float(self.csd_high_spin.value())
        self._options.csd_distance = float(self.csd_distance_spin.value())
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
        open/close cycles."""
        super().hideEvent(event)
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
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
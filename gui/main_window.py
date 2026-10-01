"""
main_window.py

Top-level QMainWindow for data_explorer. Owns the Qt application shell:
menu bar, settings.xml + continuous.dat loading, probe-stream selection,
and hosts both the ProbeMapWidget (docked, left) and TraceViewWidget
(central). Selecting channels on the probe map drives what's shown in
the trace view.

This file replaces the tkinter file-dialog / matplotlib-widget scaffolding
that used to live in neuropixels_viewer.py's create_gui() etc. That file's
data-handling logic now lives in core/trace_engine.py (TraceEngine); its
GUI half is fully superseded by this window + gui/trace_view.py.

Usage
-----
    python main.py
    python main.py path/to/settings.xml
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QFileDialog, QMessageBox, QStatusBar, QDockWidget,
    QListWidget, QListWidgetItem, QPushButton, QColorDialog, QFrame,
    QSlider, QSpinBox, QDoubleSpinBox, QGroupBox, QCheckBox, QDialog,
)
from PyQt6.QtGui import QAction, QKeySequence, QColor, QIcon, QPixmap
from gui.probe_map_widget import ProbeMapWidget
from gui.neural_trace_view import NeuralTraceViewWidget as TraceViewWidget
from gui.phase_amplitude_dialog import PhaseAmplitudeDialog
from gui.amplitude_power_dialog import AmplitudePowerDialog
from gui.ripple_dialog import RippleDialog
from core.probe_extractor import extract_probes_from_settings
from core.trace_engine import TraceEngine
from gui.theta_epoch_dialog import ThetaEpochDialog
from core.probe_definition import ProbeDefinition
from gui.probe_definition_dialog import ProbeDefinitionDialog
from gui.psd_dialog import PsdDialog
from gui.phy_units_panel import PhyUnitsPanel
from core.phy_loader import load_phy_folder, PhyLoadError

class ColorButton(QPushButton):

    """
    A push button that displays a color swatch and opens a color dialog
    when clicked. The button shows the actual color, not just a name.
    """

    colorChanged = pyqtSignal(QColor)

    def __init__(self, color: str = "#3498db", parent: QWidget | None = None,
                 label: str = ""):
        super().__init__(parent)
        self._color = QColor(color)
        self._label = label
        self.setFixedSize(120, 28)
        self.clicked.connect(self._on_clicked)
        self._update_button()

    def _update_button(self):
        """Update the button appearance to show the selected color."""
        pixmap = QPixmap(24, 24)
        pixmap.fill(self._color)

        icon = QIcon(pixmap)
        self.setIcon(icon)
        self.setIconSize(pixmap.size())

        color_name = self._color.name()
        if self._label:
            self.setText(f"{self._label} {color_name}")
        else:
            self.setText(color_name)

        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {self._color.name()};
                color: {'white' if self._color.lightness() < 128 else 'black'};
                border: 2px solid {self._color.darker(150).name()};
                border-radius: 4px;
                padding: 2px 8px;
                font-weight: bold;
                text-align: left;
            }}
            QPushButton:hover {{
                border: 2px solid {self._color.lighter(150).name()};
            }}
            QPushButton:pressed {{
                background-color: {self._color.darker(110).name()};
            }}
        """)

    def _on_clicked(self):
        """Open a color dialog when clicked."""
        color = QColorDialog.getColor(
            self._color, self, "Select Color",
            QColorDialog.ColorDialogOption.ShowAlphaChannel
        )
        if color.isValid():
            self.set_color(color)

    def set_color(self, color: QColor):
        """Set the button's color and emit the signal."""
        if color != self._color:
            self._color = color
            self._update_button()
            self.colorChanged.emit(color)

    def color(self) -> QColor:
        """Get the current color."""
        return self._color


class TraceStylePanel(QWidget):
    """
    A panel for customizing trace and background colors.
    Shows color buttons that display actual color swatches.
    Supports per-channel color customization.
    """

    def __init__(self, trace_view: TraceViewWidget, parent: QWidget | None = None):
        super().__init__(parent)
        self.trace_view = trace_view

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        title_label = QLabel("Display Settings")
        title_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(title_label)

        # Global colors
        global_group = QGroupBox("Global Colors")
        global_layout = QVBoxLayout(global_group)

        self.trace_color_btn = ColorButton(color="#3498db", label="Trace:")
        self.trace_color_btn.colorChanged.connect(self.on_trace_color_changed)
        global_layout.addWidget(self.trace_color_btn)

        self.bg_color_btn = ColorButton(color="#1e1e1e", label="Background:")
        self.bg_color_btn.colorChanged.connect(self.on_bg_color_changed)
        global_layout.addWidget(self.bg_color_btn)

        self.grid_color_btn = ColorButton(color="#555555", label="Grid:")
        self.grid_color_btn.colorChanged.connect(self.on_grid_color_changed)
        global_layout.addWidget(self.grid_color_btn)

        layout.addWidget(global_group)

        # Line settings
        line_group = QGroupBox("Line Settings")
        line_layout = QVBoxLayout(line_group)

        width_layout = QHBoxLayout()
        width_label = QLabel("Width:")
        self.line_width_spin = QDoubleSpinBox()
        self.line_width_spin.setRange(0.5, 5.0)
        self.line_width_spin.setSingleStep(0.1)
        self.line_width_spin.setValue(1.0)
        self.line_width_spin.setSuffix(" px")
        self.line_width_spin.valueChanged.connect(self.on_line_width_changed)
        width_layout.addWidget(width_label)
        width_layout.addWidget(self.line_width_spin)
        width_layout.addStretch()
        line_layout.addLayout(width_layout)

        self.grid_checkbox = QCheckBox("Show Grid")
        self.grid_checkbox.setChecked(True)
        self.grid_checkbox.toggled.connect(self.on_grid_toggled)
        line_layout.addWidget(self.grid_checkbox)

        layout.addWidget(line_group)

        # Per-channel colors
        channel_group = QGroupBox("Per-Channel Colors")
        channel_layout = QVBoxLayout(channel_group)

        channel_select_layout = QHBoxLayout()
        channel_select_layout.addWidget(QLabel("Channel:"))
        self.channel_combo = QComboBox()
        self.channel_combo.setMinimumWidth(100)
        self.channel_combo.currentIndexChanged.connect(self._on_channel_selected)
        channel_select_layout.addWidget(self.channel_combo)
        channel_select_layout.addStretch()
        channel_layout.addLayout(channel_select_layout)

        self.channel_color_btn = ColorButton(color="#3498db", label="Color:")
        self.channel_color_btn.colorChanged.connect(self.on_channel_color_changed)
        channel_layout.addWidget(self.channel_color_btn)

        self.reset_colors_btn = QPushButton("Reset All Colors")
        self.reset_colors_btn.clicked.connect(self.on_reset_colors)
        channel_layout.addWidget(self.reset_colors_btn)

        layout.addWidget(channel_group)

        # Animation
        anim_group = QGroupBox("Animation")
        anim_layout = QVBoxLayout(anim_group)

        speed_layout = QHBoxLayout()
        speed_label = QLabel("Speed:")
        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(10, 1000)
        self.speed_spin.setValue(100)
        self.speed_spin.setSuffix(" ms/frame")
        self.speed_spin.valueChanged.connect(self.on_speed_changed)
        speed_layout.addWidget(speed_label)
        speed_layout.addWidget(self.speed_spin)
        speed_layout.addStretch()
        anim_layout.addLayout(speed_layout)

        layout.addWidget(anim_group)

        layout.addStretch()

        self._sync_from_view()

    def _sync_from_view(self):
        """Sync the controls with the current trace view settings."""
        if self.trace_view:
            trace_color = self.trace_view.default_trace_color
            self.trace_color_btn.set_color(trace_color)

            bg_color = self.trace_view.background_color
            self.bg_color_btn.set_color(bg_color)

            grid_color = self.trace_view.grid_color
            self.grid_color_btn.set_color(grid_color)

            self.line_width_spin.setValue(self.trace_view.trace_width)
            self.grid_checkbox.setChecked(self.trace_view.show_grid)
            self.speed_spin.setValue(self.trace_view.animation_speed)

            self._update_channel_combo()

    def _update_channel_combo(self):
        """Update the channel combo box with currently selected channels."""
        self.channel_combo.clear()
        if self.trace_view and self.trace_view.channels:
            for ch in self.trace_view.channels:
                self.channel_combo.addItem(f"CH{ch}", ch)
            self.channel_combo.setEnabled(False)
        else:
            self.channel_combo.addItem("No channels", None)
            self.channel_combo.setEnabled(False)

    def _on_channel_selected(self, index: int):
        """Handle channel selection in the combo box."""
        if index >= 0:
            channel = self.channel_combo.itemData(index)
            if channel is not None and self.trace_view:
                color = self.trace_view.channel_colors.get(
                    channel, self.trace_view.default_trace_color
                )
                self.channel_color_btn.set_color(color)

    def on_trace_color_changed(self, color: QColor):
        if self.trace_view:
            self.trace_view.set_trace_color(color)

    def on_bg_color_changed(self, color: QColor):
        if self.trace_view:
            self.trace_view.set_background_color(color)

    def on_grid_color_changed(self, color: QColor):
        if self.trace_view:
            self.trace_view.set_grid_color(color)

    def on_grid_toggled(self, checked: bool):
        if self.trace_view:
            self.trace_view.set_grid_visible(checked)

    def on_line_width_changed(self, value: float):
        if self.trace_view:
            self.trace_view.set_trace_width(value)

    def on_channel_color_changed(self, color: QColor):
        if self.trace_view and self.channel_combo.currentIndex() >= 0:
            channel = self.channel_combo.currentData()
            if channel is not None:
                self.trace_view.set_channel_color(channel, color)

    def on_reset_colors(self):
        if self.trace_view:
            self.trace_view.reset_channel_colors()
            self.trace_view.set_trace_color("#3498db")
            self.trace_view.set_background_color("#1e1e1e")
            self.trace_view.set_grid_color("#555555")
            self._sync_from_view()

    def on_speed_changed(self, value: int):
        if self.trace_view:
            self.trace_view.set_animation_speed(value)

    def update_channels(self):
        """Public method to update channel list when selection changes."""
        self._update_channel_combo()


class MainWindow(QMainWindow):
    """
    Application shell. Current responsibilities:
      - Load a settings.xml via menu or CLI arg; pick a probe stream if
        more than one is present.
      - Load a continuous.dat and host it in a TraceViewWidget (central).
      - Host the ProbeMapWidget for the selected stream in a left dock;
        its channel selection drives the trace view.
      - Show selected channels in a dockable list + status bar.
      - Provide color customization for trace and background.
      - Provide dockable trace control panel.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("data_explorer")
        self.resize(1600, 950)

        self._settings_path: Path | None = None
        self._probes: dict = {}
        self._current_probe_key: str | None = None
        self.probe_map: ProbeMapWidget | None = None

        self.engine = TraceEngine()
        self.trace_view: TraceViewWidget | None = None
        self._open_pac_dialogs: list[PhaseAmplitudeDialog] = []
        self._open_power_dialogs: list[AmplitudePowerDialog] = []
        self._open_ripple_dialogs: list[RippleDialog] = []
        self._open_theta_dialogs: list[ThetaEpochDialog] = []
        self._open_psd_dialogs: list[PsdDialog] = []
        self.phy_units_panel: PhyUnitsPanel | None = None

        self._build_menu()
        self._build_trace_view()
        self._build_probe_map_dock()
        self._build_selected_channels_dock()
        self._build_display_settings_dock()
        self._build_phy_units_dock()
        self._build_panels_menu()
        self.setStatusBar(QStatusBar())
        self._update_status("No settings file or data loaded.")


    def _on_reset(self):
        """Reset the entire application to its just-launched state.
        The MainWindow object and its menu / toolbar / dock layout are
        preserved; every data-bound widget's *contents*, every loaded
        file, and every customization are removed. The user can then
        load a fresh settings.xml + continuous.dat as if the app had
        just been started."""
        reply = QMessageBox.question(
            self, "Reset application",
            "This will close all dialogs and unload the current probe, "
            "data, Phy folder, and every customization.\n\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # ---- 1. Close every open analysis dialog ----
        for dlg_list_name in (
            "_open_pac_dialogs",
            "_open_power_dialogs",
            "_open_ripple_dialogs",
            "_open_theta_dialogs",
            "_open_psd_dialogs",
        ):
            dlg_list = getattr(self, dlg_list_name, None)
            if dlg_list:
                for dialog in list(dlg_list):
                    try:
                        dialog.close()
                    except Exception:
                        pass
                dlg_list.clear()

        # ---- 2. Reset the trace view to a blank state ----
        self._reset_trace_view()

        # ---- 3. Tear down the probe map and its dock contents ----
        self._reset_probe_map()

        # ---- 4. Reset the Phy Units panel ----
        if self.phy_units_panel is not None:
            try:
                # Rebuild the panel from scratch so nothing survives --
                # this drops the table rows, the phy_data reference, and
                # every selection / filter / focus-mode state.
                self.phy_units_panel.set_phy_data(None)
                self.phy_units_panel.set_focus_mode(False)
                self.phy_units_panel.clear_selection()
                self.phy_units_panel.search_edit.clear()
                self.phy_units_panel.quality_combo.setCurrentIndex(0)
                self.phy_units_panel.show_only_selected_check.setChecked(False)
                self.phy_units_panel.raster_check.setChecked(True)
                self.phy_units_panel.recolor_check.setChecked(False)
                self.phy_units_panel.recolor_window_spin.setValue(1.0)
                self.phy_units_panel.focus_n_spin.setValue(5)
            except Exception:
                pass

        # ---- 5. Replace the engine with a fresh, empty one ----
        # The old engine's memmap is closed when the old instance is
        # garbage-collected; we drop our only reference here.
        self.engine = TraceEngine()
        if self.trace_view is not None:
            self.trace_view.engine = self.engine
            self.trace_view._data_loaded = False

        # ---- 6. Clear MainWindow's file/probe bookkeeping ----
        self._settings_path = None
        self._probes = {}
        self._current_probe_key = None
        # ---- 7. Reset the stream picker ----
        self.stream_picker.blockSignals(True)
        self.stream_picker.clear()
        self.stream_picker.setEnabled(False)
        self.stream_picker.blockSignals(False)

        # ---- 8. Re-run the enabled-state sweep ----
        self._update_analysis_actions_enabled()

        # ---- 9. Reset the trace view's sample rate to the default ----
        # so the filter spinboxes go back to their just-launched ranges.
        if self.trace_view is not None:
            try:
                self.trace_view.control_panel.set_sample_rate(self.engine.sr)
            except Exception:
                pass

        self._update_status("Application reset. No settings file or data loaded.")


    def _reset_trace_view(self):
        tv = self.trace_view
        if tv is None:
            return

        # Channels and geometry.
        tv.channels = []
        tv._sorted_channels = []
        tv.channel_depths = {}
        tv.channel_shanks = {}
        tv.channel_xcoords = {}
        tv.full_probe_depths = {}
        tv.full_probe_shanks = {}
        tv.full_probe_xcoords = {}
        tv._probe_data = None
        tv._csd_analyzer = None

        # Color customizations.
        tv.channel_colors = {}
        tv.default_trace_color = QColor("#ffffff")
        tv.background_color = QColor("#1e1e1e")
        tv.grid_color = QColor("#555555")
        tv.show_grid = True
        tv.trace_width = 1.0

        # Gain and scaling.
        tv.global_gain = 1.0
        tv.auto_scale = False
        tv.control_panel.set_gain(1.0)
        tv.control_panel.set_auto_scale(False)

        # Global filter state + panel.
        tv.filter_enabled = False
        tv.filter_low_freq = 1.0
        tv.filter_high_freq = 300.0
        tv.notch_enabled = False
        tv.notch_freq = 50.0
        tv.detrend_enabled = False
        tv._global_filter_enabled = False
        tv.control_panel.global_filter_checkbox.blockSignals(True)
        tv.control_panel.global_filter_checkbox.setChecked(False)
        tv.control_panel.global_filter_checkbox.blockSignals(False)
        tv.control_panel.filter_checkbox.setChecked(False)
        tv.control_panel.notch_checkbox.setChecked(False)
        tv.control_panel.detrend_checkbox.setChecked(False)

        # Per-channel customizations.
        tv._channel_options = {}
        if tv._channel_options_panel is not None:
            try:
                tv._channel_options_panel.close()
            except Exception:
                pass
            tv._channel_options_panel = None
            tv._channel_options_panel_channel = None
        tv.channel_display_modes = {}

        # Remove any channel buttons on screen.
        for ch, btn in list(tv._channel_buttons.items()):
            try:
                btn.setParent(None)
                btn.deleteLater()
            except Exception:
                pass
        tv._channel_buttons = {}

        # Phy raster and recolor.
        tv._raster_unit_ids = []
        tv._raster_unit_colors = {}
        tv._raster_overflow_warned = False
        tv.show_spike_raster = True
        tv.show_spike_recolor = False
        tv._spike_recolor_overflow_warned = False

        # Ripple overlay.
        tv.ripple_events = []
        tv._ripple_render_context = {}
        tv._ripple_selected_event = None
        tv._ripple_drag_event = None
        tv._ripple_merge_candidates = set()
        tv._ripple_undo_stack = []
        tv._ripple_snap_active_side = None
        tv._ripple_creating_armed = False
        tv._ripple_creating_channel = None
        tv._ripple_creating_start_sample = None
        tv._ripple_creating_end_sample = None

        # Theta overlay.
        tv.theta_epochs = []
        tv.theta_detection_start_time = 0.0
        tv.theta_detection_sample_offset = 0
        tv._theta_drag_epoch = None
        tv._theta_drag_side = None
        tv._theta_merge_candidate = None
        tv._theta_merge_candidates = set()
        tv._theta_selected_epoch = None
        tv._theta_creating_armed = False
        tv._theta_creating_channel = None
        tv._theta_creating_start_sample = None
        tv._theta_creating_end_sample = None

        # Spectrogram.
        tv.show_spectrogram = False
        tv.spectrogram_channel = None
        tv.spectrogram_data = None
        tv.spectrogram_freqs = None
        tv.spectrogram_times = None
        tv._spectrogram_display_data = None
        tv._spectrogram_display_freqs = None
        tv._spectrogram_image = None
        tv._spectrogram_cache_key = None
        if hasattr(tv, "spectrogram_control"):
            try:
                tv.spectrogram_control.set_enabled(False)
                tv.spectrogram_control.set_channels([])
            except Exception:
                pass

        # Time cursors.
        tv._time_cursors = []
        tv._selected_cursor = None

        # Caches.
        tv._cached_data = None
        tv._cache_start = None
        tv._cache_end = None
        tv._cache_start_idx = None
        tv._cache_end_idx = None
        tv._filtered_cache = {}
        tv._trace_path_cache = {}
        tv._trace_cache_key = None
        tv._trace_normalization_cache = {}
        tv._trace_normalization_key = None
        tv._global_scale_cache = None
        tv._global_scale_key = None
        # If the fixed-scale reference attributes don't exist on your
        # version yet, drop the two lines above -- they're harmless
        # either way when guarded with hasattr elsewhere.

        tv.update()

    def _reset_probe_map(self):
        """Destroy the probe map completely and rebuild the dock's
        placeholder from scratch.

        Two subtleties:
          - The previous placeholder widget was destroyed by Qt when the
            probe map replaced it, so it cannot be reused. We build a
            fresh one each reset.
          - deleteLater() alone does not guarantee the old probe map is
            gone before the dock shows the new widget, and if any other
            reference exists the widget survives. setParent(None) forces
            it out of the hierarchy before scheduling deletion.
        """
        old_map = self.probe_map
        self.probe_map = None

        # Build a fresh placeholder.
        placeholder = QWidget()
        vlayout = QVBoxLayout(placeholder)
        label = QLabel(
            "No probe loaded.\n\nUse File \u2192 Open settings.xml... to load one."
        )
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("color: #888; font-size: 12px;")
        vlayout.addWidget(label)

        # Replace the dock's widget (this orphans the old probe map).
        self.probe_map_dock.setWidget(placeholder)

        # Explicitly detach and destroy the old probe map.
        if old_map is not None:
            try:
                old_map.channelsSelected.disconnect(self._on_channels_selected)
            except (TypeError, RuntimeError):
                pass
            try:
                old_map.setParent(None)
            except Exception:
                pass
            try:
                old_map.deleteLater()
            except Exception:
                pass

        self.selected_list.clear()
        self.selected_channels_dock.setWindowTitle("Selected Channels")

    def _build_phy_units_dock(self):
        """Dock that lists sorted units from a loaded Phy folder and
        drives the trace view's spike raster overlay."""
        dock = QDockWidget("Phy Units", self)
        dock.setObjectName("phy_units_dock")
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)

        self.phy_units_panel = PhyUnitsPanel()
        self.phy_units_panel.selectionChanged.connect(self._on_phy_units_selection_changed)

        self.phy_units_panel.rasterVisibleChanged.connect(self._on_raster_visible_changed)
        self.phy_units_panel.recolorVisibleChanged.connect(self._on_recolor_visible_changed)
        self.phy_units_panel.recolorWindowChanged.connect(self._on_recolor_window_changed)

        self.phy_units_panel.focusModeChanged.connect(self._on_focus_mode_changed)
        self.phy_units_panel.focusNeighborhoodChanged.connect(self._on_focus_neighborhood_size_changed)
        self.phy_units_panel.focusedUnitChanged.connect(self._on_focused_unit_changed)

        self.phy_units_panel.reclassifyRequested.connect(self._on_reclassify_requested)
        self.phy_units_panel.resetClassificationRequested.connect(self._on_reset_classification_requested)
        self.phy_units_panel.importFromPhyRequested.connect(self._on_import_from_phy_requested)
        self.phy_units_panel.saveRequested.connect(self._on_save_classification_requested)

        self.phy_units_panel.prevSpikeRequested.connect(self._on_prev_spike_requested)
        self.phy_units_panel.nextSpikeRequested.connect(self._on_next_spike_requested)

        dock.setWidget(self.phy_units_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.phy_units_dock = dock

        # Not shown at startup. It is opened by loading a Phy folder,
        # or manually via the Panels menu toggle.
        dock.setVisible(False)
        dock.visibilityChanged.connect(self._on_phy_units_visibility_changed)


    def _on_reclassify_requested(self, cluster_ids: list, new_class: str):
        if self.phy_units_panel is None or not cluster_ids:
            return
        self.phy_units_panel.apply_reclassification(cluster_ids, new_class)
        self._schedule_classification_autosave()

    def _on_reset_classification_requested(self, cluster_ids: list):
        if self.phy_units_panel is None or not cluster_ids:
            return
        self.phy_units_panel.reset_classification(cluster_ids)
        self._schedule_classification_autosave()

    def _on_import_from_phy_requested(self, cluster_ids: list):
        if self.phy_units_panel is None or not cluster_ids:
            return
        self.phy_units_panel.import_from_phy(cluster_ids)
        self._schedule_classification_autosave()

    def _on_save_classification_requested(self):
        if self.engine.phy_data is None:
            return
        from core.phy_loader import save_reclassified
        try:
            path = save_reclassified(
                self.engine.phy_data, self.engine.phy_data.units
            )
        except Exception as exc:
            QMessageBox.critical(
                self, "Save failed",
                f"Could not write classification file:\n\n{exc}"
            )
            return
        self._update_status(f"Classification saved to {path.name}.")

    def _schedule_classification_autosave(self):
        """Debounce autosaves: a burst of reclassifications results in
        one file write ~600 ms after the last change, not one per click."""
        if not hasattr(self, "_classification_autosave_timer"):
            from PyQt6.QtCore import QTimer
            self._classification_autosave_timer = QTimer(self)
            self._classification_autosave_timer.setSingleShot(True)
            self._classification_autosave_timer.setInterval(600)
            self._classification_autosave_timer.timeout.connect(
                self._on_save_classification_requested
            )
        self._classification_autosave_timer.start()

    def _on_prev_spike_requested(self):
        self._jump_to_adjacent_spike(direction=-1)

    def _on_next_spike_requested(self):
        self._jump_to_adjacent_spike(direction=+1)

    def _jump_to_adjacent_spike(self, direction: int):
        """Move the trace view to the previous or next spike of the
        currently-targeted unit.

        Target selection, in order of preference:
          1. If focus mode is on and a unit is focused, use that unit.
          2. Otherwise, if exactly one unit is selected in the table,
             use it.
          3. Otherwise, print a status message and do nothing.
        """
        phy_data = getattr(self.engine, "phy_data", None)
        if phy_data is None:
            self._update_status(
                "Prev/Next spike: no Phy folder loaded."
            )
            return

        unit_id = -1
        if (self.phy_units_panel is not None
                and self.phy_units_panel.focus_mode_enabled()):
            unit_id = self.phy_units_panel.focused_unit_id()
        if unit_id < 0 and self.phy_units_panel is not None:
            selected = self.phy_units_panel.selected_cluster_ids()
            if len(selected) == 1:
                unit_id = int(selected[0])

        if unit_id < 0:
            self._update_status(
                "Prev/Next spike: select a single unit (or enable focus "
                "mode and pick one) to navigate its spikes."
            )
            return

        spikes = phy_data.spikes_for_unit(unit_id)
        if spikes.size == 0:
            self._update_status(
                f"Prev/Next spike: unit {unit_id} has no spikes."
            )
            return

        sr = float(self.engine.sr)
        spike_times = spikes.astype(np.float64) / sr

        # Current center of the visible window, in seconds.
        center_time = (
            self.trace_view.start_time
            + self.trace_view.window_duration / 2.0
        )

        if direction > 0:
            candidates = spike_times[spike_times > center_time + 1e-9]
            if candidates.size == 0:
                self._update_status(
                    f"Prev/Next spike: no later spikes for unit {unit_id}."
                )
                return
            target = float(candidates[0])
        else:
            candidates = spike_times[spike_times < center_time - 1e-9]
            if candidates.size == 0:
                self._update_status(
                    f"Prev/Next spike: no earlier spikes for unit {unit_id}."
                )
                return
            target = float(candidates[-1])

        # Center the view on the target spike.
        new_start = target - self.trace_view.window_duration / 2.0
        min_start = self.trace_view._get_min_start_time()
        max_start = self.trace_view._get_max_start_time()
        new_start = max(min_start, min(new_start, max_start))

        self.trace_view.start_time = new_start
        self.trace_view._update_time_labels()
        self.trace_view._update_scrollbar_position()
        self.trace_view._invalidate_cache()
        self.trace_view.update()

        self._update_status(
            f"Unit {unit_id}: jumped to spike at {target:.4f} s "
            f"({'(next)' if direction > 0 else '(prev)'})"
        )

    def _on_focus_mode_changed(self, enabled: bool):
        """Entering focus mode clears the current display so the user
        sees an empty trace view and an empty probe map, then waits
        for a unit to be selected. Leaving focus mode does nothing
        automatic -- the current selection stays in place, and
        multi-select is re-enabled in the panel."""
        if enabled:
            # Clear trace view + probe map.
            if self.trace_view is not None:
                self.trace_view.set_raster_units([])
                self.trace_view.set_channels([])
            if self.probe_map is not None:
                self.probe_map.set_selected_channels([])
            if hasattr(self, "selected_list"):
                self.selected_list.clear()
            self._update_status(
                "Focus mode: select one unit to open its local "
                "neighborhood."
            )

    def _on_focus_neighborhood_size_changed(self, _n: int):
        """If a unit is already focused, recompute its neighborhood
        with the new size."""
        if not self.phy_units_panel.focus_mode_enabled():
            return
        unit_id = self.phy_units_panel.focused_unit_id()
        if unit_id >= 0:
            self._apply_focus_neighborhood(unit_id)

    def _on_focused_unit_changed(self, unit_id: int):
        """Apply or clear focus for the given unit id. -1 means no
        unit is selected (either deselected or focus mode is off)."""
        if unit_id < 0:
            if self.phy_units_panel.focus_mode_enabled():
                # User deselected inside focus mode -- clear the view
                # so it's obvious nothing is focused right now.
                if self.trace_view is not None:
                    self.trace_view.set_raster_units([])
                    self.trace_view.set_channels([])
                if self.probe_map is not None:
                    self.probe_map.set_selected_channels([])
                if hasattr(self, "selected_list"):
                    self.selected_list.clear()
            return
        if not self.phy_units_panel.focus_mode_enabled():
            return
        self._apply_focus_neighborhood(unit_id)

    def _apply_focus_neighborhood(self, unit_id: int):
        phy_data = getattr(self.engine, "phy_data", None)
        if phy_data is None:
            return
        unit = phy_data.units.get(int(unit_id))
        if unit is None or unit.channel is None:
            self._update_status(
                f"Focus mode: unit {unit_id} has no assigned channel."
            )
            return

        center_ch = int(unit.channel)
        n_levels = self.phy_units_panel.focus_neighborhood_size()
        neighborhood = self.trace_view.compute_focus_neighborhood(
            center_ch, n_levels
        )
        if not neighborhood:
            self._update_status(
                f"Focus mode: could not compute neighborhood for "
                f"unit {unit_id} (CH{center_ch}) -- is the probe "
                f"geometry loaded?"
            )
            return

        self.trace_view.set_channels(neighborhood)
        self.trace_view.set_raster_units([int(unit_id)])

        if self.probe_map is not None:
            self.probe_map.set_selected_channels(neighborhood)

        if hasattr(self, "selected_list"):
            from PyQt6.QtWidgets import QListWidgetItem
            self.selected_list.clear()
            for ch in neighborhood:
                self.selected_list.addItem(QListWidgetItem(f"CH{ch}"))
            self.selected_channels_dock.setWindowTitle(
                f"Selected Channels ({len(neighborhood)})"
            )

        self._update_status(
            f"Focus mode: unit {unit_id} (CH{center_ch}) — "
            f"{len(neighborhood)} channel(s) in neighborhood."
        )




    def _on_focus_mode_changed(self, enabled: bool):
        """Entering focus mode clears the current display so the user
        sees an empty trace view and an empty probe map, then waits
        for a unit to be selected. Leaving focus mode does nothing
        automatic -- the current selection stays in place, and
        multi-select is re-enabled in the panel."""
        if enabled:
            # Clear trace view + probe map.
            if self.trace_view is not None:
                self.trace_view.set_raster_units([])
                self.trace_view.set_channels([])
            if self.probe_map is not None:
                self.probe_map.set_selected_channels([])
            if hasattr(self, "selected_list"):
                self.selected_list.clear()
            self._update_status(
                "Focus mode: select one unit to open its local "
                "neighborhood."
            )

    def _on_focus_neighborhood_size_changed(self, _n: int):
        """If a unit is already focused, recompute its neighborhood
        with the new size."""
        if not self.phy_units_panel.focus_mode_enabled():
            return
        unit_id = self.phy_units_panel.focused_unit_id()
        if unit_id >= 0:
            self._apply_focus_neighborhood(unit_id)

    def _on_focused_unit_changed(self, unit_id: int):
        """Apply or clear focus for the given unit id. -1 means no
        unit is selected (either deselected or focus mode is off)."""
        if unit_id < 0:
            if self.phy_units_panel.focus_mode_enabled():
                # User deselected inside focus mode -- clear the view
                # so it's obvious nothing is focused right now.
                if self.trace_view is not None:
                    self.trace_view.set_raster_units([])
                    self.trace_view.set_channels([])
                if self.probe_map is not None:
                    self.probe_map.set_selected_channels([])
                if hasattr(self, "selected_list"):
                    self.selected_list.clear()
            return
        if not self.phy_units_panel.focus_mode_enabled():
            return
        self._apply_focus_neighborhood(unit_id)

    def _apply_focus_neighborhood(self, unit_id: int):
        """Look up the unit's channel, compute its neighborhood, and
        push the resulting channel set to both the trace view and the
        probe map. Also set the unit as the sole raster unit."""
        phy_data = getattr(self.engine, "phy_data", None)
        if phy_data is None:
            return
        unit = phy_data.units.get(int(unit_id))
        if unit is None or unit.channel is None:
            self._update_status(
                f"Focus mode: unit {unit_id} has no assigned channel."
            )
            return

        center_ch = int(unit.channel)
        n_levels = self.phy_units_panel.focus_neighborhood_size()
        neighborhood = self.trace_view.compute_focus_neighborhood(
            center_ch, n_levels
        )
        if not neighborhood:
            self._update_status(
                f"Focus mode: could not compute neighborhood for "
                f"unit {unit_id} (CH{center_ch}) -- is the probe "
                f"geometry loaded?"
            )
            return

        # Push to the trace view first, then to the probe map. Both
        # use the same channel list, so selection stays consistent
        # between the two views.
        self.trace_view.set_channels(neighborhood)
        self.trace_view.set_raster_units([int(unit_id)])

        if self.probe_map is not None:
            self.probe_map.set_selected_channels(neighborhood)

        # Mirror the selection in the Selected Channels list.
        if hasattr(self, "selected_list"):
            from PyQt6.QtWidgets import QListWidgetItem
            self.selected_list.clear()
            for ch in neighborhood:
                self.selected_list.addItem(QListWidgetItem(f"CH{ch}"))
            self.selected_channels_dock.setWindowTitle(
                f"Selected Channels ({len(neighborhood)})"
            )

        # Push depth info to the trace view for the new channel set.
        if self._current_probe_key and self._probes:
            probe_data = self._probes["probes"][self._current_probe_key]
            depths = dict(zip(
                probe_data["coordinates"]["channels"],
                probe_data["coordinates"]["y"],
            ))
            self.trace_view.set_channel_depths(depths)
            xcoords = dict(zip(
                probe_data["coordinates"]["channels"],
                probe_data["coordinates"]["x"],
            ))
            self.trace_view.set_channel_xcoords(xcoords)
            shank_ids = (
                probe_data.get("shanks", {}).get("ids")
                or [0] * len(probe_data["coordinates"]["channels"])
            )
            shank_map = dict(zip(
                probe_data["coordinates"]["channels"], shank_ids
            ))
            self.trace_view.set_channel_shanks(shank_map)
            self.trace_view.set_full_probe_geometry(
                depths, shank_map, xcoords
            )

        self._update_status(
            f"Focus mode: unit {unit_id} (CH{center_ch}) — "
            f"{len(neighborhood)} channel(s) in neighborhood."
        )

    def _on_raster_visible_changed(self, visible: bool):
        if self.trace_view is not None:
            self.trace_view.set_spike_raster_visible(visible)

    def _on_recolor_visible_changed(self, visible: bool):
        if self.trace_view is not None:
            self.trace_view.set_spike_recolor_visible(visible)

    def _on_recolor_window_changed(self, window_ms: float):
        if self.trace_view is not None:
            self.trace_view.set_spike_recolor_window_ms(window_ms)

    def _on_phy_units_visibility_changed(self, visible: bool):
        """Keep the Panels menu checkbox in sync when the dock is
        opened or closed via its own X button or title bar."""
        if self.phy_units_dock_action is None:
            return
        self.phy_units_dock_action.blockSignals(True)
        self.phy_units_dock_action.setChecked(visible)
        self.phy_units_dock_action.blockSignals(False)
        
    def _update_analysis_actions_enabled(self):
        ready = self.probe_map is not None and self.engine.data_loaded
        self.phase_amplitude_action.setEnabled(ready)
        self.power_map_action.setEnabled(ready)
        self.ripple_detection_action.setEnabled(ready)
        self.theta_epoch_action.setEnabled(ready)
        self.psd_action.setEnabled(ready)
    # ------------------------------------------------------------------
    # UI scaffolding
    # ------------------------------------------------------------------

    def _build_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")

        open_settings_action = QAction("Open &settings.xml", self)
        open_settings_action.setShortcut(QKeySequence.StandardKey.Open)
        open_settings_action.triggered.connect(self._on_open_settings)
        file_menu.addAction(open_settings_action)

        open_data_action = QAction("Open &Data (continuous.dat)", self)
        open_data_action.triggered.connect(self._on_open_data)
        file_menu.addAction(open_data_action)

        open_definition_action = QAction("New Probe &Definition", self)
        open_definition_action.setToolTip(
            "Build a probe description by hand (or import one), then use "
            "it with a raw continuous.dat that has no settings.xml."
        )
        open_definition_action.triggered.connect(self._on_open_probe_definition)
        file_menu.addAction(open_definition_action)

        open_timestamps_action = QAction("Open Timestamps (timestamps.npy)", self)
        open_timestamps_action.triggered.connect(self._on_open_timestamps)
        file_menu.addAction(open_timestamps_action)


        open_phy_action = QAction("Open &Kilosort/Phy Output", self)
        open_phy_action.setToolTip(
            "Load a Kilosort4 / Phy output folder to overlay sorted "
            "units as a spike raster on the trace view."
        )
        open_phy_action.triggered.connect(self._on_open_phy_folder)
        file_menu.addAction(open_phy_action)



        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)


        file_menu.addSeparator()

        reset_action = QAction("&Reset Application", self)
        reset_action.setToolTip(
            "Close all dialogs and unload the current settings.xml, "
            "continuous.dat, Phy folder, and every customization. "
            "The app returns to its just-launched state."
        )
        reset_action.triggered.connect(self._on_reset)
        file_menu.addAction(reset_action)

        
        analysis_menu = menubar.addMenu("&Analysis")
        self.phase_amplitude_action = QAction("&Phase-Amplitude Coupling", self)
        self.phase_amplitude_action.setEnabled(False)
        self.phase_amplitude_action.triggered.connect(self._on_open_phase_amplitude)
        analysis_menu.addAction(self.phase_amplitude_action)


        self.power_map_action = QAction("Spatial &Power Map", self)
        self.power_map_action.setEnabled(False)
        self.power_map_action.triggered.connect(self._on_open_power_map)
        analysis_menu.addAction(self.power_map_action)


        self.psd_action = QAction("&PSD Analysis", self)
        self.psd_action.setEnabled(False)
        self.psd_action.triggered.connect(self._on_open_psd)
        analysis_menu.addAction(self.psd_action)


        self.ripple_detection_action = QAction("&Ripple Detection", self)
        self.ripple_detection_action.setEnabled(False)
        self.ripple_detection_action.triggered.connect(self._on_open_ripple_detection)
        analysis_menu.addAction(self.ripple_detection_action)

        self.theta_epoch_action = QAction("Theta Epoch Detection", self)
        self.theta_epoch_action.setEnabled(False)
        self.theta_epoch_action.triggered.connect(self._on_open_theta_epoch)
        analysis_menu.addAction(self.theta_epoch_action)

        view_menu = menubar.addMenu("&View")

        self.spectrogram_action = QAction("Show Spectrogram", self)
        self.spectrogram_action.setCheckable(True)
        self.spectrogram_action.toggled.connect(self._on_toggle_spectrogram)
        view_menu.addAction(self.spectrogram_action)

        display_menu = menubar.addMenu("&Display")

        trace_color_action = QAction("Trace Color", self)
        trace_color_action.triggered.connect(self._on_edit_trace_color)
        display_menu.addAction(trace_color_action)

        bg_color_action = QAction("Background Color", self)
        bg_color_action.triggered.connect(self._on_edit_bg_color)
        display_menu.addAction(bg_color_action)

        self.panels_menu = menubar.addMenu("&Panels")

        self.phy_units_dock_action = QAction("Phy Units", self)
        self.phy_units_dock_action.setCheckable(True)
        self.phy_units_dock_action.setChecked(False)
        self.phy_units_dock_action.toggled.connect(self._on_toggle_phy_units_dock)
        self.panels_menu.addAction(self.phy_units_dock_action)

        help_menu = menubar.addMenu("&Help")

        shortcuts_action = QAction("Keyboard Shortcuts", self)
        shortcuts_action.setShortcut(QKeySequence("F1"))
        shortcuts_action.triggered.connect(self._show_shortcuts)
        help_menu.addAction(shortcuts_action)

        about_action = QAction("About data_explorer", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _show_shortcuts(self):
        """Show keyboard shortcuts dialog."""
        shortcuts_text = """
        <h3>Navigation Shortcuts</h3>
        <table border="1" cellpadding="5" cellspacing="0">
            <tr><th>Action</th><th>Shortcut</th><th>Description</th></tr>
            <tr><td><b>Mouse wheel</b></td><td>Scroll</td><td>Scroll through time (10% of window)</td></tr>
            <tr><td><b>Ctrl + wheel</b></td><td>Zoom</td><td>Zoom in/out to cursor</td></tr>
            <tr><td><b>Shift + wheel</b></td><td>Gain</td><td>Adjust signal gain (multiply/divide by 1.15 per notch)</td></tr>
            <tr><td><b>Ctrl + Shift + wheel</b></td><td>Fast scroll</td><td>Scroll 5x faster</td></tr>
            <tr><td><b>Double-click</b></td><td>Reset</td><td>Reset view to start, default duration</td></tr>
            <tr><td><b>+ / -</b></td><td>Zoom</td><td>Zoom in/out</td></tr>
            <tr><td><b>Left / Right</b></td><td>Step</td><td>Move 10% of window</td></tr>
            <tr><td><b>Home / End</b></td><td>Jump</td><td>Go to start / end</td></tr>
            <tr><td><b>0</b></td><td>Reset</td><td>Reset view to defaults</td></tr>
            <tr><td><b>Page Up / Down</b></td><td>Zoom fast</td><td>Zoom 2x / 0.5x</td></tr>
        </table>

        <h3>Filter Shortcuts</h3>
        <table border="1" cellpadding="5" cellspacing="0">
            <tr><th>Filter Setting</th><th>Behavior</th></tr>
            <tr><td>Low=0, High=400</td><td>Low-pass filter (below 400 Hz)</td></tr>
            <tr><td>Low=4, High=0</td><td>High-pass filter (above 4 Hz)</td></tr>
            <tr><td>Low=1, High=300</td><td>Bandpass filter (1-300 Hz)</td></tr>
        </table>

        <h3>Color Controls</h3>
        <p>Use the Display Settings panel to change trace color, background color, grid color, and line width.</p>
        """
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Keyboard Shortcuts")
        msg_box.setTextFormat(Qt.TextFormat.RichText)
        msg_box.setText(shortcuts_text)
        msg_box.setIcon(QMessageBox.Icon.Information)
        msg_box.exec()

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About data_explorer",
            "<h3>data_explorer</h3>"
            "<p>A Neuropixels data visualization and analysis tool.</p>"
            "<p>Features:</p>"
            "<ul>"
            "<li>Interactive probe map</li>"
            "<li>Trace visualization with filtering</li>"
            "<li>Phase-amplitude coupling analysis</li>"
            "<li>Spatial power mapping</li>"
            "<li>Ripple detection</li>"
            "<li>Depth-based sorting</li>"
            "</ul>"
        )

    def _finalize_dock(self, dock: QDockWidget, area: Qt.DockWidgetArea):
        """
        Common tail end for every dock built at startup: register it
        with the main window, default it to FLOATING (a free-standing
        window) rather than docked, and start HIDDEN rather than shown.

        The person opens panels deliberately via the &Panels menu; a
        freshly-opened panel floats so it doesn't reflow the whole
        window layout.
        """
        self.addDockWidget(area, dock)
        dock.setFloating(True)
        dock.setVisible(False)

    def _build_panels_menu(self):
        """Populate &Panels with one checkable action per dock."""
        panel_docks = [
            ("Trace Controls", self.trace_controls_dock),
            ("Spectrogram Controls", self.spectrogram_controls_dock),
            ("Probe Map", self.probe_map_dock),
            ("Selected Channels", self.selected_channels_dock),
            ("Display Settings", self.display_settings_dock),
        ]
        for label, dock in panel_docks:
            action = QAction(label, self)
            action.setCheckable(True)
            action.setChecked(dock.isVisible())
            action.toggled.connect(dock.setVisible)
            dock.visibilityChanged.connect(action.setChecked)
            self.panels_menu.addAction(action)

    def _build_trace_view(self):
        """Build the trace view and add control panels as docks."""
        self.trace_view = TraceViewWidget(self.engine)
        self.trace_view.spikeNavigationRequested.connect(self._jump_to_adjacent_spike)

        self.setCentralWidget(self.trace_view)

        controls_dock = QDockWidget("Trace Controls", self)
        controls_dock.setObjectName("trace_controls_dock")
        controls_dock.setWidget(self.trace_view.control_panel)
        controls_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea |
            Qt.DockWidgetArea.RightDockWidgetArea |
            Qt.DockWidgetArea.TopDockWidgetArea |
            Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self.trace_controls_dock = controls_dock
        self._finalize_dock(controls_dock, Qt.DockWidgetArea.RightDockWidgetArea)

        spectrogram_dock = QDockWidget("Spectrogram Controls", self)
        spectrogram_dock.setObjectName("spectrogram_controls_dock")
        spectrogram_dock.setWidget(self.trace_view.spectrogram_control)
        spectrogram_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea |
            Qt.DockWidgetArea.RightDockWidgetArea |
            Qt.DockWidgetArea.TopDockWidgetArea |
            Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self.spectrogram_controls_dock = spectrogram_dock
        self._finalize_dock(spectrogram_dock, Qt.DockWidgetArea.RightDockWidgetArea)

    def _build_probe_map_dock(self):
        dock = QDockWidget("Probe Map", self)
        dock.setObjectName("probe_map_dock")
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(4, 4, 4, 4)
        container_layout.setSpacing(4)

        stream_row = QHBoxLayout()
        stream_row.addWidget(QLabel("Probe stream:"))
        self.stream_picker = QComboBox()
        self.stream_picker.setEnabled(False)
        self.stream_picker.currentIndexChanged.connect(self._on_stream_changed)
        stream_row.addWidget(self.stream_picker, stretch=1)
        container_layout.addLayout(stream_row)

        self._probe_map_placeholder_label = QLabel(
            "No probe loaded.\n\nUse File \u2192 Open settings.xml to load one."
        )
        self._probe_map_placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._probe_map_placeholder_label.setStyleSheet("color: #888; font-size: 12px;")
        container_layout.addWidget(self._probe_map_placeholder_label, stretch=1)

        self._probe_map_container = container
        self._probe_map_container_layout = container_layout

        dock.setWidget(container)
        self.probe_map_dock = dock
        self._finalize_dock(dock, Qt.DockWidgetArea.LeftDockWidgetArea)

    def _build_selected_channels_dock(self):
        dock = QDockWidget("Selected Channels", self)
        dock.setObjectName("selected_channels_dock")
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )

        container = QWidget()
        vlayout = QVBoxLayout(container)
        self.selected_list = QListWidget()
        vlayout.addWidget(self.selected_list)

        dock.setWidget(container)
        self.selected_channels_dock = dock
        self._finalize_dock(dock, Qt.DockWidgetArea.RightDockWidgetArea)

    def _build_display_settings_dock(self):
        """Build the display settings dock with color buttons."""
        dock = QDockWidget("Display Settings", self)
        dock.setObjectName("display_settings_dock")
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )

        self.trace_style_panel = TraceStylePanel(self.trace_view)
        dock.setWidget(self.trace_style_panel)

        self.display_settings_dock = dock
        self._finalize_dock(dock, Qt.DockWidgetArea.RightDockWidgetArea)

    # ------------------------------------------------------------------
    # settings.xml loading
    # ------------------------------------------------------------------

    def _on_open_settings(self):
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Open Open Ephys settings.xml", "", "XML files (*.xml);;All files (*)"
        )
        if not path_str:
            return
        self.load_settings_file(Path(path_str))

    def load_settings_file(self, path: Path):
        """Parse a settings.xml and populate the stream picker. Public so
        it can be called directly from main.py with a CLI-provided path."""
        try:
            probes = extract_probes_from_settings(str(path))
        except Exception as exc:
            QMessageBox.critical(
                self, "Failed to load settings",
                f"Could not parse {path.name}:\n\n{exc}"
            )
            return

        if not probes["probes"]:
            QMessageBox.warning(
                self, "No probes found",
                f"{path.name} was parsed but contains no enabled probe streams."
            )
            return

        self._settings_path = path
        self._probes = probes

        self.stream_picker.blockSignals(True)
        self.stream_picker.clear()
        for key in probes["probes"].keys():
            self.stream_picker.addItem(key)
        self.stream_picker.setEnabled(True)
        self.stream_picker.blockSignals(False)

        self._update_status(f"Loaded {path.name} — {len(probes['probes'])} stream(s).")

        self.stream_picker.setCurrentIndex(0)
        self._on_stream_changed(0)

        # Auto-open the Probe Map panel -- it starts closed like every
        # other panel (see _finalize_dock), but there's no point making
        # the person go open it by hand immediately after loading a
        # settings file.
        #
        # Explicitly size the floating dock after showing it: a freshly
        # setVisible(True) on a floating dock gets whatever default
        # size Qt feels like giving it, which was observed to be too
        # small for the probe map to be useful.
        self.probe_map_dock.setVisible(True)
        self.probe_map_dock.resize(520, 900)
        self.probe_map_dock.raise_()

    def _on_stream_changed(self, index: int):
        if index < 0 or not self._probes:
            return
        probe_key = self.stream_picker.itemText(index)
        if not probe_key or probe_key == self._current_probe_key:
            return

        probe_data = self._probes["probes"][probe_key]
        if not probe_data["coordinates"].get("x"):
            QMessageBox.warning(
                self, "No coordinates",
                f"Stream '{probe_key}' has no electrode coordinates and "
                "cannot be displayed."
            )
            return

        self._current_probe_key = probe_key
        self._load_probe_map(probe_data)

        n_ch = len(probe_data["coordinates"]["channels"])
        n_shanks = probe_data.get("num_shanks", 1)
        self._update_status(
            f"{probe_key}  \u2014  {n_ch} channels, {n_shanks} shank(s), "
            f"{probe_data.get('sample_rate', 0):.0f} Hz"
        )

    def _load_probe_map(self, probe_data: dict):

        # Tear down any previous probe map cleanly before building the new one.
        if self.probe_map is not None:
            try:
                self.probe_map.channelsSelected.disconnect(self._on_channels_selected)
            except (TypeError, RuntimeError):
                pass
            self.probe_map.deleteLater()
            self.probe_map = None

        self.probe_map = ProbeMapWidget(probe_data)

        self.probe_map.channelsSelected.connect(self._on_channels_selected)
        self.probe_map_dock.setWidget(self.probe_map)
        self.probe_map_dock.setVisible(True)

        self.selected_list.clear()
        self._update_analysis_actions_enabled()
        self._push_full_probe_geometry_to_trace_view()
              
    def _push_full_probe_geometry_to_trace_view(self):
        if not self._current_probe_key or not self._probes:
            return
        
        """Push the entire probe's depth/shank/x geometry into the
        trace view, so features that need to reason about the probe as
        a whole (CSD neighbor lookup, focus-mode neighborhoods) work
        even when nothing is currently selected for display.

        Called once per stream load, in _load_probe_map. Not called
        from _on_channels_selected -- that method's job is to update
        the *selection*, not the *geometry*, and geometry doesn't
        change when the selection does.
        """
        if not self._current_probe_key or not self._probes:
            return
        probe_data = self._probes["probes"][self._current_probe_key]
        coords = probe_data.get("coordinates", {})
        channels = coords.get("channels", [])
        if not channels:
            return

        depths = dict(zip(channels, coords.get("y", [])))
        xcoords = dict(zip(channels, coords.get("x", [])))
        shank_ids = (
            probe_data.get("shanks", {}).get("ids")
            or [0] * len(channels)
        )
        shank_map = dict(zip(channels, shank_ids))

        self.trace_view.set_channel_depths(depths)
        self.trace_view.set_channel_shanks(shank_map)
        self.trace_view.set_channel_xcoords(xcoords)
        self.trace_view.set_full_probe_geometry(depths, shank_map, xcoords)
    # ------------------------------------------------------------------
    # continuous.dat loading
    # ------------------------------------------------------------------

    def _on_open_data(self):
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Open continuous.dat", "", "DAT files (*.dat);;All files (*)"
        )
        if not path_str:
            return
        self.load_data_file(Path(path_str))

    def load_data_file(self, path: Path, n_channels: int | None = None,
                        sample_rate: float | None = None):
        """Load a continuous.dat into a fresh TraceEngine and hand it to
        the trace view."""
        if n_channels is None and self._current_probe_key:
            probe_data = self._probes["probes"][self._current_probe_key]
            n_channels = len(probe_data["coordinates"]["channels"]) or self.engine.n_channels
        if sample_rate is None and self._current_probe_key:
            probe_data = self._probes["probes"][self._current_probe_key]
            sample_rate = probe_data.get("sample_rate") or self.engine.sr

        new_engine = TraceEngine(
            n_channels=n_channels or self.engine.n_channels,
            sample_rate=sample_rate or self.engine.sr,
        )

        try:
            new_engine.load_data_file(path)
        except Exception as exc:
            QMessageBox.critical(
                self, "Failed to load data",
                f"Could not load {path.name}:\n\n{exc}"
            )
            return

        # Carry over the current probe-map selection as the initial
        # channel set.
        if self.probe_map is not None:
            selected = self.probe_map.get_selected_channels()
            if selected:
                new_engine.set_channels(selected)

        # Close any open analysis dialogs.
        for dialog in list(self._open_pac_dialogs):
            dialog.close()
        for dialog in list(self._open_power_dialogs):
            dialog.close()
        for dialog in list(self._open_ripple_dialogs):
            dialog.close()
        for dialog in list(self._open_psd_dialogs):
            dialog.close()
            
        self.engine = new_engine
        self.trace_view.set_data_source(self.engine)
        
        # Spike rasters belong to the old recording. Clear them and
        # blank the units panel until a new Phy folder is loaded.
        self.trace_view.set_raster_units([])
        if self.phy_units_panel is not None:
            self.phy_units_panel.set_phy_data(None)
        if self.phy_units_panel is not None:
            self.phy_units_panel.set_recolor_enabled(False)
        if self.trace_view is not None:
            self.trace_view.set_spike_recolor_visible(False)



        self.trace_view.clear_theta_epochs()

        if self.probe_map is not None:
            selected = self.probe_map.get_selected_channels()
            if selected:
                self.trace_view.set_channels(selected)

                # Pass depth, x, AND shank info to trace view. Shank
                # info is required for CSD: the 3-point Laplacian is
                # taken within a shank, never across. x-coords are used
                # to break ties among same-depth candidates.
                if self._current_probe_key and self._probes:
                    probe_data = self._probes["probes"][self._current_probe_key]
                    depths = dict(zip(
                        probe_data["coordinates"]["channels"],
                        probe_data["coordinates"]["y"]
                    ))
                self.trace_view.set_channel_depths(depths)

                xcoords = dict(zip(
                    probe_data["coordinates"]["channels"],
                    probe_data["coordinates"]["x"]
                ))
                self.trace_view.set_channel_xcoords(xcoords)

                shank_ids = probe_data.get("shanks", {}).get("ids") or [0] * len(probe_data["coordinates"]["channels"])
                shank_map = dict(zip(probe_data["coordinates"]["channels"], shank_ids))
                self.trace_view.set_channel_shanks(shank_map)

                # Full-probe geometry for CSD neighbor lookup: every
                # channel on the probe, not just the ones currently
                # selected for display. CSD's Laplacian must use the
                # physical neighbors of a channel, regardless of
                # whether they happen to be drawn right now.
                self.trace_view.set_full_probe_geometry(depths, shank_map, xcoords)

        self._update_analysis_actions_enabled()

        if hasattr(self, 'trace_style_panel'):
            self.trace_style_panel.update_channels()

        self._update_status(
            f"Loaded {path.name}  —  {new_engine.total_duration:.1f}s, "
            f"{new_engine.n_channels} channels @ {new_engine.sr:.0f} Hz"
        )

    def _on_toggle_phy_units_dock(self, checked: bool):
        """Show or hide the Phy Units dock via the Panels menu."""
        if self.phy_units_dock is None:
            return
        self.phy_units_dock.setVisible(checked)
        self.phy_units_dock.setFloating(True)
        if checked:
            self.phy_units_dock.raise_()

    def _on_open_probe_definition(self):
        """Open the definition editor, then use the resulting definition
        with a dat the user picks immediately after."""
        dialog = ProbeDefinitionDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted or dialog.definition is None:
            return

        defn = dialog.definition

        path_str, _ = QFileDialog.getOpenFileName(
            self, f"Open continuous.dat for '{defn.name}'", "",
            "DAT files (*.dat);;All files (*)"
        )
        if not path_str:
            return

        self.load_data_file_with_definition(defn, Path(path_str))

    def load_data_file_with_definition(self, defn: ProbeDefinition, path: Path):
        """Load a raw continuous.dat against a hand-authored (or
        imported) probe definition, bypassing the settings.xml machinery
        entirely.

        The definition is wrapped into the exact dict shape that
        extract_probes_from_settings produces, then the rest of the app
        runs unchanged.
        """
        try:
            probe_data = defn.to_probe_data_dict()
        except ValueError as exc:
            QMessageBox.critical(self, "Invalid definition", str(exc))
            return

        # Register the definition as a single-stream probe set, exactly
        # as if it had come from a settings.xml. This makes it
        # selectable in the stream picker and drives every downstream
        # consumer without special-casing.
        stream_key = f"{defn.name}-custom"
        self._probes = {"record_path": "", "probes": {stream_key: probe_data}}
        self._settings_path = None

        self.stream_picker.blockSignals(True)
        self.stream_picker.clear()
        self.stream_picker.addItem(stream_key)
        self.stream_picker.setEnabled(True)
        self.stream_picker.blockSignals(False)

        self._current_probe_key = stream_key
        self._load_probe_map(probe_data)

        # Now load the dat itself against the definition.
        new_engine = TraceEngine(
            n_channels=int(defn.n_channels),
            sample_rate=float(defn.sample_rate),
            dtype=np.dtype(defn.dtype),
        )
        try:
            new_engine.load_data_file(path)
        except Exception as exc:
            QMessageBox.critical(
                self, "Failed to load data",
                f"Could not load {path.name} against definition '{defn.name}':\n\n{exc}"
            )
            return

        # Close any open analysis dialogs (they were tied to the old data).
        for dialog in list(self._open_pac_dialogs):
            dialog.close()
        for dialog in list(self._open_power_dialogs):
            dialog.close()
        for dialog in list(self._open_ripple_dialogs):
            dialog.close()

        self.engine = new_engine
        self.trace_view.set_data_source(self.engine)
        self.trace_view.clear_theta_epochs()

        # Register the FULL probe geometry immediately so CSD, depth
        # sorting, and everything else is ready the moment the user
        # picks channels on the (freshly built) probe map. No selection
        # is carried over -- this is a new probe the user has never
        # selected channels on.
        depths = dict(zip(probe_data["coordinates"]["channels"], probe_data["coordinates"]["y"]))
        xcoords = dict(zip(probe_data["coordinates"]["channels"], probe_data["coordinates"]["x"]))
        shank_ids = probe_data.get("shanks", {}).get("ids") or [0] * len(probe_data["coordinates"]["channels"])
        shank_map = dict(zip(probe_data["coordinates"]["channels"], shank_ids))
        self.trace_view.set_full_probe_geometry(depths, shank_map, xcoords)

        self._update_analysis_actions_enabled()
        if hasattr(self, 'trace_style_panel'):
            self.trace_style_panel.update_channels()

        self._update_status(
            f"Loaded {path.name} against custom definition '{defn.name}'  —  "
            f"{new_engine.total_duration:.1f}s, {new_engine.n_channels} channels "
            f"@ {new_engine.sr:.0f} Hz ({defn.dtype})"
        )



    def _on_open_timestamps(self):
        """Open a timestamps.npy file."""
        if not self.engine.data_loaded:
            QMessageBox.warning(self, "No data loaded",
                            "Load a continuous.dat file first.")
            return

        path_str, _ = QFileDialog.getOpenFileName(
            self, "Open timestamps.npy", "",
            "NumPy files (*.npy);;All files (*)"
        )
        if not path_str:
            return

        try:
            success = self.engine.load_timestamps(Path(path_str))
            if success:
                self._update_status(
                    f"Loaded timestamps from {Path(path_str).name}"
                )
                self.trace_view._invalidate_cache()
                self.trace_view._update_time_labels()
                self.trace_view.update()
            else:
                QMessageBox.warning(
                    self, "Invalid timestamps",
                    "Timestamps file doesn't match data length or is invalid."
                )
        except Exception as exc:
            QMessageBox.critical(
                self, "Failed to load timestamps",
                f"Could not load timestamps file:\n\n{exc}"
            )

    # ------------------------------------------------------------------
    # Selection feedback
    # ------------------------------------------------------------------

    def _on_channels_selected(self, channels: list):
        """Handle channel selection from probe map.

        Geometry (depths, shanks, xcoords, full-probe maps) is pushed
        once per stream load in _push_full_probe_geometry_to_trace_view;
        this method only updates the *selection*.
        """
        self.selected_list.clear()
        for ch in channels:
            self.selected_list.addItem(QListWidgetItem(f"CH{ch}"))
        self.selected_channels_dock.setWindowTitle(
            f"Selected Channels ({len(channels)})"
        )

        if self.engine.data_loaded:
            self.trace_view.set_channels(channels)

        if hasattr(self, 'trace_style_panel'):
            self.trace_style_panel.update_channels()

        self._update_status(f"{len(channels)} channels selected")

        if hasattr(self, 'trace_view') and self.trace_view.show_spectrogram:
            if channels:
                self.trace_view.set_spectrogram_channel(channels[0])



    def get_selected_channels(self) -> list[int]:
        """Public accessor for whatever consumes the selection next."""
        if self.probe_map is None:
            return []
        return self.probe_map.get_selected_channels()

    # ------------------------------------------------------------------
    # Color editing
    # ------------------------------------------------------------------

    def _on_edit_trace_color(self):
        if self.trace_view:
            current_color = self.trace_view.get_trace_color()
            color = QColorDialog.getColor(
                QColor(current_color), self, "Select Trace Color"
            )
            if color.isValid():
                self.trace_view.set_trace_color(color.name())
                self.trace_style_panel.trace_color_btn.set_color(color)

    def _on_edit_bg_color(self):
        if self.trace_view:
            current_color = self.trace_view.get_background_color()
            color = QColorDialog.getColor(
                QColor(current_color), self, "Select Background Color"
            )
            if color.isValid():
                self.trace_view.set_background_color(color.name())
                self.trace_style_panel.bg_color_btn.set_color(color)

    # ------------------------------------------------------------------
    # Analysis dialogs
    # ------------------------------------------------------------------

    def _on_open_phase_amplitude(self):
        if self.probe_map is None or not self._current_probe_key:
            QMessageBox.warning(
                self, "No probe loaded",
                "Load a settings.xml file first to provide probe geometry."
            )
            return
        if not self.engine.data_loaded:
            QMessageBox.warning(
                self, "No data loaded",
                "Load a continuous.dat file first."
            )
            return

        probe_data = self._probes["probes"][self._current_probe_key]
        dialog = PhaseAmplitudeDialog(probe_data, self.engine, parent=self)

        selected = self.probe_map.get_selected_channels()
        if selected:
            dialog.set_channel(selected[0])

        start = self.trace_view.start_time
        end = min(
            self.engine.total_duration,
            start + max(self.trace_view.window_duration, 1.0)
        )
        dialog.start_spin.setValue(start)
        dialog.end_spin.setValue(end)

        self._open_pac_dialogs.append(dialog)
        dialog.finished.connect(lambda _res, d=dialog: self._on_pac_dialog_closed(d))
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _on_pac_dialog_closed(self, dialog: PhaseAmplitudeDialog):
        if dialog in self._open_pac_dialogs:
            self._open_pac_dialogs.remove(dialog)

    def _on_open_power_map(self):
        if self.probe_map is None or not self._current_probe_key:
            QMessageBox.warning(
                self, "No probe loaded",
                "Load a settings.xml file first to provide probe geometry."
            )
            return
        if not self.engine.data_loaded:
            QMessageBox.warning(
                self, "No data loaded",
                "Load a continuous.dat file first."
            )
            return

        probe_data = self._probes["probes"][self._current_probe_key]
        dialog = AmplitudePowerDialog(probe_data, self.engine, parent=self)

        start = self.trace_view.start_time
        end = min(
            self.engine.total_duration,
            start + max(self.trace_view.window_duration, 1.0)
        )
        dialog.start_spin.setValue(start)
        dialog.end_spin.setValue(end)

        self._open_power_dialogs.append(dialog)
        dialog.finished.connect(lambda _res, d=dialog: self._on_power_dialog_closed(d))
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _on_power_dialog_closed(self, dialog: AmplitudePowerDialog):
        if dialog in self._open_power_dialogs:
            self._open_power_dialogs.remove(dialog)


    def _on_open_psd(self):
        if not self.engine.data_loaded:
            QMessageBox.warning(self, "No data loaded",
                                "Load a continuous.dat file first.")
            return

        selected = self.get_selected_channels()
        initial_channel = selected[0] if selected else None

        dialog = PsdDialog(self.engine, initial_channel=initial_channel, parent=self)

        # Pre-fill the time range from the current trace view so the
        # analysis starts on the window you were just looking at.
        start = self.trace_view.start_time
        end = min(
            self.engine.total_duration,
            start + max(self.trace_view.window_duration, 1.0),
        )
        dialog.start_spin.setValue(start)
        dialog.end_spin.setValue(end)

        self._open_psd_dialogs.append(dialog)
        dialog.finished.connect(lambda _res, d=dialog: self._on_psd_closed(d))
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _on_psd_closed(self, dialog: PsdDialog):
        if dialog in self._open_psd_dialogs:
            self._open_psd_dialogs.remove(dialog)


    def _on_open_ripple_detection(self):
        if self.probe_map is None or not self._current_probe_key:
            QMessageBox.warning(
                self, "No probe loaded",
                "Load a settings.xml file first to provide probe geometry."
            )
            return
        if not self.engine.data_loaded:
            QMessageBox.warning(
                self, "No data loaded",
                "Load a continuous.dat file first."
            )
            return

        probe_data = self._probes["probes"][self._current_probe_key]
        selected = self.probe_map.get_selected_channels()
        dialog = RippleDialog(
            probe_data, self.engine, self.trace_view,
            initial_channels=selected, parent=self
        )

        start = self.trace_view.start_time
        end = min(
            self.engine.total_duration,
            start + max(self.trace_view.window_duration * 5, 5.0)
        )
        dialog.start_spin.setValue(start)
        dialog.end_spin.setValue(end)

        self._open_ripple_dialogs.append(dialog)
        dialog.finished.connect(lambda _res, d=dialog: self._on_ripple_dialog_closed(d))
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _on_ripple_dialog_closed(self, dialog: RippleDialog):
        if dialog in self._open_ripple_dialogs:
            self._open_ripple_dialogs.remove(dialog)

    def _on_open_theta_epoch(self):
        if self.probe_map is None or not self._current_probe_key:
            QMessageBox.warning(self, "No probe loaded", "Load settings.xml first.")
            return
        if not self.engine.data_loaded:
            QMessageBox.warning(self, "No data loaded", "Load continuous.dat first.")
            return

        probe_data = self._probes["probes"][self._current_probe_key]
        selected = self.probe_map.get_selected_channels()

        dialog = ThetaEpochDialog(probe_data, self.engine, initial_channels=selected, trace_view=self.trace_view, parent=self)

        self._open_theta_dialogs.append(dialog)

        dialog.epochsChanged.connect(
            lambda epochs, d=dialog: self.trace_view.set_theta_epochs(
                epochs,
                detection_start_time=d.start_spin.value(),
                detection_sample_offset=getattr(d, "_sample_offset", None),
                min_merge_gap_ms=d.merge_gap_spin.value(),
            )
        )
        self.trace_view.thetaEpochsChanged.connect(
            dialog.set_epochs_from_trace
        )
        dialog.mergeGapChanged.connect(self.trace_view.set_theta_merge_gap)
        self.trace_view.set_theta_merge_gap(dialog.merge_gap_spin.value())
        dialog.thetaEpochSelected.connect(self.trace_view.set_theta_selected_epoch)
        self.trace_view.thetaEpochSelected.connect(dialog.set_selected_epoch)

        dialog.finished.connect(lambda _res, d=dialog: self._on_theta_dialog_closed(d))
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _on_theta_dialog_closed(self, dialog):
        if dialog in self._open_theta_dialogs:
            self._open_theta_dialogs.remove(dialog)

    def _on_toggle_spectrogram(self, enabled: bool):
        """Toggle spectrogram display."""
        if not hasattr(self, 'trace_view'):
            return

        self.trace_view.set_spectrogram_enabled(enabled)
        self.trace_view.spectrogram_control.set_enabled(enabled)

        if enabled and self.trace_view.spectrogram_channel is None:
            selected = self.get_selected_channels()
            if selected:
                self.trace_view.set_spectrogram_channel(selected[0])
                self.trace_view.spectrogram_control.set_channel(selected[0])

        self.trace_view.set_spectrogram_enabled(enabled)

        if enabled and self.trace_view.spectrogram_channel is None:
            selected = self.get_selected_channels()
            if selected:
                self.trace_view.set_spectrogram_channel(selected[0])

    def _on_phy_units_selection_changed(self, cluster_ids: list):
        if self.trace_view is not None:
            self.trace_view.set_raster_units(cluster_ids)
        self._update_status(f"{len(cluster_ids)} unit(s) selected for raster")

    def _on_open_phy_folder(self):
        if not self.engine.data_loaded:
            QMessageBox.warning(
                self, "No data loaded",
                "Load a continuous.dat first; spike times in Phy output "
                "are sample indices into that recording."
            )
            return

        folder_str = QFileDialog.getExistingDirectory(
            self, "Open Kilosort / Phy output folder"
        )
        if not folder_str:
            return

        success = self.engine.load_phy_folder(Path(folder_str))
        if not success:
            QMessageBox.critical(
                self, "Failed to load Phy folder",
                f"{folder_str} doesn't look like a Kilosort/Phy output "
                "folder, or its spike_times.npy / spike_clusters.npy "
                "couldn't be read. See the terminal for details."
            )
            return

        # Push the loaded data into the units panel.
        if self.phy_units_panel is not None:
            self.phy_units_panel.set_phy_data(self.engine.phy_data)

        # Show the dock and sync the Panels menu checkbox so the two
        # stay consistent no matter how the user opens it.
        if self.phy_units_dock is not None:
            self.phy_units_dock.setVisible(True)
            self.phy_units_dock.setFloating(True)
            self.phy_units_dock.raise_()
        if self.phy_units_dock_action is not None:
            self.phy_units_dock_action.blockSignals(True)
            self.phy_units_dock_action.setChecked(True)
            self.phy_units_dock_action.blockSignals(False)

        # Clear any previously-selected units in the trace view; the
        # new folder's unit ids refer to a different sort.
        self.trace_view.set_raster_units([])

        n_units = len(self.engine.phy_data.units)
        self._update_status(
            f"Loaded Phy folder: {Path(folder_str).name} — {n_units} unit(s)"
        )
    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def _update_status(self, message: str):
        self.statusBar().showMessage(message)
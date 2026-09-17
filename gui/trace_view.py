"""
trace_view.py

Main trace visualization widget. Displays neural data from a TraceEngine
as scrolling lines. Supports channel selection from the probe map,
time-window navigation, customizable colors, depth-based sorting, and
signal filtering.

Display model
-------------
Two independent levels of control:

1. GLOBAL (the Trace Controls panel, unchanged): filter/notch/detrend
   settings that apply to every channel by default.

2. PER-CHANNEL (new): each channel can OPT OUT of the global pipeline
   and instead draw an explicit combination of {raw, filtered, csd}.

   - A channel with NO explicit configuration follows the global
     pipeline exactly as before: raw, or globally-filtered if a global
     filter is enabled.
   - Once the user picks any signal(s) for a channel, that channel
     shows ONLY those signals, overlaid. The global filter no longer
     applies to it.

   The three signals are alternatives, not pipeline stages:
     raw       -- the raw trace from the engine
     filtered  -- the raw trace run through the current global filter
     csd       -- the 3-point Laplacian of the raw trace and its two
                  nearest same-shank neighbors

   Each mode has its own default color:
     raw       -- the channel's own color (default_trace_color or a
                  per-channel override)
     filtered  -- light cyan (#4cc9f0), clearly "processed"
     csd       -- orange (#ff9f1c), a different physical quantity

UI
--
- Left-click on a channel's per-row button (the small ⋮ at the right
  edge of the label strip) opens that channel's display-mode menu.
- A header button at the top of the label strip opens a bulk menu
  (apply-to-all, reset-all).
- The left label strip has improved contrast and legibility.
"""

from __future__ import annotations

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollBar,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QSizePolicy,
    QGroupBox, QFrame, QToolBar, QSlider, QDockWidget, QMainWindow,
    QMenu, QToolButton, QGridLayout,
)
from PyQt6.QtGui import (
    QPainter, QPen, QColor, QBrush, QFont, QPainterPath, QIcon,
    QPixmap, QImage,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QRectF


from core.trace_engine import TraceEngine
from core.filters import bandpass_filter, notch_filter, hilbert

from scipy.signal import spectrogram, butter, sosfiltfilt, resample_poly
from matplotlib import colormaps


# Default colors for the three per-channel display signals. The 'raw'
# mode uses each channel's own color (default_trace_color or a
# per-channel override); only 'filtered' and 'csd' get fixed colors
# here, so they read as distinct regardless of the channel's base
# color.
FILTERED_COLOR = QColor("#4cc9f0")
CSD_COLOR = QColor("#ff9f1c")

# Label-strip visuals.
LABEL_STRIP_BG = QColor(0, 0, 0, 150)
LABEL_TEXT_COLOR = QColor("#e8e8e8")
LABEL_FONT_POINT_SIZE = 9
LABEL_STRIP_WIDTH = 92

# Small "menu button" drawn next to each channel's name/depth.
ROW_BUTTON_WIDTH = 20
ROW_BUTTON_HEIGHT = 16
ROW_BUTTON_BG = QColor("#2d2d2d")
ROW_BUTTON_BG_HOVER = QColor("#3d5d80")
ROW_BUTTON_BORDER = QColor("#4a4a4a")
ROW_BUTTON_BORDER_HOVER = QColor("#5a9fff")
ROW_BUTTON_GLYPH = QColor("#e8e8e8")

# Maximum allowed depth gap between a channel and a CSD neighbor.
# Comfortably above the densest Neuropixels pitch (~15 µm), well below
# any accidental cross-shank grouping.
MAX_CSD_NEIGHBOR_GAP_UM = 60.0


class TraceControlPanel(QWidget):
    """
    Control panel for trace view settings.
    Can be used as a dockable widget or embedded in the trace view.
    """

    timeChanged = pyqtSignal(float, float)
    durationChanged = pyqtSignal(float)
    gainChanged = pyqtSignal(float)
    autoScaleChanged = pyqtSignal(bool)
    animationToggled = pyqtSignal(bool)
    animationSpeedChanged = pyqtSignal(int)
    filterChanged = pyqtSignal(dict)
    zoomChanged = pyqtSignal(float)
    depthScaleChanged = pyqtSignal(bool)
    clearCursorsRequested = pyqtSignal()
    useTimestampsChanged = pyqtSignal(bool)
    goToStartRequested = pyqtSignal()
    goToEndRequested = pyqtSignal()
    stepTimeRequested = pyqtSignal(int)
    performanceChanged = pyqtSignal(dict)   

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Trace Controls")
        self.setMinimumWidth(350)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        self.channel_info_label = QLabel("No channels")
        self.channel_info_label.setStyleSheet("color: #888; font-size: 10px; font-weight: bold;")
        self.channel_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.channel_info_label)

        # ---- Navigation ----
        nav_group = QGroupBox("Navigation")
        nav_layout = QVBoxLayout(nav_group)
        nav_layout.setSpacing(2)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(2)

        self.start_btn = QPushButton("⏮")
        self.start_btn.setToolTip("Go to start")
        self.start_btn.clicked.connect(self.goToStartRequested.emit)
        btn_row.addWidget(self.start_btn)

        self.step_back_btn = QPushButton("◀")
        self.step_back_btn.setToolTip("Step back")
        self.step_back_btn.clicked.connect(lambda: self.stepTimeRequested.emit(-1))
        btn_row.addWidget(self.step_back_btn)

        self.step_fwd_btn = QPushButton("▶")
        self.step_fwd_btn.setToolTip("Step forward")
        self.step_fwd_btn.clicked.connect(lambda: self.stepTimeRequested.emit(1))
        btn_row.addWidget(self.step_fwd_btn)

        self.end_btn = QPushButton("⏭")
        self.end_btn.setToolTip("Go to end")
        self.end_btn.clicked.connect(self.goToEndRequested.emit)
        btn_row.addWidget(self.end_btn)

        btn_row.addStretch()
        nav_layout.addLayout(btn_row)

        time_row = QHBoxLayout()
        time_row.setSpacing(2)
        time_row.addWidget(QLabel("Time:"))
        self.start_label = QLabel("0.00s")
        self.start_label.setStyleSheet("font-weight: bold; color: #fff;")
        time_row.addWidget(self.start_label)
        time_row.addWidget(QLabel("—"))
        self.end_label = QLabel("10.00s")
        self.end_label.setStyleSheet("font-weight: bold; color: #fff;")
        time_row.addWidget(self.end_label)
        time_row.addStretch()
        nav_layout.addLayout(time_row)

        dur_row = QHBoxLayout()
        dur_row.setSpacing(2)
        dur_row.addWidget(QLabel("Duration:"))
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.1, 3600.0)
        self.duration_spin.setValue(10.0)
        self.duration_spin.setSuffix(" s")
        self.duration_spin.valueChanged.connect(self._on_duration_changed)
        dur_row.addWidget(self.duration_spin)
        dur_row.addStretch()
        nav_layout.addLayout(dur_row)

        layout.addWidget(nav_group)

        # ---- Playback ----
        playback_group = QGroupBox("Playback")
        playback_layout = QVBoxLayout(playback_group)
        playback_layout.setSpacing(2)

        play_row = QHBoxLayout()
        play_row.setSpacing(2)
        self.animation_btn = QPushButton("▶ Play")
        self.animation_btn.clicked.connect(self._on_animation_clicked)
        play_row.addWidget(self.animation_btn)

        play_row.addWidget(QLabel("Speed:"))
        self.animation_speed_spin = QSpinBox()
        self.animation_speed_spin.setRange(10, 1000)
        self.animation_speed_spin.setValue(100)
        self.animation_speed_spin.setSuffix(" ms")
        self.animation_speed_spin.valueChanged.connect(self._on_animation_speed_changed)
        play_row.addWidget(self.animation_speed_spin)
        play_row.addStretch()
        playback_layout.addLayout(play_row)

        layout.addWidget(playback_group)

        # ---- Gain & Scaling ----
        gain_group = QGroupBox("Gain & Scaling")
        gain_layout = QVBoxLayout(gain_group)
        gain_layout.setSpacing(2)

        gain_row = QHBoxLayout()
        gain_row.setSpacing(2)
        gain_row.addWidget(QLabel("Gain:"))
        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(0.1, 1000.0)
        self.gain_spin.setValue(1.0)
        self.gain_spin.setDecimals(1)
        self.gain_spin.valueChanged.connect(self._on_gain_changed)
        gain_row.addWidget(self.gain_spin)
        gain_row.addStretch()
        gain_layout.addLayout(gain_row)

        self.auto_scale_checkbox = QCheckBox("Auto-scale each channel")
        self.auto_scale_checkbox.setChecked(False)
        self.auto_scale_checkbox.setToolTip("When enabled, each channel is scaled to fill its vertical space")
        self.auto_scale_checkbox.toggled.connect(self._on_auto_scale_toggled)
        gain_layout.addWidget(self.auto_scale_checkbox)

        layout.addWidget(gain_group)

        # ---- Filters ----
        filter_group = QGroupBox("Filters")
        filter_layout = QVBoxLayout(filter_group)
        filter_layout.setSpacing(2)

        bp_row = QHBoxLayout()
        bp_row.setSpacing(2)
        self.filter_checkbox = QCheckBox("Bandpass:")
        self.filter_checkbox.toggled.connect(self._on_filter_toggled)
        bp_row.addWidget(self.filter_checkbox)

        self.low_freq_spin = QDoubleSpinBox()
        self.low_freq_spin.setRange(0.0, 15000.0)
        self.low_freq_spin.setValue(1.0)
        self.low_freq_spin.setSuffix(" Hz")
        self.low_freq_spin.setEnabled(False)
        self.low_freq_spin.setToolTip("Set to 0 for low-pass only")
        self.low_freq_spin.valueChanged.connect(self._on_filter_params_changed)
        bp_row.addWidget(self.low_freq_spin)

        self.high_freq_spin = QDoubleSpinBox()
        self.high_freq_spin.setRange(0.0, 15000.0)
        self.high_freq_spin.setValue(300.0)
        self.high_freq_spin.setSuffix(" Hz")
        self.high_freq_spin.setEnabled(False)
        self.high_freq_spin.setToolTip("Set to 0 for high-pass only")
        self.high_freq_spin.valueChanged.connect(self._on_filter_params_changed)
        bp_row.addWidget(self.high_freq_spin)
        bp_row.addStretch()
        filter_layout.addLayout(bp_row)

        notch_row = QHBoxLayout()
        notch_row.setSpacing(2)
        self.notch_checkbox = QCheckBox("Notch:")
        self.notch_checkbox.toggled.connect(self._on_notch_toggled)
        notch_row.addWidget(self.notch_checkbox)

        self.notch_freq_spin = QDoubleSpinBox()
        self.notch_freq_spin.setRange(0.0, 5000.0)
        self.notch_freq_spin.setValue(50.0)
        self.notch_freq_spin.setSuffix(" Hz")
        self.notch_freq_spin.setEnabled(False)
        self.notch_freq_spin.valueChanged.connect(self._on_filter_params_changed)
        notch_row.addWidget(self.notch_freq_spin)
        notch_row.addStretch()
        filter_layout.addLayout(notch_row)

        self.detrend_checkbox = QCheckBox("Detrend (remove DC offset)")
        self.detrend_checkbox.toggled.connect(self._on_detrend_toggled)
        filter_layout.addWidget(self.detrend_checkbox)

        layout.addWidget(filter_group)

        # ---- Display ----
        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout(display_group)
        display_layout.setSpacing(2)

        self.depth_scale_checkbox = QCheckBox("Show depth scale")
        self.depth_scale_checkbox.setChecked(True)
        self.depth_scale_checkbox.toggled.connect(self._on_depth_scale_toggled)
        display_layout.addWidget(self.depth_scale_checkbox)

        self.clear_cursors_btn = QPushButton("🗑 Clear All Cursors")
        self.clear_cursors_btn.clicked.connect(self._on_clear_cursors)
        display_layout.addWidget(self.clear_cursors_btn)

        layout.addWidget(display_group)

        # NOTE on scope: this checkbox controls ONLY how time is *displayed*.
        self.use_timestamps_checkbox = QCheckBox("Show real timestamp values on axis")
        self.use_timestamps_checkbox.setToolTip(
            "Display axis labels using the loaded timestamps.npy values "
            "instead of elapsed seconds. Does not change which samples "
            "are shown or where cursors are placed."
        )
        self.use_timestamps_checkbox.setChecked(False)
        self.use_timestamps_checkbox.setEnabled(False)
        self.use_timestamps_checkbox.toggled.connect(self._on_use_timestamps_toggled)
        display_layout.addWidget(self.use_timestamps_checkbox)

        # ---- Zoom ----
        zoom_group = QGroupBox("Zoom")
        zoom_layout = QHBoxLayout(zoom_group)
        zoom_layout.setSpacing(2)

        zoom_out_btn = QPushButton("−")
        zoom_out_btn.clicked.connect(lambda: self._on_zoom_changed(0.5))
        zoom_layout.addWidget(zoom_out_btn)

        self.zoom_label = QLabel("100%")
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        zoom_layout.addWidget(self.zoom_label)

        zoom_in_btn = QPushButton("+")
        zoom_in_btn.clicked.connect(lambda: self._on_zoom_changed(2.0))
        zoom_layout.addWidget(zoom_in_btn)

        zoom_layout.addStretch()
        layout.addWidget(zoom_group)

        # ---- Performance section ----
        perf_group = QGroupBox("Performance")
        perf_layout = QGridLayout(perf_group)
        perf_layout.addWidget(QLabel("Optimal pts/ch:"), 0, 0)
        self.optimal_points_spin = QSpinBox()
        self.optimal_points_spin.setRange(200, 20000)
        self.optimal_points_spin.setSingleStep(100)
        self.optimal_points_spin.setValue(2000)
        self.optimal_points_spin.setToolTip("Target number of points per channel when zoomed out (min/max decimation).")
        self.optimal_points_spin.valueChanged.connect(self._on_perf_settings_changed)
        perf_layout.addWidget(self.optimal_points_spin, 0, 1)
        perf_layout.addWidget(QLabel("Max total pts:"), 1, 0)
        self.max_total_points_spin = QSpinBox()
        self.max_total_points_spin.setRange(10000, 2000000)
        self.max_total_points_spin.setSingleStep(10000)
        self.max_total_points_spin.setValue(300000)
        self.max_total_points_spin.setToolTip("Upper bound on points summed across all visible channels.")
        self.max_total_points_spin.valueChanged.connect(self._on_perf_settings_changed)
        perf_layout.addWidget(self.max_total_points_spin, 1, 1)
        layout.addWidget(perf_group)



        

    # ---- Signal emitters ----

    def _on_perf_settings_changed(self):
        self.performanceChanged.emit({
            'optimal_points': self.optimal_points_spin.value(),
            'max_total_points': self.max_total_points_spin.value(),
        })

    def _on_use_timestamps_toggled(self, checked: bool):
        self.useTimestampsChanged.emit(checked)

    def set_timestamps_available(self, available: bool):
        self.use_timestamps_checkbox.setEnabled(available)
        if not available:
            self.use_timestamps_checkbox.setChecked(False)

    def _on_clear_cursors(self):
        self.clearCursorsRequested.emit()

    def _on_duration_changed(self, value: float):
        self.durationChanged.emit(value)

    def _on_gain_changed(self, value: float):
        self.gainChanged.emit(value)

    def _on_auto_scale_toggled(self, checked: bool):
        self.autoScaleChanged.emit(checked)

    def _on_animation_clicked(self):
        self.animationToggled.emit(True)

    def _on_animation_speed_changed(self, value: int):
        self.animationSpeedChanged.emit(value)

    def _on_filter_toggled(self, checked: bool):
        self.low_freq_spin.setEnabled(checked)
        self.high_freq_spin.setEnabled(checked)
        self._emit_filter_settings()

    def _on_notch_toggled(self, checked: bool):
        self.notch_freq_spin.setEnabled(checked)
        self._emit_filter_settings()

    def _on_detrend_toggled(self, checked: bool):
        self._emit_filter_settings()

    def _on_filter_params_changed(self):
        self._emit_filter_settings()

    def _emit_filter_settings(self):
        settings = {
            'bandpass_enabled': self.filter_checkbox.isChecked(),
            'low_freq': self.low_freq_spin.value(),
            'high_freq': self.high_freq_spin.value(),
            'notch_enabled': self.notch_checkbox.isChecked(),
            'notch_freq': self.notch_freq_spin.value(),
            'detrend': self.detrend_checkbox.isChecked(),
        }
        self.filterChanged.emit(settings)

    def _on_depth_scale_toggled(self, checked: bool):
        self.depthScaleChanged.emit(checked)

    def _on_zoom_changed(self, factor: float):
        self.zoomChanged.emit(factor)

    # ---- Public setters ----

    def set_time_display(self, start_time: float, end_time: float):
        self.start_label.setText(f"{start_time:.2f}s")
        self.end_label.setText(f"{end_time:.2f}s")

    def set_duration(self, duration: float):
        self.duration_spin.blockSignals(True)
        self.duration_spin.setValue(duration)
        self.duration_spin.blockSignals(False)

    def set_gain(self, gain: float):
        self.gain_spin.blockSignals(True)
        self.gain_spin.setValue(gain)
        self.gain_spin.blockSignals(False)

    def set_auto_scale(self, enabled: bool):
        self.auto_scale_checkbox.blockSignals(True)
        self.auto_scale_checkbox.setChecked(enabled)
        self.auto_scale_checkbox.blockSignals(False)

    def set_animation_playing(self, playing: bool):
        self.animation_btn.setText("⏸ Pause" if playing else "▶ Play")


class TraceViewWidget(QWidget):
    """
    Scrollable trace display for neural data.

    The trace view itself contains only the plot area. The control panel
    is a separate widget that can be embedded or docked separately.
    """

    timeWindowChanged = pyqtSignal(float, float)

    def __init__(self, engine: TraceEngine, parent: QWidget | None = None):
        super().__init__(parent)
        self.engine = engine

        self.setMinimumSize(600, 400)
        self.setMouseTracking(True)

        self.start_time = 0.0
        self.window_duration = 10.0
        self._zoom_level = 1.0
        self._channel_offset = 0.0
        self.spectrogram_auto_db = True

        self._last_view = {
            'start_time': 0.0,
            'window_duration': 10.0,
            'zoom_level': 1.0,
            'channel_offset': 0.0,
        }

        self.optimal_points_per_channel = 2000
        self.max_total_points = 300000
        self._trace_path_cache = {}
        self._trace_cache_key = None

        self._time_cursors: list[int] = []
        self._selected_cursor: int | None = None
        self._dragging_cursor: bool = False
        self._drag_cursor_start_x = 0.0

        self._dragging = False
        self._drag_start_pos = None
        self._drag_start_time = 0.0
        self._drag_start_offset = 0.0

        self.channels: list[int] = []
        self._sorted_channels: list[int] = []

        # ---- Display-selection geometry ----
        # Depth/shank/x for the channels the user has SELECTED for
        # display. Used for label-strip sorting and lane positioning.
        # NOT used for CSD -- see full_probe_* below.
        self.channel_depths: dict[int, float] = {}
        self.channel_shanks: dict[int, int] = {}
        self.channel_xcoords: dict[int, float] = {}

        # ---- FULL probe geometry ----
        # Depth/shank/x for EVERY channel on the loaded probe, not just
        # the ones currently selected for display. CSD neighbor lookup
        # uses ONLY these maps, so a channel's Laplacian always uses
        # its physically-adjacent same-shank neighbors -- even if those
        # neighbors aren't currently being drawn. A CSD computed from
        # only the displayed subset would silently use whatever
        # channels happened to be selected as "neighbors", which is
        # almost never what the user means.
        self.full_probe_depths: dict[int, float] = {}
        self.full_probe_shanks: dict[int, int] = {}
        self.full_probe_xcoords: dict[int, float] = {}

        self._overscroll_tolerance = 0.2

        self._use_timestamps = False

        self.default_trace_color = QColor("#ffffff")
        self.background_color = QColor("#1e1e1e")
        self.grid_color = QColor("#555555")
        self.show_grid = True
        self.trace_width = 1.0

        self._time_cursors: list[float] = []

        self.channel_colors: dict[int, QColor] = {}

        # Per-row button hitboxes, refreshed every paint.
        self._row_button_rects: dict[int, QRectF] = {}
        self._hovered_row_button_channel: int | None = None

        self.global_gain = 1.0
        self.auto_scale = False

        # GLOBAL filter settings (Trace Controls panel).
        self.filter_enabled = False
        self.filter_low_freq = 1.0
        self.filter_high_freq = 300.0
        self.filter_order = 4
        self.notch_enabled = False
        self.notch_freq = 50.0
        self.notch_q = 30.0
        self.detrend_enabled = False

        self._filtered_cache = {}

        # PER-CHANNEL display overrides.
        self.channel_display_modes: dict[int, dict[str, bool]] = {}

        # CSD neighbor table (per-probe, lazily built).
        self._csd_neighbors: dict[int, tuple] = {}
        self._csd_neighbor_table_valid = False

        self.animation_enabled = False
        self.animation_speed = 100
        self._animation_timer = QTimer(self)
        self._animation_timer.timeout.connect(self._on_animation_tick)
        self._is_animating = False

        self.show_channel_labels = True
        self.show_time_axis = True
        self.show_depth_scale = False

        self._cached_data = None
        self._cache_start = None
        self._cache_end = None
        self._cache_start_idx = None
        self._cache_end_idx = None

        # ---- Spectrogram ----
        self.show_spectrogram = False
        self.spectrogram_channel = None
        self.spectrogram_data = None
        self.spectrogram_freqs = None
        self.spectrogram_times = None
        self._spectrogram_display_data = None
        self._spectrogram_display_freqs = None
        self.spectrogram_min_freq = 0.0
        self.spectrogram_max_freq = 40.0
        self.spectrogram_vmin = -100.0
        self.spectrogram_vmax = -40.0
        self.spectrogram_cmap = "viridis"
        self.spectrogram_sr = 100.0
        self.spectrogram_lowpass = 40.0
        self.spectrogram_filter_order = 4
        self.spectrogram_nperseg = 100
        self.spectrogram_noverlap = 90
        self._spectrogram_image = None
        self._spectrogram_cache_key = None

        self._spectrogram_update_timer = QTimer(self)
        self._spectrogram_update_timer.setSingleShot(True)
        self._spectrogram_update_timer.setInterval(150)
        self._spectrogram_update_timer.timeout.connect(self._compute_spectrogram)

        self.spectrogram_control = SpectrogramControlPanel()
        self._connect_spectrogram_controls()

        self.control_panel = TraceControlPanel()
        self._connect_control_panel()

        self._data_loaded = False

        self._setup_ui()

        if engine.data_loaded:
            self._setup_from_engine()

    # ------------------------------------------------------------------
    # UI scaffolding
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(0)

        layout.addWidget(plot_container, stretch=1)

        self._build_scrollbar(layout)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def _connect_control_panel(self):
        self.control_panel.timeChanged.connect(self._on_control_time_changed)
        self.control_panel.durationChanged.connect(self._on_control_duration_changed)
        self.control_panel.gainChanged.connect(self._on_control_gain_changed)
        self.control_panel.autoScaleChanged.connect(self._on_control_auto_scale_changed)
        self.control_panel.animationToggled.connect(self._on_control_animation_toggled)
        self.control_panel.animationSpeedChanged.connect(self._on_control_animation_speed_changed)
        self.control_panel.filterChanged.connect(self._on_control_filter_changed)
        self.control_panel.zoomChanged.connect(self._on_control_zoom_changed)
        self.control_panel.depthScaleChanged.connect(self._on_control_depth_scale_changed)
        self.control_panel.clearCursorsRequested.connect(self._clear_time_cursors)
        self.control_panel.useTimestampsChanged.connect(self._on_use_timestamps_changed)
        self.control_panel.goToStartRequested.connect(self._go_to_start)
        self.control_panel.goToEndRequested.connect(self._go_to_end)
        self.control_panel.stepTimeRequested.connect(self._step_time)
        self.control_panel.performanceChanged.connect(self._on_performance_changed)


    def _build_scrollbar(self, layout: QVBoxLayout):
        self.scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        self.scrollbar.setRange(0, 1000)
        self.scrollbar.setFixedHeight(15)
        self.scrollbar.setStyleSheet("""
            QScrollBar:horizontal { background: #2d2d2d; height: 15px; margin: 0px; }
            QScrollBar::handle:horizontal { background: #4a9eff; min-width: 20px; border-radius: 7px; margin: 2px; }
            QScrollBar::handle:horizontal:hover { background: #5aaeff; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: #1e1e1e; }
        """)
        self.scrollbar.valueChanged.connect(self._on_scrollbar_changed)
        layout.addWidget(self.scrollbar)
        layout.setAlignment(self.scrollbar, Qt.AlignmentFlag.AlignBottom)

    def _setup_from_engine(self):
        if self.engine.data_loaded:
            self._data_loaded = True

            if self.engine.timestamps_loaded and self.engine.timestamps is not None:
                time_start, time_end = self.engine.get_time_range()
                self.start_time = time_start
                self.window_duration = min(10.0, time_end - time_start)
            else:
                self.start_time = 0.0
                self.window_duration = min(self.engine.total_duration, max(self.window_duration, 1.0))

            self._update_scrollbar_range()
            self._update_time_labels()
            self.update()

    # ------------------------------------------------------------------
    # Control panel handlers
    # ------------------------------------------------------------------

    def _on_performance_changed(self, settings: dict):
        self.optimal_points_per_channel = int(settings.get('optimal_points', 2000))
        self.max_total_points = int(settings.get('max_total_points', 300000))
        self._trace_cache_key = None
        self._trace_path_cache.clear()
        self.update()

    def _on_use_timestamps_changed(self, enabled: bool):
        self._use_timestamps = enabled
        self._invalidate_cache()
        self._update_time_labels()
        self.update()

    def _on_control_time_changed(self, start_time: float, duration: float):
        if not self._data_loaded:
            return
        if start_time < 0:
            max_start = max(0.0, self.engine.total_duration - self.window_duration)
            self.start_time = max_start
        elif duration != self.window_duration:
            self.window_duration = max(0.1, duration)
        else:
            self.start_time = max(0.0, start_time)
        self._update_time_labels()
        self._invalidate_cache()
        self.update()

    def _on_control_duration_changed(self, duration: float):
        self.window_duration = max(0.1, duration)
        self._update_scrollbar_range()
        self._update_time_labels()
        self._invalidate_cache()
        self.update()

    def _on_control_gain_changed(self, gain: float):
        self.global_gain = max(0.1, gain)
        self._invalidate_cache()
        self.update()

    def _on_control_auto_scale_changed(self, enabled: bool):
        self.auto_scale = enabled
        self._invalidate_cache()
        self.update()

    def _on_control_animation_toggled(self, enabled: bool):
        self._on_animation_clicked()

    def _on_control_animation_speed_changed(self, speed_ms: int):
        self.set_animation_speed(speed_ms)

    def _on_control_filter_changed(self, settings: dict):
        self.filter_enabled = settings.get('bandpass_enabled', False)
        self.filter_low_freq = settings.get('low_freq', 1.0)
        self.filter_high_freq = settings.get('high_freq', 300.0)
        self.notch_enabled = settings.get('notch_enabled', False)
        self.notch_freq = settings.get('notch_freq', 50.0)
        self.detrend_enabled = settings.get('detrend', False)
        self._filtered_cache.clear()
        self._invalidate_cache()
        self.update()

    def _on_control_zoom_changed(self, factor: float):
        self._zoom(factor)

    def _on_control_depth_scale_changed(self, enabled: bool):
        self.show_depth_scale = enabled
        self.update()

    # ------------------------------------------------------------------
    # Spectrogram
    # ------------------------------------------------------------------

    def _schedule_spectrogram_update(self):
        if not self.show_spectrogram or self.spectrogram_channel is None:
            return
        self._spectrogram_update_timer.start()

    def _get_spectrogram_height(self):
        if not self.show_spectrogram:
            return 0
        return max(120, int(self.height() * 0.25))

    def _on_spectrogram_min_db_changed(self, value: float):
        self.spectrogram_vmin = value
        if self.show_spectrogram and self.spectrogram_data is not None:
            self._render_spectrogram_image()
        self.update()

    def _on_spectrogram_max_db_changed(self, value: float):
        self.spectrogram_vmax = value
        if self.show_spectrogram and self.spectrogram_data is not None:
            self._render_spectrogram_image()
        self.update()

    def _on_spectrogram_auto_db_changed(self, enabled: bool):
        self.spectrogram_auto_db = enabled
        if self.show_spectrogram and self.spectrogram_data is not None:
            self._render_spectrogram_image()
        self.update()

    def _connect_spectrogram_controls(self):
        self.spectrogram_control.enabledChanged.connect(self._on_spectrogram_enabled_changed)
        self.spectrogram_control.channelChanged.connect(self._on_spectrogram_channel_changed)
        self.spectrogram_control.minFreqChanged.connect(self._on_spectrogram_min_freq_changed)
        self.spectrogram_control.maxFreqChanged.connect(self._on_spectrogram_max_freq_changed)
        self.spectrogram_control.minDbChanged.connect(self._on_spectrogram_min_db_changed)
        self.spectrogram_control.maxDbChanged.connect(self._on_spectrogram_max_db_changed)
        self.spectrogram_control.cmapChanged.connect(self._on_spectrogram_cmap_changed)
        self.spectrogram_control.autoDbChanged.connect(self._on_spectrogram_auto_db_changed)

    def _on_spectrogram_enabled_changed(self, enabled: bool):
        self.show_spectrogram = bool(enabled)
        if not self.show_spectrogram:
            self._spectrogram_update_timer.stop()
            self._spectrogram_image = None
        else:
            if self.spectrogram_channel is None and self.channels:
                self.spectrogram_channel = self.channels[0]
                self.spectrogram_control.set_channel(self.spectrogram_channel)
            if self.spectrogram_channel is not None:
                self._spectrogram_cache_key = None
                self._compute_spectrogram()
        self.update()

    def _on_spectrogram_channel_changed(self, channel: int):
        self.spectrogram_channel = int(channel)
        self._spectrogram_cache_key = None
        if self.show_spectrogram:
            self._schedule_spectrogram_update()
        self.update()

    def _on_spectrogram_min_freq_changed(self, value: float):
        value = max(0.0, float(value))
        if value >= self.spectrogram_max_freq:
            value = max(0.0, self.spectrogram_max_freq - 1.0)
        self.spectrogram_min_freq = value
        if self.spectrogram_data is not None:
            self._update_spectrogram_frequency_view()
        self.update()

    def _on_spectrogram_max_freq_changed(self, value: float):
        value = min(self.spectrogram_sr / 2.0, float(value))
        if value <= self.spectrogram_min_freq:
            value = self.spectrogram_min_freq + 1.0
        self.spectrogram_max_freq = value
        if self.spectrogram_data is not None:
            self._update_spectrogram_frequency_view()
        self.update()

    def _on_spectrogram_cmap_changed(self, name: str):
        self.spectrogram_cmap = str(name)
        if self.spectrogram_data is not None:
            self._render_spectrogram_image()
        self.update()

    def _compute_spectrogram(self):
        if not self.show_spectrogram:
            return
        if not self.engine.data_loaded:
            print("Spectrogram: engine has no data")
            return
        if self.spectrogram_channel is None:
            print("Spectrogram: no channel selected")
            return

        start_time = float(self.start_time)
        end_time = start_time + float(self.window_duration)
        if end_time <= start_time:
            print("Spectrogram: invalid time range")
            return

        print(f"Spectrogram: CH{self.spectrogram_channel} {start_time:.3f}–{end_time:.3f} s")

        cache_key = (self.spectrogram_channel, round(start_time, 6), round(end_time, 6),
                     self.spectrogram_sr, self.spectrogram_lowpass, self.spectrogram_filter_order,
                     self.spectrogram_nperseg, self.spectrogram_noverlap)
        if self._spectrogram_cache_key == cache_key and self.spectrogram_data is not None:
            return

        try:
            data = self.engine.get_channel_data(self.spectrogram_channel, start_time, end_time)
            data = np.asarray(data, dtype=np.float64)
            print(f"Spectrogram: raw samples = {len(data)}")

            if data.size < 10:
                print("Spectrogram: not enough samples")
                self.spectrogram_data = None
                self._spectrogram_image = None
                self._spectrogram_cache_key = None
                return

            finite = np.isfinite(data)
            if not np.all(finite):
                if not np.any(finite):
                    print("Spectrogram: all samples invalid")
                    return
                replacement = np.median(data[finite])
                data = np.nan_to_num(data, nan=replacement, posinf=replacement, neginf=replacement)

            raw_sr = float(self.engine.sr)
            target_sr = float(self.spectrogram_sr)
            print(f"Spectrogram: {raw_sr:g} Hz -> {target_sr:g} Hz")

            cutoff = min(float(self.spectrogram_lowpass), target_sr * 0.45, raw_sr * 0.45)
            sos = butter(int(self.spectrogram_filter_order), cutoff, btype="lowpass", fs=raw_sr, output="sos")
            try:
                data = sosfiltfilt(sos, data)
            except ValueError:
                print("Spectrogram: filtering skipped because the segment is too short")

            if not np.isclose(raw_sr, target_sr):
                from math import gcd
                raw_sr_int = int(round(raw_sr))
                target_sr_int = int(round(target_sr))
                divisor = gcd(raw_sr_int, target_sr_int)
                up = target_sr_int // divisor
                down = raw_sr_int // divisor
                data = resample_poly(data, up, down)

            print(f"Spectrogram: downsampled samples = {len(data)}")

            nperseg = int(self.spectrogram_nperseg)
            noverlap = int(self.spectrogram_noverlap)
            if len(data) < nperseg:
                nperseg = len(data)
                noverlap = min(noverlap, nperseg - 1)

            if nperseg < 8:
                print("Spectrogram: segment too short")
                return

            f, t, Sxx = spectrogram(data, fs=target_sr, window="hann", nperseg=nperseg,
                                    noverlap=noverlap, detrend="constant", scaling="density", mode="psd")
            print(f"Spectrogram: PSD shape = {Sxx.shape}")

            self.spectrogram_freqs = f
            self.spectrogram_times = t + start_time
            self.spectrogram_data = Sxx
            self._spectrogram_cache_key = cache_key

            self._update_spectrogram_frequency_view()

            print(f"Spectrogram: image = {self._spectrogram_image is not None}")
            if self._spectrogram_image is not None:
                print(f"Spectrogram: image size = {self._spectrogram_image.width()} x {self._spectrogram_image.height()}")
            self.update()

        except Exception as exc:
            print(f"Spectrogram calculation failed: {type(exc).__name__}: {exc}")
            self.spectrogram_data = None
            self.spectrogram_freqs = None
            self.spectrogram_times = None
            self._spectrogram_image = None
            self._spectrogram_cache_key = None
            self.update()

    def _get_spectrogram_rect(self):
        if not self.show_spectrogram:
            return None
        rect = self.rect()
        if hasattr(self, "scrollbar"):
            rect.setBottom(rect.bottom() - self.scrollbar.height())

        plot_left = rect.left() + LABEL_STRIP_WIDTH
        plot_right = rect.right() - 30

        _, _, traces_bottom = self._get_plot_bounds()
        spectrogram_height = self._get_spectrogram_height()

        spectrogram_top = traces_bottom + 4
        spectrogram_bottom = rect.bottom() - 25

        actual_height = spectrogram_bottom - spectrogram_top
        if actual_height < 20:
            return None

        return QRectF(float(plot_left), float(spectrogram_top), float(plot_right - plot_left), float(actual_height))

    def _update_spectrogram_frequency_view(self):
        if self.spectrogram_data is None or self.spectrogram_freqs is None:
            self._spectrogram_image = None
            return

        freqs = np.asarray(self.spectrogram_freqs)
        data = np.asarray(self.spectrogram_data)

        mask = (freqs >= self.spectrogram_min_freq) & (freqs <= self.spectrogram_max_freq)
        if not np.any(mask):
            print("Spectrogram: frequency mask is empty")
            self._spectrogram_image = None
            return

        self._spectrogram_display_freqs = freqs[mask]
        self._spectrogram_display_data = data[mask, :]

        print("Spectrogram display:", self._spectrogram_display_data.shape)
        self._render_spectrogram_image()

    def _render_spectrogram_image(self):
        if self._spectrogram_display_data is None:
            self._spectrogram_image = None
            return

        Sxx_db = 10.0 * np.log10(np.maximum(self._spectrogram_display_data, np.finfo(float).tiny))

        if self.spectrogram_auto_db:
            vmin = np.percentile(Sxx_db, 5)
            vmax = np.percentile(Sxx_db, 99)
            if vmax <= vmin:
                vmax = vmin + 1.0
            self.spectrogram_vmin = vmin
            self.spectrogram_vmax = vmax
            self.spectrogram_control.min_db_spin.blockSignals(True)
            self.spectrogram_control.max_db_spin.blockSignals(True)
            self.spectrogram_control.min_db_spin.setValue(vmin)
            self.spectrogram_control.max_db_spin.setValue(vmax)
            self.spectrogram_control.min_db_spin.blockSignals(False)
            self.spectrogram_control.max_db_spin.blockSignals(False)
        else:
            vmin = self.spectrogram_vmin
            vmax = self.spectrogram_vmax
            if vmax <= vmin:
                vmax = vmin + 1.0

        normalized = np.clip((Sxx_db - vmin) / (vmax - vmin), 0.0, 1.0)
        normalized = np.flipud(normalized)

        cmap = colormaps.get_cmap(self.spectrogram_cmap)
        rgba = cmap(normalized)
        rgb = (rgba[..., :3] * 255).astype(np.uint8)
        rgb = np.ascontiguousarray(rgb)

        height, width, _ = rgb.shape
        self._spectrogram_image = QImage(rgb.data, width, height, width * 3, QImage.Format.Format_RGB888).copy()

    def _draw_spectrogram(self, painter: QPainter):
        if not self.show_spectrogram:
            return
        spectrogram_rect = self._get_spectrogram_rect()
        if spectrogram_rect is None:
            return

        painter.save()
        painter.fillRect(spectrogram_rect, self.background_color)

        if self._spectrogram_image is not None:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            painter.drawImage(spectrogram_rect, self._spectrogram_image)
        else:
            painter.setPen(self.default_trace_color)
            painter.drawText(spectrogram_rect, Qt.AlignmentFlag.AlignCenter, "Spectrogram: no data")
        painter.restore()

        painter.save()
        painter.setPen(QPen(self.grid_color, 1.0))
        painter.drawRect(spectrogram_rect)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setFont(QFont("Arial", 8))

        fmin = float(self.spectrogram_min_freq)
        fmax = float(self.spectrogram_max_freq)
        tick_values = []

        if fmax > fmin:
            import math
            target_count = 5
            raw_interval = (fmax - fmin) / target_count
            magnitude = 10 ** math.floor(math.log10(raw_interval)) if raw_interval > 0 else 1
            freq_interval = magnitude
            for multiplier in (1, 2, 5, 10):
                if multiplier * magnitude >= raw_interval:
                    freq_interval = multiplier * magnitude
                    break
            first_tick = math.ceil(fmin / freq_interval) * freq_interval
            tick_values.append(fmin)
            t = first_tick
            while t <= fmax + 1e-9:
                if abs(t - fmin) > freq_interval * 0.01 and abs(t - fmax) > freq_interval * 0.01:
                    tick_values.append(t)
                t += freq_interval
            tick_values.append(fmax)
        else:
            tick_values = [fmin]

        painter.setPen(QPen(QColor(255, 255, 255, 25), 1))
        for freq in tick_values:
            y_ratio = (freq - fmin) / (fmax - fmin) if fmax > fmin else 0.5
            y_pixel = spectrogram_rect.bottom() - y_ratio * spectrogram_rect.height()
            painter.drawLine(int(spectrogram_rect.left()), int(y_pixel), int(spectrogram_rect.right()), int(y_pixel))

        metrics = painter.fontMetrics()
        text_height = metrics.height()
        pad_x, pad_y = 4, 2

        for freq in tick_values:
            text = f"{freq:g} Hz"
            text_width = metrics.horizontalAdvance(text)
            y_ratio = (freq - fmin) / (fmax - fmin) if fmax > fmin else 0.5
            y_pixel = spectrogram_rect.bottom() - y_ratio * spectrogram_rect.height()
            if y_pixel < spectrogram_rect.top() + text_height:
                y_pixel = spectrogram_rect.top() + text_height
            if y_pixel > spectrogram_rect.bottom() - 2:
                y_pixel = spectrogram_rect.bottom() - 2
            label_x = spectrogram_rect.left() + 5
            box = QRectF(label_x - pad_x, y_pixel - text_height + 1, text_width + pad_x * 2, text_height + pad_y)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, 140))
            painter.drawRoundedRect(box, 3, 3)
            painter.setPen(self.default_trace_color)
            painter.drawText(int(label_x), int(y_pixel), text)

        if self.spectrogram_channel is not None:
            text = f"CH{self.spectrogram_channel}"
            text_width = metrics.horizontalAdvance(text)
            label_x = spectrogram_rect.right() - 5 - text_width
            y_pixel = spectrogram_rect.top() + text_height
            box = QRectF(label_x - pad_x, y_pixel - text_height + 1, text_width + pad_x * 2, text_height + pad_y)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, 140))
            painter.drawRoundedRect(box, 3, 3)
            painter.setPen(self.default_trace_color)
            painter.drawText(int(label_x), int(y_pixel), text)

        painter.restore()

    def set_spectrogram_enabled(self, enabled: bool):
        enabled = bool(enabled)
        if self.show_spectrogram == enabled:
            self.spectrogram_control.set_enabled(enabled)
            return
        self.show_spectrogram = enabled
        self.spectrogram_control.set_enabled(enabled)
        if enabled:
            if self.spectrogram_channel is None and self.channels:
                self.spectrogram_channel = self.channels[0]
                self.spectrogram_control.set_channel(self.spectrogram_channel)
            if self.spectrogram_channel is not None:
                self._schedule_spectrogram_update()
        else:
            self._spectrogram_update_timer.stop()
            self._spectrogram_image = None
        self.update()

    def set_spectrogram_channel(self, channel: int):
        if channel is None:
            return
        channel = int(channel)
        if self.spectrogram_channel == channel:
            self.spectrogram_control.set_channel(channel)
            return
        self.spectrogram_channel = channel
        self.spectrogram_control.set_channel(channel)
        self._spectrogram_cache_key = None
        if self.show_spectrogram:
            self._schedule_spectrogram_update()
        self.update()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _get_min_start_time(self) -> float:
        if self.engine.timestamps_loaded and self.engine.timestamps is not None:
            return self.engine.timestamp_start
        return 0.0

    def _get_max_start_time(self) -> float:
        if self.engine.timestamps_loaded and self.engine.timestamps is not None:
            return self.engine.timestamp_end - self.window_duration
        return max(0.0, self.engine.total_duration - self.window_duration)

    def _get_total_duration(self) -> float:
        if self.engine.timestamps_loaded and self.engine.timestamps is not None:
            return self.engine.timestamp_duration
        return self.engine.total_duration

    def _go_to_start(self):
        if not self._data_loaded:
            return
        self.start_time = self._get_min_start_time()
        self._update_time_labels()
        self._invalidate_cache()
        self.update()
        self.control_panel.set_time_display(self.start_time, self.start_time + self.window_duration)

    def _go_to_end(self):
        if not self._data_loaded:
            return
        self.start_time = self._get_max_start_time()
        self._update_time_labels()
        self._invalidate_cache()
        self.update()
        self.control_panel.set_time_display(self.start_time, self.start_time + self.window_duration)

    def _step_time(self, direction: int):
        if not self._data_loaded:
            return
        time_delta = 0.1 * self.window_duration * direction
        min_start = self._get_min_start_time()
        max_start = self._get_max_start_time()
        self.start_time = min(max(min_start, self.start_time + time_delta), max_start)
        self._update_time_labels()
        self._invalidate_cache()
        self.update()
        self.control_panel.set_time_display(self.start_time, self.start_time + self.window_duration)

    def _update_time_labels(self):
        if self._data_loaded:
            if self._use_timestamps and self.engine.timestamps_loaded:
                start_idx = int(np.searchsorted(self.engine.timestamps, self.start_time, side='left'))
                end_idx = int(np.searchsorted(self.engine.timestamps, self.start_time + self.window_duration, side='right'))
                start_idx = max(0, min(start_idx, len(self.engine.timestamps) - 1))
                end_idx = max(0, min(end_idx, len(self.engine.timestamps) - 1))
                self.control_panel.set_time_display(self.engine.timestamps[start_idx], self.engine.timestamps[end_idx])
                return
            end_time = min(self.start_time + self.window_duration, self.engine.total_duration)
            self.control_panel.set_time_display(self.start_time, end_time)

    def _reset_view(self):
        if not self._data_loaded:
            return
        self.start_time = self._get_min_start_time()
        self.window_duration = min(10.0, self._get_total_duration())
        self._zoom_level = 1.0
        self._channel_offset = 0.0
        self.control_panel.zoom_label.setText("100%")
        self.control_panel.duration_spin.blockSignals(True)
        self.control_panel.duration_spin.setValue(self.window_duration)
        self.control_panel.duration_spin.blockSignals(False)
        self._update_time_labels()
        self._invalidate_cache()
        self.update()

    def _scroll_up(self):
        if not self._data_loaded or not self.channels:
            return

    def _scroll_down(self):
        if not self._data_loaded or not self.channels:
            return

    # ------------------------------------------------------------------
    # Channels / depths / shanks / x-coords
    # ------------------------------------------------------------------

    def set_channel_depths(self, depths: dict[int, float]):
        self.channel_depths = depths
        self._sort_channels_by_depth()
        self._csd_neighbor_table_valid = False
        self._invalidate_cache()
        self.update()

    def set_channel_shanks(self, shank_map: dict[int, int]):
        """Set the shank id for each channel. Needed for CSD neighbor
        lookup -- CSD is always within a shank, never across."""
        self.channel_shanks = dict(shank_map)
        self._csd_neighbor_table_valid = False
        self._invalidate_cache()

    def set_channel_xcoords(self, x_map: dict[int, float]):
        """Set the x-coordinate for each channel. Used by CSD neighbor
        selection to break ties among same-depth candidates: on some
        probes two electrodes can sit at the same y on the same shank
        (different x). The Laplacian only needs ONE channel per depth
        level, so we pick the one laterally closest to the center."""
        self.channel_xcoords = dict(x_map)
        self._csd_neighbor_table_valid = False
        self._invalidate_cache()

    def set_full_probe_geometry(self,
                                 depths: dict[int, float],
                                 shanks: dict[int, int],
                                 xcoords: dict[int, float]):
        """Set the FULL probe geometry (every channel on the probe, not
        just the ones currently selected for display). CSD neighbor
        lookup uses ONLY this geometry, so the Laplacian always uses
        each channel's physically-adjacent same-shank neighbors -- even
        if those neighbors aren't currently being drawn."""
        self.full_probe_depths = dict(depths)
        self.full_probe_shanks = dict(shanks)
        self.full_probe_xcoords = dict(xcoords)
        self._csd_neighbor_table_valid = False
        self._invalidate_cache()

    def _xcoord_of(self, channel: int) -> float | None:
        """Return the x-coordinate for a channel, or None if unknown."""
        return self.channel_xcoords.get(channel)

    def _sort_channels_by_depth(self):
        if self.channel_depths:
            self._sorted_channels = sorted(self.channels, key=lambda ch: self.channel_depths.get(ch, 0.0))
        else:
            self._sorted_channels = list(self.channels)

    def get_display_order(self) -> list[int]:
        return self._sorted_channels if self._sorted_channels else list(self.channels)

    # ------------------------------------------------------------------
    # Per-channel display modes
    # ------------------------------------------------------------------

    def _has_channel_override(self, channel: int) -> bool:
        return channel in self.channel_display_modes

    def _modes_for(self, channel: int) -> dict[str, bool]:
        return self.channel_display_modes.get(channel, {'raw': True, 'filtered': False, 'csd': False})

    def _ensure_csd_neighbor_table(self):
        """Build the channel -> (above, below, dist_above, dist_below)
        table once per probe, from the FULL probe geometry.

        Rules that matter for correctness:
          - Neighbors are looked up in full_probe_*, NOT in
            channel_depths. channel_depths only contains the channels
            the user has SELECTED for display; using it would make a
            channel's "neighbor" be whatever displayed channel happens
            to be next in y, which is almost never the physically
            adjacent electrode.
          - Neighbors are ALWAYS within the same shank. Never across.
          - A neighbor must be at a STRICTLY DIFFERENT depth (y). Two
            electrodes at the same y on the same shank (different x)
            are LATERAL neighbors and must NOT be used as the above
            or below term of a depth Laplacian.
          - Among the strictly-different-depth candidates, "above" is
            the next distinct y level above; "below" is the next
            distinct y level below.
          - When a depth level contains multiple channels (same-y
            siblings), the one laterally closest in x to the center
            channel is chosen, minimizing lateral offset in the
            estimator.
          - A neighbor must be within MAX_CSD_NEIGHBOR_GAP_UM; beyond
            that the Laplacian is meaningless and the neighbor is
            dropped.
          - The topmost and bottommost channel of each shank (after
            skipping same-y siblings) get (None, below) and
            (above, None) respectively, and can't produce a CSD.
        """
        if self._csd_neighbor_table_valid:
            return

        self._csd_neighbors = {}

        depths = self.full_probe_depths
        shanks = self.full_probe_shanks
        xcoords = self.full_probe_xcoords

        if not depths:
            self._csd_neighbor_table_valid = True
            return

        # The shank map MUST cover every channel that has depth info.
        missing_shanks = [ch for ch in depths if ch not in shanks]
        if missing_shanks:
            if not getattr(self, '_csd_warned_shank_map', False):
                print(
                    f"[CSD] full_probe_shanks is missing {len(missing_shanks)} "
                    f"channel(s) that have depth info -- refusing to build "
                    f"a neighbor table from incomplete geometry. "
                    f"First few missing: {missing_shanks[:5]}. "
                    f"Call set_full_probe_geometry() with the FULL probe map."
                )
                self._csd_warned_shank_map = True
            self._csd_neighbor_table_valid = True
            return

        # Group channels by shank.
        by_shank: dict[int, list[tuple[int, float]]] = {}
        for ch, y in depths.items():
            s = int(shanks.get(ch, 0))
            by_shank.setdefault(s, []).append((ch, float(y)))

        for shank, items in by_shank.items():
            # Sort by depth ascending (y increases toward the brain
            # surface in this codebase).
            items.sort(key=lambda cv: cv[1])

            # Distinct y-values on this shank, sorted ascending. We
            # index into this list -- not into `items` -- so same-y
            # siblings collapse to a single depth level.
            distinct_y = sorted(set(y for _, y in items))

            # Channels grouped by depth level.
            chans_at_y: dict[float, list[int]] = {}
            for ch, y in items:
                chans_at_y.setdefault(y, []).append(ch)

            for ch, y in items:
                above_ch = above_dist = None
                below_ch = below_dist = None

                try:
                    y_idx = distinct_y.index(y)
                except ValueError:
                    self._csd_neighbors[ch] = (None, None, None, None)
                    continue

                # ---- above: next distinct depth level up ----
                if y_idx + 1 < len(distinct_y):
                    cand_y = distinct_y[y_idx + 1]
                    gap = cand_y - y
                    if 0 < gap <= MAX_CSD_NEIGHBOR_GAP_UM:
                        candidates = chans_at_y[cand_y]
                        if len(candidates) == 1:
                            above_ch = candidates[0]
                        else:
                            center_x = xcoords.get(ch)
                            if center_x is None:
                                above_ch = candidates[0]
                            else:
                                above_ch = min(
                                    candidates,
                                    key=lambda c: abs((xcoords.get(c, center_x)) - center_x),
                                )
                        above_dist = gap

                # ---- below: next distinct depth level down ----
                if y_idx - 1 >= 0:
                    cand_y = distinct_y[y_idx - 1]
                    gap = y - cand_y
                    if 0 < gap <= MAX_CSD_NEIGHBOR_GAP_UM:
                        candidates = chans_at_y[cand_y]
                        if len(candidates) == 1:
                            below_ch = candidates[0]
                        else:
                            center_x = xcoords.get(ch)
                            if center_x is None:
                                below_ch = candidates[0]
                            else:
                                below_ch = min(
                                    candidates,
                                    key=lambda c: abs((xcoords.get(c, center_x)) - center_x),
                                )
                        below_dist = gap

                self._csd_neighbors[ch] = (above_ch, below_ch, above_dist, below_dist)

        self._csd_neighbor_table_valid = True

    def _compute_csd_for_channel(self, channel: int, basis: np.ndarray) -> np.ndarray | None:
        """3-point Laplacian CSD for `channel`, computed from RAW
        traces of the channel and its two nearest same-shank
        neighbors. `basis` is the raw trace of `channel` itself, so we
        don't re-fetch it. Never uses the global filter -- CSD is one
        of the three alternative signals.

        Every failure path prints a one-line diagnostic (once per
        unique reason) rather than silently returning None -- the
        most common reason CSD "doesn't plot" is that
        set_channel_shanks() was never called on this view, so the
        neighbor table can't be built correctly.
        """
        self._ensure_csd_neighbor_table()

        pair = self._csd_neighbors.get(channel)
        if pair is None:
            if not getattr(self, '_csd_warned_no_table', False):
                print(
                    f"[CSD] No neighbor-table entry for channel {channel}. "
                    f"channel_depths has {len(self.channel_depths)} entries, "
                    f"channel_shanks has {len(self.channel_shanks)} entries. "
                    f"Both must cover the FULL probe (not just selected channels), "
                    f"and set_channel_shanks() must have been called -- "
                    f"see MainWindow._on_channels_selected."
                )
                self._csd_warned_no_table = True
            return None

        above_ch, below_ch, above_dist, below_dist = pair
        if above_ch is None or below_ch is None:
            # Edge channel, or a channel whose neighbor gap exceeded
            # the sanity threshold. A 3-point Laplacian is undefined
            # without both terms, so CSD simply cannot be computed
            # here. This is normal and expected for the topmost and
            # bottommost channel of every shank, and for any channel
            # bordering a gap in the probe.
            return None

        end_time = self.start_time + self.window_duration
        try:
            above_raw = self.engine.get_channel_data(above_ch, self.start_time, end_time)
            below_raw = self.engine.get_channel_data(below_ch, self.start_time, end_time)
        except Exception as e:
            print(f"[CSD] CH{channel}: neighbor fetch failed: {e}")
            return None

        n = len(basis)
        if len(above_raw) != n or len(below_raw) != n:
            print(
                f"[CSD] CH{channel}: length mismatch basis={n} "
                f"above={len(above_raw)} below={len(below_raw)}"
            )
            return None

        spacing_um = (above_dist + below_dist) / 2.0
        if spacing_um <= 0:
            print(f"[CSD] CH{channel}: non-positive spacing ({spacing_um})")
            return None

        csd = (above_raw.astype(np.float64)
               - 2.0 * basis.astype(np.float64)
               + below_raw.astype(np.float64)) / (spacing_um ** 2)
        return csd

    def _show_channel_mode_menu(self, channel: int, global_pos):
        """Popup menu for one channel: pick which of {raw, filtered,
        csd} to draw. Checking any of them overrides the global
        pipeline for this channel."""
        is_override = self._has_channel_override(channel)
        modes = self.channel_display_modes.get(channel, {'raw': False, 'filtered': False, 'csd': False})

        # Determine whether this channel can actually compute a CSD.
        self._ensure_csd_neighbor_table()
        csd_pair = self._csd_neighbors.get(channel)
        csd_available = (csd_pair is not None and csd_pair[0] is not None and csd_pair[1] is not None)

        menu = QMenu(self)
        header = menu.addAction(f"CH{channel} — per-channel display")
        header.setEnabled(False)
        menu.addSeparator()

        act_raw = menu.addAction("Raw")
        act_raw.setCheckable(True)
        act_raw.setChecked(modes['raw'])

        act_filt = menu.addAction("Filtered (current Trace Controls filter)")
        act_filt.setCheckable(True)
        act_filt.setChecked(modes['filtered'])

        act_csd = menu.addAction("CSD (Laplacian with nearest neighbors)")
        act_csd.setCheckable(True)
        act_csd.setChecked(modes['csd'])
        if not csd_available:
            act_csd.setEnabled(False)
            if csd_pair is None:
                reason = "no depth/shank info available for this channel"
            elif csd_pair[0] is None and csd_pair[1] is None:
                reason = "no same-shank neighbors above or below"
            elif csd_pair[0] is None:
                reason = "no same-shank neighbor above (edge of shank)"
            else:
                reason = "no same-shank neighbor below (edge of shank)"
            act_csd.setToolTip(
                f"CSD is not available for CH{channel}: {reason}. "
                "A 3-point Laplacian requires both an above and below "
                "neighbor at a different depth on the same shank."
            )

        menu.addSeparator()
        act_copy = menu.addAction("Apply this to all selected channels")
        act_reset = menu.addAction("Reset this channel to global default")

        chosen = menu.exec(global_pos)
        if chosen is None:
            return

        if chosen is act_copy:
            for ch in self.channels:
                self.channel_display_modes[ch] = dict(modes)
            self._invalidate_cache()
            self.update()
            return

        if chosen is act_reset:
            self.channel_display_modes.pop(channel, None)
            self._invalidate_cache()
            self.update()
            return

        new_modes = {
            'raw': act_raw.isChecked(),
            'filtered': act_filt.isChecked(),
            'csd': act_csd.isChecked(),
        }
        if not any(new_modes.values()):
            self.channel_display_modes.pop(channel, None)
        else:
            self.channel_display_modes[channel] = new_modes

        self._invalidate_cache()
        self.update()

    def _show_bulk_mode_menu(self, global_pos):
        """Top-of-strip button: bulk actions for per-channel modes."""
        n_override = len(self.channel_display_modes)
        menu = QMenu(self)
        header = menu.addAction(
            f"Per-channel display — {n_override} channel(s) overridden"
        )
        header.setEnabled(False)
        menu.addSeparator()

        act_all_raw = menu.addAction("All channels → Raw")
        act_all_filtered = menu.addAction("All channels → Filtered")
        act_all_csd = menu.addAction("All channels → CSD")
        act_all_raw_filt = menu.addAction("All channels → Raw + Filtered")
        menu.addSeparator()
        act_reset_all = menu.addAction("Reset ALL channels to global default")

        chosen = menu.exec(global_pos)
        if chosen is None:
            return

        if chosen is act_reset_all:
            self.channel_display_modes.clear()
        elif chosen is act_all_raw:
            for ch in self.channels:
                self.channel_display_modes[ch] = {'raw': True, 'filtered': False, 'csd': False}
        elif chosen is act_all_filtered:
            for ch in self.channels:
                self.channel_display_modes[ch] = {'raw': False, 'filtered': True, 'csd': False}
        elif chosen is act_all_csd:
            for ch in self.channels:
                self.channel_display_modes[ch] = {'raw': False, 'filtered': False, 'csd': True}
        elif chosen is act_all_raw_filt:
            for ch in self.channels:
                self.channel_display_modes[ch] = {'raw': True, 'filtered': True, 'csd': False}

        self._invalidate_cache()
        self.update()

    # ------------------------------------------------------------------
    # Filter methods
    # ------------------------------------------------------------------

    def _apply_filters(self, data: np.ndarray) -> np.ndarray:
        """Apply all enabled global filters."""
        if not self.filter_enabled and not self.notch_enabled and not self.detrend_enabled:
            return data

        filtered_data = data.copy()

        if self.detrend_enabled:
            from scipy.signal import detrend as scipy_detrend
            filtered_data = scipy_detrend(filtered_data, axis=0)

        if self.filter_enabled:
            nyquist = self.engine.sr / 2
            low_freq = max(0.0, min(self.filter_low_freq, nyquist - 1))
            high_freq = max(0.0, min(self.filter_high_freq, nyquist - 1))
            if low_freq > 0 and high_freq > 0:
                if low_freq >= high_freq:
                    low_freq, high_freq = min(low_freq, high_freq), max(low_freq, high_freq)
                filtered_data = bandpass_filter(filtered_data, self.engine.sr, low_freq, high_freq, order=self.filter_order)
            elif high_freq > 0:
                filtered_data = bandpass_filter(filtered_data, self.engine.sr, 0.0, high_freq, order=self.filter_order)
            elif low_freq > 0:
                filtered_data = bandpass_filter(filtered_data, self.engine.sr, low_freq, 0.0, order=self.filter_order)
            else:
                return filtered_data

        if self.notch_enabled:
            filtered_data = notch_filter(filtered_data, self.engine.sr, self.notch_freq, quality_factor=self.notch_q)

        return filtered_data

    # ------------------------------------------------------------------
    # Public color / style API
    # ------------------------------------------------------------------
    def _trace_color_for(self, channel: int, mode: str) -> QColor:
        """Return the pen color for a given channel/mode pair."""
        base = self.channel_colors.get(channel, self.default_trace_color)
        if mode == 'raw':
            return base
        if mode == 'filtered':
            return QColor(base.red(), int(base.green() * 0.85), int(base.blue() * 0.85))
        if mode == 'csd':
            return QColor(int(base.red() * 0.6), int(base.green() * 0.9), 255)
        return base
        
    def set_trace_color(self, color: str | QColor, channel: int | None = None):
        qcolor = QColor(color) if isinstance(color, str) else color
        if not qcolor.isValid():
            return
        if channel is None:
            self.default_trace_color = qcolor
            self.channel_colors.clear()
        else:
            self.channel_colors[channel] = qcolor
        self.update()

    def get_trace_color(self, channel: int | None = None) -> str:
        if channel is not None and channel in self.channel_colors:
            return self.channel_colors[channel].name()
        return self.default_trace_color.name()

    def set_background_color(self, color: str | QColor):
        qcolor = QColor(color) if isinstance(color, str) else color
        if qcolor.isValid():
            self.background_color = qcolor
            self.update()

    def get_background_color(self) -> str:
        return self.background_color.name()

    def set_grid_color(self, color: str | QColor):
        qcolor = QColor(color) if isinstance(color, str) else color
        if qcolor.isValid():
            self.grid_color = qcolor
            self.update()

    def get_grid_color(self) -> str:
        return self.grid_color.name()

    def set_grid_visible(self, visible: bool):
        self.show_grid = visible
        self.update()

    def set_trace_width(self, width: float):
        if 0.5 <= width <= 5.0:
            self.trace_width = width
            self.update()

    def set_channel_color(self, channel: int, color: str | QColor):
        self.set_trace_color(color, channel=channel)

    def reset_channel_colors(self):
        self.channel_colors.clear()
        self.update()

    def set_animation_speed(self, speed_ms: int):
        self.animation_speed = max(10, min(speed_ms, 1000))
        if self._animation_timer.isActive():
            self._animation_timer.setInterval(self.animation_speed)
        self.control_panel.animation_speed_spin.setValue(self.animation_speed)

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------

    def _on_animation_clicked(self):
        if not self._data_loaded:
            return
        if self._is_animating:
            self._animation_timer.stop()
            self._is_animating = False
            self.control_panel.set_animation_playing(False)
        else:
            self._animation_timer.start(self.animation_speed)
            self._is_animating = True
            self.control_panel.set_animation_playing(True)

    def _on_animation_tick(self):
        if not self._data_loaded:
            return
        dt = self.animation_speed / 1000.0
        max_start = max(0.0, self.engine.total_duration - self.window_duration)
        self.start_time += dt
        if self.start_time > max_start:
            self.start_time = 0.0
        self._update_scrollbar_position()
        self._update_time_labels()
        self._invalidate_cache()
        self.update()

    # ------------------------------------------------------------------
    # Cursor helpers
    # ------------------------------------------------------------------

    def _cursor_to_time(self, fraction: float) -> float:
        if self.engine.timestamps_loaded and self.engine.timestamps is not None:
            min_time, max_time = self.engine.timestamp_start, self.engine.timestamp_end
        else:
            min_time, max_time = 0.0, self.engine.total_duration
        total_duration = max_time - min_time
        return min_time + fraction * total_duration

    def _time_to_cursor_fraction(self, time: float) -> float:
        if self.engine.timestamps_loaded and self.engine.timestamps is not None:
            min_time, max_time = self.engine.timestamp_start, self.engine.timestamp_end
        else:
            min_time, max_time = 0.0, self.engine.total_duration
        total_duration = max_time - min_time
        if total_duration > 0:
            return (time - min_time) / total_duration
        return 0.0

    def set_channels(self, channels: list[int]):
        """Set which channels to display."""
        self.channels = channels
        self._sort_channels_by_depth()

        if hasattr(self, 'control_panel') and hasattr(self.control_panel, 'channel_info_label'):
            if channels:
                self.control_panel.channel_info_label.setText(f"{len(channels)} channels selected")
            else:
                self.control_panel.channel_info_label.setText("No channels")

        if hasattr(self, 'spectrogram_control'):
            self.spectrogram_control.set_channels(channels)
            if channels:
                self.spectrogram_control.set_channel(channels[0])

        if not channels:
            self._trace_path_cache = {}
            self._trace_cache_key = None

        self._invalidate_cache()
        self.update()

    def set_data_source(self, engine: TraceEngine):
        self.engine = engine
        if engine.data_loaded:
            self._data_loaded = True
            self._setup_from_engine()
            if engine.timestamps_loaded:
                self.control_panel.set_timestamps_available(True)
        else:
            self._data_loaded = False
            self.control_panel.set_timestamps_available(False)
        self._invalidate_cache()
        self.update()

    def _on_scrollbar_changed(self, value: int):
        if self._data_loaded and self.engine.total_duration > 0:
            min_start = self._get_min_start_time()
            max_start = self._get_max_start_time()
            total_range = max_start - min_start
            if total_range > 0:
                self.start_time = min_start + (value / self.scrollbar.maximum()) * total_range
                self._update_time_labels()
                self._invalidate_cache()
                self.update()

    def _update_scrollbar_position(self):
        if self._data_loaded and hasattr(self, 'scrollbar') and self.engine.total_duration > 0:
            min_start = self._get_min_start_time()
            max_start = self._get_max_start_time()
            total_range = max_start - min_start
            if total_range > 0:
                pos = int(((self.start_time - min_start) / total_range) * self.scrollbar.maximum())
                self.scrollbar.blockSignals(True)
                self.scrollbar.setValue(pos)
                self.scrollbar.blockSignals(False)

    def _update_scrollbar_range(self):
        if self._data_loaded and hasattr(self, 'scrollbar'):
            self.scrollbar.setMaximum(1000)
            self._update_scrollbar_position()

    def _zoom_to_cursor(self, factor: float, cursor_pos):
        if not self._data_loaded:
            return
        self._last_view = {
            'start_time': self.start_time,
            'window_duration': self.window_duration,
            'zoom_level': self._zoom_level,
            'channel_offset': self._channel_offset,
        }
        plot_left, plot_right, _ = self._get_plot_bounds()
        plot_width = plot_right - plot_left
        x_ratio = (cursor_pos.x() - plot_left) / max(1, plot_width)
        cursor_time = self.start_time + x_ratio * self.window_duration
        new_duration = self.window_duration / factor
        new_duration = max(0.1, min(new_duration, self._get_total_duration()))
        new_start = cursor_time - x_ratio * new_duration
        min_start = self._get_min_start_time()
        max_start = self._get_max_start_time()
        new_start = max(min_start, min(new_start, max_start))
        self.start_time = new_start
        self.window_duration = new_duration
        self._zoom_level *= factor
        self._zoom_level = max(0.1, min(self._zoom_level, 10.0))
        self.control_panel.zoom_label.setText(f"{int(self._zoom_level * 100)}%")
        self.control_panel.duration_spin.blockSignals(True)
        self.control_panel.duration_spin.setValue(self.window_duration)
        self.control_panel.duration_spin.blockSignals(False)
        self._update_time_labels()
        self._invalidate_cache()
        self.update()

    def _zoom(self, factor: float):
        if not self._data_loaded:
            return
        self._last_view = {
            'start_time': self.start_time,
            'window_duration': self.window_duration,
            'zoom_level': self._zoom_level,
            'channel_offset': self._channel_offset,
        }
        view_center = self.start_time + self.window_duration / 2
        self._zoom_level *= factor
        self._zoom_level = max(0.1, min(self._zoom_level, 10.0))
        self.control_panel.zoom_label.setText(f"{int(self._zoom_level * 100)}%")
        new_duration = self.window_duration / factor
        new_duration = max(0.1, min(new_duration, self._get_total_duration()))
        new_start = view_center - new_duration / 2
        min_start = self._get_min_start_time()
        max_start = self._get_max_start_time()
        new_start = max(min_start, min(new_start, max_start))
        self.start_time = new_start
        self.window_duration = new_duration
        self.control_panel.duration_spin.blockSignals(True)
        self.control_panel.duration_spin.setValue(self.window_duration)
        self.control_panel.duration_spin.blockSignals(False)
        self._update_time_labels()
        self._invalidate_cache()
        self.update()

    def _get_plot_bounds(self):
        """Return the bounds of the neural trace region."""
        rect = self.rect()
        if hasattr(self, "scrollbar"):
            rect.setBottom(rect.bottom() - self.scrollbar.height())

        plot_left = rect.left() + LABEL_STRIP_WIDTH
        plot_right = rect.right() - 30

        time_axis_height = 25
        plot_bottom = rect.bottom() - time_axis_height

        if self.show_spectrogram:
            spectrogram_height = self._get_spectrogram_height()
            plot_bottom -= (spectrogram_height + 8)

        return (plot_left, plot_right, plot_bottom)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _sample_to_time(self, sample: int) -> float:
        if self.engine.timestamps_loaded and self.engine.timestamps is not None:
            sample = max(0, min(sample, len(self.engine.timestamps) - 1))
            return float(self.engine.timestamps[sample])
        return sample / self.engine.sr

    def _time_to_sample(self, time: float) -> int:
        if self.engine.timestamps_loaded and self.engine.timestamps is not None:
            idx = int(np.searchsorted(self.engine.timestamps, time, side='left'))
            return max(0, min(idx, len(self.engine.timestamps) - 1))
        return int(time * self.engine.sr)

    def _invalidate_cache(self):
        self._cached_data = None
        self._cache_start = None
        self._cache_end = None
        self._cache_start_idx = None
        self._cache_end_idx = None
        self._filtered_cache.clear()
        self._spectrogram_cache_key = None
        if self.show_spectrogram and self.spectrogram_channel is not None:
            self._schedule_spectrogram_update()
        self._trace_cache_key = None
        self._trace_path_cache.clear()

    def _get_data_for_display(self):
        """Fetch data for the current window, per channel.

        Returns { channel: [ {'mode': str, 'data': ndarray}, ... ] }.

        Two cases per channel:
          - No per-channel override (default): follow the global
            pipeline. If a global filter is active, the drawn signal
            is the filtered version (mode tag 'global' for coloring
            purposes); otherwise raw (mode tag 'global').
          - Per-channel override: draw ONLY the signals the user
            explicitly listed. Global filter is bypassed.
        """
        if not self._data_loaded or not self.channels:
            return None

        if self._cached_data is not None and self._cache_start is not None:
            if self._cache_start <= self.start_time and self._cache_end >= self.start_time + self.window_duration:
                return self._cached_data

        end_time = self.start_time + self.window_duration
        start_idx, end_idx = self.engine.get_time_window_sample_range(self.start_time, end_time)

        global_filter_active = self.filter_enabled or self.notch_enabled or self.detrend_enabled

        data: dict[int, list[dict]] = {}

        for ch in self.channels:
            try:
                raw = self.engine.get_channel_data(ch, self.start_time, end_time)
            except Exception as e:
                print(f"Error getting channel {ch}: {e}")
                continue
            if raw is None or len(raw) == 0:
                continue

            if self._has_channel_override(ch):
                modes = self.channel_display_modes[ch]
                traces: list[dict] = []

                if modes.get('raw'):
                    traces.append({'mode': 'raw', 'data': raw})

                if modes.get('filtered'):
                    filtered = self._apply_filters(raw) if global_filter_active else raw
                    traces.append({'mode': 'filtered', 'data': filtered})

                if modes.get('csd'):
                    csd = self._compute_csd_for_channel(ch, basis=raw)
                    if csd is not None:
                        traces.append({'mode': 'csd', 'data': csd})
                    # If csd is None, that means this channel has no
                    # valid both-sided neighbor pair. We deliberately
                    # do NOT fall back to raw -- falling back would
                    # make the display ambiguous.

                if not traces:
                    traces.append({'mode': 'raw', 'data': raw})

                data[ch] = traces
            else:
                # Global pipeline: exactly the previous behavior.
                if global_filter_active:
                    filtered = self._apply_filters(raw)
                    data[ch] = [{'mode': 'global', 'data': filtered}]
                else:
                    data[ch] = [{'mode': 'global', 'data': raw}]

        self._cached_data = data
        self._cache_start = self.start_time
        self._cache_end = end_time
        self._cache_start_idx = start_idx
        self._cache_end_idx = end_idx
        return data

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        if hasattr(self, 'scrollbar'):
            rect.setBottom(rect.bottom() - self.scrollbar.height())

        painter.fillRect(rect, self.background_color)

        if not self._data_loaded:
            painter.setPen(QColor("#888888"))
            font = QFont()
            font.setPointSize(14)
            painter.setFont(font)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "No data loaded.")
            return

        if not self.channels:
            painter.setPen(QColor("#888888"))
            font = QFont()
            font.setPointSize(12)
            painter.setFont(font)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Select channels from the probe map to display traces.")
            return

        if self.show_grid:
            self._draw_grid(painter, rect)

        data = self._get_data_for_display()

        self._draw_traces(painter, rect, data)

        if self.show_spectrogram:
            self._draw_spectrogram(painter)

        self._draw_time_cursors(painter, rect)

        if self.show_time_axis:
            self._draw_time_axis(painter, rect)

        if self.show_channel_labels:
            self._draw_channel_labels(painter, rect, data)

        if self.show_depth_scale:
            self._draw_depth_scale(painter, rect, data)

    def _draw_time_cursors(self, painter: QPainter, rect: QRectF):
        if not self._time_cursors:
            return

        plot_left, plot_right, _ = self._get_plot_bounds()
        plot_width = plot_right - plot_left

        self._trash_bin_rects = {}

        for i, cursor_fraction in enumerate(self._time_cursors):
            cursor_time = self._cursor_to_time(cursor_fraction)
            if cursor_time < self.start_time or cursor_time > self.start_time + self.window_duration:
                continue

            x_ratio = (cursor_time - self.start_time) / self.window_duration
            x = plot_left + x_ratio * plot_width

            is_selected = (i == self._selected_cursor)
            if is_selected:
                pen = QPen(QColor("#ff6b6b"), 3, Qt.PenStyle.DashLine)
            else:
                pen = QPen(QColor("#ff6b6b"), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawLine(int(x), int(rect.top()), int(x), int(rect.bottom()))

            painter.setPen(QPen(QColor("#ff6b6b"), 1))
            font = QFont()
            font.setPointSize(8)
            painter.setFont(font)
            painter.drawText(int(x - 25), int(rect.top() + 5), 50, 15, Qt.AlignmentFlag.AlignCenter, f"{cursor_time:.3f}s")

            if is_selected:
                self._draw_trash_bin(painter, x - 12, rect.top())
                self._trash_bin_rects[i] = QRectF(int(x) - 22, int(rect.top()) + 20, 20, 20)

            painter.setPen(pen)

    def _draw_trash_bin(self, painter: QPainter, x: float, top: float):
        bin_x = int(x) - 16
        bin_y = int(top) + 22
        bin_size = 14
        bin_color = QColor("#ff6b6b")
        bin_fill = QColor("#ff6b6b")
        bin_fill.setAlpha(50)
        painter.setPen(QPen(bin_color, 2))
        painter.setBrush(bin_fill)
        painter.drawRoundedRect(bin_x, bin_y + 3, bin_size, bin_size - 3, 2, 2)
        painter.drawLine(bin_x - 1, bin_y + 3, bin_x + bin_size + 1, bin_y + 3)
        painter.drawLine(bin_x + 3, bin_y + 3, bin_x + 3, bin_y + 1)
        painter.drawLine(bin_x + 3, bin_y + 1, bin_x + bin_size - 3, bin_y + 1)
        painter.drawLine(bin_x + bin_size - 3, bin_y + 1, bin_x + bin_size - 3, bin_y + 3)
        painter.setPen(QPen(bin_color, 1))
        painter.drawLine(bin_x + 4, bin_y + 6, bin_x + 4, bin_y + bin_size - 2)
        painter.drawLine(bin_x + bin_size - 4, bin_y + 6, bin_x + bin_size - 4, bin_y + bin_size - 2)

    def _compute_time_ticks(self, rect: QRectF):
        plot_left = rect.left() + LABEL_STRIP_WIDTH
        plot_right = rect.right() - 30
        plot_width = plot_right - plot_left

        if plot_width <= 0 or self.window_duration <= 0:
            return [], []

        start_time = self.start_time
        end_time = self.start_time + self.window_duration

        target_major_count = 6
        raw_interval = self.window_duration / target_major_count

        import math
        magnitude = 10 ** math.floor(math.log10(raw_interval))
        for multiplier in (1, 2, 5, 10):
            major_interval = multiplier * magnitude
            if major_interval >= raw_interval:
                break

        minor_interval = major_interval / 5.0
        first_major = math.ceil(start_time / major_interval) * major_interval
        first_minor = math.ceil(start_time / minor_interval) * minor_interval

        major_ticks = []
        minor_ticks = []

        t = first_major
        while t <= end_time + 1e-9:
            x_ratio = (t - start_time) / self.window_duration
            major_ticks.append((plot_left + x_ratio * plot_width, t))
            t += major_interval

        t = first_minor
        while t <= end_time + 1e-9:
            is_major = any(abs(t - mt) < minor_interval * 0.01 for _, mt in major_ticks)
            if not is_major:
                x_ratio = (t - start_time) / self.window_duration
                minor_ticks.append((plot_left + x_ratio * plot_width, t))
            t += minor_interval

        return major_ticks, minor_ticks

    def _draw_grid(self, painter: QPainter, rect: QRectF):
        major_ticks, minor_ticks = self._compute_time_ticks(rect)
        plot_left, plot_right, plot_bottom = self._get_plot_bounds()

        painter.setPen(QPen(QColor(255, 255, 255, 15), 1, Qt.PenStyle.DotLine))
        for x, _ in minor_ticks:
            painter.drawLine(int(x), int(rect.top()), int(x), int(plot_bottom))

        painter.setPen(QPen(QColor(255, 255, 255, 40), 1, Qt.PenStyle.DotLine))
        for x, _ in major_ticks:
            painter.drawLine(int(x), int(rect.top()), int(x), int(plot_bottom))

        painter.setPen(QPen(self.grid_color, 1, Qt.PenStyle.DotLine))
        num_h_lines = 5
        for i in range(num_h_lines + 1):
            y = rect.top() + ((plot_bottom - rect.top()) * i / num_h_lines)
            painter.drawLine(int(rect.left()), int(y), int(rect.right()), int(y))

    def _draw_traces(self, painter: QPainter, rect: QRectF, data: dict):
        display_channels = self.get_display_order()
        display_channels = [ch for ch in display_channels if ch in data]
        n_channels = len(display_channels)
        if n_channels == 0:
            return
        plot_left, plot_right, plot_bottom = self._get_plot_bounds()
        plot_top = rect.top()
        plot_width = plot_right - plot_left
        plot_height = plot_bottom - plot_top
        if plot_width <= 0 or plot_height <= 0:
            return
        channel_height = plot_height / n_channels
        start_time = float(self.start_time)
        end_time = start_time + float(self.window_duration)
        if end_time <= start_time:
            return
        max_total = max(1000, int(self.max_total_points))
        target_per_channel = max(64, int(self.optimal_points_per_channel))
        if target_per_channel * n_channels > max_total:
            target_per_channel = max(64, max_total // n_channels)
        global_min = None
        global_max = None
        if not self.auto_scale:
            for channel in display_channels:
                for trace in data[channel]:
                    channel_data = np.asarray(trace['data'], dtype=np.float64)
                    if channel_data.size == 0:
                        continue
                    finite = np.isfinite(channel_data)
                    if not np.any(finite):
                        continue
                    finite_vals = channel_data[finite]
                    ch_min = float(np.min(finite_vals))
                    ch_max = float(np.max(finite_vals))
                    if global_min is None or ch_min < global_min:
                        global_min = ch_min
                    if global_max is None or ch_max > global_max:
                        global_max = ch_max
        if global_min is None or global_max is None:
            global_min = -1.0
            global_max = 1.0
        global_range = global_max - global_min
        if global_range <= 0:
            global_range = 1.0
        cache_key = (
            round(start_time, 6),
            round(float(self.window_duration), 6),
            round(float(self._channel_offset), 6),
            round(float(self.global_gain), 6),
            bool(self.auto_scale),
            bool(self.filter_enabled),
            round(float(self.filter_low_freq), 3),
            round(float(self.filter_high_freq), 3),
            bool(self.notch_enabled),
            round(float(self.notch_freq), 3),
            bool(self.detrend_enabled),
            int(target_per_channel),
            tuple(display_channels),
            tuple(tuple(t['mode'] for t in data[c]) for c in display_channels),
        )
        if cache_key != self._trace_cache_key:
            self._trace_path_cache = {}
            self._trace_cache_key = cache_key
        for idx, channel in enumerate(display_channels):
            y_center = plot_bottom - (idx + 0.5) * channel_height + self._channel_offset * channel_height
            if y_center + channel_height < plot_top or y_center - channel_height > plot_bottom:
                continue
            traces = data[channel]
            if not traces:
                continue
            for trace_idx, trace in enumerate(traces):
                cached = self._trace_path_cache.get((channel, trace_idx))
                if cached is None:
                    channel_data = np.asarray(trace['data'], dtype=np.float64)
                    if channel_data.size == 0:
                        continue
                    n_samples = channel_data.size
                    if n_samples <= target_per_channel:
                        draw_y = channel_data
                        x_ratios = np.linspace(0.0, 1.0, n_samples) if n_samples > 1 else np.array([0.0])
                    else:
                        decimated_samples, decimated_values, s_idx, e_idx = self.engine.get_decimated_segment(channel, start_time, end_time, target_per_channel)
                        if decimated_values.size == 0:
                            continue
                        draw_y = decimated_values
                        if e_idx > s_idx:
                            x_ratios = (decimated_samples - s_idx) / float(e_idx - s_idx)
                        else:
                            x_ratios = np.linspace(0.0, 1.0, decimated_values.size)
                    finite = np.isfinite(draw_y)
                    if not np.any(finite):
                        continue
                    
                    if self.auto_scale:
                        finite_vals = draw_y[finite]
                        data_min = float(np.min(finite_vals))
                        data_max = float(np.max(finite_vals))
                        data_range = data_max - data_min
                        if data_range <= 0:
                            data_range = 1.0
                        normalized = (draw_y - (data_min + data_max) / 2.0) / data_range
                        y_offsets = -normalized * (channel_height * 0.40) * self.global_gain
                    else:
                        scale = (channel_height * 0.40) / global_range
                        y_offsets = -(draw_y - (global_min + global_max) / 2.0) * scale * self.global_gain

                    x_pixels = plot_left + x_ratios * plot_width
                    path = QPainterPath()
                    started = False
                    for x, y_off, ok in zip(x_pixels, y_offsets, finite):
                        if not ok:
                            started = False
                            continue
                        y = y_center + float(y_off)
                        if y < plot_top:
                            y = plot_top
                        elif y > plot_bottom:
                            y = plot_bottom
                        if not started:
                            path.moveTo(float(x), float(y))
                            started = True
                        else:
                            path.lineTo(float(x), float(y))
                    self._trace_path_cache[(channel, trace_idx)] = path
                    cached = path
                color = self._trace_color_for(channel, trace.get('mode', 'raw'))
                pen = QPen(color, self.trace_width)
                pen.setCosmetic(True)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPath(cached)

                

    def _draw_time_axis(self, painter: QPainter, rect: QRectF):
        painter.setPen(QColor("#888888"))
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)

        plot_left, plot_right, plot_bottom = self._get_plot_bounds()
        major_ticks, minor_ticks = self._compute_time_ticks(rect)

        tick_len_minor = 4
        tick_len_major = 8

        painter.setPen(QPen(QColor("#777777"), 1))
        for x, _ in minor_ticks:
            painter.drawLine(int(x), int(plot_bottom), int(x), int(plot_bottom + tick_len_minor))

        painter.setPen(QPen(QColor("#aaaaaa"), 1))
        for x, _ in major_ticks:
            painter.drawLine(int(x), int(plot_bottom), int(x), int(plot_bottom + tick_len_major))

        painter.setPen(QColor("#aaaaaa"))
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)

        for x, t in major_ticks:
            if self._use_timestamps and self.engine.timestamps_loaded:
                idx = int(np.searchsorted(self.engine.timestamps, t, side='left'))
                idx = max(0, min(idx, len(self.engine.timestamps) - 1))
                label = f"{self.engine.timestamps[idx]:.3f}s"
            else:
                label = f"{t:.2f}s"
            text_width = 60
            text_x = max(int(x - text_width / 2), 2)
            painter.drawText(text_x, int(plot_bottom + tick_len_major + 2), text_width, 14, Qt.AlignmentFlag.AlignCenter, label)

    def _label_strip_header_height(self) -> int:
        return 22

    def _label_strip_header_rect(self, rect: QRectF) -> QRectF:
        return QRectF(
            float(rect.left() + 4),
            float(rect.top() + 2),
            float(LABEL_STRIP_WIDTH - 8),
            float(self._label_strip_header_height() - 4),
        )

    def _draw_channel_labels(self, painter: QPainter, rect: QRectF, data):
        """Draw channel labels + the per-channel-modes header button,
        with a small visible button on each row that opens that
        channel's display-mode menu."""
        # Strip background.
        _, _, plot_bottom_full = self._get_plot_bounds()
        strip_rect = QRectF(
            float(rect.left()), float(rect.top()),
            float(LABEL_STRIP_WIDTH), float(plot_bottom_full - rect.top()),
        )
        painter.fillRect(strip_rect, LABEL_STRIP_BG)

        # ---- Header button (bulk per-channel menu) ----
        header = self._label_strip_header_rect(rect)
        n_override = len(self.channel_display_modes)
        if n_override > 0:
            header_bg = QColor("#3a5f8a")
            header_border = QColor("#5a9fff")
        else:
            header_bg = QColor("#2d2d2d")
            header_border = QColor("#4a4a4a")

        painter.setBrush(QBrush(header_bg))
        painter.setPen(QPen(header_border, 1.0))
        painter.drawRoundedRect(header, 3, 3)

        painter.setPen(QColor("#e8e8e8"))
        font = QFont()
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)
        header_text = "Channel display ▾"
        if n_override > 0:
            header_text = f"Channel display ({n_override}) ▾"
        painter.drawText(header, Qt.AlignmentFlag.AlignCenter, header_text)

        self._label_strip_header_hitbox = header

        # ---- Channel rows ----
        display_channels = self.get_display_order()
        display_channels = [ch for ch in display_channels if ch in data] or self.get_display_order()
        n_channels = len(display_channels)

        _, _, plot_bottom = self._get_plot_bounds()
        plot_top = rect.top() + self._label_strip_header_height()
        channel_height = (plot_bottom - plot_top) / max(1, n_channels)

        name_font = QFont()
        name_font.setPointSize(LABEL_FONT_POINT_SIZE)
        name_font.setBold(True)

        depth_font = QFont()
        depth_font.setPointSize(LABEL_FONT_POINT_SIZE - 1)
        depth_font.setBold(False)

        # Reset row-button hitboxes; they get repopulated as we draw.
        self._row_button_rects = {}

        for idx, ch in enumerate(display_channels):
            y_center = plot_bottom - (idx + 0.5) * channel_height + self._channel_offset * channel_height

            if y_center + channel_height < plot_top or y_center - channel_height > plot_bottom:
                continue

            has_override = self._has_channel_override(ch)

            # ---- Channel number ----
            painter.setFont(name_font)
            painter.setPen(LABEL_TEXT_COLOR if not has_override else QColor("#ffd23f"))
            painter.drawText(int(rect.left()) + 6, int(y_center - 9), 40, 14, Qt.AlignmentFlag.AlignLeft, f"CH{ch}")

            # ---- Depth ----
            if ch in self.channel_depths:
                depth = self.channel_depths[ch]
                painter.setFont(depth_font)
                painter.setPen(QColor("#b8b8b8"))
                painter.drawText(int(rect.left()) + 44, int(y_center + 2), 40, 12, Qt.AlignmentFlag.AlignLeft, f"{depth:.0f} µm")

            # ---- Mode indicator dot(s) ----
            dot_x = int(rect.left()) + LABEL_STRIP_WIDTH - ROW_BUTTON_WIDTH - 20
            dot_y = int(y_center) - 3

            if has_override:
                modes = self.channel_display_modes[ch]
                seg_x = dot_x
                seg_w = 4
                seg_h = 6
                painter.setPen(Qt.PenStyle.NoPen)
                if modes.get('raw'):
                    painter.setBrush(QBrush(self.channel_colors.get(ch, self.default_trace_color)))
                    painter.drawRect(seg_x, dot_y, seg_w, seg_h)
                    seg_x += seg_w
                if modes.get('filtered'):
                    painter.setBrush(QBrush(FILTERED_COLOR))
                    painter.drawRect(seg_x, dot_y, seg_w, seg_h)
                    seg_x += seg_w
                if modes.get('csd'):
                    painter.setBrush(QBrush(CSD_COLOR))
                    painter.drawRect(seg_x, dot_y, seg_w, seg_h)
            else:
                color = self.channel_colors.get(ch, self.default_trace_color)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(color))
                painter.drawRect(dot_x, dot_y, 6, 6)

            # ---- Per-row button ----
            button_rect = QRectF(
                float(rect.left() + LABEL_STRIP_WIDTH - ROW_BUTTON_WIDTH - 4),
                float(y_center - ROW_BUTTON_HEIGHT / 2.0),
                float(ROW_BUTTON_WIDTH),
                float(ROW_BUTTON_HEIGHT),
            )
            self._row_button_rects[ch] = button_rect

            hovered = (ch == self._hovered_row_button_channel)
            btn_bg = ROW_BUTTON_BG_HOVER if hovered else ROW_BUTTON_BG
            btn_border = ROW_BUTTON_BORDER_HOVER if hovered else ROW_BUTTON_BORDER

            painter.setBrush(QBrush(btn_bg))
            painter.setPen(QPen(btn_border, 1.0))
            painter.drawRoundedRect(button_rect, 3, 3)

            # Three-dot glyph, centered.
            painter.setPen(QPen(ROW_BUTTON_GLYPH, 1.5))
            glyph_cx = button_rect.center().x()
            glyph_cy = button_rect.center().y()
            dot_r = 1.6
            for dx in (-4.0, 0.0, 4.0):
                painter.drawEllipse(
                    QRectF(glyph_cx + dx - dot_r, glyph_cy - dot_r, 2 * dot_r, 2 * dot_r)
                )

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def wheelEvent(self, event):
        """Handle mouse wheel for scrolling, zooming, and gain."""
        if not self._data_loaded:
            return

        delta = event.angleDelta().y()
        modifiers = event.modifiers()

        if modifiers & Qt.KeyboardModifier.ControlModifier:
            # Ctrl + wheel: zoom to cursor.
            if delta > 0:
                self._zoom_to_cursor(1.2, event.position())
            else:
                self._zoom_to_cursor(1 / 1.2, event.position())
        elif modifiers & Qt.KeyboardModifier.ShiftModifier:
            # Shift + wheel: adjust gain. Each notch multiplies/divides
            # gain by a small factor so repeated scrolls feel gradual.
            factor = 1.15 if delta > 0 else 1 / 1.15
            new_gain = self.global_gain * factor
            new_gain = max(0.1, min(new_gain, 1000.0))
            self.global_gain = new_gain
            # Sync the control-panel spinbox without re-emitting gainChanged
            # (that would loop back through _on_control_gain_changed and
            # call _invalidate_cache twice; harmless but wasteful).
            self.control_panel.gain_spin.blockSignals(True)
            self.control_panel.gain_spin.setValue(new_gain)
            self.control_panel.gain_spin.blockSignals(False)
            self._invalidate_cache()
            self.update()
        else:
            # Plain wheel: scroll through time.
            time_delta = (delta / 120.0) * 0.1 * self.window_duration
            min_start = self._get_min_start_time()
            max_start = self._get_max_start_time()
            self.start_time = min(
                max(min_start, self.start_time - time_delta), max_start
            )
            self._update_time_labels()
            self._invalidate_cache()
            self.update()

        event.accept()

    def keyPressEvent(self, event):
        if not self._data_loaded:
            super().keyPressEvent(event)
            return
        if event.key() in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self._zoom(1.2)
        elif event.key() == Qt.Key.Key_Minus:
            self._zoom(1 / 1.2)
        elif event.key() == Qt.Key.Key_Left:
            self._step_time(-1)
        elif event.key() == Qt.Key.Key_Right:
            self._step_time(1)
        elif event.key() == Qt.Key.Key_Home:
            self._go_to_start()
        elif event.key() == Qt.Key.Key_End:
            self._go_to_end()
        elif event.key() == Qt.Key.Key_0:
            self._reset_view()
        elif event.key() == Qt.Key.Key_PageUp:
            self._zoom(2.0)
        elif event.key() == Qt.Key.Key_PageDown:
            self._zoom(0.5)
        elif event.key() == Qt.Key.Key_Up:
            self._scroll_up()
        elif event.key() == Qt.Key.Key_Down:
            self._scroll_down()
        super().keyPressEvent(event)

    def contextMenuEvent(self, event):
        if not self._data_loaded:
            return
        plot_left, plot_right, _ = self._get_plot_bounds()
        plot_width = plot_right - plot_left
        for i, cursor_fraction in enumerate(self._time_cursors):
            cursor_time = self._cursor_to_time(cursor_fraction)
            x_ratio = (cursor_time - self.start_time) / self.window_duration
            cursor_x = plot_left + x_ratio * plot_width
            if abs(event.pos().x() - cursor_x) < 10:
                self._selected_cursor = i
                self.update()
                menu = QMenu(self)
                remove_action = menu.addAction("Remove Cursor")
                remove_action.triggered.connect(lambda: self._remove_cursor(i))
                menu.exec(event.globalPos())
                return
        menu = QMenu(self)
        clear_cursors_action = menu.addAction("Clear All Time Cursors")
        clear_cursors_action.triggered.connect(self._clear_time_cursors)
        menu.exec(event.globalPos())

    def _remove_cursor(self, index: int):
        if 0 <= index < len(self._time_cursors):
            self._time_cursors.pop(index)
            self._selected_cursor = None
            self.update()

    def _clear_time_cursors(self):
        self._time_cursors.clear()
        self._selected_cursor = None
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # ---- Left label strip ----
            if event.position().x() < self.rect().left() + LABEL_STRIP_WIDTH:
                # Header button first.
                header_hitbox = getattr(self, '_label_strip_header_hitbox', None)
                if header_hitbox is not None and header_hitbox.contains(event.position()):
                    self._show_bulk_mode_menu(event.globalPosition().toPoint())
                    event.accept()
                    return

                # Per-row buttons: only the visible button opens the
                # channel menu now, not the entire row.
                for ch, r in self._row_button_rects.items():
                    if r.contains(event.position()):
                        self._show_channel_mode_menu(ch, event.globalPosition().toPoint())
                        event.accept()
                        return

            # ---- Trash bin ----
            if hasattr(self, '_trash_bin_rects'):
                for cursor_idx, trash_rect in self._trash_bin_rects.items():
                    if trash_rect.contains(event.position()):
                        self._remove_cursor(cursor_idx)
                        event.accept()
                        return

            # ---- Cursor drag ----
            if self._data_loaded and self._time_cursors:
                plot_left, plot_right, _ = self._get_plot_bounds()
                plot_width = plot_right - plot_left
                for i, cursor_fraction in enumerate(self._time_cursors):
                    cursor_time = self._cursor_to_time(cursor_fraction)
                    x_ratio = (cursor_time - self.start_time) / self.window_duration
                    cursor_x = plot_left + x_ratio * plot_width
                    if abs(event.position().x() - cursor_x) < 10:
                        self._selected_cursor = i
                        self._dragging_cursor = True
                        self._drag_cursor_start_x = event.position().x()
                        self.update()
                        event.accept()
                        return

            # ---- Pan ----
            self._dragging = True
            self._drag_start_pos = event.position()
            self._drag_start_time = self.start_time
            self._drag_start_offset = self._channel_offset
            event.accept()

        elif event.button() == Qt.MouseButton.RightButton:
            if self._data_loaded:
                plot_left, plot_right, _ = self._get_plot_bounds()
                plot_width = plot_right - plot_left
                for i, cursor_fraction in enumerate(self._time_cursors):
                    cursor_time = self._cursor_to_time(cursor_fraction)
                    x_ratio = (cursor_time - self.start_time) / self.window_duration
                    cursor_x = plot_left + x_ratio * plot_width
                    if abs(event.position().x() - cursor_x) < 10:
                        self._selected_cursor = i
                        self.update()
                        event.accept()
                        return
                x_ratio = (event.position().x() - plot_left) / max(1, plot_width)
                cursor_time = self.start_time + x_ratio * self.window_duration
                cursor_fraction = self._time_to_cursor_fraction(cursor_time)
                self._time_cursors.append(cursor_fraction)
                self._time_cursors.sort()
                self._selected_cursor = len(self._time_cursors) - 1
                self.update()
                event.accept()

    def mouseMoveEvent(self, event):
        # ---- Hover state for the per-row buttons ----
        new_hover: int | None = None
        pos = event.position()
        for ch, r in self._row_button_rects.items():
            if r.contains(pos):
                new_hover = ch
                break
        if new_hover != self._hovered_row_button_channel:
            self._hovered_row_button_channel = new_hover
            self.setCursor(
                Qt.CursorShape.PointingHandCursor if new_hover is not None
                else Qt.CursorShape.ArrowCursor
            )
            self.update()

        if self._dragging_cursor and self._selected_cursor is not None:
            plot_left, plot_right, _ = self._get_plot_bounds()
            plot_width = plot_right - plot_left
            delta_x = event.position().x() - self._drag_cursor_start_x
            time_delta_per_pixel = self.window_duration / max(1, plot_width)
            current_fraction = self._time_cursors[self._selected_cursor]
            current_time = self._cursor_to_time(current_fraction)
            new_time = current_time + delta_x * time_delta_per_pixel
            min_time = self._get_min_start_time()
            max_time = self._get_max_start_time() + self.window_duration
            new_time = max(min_time, min(new_time, max_time))
            self._time_cursors[self._selected_cursor] = self._time_to_cursor_fraction(new_time)
            self._drag_cursor_start_x = event.position().x()
            self.update()
            event.accept()
            return

        if hasattr(self, '_dragging') and self._dragging and self._drag_start_pos is not None:
            current_pos = event.position()
            delta_x = current_pos.x() - self._drag_start_pos.x()
            delta_y = current_pos.y() - self._drag_start_pos.y()
            time_delta_per_pixel = self.window_duration / max(1, self.width())
            time_delta = -delta_x * time_delta_per_pixel
            min_start = self._get_min_start_time()
            max_start = self._get_max_start_time()
            self.start_time = min(max(min_start, self._drag_start_time + time_delta), max_start)
            channel_height = self.height() / max(1, len(self.channels))
            channel_delta = delta_y / max(1, channel_height)
            self._channel_offset = self._drag_start_offset + channel_delta
            self._update_time_labels()
            self._invalidate_cache()
            self.update()
            event.accept()

    def mouseReleaseEvent(self, event):
        if self._dragging_cursor:
            self._dragging_cursor = False
            event.accept()
            return
        if hasattr(self, '_dragging') and self._dragging:
            self._dragging = False
            self._drag_start_pos = None
            self._last_view = {
                'start_time': self.start_time,
                'window_duration': self.window_duration,
                'zoom_level': self._zoom_level,
                'channel_offset': self._channel_offset,
            }
            event.accept()

    def mouseDoubleClickEvent(self, event):
        if not self._data_loaded:
            return
        cursor_pos = event.position()
        plot_left, plot_right, _ = self._get_plot_bounds()
        plot_width = plot_right - plot_left
        x_ratio = (cursor_pos.x() - plot_left) / max(1, plot_width)
        cursor_time = self.start_time + x_ratio * self.window_duration
        new_duration = min(10.0, self._get_total_duration())
        new_start = cursor_time - new_duration / 2
        min_start = self._get_min_start_time()
        max_start = self._get_max_start_time()
        new_start = max(min_start, min(new_start, max_start))
        self._channel_offset = 0.0
        self._zoom_level = 1.0
        self.start_time = new_start
        self.window_duration = new_duration
        self.control_panel.zoom_label.setText("100%")
        self.control_panel.duration_spin.blockSignals(True)
        self.control_panel.duration_spin.setValue(self.window_duration)
        self.control_panel.duration_spin.blockSignals(False)
        self._update_time_labels()
        self._invalidate_cache()
        self.update()
        event.accept()


class SpectrogramControlPanel(QWidget):
    """
    Control panel for spectrogram settings.
    """

    channelChanged = pyqtSignal(int)
    enabledChanged = pyqtSignal(bool)
    minFreqChanged = pyqtSignal(float)
    maxFreqChanged = pyqtSignal(float)
    minDbChanged = pyqtSignal(float)
    maxDbChanged = pyqtSignal(float)
    autoDbChanged = pyqtSignal(bool)
    cmapChanged = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Spectrogram Controls")
        self.setMinimumWidth(250)
        self._building_channel_list = False
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.enabled_checkbox = QCheckBox("Show Spectrogram")
        self.enabled_checkbox.setChecked(False)
        self.enabled_checkbox.toggled.connect(self._on_enabled_toggled)
        layout.addWidget(self.enabled_checkbox)

        channel_group = QGroupBox("Channel")
        channel_layout = QVBoxLayout(channel_group)
        self.channel_combo = QComboBox()
        self.channel_combo.currentIndexChanged.connect(self._on_channel_changed)
        channel_layout.addWidget(self.channel_combo)
        layout.addWidget(channel_group)

        freq_group = QGroupBox("Frequency Range (Hz)")
        freq_layout = QGridLayout(freq_group)
        freq_layout.addWidget(QLabel("Min:"), 0, 0)
        self.min_freq_spin = QDoubleSpinBox()
        self.min_freq_spin.setRange(0.0, 49.0)
        self.min_freq_spin.setDecimals(1)
        self.min_freq_spin.setSingleStep(1.0)
        self.min_freq_spin.setValue(0.0)
        self.min_freq_spin.valueChanged.connect(self._on_min_freq_changed)
        freq_layout.addWidget(self.min_freq_spin, 0, 1)
        freq_layout.addWidget(QLabel("Max:"), 1, 0)
        self.max_freq_spin = QDoubleSpinBox()
        self.max_freq_spin.setRange(1.0, 50.0)
        self.max_freq_spin.setDecimals(1)
        self.max_freq_spin.setSingleStep(1.0)
        self.max_freq_spin.setValue(40.0)
        self.max_freq_spin.valueChanged.connect(self._on_max_freq_changed)
        freq_layout.addWidget(self.max_freq_spin, 1, 1)
        layout.addWidget(freq_group)

        color_group = QGroupBox("Color Scale (dB)")
        color_layout = QGridLayout(color_group)
        self.auto_db_checkbox = QCheckBox("Automatic")
        self.auto_db_checkbox.setChecked(True)
        self.auto_db_checkbox.toggled.connect(self._on_auto_db_changed)
        color_layout.addWidget(self.auto_db_checkbox)
        color_layout.addWidget(QLabel("Color min:"))
        self.min_db_spin = QDoubleSpinBox()
        self.min_db_spin.setRange(-300.0, 100.0)
        self.min_db_spin.setSingleStep(1.0)
        self.min_db_spin.setValue(-100.0)
        self.min_db_spin.setSuffix(" dB")
        self.min_db_spin.setEnabled(False)
        self.min_db_spin.valueChanged.connect(self._on_min_db_changed)
        color_layout.addWidget(self.min_db_spin)
        color_layout.addWidget(QLabel("Color max:"))
        self.max_db_spin = QDoubleSpinBox()
        self.max_db_spin.setRange(-300.0, 100.0)
        self.max_db_spin.setSingleStep(1.0)
        self.max_db_spin.setValue(0.0)
        self.max_db_spin.setSuffix(" dB")
        self.max_db_spin.setEnabled(False)
        self.max_db_spin.valueChanged.connect(self._on_max_db_changed)
        color_layout.addWidget(self.max_db_spin)
        layout.addWidget(color_group)

        cmap_group = QGroupBox("Colormap")
        cmap_layout = QVBoxLayout(cmap_group)
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["viridis", "plasma", "inferno", "magma", "cividis", "jet"])
        self.cmap_combo.setCurrentText("viridis")
        self.cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        cmap_layout.addWidget(self.cmap_combo)
        layout.addWidget(cmap_group)

    def _on_min_db_changed(self, value: float):
        self.minDbChanged.emit(value)

    def _on_max_db_changed(self, value: float):
        self.maxDbChanged.emit(value)

    def _on_auto_db_changed(self, checked: bool):
        self.min_db_spin.setEnabled(not checked)
        self.max_db_spin.setEnabled(not checked)
        self.autoDbChanged.emit(checked)

    def _on_enabled_toggled(self, checked: bool):
        self.enabledChanged.emit(bool(checked))

    def _on_channel_changed(self, index: int):
        if self._building_channel_list or index < 0:
            return
        channel = self.channel_combo.itemData(index)
        if channel is not None:
            self.channelChanged.emit(int(channel))

    def _on_min_freq_changed(self, value: float):
        self.minFreqChanged.emit(float(value))

    def _on_max_freq_changed(self, value: float):
        self.maxFreqChanged.emit(float(value))

    def _on_cmap_changed(self, name: str):
        self.cmapChanged.emit(str(name))

    def set_channels(self, channels: list[int]):
        self._building_channel_list = True
        current_channel = None
        if self.channel_combo.currentIndex() >= 0:
            current_channel = self.channel_combo.currentData()
        self.channel_combo.clear()
        for channel in channels:
            self.channel_combo.addItem(f"CH{channel}", int(channel))
        if current_channel is not None:
            self.set_channel(int(current_channel))
        self._building_channel_list = False

    def set_channel(self, channel: int):
        for i in range(self.channel_combo.count()):
            if self.channel_combo.itemData(i) == channel:
                self._building_channel_list = True
                self.channel_combo.setCurrentIndex(i)
                self._building_channel_list = False
                return

    def set_enabled(self, enabled: bool):
        enabled = bool(enabled)
        if self.enabled_checkbox.isChecked() != enabled:
            self.enabled_checkbox.setChecked(enabled)

    def set_min_db(self, value: float):
        self.min_db_spin.setValue(float(value))

    def set_max_db(self, value: float):
        self.max_db_spin.setValue(float(value))
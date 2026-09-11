"""
trace_view.py

Main trace visualization widget. Displays neural data from a TraceEngine
as scrolling lines. Supports channel selection from the probe map,
time-window navigation, customizable colors, depth-based sorting, and
signal filtering.
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
from core.ripple_detector import RippleDetector, RippleParams

from scipy.signal import spectrogram, butter, sosfiltfilt, resample_poly
from matplotlib import colormaps


class TraceControlPanel(QWidget):
    """
    Control panel for trace view settings.
    Can be used as a dockable widget or embedded in the trace view.
    """
    
    # Signals emitted when controls change
    timeChanged = pyqtSignal(float, float)  # start_time, end_time
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
    stepTimeRequested = pyqtSignal(int)  # direction: -1 or +1

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Trace Controls")
        self.setMinimumWidth(350)
        
        self._build_ui()
    
    def _build_ui(self):
        """Build the control panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        
        # Channel info label
        self.channel_info_label = QLabel("No channels")
        self.channel_info_label.setStyleSheet("color: #888; font-size: 10px; font-weight: bold;")
        self.channel_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.channel_info_label)
        
        # ---- Navigation section ----
        nav_group = QGroupBox("Navigation")
        nav_layout = QVBoxLayout(nav_group)
        nav_layout.setSpacing(2)
        
        # Navigation buttons row
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
        
        # Time display
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
        
        # Duration
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
        
        # ---- Playback section ----
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
        
        # ---- Gain and Scaling section ----
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
        
        # ---- Filter section ----
        filter_group = QGroupBox("Filters")
        filter_layout = QVBoxLayout(filter_group)
        filter_layout.setSpacing(2)

        bp_row = QHBoxLayout()
        bp_row.setSpacing(2)
        self.filter_checkbox = QCheckBox("Bandpass:")
        self.filter_checkbox.toggled.connect(self._on_filter_toggled)
        bp_row.addWidget(self.filter_checkbox)

        # Low frequency - allow 0 for high-pass only
        self.low_freq_spin = QDoubleSpinBox()
        self.low_freq_spin.setRange(0.0, 15000.0)  # Allow 0
        self.low_freq_spin.setValue(1.0)
        self.low_freq_spin.setSuffix(" Hz")
        self.low_freq_spin.setEnabled(False)
        self.low_freq_spin.setToolTip("Set to 0 for low-pass only")
        self.low_freq_spin.valueChanged.connect(self._on_filter_params_changed)
        bp_row.addWidget(self.low_freq_spin)

        self.high_freq_spin = QDoubleSpinBox()
        self.high_freq_spin.setRange(0.0, 15000.0)  # Allow 0
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
        
        # ---- Display section ----
        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout(display_group)
        display_layout.setSpacing(2)
        
        self.depth_scale_checkbox = QCheckBox("Show depth scale")
        self.depth_scale_checkbox.setChecked(True)
        self.depth_scale_checkbox.toggled.connect(self._on_depth_scale_toggled)
        display_layout.addWidget(self.depth_scale_checkbox)
        
        # Add clear cursors button
        self.clear_cursors_btn = QPushButton("🗑 Clear All Cursors")
        self.clear_cursors_btn.clicked.connect(self._on_clear_cursors)
        display_layout.addWidget(self.clear_cursors_btn)
        
        layout.addWidget(display_group)

        
        # NOTE on scope: this checkbox controls ONLY how time is *displayed*
        # (axis tick labels / start-end readout) -- see _update_time_labels()
        # and _draw_time_axis(), the only two places self._use_timestamps is
        # read. It does NOT affect which samples are fetched, where the
        # trace is drawn, or where the vertical cursor line sits: those all
        # unconditionally use engine.timestamps_loaded (absolute recording
        # timestamps once a timestamps.npy is loaded, elapsed seconds
        # otherwise) regardless of this checkbox's state. That split is
        # intentional -- it lets you view/scroll using simple elapsed-time
        # labels even when timestamps are loaded, without changing how data
        # is actually indexed underneath. Do not "fix" this by making
        # playback/indexing depend on this checkbox too; if you want that,
        # it needs to become a real second mode, not a label toggle.
        self.use_timestamps_checkbox = QCheckBox("Show real timestamp values on axis")
        self.use_timestamps_checkbox.setToolTip(
            "Display axis labels using the loaded timestamps.npy values "
            "instead of elapsed seconds. Does not change which samples "
            "are shown or where cursors are placed -- those always use "
            "real timestamps once a timestamps.npy file is loaded."
        )
        self.use_timestamps_checkbox.setChecked(False)
        self.use_timestamps_checkbox.setEnabled(False)  # Disabled until timestamps are loaded
        self.use_timestamps_checkbox.toggled.connect(self._on_use_timestamps_toggled)
        display_layout.addWidget(self.use_timestamps_checkbox)


        # ---- Zoom section ----
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
        
        # Add stretch at bottom
        # layout.addStretch()


    # ------------------------------------------------------------------
    # Signal emitters
    # ------------------------------------------------------------------

    def _on_use_timestamps_toggled(self, checked: bool):
        self.useTimestampsChanged.emit(checked)

    def set_timestamps_available(self, available: bool):
        self.use_timestamps_checkbox.setEnabled(available)
        if not available:
            self.use_timestamps_checkbox.setChecked(False)

    def _on_clear_cursors(self):
        """Emit signal to clear cursors."""
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
        """Handle bandpass filter toggle - enable/disable spinboxes."""
        self.low_freq_spin.setEnabled(checked)
        self.high_freq_spin.setEnabled(checked)
        self._emit_filter_settings()
    
    def _on_notch_toggled(self, checked: bool):
        """Handle notch filter toggle - enable/disable spinbox."""
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
    
    # ------------------------------------------------------------------
    # Public methods to update control state
    # ------------------------------------------------------------------
    
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
        
        # View state
        self.setMinimumSize(600, 400)
        
        # Time window state - will be updated when data/timestamps are loaded
        self.start_time = 0.0
        self.window_duration = 10.0
        self._zoom_level = 1.0
        self._channel_offset = 0.0
        self.spectrogram_auto_db = True


        # Track last good view for double-click reset
        self._last_view = {
            'start_time': 0.0,
            'window_duration': 10.0,
            'zoom_level': 1.0,
            'channel_offset': 0.0
        }
        
        # Vertical time cursors - store as sample indices
        self._time_cursors: list[int] = []  # List of sample indices
        self._selected_cursor: int | None = None
        self._dragging_cursor: bool = False
        self._drag_cursor_start_x = 0.0


        # Drag state
        self._dragging = False
        self._drag_start_pos = None  # QPointF
        self._drag_start_time = 0.0
        self._drag_start_offset = 0.0

        # Channel visibility and sorting
        self.channels: list[int] = []
        self.channel_depths: dict[int, float] = {}
        self._sorted_channels: list[int] = []
        
        self._overscroll_tolerance = 0.2  # 20% of window duration

        # Axis-label display mode only -- does NOT affect sample fetching,
        # trace drawing, or cursor placement (those always follow
        # engine.timestamps_loaded). See the checkbox definition in
        # TraceControlPanel._build_ui() for the full explanation.
        self._use_timestamps = False


        # Color settings
        self.default_trace_color = QColor("#ffffff")
        self.background_color = QColor("#1e1e1e")
        self.grid_color = QColor("#555555")
        self.show_grid = True
        self.trace_width = 1.0
        
        self._time_cursors: list[float] = []  # List of time positions for vertical lines

        # Per-channel colors
        self.channel_colors: dict[int, QColor] = {}
        
        # Gain and scaling settings
        self.global_gain = 1.0
        # Default OFF: fixed-scale mode (relative amplitudes preserved
        # across channels) is the standard starting view; matches
        # TraceControlPanel.auto_scale_checkbox's default below -- keep
        # both in sync if this ever changes.
        self.auto_scale = False
        
        # Filter settings (using core.filters)
        self.filter_enabled = False
        self.filter_low_freq = 4.0
        self.filter_high_freq = 30.0
        self.filter_order = 4
        self.notch_enabled = False
        self.notch_freq = 50.0
        self.notch_q = 30.0
        self.detrend_enabled = False
        
        # Cache for filtered data
        self._filtered_cache = {}
        
        # Animation settings
        self.animation_enabled = False
        self.animation_speed = 100
        self._animation_timer = QTimer(self)
        self._animation_timer.timeout.connect(self._on_animation_tick)
        self._is_animating = False
        
        # Zoom state
        self._zoom_level = 1.0
        
        # Display options
        self.show_channel_labels = True
        self.show_time_axis = True
        self.show_depth_scale = False
        
        # Cache for data
        self._cached_data = None
        self._cache_start = None
        self._cache_end = None
        self._cache_start_idx = None
        self._cache_end_idx = None

        # Ripple overlay (or, in principle, any future per-channel
        # analysis overlay following the same shape). None when nothing
        # is active. When set, this is:
        #   {
        #     'channels': {
        #         channel_id: {
        #             'envelope': np.ndarray,       # same sample basis as 'signal'
        #             'signal': np.ndarray,          # raw/CSD signal actually analyzed (for threshold stats)
        #             'sample_offset': int,          # samples from recording start to signal[0]
        #             'sample_rate': float,           # sample rate of envelope/signal (post any decimation)
        #             'events': list[RippleEvent],   # start/end/peak sample indices relative to sample_offset
        #             'peak_threshold_sd': float,
        #             'boundary_threshold_sd': float,
        #         },
        #         ...
        #     },
        #     'show_envelope': bool,
        #     'show_thresholds': bool,
        #     'show_events': bool,
        #   }
        # Deliberately generic dict shape (not a ripple-specific class) so
        # _draw_ripple_overlay stays a thin renderer and the analysis side
        # (RippleDialog) owns all ripple-specific computation.
        self.ripple_overlay: dict | None = None
        self._ripple_detector_for_overlay = RippleDetector()  # stateless; reused to avoid re-instantiating every paint


        # ------------------------------------------------------------------
        # Spectrogram
        # ------------------------------------------------------------------
        self.show_spectrogram = False
        self.spectrogram_channel = None

        # Computed spectrogram data
        self.spectrogram_data = None
        self.spectrogram_freqs = None
        self.spectrogram_times = None

        # Frequency-masked slice actually rendered (built by
        # _update_spectrogram_frequency_view from spectrogram_data/
        # spectrogram_freqs + the min/max freq range below). Must exist
        # before any first call to _render_spectrogram_image() -- e.g. a
        # dB/colormap control changed before a full compute cycle has
        # run -- or that method would raise AttributeError instead of
        # cleanly no-op'ing like it does for spectrogram_data.
        self._spectrogram_display_data = None
        self._spectrogram_display_freqs = None

        # Display frequency range
        self.spectrogram_min_freq = 0.0
        self.spectrogram_max_freq = 40.0

        # Color scale in dB
        self.spectrogram_vmin = -100.0
        self.spectrogram_vmax = -40.0
        self.spectrogram_cmap = "viridis"

        # Spectrogram processing parameters
        #
        # Raw Neuropixels data:
        #       30 kHz
        #          ↓
        #       low-pass 40 Hz
        #          ↓
        #       resample to 100 Hz
        #          ↓
        #       spectrogram
        #
        # At 100 Hz with nperseg=100:
        #       frequency resolution = 1 Hz
        #
        # noverlap=90 (90% overlap) rather than 50%: quintuples the
        # number of time columns per window (e.g. ~19 -> ~91 for a
        # typical 10s view) for a small compute cost increase, since
        # each STFT frame is still only 100 samples. This is the main
        # lever for how "smooth" the spectrogram looks along the time
        # axis -- more real columns to draw/interpolate between, rather
        # than a handful of wide blocks stretched across the view.
        self.spectrogram_sr = 100.0
        self.spectrogram_lowpass = 40.0
        self.spectrogram_filter_order = 4
        self.spectrogram_nperseg = 100
        self.spectrogram_noverlap = 90

        # Pre-rendered Qt image
        self._spectrogram_image = None

        # Cache key for the calculated PSD
        self._spectrogram_cache_key = None

        # Prevent expensive recomputation during continuous navigation.
        # The calculation is scheduled only after navigation settles.
        self._spectrogram_update_timer = QTimer(self)
        self._spectrogram_update_timer.setSingleShot(True)
        self._spectrogram_update_timer.setInterval(150)
        self._spectrogram_update_timer.timeout.connect(
            self._compute_spectrogram
        )

        self.spectrogram_control = SpectrogramControlPanel()
        self._connect_spectrogram_controls()



        # Create control panel
        self.control_panel = TraceControlPanel()
        self._connect_control_panel()
        
        # Flag
        self._data_loaded = False
        
        # Setup UI
        self._setup_ui()
        
        # Initialize from engine if data is already loaded
        if engine.data_loaded:
            self._setup_from_engine()

    def _setup_ui(self):
        """Create the UI layout - plot area with scrollbar at bottom."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create a stretchable plot area
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(0)
        
        # The plot container takes up all available space
        layout.addWidget(plot_container, stretch=1)
        
        # Add scrollbar at the bottom (not stretched)
        self._build_scrollbar(layout)
        
        # Set focus policy
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def _connect_control_panel(self):
        """Connect control panel signals to this widget's methods."""
        self.control_panel.timeChanged.connect(self._on_control_time_changed)
        self.control_panel.durationChanged.connect(self._on_control_duration_changed)
        self.control_panel.gainChanged.connect(self._on_control_gain_changed)
        self.control_panel.autoScaleChanged.connect(self._on_control_auto_scale_changed)
        self.control_panel.animationToggled.connect(self._on_control_animation_toggled)
        self.control_panel.animationSpeedChanged.connect(self._on_control_animation_speed_changed)
        self.control_panel.filterChanged.connect(self._on_control_filter_changed)
        self.control_panel.zoomChanged.connect(self._on_control_zoom_changed)
        self.control_panel.depthScaleChanged.connect(self._on_control_depth_scale_changed)
        self.control_panel.clearCursorsRequested.connect(self._clear_time_cursors)  # ADD THIS
        self.control_panel.useTimestampsChanged.connect(self._on_use_timestamps_changed)
        self.control_panel.goToStartRequested.connect(self._go_to_start)
        self.control_panel.goToEndRequested.connect(self._go_to_end)
        self.control_panel.stepTimeRequested.connect(self._step_time)


    def _build_scrollbar(self, layout: QVBoxLayout):
        """Build the horizontal scrollbar and add it to the bottom."""
        self.scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        self.scrollbar.setRange(0, 1000)
        self.scrollbar.setFixedHeight(15)
        self.scrollbar.setStyleSheet("""
            QScrollBar:horizontal {
                background: #2d2d2d;
                height: 15px;
                margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background: #4a9eff;
                min-width: 20px;
                border-radius: 7px;
                margin: 2px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #5aaeff;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background: #1e1e1e;
            }
        """)
        self.scrollbar.valueChanged.connect(self._on_scrollbar_changed)
        layout.addWidget(self.scrollbar)
        
        # Make sure the scrollbar is at the bottom
        layout.setAlignment(self.scrollbar, Qt.AlignmentFlag.AlignBottom)

    def _setup_from_engine(self):
        """Initialize view state from the engine."""
        if self.engine.data_loaded:
            self._data_loaded = True
            
            # Get valid time range
            if self.engine.timestamps_loaded and self.engine.timestamps is not None:
                time_start, time_end = self.engine.get_time_range()
                self.start_time = time_start  # Start at first timestamp
                self.window_duration = min(10.0, time_end - time_start)
            else:
                self.start_time = 0.0
                self.window_duration = min(
                    self.engine.total_duration,
                    max(self.window_duration, 1.0)
                )
            
            self._update_scrollbar_range()
            self._update_time_labels()
            self.update()
    # ------------------------------------------------------------------
    # Control panel signal handlers
    # ------------------------------------------------------------------

    def _on_use_timestamps_changed(self, enabled: bool):
        """Handle timestamps usage toggle."""
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
    # spectrogram methods
    # ------------------------------------------------------------------
    def _schedule_spectrogram_update(self):
        """
        Schedule a spectrogram recalculation.

        Repeated navigation events restart the timer, so the expensive
        calculation happens only after navigation has settled.
        """
        if not self.show_spectrogram:
            return

        if self.spectrogram_channel is None:
            return

        self._spectrogram_update_timer.start()
        

    def _get_spectrogram_height(self):
        """Return the height reserved for the spectrogram."""
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
        """Connect spectrogram control-panel signals."""

        self.spectrogram_control.enabledChanged.connect(
            self._on_spectrogram_enabled_changed
        )

        self.spectrogram_control.channelChanged.connect(
            self._on_spectrogram_channel_changed
        )

        self.spectrogram_control.minFreqChanged.connect(
            self._on_spectrogram_min_freq_changed
        )

        self.spectrogram_control.maxFreqChanged.connect(
            self._on_spectrogram_max_freq_changed
        )

        self.spectrogram_control.minDbChanged.connect(
            self._on_spectrogram_min_db_changed
        )

        self.spectrogram_control.maxDbChanged.connect(
            self._on_spectrogram_max_db_changed
        )

        self.spectrogram_control.cmapChanged.connect(
            self._on_spectrogram_cmap_changed
        )


        self.spectrogram_control.minDbChanged.connect(
        self._on_spectrogram_min_db_changed
        )

        self.spectrogram_control.maxDbChanged.connect(
            self._on_spectrogram_max_db_changed
        )

        self.spectrogram_control.autoDbChanged.connect(
            self._on_spectrogram_auto_db_changed
        )

    def _on_spectrogram_enabled_changed(self, enabled: bool):
        """Handle spectrogram checkbox."""

        self.show_spectrogram = bool(enabled)

        if not self.show_spectrogram:

            self._spectrogram_update_timer.stop()
            self._spectrogram_image = None

        else:

            if self.spectrogram_channel is None:

                if self.channels:
                    self.spectrogram_channel = self.channels[0]
                    self.spectrogram_control.set_channel(
                        self.spectrogram_channel
                    )

            if self.spectrogram_channel is not None:

                self._spectrogram_cache_key = None

                # Immediate computation on enable.
                self._compute_spectrogram()

        self.update()
        

    def _on_spectrogram_channel_changed(self, channel: int):
        """Change the channel used for the spectrogram."""

        self.spectrogram_channel = int(channel)

        # Force recalculation for the new channel.
        self._spectrogram_cache_key = None

        if self.show_spectrogram:
            self._schedule_spectrogram_update()

        self.update()

    def _on_spectrogram_min_freq_changed(self, value: float):
        """Change lower frequency limit."""

        value = max(0.0, float(value))

        if value >= self.spectrogram_max_freq:
            value = max(0.0, self.spectrogram_max_freq - 1.0)

        self.spectrogram_min_freq = value

        # Frequency limits only affect which rows are displayed.
        # The expensive PSD calculation does not need to be repeated.
        if self.spectrogram_data is not None:
            self._update_spectrogram_frequency_view()

        self.update()


    def _on_spectrogram_max_freq_changed(self, value: float):
        """Change upper frequency limit."""

        value = min(
            self.spectrogram_sr / 2.0,
            float(value)
        )

        if value <= self.spectrogram_min_freq:
            value = self.spectrogram_min_freq + 1.0

        self.spectrogram_max_freq = value

        if self.spectrogram_data is not None:
            self._update_spectrogram_frequency_view()

        self.update()


    def _on_spectrogram_cmap_changed(self, name: str):
        """Change spectrogram colormap."""

        self.spectrogram_cmap = str(name)

        if self.spectrogram_data is not None:
            self._render_spectrogram_image()

        self.update()





    def _compute_spectrogram(self):
        """Compute the spectrogram from downsampled low-frequency data."""

        if not self.show_spectrogram:
            return

        if not self.engine.data_loaded:
            print("Spectrogram: engine has no data")
            return

        if self.spectrogram_channel is None:
            print("Spectrogram: no channel selected")
            return

        start_time = float(self.start_time)
        end_time = (
            start_time
            + float(self.window_duration)
        )

        if end_time <= start_time:
            print("Spectrogram: invalid time range")
            return

        print(
            f"Spectrogram: CH{self.spectrogram_channel} "
            f"{start_time:.3f}–{end_time:.3f} s"
        )

        # ----------------------------------------------------------
        # Cache check
        # ----------------------------------------------------------
        # Recomputing the PSD (filter + resample + scipy.signal.spectrogram)
        # on every navigation tick is the expensive part of this pipeline;
        # a cache key was already threaded through several call sites
        # (reset to None whenever something that invalidates the result
        # changes) but was never actually compared against anything, so
        # it did nothing. This is the missing other half: skip
        # recomputation entirely when nothing the PSD depends on has
        # changed since the last call.
        cache_key = (
            self.spectrogram_channel,
            round(start_time, 6),
            round(end_time, 6),
            self.spectrogram_sr,
            self.spectrogram_lowpass,
            self.spectrogram_filter_order,
            self.spectrogram_nperseg,
            self.spectrogram_noverlap,
        )
        if (self._spectrogram_cache_key == cache_key
                and self.spectrogram_data is not None):
            return

        try:

            # ----------------------------------------------------------
            # Get raw data
            # ----------------------------------------------------------

            data = self.engine.get_channel_data(
                self.spectrogram_channel,
                start_time,
                end_time,
            )

            data = np.asarray(
                data,
                dtype=np.float64,
            )

            print(
                f"Spectrogram: raw samples = {len(data)}"
            )

            if data.size < 10:
                print("Spectrogram: not enough samples")
                self.spectrogram_data = None
                self._spectrogram_image = None
                self._spectrogram_cache_key = None
                return

            # ----------------------------------------------------------
            # Replace invalid samples
            # ----------------------------------------------------------

            finite = np.isfinite(data)

            if not np.all(finite):

                if not np.any(finite):
                    print("Spectrogram: all samples invalid")
                    return

                replacement = np.median(
                    data[finite]
                )

                data = np.nan_to_num(
                    data,
                    nan=replacement,
                    posinf=replacement,
                    neginf=replacement,
                )

            # ----------------------------------------------------------
            # Raw sampling rate
            # ----------------------------------------------------------

            raw_sr = float(
                self.engine.sr
            )

            target_sr = float(
                self.spectrogram_sr
            )

            print(
                f"Spectrogram: {raw_sr:g} Hz -> "
                f"{target_sr:g} Hz"
            )

            # ----------------------------------------------------------
            # Low-pass before downsampling
            # ----------------------------------------------------------

            cutoff = min(
                float(self.spectrogram_lowpass),
                target_sr * 0.45,
                raw_sr * 0.45,
            )

            sos = butter(
                int(self.spectrogram_filter_order),
                cutoff,
                btype="lowpass",
                fs=raw_sr,
                output="sos",
            )

            try:

                data = sosfiltfilt(
                    sos,
                    data,
                )

            except ValueError:

                print(
                    "Spectrogram: filtering skipped "
                    "because the segment is too short"
                )

            # ----------------------------------------------------------
            # Downsample
            # ----------------------------------------------------------

            if not np.isclose(
                raw_sr,
                target_sr,
            ):

                from math import gcd

                raw_sr_int = int(
                    round(raw_sr)
                )

                target_sr_int = int(
                    round(target_sr)
                )

                divisor = gcd(
                    raw_sr_int,
                    target_sr_int,
                )

                up = target_sr_int // divisor
                down = raw_sr_int // divisor

                data = resample_poly(
                    data,
                    up,
                    down,
                )

            print(
                f"Spectrogram: downsampled samples = "
                f"{len(data)}"
            )

            # ----------------------------------------------------------
            # Spectrogram
            # ----------------------------------------------------------

            nperseg = int(
                self.spectrogram_nperseg
            )

            noverlap = int(
                self.spectrogram_noverlap
            )

            if len(data) < nperseg:

                nperseg = len(data)

                noverlap = min(
                    noverlap,
                    nperseg - 1,
                )

            if nperseg < 8:
                print(
                    "Spectrogram: segment too short"
                )
                return

            f, t, Sxx = spectrogram(
                data,
                fs=target_sr,
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
                detrend="constant",
                scaling="density",
                mode="psd",
            )

            print(
                f"Spectrogram: PSD shape = {Sxx.shape}"
            )

            # ----------------------------------------------------------
            # Store
            # ----------------------------------------------------------

            self.spectrogram_freqs = f

            self.spectrogram_times = (
                t + start_time
            )

            self.spectrogram_data = Sxx
            self._spectrogram_cache_key = cache_key

            # ----------------------------------------------------------
            # Frequency selection
            # ----------------------------------------------------------

            self._update_spectrogram_frequency_view()

            print(
                f"Spectrogram: image = "
                f"{self._spectrogram_image is not None}"
            )

            if self._spectrogram_image is not None:
                print(
                    f"Spectrogram: image size = "
                    f"{self._spectrogram_image.width()} x "
                    f"{self._spectrogram_image.height()}"
                )

            self.update()

        except Exception as exc:

            print(
                f"Spectrogram calculation failed: "
                f"{type(exc).__name__}: {exc}"
            )

            self.spectrogram_data = None
            self.spectrogram_freqs = None
            self.spectrogram_times = None
            self._spectrogram_image = None
            self._spectrogram_cache_key = None

            self.update()
            


    def _get_spectrogram_rect(self):
        """Return the dedicated spectrogram rectangle."""

        if not self.show_spectrogram:
            return None

        rect = self.rect()

        if hasattr(self, "scrollbar"):
            rect.setBottom(
                rect.bottom() - self.scrollbar.height()
            )

        label_width = 80
        depth_scale_width = 30

        plot_left = rect.left() + label_width
        plot_right = rect.right() - depth_scale_width

        # The spectrogram occupies the band between the trace area and the
        # time-axis labels. Use _get_plot_bounds to get the same plot_bottom
        # that the traces use, so the two stay consistent.
        _, _, traces_bottom = self._get_plot_bounds()

        spectrogram_height = self._get_spectrogram_height()

        # The spectrogram starts just below the trace area and ends where
        # the time axis labels begin (i.e. above the scrollbar with a small
        # margin for the labels).
        spectrogram_top = traces_bottom + 4
        spectrogram_bottom = (
            rect.bottom() - 25  # time axis label area
        )

        actual_height = spectrogram_bottom - spectrogram_top

        if actual_height < 20:
            return None

        return QRectF(
            float(plot_left),
            float(spectrogram_top),
            float(plot_right - plot_left),
            float(actual_height),
        )

    def _update_spectrogram_frequency_view(self):
        """Select the requested frequency range."""

        if self.spectrogram_data is None:
            self._spectrogram_image = None
            return

        if self.spectrogram_freqs is None:
            self._spectrogram_image = None
            return

        freqs = np.asarray(
            self.spectrogram_freqs
        )

        data = np.asarray(
            self.spectrogram_data
        )

        mask = (
            (freqs >= self.spectrogram_min_freq)
            & (freqs <= self.spectrogram_max_freq)
        )

        if not np.any(mask):
            print(
                "Spectrogram: frequency mask is empty"
            )
            self._spectrogram_image = None
            return

        self._spectrogram_display_freqs = (
            freqs[mask]
        )

        self._spectrogram_display_data = (
            data[mask, :]
        )

        print(
            "Spectrogram display:",
            self._spectrogram_display_data.shape
        )

        self._render_spectrogram_image()
        
    def _render_spectrogram_image(self):
        if self._spectrogram_display_data is None:
            self._spectrogram_image = None
            return

        # IMPORTANT: use the frequency-masked slice
        # (_spectrogram_display_data / _spectrogram_display_freqs, built
        # by _update_spectrogram_frequency_view from spectrogram_min_freq/
        # spectrogram_max_freq), NOT the full self.spectrogram_data. Using
        # the full array here made the min/max frequency controls appear
        # to do nothing: the mask was computed correctly but this method
        # ignored it and always rendered the whole PSD.
        Sxx_db = 10.0 * np.log10(
            np.maximum(self._spectrogram_display_data, np.finfo(float).tiny)
        )

        if self.spectrogram_auto_db:
            # Ignore extreme outliers when determining the automatic scale
            vmin = np.percentile(Sxx_db, 5)
            vmax = np.percentile(Sxx_db, 99)

            if vmax <= vmin:
                vmax = vmin + 1.0

            self.spectrogram_vmin = vmin
            self.spectrogram_vmax = vmax

            # Keep the GUI controls synchronized
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

        normalized = np.clip(
            (Sxx_db - vmin) / (vmax - vmin),
            0.0,
            1.0
        )

        # scipy.signal.spectrogram returns Sxx with row 0 = lowest
        # frequency, rows increasing with frequency. QImage row 0 is the
        # TOP of the image (raster convention, y increases downward). The
        # axis labels drawn in _draw_spectrogram put spectrogram_max_freq
        # at the top and spectrogram_min_freq at the bottom -- so without
        # flipping, the lowest frequency (row 0) was being drawn at the
        # top of the image while the label at that same position claimed
        # it was the MAXIMUM frequency. That's why a low frequency like
        # 7 Hz theta appeared near the top of a "0-40 Hz" axis instead of
        # near the bottom where it belongs. flipud() puts the highest
        # frequency row first (top of image) and lowest last (bottom of
        # image), matching the labels.
        normalized = np.flipud(normalized)

        cmap = colormaps.get_cmap(self.spectrogram_cmap)

        rgba = cmap(normalized)
        rgb = (rgba[..., :3] * 255).astype(np.uint8)
        rgb = np.ascontiguousarray(rgb)

        height, width, _ = rgb.shape

        self._spectrogram_image = QImage(
            rgb.data,
            width,
            height,
            width * 3,
            QImage.Format.Format_RGB888
        ).copy()
        
    def _draw_spectrogram(self, painter: QPainter):
        """Draw the spectrogram."""

        if not self.show_spectrogram:
            return

        spectrogram_rect = self._get_spectrogram_rect()

        if spectrogram_rect is None:
            return

        painter.save()

        # Background
        painter.fillRect(
            spectrogram_rect,
            self.background_color,
        )

        # --------------------------------------------------------------
        # Image
        # --------------------------------------------------------------

        if self._spectrogram_image is not None:

            # SmoothPixmapTransform makes drawImage interpolate (bilinear)
            # when scaling up instead of nearest-neighbor, which is what
            # was producing hard, blocky pixel edges -- the underlying
            # image is intentionally low-resolution (short time windows
            # per PSD frame, see spectrogram_nperseg/noverlap) and gets
            # stretched to fill spectrogram_rect, so without this the
            # individual PSD "blocks" are visible as sharp rectangles.
            # Scoped to just this draw call via save/restore so it
            # doesn't change how anything else in the widget renders.
            painter.setRenderHint(
                QPainter.RenderHint.SmoothPixmapTransform, True
            )

            painter.drawImage(
                spectrogram_rect,
                self._spectrogram_image,
            )

        else:

            # This should be visible if computation failed.
            painter.setPen(
                self.default_trace_color
            )

            painter.drawText(
                spectrogram_rect,
                Qt.AlignmentFlag.AlignCenter,
                "Spectrogram: no data",
            )

        painter.restore()

        # --------------------------------------------------------------
        # Border -- subtle (grid color, thin) rather than the heavy
        # trace-colored 2px border previously here, which visually
        # competed with the spectrogram content itself.
        # --------------------------------------------------------------

        painter.save()

        painter.setPen(
            QPen(
                self.grid_color,
                1.0,
            )
        )

        painter.drawRect(
            spectrogram_rect
        )

        # --------------------------------------------------------------
        # Labels -- small translucent background "pill" behind each
        # label so text stays legible over busy/bright parts of the
        # spectrogram image instead of just floating on top of it.
        # --------------------------------------------------------------

        painter.setRenderHint(
            QPainter.RenderHint.Antialiasing, True
        )

        painter.setFont(
            QFont(
                "Arial",
                8,
            )
        )

    def _draw_spectrogram(self, painter: QPainter):
        """Draw the spectrogram."""
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
        """Set spectrogram visibility."""

        enabled = bool(enabled)

        if self.show_spectrogram == enabled:
            self.spectrogram_control.set_enabled(enabled)
            return

        self.show_spectrogram = enabled
        self.spectrogram_control.set_enabled(enabled)

        if enabled:
            # If nothing has been picked yet for the spectrogram, default
            # to the first currently-selected trace channel rather than
            # silently doing nothing -- previously, enabling the
            # spectrogram before ever choosing a channel for it produced
            # no visible result and no feedback.
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
        """Set the channel used for the spectrogram."""

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
    # Navigation methods
    # ------------------------------------------------------------------

    def _get_min_start_time(self) -> float:
        """Get the minimum allowed start time."""
        if self.engine.timestamps_loaded and self.engine.timestamps is not None:
            return self.engine.timestamp_start
        else:
            return 0.0

    def _get_max_start_time(self) -> float:
        """Get the maximum allowed start time."""
        if self.engine.timestamps_loaded and self.engine.timestamps is not None:
            return self.engine.timestamp_end - self.window_duration
        else:
            return max(0.0, self.engine.total_duration - self.window_duration)

    def _get_total_duration(self) -> float:
        """Get the total duration of the data."""
        if self.engine.timestamps_loaded and self.engine.timestamps is not None:
            return self.engine.timestamp_duration
        else:
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
        """Update the start/end time labels."""
        if self._data_loaded:
            if self._use_timestamps and self.engine.timestamps_loaded:
                # Map to actual timestamp values
                start_idx = int(np.searchsorted(self.engine.timestamps, self.start_time, side='left'))
                end_idx = int(np.searchsorted(self.engine.timestamps, self.start_time + self.window_duration, side='right'))
                
                start_idx = max(0, min(start_idx, len(self.engine.timestamps) - 1))
                end_idx = max(0, min(end_idx, len(self.engine.timestamps) - 1))
                
                actual_start = self.engine.timestamps[start_idx]
                actual_end = self.engine.timestamps[end_idx]
                
                self.control_panel.set_time_display(actual_start, actual_end)
                return
            
            end_time = min(self.start_time + self.window_duration, self.engine.total_duration)
            self.control_panel.set_time_display(self.start_time, end_time)

    def _reset_view(self):
        """Reset to default view."""
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
        """Scroll up through channels (move to previous channel group)."""
        if not self._data_loaded or not self.channels:
            return
        
        # Keep the same time window but shift channels if needed
        # This could be used to scroll through channels if there are more than fit
        pass

    def _scroll_down(self):
        """Scroll down through channels."""
        if not self._data_loaded or not self.channels:
            return
        pass

        
    # ------------------------------------------------------------------
    # Depth sorting methods
    # ------------------------------------------------------------------

    def set_channel_depths(self, depths: dict[int, float]):
        self.channel_depths = depths
        self._sort_channels_by_depth()
        self._invalidate_cache()
        self.update()

    # ------------------------------------------------------------------
    # Ripple overlay API
    # ------------------------------------------------------------------

    def set_ripple_overlay(self, overlay: dict | None,
                            show_envelope: bool = True,
                            show_thresholds: bool = True,
                            show_events: bool = True):
        """
        Set (or replace) the ripple overlay data. `overlay` is
        {'channels': {channel: {...}}} as documented in __init__ --
        callers (RippleDialog) build this from detection results.
        Passing None clears the overlay.

        The three show_* flags are stored on the dict itself (defaults
        applied here) so paintEvent has one place to read them without
        the caller needing to pre-populate every key.
        """
        if overlay is not None:
            overlay = dict(overlay)  # shallow copy, don't mutate caller's dict
            overlay.setdefault('show_envelope', show_envelope)
            overlay.setdefault('show_thresholds', show_thresholds)
            overlay.setdefault('show_events', show_events)
        self.ripple_overlay = overlay
        self.update()

    def clear_ripple_overlay(self):
        self.ripple_overlay = None
        self.update()

    def set_ripple_overlay_visibility(self, show_envelope: bool | None = None,
                                       show_thresholds: bool | None = None,
                                       show_events: bool | None = None):
        """Toggle overlay layers without re-supplying all the data."""
        if self.ripple_overlay is None:
            return
        if show_envelope is not None:
            self.ripple_overlay['show_envelope'] = show_envelope
        if show_thresholds is not None:
            self.ripple_overlay['show_thresholds'] = show_thresholds
        if show_events is not None:
            self.ripple_overlay['show_events'] = show_events
        self.update()

    def _sort_channels_by_depth(self):
        if self.channel_depths:
            self._sorted_channels = sorted(
                self.channels,
                key=lambda ch: self.channel_depths.get(ch, 0.0)
            )
        else:
            self._sorted_channels = list(self.channels)

    def get_display_order(self) -> list[int]:
        return self._sorted_channels if self._sorted_channels else list(self.channels)

    # ------------------------------------------------------------------
    # Filter methods
    # ------------------------------------------------------------------

    def _apply_filters(self, data: np.ndarray) -> np.ndarray:
        """Apply all enabled filters using core/filters functions."""
        if not self.filter_enabled and not self.notch_enabled and not self.detrend_enabled:
            return data
        
        filtered_data = data.copy()
        
        # Detrend
        if self.detrend_enabled:
            from scipy.signal import detrend as scipy_detrend
            filtered_data = scipy_detrend(filtered_data, axis=0)
        
        # Bandpass/High-pass/Low-pass filter
        if self.filter_enabled:
            nyquist = self.engine.sr / 2
            low_freq = self.filter_low_freq
            high_freq = self.filter_high_freq
            
            # Validate frequencies
            low_freq = max(0.0, min(low_freq, nyquist - 1))
            high_freq = max(0.0, min(high_freq, nyquist - 1))
            
            if low_freq > 0 and high_freq > 0:
                # Bandpass filter
                if low_freq >= high_freq:
                    # Invalid range, swap or fix
                    low_freq, high_freq = min(low_freq, high_freq), max(low_freq, high_freq)
                
                filtered_data = bandpass_filter(
                    filtered_data, self.engine.sr,
                    low_freq, high_freq,
                    order=self.filter_order
                )
            elif high_freq > 0:
                # Low-pass filter (low_freq = 0)
                filtered_data = bandpass_filter(
                    filtered_data, self.engine.sr,
                    0.0, high_freq,  # low=0 means low-pass
                    order=self.filter_order
                )
            elif low_freq > 0:
                # High-pass filter (high_freq = 0)
                filtered_data = bandpass_filter(
                    filtered_data, self.engine.sr,
                    low_freq, 0.0,  # high=0 means high-pass
                    order=self.filter_order
                )
            else:
                # Both are 0 - no filtering needed, just return original
                return filtered_data
        
        # Notch filter
        if self.notch_enabled:
            filtered_data = notch_filter(
                filtered_data, self.engine.sr,
                self.notch_freq, quality_factor=self.notch_q
            )
        
        return filtered_data

    # ------------------------------------------------------------------
    # Public API for color and style customization
    # ------------------------------------------------------------------

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
    # Animation methods
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
    # Time navigation methods
    # ------------------------------------------------------------------

    def _cursor_to_time(self, fraction: float) -> float:
        """Convert fractional cursor position to time value."""
        if self.engine.timestamps_loaded and self.engine.timestamps is not None:
            min_time = self.engine.timestamp_start
            max_time = self.engine.timestamp_end
        else:
            min_time = 0.0
            max_time = self.engine.total_duration
        
        total_duration = max_time - min_time
        return min_time + fraction * total_duration

    def _time_to_cursor_fraction(self, time: float) -> float:
        """Convert time value to fractional cursor position."""
        if self.engine.timestamps_loaded and self.engine.timestamps is not None:
            min_time = self.engine.timestamp_start
            max_time = self.engine.timestamp_end
        else:
            min_time = 0.0
            max_time = self.engine.total_duration
        
        total_duration = max_time - min_time
        if total_duration > 0:
            return (time - min_time) / total_duration
        else:
            return 0.0

                
    def set_channels(self, channels: list[int]):
        """Set which channels to display."""
        self.channels = channels
        self._sort_channels_by_depth()
        
        # Update control panel channel label
        if hasattr(self, 'control_panel') and hasattr(self.control_panel, 'channel_info_label'):
            self.control_panel.channel_info_label.setText(f"{len(channels)} channels selected")
        
        # Update spectrogram control
        if hasattr(self, 'spectrogram_control'):
            self.spectrogram_control.set_channels(channels)
            if channels:
                self.spectrogram_control.set_channel(channels[0])
        
        self._invalidate_cache()
        self.update()

    def set_data_source(self, engine: TraceEngine):
        """Change data source."""
        self.engine = engine
        if engine.data_loaded:
            self._data_loaded = True
            self._setup_from_engine()
            # Enable timestamps checkbox if timestamps are available
            if engine.timestamps_loaded:
                self.control_panel.set_timestamps_available(True)
        else:
            self._data_loaded = False
            self.control_panel.set_timestamps_available(False)
        
        self._invalidate_cache()
        self.update()

    def _on_scrollbar_changed(self, value: int):
        """Handle scrollbar movement."""
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
        """Update scrollbar to reflect current start_time."""
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
        """Update scrollbar range."""
        if self._data_loaded and hasattr(self, 'scrollbar'):
            self.scrollbar.setMaximum(1000)
            self._update_scrollbar_position()



    def _zoom_to_cursor(self, factor: float, cursor_pos):
        """Zoom centered on cursor position."""
        if not self._data_loaded:
            return
        
        self._last_view = {
            'start_time': self.start_time,
            'window_duration': self.window_duration,
            'zoom_level': self._zoom_level,
            'channel_offset': self._channel_offset
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
        """Zoom in or out centered on view center."""
        if not self._data_loaded:
            return
        
        self._last_view = {
            'start_time': self.start_time,
            'window_duration': self.window_duration,
            'zoom_level': self._zoom_level,
            'channel_offset': self._channel_offset
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

        label_width = 80
        depth_scale_width = 30
        gap = 8

        plot_left = rect.left() + label_width
        plot_right = rect.right() - depth_scale_width

        time_axis_height = 25
        plot_bottom = rect.bottom() - time_axis_height

        if self.show_spectrogram:
            spectrogram_height = (self._get_spectrogram_height())
            plot_bottom -= (spectrogram_height + gap)

        return (plot_left,plot_right,plot_bottom,)
        
    # ------------------------------------------------------------------
    # Data methods
    # ------------------------------------------------------------------

    def _sample_to_time(self, sample: int) -> float:
        """Convert a sample index to a time value."""
        if self.engine.timestamps_loaded and self.engine.timestamps is not None:
            sample = max(0, min(sample, len(self.engine.timestamps) - 1))
            return float(self.engine.timestamps[sample])
        else:
            return sample / self.engine.sr

    def _time_to_sample(self, time: float) -> int:
        """Convert a time value to a sample index."""
        if self.engine.timestamps_loaded and self.engine.timestamps is not None:
            idx = int(np.searchsorted(self.engine.timestamps, time, side='left'))
            return max(0, min(idx, len(self.engine.timestamps) - 1))
        else:
            return int(time * self.engine.sr)

                    
    def _invalidate_cache(self):
        """
        Invalidate the trace-data cache.

        Spectrogram calculation is scheduled separately because it is
        considerably more expensive than retrieving/processing the trace
        data. During continuous navigation, the timer is restarted and the
        spectrogram is calculated only after navigation has settled.
        """

        self._cached_data = None
        self._cache_start = None
        self._cache_end = None
        self._cache_start_idx = None
        self._cache_end_idx = None

        # Filtered data cache
        self._filtered_cache.clear()

        # Spectrogram
        #
        # Do not calculate it immediately here. This function can be called
        # many times while dragging, scrolling, zooming, etc.
        self._spectrogram_cache_key = None

        if (
            self.show_spectrogram
            and self.spectrogram_channel is not None
        ):
            self._schedule_spectrogram_update()


    def _get_data_for_display(self):
        """Get data for the current time window."""
        if not self._data_loaded or not self.channels:
            return None
        
        # Check cache
        if self._cached_data is not None and self._cache_start is not None:
            if (self._cache_start <= self.start_time and 
                self._cache_end >= self.start_time + self.window_duration):
                return self._cached_data
        
        # Fetch new data - NO CLAMPING, just get whatever exists
        end_time = self.start_time + self.window_duration

        # Resolve the sample range ONCE, shared by every channel: all
        # channels are sliced from the same underlying (samples, n_ch)
        # array over the same time window, so they share identical
        # start/end sample indices. Capturing this here is what lets
        # _draw_traces later recover each plotted sample's TRUE time
        # (via engine.sample_to_time(start_idx + i)) instead of assuming
        # samples are spaced uniformly across the pixel width -- an
        # assumption that only holds when timestamps aren't loaded /
        # aren't jittery. Without this, the trace is drawn with uniform
        # spacing while cursors are placed at exact proportional time
        # positions, and the two silently drift apart as soon as
        # timestamps introduce any non-uniform sample spacing.
        start_idx, end_idx = self.engine.get_time_window_sample_range(
            self.start_time, end_time
        )

        data = {}
        for ch in self.channels:
            try:
                # Get channel data from engine - engine handles out-of-range gracefully
                channel_data = self.engine.get_channel_data(
                    ch, self.start_time, end_time
                )
                if channel_data is not None and len(channel_data) > 0:
                    data[ch] = channel_data
            except Exception as e:
                print(f"Error getting channel {ch}: {e}")
                continue
        
        # Apply filters if enabled
        if self.filter_enabled or self.notch_enabled or self.detrend_enabled:
            filtered_data = {}
            for ch, ch_data in data.items():
                filtered_data[ch] = self._apply_filters(ch_data)
            data = filtered_data
        
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
        """Paint the trace view."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Get plot area - leave space for scrollbar at bottom
        rect = self.rect()
        if hasattr(self, 'scrollbar'):
            scrollbar_height = self.scrollbar.height()
            rect.setBottom(rect.bottom() - scrollbar_height)
        
        # Fill background
        painter.fillRect(rect, self.background_color)
        
        # Check if data is loaded
        if not self._data_loaded:
            painter.setPen(QColor("#888888"))
            font = QFont()
            font.setPointSize(14)
            painter.setFont(font)
            painter.drawText(
                rect, Qt.AlignmentFlag.AlignCenter, "No data loaded."
            )
            return
        
        if not self.channels:
            painter.setPen(QColor("#888888"))
            font = QFont()
            font.setPointSize(12)
            painter.setFont(font)
            painter.drawText(
                rect, Qt.AlignmentFlag.AlignCenter,
                "Select channels from the probe map to display traces."
            )
            return
        
        if self.show_grid:
            self._draw_grid(painter, rect)
        
        data = self._get_data_for_display()
        
        # Draw traces (even if data is empty, just skip drawing)
        self._draw_traces(painter, rect, data,)

        if self.ripple_overlay is not None:
            self._draw_ripple_overlay(painter)

        if self.show_spectrogram:
            self._draw_spectrogram(painter)
            
        # Draw vertical time cursors
        self._draw_time_cursors(painter, rect)

        # Draw time axis
        if self.show_time_axis:
            self._draw_time_axis(painter, rect)
        
        # Draw channel labels
        if self.show_channel_labels:
            self._draw_channel_labels(painter, rect, data)
        
        # Draw depth scale
        if self.show_depth_scale:
            self._draw_depth_scale(painter, rect, data)

    def _draw_time_cursors(self, painter: QPainter, rect: QRectF):
        """Draw vertical time cursors with trash bin for selected cursor."""
        if not self._time_cursors:
            return
        
        plot_left, plot_right, _ = self._get_plot_bounds()
        plot_width = plot_right - plot_left
        
        # Store trash bin hitboxes for click detection
        self._trash_bin_rects = {}
        
        for i, cursor_fraction in enumerate(self._time_cursors):
            # Convert fractional position to time
            cursor_time = self._cursor_to_time(cursor_fraction)
            
            # Check if cursor is within current view
            if cursor_time < self.start_time or cursor_time > self.start_time + self.window_duration:
                continue
            
            # Calculate x position
            x_ratio = (cursor_time - self.start_time) / self.window_duration
            x = plot_left + x_ratio * plot_width
            
            is_selected = (i == self._selected_cursor)
            
            # Draw vertical line
            if is_selected:
                pen = QPen(QColor("#ff6b6b"), 3, Qt.PenStyle.DashLine)
            else:
                pen = QPen(QColor("#ff6b6b"), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            
            painter.drawLine(
                int(x), int(rect.top()),
                int(x), int(rect.bottom())
            )
            
            # Draw time label
            painter.setPen(QPen(QColor("#ff6b6b"), 1))
            font = QFont()
            font.setPointSize(8)
            painter.setFont(font)
            painter.drawText(
                int(x - 25), int(rect.top() + 5), 50, 15,
                Qt.AlignmentFlag.AlignCenter,
                f"{cursor_time:.3f}s"
            )
            
            # Draw trash bin for selected cursor - positioned to the LEFT of the line
            if is_selected:
                self._draw_trash_bin(painter, x - 12, rect.top())
                # Store trash bin hitbox
                self._trash_bin_rects[i] = QRectF(
                    int(x) - 22, int(rect.top()) + 20, 20, 20
                )
            
            painter.setPen(pen)
            
    def _draw_trash_bin(self, painter: QPainter, x: float, top: float):
        """Draw a small trash bin icon."""
        # Trash bin at top of cursor
        bin_x = int(x) - 16  # Center the bin on the x position
        bin_y = int(top) + 22
        bin_size = 14  # Slightly smaller
        
        # Create colors
        bin_color = QColor("#ff6b6b")
        bin_fill = QColor("#ff6b6b")
        bin_fill.setAlpha(50)
        
        # Draw trash bin body
        painter.setPen(QPen(bin_color, 2))
        painter.setBrush(bin_fill)
        
        # Draw rectangle for bin body
        painter.drawRoundedRect(
            bin_x, bin_y + 3, bin_size, bin_size - 3, 2, 2
        )
        
        # Draw lid
        painter.drawLine(
            bin_x - 1, bin_y + 3,
            bin_x + bin_size + 1, bin_y + 3
        )
        
        # Draw handle
        painter.drawLine(
            bin_x + 3, bin_y + 3,
            bin_x + 3, bin_y + 1
        )
        painter.drawLine(
            bin_x + 3, bin_y + 1,
            bin_x + bin_size - 3, bin_y + 1
        )
        painter.drawLine(
            bin_x + bin_size - 3, bin_y + 1,
            bin_x + bin_size - 3, bin_y + 3
        )
        
        # Draw vertical lines inside bin
        painter.setPen(QPen(bin_color, 1))
        painter.drawLine(bin_x + 4, bin_y + 6, bin_x + 4, bin_y + bin_size - 2)
        painter.drawLine(bin_x + bin_size - 4, bin_y + 6, bin_x + bin_size - 4, bin_y + bin_size - 2)
        

    def _compute_time_ticks(self, rect: QRectF):
        """
        Compute major and minor time tick positions.

        Returns
        -------
        major_ticks : list of (x_pixel, time_value)
        minor_ticks : list of (x_pixel, time_value)
        """
        label_width = 80
        depth_scale_width = 30

        plot_left = rect.left() + label_width
        plot_right = rect.right() - depth_scale_width
        plot_width = plot_right - plot_left

        if plot_width <= 0 or self.window_duration <= 0:
            return [], []

        start_time = self.start_time
        end_time = self.start_time + self.window_duration

        # ---- Choose a "nice" major interval ----
        # Target ~5-8 major ticks across the window.
        target_major_count = 6
        raw_interval = self.window_duration / target_major_count

        # Round to a "nice" value: 1, 2, 5, 10, 20, 50, 100...
        import math
        magnitude = 10 ** math.floor(math.log10(raw_interval))
        for multiplier in (1, 2, 5, 10):
            major_interval = multiplier * magnitude
            if major_interval >= raw_interval:
                break

        # ---- Choose a "nice" minor interval ----
        # 5 minor ticks per major tick is standard and uncluttered.
        minor_interval = major_interval / 5.0

        # ---- Snap the first major tick to a multiple of major_interval ----
        first_major = math.ceil(start_time / major_interval) * major_interval
        first_minor = math.ceil(start_time / minor_interval) * minor_interval

        major_ticks = []
        minor_ticks = []

        t = first_major
        while t <= end_time + 1e-9:
            x_ratio = (t - start_time) / self.window_duration
            x = plot_left + x_ratio * plot_width
            major_ticks.append((x, t))
            t += major_interval

        t = first_minor
        while t <= end_time + 1e-9:
            # Skip if this is also a major tick
            is_major = any(abs(t - mt) < minor_interval * 0.01 for _, mt in major_ticks)
            if not is_major:
                x_ratio = (t - start_time) / self.window_duration
                x = plot_left + x_ratio * plot_width
                minor_ticks.append((x, t))
            t += minor_interval

        return major_ticks, minor_ticks

    def _draw_grid(self, painter: QPainter, rect: QRectF):
        """Draw the grid aligned with the time ticks."""
        major_ticks, minor_ticks = self._compute_time_ticks(rect)

        plot_left, plot_right, plot_bottom = self._get_plot_bounds()

        # ---- Minor grid lines (subtle, no labels) ----
        minor_pen = QPen(QColor(255, 255, 255, 15), 1, Qt.PenStyle.DotLine)
        painter.setPen(minor_pen)
        for x, _ in minor_ticks:
            painter.drawLine(
                int(x), int(rect.top()),
                int(x), int(plot_bottom)
            )

        # ---- Major grid lines (more visible) ----
        major_pen = QPen(QColor(255, 255, 255, 40), 1, Qt.PenStyle.DotLine)
        painter.setPen(major_pen)
        for x, _ in major_ticks:
            painter.drawLine(
                int(x), int(rect.top()),
                int(x), int(plot_bottom)
            )

        # ---- Horizontal grid lines (5 divisions) ----
        painter.setPen(QPen(self.grid_color, 1, Qt.PenStyle.DotLine))
        num_h_lines = 5
        for i in range(num_h_lines + 1):
            y = rect.top() + ((plot_bottom - rect.top()) * i / num_h_lines)
            painter.drawLine(
                int(rect.left()), int(y), int(rect.right()), int(y)
            )


    def _draw_traces(
        self,
        painter: QPainter,
        rect: QRectF,
        data: dict[int, np.ndarray],
    ):
        """
        Draw neural traces.

        The trace region ends at plot_bottom, leaving the lower part of
        the widget available for the spectrogram.
        """

        if not data:
            return

        if not self._sorted_channels:
            return

        # --------------------------------------------------------------
        # Plot geometry
        # --------------------------------------------------------------
        plot_left, plot_right, plot_bottom = self._get_plot_bounds()

        plot_top = rect.top()

        plot_width = plot_right - plot_left
        plot_height = plot_bottom - plot_top

        if plot_width <= 0 or plot_height <= 0:
            return

        n_channels = len(self._sorted_channels)

        if n_channels <= 0:
            return

        channel_height = plot_height / n_channels

        # --------------------------------------------------------------
        # Time information
        # --------------------------------------------------------------
        start_time = float(self.start_time)
        end_time = start_time + float(self.window_duration)

        if end_time <= start_time:
            return

        # --------------------------------------------------------------
        # Determine display scale
        # --------------------------------------------------------------
        global_min = None
        global_max = None

        if not self.auto_scale:
            # Fixed/global scaling.
            #
            # Calculate the range once instead of once per channel.
            for channel in self._sorted_channels:

                if channel not in data:
                    continue

                channel_data = np.asarray(
                    data[channel],
                    dtype=np.float64,
                )

                if channel_data.size == 0:
                    continue

                finite = np.isfinite(channel_data)

                if not np.any(finite):
                    continue

                channel_data = channel_data[finite]

                ch_min = float(np.min(channel_data))
                ch_max = float(np.max(channel_data))

                if global_min is None or ch_min < global_min:
                    global_min = ch_min

                if global_max is None or ch_max > global_max:
                    global_max = ch_max

        if (
            global_min is None
            or global_max is None
        ):
            global_min = -1.0
            global_max = 1.0

        global_range = global_max - global_min

        if global_range <= 0:
            global_range = 1.0

        # --------------------------------------------------------------
        # Determine number of points to draw
        # --------------------------------------------------------------
        #
        # Drawing every sample from a 30 kHz recording is unnecessary.
        # Limit the number of points approximately to the number of pixels.
        #
        max_points = max(
            200,
            int(plot_width * 2),
        )

        # --------------------------------------------------------------
        # Draw each channel
        # --------------------------------------------------------------
        for idx, channel in enumerate(
            self._sorted_channels
        ):

            if channel not in data:
                continue

            channel_data = np.asarray(
                data[channel],
                dtype=np.float64,
            )

            if channel_data.size == 0:
                continue

            # ----------------------------------------------------------
            # Channel geometry
            # ----------------------------------------------------------
            y_center = (
                plot_bottom
                - (idx + 0.5) * channel_height
                + self._channel_offset * channel_height
            )

            # Keep traces inside the trace area.
            if (
                y_center + channel_height < plot_top
                or y_center - channel_height > plot_bottom
            ):
                continue

            # ----------------------------------------------------------
            # Downsample only for drawing
            # ----------------------------------------------------------
            n_samples = len(channel_data)

            if n_samples > max_points:

                step = int(
                    np.ceil(
                        n_samples / max_points
                    )
                )

                draw_data = channel_data[::step]

            else:
                draw_data = channel_data

            if draw_data.size == 0:
                continue

            # ----------------------------------------------------------
            # Handle NaNs
            # ----------------------------------------------------------
            finite = np.isfinite(draw_data)

            if not np.any(finite):
                continue

            # ----------------------------------------------------------
            # Vertical scaling
            # ----------------------------------------------------------
            if self.auto_scale:

                finite_values = draw_data[finite]

                data_min = float(
                    np.min(finite_values)
                )

                data_max = float(
                    np.max(finite_values)
                )

                data_range = data_max - data_min

                if data_range <= 0:
                    data_range = 1.0

                # Leave a small margin around each trace.
                scale = (
                    channel_height * 0.40
                ) / data_range

                y_values = (
                    y_center
                    - (
                        draw_data
                        - (data_min + data_max) / 2.0
                    ) * scale
                )

            else:

                # Global scaling.
                scale = (
                    channel_height * 0.40
                ) / global_range

                y_values = (
                    y_center
                    - (
                        draw_data
                        - (global_min + global_max) / 2.0
                    ) * scale
                )

            # ----------------------------------------------------------
            # X coordinates
            # ----------------------------------------------------------
            #
            # The cached trace data corresponds to the current time
            # interval. Use sample positions rather than assuming that
            # every sample has exactly the same timestamp.
            # ----------------------------------------------------------
            n_draw = len(draw_data)

            if n_draw == 1:
                x_values = np.array(
                    [plot_left],
                    dtype=np.float64,
                )

            else:
                x_values = np.linspace(
                    plot_left,
                    plot_right,
                    n_draw,
                )

            # ----------------------------------------------------------
            # Create painter path
            # ----------------------------------------------------------
            path = QPainterPath()

            started = False

            for x, y, valid in zip(
                x_values,
                y_values,
                np.isfinite(draw_data),
            ):

                if not valid or not np.isfinite(y):
                    started = False
                    continue

                # Keep the trace inside the trace region.
                y = float(
                    np.clip(
                        y,
                        plot_top,
                        plot_bottom,
                    )
                )

                if not started:
                    path.moveTo(
                        float(x),
                        y,
                    )
                    started = True

                else:
                    path.lineTo(
                        float(x),
                        y,
                    )

            # ----------------------------------------------------------
            # Channel color
            # ----------------------------------------------------------
            color = self.channel_colors.get(
                channel,
                self.default_trace_color,
            )

            pen = QPen(
                color,
                self.trace_width,
            )

            painter.setPen(pen)
            painter.drawPath(path)

            
    def _draw_ripple_overlay(
        self,
        painter: QPainter,
    ):
        """
        Draw ripple overlay on top of the neural traces.

        The overlay is restricted to the trace region and does not extend
        into the dedicated spectrogram region.
        """

        if self.ripple_overlay is None:
            return

        if not self._sorted_channels:
            return

        overlay = self.ripple_overlay

        overlay_channels = overlay.get(
            "channels",
            {},
        )

        if not overlay_channels:
            return

        # --------------------------------------------------------------
        # Plot geometry
        # --------------------------------------------------------------
        rect = self.rect()

        if hasattr(self, "scrollbar"):
            rect.setBottom(
                rect.bottom()
                - self.scrollbar.height()
            )

        plot_left, plot_right, plot_bottom = (
            self._get_plot_bounds()
        )

        plot_top = rect.top()

        plot_width = plot_right - plot_left
        plot_height = plot_bottom - plot_top

        if plot_width <= 0 or plot_height <= 0:
            return

        n_channels = len(
            self._sorted_channels
        )

        if n_channels <= 0:
            return

        channel_height = (
            plot_height / n_channels
        )

        # --------------------------------------------------------------
        # Time range
        # --------------------------------------------------------------
        start_time = float(
            self.start_time
        )

        end_time = (
            start_time
            + float(self.window_duration)
        )

        if end_time <= start_time:
            return

        # --------------------------------------------------------------
        # Overlay options
        # --------------------------------------------------------------
        show_envelope = overlay.get(
            "show_envelope",
            True,
        )

        show_thresholds = overlay.get(
            "show_thresholds",
            True,
        )

        show_events = overlay.get(
            "show_events",
            True,
        )

        # --------------------------------------------------------------
        # Draw each channel
        # --------------------------------------------------------------
        for idx, channel in enumerate(
            self._sorted_channels
        ):

            channel_overlay = overlay_channels.get(
                channel
            )

            if channel_overlay is None:
                continue

            # ----------------------------------------------------------
            # Channel geometry
            # ----------------------------------------------------------
            y_center = (
                plot_bottom
                - (idx + 0.5) * channel_height
                + self._channel_offset * channel_height
            )

            # ----------------------------------------------------------
            # Signal/envelope
            # ----------------------------------------------------------
            envelope = channel_overlay.get(
                "envelope"
            )

            signal = channel_overlay.get(
                "signal"
            )

            sample_offset = int(
                channel_overlay.get(
                    "sample_offset",
                    0,
                )
            )

            sample_rate = float(
                channel_overlay.get(
                    "sample_rate",
                    self.engine.sr,
                )
            )

            if sample_rate <= 0:
                continue

            # ----------------------------------------------------------
            # Select the signal to use for geometry
            # ----------------------------------------------------------
            if envelope is not None:
                envelope = np.asarray(
                    envelope,
                    dtype=np.float64,
                )

            if signal is not None:
                signal = np.asarray(
                    signal,
                    dtype=np.float64,
                )

            # ----------------------------------------------------------
            # Draw envelope
            # ----------------------------------------------------------
            if (
                show_envelope
                and envelope is not None
                and envelope.size > 0
            ):

                n_samples = len(
                    envelope
                )

                times = (
                    sample_offset
                    + np.arange(n_samples)
                ) / sample_rate

                times += start_time

                mask = (
                    (times >= start_time)
                    & (times <= end_time)
                    & np.isfinite(envelope)
                )

                if np.any(mask):

                    times_plot = times[mask]
                    env_plot = envelope[mask]

                    x = (
                        plot_left
                        + (
                            (times_plot - start_time)
                            / (end_time - start_time)
                        )
                        * plot_width
                    )

                    # --------------------------------------------------
                    # Envelope scaling
                    # --------------------------------------------------
                    finite_env = env_plot[
                        np.isfinite(env_plot)
                    ]

                    if finite_env.size:

                        env_min = float(
                            np.min(finite_env)
                        )

                        env_max = float(
                            np.max(finite_env)
                        )

                        env_range = (
                            env_max - env_min
                        )

                        if env_range <= 0:
                            env_range = 1.0

                        # Envelope is drawn in the upper/lower portion
                        # of the corresponding trace lane.
                        env_scale = (
                            channel_height * 0.30
                        ) / env_range

                        y = (
                            y_center
                            - (
                                env_plot
                                - (
                                    env_min
                                    + env_max
                                ) / 2.0
                            )
                            * env_scale
                        )

                        path = QPainterPath()

                        started = False

                        for xi, yi in zip(
                            x,
                            y,
                        ):

                            if not np.isfinite(yi):
                                started = False
                                continue

                            yi = float(
                                np.clip(
                                    yi,
                                    plot_top,
                                    plot_bottom,
                                )
                            )

                            if not started:
                                path.moveTo(
                                    float(xi),
                                    yi,
                                )
                                started = True
                            else:
                                path.lineTo(
                                    float(xi),
                                    yi,
                                )

                        painter.save()

                        pen = QPen(
                            QColor("#ffcc00"),
                            1.0,
                        )

                        painter.setPen(pen)
                        painter.drawPath(path)

                        painter.restore()

            # ----------------------------------------------------------
            # Thresholds
            # ----------------------------------------------------------
            if (
                show_thresholds
                and envelope is not None
                and envelope.size > 0
            ):

                peak_threshold_sd = channel_overlay.get(
                    "peak_threshold_sd"
                )

                boundary_threshold_sd = channel_overlay.get(
                    "boundary_threshold_sd"
                )

                finite_env = envelope[
                    np.isfinite(envelope)
                ]

                if (
                    finite_env.size > 0
                    and (
                        peak_threshold_sd is not None
                        or boundary_threshold_sd is not None
                    )
                ):

                    mean_env = float(
                        np.mean(finite_env)
                    )

                    sd_env = float(
                        np.std(finite_env)
                    )

                    # Use the same envelope range as above for
                    # converting threshold values to display coordinates.
                    env_min = float(
                        np.min(finite_env)
                    )

                    env_max = float(
                        np.max(finite_env)
                    )

                    env_range = (
                        env_max - env_min
                    )

                    if env_range <= 0:
                        env_range = 1.0

                    env_scale = (
                        channel_height * 0.30
                    ) / env_range

                    def threshold_y(
                        threshold_value
                    ):
                        return (
                            y_center
                            - (
                                threshold_value
                                - (
                                    env_min
                                    + env_max
                                ) / 2.0
                            )
                            * env_scale
                        )

                    painter.save()

                    threshold_pen = QPen(
                        QColor("#ff6666"),
                        1.0,
                        Qt.PenStyle.DashLine,
                    )

                    painter.setPen(
                        threshold_pen
                    )

                    if peak_threshold_sd is not None:

                        peak_threshold = (
                            mean_env
                            + float(
                                peak_threshold_sd
                            ) * sd_env
                        )

                        y = threshold_y(
                            peak_threshold
                        )

                        if plot_top <= y <= plot_bottom:
                            painter.drawLine(
                                int(plot_left),
                                int(y),
                                int(plot_right),
                                int(y),
                            )

                    if boundary_threshold_sd is not None:

                        boundary_threshold = (
                            mean_env
                            + float(
                                boundary_threshold_sd
                            ) * sd_env
                        )

                        y = threshold_y(
                            boundary_threshold
                        )

                        if plot_top <= y <= plot_bottom:
                            painter.drawLine(
                                int(plot_left),
                                int(y),
                                int(plot_right),
                                int(y),
                            )

                    painter.restore()

            # ----------------------------------------------------------
            # Ripple events
            # ----------------------------------------------------------
            if (
                show_events
                and channel_overlay.get(
                    "events"
                )
            ):

                events = channel_overlay.get(
                    "events",
                    [],
                )

                painter.save()

                event_pen = QPen(
                    QColor("#ff4444"),
                    1.5,
                )

                painter.setPen(
                    event_pen
                )

                for event in events:

                    # Support RippleEvent-like objects.
                    if hasattr(
                        event,
                        "start_time",
                    ):
                        event_start = float(
                            event.start_time
                        )
                    elif isinstance(
                        event,
                        dict,
                    ):
                        event_start = float(
                            event.get(
                                "start_time",
                                event.get(
                                    "start",
                                    0.0,
                                ),
                            )
                        )
                    else:
                        continue

                    if hasattr(
                        event,
                        "end_time",
                    ):
                        event_end = float(
                            event.end_time
                        )
                    elif isinstance(
                        event,
                        dict,
                    ):
                        event_end = float(
                            event.get(
                                "end_time",
                                event.get(
                                    "end",
                                    event_start,
                                ),
                            )
                        )
                    else:
                        event_end = event_start

                    # Ignore events outside the current window.
                    if (
                        event_end < start_time
                        or event_start > end_time
                    ):
                        continue

                    event_start = max(
                        event_start,
                        start_time,
                    )

                    event_end = min(
                        event_end,
                        end_time,
                    )

                    x1 = (
                        plot_left
                        + (
                            (event_start - start_time)
                            / (end_time - start_time)
                        )
                        * plot_width
                    )

                    x2 = (
                        plot_left
                        + (
                            (event_end - start_time)
                            / (end_time - start_time)
                        )
                        * plot_width
                    )

                    # Highlight the ripple interval in its channel lane.
                    event_rect = QRectF(
                        float(x1),
                        float(
                            y_center
                            - channel_height * 0.45
                        ),
                        float(
                            max(
                                1.0,
                                x2 - x1,
                            )
                        ),
                        float(
                            channel_height * 0.9
                        ),
                    )

                    painter.fillRect(
                        event_rect,
                        QBrush(
                            QColor(
                                255,
                                80,
                                80,
                                45,
                            )
                        ),
                    )

                    painter.drawRect(
                        event_rect
                    )

                painter.restore()

    def _draw_time_axis(self, painter: QPainter, rect: QRectF):
        """Draw time axis labels and tick marks, aligned with the grid."""
        painter.setPen(QColor("#888888"))
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)

        plot_left, plot_right, plot_bottom = self._get_plot_bounds()

        major_ticks, minor_ticks = self._compute_time_ticks(rect)

        # ---- Minor tick marks (short, at the bottom of the plot area) ----
        tick_len_minor = 4
        tick_len_major = 8

        painter.setPen(QPen(QColor("#777777"), 1))
        for x, _ in minor_ticks:
            painter.drawLine(
                int(x), int(plot_bottom),
                int(x), int(plot_bottom + tick_len_minor)
            )

        # ---- Major tick marks (longer) ----
        painter.setPen(QPen(QColor("#aaaaaa"), 1))
        for x, _ in major_ticks:
            painter.drawLine(
                int(x), int(plot_bottom),
                int(x), int(plot_bottom + tick_len_major)
            )

        # ---- Major tick labels ----
        painter.setPen(QColor("#aaaaaa"))
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)

        for x, t in major_ticks:
            if self._use_timestamps and self.engine.timestamps_loaded:
                idx = int(np.searchsorted(self.engine.timestamps, t, side='left'))
                idx = max(0, min(idx, len(self.engine.timestamps) - 1))
                actual_time = self.engine.timestamps[idx]
                label = f"{actual_time:.3f}s"
            else:
                label = f"{t:.2f}s"

            # Center the label on the tick
            text_width = 60
            text_x = int(x - text_width / 2)
            # Keep labels inside the widget
            text_x = max(text_x, 2)

            painter.drawText(
                text_x, int(plot_bottom + tick_len_major + 2),
                text_width, 14,
                Qt.AlignmentFlag.AlignCenter,
                label
            )


    def _draw_channel_labels(self, painter: QPainter, rect: QRectF, data: dict):
        """Draw channel labels on the left side."""
        display_channels = self.get_display_order()
        display_channels = [ch for ch in display_channels if ch in data]
        
        if not display_channels:
            display_channels = self.get_display_order()
        
        n_channels = len(display_channels)
        
        # Get plot bounds to reserve spectrogram space
        _, _, plot_bottom = self._get_plot_bounds()
        
        channel_height = (plot_bottom - rect.top()) / max(1, n_channels)
        
        label_width = 80
        label_bg = QColor(0, 0, 0, 100)
        painter.fillRect(
            int(rect.left()), int(rect.top()),
            label_width, int(plot_bottom - rect.top()),
            label_bg
        )
        
        painter.setPen(QColor("#aaaaaa"))
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        
        for idx, ch in enumerate(display_channels):
            y_center = plot_bottom - (idx + 0.5) * channel_height + self._channel_offset * channel_height
            
            # Channel number
            painter.drawText(
                int(rect.left()) + 5, int(y_center - 8), 35, 16,
                Qt.AlignmentFlag.AlignLeft,
                f"CH{ch}"
            )
            
            # Depth if available
            if ch in self.channel_depths:
                depth = self.channel_depths[ch]
                painter.drawText(
                    int(rect.left()) + 40, int(y_center - 8), 35, 16,
                    Qt.AlignmentFlag.AlignLeft,
                    f"{depth:.0f}"
                )
            
            # Color indicator
            color = self.channel_colors.get(ch, self.default_trace_color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(color))
            painter.drawRect(int(rect.left()) + 70, int(y_center - 3), 6, 6)
            painter.setPen(QColor("#aaaaaa"))
            
    '''
    def _draw_depth_scale(self, painter: QPainter, rect: QRectF, data: dict):
        """Draw depth scale on the right side."""
        painter.setPen(QColor("#888888"))
        font = QFont()
        font.setPointSize(7)
        painter.setFont(font)
        
        # Reserve space for spectrogram at bottom
        scale_bottom = rect.bottom()
        if self.show_spectrogram:
            scale_bottom = rect.bottom() - int(rect.height() * 0.25) - 25
        
        # Draw depth labels
        depth_scale_width = 30
        right_x = int(rect.right() - depth_scale_width - 50)
        
        painter.drawText(
            right_x, int(rect.top()) + 10, 25, 15,
            Qt.AlignmentFlag.AlignRight,
            "Depth"
        )
        
        painter.drawText(
            right_x, int(rect.top()) + 25, 25, 15,
            Qt.AlignmentFlag.AlignRight,
            "↑"
        )
        
        painter.drawText(
            right_x, int(scale_bottom) - 10, 25, 15,
            Qt.AlignmentFlag.AlignRight,
            "↓"
        )
    '''
    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def wheelEvent(self, event):
        """Handle mouse wheel for scrolling and zooming."""
        if not self._data_loaded:
            return
        
        delta = event.angleDelta().y()
        
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Ctrl+wheel for zoom - ZOOM TO CURSOR
            if delta > 0:
                self._zoom_to_cursor(1.2, event.position())
            else:
                self._zoom_to_cursor(1/1.2, event.position())
        elif event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            # Shift+wheel for faster scrolling (5x)
            time_delta = (delta / 120.0) * 0.5 * self.window_duration
            min_start = self._get_min_start_time()
            max_start = self._get_max_start_time()
            self.start_time = min(
                max(min_start, self.start_time - time_delta), max_start
            )
            self._update_time_labels()
            self._invalidate_cache()
            self.update()
        else:
            # Regular wheel for time scroll
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
        """Handle keyboard events."""
        if not self._data_loaded:
            super().keyPressEvent(event)
            return
        
        if event.key() == Qt.Key.Key_Plus or event.key() == Qt.Key.Key_Equal:
            self._zoom(1.2)
        elif event.key() == Qt.Key.Key_Minus:
            self._zoom(1/1.2)
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
        """Handle right-click context menu."""
        if not self._data_loaded:
            return
        
        plot_left, plot_right, _ = self._get_plot_bounds()
        plot_width = plot_right - plot_left
        
        # Check if right-clicked on a cursor
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
        
        # Right-clicked on empty area
        menu = QMenu(self)
        
        clear_cursors_action = menu.addAction("Clear All Time Cursors")
        clear_cursors_action.triggered.connect(self._clear_time_cursors)
        
        menu.exec(event.globalPos())

    def _remove_cursor(self, index: int):
        """Remove a specific cursor."""
        if 0 <= index < len(self._time_cursors):
            self._time_cursors.pop(index)
            self._selected_cursor = None
            self.update()

    def _clear_time_cursors(self):
        """Clear all vertical time cursors."""
        self._time_cursors.clear()
        self._selected_cursor = None
        self.update()
        
    def mousePressEvent(self, event):
        """Handle mouse press for panning, adding cursors, and selecting cursors."""
        if event.button() == Qt.MouseButton.LeftButton:
            # Check if we clicked on a trash bin
            if hasattr(self, '_trash_bin_rects'):
                for cursor_idx, trash_rect in self._trash_bin_rects.items():
                    if trash_rect.contains(event.position()):
                        self._remove_cursor(cursor_idx)
                        event.accept()
                        return
            
            # Check if we clicked on a cursor
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
            
            # Otherwise, start panning
            self._dragging = True
            self._drag_start_pos = event.position()
            self._drag_start_time = self.start_time
            self._drag_start_offset = self._channel_offset
            event.accept()
            
        elif event.button() == Qt.MouseButton.RightButton:
            if self._data_loaded:
                plot_left, plot_right, _ = self._get_plot_bounds()
                plot_width = plot_right - plot_left
                
                # First check if we're right-clicking on an existing cursor
                for i, cursor_fraction in enumerate(self._time_cursors):
                    cursor_time = self._cursor_to_time(cursor_fraction)
                    x_ratio = (cursor_time - self.start_time) / self.window_duration
                    cursor_x = plot_left + x_ratio * plot_width
                    
                    if abs(event.position().x() - cursor_x) < 10:
                        self._selected_cursor = i
                        self.update()
                        event.accept()
                        return
                
                # If not on a cursor, add a new one
                x_ratio = (event.position().x() - plot_left) / max(1, plot_width)
                cursor_time = self.start_time + x_ratio * self.window_duration
                
                # Convert to fractional position
                cursor_fraction = self._time_to_cursor_fraction(cursor_time)
                
                self._time_cursors.append(cursor_fraction)
                self._time_cursors.sort()
                self._selected_cursor = len(self._time_cursors) - 1
                self.update()
                
                event.accept()
                

    def mouseMoveEvent(self, event):
        """Handle mouse move for panning and dragging cursors."""
        if self._dragging_cursor and self._selected_cursor is not None:
            # Dragging a cursor
            plot_left, plot_right, _ = self._get_plot_bounds()
            plot_width = plot_right - plot_left
            
            delta_x = event.position().x() - self._drag_cursor_start_x
            time_delta_per_pixel = self.window_duration / max(1, plot_width)
            
            # Get current cursor time
            current_fraction = self._time_cursors[self._selected_cursor]
            current_time = self._cursor_to_time(current_fraction)
            
            # Calculate new time
            new_time = current_time + delta_x * time_delta_per_pixel
            
            # Clamp to valid range
            min_time = self._get_min_start_time()
            max_time = self._get_max_start_time() + self.window_duration
            new_time = max(min_time, min(new_time, max_time))
            
            # Convert back to fractional position
            new_fraction = self._time_to_cursor_fraction(new_time)
            
            self._time_cursors[self._selected_cursor] = new_fraction
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
            
            # Clamp to valid range
            min_start = self._get_min_start_time()
            max_start = self._get_max_start_time()
            self.start_time = min(
                max(min_start, self._drag_start_time + time_delta), max_start
            )
            
            channel_height = self.height() / max(1, len(self.channels))
            channel_delta = delta_y / max(1, channel_height)
            self._channel_offset = self._drag_start_offset + channel_delta
            
            self._update_time_labels()
            self._invalidate_cache()
            self.update()
            
            event.accept()

    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if self._dragging_cursor:
            self._dragging_cursor = False
            event.accept()
            return
        
        if hasattr(self, '_dragging') and self._dragging:
            self._dragging = False
            self._drag_start_pos = None
            
            # Save the current view as "last good view" - NO SNAPPING
            self._last_view = {
                'start_time': self.start_time,
                'window_duration': self.window_duration,
                'zoom_level': self._zoom_level,
                'channel_offset': self._channel_offset
            }
            
            event.accept()
            

    def mouseDoubleClickEvent(self, event):
        """Handle double-click to reset view."""
        if not self._data_loaded:
            return
        
        # Get cursor position
        cursor_pos = event.position()
        
        # Calculate time at cursor position
        plot_left, plot_right, _ = self._get_plot_bounds()
        plot_width = plot_right - plot_left
        
        x_ratio = (cursor_pos.x() - plot_left) / max(1, plot_width)
        cursor_time = self.start_time + x_ratio * self.window_duration
        
        # Reset to 10 seconds centered on cursor
        new_duration = min(10.0, self._get_total_duration())
        new_start = cursor_time - new_duration / 2
        
        min_start = self._get_min_start_time()
        max_start = self._get_max_start_time()
        new_start = max(min_start, min(new_start, max_start))
        
        # Reset channels to correct y-position
        self._channel_offset = 0.0
        self._zoom_level = 1.0
        
        self.start_time = new_start
        self.window_duration = new_duration
        
        # Update control panel
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

    Controls:
        - Enable/disable spectrogram
        - Channel
        - Frequency range
        - dB color range
        - Colormap
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

        layout.setContentsMargins(
            4, 4, 4, 4
        )

        layout.setSpacing(4)

        # --------------------------------------------------------------
        # Enable
        # --------------------------------------------------------------
        self.enabled_checkbox = QCheckBox(
            "Show Spectrogram"
        )

        self.enabled_checkbox.setChecked(False)

        self.enabled_checkbox.toggled.connect(
            self._on_enabled_toggled
        )

        layout.addWidget(
            self.enabled_checkbox
        )

        # --------------------------------------------------------------
        # Channel
        # --------------------------------------------------------------
        channel_group = QGroupBox(
            "Channel"
        )

        channel_layout = QVBoxLayout(
            channel_group
        )

        self.channel_combo = QComboBox()

        self.channel_combo.currentIndexChanged.connect(
            self._on_channel_changed
        )

        channel_layout.addWidget(
            self.channel_combo
        )

        layout.addWidget(
            channel_group
        )

        # --------------------------------------------------------------
        # Frequency range
        # --------------------------------------------------------------
        freq_group = QGroupBox(
            "Frequency Range (Hz)"
        )

        freq_layout = QGridLayout(
            freq_group
        )

        freq_layout.addWidget(
            QLabel("Min:"),
            0,
            0,
        )

        self.min_freq_spin = QDoubleSpinBox()

        self.min_freq_spin.setRange(
            0.0,
            49.0,
        )

        self.min_freq_spin.setDecimals(1)

        self.min_freq_spin.setSingleStep(
            1.0
        )

        self.min_freq_spin.setValue(
            0.0
        )

        self.min_freq_spin.valueChanged.connect(
            self._on_min_freq_changed
        )

        freq_layout.addWidget(
            self.min_freq_spin,
            0,
            1,
        )

        freq_layout.addWidget(
            QLabel("Max:"),
            1,
            0,
        )

        self.max_freq_spin = QDoubleSpinBox()

        self.max_freq_spin.setRange(
            1.0,
            50.0,
        )

        self.max_freq_spin.setDecimals(1)

        self.max_freq_spin.setSingleStep(
            1.0
        )

        self.max_freq_spin.setValue(
            40.0
        )

        self.max_freq_spin.valueChanged.connect(
            self._on_max_freq_changed
        )

        freq_layout.addWidget(
            self.max_freq_spin,
            1,
            1,
        )

        layout.addWidget(
            freq_group
        )

        # --------------------------------------------------------------
        # Color scale
        # --------------------------------------------------------------
        # layout.addWidget(QLabel("Color scale:"))

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



        # --------------------------------------------------------------
        # Colormap
        # --------------------------------------------------------------
        cmap_group = QGroupBox(
            "Colormap"
        )

        cmap_layout = QVBoxLayout(
            cmap_group
        )

        self.cmap_combo = QComboBox()

        self.cmap_combo.addItems(
            [
                "viridis",
                "plasma",
                "inferno",
                "magma",
                "cividis",
                "jet",
            ]
        )

        self.cmap_combo.setCurrentText(
            "viridis"
        )

        self.cmap_combo.currentTextChanged.connect(
            self._on_cmap_changed
        )

        cmap_layout.addWidget(
            self.cmap_combo
        )

        layout.addWidget(
            cmap_group
        )

        # layout.addStretch()

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _on_min_db_changed(self, value: float):
        self.minDbChanged.emit(value)


    def _on_max_db_changed(self, value: float):
        self.maxDbChanged.emit(value)


    def _on_auto_db_changed(self, checked: bool):
        self.min_db_spin.setEnabled(not checked)
        self.max_db_spin.setEnabled(not checked)
        self.autoDbChanged.emit(checked)

        
    def _on_enabled_toggled(self, checked: bool):
        self.enabledChanged.emit(
            bool(checked)
        )

    def _on_channel_changed(self, index: int):
        if self._building_channel_list:
            return

        if index < 0:
            return

        channel = self.channel_combo.itemData(
            index
        )

        if channel is not None:
            self.channelChanged.emit(
                int(channel)
            )

    def _on_min_freq_changed(self, value: float):
        self.minFreqChanged.emit(
            float(value)
        )

    def _on_max_freq_changed(self, value: float):
        self.maxFreqChanged.emit(
            float(value)
        )


    def _on_cmap_changed(self, name: str):
        self.cmapChanged.emit(
            str(name)
        )

    # ------------------------------------------------------------------
    # Public setters
    # ------------------------------------------------------------------

    def set_channels(self, channels: list[int]):
        """Update available spectrogram channels."""

        self._building_channel_list = True

        current_channel = None

        if self.channel_combo.currentIndex() >= 0:
            current_channel = self.channel_combo.currentData()

        self.channel_combo.clear()

        for channel in channels:
            self.channel_combo.addItem(
                f"CH{channel}",
                int(channel),
            )

        # Preserve current channel when possible.
        if current_channel is not None:
            self.set_channel(
                int(current_channel)
            )

        self._building_channel_list = False

    def set_channel(self, channel: int):
        """Select a channel in the combo box."""

        for i in range(
            self.channel_combo.count()
        ):
            if (
                self.channel_combo.itemData(i)
                == channel
            ):
                self._building_channel_list = True

                self.channel_combo.setCurrentIndex(
                    i
                )

                self._building_channel_list = False

                return

    def set_enabled(self, enabled: bool):
        """Set spectrogram visibility checkbox."""

        enabled = bool(enabled)

        if (
            self.enabled_checkbox.isChecked()
            != enabled
        ):
            self.enabled_checkbox.setChecked(
                enabled
            )

    def set_min_db(self, value: float):
        """Set minimum dB display value."""

        self.min_db_spin.setValue(
            float(value)
        )

    def set_max_db(self, value: float):
        """Set maximum dB display value."""

        self.max_db_spin.setValue(
            float(value)
        )
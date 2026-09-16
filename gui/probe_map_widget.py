"""
probe_map_widget.py

Interactive Neuropixels probe map, visually and behaviorally inspired by
Phy2 / Kilosort4's channel view.

Design goals (replacing the matplotlib-based ProbeSelector):
  - Single batched ScatterPlotItem for all electrodes instead of one
    Rectangle + one Text per channel -> one draw call instead of ~800.
  - pyqtgraph's ViewBox for GPU-accelerated, truly smooth pan/zoom
    ("infinite zoom" feel), with wheel-zoom-to-cursor like KS4.
  - Hit-testing delegated to pyqtgraph's spatial lookup (pointsAt),
    not a per-frame Python loop over every channel.
  - Level-of-detail: channel number labels only render past a zoom
    threshold and only for currently visible channels.
  - Dark theme, circular markers, shank coloring, subtle depth grid,
    consistent with Phy2/KS4 conventions.

This module has NO dependency on matplotlib and no dependency on the
rest of data_explorer other than the plain dict produced by
neuropixels_probe_extractor.extract_probes_from_settings(...)[
    'probes'][<key>] (i.e. a dict with 'coordinates' and 'shanks' keys).

Usage
-----
    from PyQt6.QtWidgets import QApplication
    from data_explorer.gui.probe_map_widget import ProbeMapWidget

    app = QApplication([])
    widget = ProbeMapWidget(probe_data)
    widget.channelsSelected.connect(lambda chans: print("selected:", chans))
    widget.show()
    app.exec()
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal, QPointF
from PyQt6.QtGui import QColor, QFont, QPen, QBrush
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QCheckBox,
    QGraphicsRectItem, QGraphicsEllipseItem, QSlider,
)

# ---------------------------------------------------------------------------
# Visual constants (tuned to look like Phy2 / Kilosort4)
# ---------------------------------------------------------------------------

BG_COLOR = "#1e1e1e"
GRID_COLOR = (255, 255, 255, 18)          # faint white, low alpha
LABEL_COLOR = (210, 210, 210, 200)
LABEL_COLOR = (255, 255, 255, 255)


COLOR_ACTIVE = QColor("#3a86ff")          # default / unselected channel
COLOR_SELECTED = QColor("#42d77d")        # selected for viewing
COLOR_EXCLUDED = QColor("#5a5a5a")        # excluded / rejected
COLOR_HOVER_RING = QColor("#ffd23f")      # hover halo

SHANK_PALETTE = [
    QColor("#3a86ff"), QColor("#ff6b6b"), QColor("#ffd23f"),
    QColor("#42d77d"), QColor("#c77dff"), QColor("#ff9f1c"),
    QColor("#4cc9f0"), QColor("#f72585"),
]

MIN_LABEL_SPACING_PX = 4  # only draw channel-number text once neighboring
                            # electrodes are at least this far apart on
                            # screen -- calibrated against actual electrode
                            # pitch (see _min_electrode_pitch_um), not a
                            # fixed px-per-um constant which breaks across
                            # probes with different geometries.


class ProbeMapWidget(QWidget):
    """
    Interactive probe map: click electrodes to select/deselect, pan/zoom
    freely (mouse drag = pan, wheel = zoom-to-cursor), double-click empty
    space to clear the current selection (mirrors phy2's behavior).

    Emits
    -----
    channelsSelected(list[int])
        Whenever the selection set changes, sorted by depth ascending.
    """

    channelsSelected = pyqtSignal(list)

    def __init__(self, probe_data: dict, parent=None):
        super().__init__(parent)

        coords = probe_data["coordinates"]
        shanks = probe_data.get("shanks", {})

        self.channels = np.asarray(coords["channels"], dtype=np.int64)
        self.xcoords = np.asarray(coords["x"], dtype=np.float64)
        self.ycoords = np.asarray(coords["y"], dtype=np.float64)

        if shanks.get("ids"):
            self.shank_ids = np.asarray(shanks["ids"], dtype=np.int64)
        else:
            self.shank_ids = np.zeros(len(self.channels), dtype=np.int64)

        n = len(self.channels)
        # Per-channel state, tracked as arrays (not per-artist attributes)
        self.state = np.zeros(n, dtype=np.uint8)  # 0=active 1=selected 2=excluded
        self._hover_idx: int | None = None

        # channel -> row index, for O(1) lookups
        self._chan_to_idx = {int(ch): i for i, ch in enumerate(self.channels)}

        self._min_pitch_um = self._min_electrode_pitch_um()
        self.base_font_size = 9  # Default label size
        
        self.setMinimumSize(400, 600)

        self._build_ui()
        self._build_scatter()
        self._update_colors()
        self._auto_range()

    def _min_electrode_pitch_um(self) -> float:
        """Smallest center-to-center distance between any two electrodes,
        used to decide when they're visually 'resolvable' enough to label."""
        if len(self.ycoords) < 2:
            return 20.0
        uy = np.unique(self.ycoords)
        if len(uy) < 2:
            return 20.0
        return float(np.min(np.diff(np.sort(uy))))

    # ------------------------------------------------------------------
    # UI scaffolding
    # ------------------------------------------------------------------

    def _build_ui(self):
        pg.setConfigOption("background", BG_COLOR)
        pg.setConfigOption("foreground", "#d0d0d0")
        pg.setConfigOption("antialias", True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ---- top bar: counts + actions (thin, phy2-style) ----
        bar = QHBoxLayout()
        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: #b0b0b0; font-size: 11px;")
        bar.addWidget(self.status_label)
        bar.addStretch(1)

        self.labels_checkbox = QCheckBox("Labels")
        self.labels_checkbox.setChecked(True)
        self.labels_checkbox.stateChanged.connect(self._on_toggle_labels)
        bar.addWidget(self.labels_checkbox)

        btn_select_all = QPushButton("Select All")
        btn_select_all.clicked.connect(self.select_all)
        bar.addWidget(btn_select_all)

        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self.clear_selection)
        bar.addWidget(btn_clear)

        layout.addLayout(bar)

        # ---- Size control bar ----
        size_bar = QHBoxLayout()
        size_bar.addWidget(QLabel("Point Size:"))
        
        # Point size slider
        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setMinimum(4)
        self.size_slider.setMaximum(50)
        self.size_slider.setValue(14)  # Default value
        self.size_slider.setTickInterval(2)
        self.size_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.size_slider.setFixedWidth(150)
        self.size_slider.valueChanged.connect(self._on_size_changed)
        size_bar.addWidget(self.size_slider)
        
        # In the size bar, add another slider for labels
        label_size_bar = QHBoxLayout()
        label_size_bar.addWidget(QLabel("Label Size:"))

        self.label_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.label_size_slider.setMinimum(4)
        self.label_size_slider.setMaximum(50)
        self.label_size_slider.setValue(9)  # Default
        self.label_size_slider.setTickInterval(2)
        self.label_size_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.label_size_slider.setFixedWidth(150)
        self.label_size_slider.valueChanged.connect(self._on_label_size_changed)
        label_size_bar.addWidget(self.label_size_slider)

        self.label_size_label = QLabel("9")
        self.label_size_label.setStyleSheet("color: #b0b0b0; font-size: 11px; min-width: 20px;")
        label_size_bar.addWidget(self.label_size_label)
        label_size_bar.addStretch(1)

        layout.addLayout(label_size_bar)


            
        # Size value display
        self.size_label = QLabel("14")
        self.size_label.setStyleSheet("color: #b0b0b0; font-size: 11px; min-width: 20px;")
        size_bar.addWidget(self.size_label)
        size_bar.addStretch(1)
        
        layout.addLayout(size_bar)

        # ---- main plot area ----
        # ... rest of the code

        # ---- main plot area ----
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setAspectLocked(True)          # true physical aspect, phy2-style
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setMenuEnabled(False)
        self.plot_widget.hideButtons()
        self.plot_widget.setLabel("bottom", "X", units="\u00b5m")
        self.plot_widget.setLabel("left", "Depth", units="\u00b5m")

        self.view_box = self.plot_widget.getViewBox()
        self.view_box.setMouseMode(pg.ViewBox.PanMode)
        # Wheel zoom already zooms to cursor by default in pyqtgraph.

        layout.addWidget(self.plot_widget, stretch=1)

        # Depth gridlines drawn manually (thin horizontal lines every
        # unique y) so they stay visually subordinate to the electrodes,
        # matching KS4's faint horizontal reference lines.
        self._draw_depth_grid()

        # Mouse interaction
        self.plot_widget.scene().sigMouseClicked.connect(self._on_mouse_clicked)
        self.plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)

        self._update_status()


    def _on_label_size_changed(self, value):
        """Update label base size when slider changes."""
        self.label_size_label.setText(str(value))
        self.base_font_size = value
        self._update_label_visibility()
            
    def _on_size_changed(self, value):
        """Update marker size when slider changes."""
        self.size_label.setText(str(value))
        self.scatter.setSize(value)
        # Remove this line - it doesn't exist:
        # self.scatter.setHoverSize(value * 1.4)
        # The hover size is set during scatter creation and can't be changed later
        
        
    def _draw_depth_grid(self):
        unique_y = np.unique(self.ycoords)
        # Thin out if there are a huge number of rows (e.g. dense NP2 probes)
        max_lines = 60
        if len(unique_y) > max_lines:
            step = max(1, len(unique_y) // max_lines)
            unique_y = unique_y[::step]

        pen = pg.mkPen(color=GRID_COLOR, width=1)
        x_min, x_max = self.xcoords.min() - 40, self.xcoords.max() + 40
        for y in unique_y:
            line = pg.PlotCurveItem(
                x=[x_min, x_max], y=[y, y], pen=pen
            )
            line.setZValue(-10)
            self.plot_widget.addItem(line)

    # ------------------------------------------------------------------
    # Scatter (electrodes) — single batched item
    # ------------------------------------------------------------------

    def _build_scatter(self):
        initial_size = self._marker_size_px()
        self.scatter = pg.ScatterPlotItem(
            x=self.xcoords,
            y=self.ycoords,
            size=initial_size,
            pxMode=True,
            symbol="o",
            pen=pg.mkPen(None),
            brush=pg.mkBrush(COLOR_ACTIVE),
            hoverable=False,
        )
        
        self.scatter.setZValue(10)
        self.plot_widget.addItem(self.scatter)

        # Text labels: created lazily/hidden, one pg.TextItem per channel
        # is still ~800 objects, but they are only made *visible* (and thus
        # only cost paint time) when zoomed in and in view -- see
        # _update_label_visibility. This matches phy2's LOD behavior.
        self._label_items: list[pg.TextItem | None] = [None] * len(self.channels)
        self.view_box.sigRangeChanged.connect(self._on_range_changed)

    def _marker_size_px(self) -> float:
        # Use slider value if available, otherwise default
        if hasattr(self, 'size_slider'):
            return float(self.size_slider.value())
        return 14.0  # Default fallback
    # ------------------------------------------------------------------
    # Color / state updates (batched — no per-artist mutation)
    # ------------------------------------------------------------------

    def _update_colors(self):
        has_multi_shank = len(np.unique(self.shank_ids)) > 1
        brushes = []
        pens = []
        for i in range(len(self.channels)):
            st = self.state[i]
            if st == 1:
                color = COLOR_SELECTED
            elif st == 2:
                color = COLOR_EXCLUDED
            elif has_multi_shank:
                color = SHANK_PALETTE[int(self.shank_ids[i]) % len(SHANK_PALETTE)]
            else:
                color = COLOR_ACTIVE

            if i == self._hover_idx:
                pens.append(pg.mkPen(COLOR_HOVER_RING, width=2))
            else:
                pens.append(pg.mkPen(None))
            brushes.append(pg.mkBrush(color))

        self.scatter.setBrush(brushes)
        self.scatter.setPen(pens)
        self._update_status()

    def _update_status(self):
        n_sel = int(np.sum(self.state == 1))
        n_exc = int(np.sum(self.state == 2))
        total = len(self.channels)
        self.status_label.setText(
            f"{n_sel} selected  \u00b7  {n_exc} excluded  \u00b7  {total} total"
        )

    # ------------------------------------------------------------------
    # Interaction: click to select/deselect
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Interaction: hover-tracks the electrode under the cursor via
    # sigMouseMoved -> _on_mouse_moved -> pointsAt() (one hit-test per
    # mouse-move event, cached in self._hover_idx); a click then just
    # toggles whatever self._hover_idx currently is, rather than
    # re-running pointsAt() a second time on click. This is why
    # ScatterPlotItem's own sigClicked/sigHovered aren't used here.
    # ------------------------------------------------------------------

    def _on_mouse_clicked(self, ev):
        # Only handle left clicks
        if ev.button() != Qt.MouseButton.LeftButton:
            return

        # If we are currently hovering over a channel, select it
        if self._hover_idx is not None:
            idx = self._hover_idx

            if self.state[idx] == 1:
                self.state[idx] = 0
            else:
                self.state[idx] = 1

            self._update_colors()
            self._emit_selection()

            ev.accept()
            

    def _on_scatter_hovered(self, plot_item, points, ev):
        # Check if points is a numpy array and has elements
        if points is not None and hasattr(points, 'size') and points.size > 0:
            new_hover = points[0].index()
        else:
            new_hover = None
        
        if new_hover != self._hover_idx:
            self._hover_idx = new_hover
            self._update_colors()



    def _on_mouse_moved(self, scene_pos):
        if not self.plot_widget.sceneBoundingRect().contains(scene_pos):
            return
        
        points_at = self.scatter.pointsAt(
            self.view_box.mapSceneToView(scene_pos)
        )
        
        # Pass the points to hover handler
        self._on_scatter_hovered(self.scatter, points_at, None)

        


    def _emit_selection(self):
        selected_channels = self.channels[self.state == 1]
        order = np.argsort(self.ycoords[self.state == 1])
        selected_sorted = selected_channels[order].tolist()
        self.channelsSelected.emit(selected_sorted)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_all(self):
        self.state[self.state != 2] = 1
        self._update_colors()
        self._emit_selection()

    def clear_selection(self):
        self.state[self.state == 1] = 0
        self._update_colors()
        self._emit_selection()

    def set_excluded(self, channel_list):
        excluded = set(int(c) for c in channel_list)
        for ch, idx in self._chan_to_idx.items():
            self.state[idx] = 2 if ch in excluded else (
                0 if self.state[idx] == 2 else self.state[idx]
            )
        self._update_colors()

    def set_selected_channels(self, channel_list):
        """Select exactly the given channels (replacing any current
        selection), leaving excluded channels untouched. Used to seed
        this widget with an externally-provided starting selection --
        e.g. RippleTriggeredAverageDialog opens a scoped ProbeMapWidget
        instance pre-populated with whatever channels were already
        active in the main trace view."""
        wanted = set(int(c) for c in channel_list)
        for ch, idx in self._chan_to_idx.items():
            if self.state[idx] == 2:  # leave excluded channels alone
                continue
            self.state[idx] = 1 if ch in wanted else 0
        self._update_colors()
        self._emit_selection()

    def get_selected_channels(self) -> list[int]:
        selected_channels = self.channels[self.state == 1]
        order = np.argsort(self.ycoords[self.state == 1])
        return selected_channels[order].tolist()

    # ------------------------------------------------------------------
    # Level-of-detail labels + view range handling
    # ------------------------------------------------------------------

    def _on_range_changed(self, *_):
        self._update_label_visibility()

    def _update_label_visibility(self):
        (x0, x1), (y0, y1) = self.view_box.viewRange()
        view_height_um = max(y1 - y0, 1e-6)
        
        # Use slider value for base font size
        base_font_size = self.base_font_size if hasattr(self, 'base_font_size') else 9
        reference_height = 500
        zoom_factor = max(0.5, min(3.0, reference_height / view_height_um))
        
        # Get max from slider if available
        max_label_size = self.label_size_slider.maximum() if hasattr(self, 'label_size_slider') else 30
        font_size = max(4, min(max_label_size, base_font_size * zoom_factor))


        show_labels = self.labels_checkbox.isChecked()
        
        if not show_labels:
            for item in self._label_items:
                if item is not None:
                    item.setVisible(False)
            return
        
        visible_mask = (
            (self.xcoords >= x0) & (self.xcoords <= x1)
            & (self.ycoords >= y0) & (self.ycoords <= y1)
        )
        visible_idx = np.where(visible_mask)[0]
        
        # Don't cap labels - show all visible ones (pyqtgraph handles performance)
        # If performance is an issue, keep MAX_LABELS but increase it
        MAX_LABELS = 800  # Increased to show all 384 channels
        if len(visible_idx) > MAX_LABELS:
            visible_idx = visible_idx[:MAX_LABELS]
        
        visible_set = set(visible_idx.tolist())
        
        for i in range(len(self.channels)):
            want_visible = i in visible_set
            item = self._label_items[i]
            
            if want_visible and item is None:
                # Create new label
                item = pg.TextItem(
                    text=str(int(self.channels[i])),
                    color=LABEL_COLOR,
                    anchor=(0.5, 0.5),
                )
                font = QFont()
                font.setPointSizeF(font_size)
                font.setBold(True)
                item.setFont(font)
                item.setPos(self.xcoords[i], self.ycoords[i])
                item.setZValue(20)
                self.plot_widget.addItem(item)
                self._label_items[i] = item
                
            elif item is not None:
                # Update visibility and font size
                item.setVisible(want_visible)
                if want_visible:
                    # Update font size - need to create new QFont and set it
                    new_font = QFont()
                    new_font.setPointSizeF(font_size)
                    new_font.setBold(True)
                    item.setFont(new_font)
                    
                    # Make sure color is correct
                    item.setColor(QColor(255, 255, 255, 255))



                    

    def _on_toggle_labels(self, _state):
        self._update_label_visibility()

    def _auto_range(self):
        pad_x = max((self.xcoords.max() - self.xcoords.min()) * 0.15, 30)
        pad_y = max((self.ycoords.max() - self.ycoords.min()) * 0.03, 30)
        self.view_box.setRange(
            xRange=(self.xcoords.min() - pad_x, self.xcoords.max() + pad_x),
            yRange=(self.ycoords.min() - pad_y, self.ycoords.max() + pad_y),
            padding=0,
        )
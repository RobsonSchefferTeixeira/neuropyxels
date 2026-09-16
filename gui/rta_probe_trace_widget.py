"""
rta_probe_trace_widget.py

Renders ripple-triggered-average results as small trace glyphs
positioned at each electrode's position on the probe -- phy2's
waveform-view idiom applied to ripple-triggered averages instead of
spike waveforms.

Layout (phy-style)
------------------
- Only the averaged trace (and optional SEM band) is drawn -- no box,
  no per-glyph baseline, no channel label. Each glyph is just the
  waveform itself.
- Channels are grouped by shank_id. Each shank gets its own horizontal
  column, ordered left-to-right by ASCENDING shank ID (0, 1, 3, ...
  if some shanks are missing from the current selection). Shanks with
  no channels in the selection simply don't produce a column -- the
  remaining shanks keep their ID order.
- Within a column, channels are placed by RELATIVE depth rank (sorted
  by real y). Spacing between neighbors is uniform, deliberately not
  proportional to real µm pitch -- matching phy.
- A vertical zero-trigger line is drawn at t=0 through every glyph in
  every column.
- Optional SINGLE-COLUMN mode (via the "Single column" checkbox):
  all shanks' channels are merged into one column, ranked by relative
  depth across the entire selection. Per-shank coloring is retained so
  shanks stay distinguishable without a column each.

Zoom / pan model
----------------
Three INDEPENDENT controls, all driven from the mouse:

  - Plain wheel        -> geometric ZOOM (both axes).
  - Shift + wheel      -> LATERAL zoom (horizontal only).
  - Ctrl + wheel       -> amplitude GAIN (vertical trace deflection
                          only).

  - Drag               -> pan (both axes).
  - Double-click       -> full reset (all axes + pan + gain).
  - F                  -> fit whole LATERAL screen (horizontal-only
                          reset).
  - 0                  -> full reset (same as double-click).

See the previous version of this module's docstring for the full
explanation of the zoom/DC-offset model; this version adds column
ordering by shank ID and the single-column toggle but leaves all of
that behavior unchanged.
"""

from __future__ import annotations

import numpy as np

from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPolygonF
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel


SHANK_PALETTE = [
    QColor("#4a9eff"), QColor("#ff6b6b"), QColor("#42d77d"),
    QColor("#ffd23f"), QColor("#c77dff"), QColor("#ff9f1c"),
    QColor("#4cc9f0"), QColor("#f72585"),
]

BG_COLOR = QColor("#1e1e1e")
DIVIDER_COLOR = QColor(255, 255, 255, 50)
ZERO_TRIGGER_COLOR = QColor(255, 255, 255, 70)
HUD_COLOR = QColor(200, 200, 200, 200)
HEADER_TEXT_COLOR = QColor("#b0b0b0")


def _shank_color(shank_id: int) -> QColor:
    """Color for a shank ID. Used for BOTH per-column trace tinting and
    (in single-column mode) per-trace tinting -- same mapping either
    way, so a shank has one consistent color throughout the display."""
    return SHANK_PALETTE[int(shank_id) % len(SHANK_PALETTE)]


class _RTACanvas(QWidget):
    """The actual painted canvas. All the layout math and rendering
    lives here; RTAProbeTraceWidget wraps it with a small header."""

    # Fixed design dimensions at zoom == 1.0.
    _DESIGN_GLYPH_WIDTH_PX = 90.0
    _DESIGN_SLOT_SPACING_PX = 42.0
    _DESIGN_COLUMN_SPACING_PX = 140.0

    _ZOOM_MIN = 0.1
    _ZOOM_MAX = 20.0
    _GAIN_MIN = 0.05
    _GAIN_MAX = 50.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 260)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Geometry (set via set_geometry_from_probe)
        self._channels: np.ndarray = np.array([], dtype=np.int64)
        self._xcoords: np.ndarray = np.array([], dtype=np.float64)
        self._ycoords: np.ndarray = np.array([], dtype=np.float64)
        self._shank_ids: np.ndarray = np.array([], dtype=np.int64)
        self._chan_to_idx: dict[int, int] = {}

        # Result data
        self._time_axis: np.ndarray | None = None
        self._channel_data: dict[int, dict] = {}
        self._show_sem = True

        # Layout mode: False = one column per shank (default), True =
        # merge every shank into a single column.
        self._single_column = False

        # Layout cache
        self._layout_valid = False
        self._column_of_channel: dict[int, int] = {}
        self._column_order: list[int] = []                # shank ids, or [-1] in single-column mode
        self._column_shank_color: dict[int, QColor] = {}  # column index -> column tint
        self._shank_of_channel: dict[int, int] = {}       # per-channel shank id (for trace color)
        self._slot_of_channel: dict[int, int] = {}
        self._slots_per_column: dict[int, int] = {}

        # View state
        self._zoom_x = 1.0
        self._zoom_y = 1.0
        self._amplitude_gain = 1.0
        self._view_x_center = 0.0
        self._view_y_center = 0.0

        self._dragging = False
        self._drag_start_pos: QPointF | None = None
        self._drag_start_view_center = (0.0, 0.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_geometry_from_probe(self, probe_data: dict, channels: list[int]):
        coords = probe_data["coordinates"]
        shanks = probe_data.get("shanks", {})

        all_channels = np.asarray(coords["channels"], dtype=np.int64)
        all_x = np.asarray(coords["x"], dtype=np.float64)
        all_y = np.asarray(coords["y"], dtype=np.float64)
        if shanks.get("ids"):
            all_shank_ids = np.asarray(shanks["ids"], dtype=np.int64)
        else:
            all_shank_ids = np.zeros(len(all_channels), dtype=np.int64)

        wanted = set(int(c) for c in channels)
        mask = np.array([int(c) in wanted for c in all_channels], dtype=bool)

        self._channels = all_channels[mask]
        self._xcoords = all_x[mask]
        self._ycoords = all_y[mask]
        self._shank_ids = all_shank_ids[mask]
        self._chan_to_idx = {int(ch): i for i, ch in enumerate(self._channels)}

        self._layout_valid = False
        self._auto_range()
        self.update()

    def set_result(self, time_axis: np.ndarray, channel_data: dict[int, dict], show_sem: bool = True):
        self._time_axis = np.asarray(time_axis, dtype=np.float64)
        self._channel_data = channel_data
        self._show_sem = show_sem
        self.update()

    def set_show_sem(self, show_sem: bool):
        self._show_sem = show_sem
        self.update()

    def set_single_column(self, single: bool):
        """Toggle single-column mode. Rebuilds the layout; preserves
        current zoom/pan since the user's view shouldn't be reset just
        because the grouping changed."""
        single = bool(single)
        if single == self._single_column:
            return
        self._single_column = single
        self._layout_valid = False
        self._ensure_layout()
        # Re-center horizontally so the new layout is visible, but keep
        # zoom_x/zoom_y and vertical center untouched -- the user was
        # probably inspecting something in particular.
        n_columns = len(self._column_order) or 1
        self._view_x_center = (n_columns - 1) / 2.0
        self._zoom_x = self._fit_zoom_x_for_columns(n_columns)
        self.update()

    def is_single_column(self) -> bool:
        return self._single_column

    def reset_view(self):
        self._auto_range()

    def fit_horizontal(self):
        self._fit_horizontal()
        self.update()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _ensure_layout(self):
        if self._layout_valid or len(self._channels) == 0:
            return

        self._column_of_channel = {}
        self._slot_of_channel = {}
        self._slots_per_column = {}
        self._column_shank_color = {}
        self._shank_of_channel = {}

        if self._single_column:
            # -------- Single-column mode --------
            # One column (index 0) containing every selected channel
            # across every shank. Slot assignment is by relative depth
            # rank across ALL channels, sorted shallowest-first.
            #
            # The column's "shank id" is set to -1 (sentinel), which
            # means the column divider / zero-trigger code treats it as
            # just a normal column and never tries to label it with a
            # shank id. Per-trace color is still driven by each
            # channel's REAL shank id (via _shank_of_channel), so
            # shanks stay distinguishable.
            self._column_order = [-1]
            self._column_shank_color[0] = QColor("#e0e0e0")  # unused in this mode, kept for safety

            order = np.argsort(-self._ycoords)
            sorted_channels = self._channels[order]

            n = len(sorted_channels)
            self._slots_per_column[0] = n
            for slot, ch in enumerate(sorted_channels):
                ch_i = int(ch)
                self._column_of_channel[ch_i] = 0
                self._slot_of_channel[ch_i] = slot
                self._shank_of_channel[ch_i] = int(self._shank_ids[self._chan_to_idx[ch_i]])

        else:
            # -------- One column per shank --------
            # Order columns by ASCENDING shank ID. Missing shanks
            # (present on the probe but with no selected channels) are
            # simply absent -- no empty column is inserted for them,
            # matching "if one shank is skipped, you just draw 0, 1, 3
            # side by side but keeping the shank order."
            unique_shanks_present = sorted(set(int(s) for s in self._shank_ids))
            self._column_order = list(unique_shanks_present)

            for col_idx, shank in enumerate(self._column_order):
                self._column_shank_color[col_idx] = _shank_color(shank)

                shank_mask = self._shank_ids == shank
                shank_channels = self._channels[shank_mask]
                shank_ys = self._ycoords[shank_mask]

                order = np.argsort(-shank_ys)  # shallowest first
                sorted_channels = shank_channels[order]

                n = len(sorted_channels)
                self._slots_per_column[col_idx] = n
                for slot, ch in enumerate(sorted_channels):
                    ch_i = int(ch)
                    self._column_of_channel[ch_i] = col_idx
                    self._slot_of_channel[ch_i] = slot
                    self._shank_of_channel[ch_i] = shank

        self._layout_valid = True

    def _max_slots(self) -> int:
        if not self._slots_per_column:
            return 1
        return max(self._slots_per_column.values())

    def _fit_zoom_x_for_columns(self, n_columns: int) -> float:
        rect = self.rect()
        available_w = max(50, rect.width() - 40)
        needed_w = (max(1, n_columns) - 1) * self._DESIGN_COLUMN_SPACING_PX + self._DESIGN_GLYPH_WIDTH_PX
        if needed_w <= 0:
            return 1.0
        return max(self._ZOOM_MIN, min(available_w / needed_w, self._ZOOM_MAX))

    def _auto_range(self):
        self._ensure_layout()
        n_columns = len(self._column_order) or 1

        self._fit_horizontal()

        max_slots = self._max_slots()
        self._view_y_center = (max_slots - 1) / 2.0

        self._amplitude_gain = 1.0

        rect = self.rect()
        available_h = max(50, rect.height() - 40)
        needed_h = max(1, max_slots) * self._DESIGN_SLOT_SPACING_PX
        if needed_h > 0:
            self._zoom_y = min(1.0, available_h / needed_h)
        else:
            self._zoom_y = 1.0
        self._zoom_y = max(self._ZOOM_MIN, min(self._zoom_y, self._ZOOM_MAX))

    def _fit_horizontal(self):
        self._ensure_layout()
        n_columns = len(self._column_order)
        if n_columns == 0:
            return
        self._zoom_x = self._fit_zoom_x_for_columns(n_columns)
        self._view_x_center = (n_columns - 1) / 2.0

    def showEvent(self, event):
        super().showEvent(event)
        if not getattr(self, "_shown_once", False):
            self._shown_once = True
            self._auto_range()
            self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)

    # ------------------------------------------------------------------
    # Design-space -> pixel mapping
    # ------------------------------------------------------------------

    def _slot_spacing_px(self) -> float:
        return self._DESIGN_SLOT_SPACING_PX * self._zoom_y

    def _column_spacing_px(self) -> float:
        return self._DESIGN_COLUMN_SPACING_PX * self._zoom_x

    def _glyph_width_px(self) -> float:
        return self._DESIGN_GLYPH_WIDTH_PX * self._zoom_x

    def _slot_to_y_px(self, slot: float) -> float:
        rect = self.rect()
        center_y = rect.center().y()
        return center_y - (self._view_y_center - slot) * self._slot_spacing_px()

    def _column_to_x_px(self, column: float) -> float:
        rect = self.rect()
        center_x = rect.center().x()
        return center_x + (column - self._view_x_center) * self._column_spacing_px()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        painter.fillRect(rect, BG_COLOR)

        if len(self._channels) == 0:
            self._draw_centered_message(painter, "No channels selected.")
            return

        self._ensure_layout()

        if self._time_axis is None or not self._channel_data:
            self._draw_centered_message(painter, "Click Compute Average to see traces.")
            return

        self._draw_column_dividers(painter, rect)
        self._draw_zero_trigger_lines(painter, rect)
        self._draw_glyphs(painter)
        self._draw_hud(painter, rect)

    def _draw_centered_message(self, painter: QPainter, text: str):
        painter.setPen(QColor("#888888"))
        font = QFont()
        font.setPointSize(11)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, text)

    def _draw_column_dividers(self, painter: QPainter, rect: QRectF):
        n_columns = len(self._column_order)
        if n_columns < 2:
            return

        pen = QPen(DIVIDER_COLOR, 1.0, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        for col in range(n_columns - 1):
            x_left = self._column_to_x_px(col)
            x_right = self._column_to_x_px(col + 1)
            x_mid = (x_left + x_right) / 2.0
            if rect.left() - 10 <= x_mid <= rect.right() + 10:
                painter.drawLine(int(x_mid), rect.top(), int(x_mid), rect.bottom())

    def _draw_zero_trigger_lines(self, painter: QPainter, rect: QRectF):
        t = self._time_axis
        if t is None or len(t) < 2:
            return
        t_min, t_max = float(t[0]), float(t[-1])
        t_range = t_max - t_min
        if t_range <= 0:
            return

        zero_frac = (0.0 - t_min) / t_range
        if not (0.0 <= zero_frac <= 1.0):
            return

        half_w = self._glyph_width_px() / 2.0
        pen = QPen(ZERO_TRIGGER_COLOR, 1.0, Qt.PenStyle.DotLine)
        painter.setPen(pen)

        for col_idx in range(len(self._column_order)):
            n_slots = self._slots_per_column.get(col_idx, 0)
            if n_slots == 0:
                continue
            top_y = self._slot_to_y_px(0)
            bot_y = self._slot_to_y_px(n_slots - 1)
            half_h = self._slot_spacing_px() / 2.0
            top_y -= half_h
            bot_y += half_h

            cx = self._column_to_x_px(col_idx)
            x_zero = cx - half_w + zero_frac * self._glyph_width_px()

            if rect.left() - 5 <= x_zero <= rect.right() + 5:
                painter.drawLine(int(x_zero), int(top_y), int(x_zero), int(bot_y))

    def _draw_glyphs(self, painter: QPainter):
        rect = self.rect()
        glyph_w = self._glyph_width_px()
        half_w = glyph_w / 2.0
        t = self._time_axis
        if t is None or len(t) < 2:
            return
        t_min, t_max = float(t[0]), float(t[-1])
        t_range = max(t_max - t_min, 1e-9)

        max_points = max(64, int(glyph_w * 2))
        n_total = len(t)
        if n_total > max_points:
            step = int(np.ceil(n_total / max_points))
            idx = np.arange(0, n_total, step)
        else:
            idx = np.arange(n_total)
        t_ds = t[idx]
        x_norm = (t_ds - t_min) / t_range
        x_norm_list = x_norm.tolist()

        # Shared amplitude scale (DC-corrected, full window).
        all_peak = 0.0
        for ch in self._channel_data:
            if ch not in self._column_of_channel:
                continue
            m = self._channel_data[ch]["mean"]
            if len(m) == 0:
                continue
            m_centered = m - float(np.mean(m))
            p = float(np.max(np.abs(m_centered)))
            if p > all_peak:
                all_peak = p
        if all_peak <= 0:
            all_peak = 1.0
        global_max_abs = all_peak

        glyph_half_h = self._slot_spacing_px() * 0.42 * self._amplitude_gain
        y_scale = glyph_half_h / global_max_abs

        for ch, data in self._channel_data.items():
            col = self._column_of_channel.get(ch)
            if col is None:
                continue
            slot = self._slot_of_channel.get(ch)
            if slot is None:
                continue

            cx = self._column_to_x_px(col)
            cy = self._slot_to_y_px(slot)
            glyph_left = cx - half_w

            if (glyph_left + glyph_w < rect.left() - 2 or glyph_left > rect.right() + 2
                    or cy + glyph_half_h < rect.top() - 2 or cy - glyph_half_h > rect.bottom() + 2):
                continue

            # Trace color: always driven by the CHANNEL's own shank id,
            # not the column. In per-shank-column mode this is
            # redundant with the column tint (both derive from the
            # same shank id, so they agree). In single-column mode it's
            # the only thing distinguishing shanks visually.
            shank_id = self._shank_of_channel.get(ch, 0)
            color = _shank_color(shank_id)
            trace_pen = QPen(color, 1.2)
            sem_brush = QBrush(QColor(color.red(), color.green(), color.blue(), 40))

            full_mean = float(np.mean(data["mean"]))
            mean_ds = data["mean"][idx] - full_mean
            sem_ds = data["sem"][idx] if self._show_sem and "sem" in data else None

            x_px_list = [glyph_left + xn * glyph_w for xn in x_norm_list]
            y_mean_px_list = (cy - mean_ds * y_scale).tolist()

            if sem_ds is not None and np.any(sem_ds > 0):
                upper_px = (cy - (mean_ds + sem_ds) * y_scale).tolist()
                lower_px = (cy - (mean_ds - sem_ds) * y_scale).tolist()
                sem_points = [QPointF(xi, yi) for xi, yi in zip(x_px_list, upper_px)]
                sem_points += [QPointF(xi, yi) for xi, yi in zip(reversed(x_px_list), reversed(lower_px))]
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(sem_brush)
                painter.drawPolygon(QPolygonF(sem_points))

            mean_points = [QPointF(xi, yi) for xi, yi in zip(x_px_list, y_mean_px_list)]
            painter.setPen(trace_pen)
            painter.drawPolyline(QPolygonF(mean_points))

    def _draw_hud(self, painter: QPainter, rect: QRectF):
        if abs(self._zoom_x - self._zoom_y) < 1e-6:
            zoom_part = f"zoom {self._zoom_x:.2f}x"
        else:
            zoom_part = f"zoom x {self._zoom_x:.2f}x  y {self._zoom_y:.2f}x"
        mode_part = "single-col" if self._single_column else "per-shank"
        text = f"{zoom_part}   gain {self._amplitude_gain:.2f}x   {mode_part}"

        painter.setPen(HUD_COLOR)
        font = QFont()
        font.setPointSize(9)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        w = metrics.horizontalAdvance(text)
        painter.drawText(int(rect.right() - w - 10), int(rect.top() + 18), text)

    # ------------------------------------------------------------------
    # Zoom helpers
    # ------------------------------------------------------------------

    def _apply_horizontal_zoom(self, factor: float, cursor: QPointF, update: bool = True):
        new_zoom = self._zoom_x * factor
        new_zoom = max(self._ZOOM_MIN, min(new_zoom, self._ZOOM_MAX))
        if new_zoom == self._zoom_x:
            return

        cx_px = self.rect().center().x()
        old_spacing = self._column_spacing_px()

        design_x = self._view_x_center + (cursor.x() - cx_px) / old_spacing

        self._zoom_x = new_zoom
        new_spacing = self._column_spacing_px()

        self._view_x_center = design_x - (cursor.x() - cx_px) / new_spacing

        if update:
            self.update()

    def _apply_vertical_zoom(self, factor: float, cursor: QPointF, update: bool = True):
        new_zoom = self._zoom_y * factor
        new_zoom = max(self._ZOOM_MIN, min(new_zoom, self._ZOOM_MAX))
        if new_zoom == self._zoom_y:
            return

        cy_px = self.rect().center().y()
        old_spacing = self._slot_spacing_px()

        design_y = self._view_y_center - (cursor.y() - cy_px) / old_spacing

        self._zoom_y = new_zoom
        new_spacing = self._slot_spacing_px()

        self._view_y_center = design_y + (cursor.y() - cy_px) / new_spacing

        if update:
            self.update()

    # ------------------------------------------------------------------
    # Mouse / key interaction
    # ------------------------------------------------------------------

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            event.accept()
            return

        factor = 1.15 if delta > 0 else 1 / 1.15
        cursor = event.position()

        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self._amplitude_gain *= factor
            self._amplitude_gain = max(self._GAIN_MIN, min(self._amplitude_gain, self._GAIN_MAX))
            self.update()
            event.accept()
            return

        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self._apply_horizontal_zoom(factor, cursor)
            event.accept()
            return

        self._apply_horizontal_zoom(factor, cursor, update=False)
        self._apply_vertical_zoom(factor, cursor, update=False)
        self.update()
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_start_pos = event.position()
            self._drag_start_view_center = (self._view_x_center, self._view_y_center)
            event.accept()

    def mouseMoveEvent(self, event):
        if self._dragging and self._drag_start_pos is not None:
            delta = event.position() - self._drag_start_pos
            dx_columns = -delta.x() / max(1.0, self._column_spacing_px())
            dy_slots = -delta.y() / max(1.0, self._slot_spacing_px())
            self._view_x_center = self._drag_start_view_center[0] + dx_columns
            self._view_y_center = self._drag_start_view_center[1] + dy_slots
            self.update()
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            event.accept()

    def mouseDoubleClickEvent(self, event):
        self._auto_range()
        self.update()
        event.accept()

    def keyPressEvent(self, event):
        key = event.key()
        ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)

        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            if ctrl:
                self._amplitude_gain = min(self._GAIN_MAX, self._amplitude_gain * 1.15)
            elif shift:
                self._apply_horizontal_zoom(1.15, QPointF(self.rect().center()))
            else:
                center = QPointF(self.rect().center())
                self._apply_horizontal_zoom(1.15, center, update=False)
                self._apply_vertical_zoom(1.15, center, update=False)
                self.update()
            event.accept()
            return

        if key == Qt.Key.Key_Minus:
            if ctrl:
                self._amplitude_gain = max(self._GAIN_MIN, self._amplitude_gain / 1.15)
            elif shift:
                self._apply_horizontal_zoom(1 / 1.15, QPointF(self.rect().center()))
            else:
                center = QPointF(self.rect().center())
                self._apply_horizontal_zoom(1 / 1.15, center, update=False)
                self._apply_vertical_zoom(1 / 1.15, center, update=False)
                self.update()
            event.accept()
            return

        if key == Qt.Key.Key_F:
            self._fit_horizontal()
            self.update()
            event.accept()
            return

        if key == Qt.Key.Key_0:
            self._auto_range()
            self.update()
            event.accept()
            return

        super().keyPressEvent(event)


class RTAProbeTraceWidget(QWidget):
    """
    Container: a thin header row (layout-mode toggle) above the painted
    canvas. Public API forwards to the canvas so callers that already
    use this class (rta_probe_trace_dialog.py) don't need changes.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ---- Header row ----
        header = QWidget()
        header.setStyleSheet("background-color: #252525;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 4, 8, 4)
        header_layout.setSpacing(12)

        self.single_column_check = QCheckBox("Single column")
        self.single_column_check.setToolTip(
            "Merge every shank into one column, ranked by relative "
            "depth. Per-shank colors are preserved so shanks remain "
            "distinguishable without a column each."
        )
        self.single_column_check.setStyleSheet(f"color: {HEADER_TEXT_COLOR.name()};")
        self.single_column_check.toggled.connect(self._on_single_column_toggled)
        header_layout.addWidget(self.single_column_check)

        self.shank_legend_label = QLabel("")
        self.shank_legend_label.setStyleSheet(f"color: {HEADER_TEXT_COLOR.name()};")
        header_layout.addWidget(self.shank_legend_label)

        header_layout.addStretch(1)
        layout.addWidget(header)

        # ---- Canvas ----
        self.canvas = _RTACanvas()
        layout.addWidget(self.canvas, stretch=1)

    # ------------------------------------------------------------------
    # Public API -- forwards to canvas
    # ------------------------------------------------------------------

    def set_geometry_from_probe(self, probe_data: dict, channels: list[int]):
        self.canvas.set_geometry_from_probe(probe_data, channels)
        self._update_shank_legend()

    def set_result(self, time_axis: np.ndarray, channel_data: dict[int, dict], show_sem: bool = True):
        self.canvas.set_result(time_axis, channel_data, show_sem)

    def set_show_sem(self, show_sem: bool):
        self.canvas.set_show_sem(show_sem)

    def reset_view(self):
        self.canvas.reset_view()

    def fit_horizontal(self):
        self.canvas.fit_horizontal()

    def set_single_column(self, single: bool):
        self.single_column_check.setChecked(bool(single))

    def is_single_column(self) -> bool:
        return self.canvas.is_single_column()

    # ------------------------------------------------------------------
    # Header helpers
    # ------------------------------------------------------------------

    def _on_single_column_toggled(self, checked: bool):
        self.canvas.set_single_column(checked)

    def _update_shank_legend(self):
        """Show a compact color-coded legend of which shank colors are
        present, so the color coding is discoverable even in
        per-shank-column mode where it's visually obvious."""
        shanks = sorted(set(int(s) for s in self.canvas._shank_ids))
        if not shanks:
            self.shank_legend_label.setText("")
            return
        # Build a small HTML legend using the shank palette.
        parts = []
        for s in shanks:
            c = _shank_color(s)
            parts.append(
                f'<span style="color:{c.name()};">&#9632;</span>&nbsp;Shank {s}'
            )
        self.shank_legend_label.setText("&nbsp;&nbsp;".join(parts))
        self.shank_legend_label.setTextFormat(Qt.TextFormat.RichText)
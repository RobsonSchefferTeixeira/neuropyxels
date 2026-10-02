"""
phy_units_panel.py

Dockable panel that lists the units in a loaded Phy folder, allows
reclassification with the app's extended class scheme, and drives the
spike overlays on the trace view.

Signals
-------
selectionChanged(list[int])
    Emitted whenever the set of selected cluster ids changes.
rasterVisibleChanged(bool)
    Emitted when the "Raster ticks" checkbox is toggled.
recolorVisibleChanged(bool)
    Emitted when the "Recolor trace at spikes" checkbox is toggled.
recolorWindowChanged(float)
    Emitted when the "Window (ms)" spinbox value changes.
focusModeChanged(bool)
    Emitted when the "Focus mode" checkbox is toggled.
focusNeighborhoodChanged(int)
    Emitted when the "± channels" spinbox value changes.
focusedUnitChanged(int)
    Emitted when the focused unit changes. -1 when no unit is focused.
reclassifyRequested(list[int], str)
    Emitted when the user asks to reclassify the selected units
    (via the Reclassify button or the table's context menu).
resetClassificationRequested(list[int])
    Emitted when the user asks to clear the reclassified column for
    the selected units.
importFromPhyRequested(list[int])
    Emitted when the user asks to seed the reclassified column from
    Phy's own label for the selected units.
saveRequested()
    Emitted when the user clicks "Save classification".
prevSpikeRequested()
    Emitted when the user clicks "◄ Prev Spike" or presses PageUp.
nextSpikeRequested()
    Emitted when the user clicks "Next Spike ►" or presses PageDown.
reclassifyAndAdvanceRequested(list[int], str)
    Emitted on bare E/G/P/U/M/N: reclassify the focused unit and
    advance to the next visible one. Fired only when exactly one unit
    is selected.

Keyboard shortcuts
------------------
The shortcuts live on the table (via _ReclassifyTable) so bare letters
intercept before QAbstractItemView's own keyboard-search behavior.
The panel itself also handles the same keys as a fallback for when the
panel -- not the table -- has focus.

  Bare E/G/P/U/M/N   -> reclassifyAndAdvanceRequested
  Bare PageUp/Down   -> prevSpikeRequested / nextSpikeRequested

Auto-repeat is ignored. Any modifier (Ctrl/Alt/Shift/Meta) disqualifies
the shortcut, so e.g. Ctrl+PageUp still zooms the trace view.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QCheckBox, QPushButton, QDoubleSpinBox, QSpinBox, QMenu,
)

from core.phy_loader import PhyData, Unit, KNOWN_CLASSES


# Color gradient for the classification. Monotonic from worst to best.
CLASS_COLORS = {
    "noise":     "#666666",
    "mua":       "#a06030",
    "unsorted":  "#8888aa",
    "poor":      "#cc6644",
    "good":      "#3fa34d",
    "excellent": "#5ad1ff",
}


# Bare-letter -> class, for the reclassify-and-advance shortcut.
# Deliberately no modifiers: plain letters match Phy's own mnemonics.
# The _ReclassifyTable subclass intercepts these at the table level so
# QAbstractItemView's keyboard-search doesn't eat them first.
_RECLASSIFY_SHORTCUTS: dict[int, str] = {
    Qt.Key.Key_E: "excellent",
    Qt.Key.Key_G: "good",
    Qt.Key.Key_P: "poor",
    Qt.Key.Key_U: "unsorted",
    Qt.Key.Key_M: "mua",
    Qt.Key.Key_N: "noise",
}


def _is_bare_key(event) -> bool:
    """True if the event has no Ctrl/Alt/Shift/Meta modifiers. Bare
    keys are the only ones the shortcut handlers act on; anything with
    a modifier is left for other widgets (e.g. Ctrl+PageUp -> zoom)."""
    mods = event.modifiers()
    disqualifying = (
        Qt.KeyboardModifier.ControlModifier
        | Qt.KeyboardModifier.AltModifier
        | Qt.KeyboardModifier.ShiftModifier
        | Qt.KeyboardModifier.MetaModifier
    )
    return not (mods & disqualifying)


class _ReclassifyTable(QTableWidget):
    """QTableWidget subclass that intercepts bare E/G/P/U/M/N and bare
    PageUp/PageDown before QAbstractItemView's own key handling. Emits
    the results as signals rather than reaching up to the parent, so
    PhyUnitsPanel just wires them to its own public signals.

    Why a subclass rather than an event filter: key events reach the
    focused widget (the table) first, and QAbstractItemView's default
    keyPressEvent runs a keyboard-search for printable characters --
    typing G would jump to the first cell whose text starts with "G".
    Overriding keyPressEvent here is the only way to consume the
    printable-letter presses cleanly before that runs.
    """

    reclassifyShortcutActivated = pyqtSignal(str)   # new class name
    prevSpikeRequested = pyqtSignal()
    nextSpikeRequested = pyqtSignal()

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            super().keyPressEvent(event)
            return

        if _is_bare_key(event):
            key = event.key()

            if key in (Qt.Key.Key_PageUp, Qt.Key.Key_PageDown):
                if key == Qt.Key.Key_PageUp:
                    self.prevSpikeRequested.emit()
                else:
                    self.nextSpikeRequested.emit()
                event.accept()
                return

            class_name = _RECLASSIFY_SHORTCUTS.get(key)
            if class_name is not None:
                self.reclassifyShortcutActivated.emit(class_name)
                event.accept()
                return

        super().keyPressEvent(event)


class PhyUnitsPanel(QWidget):
    """Table of Phy units with search, quality filter, multi-select,
    reclassification, and spike navigation."""

    selectionChanged = pyqtSignal(list)
    rasterVisibleChanged = pyqtSignal(bool)
    recolorVisibleChanged = pyqtSignal(bool)
    recolorWindowChanged = pyqtSignal(float)
    focusModeChanged = pyqtSignal(bool)
    focusNeighborhoodChanged = pyqtSignal(int)
    focusedUnitChanged = pyqtSignal(int)
    reclassifyRequested = pyqtSignal(list, str)              # cluster ids, new class
    resetClassificationRequested = pyqtSignal(list)          # cluster ids
    importFromPhyRequested = pyqtSignal(list)                # cluster ids
    saveRequested = pyqtSignal()
    prevSpikeRequested = pyqtSignal()
    nextSpikeRequested = pyqtSignal()
    reclassifyAndAdvanceRequested = pyqtSignal(list, str)    # cluster ids, new class

    def __init__(self, parent=None):
        super().__init__(parent)
        self.phy_data: PhyData | None = None
        self._rows: list[Unit] = []

        # The panel itself accepts focus so the shortcut fallback below
        # fires when the user has clicked a non-table area of the panel
        # (e.g. the summary label). Without this, focus stays on
        # whatever had it before, and the shortcut would appear to do
        # nothing from the user's point of view.
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ---- Filter row ----
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("cluster id, channel, class...")
        self.search_edit.textChanged.connect(self._apply_filter)
        filter_row.addWidget(self.search_edit, stretch=1)

        filter_row.addWidget(QLabel("Class:"))
        self.quality_combo = QComboBox()
        # Reverse quality order, with 'all' at the end.
        for cls in KNOWN_CLASSES:
            self.quality_combo.addItem(cls)
        self.quality_combo.addItem("all")
        self.quality_combo.setCurrentText("all")
        self.quality_combo.currentTextChanged.connect(self._apply_filter)
        filter_row.addWidget(self.quality_combo)
        layout.addLayout(filter_row)

        # ---- Bulk actions ----
        action_row = QHBoxLayout()
        self.select_good_btn = QPushButton("Select all 'good'")
        self.select_good_btn.setAutoDefault(False)
        self.select_good_btn.clicked.connect(self._select_all_good)
        action_row.addWidget(self.select_good_btn)

        self.clear_btn = QPushButton("Clear selection")
        self.clear_btn.setAutoDefault(False)
        self.clear_btn.clicked.connect(self._clear_selection)
        action_row.addWidget(self.clear_btn)

        self.show_only_selected_check = QCheckBox("Only selected")
        self.show_only_selected_check.setToolTip(
            "Hide rows for units that are not currently selected."
        )
        self.show_only_selected_check.toggled.connect(self._apply_filter)
        action_row.addWidget(self.show_only_selected_check)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        # ---- Focus mode ----
        focus_row = QHBoxLayout()
        self.focus_check = QCheckBox("Focus mode")
        self.focus_check.setChecked(False)
        self.focus_check.setToolTip(
            "Select exactly one unit at a time and automatically open "
            "its local neighborhood on the trace view and probe map."
        )
        self.focus_check.toggled.connect(self._on_focus_toggled)
        focus_row.addWidget(self.focus_check)

        focus_row.addWidget(QLabel("± channels:"))
        self.focus_n_spin = QSpinBox()
        self.focus_n_spin.setRange(1, 20)
        self.focus_n_spin.setValue(5)
        self.focus_n_spin.valueChanged.connect(self.focusNeighborhoodChanged.emit)
        focus_row.addWidget(self.focus_n_spin)
        focus_row.addStretch(1)
        layout.addLayout(focus_row)

        # ---- Spike navigation ----
        nav_row = QHBoxLayout()
        self.prev_spike_btn = QPushButton("\u25c4 Prev Spike")
        self.prev_spike_btn.setAutoDefault(False)
        self.prev_spike_btn.setDefault(False)
        self.prev_spike_btn.setToolTip(
            "Jump the trace view to the previous spike of the focused "
            "unit (or the single selected unit if focus mode is off).\n"
            "Shortcut: PageUp"
        )
        self.prev_spike_btn.clicked.connect(self.prevSpikeRequested.emit)
        nav_row.addWidget(self.prev_spike_btn)

        self.next_spike_btn = QPushButton("Next Spike \u25ba")
        self.next_spike_btn.setAutoDefault(False)
        self.next_spike_btn.setDefault(False)
        self.next_spike_btn.setToolTip(
            "Jump the trace view to the next spike of the focused unit "
            "(or the single selected unit if focus mode is off).\n"
            "Shortcut: PageDown"
        )
        self.next_spike_btn.clicked.connect(self.nextSpikeRequested.emit)
        nav_row.addWidget(self.next_spike_btn)
        nav_row.addStretch(1)
        layout.addLayout(nav_row)

        # ---- Spike visualization ----
        spike_row = QHBoxLayout()
        self.raster_check = QCheckBox("Raster ticks")
        self.raster_check.setChecked(True)
        self.raster_check.toggled.connect(self.rasterVisibleChanged.emit)
        spike_row.addWidget(self.raster_check)

        self.recolor_check = QCheckBox("Recolor trace at spikes")
        self.recolor_check.setChecked(False)
        self.recolor_check.toggled.connect(self.recolorVisibleChanged.emit)
        spike_row.addWidget(self.recolor_check)

        spike_row.addWidget(QLabel("Window (ms):"))
        self.recolor_window_spin = QDoubleSpinBox()
        self.recolor_window_spin.setRange(0.05, 20.0)
        self.recolor_window_spin.setDecimals(2)
        self.recolor_window_spin.setSingleStep(0.1)
        self.recolor_window_spin.setValue(1.0)
        self.recolor_window_spin.valueChanged.connect(self.recolorWindowChanged.emit)
        spike_row.addWidget(self.recolor_window_spin)
        spike_row.addStretch(1)
        layout.addLayout(spike_row)

        # ---- Classification actions ----
        class_row = QHBoxLayout()
        self.reclassify_btn = QPushButton("Reclassify…")
        self.reclassify_btn.setAutoDefault(False)
        self.reclassify_btn.setToolTip(
            "Change the class of the selected unit(s). Right-click the "
            "table for the same menu.\n"
            "Shortcuts: E=excellent  G=good  P=poor  "
            "U=unsorted  M=mua  N=noise"
        )
        self.reclassify_btn.clicked.connect(self._on_reclassify_button_clicked)
        class_row.addWidget(self.reclassify_btn)

        self.import_phy_btn = QPushButton("Import Phy classification")
        self.import_phy_btn.setAutoDefault(False)
        self.import_phy_btn.setToolTip(
            "Set the reclassified column to Phy's own label for the "
            "selected unit(s). Useful before editing."
        )
        self.import_phy_btn.clicked.connect(self._on_import_phy_clicked)
        class_row.addWidget(self.import_phy_btn)

        self.reset_class_btn = QPushButton("Reset classification")
        self.reset_class_btn.setAutoDefault(False)
        self.reset_class_btn.setToolTip(
            "Clear the reclassified column for the selected unit(s)."
        )
        self.reset_class_btn.clicked.connect(self._on_reset_class_clicked)
        class_row.addWidget(self.reset_class_btn)

        self.save_btn = QPushButton("Save classification")
        self.save_btn.setAutoDefault(False)
        self.save_btn.setToolTip(
            "Write cluster_info_reclassified.tsv in the loaded Phy folder."
        )
        self.save_btn.clicked.connect(self.saveRequested.emit)
        class_row.addWidget(self.save_btn)
        class_row.addStretch(1)
        layout.addLayout(class_row)

        # ---- Table ----
        # Use _ReclassifyTable so bare-letter shortcuts are intercepted
        # before QAbstractItemView's keyboard-search behavior.
        self.table = _ReclassifyTable(0, 8)
        self.table.setHorizontalHeaderLabels([
            "Unit", "Channel", "Depth (µm)", "Class",
            "Reclassified", "n spikes", "FR (Hz)", "Shank",
        ])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        header.setDefaultAlignment(
            Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_context_menu)

        # Relay the table's shortcut signals to the panel's public ones,
        # so MainWindow only needs to know about the panel.
        self.table.reclassifyShortcutActivated.connect(
            self._on_table_reclassify_shortcut
        )
        self.table.prevSpikeRequested.connect(self.prevSpikeRequested.emit)
        self.table.nextSpikeRequested.connect(self.nextSpikeRequested.emit)

        layout.addWidget(self.table, stretch=1)

        self.summary_label = QLabel("No Phy folder loaded.")
        self.summary_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self.summary_label)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_phy_data(self, phy_data: PhyData | None):
        self.phy_data = phy_data
        self._rows = []
        if phy_data is not None:
            self._rows = sorted(phy_data.units.values(), key=lambda u: u.cluster_id)
        self._populate_table()

        if phy_data is None:
            self.summary_label.setText("No Phy folder loaded.")
        else:
            n_total = len(self._rows)
            # "good+" counts Phy's own labels only -- the app's
            # reclassifications are a separate axis and deliberately do
            # not feed into this number. Same source as the Class
            # column, the class filter, and "Select all 'good'".
            n_good_phy = sum(
                1 for u in self._rows
                if (u.group or u.kslabel or "").strip().lower()
                in ("good", "excellent")
            )
            n_reclassified = sum(1 for u in self._rows if u.reclassified)
            n_spikes = int(phy_data.spike_times.size)
            src = phy_data.info_source.name if phy_data.info_source else "—"
            self.summary_label.setText(
                f"{phy_data.folder.name}: {n_total} unit(s), "
                f"{n_good_phy} good+ (Phy), "
                f"{n_reclassified} reclassified, "
                f"{n_spikes} total spikes. "
                f"Loaded from: {src}"
            )

    def selected_cluster_ids(self) -> list[int]:
        ids = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is None:
                continue
            if item.isSelected():
                cid = item.data(Qt.ItemDataRole.UserRole)
                if cid is not None:
                    ids.append(int(cid))
        return ids

    def apply_reclassification(self, cluster_ids: list[int], new_class: str):
        """Update the in-memory Unit.quality for the given units and
        refresh the table immediately. Does NOT write to disk; the
        caller decides when to save."""
        if self.phy_data is None:
            return
        for cid in cluster_ids:
            unit = self.phy_data.units.get(int(cid))
            if unit is None:
                continue
            unit.quality = new_class if new_class else None
            unit.reclassified = bool(new_class)
        self._populate_table()

    def reset_classification(self, cluster_ids: list[int]):
        self.apply_reclassification(cluster_ids, "")

    def import_from_phy(self, cluster_ids: list[int]):
        """Copy Phy's own label (unit.group or unit.kslabel) into the
        app's reclassified column for the given units."""
        if self.phy_data is None:
            return
        for cid in cluster_ids:
            unit = self.phy_data.units.get(int(cid))
            if unit is None:
                continue
            label = unit.group or unit.kslabel
            if label:
                unit.quality = label
                unit.reclassified = True
        self._populate_table()

    def focus_next_visible_unit(self) -> bool:
        """Select the next visible row after the currently-selected one,
        using the table's current sort/filter order. Stops at the last
        row -- no wrap. Returns True if the selection moved.

        Called by MainWindow after a successful reclassify-and-advance
        so the bare-letter shortcut behaves like Phy's: label this unit,
        move on.
        """
        current = self.selected_cluster_ids()
        if not current:
            if self.table.rowCount() > 0:
                self.table.selectRow(0)
                self.selectionChanged.emit(self.selected_cluster_ids())
                return True
            return False

        current_cid = current[0]
        current_row = -1
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == current_cid:
                current_row = row
                break

        if current_row < 0 or current_row + 1 >= self.table.rowCount():
            return False

        self.table.selectRow(current_row + 1)
        self.selectionChanged.emit(self.selected_cluster_ids())
        return True

    def set_recolor_enabled(self, enabled: bool):
        self.recolor_check.blockSignals(True)
        self.recolor_check.setChecked(bool(enabled))
        self.recolor_check.blockSignals(False)

    def recolor_window_ms(self) -> float:
        return float(self.recolor_window_spin.value())

    def focus_mode_enabled(self) -> bool:
        return bool(self.focus_check.isChecked())

    def focus_neighborhood_size(self) -> int:
        return int(self.focus_n_spin.value())

    def focused_unit_id(self) -> int:
        ids = self.selected_cluster_ids()
        return int(ids[0]) if ids else -1

    def set_focus_mode(self, enabled: bool):
        self.focus_check.blockSignals(True)
        self.focus_check.setChecked(bool(enabled))
        self.focus_check.blockSignals(False)
        self._on_focus_toggled(bool(enabled))

    def clear_selection(self):
        self._clear_selection()

    # ------------------------------------------------------------------
    # Shortcut handling
    # ------------------------------------------------------------------

    def _on_table_reclassify_shortcut(self, class_name: str):
        """Table emitted a bare-letter shortcut. Only act when exactly
        one row is selected -- which focus mode guarantees, but we
        check defensively in case a future change loosens that."""
        selected = self.selected_cluster_ids()
        if len(selected) == 1:
            self.reclassifyAndAdvanceRequested.emit(selected, class_name)

    def keyPressEvent(self, event):
        """Fallback for when the panel itself (not the table) has
        focus. Same behavior as the table subclass -- see _ReclassifyTable.
        """
        if event.isAutoRepeat():
            super().keyPressEvent(event)
            return

        if _is_bare_key(event):
            key = event.key()

            if key in (Qt.Key.Key_PageUp, Qt.Key.Key_PageDown):
                if key == Qt.Key.Key_PageUp:
                    self.prevSpikeRequested.emit()
                else:
                    self.nextSpikeRequested.emit()
                event.accept()
                return

            class_name = _RECLASSIFY_SHORTCUTS.get(key)
            if class_name is not None:
                selected = self.selected_cluster_ids()
                if len(selected) == 1:
                    self.reclassifyAndAdvanceRequested.emit(selected, class_name)
                    event.accept()
                    return

        super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Filtering / population
    # ------------------------------------------------------------------

    def _apply_filter(self):
        self._populate_table()

    def _populate_table(self):
        if not self._rows:
            self.table.setRowCount(0)
            return

        search = self.search_edit.text().strip().lower()
        class_filter = self.quality_combo.currentText()
        only_selected = self.show_only_selected_check.isChecked()
        prev_selected = set(self.selected_cluster_ids())

        def matches(u: Unit) -> bool:
            if only_selected and u.cluster_id not in prev_selected:
                return False
            if class_filter != "all":
                # Filter on Phy's label only -- the dropdown's options
                # are Phy's vocabulary, and the user is choosing which
                # of Phy's units to look at, not which of their own
                # classifications.
                phy_label = (u.group or u.kslabel or "").lower()
                if phy_label != class_filter.lower():
                    return False
            if not search:
                return True
            # The free-text search is broader: it matches both Phy's
            # label and the app's classification, plus the other
            # identifying fields.
            haystack = " ".join(str(x) for x in (
                u.cluster_id, u.channel, u.quality, u.group, u.kslabel, u.shank
            ) if x is not None).lower()
            return search in haystack

        visible = [u for u in self._rows if matches(u)]

        self.table.setSortingEnabled(False)
        self.table.blockSignals(True)
        self.table.setRowCount(len(visible))
        for row, u in enumerate(visible):
            
            
            # Column 3 (Class)     = Phy's own label, never touched by
            #                        the app -- falls back to kslabel so
            #                        uncurated folders still show
            #                        something.
            # Column 4 (Reclassified) = the app's classification as text;
            #                        empty unless the user has actually
            #                        reclassified this unit.
            phy_label = (u.group or u.kslabel or "").strip()
            app_label = (u.quality or "") if u.reclassified else ""

            cells = [
                str(u.cluster_id),
                "" if u.channel is None else str(u.channel),
                "" if u.depth is None else f"{u.depth:.0f}",
                phy_label,
                app_label,
                str(u.n_spikes),
                "" if u.firing_rate is None else f"{u.firing_rate:.2f}",
                "" if u.shank is None else str(u.shank),
            ]

            
            item_id = None
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
                )
                self.table.setItem(row, col, item)
                if col == 0:
                    item_id = item
            item_id.setData(Qt.ItemDataRole.UserRole, u.cluster_id)

            # Row color: the app's class if reclassified, else Phy's
            # own label. This keeps the visual gradient meaningful on a
            # fresh load (colored by Phy's verdict) and switches to
            # reflecting the user's judgments as they curate.
            color_source = (app_label or phy_label).lower()
            color_hex = CLASS_COLORS.get(color_source)
            if color_hex is not None:
                from PyQt6.QtGui import QColor
                c = QColor(color_hex)
                for col in range(len(cells)):
                    self.table.item(row, col).setForeground(c)

            if u.cluster_id in prev_selected:
                item_id.setSelected(True)

        self.table.blockSignals(False)
        self.table.setSortingEnabled(True)

    # ------------------------------------------------------------------
    # Selection events
    # ------------------------------------------------------------------

    def _on_selection_changed(self):
        selected = self.selected_cluster_ids()
        if self.focus_check.isChecked() and len(selected) > 1:
            selected = [selected[0]]
            self.table.blockSignals(True)
            first_cid = selected[0]
            for row in range(self.table.rowCount()):
                item = self.table.item(row, 0)
                if item is None:
                    continue
                if item.data(Qt.ItemDataRole.UserRole) == first_cid:
                    self.table.clearSelection()
                    item.setSelected(True)
                    break
            self.table.blockSignals(False)

        self.selectionChanged.emit(selected)

        if self.focus_check.isChecked():
            self.focusedUnitChanged.emit(
                int(selected[0]) if selected else -1
            )

    def _select_all_good(self):
        """Select every row whose Class column (Phy's own label) is
        'good' or 'excellent'. Deliberately ignores the app's
        reclassifications -- this is a starting point for curation, not
        a reflection of it."""
        self.table.blockSignals(True)
        self.table.clearSelection()
        for row in range(self.table.rowCount()):
            q_item = self.table.item(row, 3)   # Class column = Phy label
            if q_item is not None and q_item.text().strip().lower() in ("good", "excellent"):
                self.table.selectRow(row)
        self.table.blockSignals(False)
        self.selectionChanged.emit(self.selected_cluster_ids())

    def _clear_selection(self):
        self.table.blockSignals(True)
        self.table.clearSelection()
        self.table.blockSignals(False)
        self.selectionChanged.emit([])

    # ------------------------------------------------------------------
    # Reclassification
    # ------------------------------------------------------------------

    def _on_reclassify_button_clicked(self):
        selected = self.selected_cluster_ids()
        if not selected:
            return
        menu = QMenu(self)
        for cls in KNOWN_CLASSES:
            action = menu.addAction(cls)
            action.triggered.connect(
                lambda _checked, c=cls: self.reclassifyRequested.emit(
                    self.selected_cluster_ids(), c
                )
            )
        menu.exec(self.reclassify_btn.mapToGlobal(
            self.reclassify_btn.rect().bottomLeft()
        ))

    def _on_import_phy_clicked(self):
        selected = self.selected_cluster_ids()
        if selected:
            self.importFromPhyRequested.emit(selected)

    def _on_reset_class_clicked(self):
        selected = self.selected_cluster_ids()
        if selected:
            self.resetClassificationRequested.emit(selected)

    def _on_context_menu(self, pos):
        selected = self.selected_cluster_ids()
        if not selected:
            return
        menu = QMenu(self)

        reclass_sub = menu.addMenu("Reclassify as")
        for cls in KNOWN_CLASSES:
            act = reclass_sub.addAction(cls)
            act.triggered.connect(
                lambda _checked, c=cls: self.reclassifyRequested.emit(
                    self.selected_cluster_ids(), c
                )
            )

        menu.addSeparator()

        import_act = menu.addAction("Import Phy classification")
        import_act.triggered.connect(self._on_import_phy_clicked)

        reset_act = menu.addAction("Reset classification")
        reset_act.triggered.connect(self._on_reset_class_clicked)

        menu.exec(self.table.viewport().mapToGlobal(pos))

    # ------------------------------------------------------------------
    # Focus mode
    # ------------------------------------------------------------------

    def _on_focus_toggled(self, checked: bool):
        if checked:
            self.table.setSelectionMode(
                QAbstractItemView.SelectionMode.SingleSelection
            )
            selected = self.selected_cluster_ids()
            if len(selected) > 1:
                self.table.blockSignals(True)
                first_cid = selected[0]
                for row in range(self.table.rowCount()):
                    item = self.table.item(row, 0)
                    if item is None:
                        continue
                    if item.data(Qt.ItemDataRole.UserRole) == first_cid:
                        self.table.clearSelection()
                        item.setSelected(True)
                        break
                self.table.blockSignals(False)
                self.selectionChanged.emit([first_cid])
            elif len(selected) == 1:
                self.selectionChanged.emit(selected)
        else:
            self.table.setSelectionMode(
                QAbstractItemView.SelectionMode.ExtendedSelection
            )
        self.focusModeChanged.emit(bool(checked))
        if checked:
            selected = self.selected_cluster_ids()
            self.focusedUnitChanged.emit(
                int(selected[0]) if selected else -1
            )
        else:
            self.focusedUnitChanged.emit(-1)
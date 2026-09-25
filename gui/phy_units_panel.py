"""
phy_units_panel.py

Dockable panel that lists the units in a loaded Phy folder and lets the
user pick which ones to overlay as a spike raster on the trace view.

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
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal


from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QCheckBox, QPushButton, QDoubleSpinBox, QSpinBox,
)

from core.phy_loader import PhyData, Unit


class PhyUnitsPanel(QWidget):
    """Table of Phy units with search, quality filter, and multi-select.

    Selection is communicated via selectionChanged(list[int]). The
    panel does NOT know anything about the trace view -- it just reports
    which cluster ids the user has selected, and what visual options
    they want applied to those selections.
    """

    selectionChanged = pyqtSignal(list)
    rasterVisibleChanged = pyqtSignal(bool)
    recolorVisibleChanged = pyqtSignal(bool)
    recolorWindowChanged = pyqtSignal(float)
    focusModeChanged = pyqtSignal(bool)
    focusNeighborhoodChanged = pyqtSignal(int)
    focusedUnitChanged = pyqtSignal(int)   # -1 when no focused unit
        

    def __init__(self, parent=None):
        super().__init__(parent)
        self.phy_data: PhyData | None = None
        self._rows: list[Unit] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ---- Filter row ----
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("cluster id, channel, quality...")
        self.search_edit.textChanged.connect(self._apply_filter)
        filter_row.addWidget(self.search_edit, stretch=1)

        filter_row.addWidget(QLabel("Quality:"))
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["all", "good", "mua", "noise"])
        self.quality_combo.currentTextChanged.connect(self._apply_filter)
        filter_row.addWidget(self.quality_combo)
        layout.addLayout(filter_row)

        # ---- Bulk actions ----
        action_row = QHBoxLayout()
        self.select_good_btn = QPushButton("Select all 'good'")
        self.select_good_btn.setAutoDefault(False)
        self.select_good_btn.setDefault(False)
        self.select_good_btn.clicked.connect(self._select_all_good)
        action_row.addWidget(self.select_good_btn)

        self.clear_btn = QPushButton("Clear selection")
        self.clear_btn.setAutoDefault(False)
        self.clear_btn.setDefault(False)
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


        # ---- Focus mode row ----
        focus_row = QHBoxLayout()

        self.focus_check = QCheckBox("Focus mode")
        self.focus_check.setChecked(False)
        self.focus_check.setToolTip(
            "Select exactly one unit at a time and automatically open "
            "that unit's local neighborhood on the trace view and probe "
            "map. Channels are replaced, not added -- turning focus mode "
            "off leaves the current selection in place."
        )
        self.focus_check.toggled.connect(self._on_focus_toggled)
        focus_row.addWidget(self.focus_check)

        focus_row.addWidget(QLabel("± channels:"))
        self.focus_n_spin = QSpinBox()
        self.focus_n_spin.setRange(1, 20)
        self.focus_n_spin.setValue(5)
        self.focus_n_spin.setToolTip(
            "Number of depth levels above and below the focused unit's "
            "depth to include. All channels at each selected depth on "
            "the same shank are included (across all x positions on "
            "that shank)."
        )
        self.focus_n_spin.valueChanged.connect(self.focusNeighborhoodChanged.emit)
        focus_row.addWidget(self.focus_n_spin)

        focus_row.addStretch(1)
        layout.addLayout(focus_row)


        # ---- Spike visualization controls ----
        spike_row = QHBoxLayout()

        self.raster_check = QCheckBox("Raster ticks")
        self.raster_check.setChecked(True)
        self.raster_check.setToolTip(
            "Draw one vertical tick per spike, in the lane of the "
            "unit's assigned channel."
        )
        self.raster_check.toggled.connect(self.rasterVisibleChanged.emit)
        spike_row.addWidget(self.raster_check)

        self.recolor_check = QCheckBox("Recolor trace at spikes")
        self.recolor_check.setChecked(False)
        self.recolor_check.setToolTip(
            "Redraw every displayed trace in the unit's color over a "
            "short window around each spike time. Works across all "
            "channels and shanks, so you can see what every channel "
            "was doing at that instant."
        )
        self.recolor_check.toggled.connect(self.recolorVisibleChanged.emit)
        spike_row.addWidget(self.recolor_check)

        spike_row.addWidget(QLabel("Window (ms):"))
        self.recolor_window_spin = QDoubleSpinBox()
        self.recolor_window_spin.setRange(0.05, 20.0)
        self.recolor_window_spin.setDecimals(2)
        self.recolor_window_spin.setSingleStep(0.1)
        self.recolor_window_spin.setValue(1.0)
        self.recolor_window_spin.setToolTip(
            "Total window width, centered on each spike. The trace is "
            "recolored within this interval."
        )
        self.recolor_window_spin.valueChanged.connect(self.recolorWindowChanged.emit)
        spike_row.addWidget(self.recolor_window_spin)

        spike_row.addStretch(1)
        layout.addLayout(spike_row)

        # ---- Table ----
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels([
            "Unit", "Channel", "Depth (µm)", "Quality", "n spikes", "FR (Hz)", "Shank"
        ])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
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
            n_good = sum(1 for u in self._rows if (u.quality or "").lower() == "good")
            n_spikes = int(phy_data.spike_times.size)
            self.summary_label.setText(
                f"{phy_data.folder.name}: {n_total} unit(s), {n_good} good, "
                f"{n_spikes} total spikes."
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

    def set_recolor_enabled(self, enabled: bool):
        """Programmatically set the recolor checkbox state without
        re-emitting recolorVisibleChanged (avoids a redundant round-
        trip through MainWindow when resetting after a data reload)."""
        self.recolor_check.blockSignals(True)
        self.recolor_check.setChecked(bool(enabled))
        self.recolor_check.blockSignals(False)

    def recolor_window_ms(self) -> float:
        return float(self.recolor_window_spin.value())

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
        quality = self.quality_combo.currentText()
        only_selected = self.show_only_selected_check.isChecked()
        prev_selected = set(self.selected_cluster_ids())

        def matches(u: Unit) -> bool:
            if only_selected and u.cluster_id not in prev_selected:
                return False
            if quality != "all":
                q = (u.quality or "").lower()
                if q != quality:
                    return False
            if not search:
                return True
            haystack = " ".join(str(x) for x in (
                u.cluster_id, u.channel, u.quality, u.kslabel, u.shank
            ) if x is not None).lower()
            return search in haystack

        visible = [u for u in self._rows if matches(u)]

        self.table.setSortingEnabled(False)
        self.table.blockSignals(True)
        self.table.setRowCount(len(visible))
        for row, u in enumerate(visible):
            item_id = QTableWidgetItem(str(u.cluster_id))
            item_id.setData(Qt.ItemDataRole.UserRole, u.cluster_id)
            self.table.setItem(row, 0, item_id)
            self.table.setItem(row, 1, QTableWidgetItem("" if u.channel is None else str(u.channel)))
            self.table.setItem(row, 2, QTableWidgetItem("" if u.depth is None else f"{u.depth:.0f}"))
            self.table.setItem(row, 3, QTableWidgetItem(u.quality or ""))
            self.table.setItem(row, 4, QTableWidgetItem(str(u.n_spikes)))
            self.table.setItem(row, 5, QTableWidgetItem("" if u.firing_rate is None else f"{u.firing_rate:.2f}"))
            self.table.setItem(row, 6, QTableWidgetItem("" if u.shank is None else str(u.shank)))

            if u.cluster_id in prev_selected:
                item_id.setSelected(True)

            # Color-code quality for readability.
            q = (u.quality or "").lower()
            if q == "good":
                for col in range(7):
                    self.table.item(row, col).setForeground(Qt.GlobalColor.green)
            elif q == "noise":
                for col in range(7):
                    self.table.item(row, col).setForeground(Qt.GlobalColor.darkGray)
            elif q == "mua":
                for col in range(7):
                    self.table.item(row, col).setForeground(Qt.GlobalColor.yellow)

        self.table.blockSignals(False)
        self.table.setSortingEnabled(True)

    # ------------------------------------------------------------------
    # Selection events
    # ------------------------------------------------------------------
    def _on_selection_changed(self):
        selected = self.selected_cluster_ids()
        if self.focus_check.isChecked() and len(selected) > 1:
            # Belt-and-suspenders: the table's SelectionMode should
            # already prevent this, but if anything ever slips through,
            # collapse to the first.
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
        self.table.blockSignals(True)
        for row in range(self.table.rowCount()):
            q_item = self.table.item(row, 3)
            if q_item is not None and q_item.text().strip().lower() == "good":
                self.table.selectRow(row)
        self.table.blockSignals(False)
        self.selectionChanged.emit(self.selected_cluster_ids())

    def _clear_selection(self):
        self.table.blockSignals(True)
        self.table.clearSelection()
        self.table.blockSignals(False)
        self.selectionChanged.emit([])


    def _on_focus_toggled(self, checked: bool):
        """Focus mode restricts the table to single-selection and
        emits the new state. The panel itself doesn't know how to
        compute neighborhoods -- it just reports intent; MainWindow
        does the computation."""
        if checked:
            self.table.setSelectionMode(
                QAbstractItemView.SelectionMode.SingleSelection
            )
            # Collapse an existing multi-selection down to the first
            # selected row, so turning focus mode on doesn't leave the
            # table in an inconsistent state.
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
                # Already fine; just re-emit so MainWindow runs its
                # focus pipeline on the now-active mode.
                self.selectionChanged.emit(selected)
            # If nothing selected, nothing to do until the user picks.
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

    def focus_mode_enabled(self) -> bool:
        return bool(self.focus_check.isChecked())

    def focus_neighborhood_size(self) -> int:
        return int(self.focus_n_spin.value())

    def focused_unit_id(self) -> int:
        ids = self.selected_cluster_ids()
        return int(ids[0]) if ids else -1

    def set_focus_mode(self, enabled: bool):
        """Programmatically set focus mode without re-emitting the
        toggle handler's full side-effect chain beyond what's needed."""
        self.focus_check.blockSignals(True)
        self.focus_check.setChecked(bool(enabled))
        self.focus_check.blockSignals(False)
        self._on_focus_toggled(bool(enabled))
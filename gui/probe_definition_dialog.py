"""
probe_definition_dialog.py

Editor dialog for a ProbeDefinition. Lets the user describe a probe
manually (or import one from an existing settings.xml or JSON), then
save it as JSON, export it as settings.xml, or accept it and use it
directly with a dat file for this session.

The design deliberately keeps authoring simple:

  - Top: acquisition params (name, n_channels, sample_rate, dtype,
    bit_volts, probe type).
  - Middle: a "generate linear layout" control (shanks, depth pitch,
    shank x spacing) that fills the electrode table.
  - Bottom: the electrode table itself, editable cell-by-cell for cases
    where the layout isn't perfectly regular.

On accept, `definition` holds the validated ProbeDefinition.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QFileDialog, QMessageBox, QWidget,
)

from core.probe_definition import ProbeDefinition, SUPPORTED_DTYPES


class ProbeDefinitionDialog(QDialog):
    """Modal editor. On exec() == Accepted, `self.definition` is set to
    a validated ProbeDefinition."""

    def __init__(self, parent=None, initial: ProbeDefinition | None = None):
        super().__init__(parent)
        self.setWindowTitle("Probe Definition")
        self.setModal(True)
        self.resize(760, 720)

        self.definition: ProbeDefinition | None = None

        self._build_ui()

        # Seed with a sensible linear layout so the user has something
        # concrete to edit rather than an empty table.
        if initial is not None:
            self._load_definition_into_ui(initial)
        else:
            self._generate_linear_layout()

    # ------------------------------------------------------------------
    # UI scaffolding
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        layout.addWidget(self._build_meta_group())
        layout.addWidget(self._build_layout_generator_group())
        layout.addWidget(self._build_electrode_table(), stretch=1)
        layout.addWidget(self._build_validation_label())
        layout.addLayout(self._build_button_row())

        self._refresh_validation()

    def _build_meta_group(self) -> QGroupBox:
        group = QGroupBox("Acquisition & identification")
        grid = QGridLayout(group)

        self.name_edit = QLineEdit("CustomProbe")
        grid.addWidget(QLabel("Name:"), 0, 0)
        grid.addWidget(self.name_edit, 0, 1)

        self.n_channels_spin = QSpinBox()
        self.n_channels_spin.setRange(1, 100000)
        self.n_channels_spin.setValue(384)
        grid.addWidget(QLabel("n_channels:"), 0, 2)
        grid.addWidget(self.n_channels_spin, 0, 3)

        self.sample_rate_spin = QDoubleSpinBox()
        self.sample_rate_spin.setRange(1.0, 1e6)
        self.sample_rate_spin.setDecimals(1)
        self.sample_rate_spin.setValue(30000.0)
        self.sample_rate_spin.setSuffix(" Hz")
        grid.addWidget(QLabel("Sample rate:"), 0, 4)
        grid.addWidget(self.sample_rate_spin, 0, 5)

        self.dtype_combo = QComboBox()
        self.dtype_combo.addItems(list(SUPPORTED_DTYPES))
        grid.addWidget(QLabel("dtype:"), 1, 0)
        grid.addWidget(self.dtype_combo, 1, 1)

        self.bit_volts_spin = QDoubleSpinBox()
        self.bit_volts_spin.setRange(0.0, 1e6)
        self.bit_volts_spin.setDecimals(6)
        self.bit_volts_spin.setValue(1.0)
        self.bit_volts_spin.setToolTip(
            "Microvolts per raw sample count. Leave at 1.0 to keep the "
            "traces in their native int16 units; set to 0.195 for the "
            "standard Neuropixels LFP scaling."
        )
        grid.addWidget(QLabel("bit_volts:"), 1, 2)
        grid.addWidget(self.bit_volts_spin, 1, 3)

        self.probe_type_edit = QLineEdit("Custom")
        grid.addWidget(QLabel("Probe type:"), 1, 4)
        grid.addWidget(self.probe_type_edit, 1, 5)

        return group

    def _build_layout_generator_group(self) -> QGroupBox:
        group = QGroupBox("Layout generator")
        grid = QGridLayout(group)

        self.n_shanks_spin = QSpinBox()
        self.n_shanks_spin.setRange(1, 64)
        self.n_shanks_spin.setValue(1)
        grid.addWidget(QLabel("Shanks:"), 0, 0)
        grid.addWidget(self.n_shanks_spin, 0, 1)

        self.pitch_spin = QDoubleSpinBox()
        self.pitch_spin.setRange(0.1, 1000.0)
        self.pitch_spin.setDecimals(2)
        self.pitch_spin.setValue(20.0)
        self.pitch_spin.setSuffix(" µm")
        grid.addWidget(QLabel("Depth pitch:"), 0, 2)
        grid.addWidget(self.pitch_spin, 0, 3)

        self.x_spacing_spin = QDoubleSpinBox()
        self.x_spacing_spin.setRange(1.0, 10000.0)
        self.x_spacing_spin.setDecimals(1)
        self.x_spacing_spin.setValue(250.0)
        self.x_spacing_spin.setSuffix(" µm")
        grid.addWidget(QLabel("Shank x spacing:"), 0, 4)
        grid.addWidget(self.x_spacing_spin, 0, 5)

        self.generate_btn = QPushButton("Generate linear layout")
        self.generate_btn.setToolTip(
            "Fill the electrode table with n_channels electrodes laid out "
            "regularly: each shank a column, depth increasing by the "
            "given pitch, shanks separated by the x spacing. This "
            "overwrites any edits in the table."
        )
        self.generate_btn.clicked.connect(self._generate_linear_layout)
        grid.addWidget(self.generate_btn, 1, 0, 1, 6)

        return group

    def _build_electrode_table(self) -> QWidget:
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Channel", "X (µm)", "Y (µm)", "Shank"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self.table.itemChanged.connect(lambda _i: self._refresh_validation())
        return self.table

    def _build_validation_label(self) -> QWidget:
        self.validation_label = QLabel("")
        self.validation_label.setWordWrap(True)
        self.validation_label.setStyleSheet("font-size: 11px;")
        return self.validation_label

    def _build_button_row(self) -> QHBoxLayout:
        row = QHBoxLayout()

        self.load_json_btn = QPushButton("Load JSON...")
        self.load_json_btn.setAutoDefault(False)
        self.load_json_btn.clicked.connect(self._on_load_json)
        row.addWidget(self.load_json_btn)

        self.load_xml_btn = QPushButton("Load settings.xml...")
        self.load_xml_btn.setAutoDefault(False)
        self.load_xml_btn.setToolTip(
            "Import a probe from an existing Open Ephys settings.xml. "
            "The first enabled probe in the file is loaded into the "
            "editor."
        )
        self.load_xml_btn.clicked.connect(self._on_load_settings_xml)
        row.addWidget(self.load_xml_btn)

        row.addStretch(1)

        self.save_json_btn = QPushButton("Save JSON...")
        self.save_json_btn.setAutoDefault(False)
        self.save_json_btn.clicked.connect(self._on_save_json)
        row.addWidget(self.save_json_btn)

        self.save_xml_btn = QPushButton("Save settings.xml...")
        self.save_xml_btn.setAutoDefault(False)
        self.save_xml_btn.setToolTip("Export as an Open Ephys-compatible settings.xml.")
        self.save_xml_btn.clicked.connect(self._on_save_settings_xml)
        row.addWidget(self.save_xml_btn)

        row.addStretch(1)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setAutoDefault(False)
        self.cancel_btn.clicked.connect(self.reject)
        row.addWidget(self.cancel_btn)

        self.ok_btn = QPushButton("Use this definition")
        self.ok_btn.setAutoDefault(False)
        self.ok_btn.setDefault(True)
        self.ok_btn.clicked.connect(self._on_accept)
        row.addWidget(self.ok_btn)

        return row

    # ------------------------------------------------------------------
    # Generator / table sync
    # ------------------------------------------------------------------

    def _generate_linear_layout(self):
        try:
            defn = ProbeDefinition.linear_layout(
                n_channels=self.n_channels_spin.value(),
                sample_rate=self.sample_rate_spin.value(),
                dtype=self.dtype_combo.currentText(),
                n_shanks=self.n_shanks_spin.value(),
                pitch_um=self.pitch_spin.value(),
                x_shank_spacing_um=self.x_spacing_spin.value(),
                name=self.name_edit.text().strip() or "CustomProbe",
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot generate layout", str(exc))
            return
        defn.bit_volts = self.bit_volts_spin.value()
        defn.probe_type = self.probe_type_edit.text().strip() or "Custom"
        self._load_definition_into_ui(defn)

    def _load_definition_into_ui(self, defn: ProbeDefinition):
        # Block signals while populating so the itemChanged handler
        # doesn't fire validation N times during a bulk load.
        self.table.blockSignals(True)
        try:
            self.name_edit.setText(defn.name)
            self.n_channels_spin.setValue(int(defn.n_channels))
            self.sample_rate_spin.setValue(float(defn.sample_rate))
            if defn.dtype in SUPPORTED_DTYPES:
                self.dtype_combo.setCurrentText(defn.dtype)
            self.bit_volts_spin.setValue(float(defn.bit_volts))
            self.probe_type_edit.setText(defn.probe_type)

            n = len(defn.channels)
            self.table.setRowCount(n)
            for row in range(n):
                self._set_row(row, defn.channels[row], defn.x[row], defn.y[row], defn.shank_ids[row])
        finally:
            self.table.blockSignals(False)
        self._refresh_validation()

    def _set_row(self, row: int, ch: int, x: float, y: float, shank: int):
        self.table.setItem(row, 0, QTableWidgetItem(str(int(ch))))
        self.table.setItem(row, 1, QTableWidgetItem(f"{float(x):.2f}"))
        self.table.setItem(row, 2, QTableWidgetItem(f"{float(y):.2f}"))
        self.table.setItem(row, 3, QTableWidgetItem(str(int(shank))))

    # ------------------------------------------------------------------
    # Reading the UI back into a definition
    # ------------------------------------------------------------------

    def _definition_from_ui(self) -> ProbeDefinition:
        n_rows = self.table.rowCount()

        channels: list[int] = []
        xs: list[float] = []
        ys: list[float] = []
        shanks: list[int] = []

        for row in range(n_rows):
            def cell_text(col: int) -> str:
                item = self.table.item(row, col)
                return item.text().strip() if item is not None else ""

            try:
                channels.append(int(cell_text(0)))
            except ValueError:
                channels.append(-1)
            try:
                xs.append(float(cell_text(1)))
            except ValueError:
                xs.append(0.0)
            try:
                ys.append(float(cell_text(2)))
            except ValueError:
                ys.append(0.0)
            try:
                shanks.append(int(cell_text(3)))
            except ValueError:
                shanks.append(0)

        return ProbeDefinition(
            name=self.name_edit.text().strip() or "CustomProbe",
            n_channels=int(self.n_channels_spin.value()),
            sample_rate=float(self.sample_rate_spin.value()),
            dtype=self.dtype_combo.currentText(),
            bit_volts=float(self.bit_volts_spin.value()),
            probe_type=self.probe_type_edit.text().strip() or "Custom",
            channels=channels, x=xs, y=ys, shank_ids=shanks,
        )

    def _refresh_validation(self):
        defn = self._definition_from_ui()
        problems = defn.validate()
        if problems:
            self.validation_label.setText("⚠  " + "\n⚠  ".join(problems))
            self.validation_label.setStyleSheet("color: #e57373; font-size: 11px;")
            self.ok_btn.setEnabled(False)
        else:
            self.validation_label.setText("✓  Definition looks valid.")
            self.validation_label.setStyleSheet("color: #4caf50; font-size: 11px;")
            self.ok_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Import / export
    # ------------------------------------------------------------------

    def _on_load_json(self):
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Load Probe Definition (JSON)", "", "JSON files (*.json);;All files (*)"
        )
        if not path_str:
            return
        try:
            defn = ProbeDefinition.load_json(path_str)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to load", f"Could not read {path_str}:\n\n{exc}")
            return
        self._load_definition_into_ui(defn)

    def _on_load_settings_xml(self):
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Load Open Ephys settings.xml", "", "XML files (*.xml);;All files (*)"
        )
        if not path_str:
            return
        try:
            from core.probe_extractor import extract_probes_from_settings
            probes = extract_probes_from_settings(path_str)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to parse", f"Could not parse {path_str}:\n\n{exc}")
            return

        if not probes.get("probes"):
            QMessageBox.warning(self, "No probes", f"{path_str} contained no enabled probes.")
            return

        # Take the first probe. Importing every probe in a multi-probe
        # file into ONE definition isn't meaningful (they're different
        # physical devices), and this dialog is scoped to one.
        first_key = next(iter(probes["probes"]))
        probe_data = probes["probes"][first_key]
        defn = ProbeDefinition.load_from_probe_data_dict(probe_data, name=first_key)
        self._load_definition_into_ui(defn)

    def _on_save_json(self):
        defn = self._definition_from_ui()
        if defn.validate():
            QMessageBox.warning(self, "Invalid definition", "\n".join(defn.validate()))
            return
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Save Probe Definition (JSON)", f"{defn.name}.json",
            "JSON files (*.json);;All files (*)"
        )
        if not path_str:
            return
        try:
            defn.save_json(path_str)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to save", str(exc))
            return
        QMessageBox.information(self, "Saved", f"Saved definition to {path_str}.")

    def _on_save_settings_xml(self):
        defn = self._definition_from_ui()
        if defn.validate():
            QMessageBox.warning(self, "Invalid definition", "\n".join(defn.validate()))
            return
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Export as settings.xml", f"{defn.name}_settings.xml",
            "XML files (*.xml);;All files (*)"
        )
        if not path_str:
            return
        try:
            defn.save_settings_xml(path_str)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to save", str(exc))
            return
        QMessageBox.information(self, "Saved", f"Wrote settings.xml to {path_str}.")

    # ------------------------------------------------------------------
    # Accept
    # ------------------------------------------------------------------

    def _on_accept(self):
        defn = self._definition_from_ui()
        problems = defn.validate()
        if problems:
            QMessageBox.warning(
                self, "Cannot use this definition",
                "Fix the following first:\n\n" + "\n".join(f"• {p}" for p in problems)
            )
            return
        self.definition = defn
        self.accept()
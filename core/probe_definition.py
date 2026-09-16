"""
probe_definition.py

A user-authorable probe description: enough geometry + acquisition
metadata to drive the whole pipeline (trace view, probe map, CSD, ripple,
theta, PAC, spatial power, RTA) against a raw continuous.dat that has no
accompanying Open Ephys settings.xml.

The output of to_probe_data_dict() is deliberately IDENTICAL in shape to
what extract_probes_from_settings()["probes"][<key>] returns, so anything
downstream that consumes a probe dict works unchanged.

Two on-disk formats are supported:

  - JSON (.json)             -- primary, self-contained, human-editable.
  - Open Ephys settings.xml  -- import/export, so a hand-built definition
                                can round-trip through Open Ephys.

JSON schema (all keys required except where noted):

    {
      "name": "MyProbe",                  # display name / stream key
      "n_channels": 384,                  # channels per sample in the dat
      "sample_rate": 30000.0,             # Hz
      "dtype": "int16",                   # numpy dtype string
      "bit_volts": 0.195,                 # optional; µV per int16 count
      "probe_type": "Neuropixels 2.0",    # optional; display only
      "serial_number": "PRB_1234",        # optional; display only
      "coordinates": {
        "channels": [0, 1, 2, ...],       # per-row: which dat row
        "x":        [0, 0, 0, ...],       # µm
        "y":        [0, 20, 40, ...],     # µm
      },
      "shanks": {
        "ids": [0, 0, 0, ...],            # one per row of coordinates
      }
    }

The three coordinate arrays and shanks.ids must all be the same length,
which must equal n_channels. to_probe_data_dict() enforces this.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# NumPy dtype strings we accept. Kept narrow on purpose -- these are the
# ones Open Ephys and SpikeGLX actually write, and anything else is much
# more likely to be a user mistake than an exotic valid case.
SUPPORTED_DTYPES = ("int16", "int32", "float32")


@dataclass
class ProbeDefinition:
    """A complete, self-contained probe description."""

    name: str = "CustomProbe"
    n_channels: int = 384
    sample_rate: float = 30000.0
    dtype: str = "int16"
    bit_volts: float = 1.0
    probe_type: str = "Custom"
    serial_number: str = ""

    # Per-electrode arrays. All three plus shank_ids must be the same
    # length (= n_channels). Stored as plain Python lists so JSON
    # round-trips are trivial and there is no numpy-vs-list ambiguity.
    channels: list[int] = field(default_factory=list)
    x: list[float] = field(default_factory=list)
    y: list[float] = field(default_factory=list)
    shank_ids: list[int] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Return a list of human-readable problems; empty means valid.
        Non-fatal warnings (things that would still load but look odd)
        are returned the same way -- the caller decides how to surface
        them."""
        problems: list[str] = []

        if self.n_channels <= 0:
            problems.append("n_channels must be positive.")
        if self.sample_rate <= 0:
            problems.append("sample_rate must be positive.")
        if self.dtype not in SUPPORTED_DTYPES:
            problems.append(f"dtype must be one of {SUPPORTED_DTYPES}, got {self.dtype!r}.")

        n = self.n_channels
        for name in ("channels", "x", "y", "shank_ids"):
            arr = getattr(self, name)
            if len(arr) != n:
                problems.append(f"{name} has {len(arr)} entries but n_channels is {n}.")

        if len(self.channels) != len(set(self.channels)):
            dupes = [c for c in set(self.channels) if self.channels.count(c) > 1]
            problems.append(f"duplicate channel numbers in coordinates: {sorted(dupes)[:5]}...")

        # A channel index outside [0, n_channels) means a coordinate row
        # points at a dat row that doesn't exist -- that will slice off
        # the end of the memmap and look like missing channels.
        bad = [c for c in self.channels if c < 0 or c >= n]
        if bad:
            problems.append(f"{len(bad)} channel(s) fall outside [0, {n}): {bad[:5]}...")

        return problems

    def is_valid(self) -> bool:
        return not self.validate()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def linear_layout(cls, n_channels: int = 384, sample_rate: float = 30000.0,
                       dtype: str = "int16", n_shanks: int = 1,
                       pitch_um: float = 20.0, x_shank_spacing_um: float = 250.0,
                       name: str = "CustomProbe") -> "ProbeDefinition":
        """Build a definition with a simple, regular layout: n_shanks
        columns of electrodes, each column evenly spaced in depth by
        pitch_um, columns separated in x by x_shank_spacing_um. Channel
        numbers run 0..n_channels-1 in the order the electrodes are laid
        out (shank-major, then depth), which is what the dat rows will
        be assumed to contain.

        If n_channels isn't divisible by n_shanks, the remainder goes
        to the first shank(s) so the total is exact."""
        if n_channels <= 0:
            raise ValueError("n_channels must be positive.")
        if n_shanks <= 0:
            raise ValueError("n_shanks must be positive.")

        per_shank = [n_channels // n_shanks] * n_shanks
        for i in range(n_channels % n_shanks):
            per_shank[i] += 1

        channels: list[int] = []
        xs: list[float] = []
        ys: list[float] = []
        shank_ids: list[int] = []

        next_ch = 0
        for shank, count in enumerate(per_shank):
            # Center the shank columns on x=0 so the probe map opens
            # centered on screen.
            x = (shank - (n_shanks - 1) / 2.0) * x_shank_spacing_um
            for row in range(count):
                channels.append(next_ch)
                xs.append(float(x))
                ys.append(float(row) * float(pitch_um))
                shank_ids.append(int(shank))
                next_ch += 1

        return cls(name=name, n_channels=n_channels, sample_rate=sample_rate,
                   dtype=dtype, channels=channels, x=xs, y=ys, shank_ids=shank_ids)

    # ------------------------------------------------------------------
    # Downstream conversion
    # ------------------------------------------------------------------

    def to_probe_data_dict(self) -> dict:
        """Return the same dict shape extract_probes_from_settings
        produces for one probe. This is the bridge that lets a
        hand-authored definition drive the rest of the app unchanged.

        Raises ValueError if validate() fails -- callers should have
        checked already; this is a guard against silently shipping a
        malformed definition into the pipeline."""
        problems = self.validate()
        if problems:
            raise ValueError("Invalid probe definition:\n  - " + "\n  - ".join(problems))

        return {
            "stream_name": self.name,
            "sample_rate": float(self.sample_rate),
            "channel_count": int(self.n_channels),
            "device_name": self.probe_type,
            "coordinates": {
                "channels": [int(c) for c in self.channels],
                "x": [float(v) for v in self.x],
                "y": [float(v) for v in self.y],
            },
            "shanks": {
                "ids": [int(s) for s in self.shank_ids],
                "mapping": {int(c): int(s) for c, s in zip(self.channels, self.shank_ids)},
            },
            "num_shanks": len(set(self.shank_ids)) if self.shank_ids else 1,
            "electrode_config": "CUSTOM",
            "serial_number": self.serial_number,
            "enabled": True,
            "node_id": "custom",
            # Extras kept on the dict so future features (bit_volts
            # scaling, dtype-aware loading) can read them without
            # re-plumbing. Downstream code ignores unknown keys.
            "dtype": self.dtype,
            "bit_volts": float(self.bit_volts),
            "is_custom": True,
        }

    # ------------------------------------------------------------------
    # JSON I/O
    # ------------------------------------------------------------------

    def to_json_dict(self) -> dict:
        return {
            "name": self.name,
            "n_channels": int(self.n_channels),
            "sample_rate": float(self.sample_rate),
            "dtype": self.dtype,
            "bit_volts": float(self.bit_volts),
            "probe_type": self.probe_type,
            "serial_number": self.serial_number,
            "coordinates": {
                "channels": [int(c) for c in self.channels],
                "x": [float(v) for v in self.x],
                "y": [float(v) for v in self.y],
            },
            "shanks": {
                "ids": [int(s) for s in self.shank_ids],
            },
        }

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_json_dict(), f, indent=2)
        return path

    @classmethod
    def load_json(cls, path: str | Path) -> "ProbeDefinition":
        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)

        coords = data.get("coordinates", {})
        shanks = data.get("shanks", {})

        return cls(
            name=str(data.get("name", "CustomProbe")),
            n_channels=int(data.get("n_channels", 0)),
            sample_rate=float(data.get("sample_rate", 30000.0)),
            dtype=str(data.get("dtype", "int16")),
            bit_volts=float(data.get("bit_volts", 1.0)),
            probe_type=str(data.get("probe_type", "Custom")),
            serial_number=str(data.get("serial_number", "")),
            channels=[int(c) for c in coords.get("channels", [])],
            x=[float(v) for v in coords.get("x", [])],
            y=[float(v) for v in coords.get("y", [])],
            shank_ids=[int(s) for s in shanks.get("ids", [])],
        )

    # ------------------------------------------------------------------
    # Open Ephys settings.xml I/O
    # ------------------------------------------------------------------
    #
    # The XML we emit is a minimal-but-valid skeleton that OUR OWN
    # extract_probes_from_settings() reads correctly, so a definition
    # saved this way can be re-imported via the normal File -> Open
    # settings.xml path. It is NOT a byte-for-byte clone of a real
    # recording settings.xml -- those have hundreds of fields that Open
    # Ephys adds and we don't use.

    def save_settings_xml(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        root = ET.Element("SETTINGS")
        control = ET.SubElement(root, "CONTROLPANEL")
        control.set("recordPath", str(path.parent))

        proc = ET.SubElement(root, "PROCESSOR")
        proc.set("pluginName", "Neuropix-PXI")
        proc.set("name", "ProbeA")
        proc.set("nodeId", "100")

        stream = ET.SubElement(proc, "STREAM")
        stream.set("name", "ProbeA-AP")
        stream.set("sample_rate", str(self.sample_rate))
        stream.set("channel_count", str(self.n_channels))
        stream.set("device_name", self.probe_type or "Custom")

        np_probe = ET.SubElement(proc, "NP_PROBE")
        np_probe.set("probe_name", self.probe_type or "Custom")
        np_probe.set("probe_serial_number", self.serial_number)
        np_probe.set("electrodeConfigurationPreset", "CUSTOM")
        np_probe.set("isEnabled", "1")

        xpos = ET.SubElement(np_probe, "ELECTRODE_XPOS")
        for ch, xv in zip(self.channels, self.x):
            xpos.set(f"CH{int(ch)}", str(int(round(xv))))

        ypos = ET.SubElement(np_probe, "ELECTRODE_YPOS")
        for ch, yv in zip(self.channels, self.y):
            ypos.set(f"CH{int(ch)}", str(int(round(yv))))

        chan_elem = ET.SubElement(np_probe, "CHANNELS")
        for ch, sh in zip(self.channels, self.shank_ids):
            # Open Ephys encodes CHANNELS values as "electrode:shank".
            chan_elem.set(f"CH{int(ch)}", f"{int(ch)}:{int(sh)}")

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(path, encoding="utf-8", xml_declaration=True)
        return path

    @classmethod
    def load_from_probe_data_dict(cls, probe_data: dict, name: str | None = None) -> "ProbeDefinition":
        """Build a definition from one probe's extract_probes_from_settings
        output. This is how a real recording's probe gets imported into
        the editor for inspection or modification."""
        coords = probe_data.get("coordinates", {})
        shanks = probe_data.get("shanks", {})

        return cls(
            name=name or probe_data.get("stream_name", "ImportedProbe"),
            n_channels=int(probe_data.get("channel_count", len(coords.get("channels", [])))),
            sample_rate=float(probe_data.get("sample_rate", 30000.0)),
            dtype=str(probe_data.get("dtype", "int16")),
            bit_volts=float(probe_data.get("bit_volts", 1.0)),
            probe_type=str(probe_data.get("device_name", "Custom")),
            serial_number=str(probe_data.get("serial_number", "")),
            channels=[int(c) for c in coords.get("channels", [])],
            x=[float(v) for v in coords.get("x", [])],
            y=[float(v) for v in coords.get("y", [])],
            shank_ids=[int(s) for s in shanks.get("ids", [])],
        )


def dat_implied_duration_seconds(dat_path: str | Path, n_channels: int, dtype: str) -> float:
    """Return the recording duration (in seconds) implied by a dat file
    of the given channel count and dtype, assuming a sample rate of 1 Hz
    -- i.e. this is the number of samples. Callers divide by their
    sample_rate to get seconds. Broken out so a UI can preview the
    duration for candidate parameters without committing to them."""
    dat_path = Path(dat_path)
    if not dat_path.exists() or n_channels <= 0:
        return 0.0
    size = dat_path.stat().st_size
    itemsize = np.dtype(dtype).itemsize
    return size / (n_channels * itemsize)
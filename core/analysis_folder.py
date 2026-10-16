"""
analysis_folder.py

Locates and creates the per-recording analysis folder that detection
dialogs (ripples, theta epochs, theta cycles) write their CSVs and
sidecars into.

Layout
------
    <dat_dir>/
      continuous.dat
      neuropyxels/
        ripples.csv
        ripples.meta.json

The folder is named "neuropyxels" and lives next to the loaded .dat, so
it travels with the recording and one recording's outputs never mix
with another's.

No GUI dependency.
"""

from __future__ import annotations

from pathlib import Path


FOLDER_NAME = "neuropyxels"


def analysis_folder_for(engine) -> Path | None:
    """Return the analysis folder path for the engine's loaded .dat, or
    None if the engine has no data file (nothing to anchor the folder
    to). The folder is NOT created here -- callers create it on first
    write via .mkdir(parents=True, exist_ok=True)."""
    data_file = getattr(engine, "data_file", None)
    if data_file is None:
        return None
    data_file = Path(data_file)
    return data_file.parent / FOLDER_NAME


def ensure_analysis_folder(engine) -> Path | None:
    """Like analysis_folder_for, but also creates the folder if it
    doesn't exist. Returns None if the engine has no data file."""
    folder = analysis_folder_for(engine)
    if folder is None:
        return None
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def ripples_csv_path(engine) -> Path | None:
    folder = analysis_folder_for(engine)
    if folder is None:
        return None
    return folder / "ripples.csv"


def ripples_meta_path(engine) -> Path | None:
    folder = analysis_folder_for(engine)
    if folder is None:
        return None
    return folder / "ripples.meta.json"


def theta_epochs_csv_path(engine) -> Path | None:
    folder = analysis_folder_for(engine)
    if folder is None:
        return None
    return folder / "theta_epochs.csv"


def theta_epochs_meta_path(engine) -> Path | None:
    folder = analysis_folder_for(engine)
    if folder is None:
        return None
    return folder / "theta_epochs.meta.json"


def theta_cycles_csv_path(engine) -> Path | None:
    folder = analysis_folder_for(engine)
    if folder is None:
        return None
    return folder / "theta_cycles.csv"


def theta_cycles_meta_path(engine) -> Path | None:
    folder = analysis_folder_for(engine)
    if folder is None:
        return None
    return folder / "theta_cycles.meta.json"
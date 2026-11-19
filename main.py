"""
main.py

Entry point for data_explorer.

Usage:
    python main.py
    python main.py path/to/settings.xml
"""
import os
import sys
from pathlib import Path

# Wayland's QDockWidget grab is broken in Qt6 (QTBUG-129362 and
# related): floating dock widgets cannot be dragged under native
# Wayland. Run under XWayland on Linux to restore drag behavior.
# Guarded to Linux only -- on Windows and macOS the default platform
# plugin is already correct, and forcing "xcb" there would fail with
# "Could not find the Qt platform plugin xcb".
if sys.platform.startswith("linux"):
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

from PyQt6.QtWidgets import QApplication

from gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("data_explorer")

    window = MainWindow()
    window.show()

    if len(sys.argv) > 1:
        settings_path = Path(sys.argv[1])
        if settings_path.exists():
            window.load_settings_file(settings_path)
        else:
            print(f"Warning: {settings_path} does not exist, skipping auto-load.")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
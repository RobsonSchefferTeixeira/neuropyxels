"""
main.py

Entry point for data_explorer.

Usage:
    python main.py
    python main.py path/to/settings.xml
"""
import sys
from pathlib import Path

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
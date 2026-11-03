"""
gui/plot_utils.py

Small helpers shared by the pyqtgraph-based analysis figures.
"""

from __future__ import annotations

import pyqtgraph as pg

def attach_double_click_fit(plot_widget: pg.PlotWidget, fit_callback=None):
    """Wire a double-click on `plot_widget`'s scene to a "restore fit
    to screen" action.
    """
    scene = plot_widget.scene()

    def _on_mouse_clicked(evt):
        # GraphicsScene has no sigMouseDoubleClicked; the click event
        # itself carries a .double() flag that is True on the second
        # click of a double-click.
        if not evt.double():
            return
        if fit_callback is not None:
            try:
                fit_callback()
            except Exception as exc:
                print(f"[plot-utils] fit callback failed: {exc}")
        else:
            try:
                plot_widget.enableAutoRange()
                plot_widget.autoRange()
            except Exception as exc:
                print(f"[plot-utils] default fit failed: {exc}")

    scene.sigMouseClicked.connect(_on_mouse_clicked)
    
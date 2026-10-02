"""
neural_trace_view.py

Combines the interactive ripple-event overlay and theta-epoch overlay
into a single trace view widget.

IMPORTANT: this is a single-parent subclass, NOT multiple inheritance.
An earlier version combined RippleTraceViewWidget and
ThetaEpochTraceViewWidget via
class NeuralTraceViewWidget(RippleTraceViewWidget, ThetaEpochTraceViewWidget)
-- two independent QWidget subclasses composed side-by-side. That
pattern is NOT supported by PyQt6's meta-object system: pyqtSignal
attributes declared on whichever base ends up second in the MRO fail to
bind at connect() time (confirmed directly: connecting to
thetaEpochsChanged raised "QObject::connect: Use the SIGNAL macro to
bind NeuralTraceViewWidget::(PyQt_PyObject)" / "TypeError: connect()
failed between [object] and <slot>"). Ripple's signals, declared on
whichever base was first in the MRO, bound fine -- only the second
base's signals broke, which is exactly what this fix addresses.

The real fix is that RippleTraceViewWidget (gui/ripple_trace_view.py)
now inherits DIRECTLY from ThetaEpochTraceViewWidget instead of from
TraceViewWidget, giving PyQt a normal single, linear class hierarchy:

    NeuralTraceViewWidget -> RippleTraceViewWidget
                           -> ThetaEpochTraceViewWidget
                           -> TraceViewWidget -> QWidget

This is the same kind of chain that already worked reliably for
ThetaEpochTraceViewWidget -> TraceViewWidget before ripples existed;
it's just one link longer now. This file is consequently just an
identity subclass -- it exists so the combined widget has its own
class name/type (useful for isinstance checks, debugging, and as the
single import MainWindow uses) without repeating any logic.

Paint/event ordering (unchanged from the original multiple-inheritance
design's intent, now achieved via the linear chain instead): a single
paintEvent() call unwinds through each class's super().paintEvent()
call in turn, so base traces draw first (TraceViewWidget), then theta
epochs on top of that (ThetaEpochTraceViewWidget), then ripple events
on top of everything (RippleTraceViewWidget). Mouse/key events work the
same way in reverse: RippleTraceViewWidget's handlers get first chance
to consume a click/keypress (its own hit-test), falling through via
super() to ThetaEpochTraceViewWidget's handlers, which fall through to
TraceViewWidget's base pan/zoom/cursor logic if neither overlay claims
the event.

Shortcut relay
--------------
This widget also relays two shortcut families upward so the user can
curate units without clicking back to the Phy Units panel:

  * Bare E/G/P/U/M/N -> reclassifyShortcutRequested(str), a class name.
    MainWindow resolves which unit is focused and applies the change.
  * Bare PageUp / PageDown -> spikeNavigationRequested(-1/+1), the
    existing signal declared on TraceViewWidget. MainWindow already
    routes that into _jump_to_adjacent_spike.

Ctrl+PageUp / Ctrl+PageDown are deliberately NOT consumed here -- they
fall through to TraceViewWidget.keyPressEvent, which handles them as
zoom in / zoom out.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal

from gui.ripple_trace_view import RippleTraceViewWidget


# Bare-letter -> class. Same mapping as gui/phy_units_panel.py's
# _ReclassifyTable. Duplicated here rather than imported so the trace
# view has no dependency on the panel module.
_RECLASSIFY_SHORTCUTS: dict[int, str] = {
    Qt.Key.Key_E: "excellent",
    Qt.Key.Key_G: "good",
    Qt.Key.Key_P: "poor",
    Qt.Key.Key_U: "unsorted",
    Qt.Key.Key_M: "mua",
    Qt.Key.Key_N: "noise",
}


class NeuralTraceViewWidget(RippleTraceViewWidget):
    """TraceViewWidget with both ripple-event and theta-epoch
    interactive overlays active simultaneously, via a single linear
    inheritance chain (see module docstring for why NOT multiple
    inheritance).

    Also relays bare-letter reclassify shortcuts and bare PageUp/Down
    spike navigation upward -- see module docstring.
    """

    reclassifyShortcutRequested = pyqtSignal(str)  # new class name

    def keyPressEvent(self, event):
        # Auto-repeat off for all shortcut handling -- holding a key
        # must not blow through dozens of units / spikes in a second.
        if event.isAutoRepeat():
            super().keyPressEvent(event)
            return

        mods = event.modifiers()
        key = event.key()

        ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
        alt = bool(mods & Qt.KeyboardModifier.AltModifier)
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        meta = bool(mods & Qt.KeyboardModifier.MetaModifier)

        # Ctrl+PageUp / Ctrl+PageDown: pass through to the base class
        # (zoom in / zoom out). Must be checked before the bare
        # PageUp/PageDown branch below, or Ctrl-modified presses would
        # be swallowed as spike navigation.
        if ctrl and not (alt or shift or meta) and key in (
            Qt.Key.Key_PageUp, Qt.Key.Key_PageDown
        ):
            super().keyPressEvent(event)
            return

        # Anything with a modifier other than bare Shift is not ours.
        if alt or ctrl or meta or shift:
            super().keyPressEvent(event)
            return

        # Bare PageUp / PageDown: spike navigation of the focused unit
        # (or the single selected unit if focus mode is off). The
        # routing decision -- including the "multiple units selected,
        # show a status message" case -- lives in MainWindow's
        # _jump_to_adjacent_spike, which this signal already reaches.
        if key == Qt.Key.Key_PageUp:
            self.spikeNavigationRequested.emit(-1)
            event.accept()
            return
        if key == Qt.Key.Key_PageDown:
            self.spikeNavigationRequested.emit(+1)
            event.accept()
            return

        # Bare E/G/P/U/M/N: reclassify the focused unit, then advance.
        class_name = _RECLASSIFY_SHORTCUTS.get(key)
        if class_name is not None:
            self.reclassifyShortcutRequested.emit(class_name)
            event.accept()
            return

        super().keyPressEvent(event)
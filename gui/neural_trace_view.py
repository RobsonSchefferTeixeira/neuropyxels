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
"""

from __future__ import annotations

from gui.ripple_trace_view import RippleTraceViewWidget


class NeuralTraceViewWidget(RippleTraceViewWidget):
    """TraceViewWidget with both ripple-event and theta-epoch
    interactive overlays active simultaneously, via a single linear
    inheritance chain (see module docstring for why NOT multiple
    inheritance)."""
    pass


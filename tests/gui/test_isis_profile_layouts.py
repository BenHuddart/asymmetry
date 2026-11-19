"""Actual ISIS renderer and hit selection retain the manual acquired ordinal."""

import math
from types import SimpleNamespace

import pytest
from PySide6.QtTest import QSignalSpy

from asymmetry.core.instrument import get_instrument_layout
from asymmetry.gui.widgets.detector_schematic import DetectorSchematicWidget

pytestmark = pytest.mark.gui


@pytest.mark.parametrize("name,angle", [("HiFi", 270.0), ("MuSR", 230.625)])
def test_isis_declarations_render_and_pick_same_histogram(qapp, name, angle):
    widget = DetectorSchematicWidget(get_instrument_layout(name))
    widget.resize(1000, 600)
    widget.show()
    qapp.processEvents()
    widget._canvas.draw()
    assert set(widget._patches) == set(range(1, 65))
    signal = QSignalSpy(widget.detector_toggled)
    widget._on_click(
        SimpleNamespace(
            button=1,
            inaxes=widget._axes[0],
            xdata=0.64 * math.cos(math.radians(angle)),
            ydata=0.64 * math.sin(math.radians(angle)),
        )
    )
    assert signal.count() == 1
    assert signal.at(0)[0] == 1

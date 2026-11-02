"""The Auto Phase documentation example must render through the GUI workflow."""

import pytest
from PySide6.QtGui import QImage

from docs.screenshots.scenarios._base import CaptureContext
from docs.screenshots.scenarios.fourier_tf import FourierAutoPhaseScenario


@pytest.mark.gui
def test_auto_phase_documentation_scenario_renders(qapp, tmp_path):
    path = FourierAutoPhaseScenario().capture(CaptureContext(output_dir=tmp_path))

    assert not QImage(str(path)).isNull()

"""The public preset cookbook runs from synthetic file loading to calibrated reduction."""

from pathlib import Path

import numpy as np
import pytest


def test_synthetic_preset_cookbook():
    pytest.importorskip("h5py")
    cookbook = Path(__file__).resolve().parents[2] / "docs/reference/cookbook.rst"
    section = cookbook.read_text().split(".. _preset-scripting-example:", 1)[1]
    block = section.split(".. code-block:: python\n\n", 1)[1]
    lines = []
    for line in block.splitlines():
        if line and not line.startswith("   "):
            break
        lines.append(line[3:] if line else "")
    namespace = {}
    exec("\n".join(lines), namespace)
    reduced = namespace["reduced"]
    assert np.all(np.isfinite(reduced.asymmetry))
    assert np.all(reduced.error > 0)
    assert namespace["loaded"].run.grouping != namespace["prepared"].run.grouping

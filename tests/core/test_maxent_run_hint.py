"""Discoverability hint for passing a MuonDataset to MaxEnt.

MaxEnt is a grouped raw-count algorithm and needs a :class:`Run` (the raw
detector histograms), not a :class:`MuonDataset` (the reduced asymmetry curve).
Passing the dataset previously failed with a cryptic
``AttributeError: 'MuonDataset' object has no attribute 'grouping'``; it must
now raise a TypeError that names the fix (``pass load(path).run``).
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.maxent import build_maxent_input, maxent


def _reduced_dataset() -> MuonDataset:
    return MuonDataset(
        time=np.arange(10.0),
        asymmetry=np.zeros(10),
        error=np.ones(10),
        metadata={},
        run=None,
    )


@pytest.mark.parametrize("entry_point", [maxent, build_maxent_input])
def test_dataset_raises_pointing_type_error(entry_point) -> None:
    with pytest.raises(TypeError) as excinfo:
        entry_point(_reduced_dataset())

    message = str(excinfo.value)
    assert "Run" in message
    assert "MuonDataset" in message
    assert ".run" in message


@pytest.mark.parametrize("entry_point", [maxent, build_maxent_input])
def test_unexpected_type_raises_pointing_type_error(entry_point) -> None:
    with pytest.raises(TypeError) as excinfo:
        entry_point(object())

    assert ".run" in str(excinfo.value)

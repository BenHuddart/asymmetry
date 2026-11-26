"""Half-recovery interpolation for the corannulene documentation figure."""

import numpy as np
import pytest

from docs.screenshots.scenarios.corpus.corannulene_ulcr import _normalised_recovery

pytestmark = pytest.mark.gui


def test_half_recovery_is_independent_of_step_height():
    field = np.array([0.0, 3.0, 60.0, 140.0, 1_000.0, 1_800.0, 2_500.0])
    shape = np.clip((field - 20.0) / 160.0, 0.0, 1.0)
    small_step = 0.12 + 0.08 * shape
    large_step = 0.22 + 0.20 * shape

    assert np.ptp(small_step) != pytest.approx(np.ptp(large_step))
    for asymmetry in (small_step, large_step):
        recovery, half_field = _normalised_recovery(field, asymmetry)
        np.testing.assert_allclose(recovery, shape)
        assert half_field == pytest.approx(100.0)


@pytest.mark.parametrize(
    "asymmetry",
    [
        [0.0, 0.0, 0.2, 0.3, 0.35, 0.4, 1.2, 1.4],
        [0.0, 0.0, 0.48, 0.51, 0.49, 1.0, 1.0, 1.0],
    ],
    ids=["no-bracket", "noise-recrosses-half-recovery"],
)
def test_half_recovery_requires_one_rising_bracket(asymmetry):
    field = np.array([0.0, 3.0, 40.0, 80.0, 120.0, 1_000.0, 1_800.0, 2_500.0])

    with pytest.raises(ValueError, match="one rising bracket"):
        _normalised_recovery(field, np.asarray(asymmetry))

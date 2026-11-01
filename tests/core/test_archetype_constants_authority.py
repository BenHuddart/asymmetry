"""Rounded teaching constants are derived from the physical-constant authority."""

from __future__ import annotations

import pytest

from asymmetry.core import simulate_presets
from asymmetry.core.utils import constants
from docs.screenshots.data import archetypes


@pytest.mark.parametrize(
    ("preset", "screenshot", "expected"),
    [
        (simulate_presets.GAMMA_MU_MHZ_PER_G, archetypes.GAMMA_MU_MHZ_PER_G, 0.01355),
        (simulate_presets.DELTA_AG_PER_US, archetypes.DELTA_AG_PER_US, 0.39),
        (simulate_presets.TC_EUO_K, archetypes.TC_EUO_K, 69.0),
        (simulate_presets.R_MUF_ANG, archetypes.R_MUF_ANG, 1.17),
    ],
)
def test_presets_and_screenshots_share_the_teaching_values(
    preset: float, screenshot: float, expected: float
) -> None:
    assert preset == expected
    assert screenshot is preset


def test_frequency_and_lifetime_rounding_comes_from_the_authority() -> None:
    assert simulate_presets.GAMMA_MU_MHZ_PER_G == round(
        constants.MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * constants.GAUSS_TO_TESLA, 5
    )
    assert archetypes.GAMMA_MU_MHZ_PER_T == round(constants.MUON_GYROMAGNETIC_RATIO_MHZ_PER_T, 1)
    assert archetypes.MUON_LIFETIME_US == round(constants.MUON_LIFETIME_US, 3)
    assert archetypes.MUON_LIFETIME_US == 2.197

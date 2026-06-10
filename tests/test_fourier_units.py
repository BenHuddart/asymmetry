"""Field ↔ frequency unit conversions (shared spectral-display helper)."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.fourier.units import (
    FieldUnit,
    axis_label,
    convert,
    frequency_resolution_mhz,
    gauss_to_mhz,
    gauss_to_tesla,
    mhz_to_gauss,
    mhz_to_tesla,
    tesla_to_mhz,
)
from asymmetry.core.utils.constants import (
    GAUSS_TO_TESLA,
    MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
)


def test_mhz_tesla_uses_codata_gyromagnetic_ratio() -> None:
    # ν = γ_μ B / 2π with γ_μ/2π = 135.538817 MHz/T.
    assert mhz_to_tesla(MUON_GYROMAGNETIC_RATIO_MHZ_PER_T) == pytest.approx(1.0)
    assert tesla_to_mhz(1.0) == pytest.approx(MUON_GYROMAGNETIC_RATIO_MHZ_PER_T)


def test_mhz_gauss_round_trip() -> None:
    values = np.array([0.0, 1.5, 13.55, 135.5])
    np.testing.assert_allclose(gauss_to_mhz(mhz_to_gauss(values)), values, atol=1e-12)
    # 1 T = 1e4 G, so 1 G ↔ γ·1e-4 MHz.
    assert gauss_to_mhz(1.0) == pytest.approx(MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA)


def test_gauss_tesla_scale() -> None:
    assert gauss_to_tesla(1.0) == pytest.approx(1.0e-4)


def test_convert_pivots_through_mhz() -> None:
    # 100 G expressed in MHz then Tesla must agree with the direct conversions.
    mhz = convert(100.0, FieldUnit.GAUSS, FieldUnit.MHZ)
    assert mhz == pytest.approx(gauss_to_mhz(100.0))
    tesla = convert(100.0, "gauss", "tesla")
    assert tesla == pytest.approx(100.0 * GAUSS_TO_TESLA)
    # Identity and full round trip.
    assert convert(5.0, "mhz", "mhz") == pytest.approx(5.0)
    assert convert(convert(5.0, "mhz", "gauss"), "gauss", "mhz") == pytest.approx(5.0)


def test_field_unit_coerce_and_labels() -> None:
    assert FieldUnit.coerce("GAUSS") == FieldUnit.GAUSS
    assert FieldUnit.coerce("nonsense") == FieldUnit.MHZ
    assert FieldUnit.coerce("nonsense", FieldUnit.TESLA) == FieldUnit.TESLA
    assert axis_label("mhz") == "Frequency (MHz)"
    assert axis_label(FieldUnit.GAUSS) == "Field (G)"
    assert axis_label("tesla") == "Field (T)"


def test_frequency_resolution() -> None:
    # 1/(2·Δt·N): WiMDA's fres for the spectrum grid.
    assert frequency_resolution_mhz(0.016, 1024) == pytest.approx(1.0 / (2.0 * 0.016 * 1024))
    assert frequency_resolution_mhz(0.0, 1024) == float("inf")


def test_converters_are_array_friendly() -> None:
    arr = np.linspace(0.0, 10.0, 5)
    out = mhz_to_gauss(arr)
    assert out.shape == arr.shape
    np.testing.assert_allclose(gauss_to_mhz(out), arr, atol=1e-12)


@pytest.mark.parametrize("bad_ratio", [0.0, -1.0, float("nan"), float("inf")])
def test_gyromagnetic_ratio_override_rejects_non_positive_or_non_finite(bad_ratio: float) -> None:
    # A bad probe constant on the override path would yield inf/sign-flipped
    # fields; the converters fail fast instead.
    for converter in (mhz_to_gauss, gauss_to_mhz, mhz_to_tesla, tesla_to_mhz):
        with pytest.raises(ValueError, match="positive and finite"):
            converter(1.0, gyromagnetic_ratio_mhz_per_t=bad_ratio)
    with pytest.raises(ValueError, match="positive and finite"):
        convert(1.0, "mhz", "gauss", gyromagnetic_ratio_mhz_per_t=bad_ratio)

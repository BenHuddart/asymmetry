"""Asymmetry percent/fraction scale convention: accessors, primitive, and guard.

These protect the documented contract that a loaded ``MuonDataset`` exposes
asymmetry on the **percent** scale while the low-level
:func:`compute_asymmetry` primitive returns the dimensionless **fraction** — and
that seeding a fit with the wrong scale is surfaced loudly rather than silently
converging to the wrong minimum.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting import (
    AsymmetryScaleWarning,
    FitEngine,
    Parameter,
    ParameterSet,
)
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.models import MODELS
from asymmetry.core.transform import compute_asymmetry

pytestmark = [pytest.mark.unit]


def _percent_dataset() -> MuonDataset:
    """A synthetic percent-scale dataset: a 20 % exponential relaxation."""
    time = np.linspace(0.0, 8.0, 64)
    asymmetry = 20.0 * np.exp(-0.3 * time)  # percent scale (peak ~20)
    error = np.full_like(asymmetry, 0.5)
    return MuonDataset(time=time, asymmetry=asymmetry, error=error)


def test_dataset_scale_accessors_are_mutually_consistent() -> None:
    ds = MuonDataset(
        time=np.array([0.0, 1.0, 2.0]),
        asymmetry=np.array([16.0, -8.0, 4.0]),  # stored percent
        error=np.array([1.0, 1.0, 1.0]),
    )

    # percent == stored; fraction == percent / 100.
    np.testing.assert_allclose(ds.asymmetry_percent, ds.asymmetry)
    np.testing.assert_allclose(ds.asymmetry_fraction, ds.asymmetry / 100.0)
    np.testing.assert_allclose(ds.error_percent, ds.error)
    np.testing.assert_allclose(ds.error_fraction, ds.error / 100.0)
    # Round-trips: fraction * 100 == percent.
    np.testing.assert_allclose(ds.asymmetry_fraction * 100.0, ds.asymmetry_percent)
    np.testing.assert_allclose(ds.error_fraction * 100.0, ds.error_percent)


def test_compute_asymmetry_returns_fraction() -> None:
    # The primitive returns the dimensionless A in [-1, 1], NOT percent.
    forward = np.array([100.0, 100.0])
    backward = np.array([80.0, 60.0])
    asym, _err = compute_asymmetry(forward, backward, alpha=1.0)

    assert np.all(np.abs(asym) <= 1.0)
    np.testing.assert_allclose(asym, [(100 - 80) / 180, (100 - 60) / 160])


def test_fit_warns_on_fraction_seed_against_percent_data() -> None:
    ds = _percent_dataset()
    model_fn = MODELS["ExponentialRelaxation"].function

    fraction_seed = ParameterSet()
    fraction_seed.add(Parameter(name="A0", value=0.2))  # fraction-scale amplitude
    fraction_seed.add(Parameter(name="Lambda", value=0.3))
    fraction_seed.add(Parameter(name="baseline", value=0.0))

    with pytest.warns(AsymmetryScaleWarning):
        FitEngine().fit(ds, model_fn, fraction_seed)


def test_fit_warns_on_low_asymmetry_percent_data_with_fraction_seed() -> None:
    # Regression: a small (but percent-scale, peak > 1) asymmetry seeded with a
    # fraction-scale amplitude must still warn — the trap is the scale crossing,
    # not the absolute size ratio.
    time = np.linspace(0.0, 8.0, 64)
    asymmetry = 4.0 * np.exp(-0.3 * time)  # percent, peak ~4
    ds = MuonDataset(time=time, asymmetry=asymmetry, error=np.full_like(asymmetry, 0.2))
    model_fn = MODELS["ExponentialRelaxation"].function

    fraction_seed = ParameterSet()
    fraction_seed.add(Parameter(name="A0", value=0.2))
    fraction_seed.add(Parameter(name="Lambda", value=0.3))
    fraction_seed.add(Parameter(name="baseline", value=0.0))

    with pytest.warns(AsymmetryScaleWarning):
        FitEngine().fit(ds, model_fn, fraction_seed)


def test_fit_does_not_warn_on_consistent_fraction_scale() -> None:
    # Both data and seed on the fraction scale (≤ 1.5): no boundary crossing,
    # so no warning even though magnitudes are small.
    time = np.linspace(0.0, 8.0, 64)
    asymmetry = 0.2 * np.exp(-0.3 * time)  # fraction, peak ~0.2
    ds = MuonDataset(time=time, asymmetry=asymmetry, error=np.full_like(asymmetry, 0.005))
    model_fn = MODELS["ExponentialRelaxation"].function

    fraction_seed = ParameterSet()
    fraction_seed.add(Parameter(name="A0", value=0.2))
    fraction_seed.add(Parameter(name="Lambda", value=0.3))
    fraction_seed.add(Parameter(name="baseline", value=0.0))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        FitEngine().fit(ds, model_fn, fraction_seed)

    scale_warnings = [w for w in caught if issubclass(w.category, AsymmetryScaleWarning)]
    assert not scale_warnings


def test_fit_does_not_warn_on_matching_percent_seed() -> None:
    ds = _percent_dataset()
    model_fn = MODELS["ExponentialRelaxation"].function

    percent_seed = ParameterSet()
    percent_seed.add(Parameter(name="A0", value=20.0))  # percent-scale amplitude
    percent_seed.add(Parameter(name="Lambda", value=0.3))
    percent_seed.add(Parameter(name="baseline", value=0.0))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        FitEngine().fit(ds, model_fn, percent_seed)

    scale_warnings = [w for w in caught if issubclass(w.category, AsymmetryScaleWarning)]
    assert not scale_warnings


def _scale_warnings_for(model_fn, seed: ParameterSet) -> list[warnings.WarningMessage]:
    """The scale warnings a fit of *seed* against a percent record with a ~1 % tail raises."""
    time = np.linspace(0.1, 10.0, 200)
    asymmetry = 31.0 * np.exp(-0.5 * time) + 1.0
    ds = MuonDataset(time=time, asymmetry=asymmetry, error=np.full_like(asymmetry, 0.3))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        FitEngine().fit(ds, model_fn, seed)
    return [w for w in caught if issubclass(w.category, AsymmetryScaleWarning)]


def test_fit_does_not_warn_on_constant_seeded_at_a_percent_records_tail() -> None:
    # The fit wizard's constant-only null baseline seeds A_bg at the record's
    # tail, ~1 on this percent record: a flat curve has no amplitude whose scale
    # could be confused with the data's.
    seed = ParameterSet()
    seed.add(Parameter(name="A_bg", value=1.03))

    assert not _scale_warnings_for(CompositeModel.from_expression("Constant").function, seed)


def test_fit_does_not_warn_on_term_decayed_before_the_first_bin() -> None:
    # A resolution probe pins a rate at its fast edge: the term has decayed
    # before the first bin, so the seeded curve is the background alone.
    seed = ParameterSet()
    seed.add(Parameter(name="A0", value=30.0))
    seed.add(Parameter(name="Lambda", value=1.0e3, fixed=True))
    seed.add(Parameter(name="baseline", value=1.16))

    assert not _scale_warnings_for(MODELS["ExponentialRelaxation"].function, seed)


def test_fit_warns_on_fraction_amplitude_with_tail_background_against_percent_data() -> None:
    # The flat-curve exemption must not hide the trap itself: a fraction-scale
    # amplitude still varies across the window.
    seed = ParameterSet()
    seed.add(Parameter(name="A0", value=0.3))
    seed.add(Parameter(name="Lambda", value=0.5))
    seed.add(Parameter(name="baseline", value=1.03))

    assert _scale_warnings_for(MODELS["ExponentialRelaxation"].function, seed)

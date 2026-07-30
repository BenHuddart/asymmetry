"""The asymmetry fraction/percent convention, pinned numerically.

Every public asymmetry-valued surface is exercised on one tiny synthetic
forward/backward pair whose asymmetry is exact:

    F = 120, B = 80, alpha = beta = 1  ->  A = (120 - 80) / (120 + 80) = 0.2

so a **fraction**-scale API must return exactly ``0.2`` and a **percent**-scale
one exactly ``20.0``. These assertions are the contract that stops silent unit
drift: a function that changes scale fails here rather than a factor of 100
later, in a fit amplitude. All fixtures are synthetic with invented metadata.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.fitting.engine import FitEngine
from asymmetry.core.fitting.models import MODELS
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.transform.asymmetry import (
    compute_asymmetry,
    compute_asymmetry_with_count_errors,
)
from asymmetry.core.transform.integral import build_field_scan, integrate_asymmetry, integrate_run
from asymmetry.core.transform.rebin import binned_fb_asymmetry, rebin
from asymmetry.core.transform.reduce import reduce_grouped_asymmetry
from asymmetry.core.transform.rrf import rrf_demodulate
from asymmetry.core.transform.units import (
    ASYMMETRY_FRACTION,
    ASYMMETRY_PERCENT,
    PERCENT_PER_FRACTION,
    AsymmetryUnit,
    convert_asymmetry,
    to_fraction,
    to_percent,
)

#: The exact asymmetry of the fixture below, as a fraction and in percent.
EXPECTED_FRACTION = 0.2
EXPECTED_PERCENT = 20.0

_FORWARD_COUNTS = 120.0
_BACKWARD_COUNTS = 80.0
_N_BINS = 64
_BIN_WIDTH = 0.05


def _counts() -> tuple[np.ndarray, np.ndarray]:
    """Flat forward/backward counts with the exact asymmetry ``0.2``."""
    forward = np.full(_N_BINS, _FORWARD_COUNTS)
    backward = np.full(_N_BINS, _BACKWARD_COUNTS)
    return forward, backward


def _run() -> Run:
    forward, backward = _counts()
    histograms = [
        Histogram(
            counts=arr,
            bin_width=_BIN_WIDTH,
            t0_bin=0,
            good_bin_start=0,
            good_bin_end=_N_BINS - 1,
        )
        for arr in (forward, backward)
    ]
    return Run(
        run_number=1,
        histograms=histograms,
        metadata={"instrument": "TESTINST", "field": 100.0, "temperature": 10.0},
        grouping=_grouping(),
    )


def _grouping() -> dict:
    return {
        "groups": {1: [1], 2: [2]},
        "forward_group": 1,
        "backward_group": 2,
        "first_good_bin": 0,
        "last_good_bin": _N_BINS - 1,
        "alpha": 1.0,
    }


# --- the fraction side -----------------------------------------------------


def test_compute_asymmetry_returns_a_fraction() -> None:
    forward, backward = _counts()
    asym, _ = compute_asymmetry(forward, backward)
    assert asym == pytest.approx(EXPECTED_FRACTION)
    assert np.max(np.abs(asym)) <= 1.0


def test_compute_asymmetry_with_count_errors_returns_a_fraction() -> None:
    forward, backward = _counts()
    asym, _ = compute_asymmetry_with_count_errors(
        forward, backward, np.sqrt(forward), np.sqrt(backward)
    )
    assert asym == pytest.approx(EXPECTED_FRACTION)


def test_binned_fb_asymmetry_returns_a_fraction() -> None:
    forward, backward = _counts()
    _, asym, _ = binned_fb_asymmetry(
        forward,
        backward,
        grouping=_grouping(),
        common_t0=0,
        bin_width_us=_BIN_WIDTH,
        alpha=1.0,
        first_good_bin=0,
        last_good_bin=_N_BINS - 1,
    )
    assert asym == pytest.approx(EXPECTED_FRACTION)


@pytest.mark.parametrize("method", ["integral", "differential"])
def test_integrate_asymmetry_returns_a_fraction(method: str) -> None:
    forward, backward = _counts()
    value, _ = integrate_asymmetry(forward, backward, method=method)
    assert value == pytest.approx(EXPECTED_FRACTION)


def test_integrate_run_returns_a_fraction() -> None:
    value, _ = integrate_run(_run())
    assert value == pytest.approx(EXPECTED_FRACTION)


def test_build_field_scan_is_fraction_scale_and_says_so() -> None:
    scan = build_field_scan([_run()], order_key="field")
    assert scan.units is ASYMMETRY_FRACTION
    assert scan.value == pytest.approx(EXPECTED_FRACTION)


# --- the percent side ------------------------------------------------------


def test_reduce_grouped_asymmetry_is_percent_scale_and_says_so() -> None:
    run = _run()
    reduction = reduce_grouped_asymmetry(
        histograms=run.histograms,
        grouping=_grouping(),
        forward_idx=[0],
        backward_idx=[1],
        alpha=1.0,
        use_deadtime=False,
        deadtime_mode="off",
        use_background=False,
    )
    assert reduction.units is ASYMMETRY_PERCENT
    assert reduction.asymmetry == pytest.approx(EXPECTED_PERCENT)


def test_the_reduction_is_exactly_one_hundred_times_the_primitive() -> None:
    """The whole trap, stated as an equation on one input."""
    run = _run()
    forward, backward = _counts()
    _, fraction_asym, fraction_err = binned_fb_asymmetry(
        forward,
        backward,
        grouping=_grouping(),
        common_t0=0,
        bin_width_us=_BIN_WIDTH,
        alpha=1.0,
        first_good_bin=0,
        last_good_bin=_N_BINS - 1,
    )
    reduction = reduce_grouped_asymmetry(
        histograms=run.histograms,
        grouping=_grouping(),
        forward_idx=[0],
        backward_idx=[1],
        alpha=1.0,
        use_deadtime=False,
        deadtime_mode="off",
        use_background=False,
    )
    assert reduction.asymmetry == pytest.approx(fraction_asym * PERCENT_PER_FRACTION)
    assert reduction.error == pytest.approx(fraction_err * PERCENT_PER_FRACTION)


def test_muon_dataset_accessors_name_both_scales() -> None:
    time = np.linspace(0.0, 1.0, 8)
    ds = MuonDataset(
        time=time,
        asymmetry=np.full(8, EXPECTED_PERCENT),
        error=np.full(8, 0.5),
        metadata={"run_number": 1},
    )
    assert ds.asymmetry_percent == pytest.approx(EXPECTED_PERCENT)
    assert ds.asymmetry_fraction == pytest.approx(EXPECTED_FRACTION)
    assert ds.error_percent == pytest.approx(0.5)
    assert ds.error_fraction == pytest.approx(0.005)


def test_builtin_model_amplitude_seeds_are_percent_scale() -> None:
    """A0 = 25 (not 0.25) is the percent convention, pinned at the registry."""
    for name in ("ExponentialRelaxation", "GaussianRelaxation", "Oscillatory"):
        defaults = MODELS[name].param_defaults
        assert defaults["A0"] > 1.5, f"{name} A0 seed is not on the percent scale"


# --- scale-preserving transforms -------------------------------------------


def test_rebin_preserves_whichever_scale_it_is_given() -> None:
    time = np.arange(8, dtype=float)
    fraction = np.full(8, EXPECTED_FRACTION)
    errors = np.full(8, 0.01)
    _, rebinned_fraction, _ = rebin(time, fraction, errors, 2)
    _, rebinned_percent, _ = rebin(time, fraction * PERCENT_PER_FRACTION, errors, 2)
    assert rebinned_fraction == pytest.approx(EXPECTED_FRACTION)
    assert rebinned_percent == pytest.approx(rebinned_fraction * PERCENT_PER_FRACTION)


def test_rrf_demodulation_preserves_scale_exactly() -> None:
    time = np.arange(256) * 0.01
    fraction = 0.2 * np.cos(2.0 * np.pi * 5.0 * time)
    errors = np.full_like(time, 0.002)
    frac_curve = rrf_demodulate(time, fraction, errors, frequency_mhz=5.0)
    pct_curve = rrf_demodulate(
        time,
        fraction * PERCENT_PER_FRACTION,
        errors * PERCENT_PER_FRACTION,
        frequency_mhz=5.0,
    )
    assert pct_curve.real == pytest.approx(frac_curve.real * PERCENT_PER_FRACTION)
    assert pct_curve.magnitude == pytest.approx(frac_curve.magnitude * PERCENT_PER_FRACTION)
    assert pct_curve.real_error == pytest.approx(frac_curve.real_error * PERCENT_PER_FRACTION)


def test_the_fit_engine_reports_amplitudes_on_the_data_scale() -> None:
    """The engine never rescales: percent in, percent out; fraction in, fraction out."""
    time = np.linspace(0.05, 6.0, 120)
    percent_curve = EXPECTED_PERCENT * np.exp(-0.3 * time)
    percent_error = np.full_like(time, 0.2)
    model = MODELS["ExponentialRelaxation"].function

    def _params(scale: float) -> ParameterSet:
        params = ParameterSet()
        params.add(Parameter(name="A0", value=15.0 * scale, min=0.0))
        params.add(Parameter(name="Lambda", value=0.3, min=0.0))
        params.add(Parameter(name="baseline", value=0.0))
        return params

    engine = FitEngine()
    in_percent = engine.fit_arrays(time, percent_curve, percent_error, model, _params(1.0))
    in_fraction = engine.fit_arrays(
        time,
        to_fraction(percent_curve),
        to_fraction(percent_error),
        model,
        _params(1.0 / PERCENT_PER_FRACTION),
    )
    assert in_percent.success and in_fraction.success
    # Tolerances are loose (Migrad's convergence, not the scale, is the noise
    # floor here); the assertion that matters is the factor of 100 between them.
    assert in_percent.parameters["A0"].value == pytest.approx(EXPECTED_PERCENT, rel=1e-4)
    assert in_fraction.parameters["A0"].value == pytest.approx(EXPECTED_FRACTION, rel=1e-4)
    assert in_percent.parameters["A0"].value == pytest.approx(
        in_fraction.parameters["A0"].value * PERCENT_PER_FRACTION, rel=1e-4
    )
    # Same relaxation rate either way — only the amplitude carries the scale.
    assert in_fraction.parameters["Lambda"].value == pytest.approx(
        in_percent.parameters["Lambda"].value, rel=1e-4
    )


# --- the units mechanism itself --------------------------------------------


def test_the_enum_has_exactly_the_two_real_conventions() -> None:
    assert set(AsymmetryUnit) == {ASYMMETRY_FRACTION, ASYMMETRY_PERCENT}
    assert ASYMMETRY_FRACTION is AsymmetryUnit.FRACTION
    assert ASYMMETRY_PERCENT is AsymmetryUnit.PERCENT
    assert ASYMMETRY_PERCENT.label == "%"
    assert ASYMMETRY_FRACTION.label == ""


def test_scale_to_is_the_factor_of_one_hundred() -> None:
    assert ASYMMETRY_FRACTION.scale_to(ASYMMETRY_PERCENT) == PERCENT_PER_FRACTION
    assert ASYMMETRY_PERCENT.scale_to(ASYMMETRY_FRACTION) == pytest.approx(0.01)
    assert ASYMMETRY_FRACTION.scale_to(ASYMMETRY_FRACTION) == 1.0
    assert ASYMMETRY_PERCENT.scale_to(ASYMMETRY_PERCENT) == 1.0


def test_conversions_round_trip_and_never_alias_the_input() -> None:
    values = np.array([EXPECTED_FRACTION, -EXPECTED_FRACTION, 0.0])
    as_percent = to_percent(values)
    assert as_percent == pytest.approx([EXPECTED_PERCENT, -EXPECTED_PERCENT, 0.0])
    assert to_fraction(as_percent) == pytest.approx(values)
    # A same-unit conversion still copies, so callers cannot mutate the source.
    same = convert_asymmetry(values, ASYMMETRY_FRACTION, ASYMMETRY_FRACTION)
    assert same is not values
    same[0] = 99.0
    assert values[0] == pytest.approx(EXPECTED_FRACTION)


def test_conversion_accepts_an_explicit_source_unit() -> None:
    assert to_percent(EXPECTED_PERCENT, frm=ASYMMETRY_PERCENT) == pytest.approx(EXPECTED_PERCENT)
    assert to_fraction(EXPECTED_FRACTION, frm=ASYMMETRY_FRACTION) == pytest.approx(
        EXPECTED_FRACTION
    )

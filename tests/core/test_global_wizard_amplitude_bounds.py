"""Series amplitude defaults share a scale envelope without changing authored limits."""

import pytest

from asymmetry.core.fitting.component_tags import ParameterKind
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.fit_wizard import CandidateTemplate
from asymmetry.core.fitting.global_fit_wizard import (
    _configured_single_fit_parameter_set,
    _initial_parameter_sets_for_candidate,
    fingerprint_spectrum,
)
from tests.core.sharing_series import glassy_series


@pytest.fixture(scope="module")
def glassy_context():
    series = glassy_series(runs_with_lost_asymmetry=3)
    fingerprints = {int(ds.run_number): fingerprint_spectrum(ds) for ds in series.datasets}
    template = CandidateTemplate("stretched_constant", "Stretched", "General", "test", series.model)
    return series.datasets, fingerprints, template


def test_series_envelope_contains_an_amplitude_excluded_by_a_weak_member(glassy_context):
    datasets, fingerprints, template = glassy_context
    defaults = {
        int(ds.run_number): _configured_single_fit_parameter_set(
            ds,
            fingerprints[int(ds.run_number)],
            template,
            current_values={},
            parameter_bounds={},
            fixed_param_names=(),
        )
        for ds in datasets
    }
    result = _initial_parameter_sets_for_candidate(
        datasets,
        fingerprints,
        template,
        current_values={},
        parameter_bounds={},
        fixed_param_names=(),
    )
    # A flat slow trace's span cannot bound its extrapolated zero-time amplitude.
    assert defaults[10]["A_1"].max < 20.0
    envelope = max(parameters["A_1"].max for parameters in defaults.values())
    assert envelope > 20.0
    for run, parameters in result.items():
        assert parameters["A_1"].max == envelope
        for before, after in zip(defaults[run], parameters, strict=True):
            assert (after.name, after.value, after.min, after.fixed) == (
                before.name,
                before.value,
                before.min,
                before.fixed,
            )
            if before.name != "A_1":
                assert after.max == before.max


@pytest.mark.parametrize("fixed", [False, True])
def test_authored_amplitude_bounds_and_fixed_values_keep_their_authority(glassy_context, fixed):
    datasets, fingerprints, template = glassy_context
    options = dict(
        current_values={"A_1": 12.3} if fixed else {},
        parameter_bounds={} if fixed else {"A_1": (2.0, 19.5)},
        fixed_param_names=("A_1",) if fixed else (),
    )
    result = _initial_parameter_sets_for_candidate(datasets, fingerprints, template, **options)
    for ds in datasets:
        original = _configured_single_fit_parameter_set(
            ds,
            fingerprints[int(ds.run_number)],
            template,
            **options,
        )
        assert list(result[int(ds.run_number)]) == list(original)


def test_authored_fraction_limits_remain_bounded_under_a_group_amplitude(glassy_context):
    datasets, fingerprints, _ = glassy_context
    model = CompositeModel.from_expression("(Exponential + Exponential){frac} + Constant")
    template = CandidateTemplate("fraction_context", "Fraction", "General", "test", model)
    fractions = [
        name for name, kind in model.parameter_kinds().items() if kind is ParameterKind.FRACTION
    ]
    assert fractions
    bounds = {name: (0.2, 0.8) for name in fractions}
    result = _initial_parameter_sets_for_candidate(
        datasets,
        fingerprints,
        template,
        current_values={},
        parameter_bounds=bounds,
        fixed_param_names=(),
    )
    for parameters in result.values():
        for name in fractions:
            assert (parameters[name].min, parameters[name].max) == bounds[name]

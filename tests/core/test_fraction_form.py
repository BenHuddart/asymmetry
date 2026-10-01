"""A model's signal terms as one fraction group (plan D11, D16)."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.fraction_form import signal_fraction_form


@pytest.mark.parametrize(
    ("expression", "grouped", "amplitudes", "total", "fractions"),
    [
        (
            "Exponential + Gaussian + Constant",
            "(Exponential + Gaussian){frac} + Constant",
            ("A_1", "A_2"),
            "A_1",
            ("f_Exponential",),
        ),
        (
            "Oscillatory * Gaussian + Oscillatory * Gaussian + Constant",
            "(Oscillatory * Gaussian + Oscillatory * Gaussian){frac} + Constant",
            ("A_1", "A_3"),
            "A_1",
            ("f_Oscillatory",),
        ),
        (
            "Exponential + Gaussian + Exponential",
            "(Exponential + Gaussian + Exponential){frac}",
            ("A_1", "A_2", "A_3"),
            "A_1",
            ("f_Exponential", "f_Gaussian"),
        ),
        (
            "Constant + Exponential + Gaussian",
            "Constant + (Exponential + Gaussian){frac}",
            ("A_2", "A_3"),
            "A_2",
            ("f_Exponential",),
        ),
        (
            "(Oscillatory * Exponential) + Exponential - Constant",
            "((Oscillatory * Exponential) + Exponential){frac} - Constant",
            ("A_1", "A_3"),
            "A_1",
            ("f_Oscillatory",),
        ),
    ],
)
def test_signal_terms_are_grouped_and_the_background_is_left_outside(
    expression: str,
    grouped: str,
    amplitudes: tuple[str, ...],
    total: str,
    fractions: tuple[str, ...],
) -> None:
    model = CompositeModel.from_expression(expression)
    form = signal_fraction_form(model)

    assert form.grouped.component_expression_string() == grouped
    assert (form.amplitudes, form.total, form.fractions) == (amplitudes, total, fractions)
    # n amplitudes become one total and n − 1 fractions: the count is unchanged.
    assert len(form.grouped.param_names) == len(model.param_names)
    assert set(form.grouped.param_names) == {total, *fractions, *form.carried}
    assert set(model.param_names) == {*amplitudes, *form.carried.values()}


def test_values_map_between_the_two_forms_and_back() -> None:
    model = CompositeModel.from_expression(
        "Oscillatory * Gaussian + Oscillatory * Gaussian + Exponential + Constant"
    )
    form = signal_fraction_form(model)
    plain = {
        "A_1": 6.0,
        "frequency_1": 2.0,
        "phase_1": 0.3,
        "sigma_2": 0.8,
        "A_3": 10.0,
        "frequency_3": 2.4,
        "phase_3": 0.1,
        "sigma_4": 0.2,
        "A_5": 4.0,
        "Lambda": 0.5,
        "A_bg": 1.5,
    }
    assert set(plain) == set(model.param_names)

    grouped = form.grouped_values(plain)

    assert grouped[form.total] == 20.0
    assert [grouped[name] for name in form.fractions] == [0.3, 0.5]
    assert grouped["sigma_1"] == 0.8 and grouped["sigma_2"] == 0.2
    assert form.plain_values(grouped) == pytest.approx(plain)
    time = np.linspace(0.0, 8.0, 200)
    assert np.allclose(model.function(time, **plain), form.grouped.function(time, **grouped))


def test_amplitudes_summing_to_zero_have_no_fractions() -> None:
    form = signal_fraction_form(CompositeModel.from_expression("Exponential + Gaussian"))

    with pytest.raises(ValueError, match="summing to zero"):
        form.grouped_values({"A_1": 3.0, "Lambda": 1.0, "A_2": -3.0, "sigma": 1.0})


@pytest.mark.parametrize(
    "expression",
    [
        "Exponential + Constant",  # one signal amplitude
        "Oscillatory * Exponential",  # not a sum
        "(Exponential + Gaussian){frac} + Constant",  # already grouped
        "Exponential - Gaussian + Constant",  # a subtracted signal term
        "Exponential + Constant + Gaussian",  # the signal terms do not stand side by side
        "Oscillatory * (Exponential + Gaussian) + Exponential + Constant",  # a sum inside a term
        "GaussianPeak + LorentzianPeak + ConstantBackground",  # heights are not group scales
    ],
)
def test_model_without_two_signal_terms_to_group_has_no_fraction_form(expression: str) -> None:
    assert signal_fraction_form(CompositeModel.from_expression(expression)) is None

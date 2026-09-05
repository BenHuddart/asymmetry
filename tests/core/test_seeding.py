"""The one seeding function: its layers, their order, and what is run-bound."""

from __future__ import annotations

import math

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.fit_wizard import fingerprint_spectrum
from asymmetry.core.fitting.parameter_models import ParameterCompositeModel
from asymmetry.core.fitting.seeding import (
    Seed,
    SeedContext,
    record_scale_estimate,
    seed_parameters,
    seed_trend_parameters,
)
from asymmetry.core.fitting.spectral import default_frequency_model

# ── models and data ─────────────────────────────────────────────────────────


def _exponential_model() -> CompositeModel:
    return CompositeModel(["Exponential", "Constant"], operators=["+"])


def _damped_oscillation_model() -> CompositeModel:
    return CompositeModel(["Oscillatory", "Exponential", "Constant"], operators=["*", "+"])


def _lf_kt_model() -> CompositeModel:
    return CompositeModel(["LongitudinalFieldKT", "Constant"], operators=["+"])


def _decaying_record(*, amplitude: float = 20.0, tail: float = 3.0, n: int = 400) -> MuonDataset:
    """A clean exponential decay from ``tail + amplitude`` down to ``tail``."""
    time = np.linspace(0.0, 10.0, n)
    asymmetry = tail + amplitude * np.exp(-time / 0.8)
    return MuonDataset(
        time=time,
        asymmetry=asymmetry,
        error=np.full_like(time, 0.1),
        metadata={"run_number": 1},
    )


def _peak_spectrum(*, peak_freq: float = 30.0, baseline: float = 0.2) -> MuonDataset:
    frequency = np.linspace(0.0, 100.0, 2000)
    values = baseline + 10.0 * np.exp(-0.5 * ((frequency - peak_freq) / 1.5) ** 2)
    return MuonDataset(
        time=frequency,
        asymmetry=values,
        error=np.full_like(frequency, 0.05),
        metadata={"plot_domain": "frequency"},
    )


# ── invariants ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "model, context",
    [
        (_exponential_model(), SeedContext()),
        (_exponential_model(), SeedContext(dataset=_decaying_record(), field_gauss=150.0)),
        (_damped_oscillation_model(), SeedContext(dataset=_decaying_record())),
        (_lf_kt_model(), SeedContext(field_gauss=250.0, individual_groups=True)),
        (
            default_frequency_model(),
            SeedContext(dataset=_peak_spectrum(), domain="frequency"),
        ),
    ],
)
def test_every_seeded_name_is_a_model_parameter(
    model: CompositeModel, context: SeedContext
) -> None:
    seeds = seed_parameters(model, context)

    assert set(seeds) == set(model.param_names)


# ── layer 1: static component defaults ──────────────────────────────────────


def test_no_data_seeds_static_defaults_only() -> None:
    model = _exponential_model()

    seeds = seed_parameters(model, SeedContext())

    assert seeds == {
        "A_1": Seed(value=25.0, fixed=False, min=0.0, max=math.inf, run_bound=False),
        "Lambda": Seed(value=0.5, fixed=False, min=0.0, max=math.inf, run_bound=False),
        "A_bg": Seed(value=0.0, fixed=False, min=-math.inf, max=math.inf, run_bound=False),
    }


def test_static_layer_carries_default_bounds_and_fixed_by_default() -> None:
    model = CompositeModel(["VortexLattice"])

    seeds = seed_parameters(model, SeedContext())

    # ``field`` is declared fixed-by-default by the component itself.
    assert seeds["field"].fixed is True
    assert seeds["A_1"].fixed is False
    # ``default_min`` where the parameter declares one, -inf where it does not.
    assert seeds["A_1"].min == 0.0
    assert seeds["phase"].min == -math.inf
    assert all(seed.max == math.inf for seed in seeds.values())


# ── layer 2: record scale ───────────────────────────────────────────────────


def test_record_scale_seeds_amplitude_and_background_from_the_record() -> None:
    dataset = _decaying_record(amplitude=20.0, tail=3.0)
    model = _exponential_model()

    seeds = seed_parameters(model, SeedContext(dataset=dataset))

    amplitude, tail = record_scale_estimate(dataset.time, dataset.asymmetry)
    assert seeds["A_1"].value == pytest.approx(amplitude)
    assert seeds["A_bg"].value == pytest.approx(tail)
    # The early window sits inside the decay, so the amplitude reads a little
    # below the true 20 % and the tail a little above the true 3 %.
    assert seeds["A_1"].value == pytest.approx(20.0, rel=0.25)
    assert seeds["A_bg"].value == pytest.approx(3.0, rel=0.05)
    # A record-scale value describes the physics, not the run: it travels.
    assert seeds["A_1"].run_bound is False
    assert seeds["A_bg"].run_bound is False


def test_record_scale_leaves_rate_and_shape_parameters_alone() -> None:
    model = _damped_oscillation_model()

    seeds = seed_parameters(model, SeedContext(dataset=_decaying_record()))

    assert seeds["Lambda"].value == model.param_defaults["Lambda"]
    assert seeds["frequency"].value == model.param_defaults["frequency"]
    assert seeds["phase"].value == model.param_defaults["phase"]


def test_record_scale_does_not_sweep_in_physics_named_like_an_amplitude() -> None:
    """``A_hf`` is a hyperfine coupling, not the record's asymmetry amplitude."""
    model = CompositeModel(["MuoniumLFRelax"])

    seeds = seed_parameters(model, SeedContext(dataset=_decaying_record()))

    assert seeds["A_hf"].value == model.param_defaults["A_hf"]
    assert seeds["A_1"].value != model.param_defaults["A_1"]


def test_record_scale_is_time_domain_only() -> None:
    model = _exponential_model()
    dataset = _decaying_record()

    seeds = seed_parameters(model, SeedContext(dataset=dataset, domain="frequency"))

    assert seeds["A_1"].value == model.param_defaults["A_1"]


def test_record_scale_estimate_matches_the_wizard_fingerprint() -> None:
    dataset = _decaying_record(amplitude=12.5, tail=-1.5, n=317)

    amplitude, tail = record_scale_estimate(dataset.time, dataset.asymmetry)
    fingerprint = fingerprint_spectrum(dataset)

    assert amplitude == fingerprint.initial_amplitude_estimate
    assert tail == fingerprint.tail_estimate


# ── layer 3: applied field ──────────────────────────────────────────────────


def test_applied_field_seeds_field_and_b_l_and_marks_them_run_bound() -> None:
    model = _lf_kt_model()

    seeds = seed_parameters(model, SeedContext(field_gauss=250.0))

    assert seeds["B_L"].value == 250.0
    assert seeds["B_L"].run_bound is True
    # Nothing else moves, and nothing else is run-bound.
    assert seeds["Delta"].value == model.param_defaults["Delta"]
    assert [name for name, seed in seeds.items() if seed.run_bound] == ["B_L"]


def test_applied_field_seeds_every_field_parameter_of_a_multi_component_model() -> None:
    model = CompositeModel(["OscillatoryField", "OscillatoryField"], operators=["+"])

    seeds = seed_parameters(model, SeedContext(field_gauss=125.0))

    field_names = [name for name in model.param_names if name.startswith("field")]
    assert len(field_names) == 2
    assert all(seeds[name].value == 125.0 for name in field_names)
    assert all(seeds[name].run_bound for name in field_names)


@pytest.mark.parametrize("field_gauss", [None, 0.0])
def test_no_applied_field_keeps_the_component_default(field_gauss: float | None) -> None:
    model = CompositeModel(["OscillatoryField"])

    seeds = seed_parameters(model, SeedContext(field_gauss=field_gauss))

    assert seeds["field"].value == model.param_defaults["field"]
    assert seeds["field"].run_bound is False


def test_applied_field_seeds_the_value_of_a_fixed_by_default_parameter() -> None:
    """The field layer overrides the value; the static Fix state stands."""
    model = CompositeModel(["VortexLattice"])

    seeds = seed_parameters(model, SeedContext(field_gauss=310.0))

    assert seeds["field"].value == 310.0
    assert seeds["field"].fixed is True


# ── layer 4: frequency-domain peaks ─────────────────────────────────────────


def test_frequency_peaks_seed_from_the_spectrum_and_are_run_bound() -> None:
    model = default_frequency_model()

    seeds = seed_parameters(
        model, SeedContext(dataset=_peak_spectrum(peak_freq=30.0), domain="frequency")
    )

    assert seeds["nu0"].value == pytest.approx(30.0, abs=1.0)
    assert seeds["height"].value == pytest.approx(10.0, rel=0.1)
    assert all(seeds[name].run_bound for name in ("height", "nu0", "fwhm", "bg"))


def test_frequency_peaks_seed_a_batch_from_the_mean_across_its_datasets() -> None:
    model = default_frequency_model()
    datasets = (_peak_spectrum(peak_freq=20.0), _peak_spectrum(peak_freq=40.0))

    seeds = seed_parameters(model, SeedContext(datasets=datasets, domain="frequency"))

    assert seeds["nu0"].value == pytest.approx(30.0, abs=1.0)
    assert seeds["nu0"].run_bound is True


def test_frequency_peaks_are_not_seeded_in_the_time_domain() -> None:
    model = default_frequency_model()

    seeds = seed_parameters(model, SeedContext(dataset=_peak_spectrum(), domain="time"))

    assert seeds["nu0"].value == model.param_defaults["nu0"]
    assert not any(seed.run_bound for seed in seeds.values())


# ── layer 5: individual-groups overrides ────────────────────────────────────


def test_individual_groups_holds_background_and_phase_at_zero() -> None:
    model = _damped_oscillation_model()

    seeds = seed_parameters(model, SeedContext(dataset=_decaying_record(), individual_groups=True))

    for name in ("A_bg", "phase"):
        assert seeds[name].value == 0.0
        assert seeds[name].fixed is True
    # The amplitude still takes its record-scale seed and stays free.
    assert seeds["A_1"].value != model.param_defaults["A_1"]
    assert seeds["A_1"].fixed is False


def test_individual_groups_overrides_the_record_scale_background() -> None:
    dataset = _decaying_record(tail=7.0)
    model = _exponential_model()

    grouped = seed_parameters(model, SeedContext(dataset=dataset, individual_groups=True))
    single = seed_parameters(model, SeedContext(dataset=dataset))

    assert single["A_bg"].value == pytest.approx(7.0, rel=0.05)
    assert grouped["A_bg"].value == 0.0


# ── layer order ─────────────────────────────────────────────────────────────


def test_field_layer_overrides_the_static_default_which_survives_where_it_does_not_apply() -> None:
    model = _lf_kt_model()
    dataset = _decaying_record(amplitude=18.0, tail=2.0)

    seeds = seed_parameters(model, SeedContext(dataset=dataset, field_gauss=90.0))

    assert seeds["A_1"].value == pytest.approx(18.0, rel=0.3)  # layer 2 over layer 1
    assert seeds["B_L"].value == 90.0  # layer 3 over layer 1
    assert seeds["Delta"].value == model.param_defaults["Delta"]  # layer 1 stands


def test_individual_groups_is_the_last_word_on_phase() -> None:
    """Layer 5 wins over the applied-field and record-scale layers before it."""
    model = CompositeModel(["OscillatoryField", "Constant"], operators=["+"])
    dataset = _decaying_record(tail=4.0)

    seeds = seed_parameters(
        model,
        SeedContext(dataset=dataset, field_gauss=200.0, individual_groups=True),
    )

    assert seeds["phase"] == Seed(value=0.0, fixed=True, min=-math.inf, max=math.inf)
    assert seeds["A_bg"].value == 0.0
    assert seeds["A_bg"].fixed is True
    # The field is neither background nor phase, so it keeps its run-bound seed.
    assert seeds["field"].value == 200.0
    assert seeds["field"].run_bound is True


# ── trend seeds ─────────────────────────────────────────────────────────────


def test_trend_seeds_place_tc_beyond_the_fitted_range() -> None:
    model = ParameterCompositeModel(["OrderParameter"])
    x = np.linspace(5.0, 60.0, 25)
    y = np.linspace(10.0, 0.5, 25)

    seeds = seed_trend_parameters(model, x, y)

    assert set(seeds) == set(model.param_names)
    assert seeds["Tc"].value > 60.0
    assert seeds["y0"].value == pytest.approx(10.0)
    # Exponents keep their physical defaults.
    assert seeds["beta"].value == model.param_defaults["beta"]
    # Nothing about a trend describes one run.
    assert not any(seed.run_bound for seed in seeds.values())


def test_trend_seeds_place_a_plain_component_on_the_series_scale() -> None:
    """Without a critical-temperature component the series' own scale seeds it."""
    model = ParameterCompositeModel(["Linear"])
    x = np.linspace(0.0, 10.0, 11)
    y = np.linspace(0.0, 5.0, 11)

    seeds = seed_trend_parameters(model, x, y)

    # Intercept at the mean of y, slope at its span — the reading the trend
    # dialog used to make inline, now the layer beneath suggest_trend_seeds.
    assert seeds["b"].value == pytest.approx(2.5)
    assert seeds["m"].value == pytest.approx(5.0)


def test_trend_seeds_keep_the_component_default_for_a_flat_series() -> None:
    """A span of zero is no scale at all, so the component's own default stands."""
    model = ParameterCompositeModel(["Linear"])
    x = np.linspace(0.0, 10.0, 11)
    y = np.full(11, 3.0)

    seeds = seed_trend_parameters(model, x, y)

    assert seeds["m"].value == model.param_defaults["m"]
    assert seeds["b"].value == pytest.approx(3.0)


def test_trend_seeds_put_a_characteristic_x_at_half_the_fitted_range() -> None:
    """``B0``/``tau``/``nu`` are x-like: they start at half the x-range."""
    model = ParameterCompositeModel(["ExponentialDecay"])
    x = np.linspace(0.0, 40.0, 21)
    y = 10.0 * np.exp(-x / 12.0)

    seeds = seed_trend_parameters(model, x, y)

    assert "tau" in seeds
    assert seeds["tau"].value == pytest.approx(20.0)


def test_trend_seeds_hold_the_shape_factor_switch() -> None:
    model = ParameterCompositeModel(["SC_PWaveAxial"])
    x = np.linspace(1.0, 15.0, 20)
    y = np.linspace(1.0, 0.0, 20)

    seeds = seed_trend_parameters(model, x, y)

    assert seeds["shape_factor_a"].value == 0.0
    assert seeds["shape_factor_a"].fixed is True
    assert seeds["sigma_0"].fixed is False

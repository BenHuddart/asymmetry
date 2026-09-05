"""Applied-field seeding policy in the fit wizard.

``field`` (muonium/vortex) and ``B_L`` (the longitudinal component carried by
the Kubo-Toyabe family) are *recorded* quantities, not fitted ones. These tests
pin the rules in ``_pinned_longitudinal_field`` /
``_free_longitudinal_field_seed`` and the end-to-end consequence on a zero-field
record: a free ``B_L`` seeded on its own 0 lower bound cannot be started by
Migrad, which used to cost the record its own model.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.component_tags import FieldGeometry
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.fit_wizard import (
    CandidateTemplate,
    TemplateSeedContext,
    _initial_parameters_for_template,
    _pinned_longitudinal_field,
    build_fit_wizard_recommendation,
    fingerprint_spectrum,
)
from asymmetry.core.fitting.models import (
    dynamic_lorentzian_kt,
    field_decoupling_threshold_gauss,
)
from asymmetry.core.fitting.parameters import split_parameter_name

MUON_LIFETIME_US = 2.197


def _record(*, metadata: dict | None = None, n_points: int = 400) -> MuonDataset:
    """A cheap Kubo-Toyabe-shaped record; only its metadata matters here."""
    t = np.linspace(0.008, 12.0, n_points)
    y = 20.0 * dynamic_lorentzian_kt(t, 1.0, 0.4, 0.3, 0.0) + 2.0
    payload = {"run_number": 1}
    payload.update(metadata or {})
    return MuonDataset(
        time=t,
        asymmetry=y,
        error=np.full_like(t, 0.3),
        metadata=payload,
    )


def _template(components: list[str], operators: list[str], key: str) -> CandidateTemplate:
    return CandidateTemplate(
        key=key,
        title=key,
        category="Kubo-Toyabe",
        rationale="",
        model=CompositeModel(components, operators=operators),
    )


def _seeded(
    template: CandidateTemplate,
    *,
    field_gauss: float | None,
    geometry: FieldGeometry | None,
):
    dataset = _record()
    return _initial_parameters_for_template(
        dataset,
        fingerprint_spectrum(dataset),
        template,
        seed_context=TemplateSeedContext(field_gauss=field_gauss, geometry=geometry),
    )


def _lkt_template() -> CandidateTemplate:
    return _template(["DynamicLorentzianKT", "Constant"], ["+"], "dynamic_lkt_constant")


# --- the pinning policy in isolation ---------------------------------------


@pytest.mark.parametrize(
    ("field_gauss", "geometry", "expected"),
    [
        # A ZF label means the magnet is nulled, whatever magnitude is recorded.
        (0.0, FieldGeometry.ZF, 0.0),
        (None, FieldGeometry.ZF, 0.0),
        (35.0, FieldGeometry.ZF, 0.0),
        # A zero applied field has a zero longitudinal component in any geometry.
        (0.0, None, 0.0),
        (0.0, FieldGeometry.LF, 0.0),
        (0.0, FieldGeometry.TF, 0.0),
        (-0.05, None, 0.0),
        # A known LF setpoint is the longitudinal component.
        (200.0, FieldGeometry.LF, 200.0),
        (-200.0, FieldGeometry.LF, 200.0),
        # Nothing to pin to.
        (None, None, None),
        (None, FieldGeometry.LF, None),
        # A TF setpoint says nothing about the longitudinal component.
        (150.0, FieldGeometry.TF, None),
        (150.0, None, None),
    ],
)
def test_longitudinal_field_pinning_policy(field_gauss, geometry, expected) -> None:
    assert _pinned_longitudinal_field(field_gauss, geometry) == expected


# --- how that reaches a seeded ParameterSet --------------------------------


def test_zero_field_run_pins_b_l_at_zero() -> None:
    seeded = _seeded(_lkt_template(), field_gauss=0.0, geometry=FieldGeometry.ZF)
    b_l = seeded["B_L"]
    assert b_l.fixed is True
    assert b_l.value == pytest.approx(0.0)


def test_zero_magnitude_with_unknown_geometry_pins_b_l_at_zero() -> None:
    seeded = _seeded(_lkt_template(), field_gauss=0.0, geometry=None)
    assert seeded["B_L"].fixed is True
    assert seeded["B_L"].value == pytest.approx(0.0)


def test_zf_label_with_a_stale_setpoint_still_pins_zero() -> None:
    # A non-zero magnitude alongside a "zero field" label is a nominal/stale
    # reading, not a longitudinal field.
    seeded = _seeded(_lkt_template(), field_gauss=50.0, geometry=FieldGeometry.ZF)
    assert seeded["B_L"].fixed is True
    assert seeded["B_L"].value == pytest.approx(0.0)


def test_longitudinal_run_pins_b_l_at_the_setpoint() -> None:
    seeded = _seeded(_lkt_template(), field_gauss=250.0, geometry=FieldGeometry.LF)
    assert seeded["B_L"].fixed is True
    assert seeded["B_L"].value == pytest.approx(250.0)
    assert seeded["B_L"].min <= 250.0 <= seeded["B_L"].max


def test_unrecorded_field_leaves_b_l_free_and_strictly_inside_bounds() -> None:
    template = _lkt_template()
    seeded = _seeded(template, field_gauss=None, geometry=None)
    b_l = seeded["B_L"]
    assert b_l.fixed is False
    # Strictly inside — Migrad cannot start on a bound.
    assert b_l.min < b_l.value < b_l.max
    # ...and clear of the zero-field branch, where the objective is exactly
    # flat in B_L because the line shape ignores it entirely.
    assert b_l.value > field_decoupling_threshold_gauss(seeded["a_L"].value)


def test_transverse_run_leaves_b_l_free() -> None:
    seeded = _seeded(_lkt_template(), field_gauss=150.0, geometry=FieldGeometry.TF)
    assert seeded["B_L"].fixed is False
    assert seeded["B_L"].min < seeded["B_L"].value


def test_muonium_lf_relax_keeps_its_own_default_when_b_l_is_free() -> None:
    # A real longitudinal-field model seeds B_L at 10 G; the free-seed floor
    # must not drag that down to a near-zero weak-field guess.
    template = _template(["MuoniumLFRelax", "Constant"], ["+"], "muonium_lf_relax_constant")
    seeded = _seeded(template, field_gauss=None, geometry=None)
    assert seeded["B_L"].fixed is False
    assert seeded["B_L"].value == pytest.approx(10.0)


def test_every_b_l_carrier_in_a_composite_is_pinned() -> None:
    template = _template(
        ["DynamicLorentzianKT", "LongitudinalFieldKT", "Constant"],
        ["+", "+"],
        "two_carrier_lkt",
    )
    names = [name for name in template.model.param_names if split_parameter_name(name)[0] == "B_L"]
    assert len(names) == 2, names
    seeded = _seeded(template, field_gauss=120.0, geometry=FieldGeometry.LF)
    for name in names:
        assert seeded[name].fixed is True
        assert seeded[name].value == pytest.approx(120.0)


def test_muonium_field_parameter_is_still_pinned() -> None:
    template = _template(["MuoniumLowTF", "Constant"], ["+"], "muonium_low_tf_constant")
    seeded = _seeded(template, field_gauss=100.0, geometry=FieldGeometry.TF)
    assert seeded["field"].fixed is True
    assert seeded["field"].value == pytest.approx(100.0)


@pytest.mark.parametrize("geometry", [FieldGeometry.TF, None])
def test_recorded_zero_setpoint_pins_field_at_zero(geometry) -> None:
    # Recorded is recorded: a 0 G setpoint is metadata like any other value,
    # and the transverse applied-field carriers have nothing to precess in.
    template = _template(["MuoniumLowTF", "Constant"], ["+"], "muonium_low_tf_constant")
    seeded = _seeded(template, field_gauss=0.0, geometry=geometry)
    assert seeded["field"].fixed is True
    assert seeded["field"].value == pytest.approx(0.0)


def test_unrecorded_field_leaves_field_free() -> None:
    template = _template(["MuoniumLowTF", "Constant"], ["+"], "muonium_low_tf_constant")
    seeded = _seeded(template, field_gauss=None, geometry=None)
    assert seeded["field"].fixed is False


# --- end to end -------------------------------------------------------------


def test_wizard_recovers_dynamic_lorentzian_kt_on_a_zero_field_record() -> None:
    """The whole point: a genuine ZF dynamic-LKT record keeps its own model.

    With ``B_L`` free it was seeded on its 0 lower bound, Migrad refused to
    start ("parameters at limit"), the true candidate was discarded, and the
    wizard recommended a stretched exponential instead.
    """
    rng = np.random.default_rng(20240517)
    t = np.linspace(0.008, 16.0, 2000)
    sigma = 0.3 * np.exp(t / (2.0 * MUON_LIFETIME_US))
    y = 20.0 * dynamic_lorentzian_kt(t, 1.0, 0.4, 0.3, 0.0) + 2.0 + rng.normal(0.0, sigma)
    dataset = MuonDataset(
        time=t,
        asymmetry=y,
        error=sigma,
        metadata={"run_number": 7, "field": 0.0, "field_direction": "ZF"},
    )

    recommendation = build_fit_wizard_recommendation(dataset, max_workers=1)

    assert recommendation.recommended_key == "dynamic_lkt_constant"
    assessment = recommendation.recommended_assessment
    assert assessment is not None
    assert assessment.fit_result.success is True
    b_l = assessment.fit_result.parameters["B_L"]
    assert b_l.fixed is True
    assert b_l.value == pytest.approx(0.0)
    # A pinned parameter must not be counted against the information criteria.
    assert "B_L" not in {p.name for p in assessment.fit_result.parameters.free_parameters}


def test_global_wizard_candidate_sets_follow_the_same_policy() -> None:
    # The series candidates are built from the same seeding function; without a
    # field seed context a zero-field series would carry a free B_L per run
    # that each run's own screening had already pinned.
    from asymmetry.core.fitting.global_fit_wizard import _configured_single_fit_parameter_set

    dataset = _record(metadata={"field": 0.0, "field_direction": "ZF"})
    parameters = _configured_single_fit_parameter_set(
        dataset,
        fingerprint_spectrum(dataset),
        _lkt_template(),
        current_values={},
        parameter_bounds={},
        fixed_param_names=(),
    )

    assert parameters["B_L"].fixed is True
    assert parameters["B_L"].value == pytest.approx(0.0)

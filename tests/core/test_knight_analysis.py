"""Knight-shift analysis session: snapshot -> branches + crossings (Phase 3c)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from asymmetry.core.fitting.knight_analysis import (
    KnightAnalysisInput,
    KnightAnalysisState,
    KnightPoint,
    migrate_legacy_state,
    selected_components,
    snapshot_from_rows,
)
from asymmetry.core.fitting.knight_analysis import evaluate as knight_evaluate
from asymmetry.core.fitting.knight_shift import (
    REFERENCE_APPLIED_FIELD,
    REFERENCE_COMPONENT,
    KnightShiftConfig,
    KnightShiftUnit,
    knight_shift,
    label_for_unit,
    larmor_frequency_mhz,
    scale_for_unit,
)


def _point(
    run_number: int,
    x: float,
    field_gauss: float,
    values: dict,
    errors: dict | None = None,
    covariance: dict | None = None,
    include: bool = True,
) -> KnightPoint:
    return KnightPoint(
        run_number=run_number,
        run_label=f"run{run_number}",
        x=x,
        field_gauss=field_gauss,
        values=values,
        errors=errors or {name: 0.0 for name in values},
        covariance=covariance,
        include=include,
    )


def _input(
    points,
    components=(("frequency", "frequency"),),
    x_key="angle",
    x_label="Angle",
) -> KnightAnalysisInput:
    return KnightAnalysisInput(
        x_key=x_key,
        x_label=x_label,
        components=tuple(components),
        points=tuple(points),
    )


# --- KnightPoint / KnightAnalysisInput construction ------------------------


def test_component_names_returns_names_in_order():
    analysis_input = _input(
        [_point(1, 0.0, 7000.0, {"frequency": 95.0, "field": 500.0})],
        components=(("frequency", "frequency"), ("field", "field")),
    )
    assert analysis_input.component_names() == ("frequency", "field")


def test_knight_point_defaults():
    point = KnightPoint(
        run_number=1, run_label="run1", x=0.0, field_gauss=7000.0, values={}, errors={}
    )
    assert point.covariance is None
    assert point.include is True


# --- evaluate(): applied-field reference -----------------------------------


def test_applied_field_frequency_kind_matches_direct_call():
    field_gauss = 7000.0
    nu = larmor_frequency_mhz(field_gauss) * 1.0023
    points = [_point(1, 10.0, field_gauss, {"frequency": nu}, {"frequency": 0.01})]
    analysis_input = _input(points)
    config = KnightShiftConfig(enabled=True, unit=KnightShiftUnit.FRACTION)

    result = knight_evaluate(analysis_input, config)

    expected_k, expected_sigma = knight_shift(nu, larmor_frequency_mhz(field_gauss), sigma_nu=0.01)
    branch = result.branch("K[frequency]")
    assert branch is not None
    assert branch.k[0] == pytest.approx(expected_k)
    assert branch.k_err[0] == pytest.approx(expected_sigma)


def test_applied_field_field_kind_uses_field_itself():
    field_gauss = 500.0
    b_mu = field_gauss * 1.001
    points = [_point(1, 10.0, field_gauss, {"field": b_mu}, {"field": 0.5})]
    analysis_input = _input(points, components=(("field", "field"),))
    config = KnightShiftConfig(enabled=True, unit=KnightShiftUnit.FRACTION)

    result = knight_evaluate(analysis_input, config)

    expected_k, expected_sigma = knight_shift(b_mu, field_gauss, sigma_nu=0.5)
    branch = result.branch("K[field]")
    assert branch is not None
    assert branch.k[0] == pytest.approx(expected_k)
    assert branch.k_err[0] == pytest.approx(expected_sigma)


def test_evaluate_branch_x_is_sorted_ascending_and_aligned_with_run_numbers():
    field_gauss = 7000.0
    points = [
        _point(3, 60.0, field_gauss, {"frequency": 95.5}),
        _point(1, 0.0, field_gauss, {"frequency": 95.0}),
        _point(2, 30.0, field_gauss, {"frequency": 95.2}),
    ]
    analysis_input = _input(points)
    config = KnightShiftConfig(enabled=True, unit=KnightShiftUnit.FRACTION)

    result = knight_evaluate(analysis_input, config)

    branch = result.branch("K[frequency]")
    assert branch.x == (0.0, 30.0, 60.0)
    assert branch.run_numbers == (1, 2, 3)
    assert branch.included == (True, True, True)


# --- evaluate(): component reference ----------------------------------------


def test_component_reference_matches_direct_call_and_carries_covariance():
    points = [
        _point(
            1,
            0.0,
            7000.0,
            {"frequency": 95.5, "frequency_2": 95.0},
            {"frequency": 0.02, "frequency_2": 0.01},
            covariance={"frequency": {"frequency_2": 0.0002}, "frequency_2": {"frequency": 0.0002}},
        )
    ]
    analysis_input = _input(
        points,
        components=(("frequency", "frequency"), ("frequency_2", "frequency")),
    )
    config = KnightShiftConfig(
        enabled=True,
        reference_mode=REFERENCE_COMPONENT,
        reference_component="frequency_2",
        unit=KnightShiftUnit.FRACTION,
    )

    result = knight_evaluate(analysis_input, config)

    expected_k, expected_sigma = knight_shift(95.5, 95.0, sigma_nu=0.02, sigma_ref=0.01, cov=0.0002)
    branch = result.branch("K[frequency]")
    assert branch.k[0] == pytest.approx(expected_k)
    assert branch.k_err[0] == pytest.approx(expected_sigma)
    # Reference itself must be excluded from the branches.
    assert result.branch("K[frequency_2]") is None


def test_component_reference_same_kind_restriction():
    # A 'field' reference never converts a 'frequency' component and vice versa.
    points = [
        _point(
            1, 0.0, 7000.0, {"frequency": 95.5, "field": 500.0}, {"frequency": 0.02, "field": 0.5}
        )
    ]
    analysis_input = _input(points, components=(("frequency", "frequency"), ("field", "field")))
    config = KnightShiftConfig(
        enabled=True, reference_mode=REFERENCE_COMPONENT, reference_component="field"
    )

    result = knight_evaluate(analysis_input, config)

    assert result.branch("K[frequency]") is None
    assert result.branch("K[field]") is None  # field is the reference, excluded from its own branch


def test_covariance_changes_sigma_relative_to_zero_covariance():
    common = dict(
        values={"frequency": 95.5, "frequency_2": 95.0},
        errors={"frequency": 0.02, "frequency_2": 0.01},
    )
    analysis_input_no_cov = _input(
        [_point(1, 0.0, 7000.0, **common)],
        components=(("frequency", "frequency"), ("frequency_2", "frequency")),
    )
    analysis_input_cov = _input(
        [
            _point(
                1,
                0.0,
                7000.0,
                covariance={"frequency": {"frequency_2": 0.0002}},
                **common,
            )
        ],
        components=(("frequency", "frequency"), ("frequency_2", "frequency")),
    )
    config = KnightShiftConfig(
        enabled=True, reference_mode=REFERENCE_COMPONENT, reference_component="frequency_2"
    )

    sigma_no_cov = knight_evaluate(analysis_input_no_cov, config).branch("K[frequency]").k_err[0]
    sigma_cov = knight_evaluate(analysis_input_cov, config).branch("K[frequency]").k_err[0]

    assert sigma_no_cov != pytest.approx(sigma_cov)


def test_missing_reference_component_in_point_gives_no_branches_data():
    # Reference is a valid component overall, but this particular point lacks it:
    # selected_components() still returns 'frequency', but the per-point value is NaN.
    points = [_point(1, 0.0, 7000.0, {"frequency": 95.5})]
    analysis_input = _input(
        points, components=(("frequency", "frequency"), ("frequency_2", "frequency"))
    )
    config = KnightShiftConfig(
        enabled=True, reference_mode=REFERENCE_COMPONENT, reference_component="frequency_2"
    )

    result = knight_evaluate(analysis_input, config)
    branch = result.branch("K[frequency]")
    assert branch is not None
    assert math.isnan(branch.k[0])
    assert math.isnan(branch.k_err[0])


def test_reference_not_in_snapshot_components_gives_no_branches():
    points = [_point(1, 0.0, 7000.0, {"frequency": 95.5})]
    analysis_input = _input(points, components=(("frequency", "frequency"),))
    config = KnightShiftConfig(
        enabled=True, reference_mode=REFERENCE_COMPONENT, reference_component="not_present"
    )
    result = knight_evaluate(analysis_input, config)
    assert result.branches == ()


# --- config component subset -------------------------------------------------


def test_empty_components_tuple_selects_all():
    analysis_input = _input(
        [_point(1, 0.0, 7000.0, {"frequency": 95.0, "field": 500.0})],
        components=(("frequency", "frequency"), ("field", "field")),
    )
    config = KnightShiftConfig(enabled=True, components=())
    selected = selected_components(analysis_input, config)
    assert set(selected) == {("frequency", "frequency"), ("field", "field")}


def test_subset_converts_only_listed_components():
    analysis_input = _input(
        [_point(1, 0.0, 7000.0, {"frequency": 95.0, "field": 500.0})],
        components=(("frequency", "frequency"), ("field", "field")),
    )
    config = KnightShiftConfig(enabled=True, components=("frequency",))
    selected = selected_components(analysis_input, config)
    assert selected == (("frequency", "frequency"),)


# --- disabled config ---------------------------------------------------------


def test_disabled_config_yields_no_branches_no_crossings_but_resolved_unit():
    points = [_point(1, 0.0, 7000.0, {"frequency": 95.0})]
    analysis_input = _input(points)
    config = KnightShiftConfig(enabled=False, unit=KnightShiftUnit.PERCENT)

    result = knight_evaluate(analysis_input, config)

    assert result.branches == ()
    assert result.crossings == ()
    assert result.unit is KnightShiftUnit.PERCENT
    assert result.scale == scale_for_unit(KnightShiftUnit.PERCENT)
    assert result.unit_label == label_for_unit(KnightShiftUnit.PERCENT)


# --- AUTO unit resolution -----------------------------------------------------


def test_auto_unit_resolves_ppm_for_small_shifts():
    field_gauss = 7000.0
    nu_ref = larmor_frequency_mhz(field_gauss)
    nu = nu_ref * (1.0 + 1e-5)  # well under the 1e-3 ppm/percent threshold
    points = [_point(1, 0.0, field_gauss, {"frequency": nu})]
    analysis_input = _input(points)
    config = KnightShiftConfig(enabled=True, unit=KnightShiftUnit.AUTO)

    result = knight_evaluate(analysis_input, config)

    assert result.unit is KnightShiftUnit.PPM
    assert result.scale == scale_for_unit(KnightShiftUnit.PPM)
    assert result.unit_label == label_for_unit(KnightShiftUnit.PPM)


def test_auto_unit_resolves_percent_for_large_shifts():
    field_gauss = 7000.0
    nu_ref = larmor_frequency_mhz(field_gauss)
    nu = nu_ref * 1.02  # 2% shift, well above the ppm threshold
    points = [_point(1, 0.0, field_gauss, {"frequency": nu})]
    analysis_input = _input(points)
    config = KnightShiftConfig(enabled=True, unit=KnightShiftUnit.AUTO)

    result = knight_evaluate(analysis_input, config)

    assert result.unit is KnightShiftUnit.PERCENT
    assert result.scale == scale_for_unit(KnightShiftUnit.PERCENT)
    assert result.unit_label == label_for_unit(KnightShiftUnit.PERCENT)


def test_explicit_unit_passes_through_regardless_of_data():
    field_gauss = 7000.0
    nu_ref = larmor_frequency_mhz(field_gauss)
    nu = nu_ref * 1.02
    points = [_point(1, 0.0, field_gauss, {"frequency": nu})]
    analysis_input = _input(points)
    config = KnightShiftConfig(enabled=True, unit=KnightShiftUnit.FRACTION)

    result = knight_evaluate(analysis_input, config)

    assert result.unit is KnightShiftUnit.FRACTION
    assert result.scale == 1.0
    assert result.unit_label == ""


# --- dropped / included points -----------------------------------------------


def test_nan_x_point_is_dropped_and_counted_in_skipped():
    points = [
        _point(1, 0.0, 7000.0, {"frequency": 95.0}),
        _point(2, float("nan"), 7000.0, {"frequency": 95.1}),
        _point(3, 30.0, 7000.0, {"frequency": 95.2}),
    ]
    analysis_input = _input(points)
    config = KnightShiftConfig(enabled=True)

    result = knight_evaluate(analysis_input, config)

    branch = result.branch("K[frequency]")
    assert branch.run_numbers == (1, 3)
    assert result.skipped_points == 1


def test_missing_component_point_is_dropped_and_counted_in_skipped():
    points = [
        _point(1, 0.0, 7000.0, {"frequency": 95.0}),
        _point(2, 30.0, 7000.0, {}),  # missing 'frequency' entirely
    ]
    analysis_input = _input(points)
    config = KnightShiftConfig(enabled=True)

    result = knight_evaluate(analysis_input, config)

    branch = result.branch("K[frequency]")
    assert branch.run_numbers == (1,)
    assert result.skipped_points == 1


def test_skipped_points_counts_only_points_dropped_from_every_branch():
    # A point missing just one of two selected components is kept in the branch
    # for the component it does have and is NOT counted as skipped; only a point
    # retained by no branch at all (NaN abscissa, or every component missing) is.
    points = [
        _point(1, 0.0, 7000.0, {"frequency": 95.0, "frequency_2": 95.1}),
        _point(2, 30.0, 7000.0, {"frequency": 95.2}),  # missing frequency_2 only
        _point(3, float("nan"), 7000.0, {"frequency": 95.3, "frequency_2": 95.4}),
        _point(4, 60.0, 7000.0, {}),  # no components at all
    ]
    analysis_input = _input(
        points, components=(("frequency", "frequency"), ("frequency_2", "frequency"))
    )
    config = KnightShiftConfig(enabled=True)

    result = knight_evaluate(analysis_input, config)

    # Run 2 is retained in the frequency branch and dropped only from
    # frequency_2's — a partial miss, so it does not count as skipped.
    assert result.branch("K[frequency]").run_numbers == (1, 2)
    assert result.branch("K[frequency_2]").run_numbers == (1,)
    # Runs 3 (NaN abscissa) and 4 (no components) appear in no branch: skipped.
    assert result.skipped_points == 2


def test_excluded_point_is_kept_in_branch_flagged_not_included():
    points = [
        _point(1, 0.0, 7000.0, {"frequency": 95.0}, include=True),
        _point(2, 30.0, 7000.0, {"frequency": 95.1}, include=False),
    ]
    analysis_input = _input(points)
    config = KnightShiftConfig(enabled=True)

    result = knight_evaluate(analysis_input, config)

    branch = result.branch("K[frequency]")
    assert branch.run_numbers == (1, 2)
    assert branch.included == (True, False)
    assert result.skipped_points == 0


# --- crossing detection -------------------------------------------------------


def test_two_components_swapping_order_are_flagged_as_crossing():
    field_gauss = 7000.0
    points = [
        _point(1, 0.0, field_gauss, {"frequency": 10.0, "frequency_2": 20.0}),
        _point(2, 30.0, field_gauss, {"frequency": 21.0, "frequency_2": 11.0}),  # swapped
    ]
    analysis_input = _input(
        points, components=(("frequency", "frequency"), ("frequency_2", "frequency"))
    )
    config = KnightShiftConfig(enabled=True)

    result = knight_evaluate(analysis_input, config)

    assert any(e.kind == "order_swap" for e in result.crossings)


def test_mixed_kind_components_never_cross_compared():
    # A Gauss-scale 'field' pair whose values happen to swap, plus an MHz-scale
    # 'frequency' singleton that cannot cross (only one frequency component) —
    # the frequency/field groups must never be compared against each other.
    points = [
        _point(1, 0.0, 7000.0, {"field": 100.0, "field_2": 200.0, "frequency": 95.0}),
        _point(2, 30.0, 7000.0, {"field": 210.0, "field_2": 110.0, "frequency": 95.2}),
    ]
    analysis_input = _input(
        points,
        components=(("field", "field"), ("field_2", "field"), ("frequency", "frequency")),
    )
    config = KnightShiftConfig(enabled=True)

    result = knight_evaluate(analysis_input, config)

    # Only the field pair can produce a swap; no spurious cross-kind events.
    assert all(e.component_pair in {(0, 1)} for e in result.crossings if e.kind == "order_swap")
    assert any(e.kind == "order_swap" for e in result.crossings)


# --- selected_components() unit tests -----------------------------------------


def test_selected_components_applied_field_ignores_reference_restriction():
    analysis_input = _input(
        [_point(1, 0.0, 7000.0, {"frequency": 95.0, "field": 500.0})],
        components=(("frequency", "frequency"), ("field", "field")),
    )
    config = KnightShiftConfig(enabled=True, reference_mode=REFERENCE_APPLIED_FIELD)
    assert set(selected_components(analysis_input, config)) == {
        ("frequency", "frequency"),
        ("field", "field"),
    }


def test_selected_components_component_reference_restricts_to_same_kind():
    analysis_input = _input(
        [_point(1, 0.0, 7000.0, {"frequency": 95.0, "frequency_2": 96.0, "field": 500.0})],
        components=(("frequency", "frequency"), ("frequency_2", "frequency"), ("field", "field")),
    )
    config = KnightShiftConfig(
        enabled=True, reference_mode=REFERENCE_COMPONENT, reference_component="frequency_2"
    )
    selected = selected_components(analysis_input, config)
    assert selected == (("frequency", "frequency"),)  # field excluded (wrong kind), ref excluded


def test_selected_components_reference_missing_from_snapshot_returns_empty():
    analysis_input = _input(
        [_point(1, 0.0, 7000.0, {"frequency": 95.0})], components=(("frequency", "frequency"),)
    )
    config = KnightShiftConfig(
        enabled=True, reference_mode=REFERENCE_COMPONENT, reference_component="ghost"
    )
    assert selected_components(analysis_input, config) == ()


# --- KnightAnalysisState round-trip -------------------------------------------


def test_knight_analysis_state_round_trip():
    config = KnightShiftConfig(
        enabled=True,
        reference_mode=REFERENCE_COMPONENT,
        reference_component="frequency_2",
        unit=KnightShiftUnit.PERCENT,
        components=("frequency",),
    )
    state = KnightAnalysisState(
        config=config,
        source_batch_id="batch-1",
        source_group_id="group-1",
        x_key="temperature",
        fold_180=True,
        show_markers=False,
    )

    restored = KnightAnalysisState.from_dict(state.to_dict())

    assert restored.config.to_dict() == config.to_dict()
    assert restored.source_batch_id == "batch-1"
    assert restored.source_group_id == "group-1"
    assert restored.x_key == "temperature"
    assert restored.fold_180 is True
    assert restored.show_markers is False


@pytest.mark.parametrize("garbage", [None, [], "not a dict", 5, 3.14])
def test_knight_analysis_state_from_dict_garbage_gives_defaults(garbage):
    state = KnightAnalysisState.from_dict(garbage)
    default = KnightAnalysisState()
    assert state.to_dict() == default.to_dict()


# --- migrate_legacy_state() ---------------------------------------------------


@pytest.mark.parametrize("garbage", [None, [], "nope", 5])
def test_migrate_legacy_state_non_dict_returns_none(garbage):
    assert migrate_legacy_state(garbage) is None


def test_migrate_legacy_state_missing_block_returns_none():
    assert migrate_legacy_state({"x_axis_key": "angle"}) is None


def test_migrate_legacy_state_disabled_returns_none():
    legacy = {"knight_shift": {"enabled": False, "unit": "ppm"}, "x_axis_key": "angle"}
    assert migrate_legacy_state(legacy) is None


def test_migrate_legacy_state_realistic_block_maps_fields():
    legacy = {
        "knight_shift": {
            "enabled": True,
            "unit": "percent",
            "reference_mode": REFERENCE_COMPONENT,
            "reference_component": "frequency_2",
            "components": ["frequency"],
        },
        "x_axis_key": "angle",
        "active_group_id": "g1",
    }

    state = migrate_legacy_state(legacy)

    assert state is not None
    assert state.config.enabled is True
    assert state.config.unit is KnightShiftUnit.PERCENT
    assert state.config.reference_mode == REFERENCE_COMPONENT
    assert state.config.reference_component == "frequency_2"
    assert state.config.components == ("frequency",)
    assert state.x_key == "angle"
    assert state.source_group_id == "g1"
    assert state.source_batch_id is None


# --- snapshot_from_rows() -----------------------------------------------------


@dataclass
class _Row:
    run_number: int
    run_label: str
    field: float
    values: dict
    errors: dict
    covariance: dict | None = None
    include_in_trend: bool = True


def test_snapshot_from_rows_maps_fields_correctly():
    rows = [
        _Row(
            1, "run1", 7000.0, {"frequency": 95.0}, {"frequency": 0.1}, covariance={"a": {"b": 1.0}}
        ),
        _Row(2, "run2", 7050.0, {"frequency": 95.5}, {"frequency": 0.2}, include_in_trend=False),
    ]
    x_values = [10.0, 20.0]

    snapshot = snapshot_from_rows(
        rows,
        x_values=x_values,
        x_key="angle",
        x_label="Angle",
        components=[("frequency", "frequency")],
        source_label="my source",
        batch_id="b1",
        group_id="g1",
    )

    assert snapshot.x_key == "angle"
    assert snapshot.x_label == "Angle"
    assert snapshot.components == (("frequency", "frequency"),)
    assert snapshot.source_label == "my source"
    assert snapshot.batch_id == "b1"
    assert snapshot.group_id == "g1"
    assert len(snapshot.points) == 2

    p0, p1 = snapshot.points
    assert p0.run_number == 1
    assert p0.run_label == "run1"
    assert p0.x == 10.0
    assert p0.field_gauss == 7000.0
    assert p0.values == {"frequency": 95.0}
    assert p0.covariance == {"a": {"b": 1.0}}
    assert p0.include is True

    assert p1.run_number == 2
    assert p1.x == 20.0
    assert p1.include is False
    assert p1.covariance is None


def test_snapshot_from_rows_length_mismatch_raises():
    rows = [_Row(1, "run1", 7000.0, {}, {})]
    with pytest.raises(ValueError):
        snapshot_from_rows(rows, x_values=[1.0, 2.0], x_key="angle", x_label="Angle", components=[])


def test_snapshot_from_rows_non_finite_field_and_x_coerced_to_nan():
    rows = [_Row(1, "run1", float("inf"), {}, {})]
    snapshot = snapshot_from_rows(
        rows, x_values=[float("nan")], x_key="angle", x_label="Angle", components=[]
    )
    point = snapshot.points[0]
    assert math.isnan(point.x)
    assert math.isnan(point.field_gauss)

"""Tests for the fit-wizard scoping module."""

from __future__ import annotations

import json

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset, Run
from asymmetry.core.fitting.component_tags import ComputationalCost, FieldGeometry, PhysicsClass
from asymmetry.core.fitting.composite import COMPONENTS
from asymmetry.core.fitting.user_functions import register_component
from asymmetry.core.fitting.wizard_scope import (
    DEFAULT_EFFORT_TIER,
    EFFORT_TIER_DESCRIPTIONS,
    EFFORT_TIER_LABELS,
    FAMILY_ORDER,
    MUONIUM_HIGH_TF_MIN_GAUSS,
    MUONIUM_LOW_TF_MAX_GAUSS,
    USER_FAMILY_TITLE,
    ZERO_FIELD_MAX_GAUSS,
    EffortTier,
    ExcludedComponent,
    ScopeResolution,
    WizardScope,
    dataset_field_geometry,
    describe_scope,
    effort_tier_from_payload,
    effort_tier_to_payload,
    infer_run_geometries,
    resolve_scope,
    resolve_scope_for_dataset,
    resolve_scope_for_datasets,
    set_user_field_direction,
    user_field_direction_overrides,
)

MAGNETISM = frozenset({PhysicsClass.MAGNETISM})
MOLECULAR = frozenset({PhysicsClass.MOLECULAR})
LF_DYNAMICS = frozenset({PhysicsClass.DYNAMICS, PhysicsClass.MAGNETISM})
SUPERCONDUCTOR = frozenset({PhysicsClass.SUPERCONDUCTIVITY, PhysicsClass.MAGNETISM})


def _time_component_names() -> set[str]:
    return {n for n, d in COMPONENTS.items() if d.domain == "time"}


def _frequency_component_names() -> set[str]:
    return {n for n, d in COMPONENTS.items() if d.domain == "frequency"}


def _resolve(physics: frozenset[PhysicsClass], **kw) -> ScopeResolution:
    return resolve_scope(WizardScope(physics=physics), **kw)


# --- registry labels ----------------------------------------------------


def test_every_component_has_a_label_and_a_use_when_line():
    for name, definition in COMPONENTS.items():
        assert definition.label.strip(), name
        assert definition.use_when.strip(), name


def test_time_component_labels_are_distinct():
    labels = [d.label for d in COMPONENTS.values() if d.domain == "time"]
    assert len(labels) == len(set(labels))


# --- physics classes ----------------------------------------------------


def test_magnetism_in_zero_field_keeps_zf_magnetism_and_envelopes():
    res = _resolve(MAGNETISM, field_direction="ZF")
    assert {"Exponential", "Constant", "StaticGKT_ZF", "Oscillatory"} <= res.included_set
    # A TF-only muonium form is excluded with a geometry reason.
    assert "MuoniumTF" not in res.included_set
    reason = next(e.reason for e in res.excluded_components if e.name == "MuoniumTF")
    assert "geometry" in reason
    # A molecular component is excluded with a physics reason.
    reason = next(e.reason for e in res.excluded_components if e.name == "FmuF_Linear")
    assert "molecular" in reason and "magnetism" in reason


def test_physics_classes_combine():
    res = _resolve(frozenset({PhysicsClass.DYNAMICS, PhysicsClass.MUONIUM}), field_direction="LF")
    assert {"MuoniumLFRelax", "DynamicGaussianKT", "Keren"} <= res.included_set
    # LongitudinalFieldKT is static magnetism only.
    assert "LongitudinalFieldKT" not in res.included_set


def test_superconductor_in_tf_includes_vortex_lattice():
    res = _resolve(SUPERCONDUCTOR, field_direction="TF", field_gauss=100.0)
    assert {"VortexLattice", "Exponential", "Constant"} <= res.included_set


def test_molecular_membership():
    res = _resolve(MOLECULAR)
    assert {"FmuF_Linear", "Exponential", "Constant"} <= res.included_set
    # Oscillatory is MAGNETISM, not looked for.
    assert "Oscillatory" not in res.included_set


def test_empty_physics_with_no_geometry_includes_every_time_domain_component():
    res = resolve_scope(WizardScope())
    assert res.included_set == _time_component_names()


@pytest.mark.parametrize("physics", [MAGNETISM, MOLECULAR, LF_DYNAMICS, SUPERCONDUCTOR])
def test_every_physics_choice_keeps_exponential_and_constant(physics):
    res = _resolve(physics, field_direction="ZF")
    assert "Exponential" in res.included_set
    assert "Constant" in res.included_set


def test_frequency_domain_components_excluded_everywhere():
    freq = _frequency_component_names()
    assert freq  # sanity: there are frequency-domain components
    for physics in (frozenset(), MAGNETISM, MOLECULAR):
        res = _resolve(physics)
        assert not (res.included_set & freq), physics
        excluded_names = {e.name for e in res.excluded_components}
        assert freq <= excluded_names, physics


def test_physics_note_names_the_classes_looked_for():
    res = _resolve(LF_DYNAMICS, field_direction="ZF")
    assert "looking for dynamics, magnetism" in res.notes


# --- slow models --------------------------------------------------------


def test_skip_slow_caps_the_cost_at_moderate():
    res = resolve_scope(WizardScope(skip_slow=True))
    assert res.query.max_cost is ComputationalCost.MODERATE
    slow = {n for n, d in COMPONENTS.items() if d.cost is ComputationalCost.EXPENSIVE}
    assert slow and not (slow & res.included_set)
    reason = next(e.reason for e in res.excluded_components if e.name == "DynamicGaussianKT")
    assert "slow" in reason


def test_skip_slow_off_sets_no_cost_cap():
    assert resolve_scope(WizardScope()).query.max_cost is None


# --- geometry from the run ----------------------------------------------


@pytest.mark.parametrize(
    ("field_direction", "expected"),
    [
        ("Transverse", {FieldGeometry.TF}),
        ("Longitudinal", {FieldGeometry.LF}),
        ("Zero field", {FieldGeometry.ZF}),
        ("TF", {FieldGeometry.TF}),
        ("LF", {FieldGeometry.LF}),
        ("ZF", {FieldGeometry.ZF}),
        ("", set(FieldGeometry)),
    ],
)
def test_geometries_come_from_the_recorded_direction(field_direction, expected):
    res = resolve_scope(WizardScope(physics=MAGNETISM), field_direction=field_direction)
    assert res.query.geometries == expected


def test_note_names_geometry_source():
    res = resolve_scope(WizardScope(), field_direction="Zero field")
    assert "zero field" in res.inference_note.lower()


def test_tf_low_field_regime():
    res = resolve_scope(WizardScope(), field_direction="TF", field_gauss=20.0)
    assert "MuoniumLowTF" in res.included_set
    assert "MuoniumHighTF" not in res.included_set
    assert "MuoniumHighTFAniso" not in res.included_set
    # MuoniumTF (exact four-frequency form) is never field-excluded.
    assert "MuoniumTF" in res.included_set
    high_reason = next(e.reason for e in res.excluded_components if e.name == "MuoniumHighTF")
    assert str(int(MUONIUM_HIGH_TF_MIN_GAUSS)) in high_reason or "20" in high_reason
    assert any("muonium" in note for note in res.notes)


def test_tf_high_field_regime():
    res = resolve_scope(WizardScope(), field_direction="TF", field_gauss=3000.0)
    assert "MuoniumHighTF" in res.included_set
    assert "MuoniumHighTFAniso" in res.included_set
    assert "MuoniumLowTF" not in res.included_set
    assert "MuoniumTF" in res.included_set
    low_reason = next(e.reason for e in res.excluded_components if e.name == "MuoniumLowTF")
    assert str(int(MUONIUM_LOW_TF_MAX_GAUSS)) in low_reason or "3000" in low_reason


def test_tf_unknown_field_excludes_no_muonium_regime():
    res = resolve_scope(WizardScope(), field_direction="TF", field_gauss=None)
    assert "MuoniumLowTF" in res.included_set
    assert "MuoniumHighTF" in res.included_set
    assert "MuoniumHighTFAniso" in res.included_set
    assert not any("muonium" in note for note in res.notes)


def test_unknown_geometry_is_superset_of_zero_field():
    unknown = resolve_scope(WizardScope(), field_direction="")
    zf = resolve_scope(WizardScope(), field_direction="ZF")
    assert zf.included_set <= unknown.included_set
    assert unknown.included_set == _time_component_names()


# --- B≈0 geometry-label override -----------------------------------------
#
# Real ISIS runs are sometimes recorded "TF" with the applied-field setpoint
# at (or near) zero — a ZF measurement on a TF-capable beamline (see
# docs/porting/field-geometry/: "MUSR00044991.nxs: magnetic_field_state='TF'
# at magnetic_field=0 G"). The scope must widen to include ZF families in that
# case without dropping the labelled TF family, which may still be
# hardware-correct.


def test_tf_label_zero_field_widens_to_include_zf_families():
    res = resolve_scope(WizardScope(), field_direction="TF", field_gauss=0.0)
    # ZF-only molecular family now in scope...
    assert "FmuF_Linear" in res.included_set
    # ...without losing the labelled TF family (VortexLattice is TF-only).
    assert "VortexLattice" in res.included_set
    assert "zero" in res.inference_note.lower()


def test_tf_label_nonzero_field_unchanged_behavior():
    res = resolve_scope(WizardScope(), field_direction="TF", field_gauss=100.0)
    assert "FmuF_Linear" not in res.included_set
    assert "VortexLattice" in res.included_set


def test_tf_label_at_override_threshold_widens():
    res = resolve_scope(WizardScope(), field_direction="TF", field_gauss=ZERO_FIELD_MAX_GAUSS)
    assert "FmuF_Linear" in res.included_set


def test_tf_label_just_above_threshold_does_not_widen():
    res = resolve_scope(WizardScope(), field_direction="TF", field_gauss=ZERO_FIELD_MAX_GAUSS + 0.5)
    assert "FmuF_Linear" not in res.included_set


def test_tf_label_negative_near_zero_field_widens():
    # A signed setpoint near zero (e.g. a small negative residual) still counts.
    res = resolve_scope(WizardScope(), field_direction="TF", field_gauss=-1.0)
    assert "FmuF_Linear" in res.included_set


def test_lf_label_zero_field_widens_to_include_zf_families():
    res = resolve_scope(WizardScope(), field_direction="LF", field_gauss=0.5)
    assert "FmuF_Linear" in res.included_set
    # LF-only dynamics family (labelled geometry) is preserved.
    zf_only = resolve_scope(WizardScope(), field_direction="ZF")
    lf_only = resolve_scope(WizardScope(), field_direction="LF")
    assert lf_only.included_set <= res.included_set
    assert zf_only.included_set <= res.included_set


def test_lf_label_nonzero_field_unchanged_behavior():
    res = resolve_scope(WizardScope(), field_direction="LF", field_gauss=680.0)
    assert "FmuF_Linear" not in res.included_set


def test_tf_label_unknown_field_does_not_widen():
    # No override without a recorded setpoint magnitude.
    res = resolve_scope(WizardScope(), field_direction="TF", field_gauss=None)
    assert "FmuF_Linear" not in res.included_set


def test_zf_label_unaffected_by_override():
    # ZF geometry is already the override's target; a field_gauss value must
    # not further change the ZF behaviour one way or the other.
    with_field = resolve_scope(WizardScope(), field_direction="ZF", field_gauss=0.0)
    without_field = resolve_scope(WizardScope(), field_direction="ZF", field_gauss=None)
    assert with_field.included_set == without_field.included_set
    assert with_field.query.geometries == {FieldGeometry.ZF}


def test_zero_field_override_via_dataset_wrapper():
    dataset = _fake_dataset("TF", field=0.0)
    res = resolve_scope_for_dataset(dataset, WizardScope())
    assert "FmuF_Linear" in res.included_set
    assert res.query.geometries == {FieldGeometry.TF, FieldGeometry.ZF}


# --- fluorine sniff -----------------------------------------------------


@pytest.mark.parametrize(
    "sample", ["PbF2", "CaF2", "LiF", "NaF", "KTCNQF4 T=300.0 F=100.0", "CaF2 TF20"]
)
def test_fluorine_sniff_positive(sample):
    _, notes, _ = infer_run_geometries("Zero field", None, sample)
    assert any("fluorine" in note for note in notes)


@pytest.mark.parametrize(
    "sample",
    [
        "Fe",
        "FeSe",
        "Fer",
        "",
        # ISIS run titles spell the applied field as ``F=<gauss>``; that is a
        # field, not a fluoride, and must not promote the F-mu-F family.
        "nickel_T=100_F=0",
        "Y(MnAl)2 T=75.0 F=110",
        "Teflon T=5.0 F=20.0",
        # Geometry tokens in PSI and ISIS titles: a field, not a fluoride.
        "EuO TF60G",
        "EuO ZF",
        "sample LF100",
    ],
)
def test_fluorine_sniff_negative(sample):
    _, notes, _ = infer_run_geometries("Zero field", None, sample)
    assert not any("fluorine" in note for note in notes)


# --- overrides ----------------------------------------------------------


def test_include_resurrects_query_excluded_component():
    scope = WizardScope(physics=MAGNETISM, include_components=frozenset({"VortexLattice"}))
    res = resolve_scope(scope, field_direction="ZF")
    assert "VortexLattice" in res.included_set
    assert "VortexLattice" not in {e.name for e in res.excluded_components}


def test_exclude_beats_include_for_same_name():
    scope = WizardScope(
        include_components=frozenset({"Exponential"}),
        exclude_components=frozenset({"Exponential"}),
    )
    res = resolve_scope(scope)
    assert "Exponential" not in res.included_set
    reason = next(e.reason for e in res.excluded_components if e.name == "Exponential")
    assert reason == "excluded by user"


def test_unknown_override_names_are_noted_not_crashing():
    scope = WizardScope(
        include_components=frozenset({"NoSuchComponent"}),
        exclude_components=frozenset({"AlsoMissing"}),
    )
    res = resolve_scope(scope)
    assert "NoSuchComponent" in res.inference_note
    assert "AlsoMissing" in res.inference_note


# --- exclusion reason quality -------------------------------------------


@pytest.mark.parametrize(("physics", "direction"), [(MAGNETISM, "ZF"), (SUPERCONDUCTOR, "TF")])
def test_every_exclusion_has_a_specific_nonempty_reason(physics, direction):
    res = _resolve(physics, field_direction=direction)
    assert res.excluded_components
    for exc in res.excluded_components:
        assert isinstance(exc, ExcludedComponent)
        assert exc.reason.strip()


# --- payload round-trip -------------------------------------------------


def test_payload_round_trip_and_json_safe():
    scope = WizardScope(
        physics=LF_DYNAMICS,
        include_components=frozenset({"VortexLattice"}),
        exclude_components=frozenset({"Exponential"}),
        skip_slow=True,
    )
    payload = scope.to_payload()
    assert payload == {
        "version": 2,
        "physics": ["dynamics", "magnetism"],
        "include": ["VortexLattice"],
        "exclude": ["Exponential"],
        "skip_slow": True,
    }
    json.dumps(payload)  # must not raise
    assert WizardScope.from_payload(payload) == scope


_V2 = {"version": 2, "physics": [], "include": [], "exclude": [], "skip_slow": False}


@pytest.mark.parametrize(
    ("payload", "invariant"),
    [
        (None, "mapping"),
        ([], "mapping"),
        ({"version": 1, "preset": "auto", "include": [], "exclude": []}, "exactly the keys"),
        ({**_V2, "version": 1}, "version 2"),
        ({**_V2, "physics": ["made-up"]}, "unknown physics class"),
        ({**_V2, "include": "Exponential"}, "list of strings"),
        ({**_V2, "exclude": [1]}, "list of strings"),
        ({**_V2, "skip_slow": "yes"}, "bool"),
    ],
)
def test_from_payload_rejects_anything_but_version_2(payload, invariant):
    with pytest.raises(ValueError, match=invariant):
        WizardScope.from_payload(payload)


# --- user component ubiquity --------------------------------------------


@pytest.fixture
def throwaway_user_component():
    name = "ZZWizardScopeProbe"

    def fn(t, A):  # noqa: N803 — param name must match the registered "A"
        return np.full_like(np.asarray(t, dtype=float), float(A))

    register_component(
        name,
        fn,
        ["A"],
        domain="time",
        description="throwaway probe component",
        formula_template="A",
        param_defaults={"A": 1.0},
    )
    try:
        yield name
    finally:
        COMPONENTS.pop(name, None)


def test_user_component_appears_in_every_scope(throwaway_user_component):
    name = throwaway_user_component
    for physics in (frozenset(), MAGNETISM, MOLECULAR, LF_DYNAMICS):
        res = _resolve(physics, field_direction="Zero field")
        assert name in res.included_set, physics


def test_user_component_can_still_be_excluded(throwaway_user_component):
    name = throwaway_user_component
    scope = WizardScope(physics=MAGNETISM, exclude_components=frozenset({name}))
    res = resolve_scope(scope)
    assert name not in res.included_set


def test_user_component_label_is_its_name_and_use_when_its_description(
    throwaway_user_component,
):
    definition = COMPONENTS[throwaway_user_component]
    assert definition.label == throwaway_user_component
    assert definition.use_when == "throwaway probe component"


# --- dataset wrappers ---------------------------------------------------


def _fake_dataset(field_state: str, field: float | None = None, title: str = "") -> MuonDataset:
    metadata = {"field_state": field_state}
    if field is not None:
        metadata["field"] = field
    if title:
        metadata["title"] = title
    return MuonDataset(
        time=np.linspace(0.0, 8.0, 4),
        asymmetry=np.zeros(4),
        error=np.ones(4),
        metadata=metadata,
    )


def test_resolve_for_dataset_reads_geometry_and_field():
    dataset = _fake_dataset("TF", field=20.0)
    res = resolve_scope_for_dataset(dataset, WizardScope())
    assert res.query.geometries == {FieldGeometry.TF}
    assert "MuoniumLowTF" in res.included_set
    assert "MuoniumHighTF" not in res.included_set


def test_resolve_for_datasets_unions_geometries():
    tf = _fake_dataset("TF")
    zf = _fake_dataset("ZF")
    res = resolve_scope_for_datasets([tf, zf], WizardScope())
    # VortexLattice (TF-only superconductivity) in scope for the TF run.
    assert "VortexLattice" in res.included_set
    # FmuF_Linear (ZF-only molecular) in scope for the ZF run.
    assert "FmuF_Linear" in res.included_set


def test_resolve_for_datasets_excluded_in_all_keeps_all_runs_reason():
    freq_names = _frequency_component_names()
    assert freq_names
    tf = _fake_dataset("TF")
    zf = _fake_dataset("ZF")
    res = resolve_scope_for_datasets([tf, zf], WizardScope())
    excluded = {e.name: e.reason for e in res.excluded_components}
    for name in freq_names:
        assert name in excluded
        assert excluded[name].startswith("all runs: ")


def test_resolve_for_datasets_notes_come_from_a_run_that_records_geometry():
    res = resolve_scope_for_datasets([_fake_dataset(""), _fake_dataset("LF")], WizardScope())
    assert "longitudinal field" in res.inference_note


# --- describe_scope -------------------------------------------------------


def test_describe_scope_summarises_the_recorded_geometry():
    view = describe_scope(
        [_fake_dataset("LF"), _fake_dataset("LF"), _fake_dataset("")], WizardScope()
    )
    assert view.geometry.counts == ((FieldGeometry.LF, 2),)
    assert view.geometry.answered == ()
    assert view.geometry.unrecorded == 1
    assert view.geometry.editable
    assert view.geometry.answer is None


def test_describe_scope_direction_is_read_only_when_every_run_records_one():
    view = describe_scope([_fake_dataset("ZF"), _fake_dataset("TF")], WizardScope())
    assert view.geometry.counts == ((FieldGeometry.ZF, 1), (FieldGeometry.TF, 1))
    assert not view.geometry.editable


def test_describe_scope_lists_families_in_display_order_with_every_time_component():
    view = describe_scope([_fake_dataset("ZF")], WizardScope())
    assert tuple(family.title for family in view.families) == FAMILY_ORDER
    names = [c.name for family in view.families for c in family.components]
    assert sorted(names) == sorted(_time_component_names())


def test_describe_scope_lists_a_component_outside_the_geometry_with_its_reason():
    view = describe_scope([_fake_dataset("LF", field=100.0)], WizardScope())
    components = {c.name: c for family in view.families for c in family.components}
    static = components["StaticGKT_ZF"]
    assert not static.applies
    assert not static.included
    assert "geometry" in static.reason
    dynamic = components["DynamicGaussianKT"]
    assert dynamic.applies and dynamic.included and dynamic.reason == ""
    assert dynamic.label == "Dynamic Gaussian KT"
    assert dynamic.use_when == "Fluctuating Gaussian fields (strong collision)"
    assert dynamic.slow
    assert dynamic.geometries == {FieldGeometry.LF, FieldGeometry.ZF}


def test_describe_scope_physics_exclusion_still_applies():
    view = describe_scope([_fake_dataset("ZF")], WizardScope(physics=MOLECULAR))
    components = {c.name: c for family in view.families for c in family.components}
    oscillatory = components["Oscillatory"]
    assert oscillatory.applies and not oscillatory.included
    assert "magnetism" in oscillatory.reason


def test_describe_scope_counts_follow_inclusion():
    datasets = [_fake_dataset("ZF")]
    everything = describe_scope(datasets, WizardScope())
    fast = describe_scope(datasets, WizardScope(skip_slow=True))
    assert everything.slow_included
    assert all(c.slow and c.included for c in everything.slow_included)
    assert fast.slow_included == ()
    assert fast.included_count == everything.included_count - len(everything.slow_included)
    assert everything.included_count == len(
        resolve_scope_for_datasets(datasets, WizardScope()).included_components
    )


def test_describe_scope_notes_match_the_resolution():
    datasets = [_fake_dataset("TF", field=20.0, title="CaF2")]
    view = describe_scope(datasets, WizardScope())
    assert view.notes == resolve_scope_for_datasets(datasets, WizardScope()).notes
    assert any("fluorine" in note for note in view.notes)
    assert any("muonium" in note for note in view.notes)


def test_describe_scope_puts_user_functions_last(throwaway_user_component):
    view = describe_scope([_fake_dataset("ZF")], WizardScope())
    assert view.families[-1].title == USER_FAMILY_TITLE
    (component,) = view.families[-1].components
    assert component.name == throwaway_user_component
    assert component.included


# --- the user's field-direction answer (D4) ---------------------------------


def _run_dataset(**metadata: str) -> MuonDataset:
    """A dataset whose run holds its own copy of the metadata, as a loader builds it."""
    return MuonDataset(
        time=np.linspace(0.0, 8.0, 4),
        asymmetry=np.zeros(4),
        error=np.ones(4),
        metadata=dict(metadata),
        run=Run(run_number=1, metadata=dict(metadata)),
    )


def _direction_keys(metadata: dict) -> dict:
    return {k: metadata[k] for k in ("field_direction", "field_direction_source") if k in metadata}


USER_LF = {"field_direction": "Longitudinal", "field_direction_source": "user"}


def test_user_direction_fills_a_run_that_records_none_on_dataset_and_run():
    dataset = _run_dataset(field_direction="")
    set_user_field_direction([dataset], FieldGeometry.LF)
    assert _direction_keys(dataset.metadata) == USER_LF
    assert _direction_keys(dataset.run.metadata) == USER_LF
    assert dataset_field_geometry(dataset) is FieldGeometry.LF


@pytest.mark.parametrize(
    "recorded",
    [
        {"field_direction": "Transverse"},
        {"field_state": "TF"},
        {"field_direction": "Transverse", "field_direction_source": "icp_log"},
    ],
)
def test_user_direction_never_touches_a_direction_the_file_records(recorded):
    dataset = _run_dataset(**recorded)
    for geometry in (FieldGeometry.LF, None):
        set_user_field_direction([dataset], geometry)
        assert dataset.metadata == recorded
        assert dataset.run.metadata == recorded


def test_user_direction_replaces_its_own_earlier_answer():
    dataset = _run_dataset()
    set_user_field_direction([dataset], FieldGeometry.LF)
    set_user_field_direction([dataset], FieldGeometry.ZF)
    expected = {"field_direction": "Zero field", "field_direction_source": "user"}
    assert _direction_keys(dataset.metadata) == expected
    assert _direction_keys(dataset.run.metadata) == expected


def test_not_recorded_withdraws_only_the_user_answer():
    answered = _run_dataset(field_state="")
    untouched = _run_dataset(field_direction="")
    recorded = _run_dataset(field_state="ZF")
    set_user_field_direction([answered], FieldGeometry.TF)
    set_user_field_direction([answered, untouched, recorded], None)
    assert answered.metadata == {"field_state": ""} == answered.run.metadata
    assert untouched.metadata == {"field_direction": ""} == untouched.run.metadata
    assert recorded.metadata == {"field_state": "ZF"} == recorded.run.metadata
    assert dataset_field_geometry(answered) is None


def test_user_direction_answers_a_mixed_series_only_where_the_file_is_silent():
    silent, recorded = _run_dataset(), _run_dataset(field_state="LF")
    set_user_field_direction([silent, recorded], FieldGeometry.TF)
    assert dataset_field_geometry(silent) is FieldGeometry.TF
    assert recorded.metadata == {"field_state": "LF"}


def test_user_direction_on_a_dataset_sharing_its_runs_metadata():
    dataset = _run_dataset()
    dataset.run.metadata = dataset.metadata
    set_user_field_direction([dataset], FieldGeometry.LF)
    assert _direction_keys(dataset.metadata) == USER_LF
    set_user_field_direction([dataset], None)
    assert dataset.metadata == {}


def test_user_direction_on_a_dataset_without_a_run():
    dataset = _fake_dataset("")
    set_user_field_direction([dataset], FieldGeometry.ZF)
    assert dataset_field_geometry(dataset) is FieldGeometry.ZF


def test_user_direction_overrides_carry_only_the_users_answer():
    answered, recorded = _run_dataset(), _run_dataset(field_direction="Transverse")
    set_user_field_direction([answered, recorded], FieldGeometry.LF)
    assert user_field_direction_overrides(answered) == USER_LF
    assert user_field_direction_overrides(recorded) == {}


def test_describe_scope_counts_a_user_answer_apart_from_the_files():
    runs = [_run_dataset(field_state="LF"), _run_dataset(), _run_dataset()]
    set_user_field_direction(runs, FieldGeometry.LF)
    geometry = describe_scope(runs, WizardScope()).geometry
    assert geometry.counts == ((FieldGeometry.LF, 1),)
    assert geometry.answered == ((FieldGeometry.LF, 2),)
    assert geometry.unrecorded == 2
    assert geometry.editable
    assert geometry.answer is FieldGeometry.LF


def test_describe_scope_has_no_single_answer_while_a_silent_run_is_unanswered():
    answered = _run_dataset()
    set_user_field_direction([answered], FieldGeometry.TF)
    geometry = describe_scope([answered, _run_dataset()], WizardScope()).geometry
    assert geometry.answered == ((FieldGeometry.TF, 1),)
    assert geometry.unrecorded == 2
    assert geometry.answer is None


def test_describe_scope_has_no_single_answer_when_answers_differ():
    first, second = _run_dataset(), _run_dataset()
    set_user_field_direction([first], FieldGeometry.TF)
    set_user_field_direction([second], FieldGeometry.ZF)
    geometry = describe_scope([first, second], WizardScope()).geometry
    assert geometry.answered == ((FieldGeometry.ZF, 1), (FieldGeometry.TF, 1))
    assert geometry.answer is None


def test_a_user_answer_scopes_the_screen_like_a_recorded_one():
    answered = _run_dataset()
    set_user_field_direction([answered], FieldGeometry.LF)
    assert resolve_scope_for_dataset(answered, WizardScope()).query.geometries == {FieldGeometry.LF}


# --- effort tier (PR 5) ---------------------------------------------------


def test_effort_tier_default_is_exhaustive():
    # PR 5 rework: every tier resolves to the exact engine, so the persisted/
    # legacy-restore default is the exact (Exhaustive) tier — the single honest
    # user-facing optimisation mode.
    assert DEFAULT_EFFORT_TIER is EffortTier.EXHAUSTIVE


def test_effort_tier_has_four_values():
    assert {tier.value for tier in EffortTier} == {"low", "balanced", "thorough", "exhaustive"}


def test_effort_tier_every_value_has_a_label_and_description():
    for tier in EffortTier:
        assert EFFORT_TIER_LABELS[tier]
        assert EFFORT_TIER_DESCRIPTIONS[tier]


def test_low_label_says_screening_grade():
    assert "screening-grade" in EFFORT_TIER_LABELS[EffortTier.LOW].lower()


@pytest.mark.parametrize("tier", list(EffortTier))
def test_effort_tier_payload_round_trips(tier: EffortTier):
    payload = effort_tier_to_payload(tier)
    assert isinstance(payload, str)
    assert effort_tier_from_payload(payload) is tier


def test_effort_tier_from_payload_tolerates_garbage():
    assert effort_tier_from_payload(None) is DEFAULT_EFFORT_TIER
    assert effort_tier_from_payload("not-a-tier") is DEFAULT_EFFORT_TIER
    assert effort_tier_from_payload(123) is DEFAULT_EFFORT_TIER
    assert effort_tier_from_payload({}) is DEFAULT_EFFORT_TIER


def test_effort_tier_from_payload_accepts_enum_member_directly():
    assert effort_tier_from_payload(EffortTier.LOW) is EffortTier.LOW

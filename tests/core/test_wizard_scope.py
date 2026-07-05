"""Tests for the fit-wizard scoping module."""

from __future__ import annotations

import json

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import COMPONENTS
from asymmetry.core.fitting.user_functions import register_component
from asymmetry.core.fitting.wizard_scope import (
    DEFAULT_EFFORT_TIER,
    EFFORT_TIER_DESCRIPTIONS,
    EFFORT_TIER_LABELS,
    MUONIUM_HIGH_TF_MIN_GAUSS,
    MUONIUM_LOW_TF_MAX_GAUSS,
    ZERO_FIELD_MAX_GAUSS,
    EffortTier,
    ExcludedComponent,
    ScopeResolution,
    WizardScope,
    WizardScopePreset,
    effort_tier_from_payload,
    effort_tier_to_payload,
    estimate_screening_cost,
    infer_auto_query,
    resolve_scope,
    resolve_scope_for_dataset,
    resolve_scope_for_datasets,
)


def _time_component_names() -> set[str]:
    return {n for n, d in COMPONENTS.items() if d.domain == "time"}


def _frequency_component_names() -> set[str]:
    return {n for n, d in COMPONENTS.items() if d.domain == "frequency"}


def _resolve(preset: WizardScopePreset, **kw) -> ScopeResolution:
    return resolve_scope(WizardScope(preset=preset), **kw)


# --- named presets ------------------------------------------------------


def test_zf_static_magnetism_preset_membership():
    res = _resolve(WizardScopePreset.ZF_STATIC_MAGNETISM)
    assert "Exponential" in res.included_set
    assert "Constant" in res.included_set
    # A TF-only muonium form is excluded with a geometry reason.
    assert "MuoniumTF" not in res.included_set
    reason = next(e.reason for e in res.excluded_components if e.name == "MuoniumTF")
    assert "geometr" in reason.lower()
    # A TF superconductivity component is excluded.
    assert "VortexLattice" not in res.included_set


def test_tf_superconductor_includes_vortex_lattice():
    res = _resolve(WizardScopePreset.TF_SUPERCONDUCTOR)
    assert "VortexLattice" in res.included_set
    assert "Exponential" in res.included_set
    assert "Constant" in res.included_set


def test_fluoride_fmuf_preset_membership():
    res = _resolve(WizardScopePreset.FLUORIDE_FMUF)
    assert "FmuF_Linear" in res.included_set
    assert "Exponential" in res.included_set
    assert "Constant" in res.included_set
    # Oscillatory is MAGNETISM, outside the molecular preset.
    assert "Oscillatory" not in res.included_set


def test_all_preset_includes_every_time_domain_component():
    res = _resolve(WizardScopePreset.ALL)
    assert res.included_set == _time_component_names()


@pytest.mark.parametrize(
    "preset",
    [p for p in WizardScopePreset if p is not WizardScopePreset.AUTO],
)
def test_every_preset_includes_exponential_and_constant(preset):
    res = _resolve(preset)
    assert "Exponential" in res.included_set
    assert "Constant" in res.included_set


def test_frequency_domain_components_excluded_everywhere():
    freq = _frequency_component_names()
    assert freq  # sanity: there are frequency-domain components
    for preset in WizardScopePreset:
        if preset is WizardScopePreset.AUTO:
            res = resolve_scope(WizardScope(preset=preset))
        else:
            res = _resolve(preset)
        assert not (res.included_set & freq), preset
        excluded_names = {e.name for e in res.excluded_components}
        assert freq <= excluded_names, preset


# --- Auto inference -----------------------------------------------------


@pytest.mark.parametrize(
    ("field_direction", "expected"),
    [
        ("Transverse", WizardScopePreset.TF_KNIGHT_PRECESSION),
        ("Longitudinal", WizardScopePreset.LF_DYNAMICS),
        ("Zero field", WizardScopePreset.ZF_STATIC_MAGNETISM),
        ("TF", WizardScopePreset.TF_KNIGHT_PRECESSION),
        ("LF", WizardScopePreset.LF_DYNAMICS),
        ("ZF", WizardScopePreset.ZF_STATIC_MAGNETISM),
        ("", WizardScopePreset.ALL),
    ],
)
def test_auto_effective_preset(field_direction, expected):
    res = resolve_scope(WizardScope(), field_direction=field_direction)
    assert res.effective_preset is expected


def test_auto_note_names_geometry_source():
    res = resolve_scope(WizardScope(), field_direction="Zero field")
    assert "zero field" in res.inference_note.lower()


def test_auto_tf_low_field_regime():
    res = resolve_scope(WizardScope(), field_direction="TF", field_gauss=20.0)
    assert "MuoniumLowTF" in res.included_set
    assert "MuoniumHighTF" not in res.included_set
    assert "MuoniumHighTFAniso" not in res.included_set
    # MuoniumTF (exact four-frequency form) is never field-excluded.
    assert "MuoniumTF" in res.included_set
    high_reason = next(e.reason for e in res.excluded_components if e.name == "MuoniumHighTF")
    assert str(int(MUONIUM_HIGH_TF_MIN_GAUSS)) in high_reason or "20" in high_reason


def test_auto_tf_high_field_regime():
    res = resolve_scope(WizardScope(), field_direction="TF", field_gauss=3000.0)
    assert "MuoniumHighTF" in res.included_set
    assert "MuoniumHighTFAniso" in res.included_set
    assert "MuoniumLowTF" not in res.included_set
    assert "MuoniumTF" in res.included_set
    low_reason = next(e.reason for e in res.excluded_components if e.name == "MuoniumLowTF")
    assert str(int(MUONIUM_LOW_TF_MAX_GAUSS)) in low_reason or "3000" in low_reason


def test_auto_tf_unknown_field_excludes_no_muonium_regime():
    res = resolve_scope(WizardScope(), field_direction="TF", field_gauss=None)
    assert "MuoniumLowTF" in res.included_set
    assert "MuoniumHighTF" in res.included_set
    assert "MuoniumHighTFAniso" in res.included_set


def test_auto_unknown_geometry_is_superset_of_zf_preset():
    unknown = resolve_scope(WizardScope(), field_direction="")
    zf = _resolve(WizardScopePreset.ZF_STATIC_MAGNETISM)
    assert zf.included_set <= unknown.included_set
    # And it equals the ALL query set (every time-domain component).
    assert unknown.included_set == _time_component_names()


# --- B≈0 geometry-label override -----------------------------------------
#
# Real ISIS runs are sometimes recorded "TF" with the applied-field setpoint
# at (or near) zero — a ZF measurement on a TF-capable beamline (see
# docs/porting/field-geometry/: "MUSR00044991.nxs: magnetic_field_state='TF'
# at magnetic_field=0 G"). Auto must widen to include ZF families in that case
# without dropping the labelled TF family, which may still be hardware-correct.


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
    zf_only = _resolve(WizardScopePreset.ZF_STATIC_MAGNETISM)
    lf_only = _resolve(WizardScopePreset.LF_DYNAMICS)
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
    # not further change Auto's ZF behaviour one way or the other.
    with_field = resolve_scope(WizardScope(), field_direction="ZF", field_gauss=0.0)
    without_field = resolve_scope(WizardScope(), field_direction="ZF", field_gauss=None)
    assert with_field.included_set == without_field.included_set
    assert with_field.effective_preset is WizardScopePreset.ZF_STATIC_MAGNETISM


def test_zero_field_override_via_dataset_wrapper():
    dataset = _fake_dataset("TF", field=0.0)
    res = resolve_scope_for_dataset(dataset, WizardScope())
    assert "FmuF_Linear" in res.included_set
    assert res.effective_preset is WizardScopePreset.TF_KNIGHT_PRECESSION


# --- fluorine sniff -----------------------------------------------------


@pytest.mark.parametrize("sample", ["PbF2", "CaF2", "LiF", "NaF"])
def test_fluorine_sniff_positive(sample):
    _, _, note, _ = infer_auto_query("Zero field", None, sample)
    assert "fluorine" in note.lower()


@pytest.mark.parametrize("sample", ["Fe", "FeSe", "Fer", ""])
def test_fluorine_sniff_negative(sample):
    _, _, note, _ = infer_auto_query("Zero field", None, sample)
    assert "fluorine" not in note.lower()


# --- overrides ----------------------------------------------------------


def test_include_resurrects_query_excluded_component():
    scope = WizardScope(
        preset=WizardScopePreset.ZF_STATIC_MAGNETISM,
        include_components=frozenset({"VortexLattice"}),
    )
    res = resolve_scope(scope)
    assert "VortexLattice" in res.included_set
    assert "VortexLattice" not in {e.name for e in res.excluded_components}


def test_exclude_beats_include_for_same_name():
    scope = WizardScope(
        preset=WizardScopePreset.ALL,
        include_components=frozenset({"Exponential"}),
        exclude_components=frozenset({"Exponential"}),
    )
    res = resolve_scope(scope)
    assert "Exponential" not in res.included_set
    reason = next(e.reason for e in res.excluded_components if e.name == "Exponential")
    assert reason == "excluded by user"


def test_unknown_override_names_are_noted_not_crashing():
    scope = WizardScope(
        preset=WizardScopePreset.ALL,
        include_components=frozenset({"NoSuchComponent"}),
        exclude_components=frozenset({"AlsoMissing"}),
    )
    res = resolve_scope(scope)
    assert "NoSuchComponent" in res.inference_note
    assert "AlsoMissing" in res.inference_note


# --- exclusion reason quality -------------------------------------------


@pytest.mark.parametrize(
    "preset",
    [WizardScopePreset.ZF_STATIC_MAGNETISM, WizardScopePreset.TF_SUPERCONDUCTOR],
)
def test_every_exclusion_has_a_specific_nonempty_reason(preset):
    res = _resolve(preset)
    assert res.excluded_components
    for exc in res.excluded_components:
        assert isinstance(exc, ExcludedComponent)
        assert exc.reason.strip()


# --- payload round-trip -------------------------------------------------


def test_payload_round_trip_and_json_safe():
    scope = WizardScope(
        preset=WizardScopePreset.LF_DYNAMICS,
        include_components=frozenset({"VortexLattice"}),
        exclude_components=frozenset({"Exponential"}),
    )
    payload = scope.to_payload()
    json.dumps(payload)  # must not raise
    restored = WizardScope.from_payload(payload)
    assert restored == scope


@pytest.mark.parametrize("garbage", [None, 42, "nonsense", [], {"preset": "made-up"}])
def test_from_payload_tolerates_garbage(garbage):
    assert WizardScope.from_payload(garbage).preset is WizardScopePreset.AUTO


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


def test_user_component_appears_in_every_preset(throwaway_user_component):
    name = throwaway_user_component
    for preset in WizardScopePreset:
        if preset is WizardScopePreset.AUTO:
            res = resolve_scope(WizardScope(preset=preset), field_direction="Zero field")
        else:
            res = _resolve(preset)
        assert name in res.included_set, preset


def test_user_component_can_still_be_excluded(throwaway_user_component):
    name = throwaway_user_component
    scope = WizardScope(
        preset=WizardScopePreset.ZF_STATIC_MAGNETISM,
        exclude_components=frozenset({name}),
    )
    res = resolve_scope(scope)
    assert name not in res.included_set


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
    assert res.effective_preset is WizardScopePreset.TF_KNIGHT_PRECESSION
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


# --- screening cost estimate --------------------------------------------


def test_estimate_screening_cost_positive_and_ordered():
    all_res = _resolve(WizardScopePreset.ALL)
    zf_res = _resolve(WizardScopePreset.ZF_STATIC_MAGNETISM)
    all_candidates, all_fits = estimate_screening_cost(all_res)
    zf_candidates, zf_fits = estimate_screening_cost(zf_res)
    assert all_candidates > 0 and all_fits > 0
    assert zf_candidates > 0 and zf_fits > 0
    assert all_candidates > zf_candidates
    assert all_fits > zf_fits


# --- effort tier (PR 5) ---------------------------------------------------


def test_effort_tier_default_is_balanced():
    assert DEFAULT_EFFORT_TIER is EffortTier.BALANCED


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

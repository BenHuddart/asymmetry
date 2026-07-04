"""Round-trip and migration tests for :class:`GlobalFitStudy`.

Pure-core: no Qt. Covers full to_dict/from_dict round trips (including xerr
groups and correlations), tolerant/defensive from_dict behaviour, the
group-input digest's staleness semantics (order-independence, sensitivity to
data changes), and the legacy single-slot payload migration helper.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.fitting.parameter_models import (
    CrossGroupFitResult,
    ParameterCompositeModel,
    ParameterGroupData,
    global_fit_parameter_model,
)
from asymmetry.core.representation.global_fit_study import (
    GlobalFitStudy,
    compute_group_input_digest,
    study_from_legacy_cross_group_payload,
)


def _groups(*, with_xerr: bool = False, seed: int = 3) -> list[ParameterGroupData]:
    rng = np.random.default_rng(seed)
    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    groups = []
    for idx, b in enumerate([1.0, -0.5, 3.0]):
        y = 2.0 * x + b + rng.normal(scale=0.01, size=x.shape)
        groups.append(
            ParameterGroupData(
                group_id=f"g{idx}",
                group_name=f"G{idx}",
                x=x,
                y=y,
                yerr=np.full_like(x, 0.05),
                group_variable_value=float(idx),
                xerr=np.full_like(x, 0.02) if with_xerr else None,
            )
        )
    return groups


def _fit_result(groups: list[ParameterGroupData]) -> CrossGroupFitResult:
    model = ParameterCompositeModel(["Linear"])
    return global_fit_parameter_model(
        groups=groups,
        model=model,
        global_params=["m"],
        local_params=["b"],
        fixed_params={},
        initial_params={"m": 1.0, "b": 0.0},
    )


def _study(
    *,
    study_id: str = "modelfit-abc123",
    with_xerr: bool = False,
    seed: int = 3,
) -> GlobalFitStudy:
    groups = _groups(with_xerr=with_xerr, seed=seed)
    model = ParameterCompositeModel(["Linear"])
    result = _fit_result(groups)
    assert result.success
    return GlobalFitStudy(
        study_id=study_id,
        name="lambda(B) at 8 temperatures",
        parameter_name="lambda",
        x_key="field",
        x_label="Field (G)",
        group_variable_key="temperature",
        group_variable_label="Temperature (K)",
        created="2026-07-04T00:00:00",
        updated="2026-07-04T00:00:00",
        source_group_ids=["series-1", "series-2"],
        groups=groups,
        model=model,
        config={"error_mode": "column", "roles": {"m": "global", "b": "local"}},
        result=result,
        fit_x_min=0.0,
        fit_x_max=3.0,
        input_digest=compute_group_input_digest(groups),
    )


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


def test_round_trip_basic_study() -> None:
    study = _study()
    restored = GlobalFitStudy.from_dict(study.to_dict())

    assert restored is not None
    assert restored.study_id == study.study_id
    assert restored.name == study.name
    assert restored.parameter_name == study.parameter_name
    assert restored.x_key == study.x_key
    assert restored.x_label == study.x_label
    assert restored.group_variable_key == study.group_variable_key
    assert restored.group_variable_label == study.group_variable_label
    assert restored.created == study.created
    assert restored.updated == study.updated
    assert restored.source_group_ids == study.source_group_ids
    assert restored.config == study.config
    assert restored.input_digest == study.input_digest
    assert restored.fit_x_min == pytest.approx(study.fit_x_min)
    assert restored.fit_x_max == pytest.approx(study.fit_x_max)

    assert len(restored.groups) == len(study.groups)
    for orig, back in zip(study.groups, restored.groups, strict=True):
        assert back.group_id == orig.group_id
        np.testing.assert_allclose(back.x, orig.x)
        np.testing.assert_allclose(back.y, orig.y)

    assert restored.model is not None
    assert restored.model.to_dict() == study.model.to_dict()

    assert restored.result is not None
    assert restored.result.success == study.result.success
    assert restored.result.chi_squared == pytest.approx(study.result.chi_squared)


def test_round_trip_study_with_xerr_groups() -> None:
    study = _study(study_id="modelfit-xerr", with_xerr=True, seed=11)
    restored = GlobalFitStudy.from_dict(study.to_dict())

    assert restored is not None
    for orig, back in zip(study.groups, restored.groups, strict=True):
        assert back.xerr is not None
        np.testing.assert_allclose(back.xerr, orig.xerr)


def test_round_trip_study_with_correlations_in_result() -> None:
    groups = _groups(seed=5)
    model = ParameterCompositeModel(["Linear"])
    result = global_fit_parameter_model(
        groups=groups,
        model=model,
        global_params=["m", "b"],
        local_params=[],
        fixed_params={},
        initial_params={"m": 1.0, "b": 0.0},
    )
    assert result.success
    assert result.global_correlations is not None

    study = GlobalFitStudy(
        study_id="modelfit-corr",
        name="Global m/b fit",
        parameter_name="lambda",
        x_key="field",
        x_label="Field (G)",
        group_variable_key="temperature",
        group_variable_label="Temperature (K)",
        created="2026-07-04T00:00:00",
        updated="2026-07-04T00:00:00",
        groups=groups,
        model=model,
        result=result,
        input_digest=compute_group_input_digest(groups),
    )

    restored = GlobalFitStudy.from_dict(study.to_dict())
    assert restored is not None
    assert restored.result.global_correlations is not None
    names, matrix = result.global_correlations
    restored_names, restored_matrix = restored.result.global_correlations
    assert restored_names == names
    np.testing.assert_allclose(np.array(restored_matrix), np.array(matrix))


def test_two_studies_are_independent_round_trips() -> None:
    """Two distinct studies (as would be kept side by side in a project) each
    round trip to their own values without cross-contamination."""
    study_a = _study(study_id="modelfit-a", seed=1)
    study_b = _study(study_id="modelfit-b", with_xerr=True, seed=2)

    restored_a = GlobalFitStudy.from_dict(study_a.to_dict())
    restored_b = GlobalFitStudy.from_dict(study_b.to_dict())

    assert restored_a is not None
    assert restored_b is not None
    assert restored_a.study_id == "modelfit-a"
    assert restored_b.study_id == "modelfit-b"
    assert restored_a.input_digest != restored_b.input_digest


# ---------------------------------------------------------------------------
# Tolerant / defensive from_dict
# ---------------------------------------------------------------------------


def test_from_dict_returns_none_on_missing_result() -> None:
    study = _study()
    data = study.to_dict()
    data["result"] = None
    assert GlobalFitStudy.from_dict(data) is None


def test_from_dict_returns_none_on_missing_model() -> None:
    study = _study()
    data = study.to_dict()
    data["model"] = None
    assert GlobalFitStudy.from_dict(data) is None


def test_from_dict_returns_none_on_non_dict_payload() -> None:
    assert GlobalFitStudy.from_dict(None) is None  # type: ignore[arg-type]
    assert GlobalFitStudy.from_dict([]) is None  # type: ignore[arg-type]


def test_from_dict_returns_none_on_malformed_model() -> None:
    study = _study()
    data = study.to_dict()
    data["model"] = {"component_names": ["NotAComponent"]}
    assert GlobalFitStudy.from_dict(data) is None


def test_from_dict_skips_malformed_group_entries() -> None:
    study = _study()
    data = study.to_dict()
    data["groups"].append("not-a-dict")
    data["groups"].append({"group_id": "gX"})  # missing x/y/yerr -> tolerant defaults

    restored = GlobalFitStudy.from_dict(data)
    assert restored is not None
    # The junk string was dropped; the sparse-but-dict entry survives via
    # ParameterGroupData.from_dict's own tolerant defaults.
    assert len(restored.groups) == len(study.groups) + 1
    assert restored.groups[-1].group_id == "gX"
    assert restored.groups[-1].x.size == 0


def test_from_dict_defaults_missing_optional_keys() -> None:
    study = _study()
    data = study.to_dict()
    for key in (
        "source_group_ids",
        "config",
        "x_label",
        "group_variable_key",
        "group_variable_label",
        "created",
        "updated",
        "input_digest",
    ):
        data.pop(key, None)

    restored = GlobalFitStudy.from_dict(data)
    assert restored is not None
    assert restored.source_group_ids == []
    assert restored.config == {}
    assert restored.x_label == ""
    assert restored.group_variable_key == ""
    assert restored.group_variable_label == ""
    assert restored.created == ""
    assert restored.updated == ""
    assert restored.input_digest == ""


def test_from_dict_defaults_missing_fit_x_bounds_to_nan() -> None:
    study = _study()
    data = study.to_dict()
    data.pop("fit_x_min", None)
    data.pop("fit_x_max", None)

    restored = GlobalFitStudy.from_dict(data)
    assert restored is not None
    assert np.isnan(restored.fit_x_min)
    assert np.isnan(restored.fit_x_max)


# ---------------------------------------------------------------------------
# Digest semantics
# ---------------------------------------------------------------------------


def test_digest_identical_data_is_identical() -> None:
    groups_a = _groups(seed=42)
    groups_b = _groups(seed=42)
    assert compute_group_input_digest(groups_a) == compute_group_input_digest(groups_b)


def test_digest_changes_when_a_single_y_value_changes() -> None:
    groups = _groups(seed=42)
    baseline = compute_group_input_digest(groups)

    mutated = [
        ParameterGroupData(
            group_id=g.group_id,
            group_name=g.group_name,
            x=g.x.copy(),
            y=g.y.copy(),
            yerr=g.yerr.copy(),
            group_variable_value=g.group_variable_value,
            xerr=None if g.xerr is None else g.xerr.copy(),
        )
        for g in groups
    ]
    mutated[0].y[0] += 1.0

    assert compute_group_input_digest(mutated) != baseline


def test_digest_is_order_independent() -> None:
    """PINNED behaviour: groups are sorted by group_id before hashing, so
    permuting the input list (e.g. a different multi-select order) does not
    change the digest -- only content changes should register as stale."""
    groups = _groups(seed=7)
    forward = compute_group_input_digest(groups)
    reversed_order = compute_group_input_digest(list(reversed(groups)))
    assert forward == reversed_order

    import itertools

    for perm in itertools.permutations(groups):
        assert compute_group_input_digest(list(perm)) == forward


def test_digest_changes_when_group_variable_value_changes() -> None:
    groups = _groups(seed=9)
    baseline = compute_group_input_digest(groups)
    mutated = list(groups)
    mutated[0] = ParameterGroupData(
        group_id=mutated[0].group_id,
        group_name=mutated[0].group_name,
        x=mutated[0].x,
        y=mutated[0].y,
        yerr=mutated[0].yerr,
        group_variable_value=mutated[0].group_variable_value + 100.0,
    )
    assert compute_group_input_digest(mutated) != baseline


# ---------------------------------------------------------------------------
# Legacy migration helper
# ---------------------------------------------------------------------------


def _legacy_payload() -> dict:
    """Mirror the on-disk shape written by
    ``FitParametersPanel._serialize_last_cross_group_fit`` in
    ``src/asymmetry/gui/panels/fit_parameters_panel.py`` (lines ~4198-4286 as
    of this writing). Notably: group entries have no ``xerr`` key, and
    ``fit_result`` carries only the pre-Phase-1A subset of
    ``CrossGroupFitResult`` fields (no error_mode/n_points/per_group_*/
    global_correlations)."""
    return {
        "parameter_name": "lambda",
        "x_key": "field",
        "fit_x_min": 0.0,
        "fit_x_max": 3.0,
        "config": {"error_mode": "column"},
        "config_key": "lambda|field|g0,g1",
        "groups": [
            {
                "group_id": "g0",
                "group_name": "G0",
                "x": [0.0, 1.0, 2.0, 3.0],
                "y": [1.0, 3.0, 5.0, 7.0],
                "yerr": [0.05, 0.05, 0.05, 0.05],
                "group_variable_value": 10.0,
            },
            {
                "group_id": "g1",
                "group_name": "G1",
                "x": [0.0, 1.0, 2.0, 3.0],
                "y": [-0.5, 1.5, 3.5, 5.5],
                "yerr": [0.05, 0.05, 0.05, 0.05],
                "group_variable_value": 20.0,
            },
        ],
        "model": ParameterCompositeModel(["Linear"]).to_dict(),
        "fit_result": {
            "success": True,
            "chi_squared": 1.2,
            "reduced_chi_squared": 0.3,
            "message": "Fit successful",
            "global_parameters": [
                {"name": "m", "value": 2.0, "min": -1e300, "max": 1e300, "fixed": False}
            ],
            "global_uncertainties": {"m": 0.01},
            "local_parameters": {
                "g0": [{"name": "b", "value": 1.0, "min": -1e300, "max": 1e300, "fixed": False}],
                "g1": [{"name": "b", "value": -0.5, "min": -1e300, "max": 1e300, "fixed": False}],
            },
            "fixed_parameters": [],
            "local_uncertainties": {"g0": {"b": 0.02}, "g1": {"b": 0.02}},
        },
    }


def test_migration_helper_builds_valid_study_from_legacy_payload() -> None:
    payload = _legacy_payload()
    study = study_from_legacy_cross_group_payload(
        payload,
        study_id="modelfit-legacy1",
        name="lambda",
        created="2026-07-04T00:00:00",
    )

    assert study is not None
    assert study.study_id == "modelfit-legacy1"
    assert study.name == "lambda"
    assert study.parameter_name == "lambda"
    assert study.x_key == "field"
    # No stored label in the legacy shape -- defaults to the bare x_key.
    assert study.x_label == "field"
    assert study.group_variable_key == ""
    assert study.group_variable_label == ""
    assert study.source_group_ids == []
    assert study.fit_x_min == pytest.approx(0.0)
    assert study.fit_x_max == pytest.approx(3.0)
    assert study.config == {"error_mode": "column"}

    assert len(study.groups) == 2
    assert {g.group_id for g in study.groups} == {"g0", "g1"}
    g0 = next(g for g in study.groups if g.group_id == "g0")
    np.testing.assert_allclose(g0.y, [1.0, 3.0, 5.0, 7.0])
    assert g0.xerr is None

    assert study.model is not None
    assert study.model.component_names == ["Linear"]

    assert study.result is not None
    assert study.result.success is True
    assert study.result.chi_squared == pytest.approx(1.2)
    # Phase-1A-only fields are defaulted by CrossGroupFitResult.from_dict.
    assert study.result.per_group_chi_squared == {}
    assert study.result.global_correlations is None

    # The digest is computed fresh over the migrated groups.
    assert study.input_digest == compute_group_input_digest(study.groups)

    # The migrated study itself must round trip cleanly going forward.
    restored = GlobalFitStudy.from_dict(study.to_dict())
    assert restored is not None
    assert restored.study_id == study.study_id


def test_migration_helper_returns_none_on_missing_model() -> None:
    payload = _legacy_payload()
    payload["model"] = None
    assert (
        study_from_legacy_cross_group_payload(
            payload, study_id="x", name="x", created="2026-07-04T00:00:00"
        )
        is None
    )


def test_migration_helper_returns_none_on_missing_fit_result() -> None:
    payload = _legacy_payload()
    payload.pop("fit_result")
    assert (
        study_from_legacy_cross_group_payload(
            payload, study_id="x", name="x", created="2026-07-04T00:00:00"
        )
        is None
    )


def test_migration_helper_returns_none_on_non_dict_payload() -> None:
    assert (
        study_from_legacy_cross_group_payload(
            None,
            study_id="x",
            name="x",
            created="2026-07-04T00:00:00",  # type: ignore[arg-type]
        )
        is None
    )


def test_migration_helper_tolerates_malformed_group_entry() -> None:
    payload = _legacy_payload()
    payload["groups"].append("not-a-dict")
    study = study_from_legacy_cross_group_payload(
        payload, study_id="x", name="x", created="2026-07-04T00:00:00"
    )
    assert study is not None
    assert len(study.groups) == 2  # junk entry dropped

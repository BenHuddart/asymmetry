"""Unit tests for :class:`JointFit`, its series stamp, and its ``ProjectModel``
cascade (Phase 2 of ``docs/plans/joint-fit.md``).

Pure-core: no Qt. Covers ``to_dict``/``from_dict`` round trips (including the
tolerant file-boundary behaviour on a malformed entry), the D10 delete
cascades (deleting a member, deleting the record itself), the D9 staleness
verdict, and that stamping a series never changes its ``recipe_identity``.
"""

from __future__ import annotations

from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.representation import RepresentationType
from asymmetry.core.representation.group import DataGroup
from asymmetry.core.representation.joint_fit import JointFit
from asymmetry.core.representation.project_model import ProjectModel
from asymmetry.core.representation.series import FitSeries

_FB = RepresentationType.TIME_FB_ASYMMETRY


def _series(batch_id: str, **kwargs) -> FitSeries:
    defaults = dict(
        rep_type=_FB,
        member_run_numbers=[10, 11],
        canonical_model=CompositeModel(["Exponential"]).to_dict(),
        param_roles={"A": "global"},
    )
    defaults.update(kwargs)
    return FitSeries(batch_id, **defaults)


def _joint(joint_id: str = "j1", members=("b1", "b2"), **kwargs) -> JointFit:
    defaults = dict(
        label=None,
        rep_type=_FB,
        member_batch_ids=list(members),
        shared=[
            {
                "name": "A_shared",
                "members": {m: "A" for m in members},
                "value": 0.2,
                "min": None,
                "max": None,
            }
        ],
    )
    defaults.update(kwargs)
    return JointFit(joint_id, **defaults)


def _model_with_joint(joint_id: str = "j1", members=("b1", "b2")) -> ProjectModel:
    model = ProjectModel()
    for batch_id in members:
        series = _series(batch_id, joint_fit_id=joint_id, shared_params={"A": "A_shared"})
        model.add_batch(series)
    model.add_joint_fit(_joint(joint_id, members))
    return model


# ── JointFit construction / normalisation ───────────────────────────────────


def test_rep_type_accepts_string():
    joint = _joint(rep_type="time_fb_asymmetry")
    assert joint.rep_type is RepresentationType.TIME_FB_ASYMMETRY


def test_label_blank_becomes_none():
    joint = _joint(label="   ")
    assert joint.label is None


def test_display_name_falls_back():
    joint = _joint(label=None)
    assert joint.display_name("fallback") == "fallback"
    joint.label = "My joint fit"
    assert joint.display_name("fallback") == "My joint fit"


def test_shared_row_bounds_default_unbounded():
    joint = _joint(shared=[{"name": "A_shared", "members": {"b1": "A", "b2": "A"}, "value": 0.2}])
    row = joint.shared[0]
    assert row["min"] == float("-inf")
    assert row["max"] == float("inf")


# ── to_dict / from_dict round trip ───────────────────────────────────────────


def test_to_dict_from_dict_round_trip():
    joint = _joint(
        label="Ordered + Para",
        shared=[
            {
                "name": "A_shared",
                "members": {"b1": "A", "b2": "A_scale"},
                "value": 0.25,
                "min": 0.0,
                "max": 1.0,
            }
        ],
    )
    joint.result = {
        "shared_values": {"A_shared": 0.251},
        "shared_uncertainties": {"A_shared": 0.003},
        "shared_covariance": [[9e-6]],
        "chi_squared": 120.0,
        "dof": 100,
        "reduced_chi_squared": 1.2,
        "series_reduced_chi_squared": {"b1": 1.1, "b2": 1.3},
        "fitted_at": "2026-09-18T00:00:00",
    }
    restored = JointFit.from_dict(joint.to_dict())
    assert restored is not None
    assert restored.joint_id == joint.joint_id
    assert restored.label == "Ordered + Para"
    assert restored.rep_type == _FB
    assert restored.member_batch_ids == ["b1", "b2"]
    assert restored.shared == joint.shared
    assert restored.result == joint.result


def test_shared_row_infinite_bounds_serialise_as_none():
    joint = _joint()
    data = joint.to_dict()
    assert data["shared"][0]["min"] is None
    assert data["shared"][0]["max"] is None


def test_from_dict_unknown_rep_type_returns_none():
    data = _joint().to_dict()
    data["rep_type"] = "not-a-real-rep-type"
    assert JointFit.from_dict(data) is None


def test_from_dict_missing_rep_type_returns_none():
    data = _joint().to_dict()
    del data["rep_type"]
    assert JointFit.from_dict(data) is None


def test_from_dict_fewer_than_two_members_returns_none():
    data = _joint().to_dict()
    data["member_batch_ids"] = ["b1"]
    assert JointFit.from_dict(data) is None


def test_from_dict_not_a_dict_returns_none():
    assert JointFit.from_dict("not-a-dict") is None  # type: ignore[arg-type]
    assert JointFit.from_dict(None) is None  # type: ignore[arg-type]


def test_from_dict_drops_malformed_shared_row_but_keeps_the_rest():
    data = _joint().to_dict()
    data["shared"].append({"name": "orphan"})  # no "members": dropped, not fatal
    data["shared"].append("not-a-dict")
    restored = JointFit.from_dict(data)
    assert restored is not None
    assert len(restored.shared) == 1
    assert restored.shared[0]["name"] == "A_shared"


def test_from_dict_malformed_result_degrades_to_none():
    data = _joint().to_dict()
    data["result"] = "not-a-dict"
    restored = JointFit.from_dict(data)
    assert restored is not None
    assert restored.result is None


# ── FitSeries joint stamp ────────────────────────────────────────────────────


def test_series_defaults_unstamped():
    series = _series("b1")
    assert series.joint_fit_id is None
    assert series.shared_params == {}


def test_series_stamp_round_trips():
    series = _series("b1", joint_fit_id="j1", shared_params={"A": "A_shared"})
    restored = FitSeries.from_dict(series.to_dict())
    assert restored.joint_fit_id == "j1"
    assert restored.shared_params == {"A": "A_shared"}


def test_series_stamp_absent_key_defaults_unstamped():
    payload = _series("b1").to_dict()
    del payload["joint_fit_id"]
    del payload["shared_params"]
    restored = FitSeries.from_dict(payload)
    assert restored.joint_fit_id is None
    assert restored.shared_params == {}


def test_clear_joint_stamp():
    series = _series("b1", joint_fit_id="j1", shared_params={"A": "A_shared"})
    series.clear_joint_stamp()
    assert series.joint_fit_id is None
    assert series.shared_params == {}


def test_recipe_identity_unchanged_by_stamping():
    plain = _series("b1")
    stamped = _series("b1", joint_fit_id="j1", shared_params={"A": "A_shared"})
    assert plain.recipe_identity() == stamped.recipe_identity()


# ── ProjectModel joint-fit registry ──────────────────────────────────────────


def test_add_and_fetch_joint_fit():
    model = ProjectModel()
    joint = _joint()
    model.add_joint_fit(joint)
    assert model.joint_fit("j1") is joint


def test_joint_fit_unknown_id_raises_keyerror():
    model = ProjectModel()
    try:
        model.joint_fit("missing")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")


def test_joint_fit_for_series_reads_the_stamp():
    model = _model_with_joint()
    found = model.joint_fit_for_series("b1")
    assert found is not None
    assert found.joint_id == "j1"


def test_joint_fit_for_series_none_when_unstamped():
    model = ProjectModel()
    model.add_batch(_series("b1"))
    assert model.joint_fit_for_series("b1") is None


def test_joint_fit_for_series_none_for_unknown_batch():
    model = ProjectModel()
    assert model.joint_fit_for_series("missing") is None


def test_remove_joint_fit_clears_stamps_and_keeps_results():
    model = _model_with_joint()
    model.batch("b1").results_by_run = {10: {"success": True, "parameters": {"A": 0.2}}}
    removed = model.remove_joint_fit("j1")
    assert removed is not None
    assert model.joint_fits == {}
    assert model.batch("b1").joint_fit_id is None
    assert model.batch("b1").shared_params == {}
    # Results are untouched — deleting the record never touches a member's fit.
    assert model.batch("b1").results_by_run == {10: {"success": True, "parameters": {"A": 0.2}}}
    assert model.batch("b2").joint_fit_id is None


def test_remove_joint_fit_unknown_id_returns_none():
    model = ProjectModel()
    assert model.remove_joint_fit("missing") is None


def test_remove_batch_drops_member_from_a_three_member_joint_fit():
    model = ProjectModel()
    for batch_id in ("b1", "b2", "b3"):
        model.add_batch(_series(batch_id, joint_fit_id="j1", shared_params={"A": "A_shared"}))
    model.add_joint_fit(
        _joint(
            "j1",
            members=("b1", "b2", "b3"),
            shared=[
                {
                    "name": "A_shared",
                    "members": {"b1": "A", "b2": "A", "b3": "A"},
                    "value": 0.2,
                }
            ],
        )
    )
    model.remove_batch("b3")
    joint = model.joint_fit("j1")
    assert joint.member_batch_ids == ["b1", "b2"]
    assert set(joint.shared[0]["members"]) == {"b1", "b2"}
    # Still two members: not deleted, survivors keep their stamp.
    assert model.batch("b1").joint_fit_id == "j1"
    assert model.batch("b2").joint_fit_id == "j1"


def test_remove_batch_second_delete_removes_the_joint_fit_and_clears_survivor():
    model = ProjectModel()
    for batch_id in ("b1", "b2", "b3"):
        model.add_batch(_series(batch_id, joint_fit_id="j1", shared_params={"A": "A_shared"}))
    model.add_joint_fit(
        _joint(
            "j1",
            members=("b1", "b2", "b3"),
            shared=[
                {
                    "name": "A_shared",
                    "members": {"b1": "A", "b2": "A", "b3": "A"},
                    "value": 0.2,
                }
            ],
        )
    )
    model.remove_batch("b3")
    model.remove_batch("b2")
    assert model.joint_fits == {}
    assert model.batch("b1").joint_fit_id is None
    assert model.batch("b1").shared_params == {}


def test_remove_batch_of_a_non_member_series_leaves_joint_fit_untouched():
    model = _model_with_joint()
    model.add_batch(_series("solo"))
    model.remove_batch("solo")
    assert "j1" in model.joint_fits
    assert model.joint_fit("j1").member_batch_ids == ["b1", "b2"]


# ── staleness (D9) ────────────────────────────────────────────────────────────


def test_fresh_joint_fit_is_not_stale():
    model = _model_with_joint()
    joint = model.joint_fit("j1")
    assert not joint.is_stale(model)
    assert joint.stale_reason(model) == ""


def test_removing_the_only_extra_member_deletes_the_record_via_cascade():
    model = _model_with_joint()
    model.remove_batch("b2")
    # b2 was the second (and last) member: the cascade removes the record too,
    # so there is nothing left to ask "is this stale" about.
    assert "j1" not in model.joint_fits


def test_missing_member_reason_when_bypassing_the_cascade():
    """``stale_reason``'s "was deleted" branch, isolated from ``remove_batch``.

    ``ProjectModel.remove_batch`` always keeps a joint fit's membership in
    sync (D10), so the only way to observe a genuinely *missing* member is to
    bypass it — exactly the situation a corrupted or hand-edited project file
    could leave behind, which is what this staleness branch exists for.
    """
    model = ProjectModel()
    for batch_id in ("b1", "b2", "b3"):
        model.add_batch(_series(batch_id, joint_fit_id="j1", shared_params={"A": "A_shared"}))
    joint = _joint(
        "j1",
        members=("b1", "b2", "b3"),
        shared=[{"name": "A_shared", "members": {"b1": "A", "b2": "A", "b3": "A"}, "value": 0.2}],
    )
    model.add_joint_fit(joint)
    del model.batches["b3"]
    assert joint.is_stale(model)
    assert "was deleted" in joint.stale_reason(model)


def test_detached_member_makes_it_stale():
    model = _model_with_joint()
    model.batch("b1").clear_joint_stamp()
    joint = model.joint_fit("j1")
    assert joint.is_stale(model)
    assert "re-run on its own" in joint.stale_reason(model)


def test_member_group_staleness_propagates():
    model = _model_with_joint()
    group = DataGroup(group_id="g1", name="scan", member_run_numbers=[10, 11, 12])
    model.add_data_group(group)
    series = model.batch("b1")
    series.group_id = "g1"
    series.last_fitted_members = [10, 11]
    joint = model.joint_fit("j1")
    assert joint.is_stale(model)
    assert "membership changed" in joint.stale_reason(model)

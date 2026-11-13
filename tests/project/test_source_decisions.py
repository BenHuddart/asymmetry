"""Owner-specific replacement preserves immutable histories and legacy expectations."""

from copy import deepcopy
from dataclasses import replace

import pytest

from asymmetry.core.data.acquisition import ContentIdentity, SourceBinding
from asymmetry.core.io import load
from asymmetry.core.project import (
    CURRENT_SCHEMA_VERSION,
    apply_source_decisions,
    load_prepared_project,
    prepare_project,
    save_project,
    verify_project_sources,
)
from asymmetry.core.project.sources import source_record
from asymmetry.core.workflow.reduction import ReductionSettings, reduce_run
from tests.io.test_acquisition_content import acquisition

pytestmark = pytest.mark.io


def owned_project(tmp_path):
    path = acquisition(tmp_path)
    dataset = load(str(path))
    recipe = reduce_run(
        dataset.run, ReductionSettings(alpha=1.0, alpha_source="user")
    ).effective_recipe.to_dict()
    state = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "datasets": [
            {
                "run_number": dataset.run_number,
                "source_file": str(path),
                "source_binding": dataset.run.source_binding.to_dict(),
                "representations": {
                    "time_fb_asymmetry": {
                        "rep_type": "time_fb_asymmetry",
                        "recipe": {},
                        "result_metadata": {
                            "effective_recipes": [
                                {
                                    "projection": None,
                                    "effective_recipe": recipe,
                                    "computation_revision": "1",
                                }
                            ]
                        },
                        "single_fits": {
                            "": {
                                "open_id": "original",
                                "fits": [
                                    {
                                        "fit_id": "original",
                                        "result": {
                                            "input_attribution": {
                                                "identity": "owned",
                                                "effective_recipe": recipe,
                                            },
                                            "computation_revision": "1",
                                            "parameters": {"A": 0.1},
                                        },
                                        "analysis_history": {
                                            "reason": "constraints",
                                            "original": {"parameters": []},
                                        },
                                    }
                                ],
                            }
                        },
                    }
                },
            }
        ],
    }
    prepare_project(state).model.write_to_project_state(state)
    return path, state


def test_selected_replacement_resave_does_not_promote_history(tmp_path):
    path, state = owned_project(tmp_path)
    original = deepcopy(state)
    prepared = prepare_project(state)
    acquisition(tmp_path, 5)
    verified = verify_project_sources(prepared, tmp_path / "project.asymp")
    facts = verified.facts
    assert facts[0].status == "changed"
    assert facts[0].source.owner.active
    assert all(not f.source.owner.active for f in facts[1:])
    history = apply_source_decisions(verified)
    assert not history.available(0)
    assert history.prepared.state == original
    accepted = apply_source_decisions(verified, replacements=[0])
    observed = facts[0].observed
    assert (
        accepted.prepared.state["datasets"][0]["source_binding"]
        == SourceBinding(observed, observed).to_dict()
    )
    assert (
        accepted.prepared.state["datasets"][0]["representations"]
        == original["datasets"][0]["representations"]
    )
    assert prepared.state == original
    project = tmp_path / "saved.asymp"
    save_project(accepted.prepared.state, project)
    reopened = load_prepared_project(project)
    verified = verify_project_sources(reopened, project)
    fresh = verified.facts
    assert fresh[0].status == "matched"
    assert all(f.status == "changed" and not f.source.owner.active for f in fresh[1:])
    assert not apply_source_decisions(verified).replacements
    assert str(path) not in str(accepted.report())
    assert all("candidate_file" not in row and "diagnostic" not in row for row in accepted.report())


def test_owner_order_selectors_and_frozen_calibration(tmp_path):
    path, state = owned_project(tmp_path)
    identity = load(str(path)).run.source_binding.expected
    state["datasets"][0]["grouping_overrides"] = {
        "background_correction": True,
        "background_mode": "reference_run",
        "background_run": {
            "source_file": str(path),
            "run_number": 7,
            "period_number": 2,
            "content_identity": identity.to_dict(),
        },
    }
    state["combined_datasets"] = [
        {
            "combined_run_number": -1,
            "source_run_numbers": [1, 2],
            "constituent_sources": [
                {
                    "source_file": str(path),
                    "source_run_number": 7,
                    "data_period_number": p,
                    "dae_period_number": p + 2,
                    "input_ordinal": p,
                    "coefficient": c,
                    "source_binding": SourceBinding(identity, identity).to_dict(),
                }
                for p, c in ((1, 1.0), (2, -2.0))
            ],
        }
    ]
    prepared = prepare_project(state)
    acquisition(tmp_path, 5)
    verified = verify_project_sources(prepared, tmp_path / "project.asymp")
    facts = verified.facts
    active = [i for i, f in enumerate(facts) if f.source.owner.active]
    terms = [facts[i].source for i in active if facts[i].source.role == "constituent"]
    assert [
        (s.period_number, s.dae_period_number, s.input_number, s.coefficient) for s in terms
    ] == [(1, 3, 1, 1.0), (2, 4, 2, -2.0)]
    ref_index = next(i for i in active if facts[i].source.role == "reference")
    accepted = apply_source_decisions(verified, replacements=[ref_index])
    assert not accepted.available(0)
    assert (
        source_record(accepted.prepared.state, facts[ref_index].source.owner.address)[
            "content_identity"
        ]
        == facts[ref_index].observed.to_dict()
    )
    assert accepted.prepared.state["combined_datasets"] == state["combined_datasets"]
    with pytest.raises(ValueError, match="unavailable"):
        history = apply_source_decisions(verified)
        history.reference_inputs(
            history.prepared.state["datasets"][0]["grouping_overrides"]["background_run"]
        )
    for i, f in enumerate(facts):
        if not f.source.owner.active:
            with pytest.raises(ValueError, match="current input"):
                apply_source_decisions(verified, replacements=[i])


def test_relative_identical_relocation_legacy_and_capture_race(tmp_path):
    path, state = owned_project(tmp_path)
    state["datasets"][0]["source_file"] = path.name
    state["datasets"].append({"run_number": 8, "source_file": path.name})
    prepared = prepare_project(state)
    verified = verify_project_sources(prepared, tmp_path / "project.asymp")
    facts = verified.facts
    legacy = next(
        i for i, f in enumerate(facts) if f.source.owner.active and f.source.run_number == 8
    )
    assert facts[0].status == "matched" and facts[legacy].status == "legacy_unknown"
    applied = apply_source_decisions(verified)
    assert "source_binding" not in applied.prepared.state["datasets"][1]
    applied.check_capture(legacy, facts[legacy].observed)
    for observed in (None, ContentIdentity("d" * 64)):
        with pytest.raises(ValueError, match="after verification"):
            applied.check_capture(0, observed)
    with pytest.raises(ValueError, match="complete verification"):
        apply_source_decisions(replace(verified, facts=facts[::-1]))
    for index in (-1, True, 999):
        with pytest.raises(ValueError, match="index"):
            apply_source_decisions(verified, replacements=[index])


@pytest.mark.parametrize("value", [{}, "bad", [None]])
def test_owned_recipe_list_boundary_is_contextual(value):
    with pytest.raises(ValueError, match="effective_recipes"):
        prepare_project(
            {
                "schema_version": CURRENT_SCHEMA_VERSION,
                "datasets": [
                    {
                        "run_number": 1,
                        "representations": {
                            "time_fb_asymmetry": {
                                "result_metadata": {"effective_recipes": value},
                            }
                        },
                    }
                ],
            }
        )


def test_content_only_change_keeps_equal_observations_and_separate_fit_history(tmp_path):
    import numpy as np

    from asymmetry.core.fitting.engine import FitEngine
    from asymmetry.core.fitting.input_attribution import saved_fit_matches
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet
    from asymmetry.core.fitting.result_summary import fit_result_summary
    from asymmetry.core.representation.base import FitSlot, SingleFitSet
    from asymmetry.core.representation.series import FitSeries
    from tests.core.test_computation_revision import constant

    path = acquisition(tmp_path)
    before = reduce_run(
        load(str(path)).run, ReductionSettings(alpha=1.0, alpha_source="user")
    ).time_range(0.3, 0.5)
    # HDF5 ignores trailing bytes; the acquisition identity still covers them.
    with path.open("ab") as stream:
        stream.write(b"owned non-numerical acquisition annotation")
    after = reduce_run(
        load(str(path)).run, ReductionSettings(alpha=1.0, alpha_source="user")
    ).time_range(0.3, 0.5)
    np.testing.assert_array_equal(before.asymmetry, after.asymmetry)
    np.testing.assert_array_equal(before.error, after.error)
    summaries = [
        fit_result_summary(FitEngine().fit(curve, constant, ParameterSet([Parameter("c", 0.0)])))
        for curve in (before, after)
    ]
    assert (
        summaries[0]["input_attribution"]["identity"]
        == summaries[1]["input_attribution"]["identity"]
    )
    assert before.effective_recipe.identity == after.effective_recipe.identity
    assert (
        before.effective_recipe.sources()[0].expected
        != after.effective_recipe.sources()[0].expected
    )
    for summary in summaries:
        summary.update(npar=1, ndof=before.n_points - 1)
    fits = SingleFitSet()
    for summary in summaries:
        fits.record(FitSlot(result=summary), detached=False, default_name=lambda slot: "Constant")
    assert len(fits.fits) == 2
    assert fits.fits[0].data_key() is not None
    assert fits.fits[0].data_key() == fits.fits[1].data_key()
    from asymmetry.core.fitting.fit_wizard import SelectionMetric
    from asymmetry.core.fitting.model_comparison import summarise_saved_fits

    ranked = summarise_saved_fits(
        [(slot, slot.fit_id) for slot in fits.fits],
        after,
        SelectionMetric.AICC,
        {},
        first=fits.fits[0].fit_id,
    )
    assert all(np.isinf(row.delta) and row.weight == 0 for row in ranked)
    assert not saved_fit_matches(fits.fits[0], after)
    assert saved_fit_matches(fits.fits[1], after)
    old, new = [
        FitSeries(str(i), "time_fb_asymmetry", results_by_run={7: summary})
        for i, summary in enumerate(summaries)
    ]
    assert not old.same_observations(new)


def test_zero_terms_and_calibration_archives_are_not_current_replacements(tmp_path):
    from asymmetry.core.project.calibration import CalibrationRecord, calibration_payload
    from tests.project.test_tf_calibration_records import candidate

    path, state = owned_project(tmp_path)
    old = load(str(path)).run.source_binding.expected
    state["datasets"][0]["grouping_overrides"] = {
        "calibration_records": [
            calibration_payload(
                CalibrationRecord("frozen", replace(candidate(), content_identity=old))
            )
        ]
    }
    state["combined_datasets"] = [
        {
            "combined_run_number": -1,
            "source_run_numbers": [7, 8],
            "constituent_sources": [
                {
                    "source_file": str(path),
                    "coefficient": 0.0,
                    "source_binding": SourceBinding(old, old).to_dict(),
                }
            ],
        }
    ]
    prepared = prepare_project(state)
    acquisition(tmp_path, 5)
    verified = verify_project_sources(prepared, tmp_path / "test.asymp")
    facts = verified.facts
    zero = next(f for f in facts if f.source.role == "constituent")
    calibration = next(f for f in facts if f.source.role == "calibration")
    assert not zero.source.owner.active
    assert not calibration.source.owner.active
    applied = apply_source_decisions(verified, replacements=[0])
    applied.check_constituents(0, [])
    assert (
        applied.prepared.state["datasets"][0]["grouping_overrides"]
        == state["datasets"][0]["grouping_overrides"]
    )


def test_profile_preflight_propagates_one_shot_checkpoint_cancel(tmp_path, monkeypatch):
    import asymmetry.core.io as io
    from asymmetry.core.project import BackgroundPolicy, GroupingProfile
    from asymmetry.core.project.profiles import profile_fingerprint_for_run

    path, state = owned_project(tmp_path)
    run = load(str(path)).run
    profile = GroupingProfile(
        "current",
        profile_fingerprint_for_run(run),
        background_policy=BackgroundPolicy(
            "reference_run", {"background_run": {"source_file": str(path)}}
        ),
    )
    state["grouping_profiles"] = [profile.to_dict()]
    prepared = prepare_project(state)
    calls = []

    def checkpoint():
        if calls == ["inside"]:
            calls.append("cancelled")
            raise InterruptedError("one-shot cancellation")

    def decode(path, *, checkpoint):
        calls.append("inside")
        checkpoint()
        return load(path)

    monkeypatch.setattr(io, "load", decode)
    with pytest.raises(InterruptedError, match="one-shot"):
        verify_project_sources(prepared, tmp_path / "test.asymp", checkpoint=checkpoint)


def test_nullable_legacy_locators_stay_unknown_without_editing_history(tmp_path):
    state = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "datasets": [
            {
                "run_number": 7,
                "grouping_overrides": {
                    "background_correction": True,
                    "background_mode": "reference_run",
                    "background_run": {"run_number": 8, "source_file": None},
                },
                "representations": {
                    "time_fb_asymmetry": {
                        "result_metadata": {
                            "effective_recipes": [
                                {
                                    "effective_recipe": {
                                        "version": 1,
                                        "operation": "prepared_counts",
                                        "source": {
                                            "source_file": None,
                                            "run_number": 7,
                                            "content_identity": None,
                                        },
                                    }
                                }
                            ]
                        }
                    }
                },
            }
        ],
    }
    prepared = prepare_project(state)
    assert all(source.source_file == "" for source in prepared.sources)
    assert prepared.state == state
    verified = verify_project_sources(prepared, tmp_path / "legacy.asymp")
    assert all(f.source.expected is None for f in verified.facts)
    assert apply_source_decisions(verified).prepared.state == state
    state["datasets"][0]["representations"]["time_fb_asymmetry"]["result_metadata"][
        "effective_recipes"
    ][0]["effective_recipe"]["source"]["source_file"] = []
    with pytest.raises(ValueError, match="project.datasets.*source_file"):
        prepare_project(state)

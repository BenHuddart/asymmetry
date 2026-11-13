"""Acquisition bytes, observation ownership, and historical expectations."""

import hashlib
import os
import shutil
from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.data.acquisition import (
    ContentIdentity,
    SourceBinding,
    SourceDeclaration,
    verify_sources,
)
from asymmetry.core.io import load
from asymmetry.core.io.nexus import NexusLoader
from asymmetry.core.io.nexus_writer import write_nexus_v1
from asymmetry.core.project import (
    CURRENT_SCHEMA_VERSION,
    load_prepared_project,
    prepare_project,
    save_project,
)
from asymmetry.core.workflow.reduction import ReductionSettings, reduce_run
from tests.core.test_workflow_reduction import _continuous_run

pytestmark = pytest.mark.io


def acquisition(tmp_path, offset=0):
    path = tmp_path / "owned.nxs"
    run = _continuous_run(0)
    run.histograms[0].counts[:] += offset
    write_nexus_v1(run, path)
    return path


def test_actual_loader_capture_and_reduction(tmp_path):
    path = acquisition(tmp_path)
    ds = NexusLoader().load(str(path))
    identity = ContentIdentity(hashlib.sha256(path.read_bytes()).hexdigest())
    assert ds.run.source_binding == SourceBinding(identity, identity)
    reduced = reduce_run(ds.run, ReductionSettings(alpha=1.0, alpha_source="user"))
    assert (
        reduced.effective_recipe.to_dict()["periods"]["red"]["source"]["content_identity"]
        == identity.to_dict()
    )
    assert str(path) not in str(reduced.effective_recipe.publishable())
    moved = tmp_path / "moved.nxs"
    shutil.copyfile(path, moved)
    os.utime(moved, (1, 1))
    declarations = [
        SourceDeclaration("sample", str(path), identity),
        SourceDeclaration("reference", str(moved), identity),
    ]
    assert [v.status for v in verify_sources(declarations)] == ["matched", "matched"]
    before = ds.asymmetry.copy()
    acquisition(tmp_path, 5)
    assert verify_sources(declarations)[0].status == "changed"
    np.testing.assert_array_equal(ds.asymmetry, before)
    assert ds.run.source_binding.observed == identity


def test_legacy_and_expected_observed_project_transport(tmp_path):
    path = acquisition(tmp_path)
    observed = load(str(path)).run.source_binding.observed
    historical = ContentIdentity("a" * 64)
    binding = SourceBinding(historical, observed)
    state = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "datasets": [
            {"run_number": 7, "source_file": str(path), "source_binding": binding.to_dict()},
            {"run_number": 8, "source_file": str(path)},
        ],
    }
    prepared = prepare_project(state)
    target = tmp_path / "owned.asymp"
    save_project(prepared.state, target)
    reopened = load_prepared_project(target)
    assert reopened.state["datasets"][0]["source_binding"] == binding.to_dict()
    assert "source_binding" not in reopened.state["datasets"][1]
    assert [d.expected for d in reopened.sources] == [historical, None]


def test_tail_change_missing_unknown_and_cancel(tmp_path):
    path = tmp_path / "bytes"
    path.write_bytes(b"a" * (2 * 1024 * 1024))
    original = ContentIdentity.read(path)
    status = path.stat()
    with path.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"b")
    os.utime(path, ns=(status.st_atime_ns, status.st_mtime_ns))
    assert verify_sources([SourceDeclaration("sample", str(path), original)])[0].status == "changed"
    assert (
        verify_sources([SourceDeclaration("sample", str(path), None)])[0].status == "legacy_unknown"
    )
    assert (
        verify_sources([SourceDeclaration("sample", str(path) + "missing", original)])[0].status
        == "missing"
    )
    calls = []

    def cancel():
        calls.append(1)
        if len(calls) == 3:
            raise InterruptedError("cancelled")

    with pytest.raises(InterruptedError, match="cancelled"):
        verify_sources([SourceDeclaration("sample", str(path), original)], checkpoint=cancel)


def test_decode_race_never_stamps_new_bytes_on_old_counts(tmp_path, monkeypatch):
    path = acquisition(tmp_path)
    original = NexusLoader._reduce_handle

    def racing_decode(self, handle, source_file):
        result = original(self, handle, source_file)
        status = path.stat()
        with path.open("r+b") as stream:
            stream.seek(-1, 2)
            old = stream.read(1)
            stream.seek(-1, 2)
            stream.write(bytes([old[0] ^ 1]))
        os.utime(path, ns=(status.st_atime_ns, status.st_mtime_ns))
        return result

    monkeypatch.setattr(NexusLoader, "_reduce_handle", racing_decode)
    result = load(str(path))
    assert result.n_points > 0
    assert result.run.source_binding == SourceBinding()


def test_hash_io_unavailable_does_not_reject_decodable_acquisition(tmp_path, monkeypatch):
    from asymmetry.core.utils.content import ContentReadError

    path = acquisition(tmp_path)

    def unavailable(*args, **kwargs):
        raise ContentReadError("read unavailable")

    monkeypatch.setattr(ContentIdentity, "read", unavailable)
    assert load(str(path)).run.source_binding == SourceBinding()


@pytest.mark.parametrize("exception", [InterruptedError, RuntimeError])
def test_loader_and_verification_propagate_checkpoint_cancel(tmp_path, exception):
    path = acquisition(tmp_path)

    def cancelled():
        raise exception("cancelled")

    with pytest.raises(exception, match="cancelled"):
        load(str(path), checkpoint=cancelled)
    with pytest.raises(exception, match="cancelled"):
        verify_sources([SourceDeclaration("sample", str(path), None)], checkpoint=cancelled)


def test_period_views_share_capture_and_verification_retains_terms(tmp_path, monkeypatch):
    from asymmetry.core.io import select_period
    from tests.io.test_nexus_loader import _write_v2_file

    path = tmp_path / "periods.nxs"
    _write_v2_file(path, multiperiod=True)
    calls = []
    real_read = ContentIdentity.read

    def counted(path, **kwargs):
        calls.append(path)
        return real_read(path, **kwargs)

    monkeypatch.setattr(ContentIdentity, "read", counted)
    loaded = load(str(path))
    assert len(calls) == 2  # Whole hashes bracket one decode, not each period.
    red, green = [select_period(loaded, p) for p in (1, 2)]
    identity = red.run.source_binding.observed
    assert green.run.source_binding.observed is identity
    assert all(r.source_binding.observed is identity for r in loaded.run.grouping["period_inputs"])
    terms = [
        SourceDeclaration("constituent", str(path), identity, period_number=p, coefficient=c)
        for p, c in ((1, 1.0), (2, -2.0))
    ]
    calls.clear()
    progress = []
    facts = verify_sources(terms, progress=lambda done, total: progress.append((done, total)))
    assert len(calls) == 1
    assert [f.source.coefficient for f in facts] == [1.0, -2.0]
    assert all(f.status == "matched" for f in facts)
    assert progress[0][0] == 0 and progress[-1][0] == path.stat().st_size


def test_unreadable_and_callback_io_error_are_distinct(tmp_path, monkeypatch):
    from pathlib import Path

    path = acquisition(tmp_path)
    real_open = Path.open

    def denied(self, *args, **kwargs):
        if self == path:
            raise PermissionError("denied")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denied)
    fact = verify_sources([SourceDeclaration("sample", str(path), None)])[0]
    assert fact.status == "unreadable" and fact.observed is None
    monkeypatch.undo()

    def cancelled(done, total):
        raise InterruptedError("callback cancelled")

    with pytest.raises(InterruptedError, match="callback cancelled"):
        verify_sources([SourceDeclaration("sample", str(path), None)], progress=cancelled)


def test_reference_cache_keeps_its_counts_identity_when_stat_tags_match(tmp_path):
    from asymmetry.core.io import resolve_background_reference
    from asymmetry.core.io.reference_selection import reference_selection
    from asymmetry.core.transform.reference_scale import ReferenceExposure

    path = acquisition(tmp_path)
    loaded = load(str(path))
    payload = dict(
        reference_selection(loaded.run),
        source_file=str(path),
        scale_request={"mode": "manual", "coefficient": 1.0},
    )
    cache = {}
    before = resolve_background_reference(payload, cache=cache, sample_exposure=ReferenceExposure())
    status = path.stat()
    acquisition(tmp_path, 5)
    # Rewriting with HDF5 may change file size; cache-hit control uses the explicit
    # retained stat tag, independently of content verification.
    from asymmetry.core.io import reference_file_identity

    key, run = next(iter(cache.items()))
    cache.clear()
    cache[(*key[:3], reference_file_identity(payload))] = run
    after = resolve_background_reference(payload, cache=cache)
    assert after.content_identity == before.content_identity
    facts = verify_sources([SourceDeclaration("reference", str(path), before.content_identity)])
    assert facts[0].status == "changed"
    assert status.st_size > 0


def test_restore_keeps_expected_unknown_and_fresh_observation_separate(tmp_path):
    from asymmetry.core.io.acquisition import restore_source_binding

    path = acquisition(tmp_path)
    ds = load(str(path))
    historical = ContentIdentity("b" * 64)
    for saved, expected in (
        (None, None),
        (SourceBinding(historical, historical).to_dict(), historical),
    ):
        restored = restore_source_binding(ds, saved)
        assert restored.run.source_binding.expected == expected
        assert restored.run.source_binding.observed == ds.run.source_binding.observed
        fresh = reduce_run(restored.run, ReductionSettings(alpha=1.0, alpha_source="user"))
        assert (
            fresh.effective_recipe.to_dict()["periods"]["red"]["source"]["content_identity"]
            == ds.run.source_binding.observed.to_dict()
        )
        assert ds.run.source_binding.expected == ds.run.source_binding.observed


@pytest.mark.parametrize("method", ["ratio", "tf_calibration_fit"])
def test_fresh_calibration_captures_source_and_legacy_codec_stays_unknown(tmp_path, method):
    from asymmetry.core.fitting.tf_calibration import TFEnvelope
    from asymmetry.core.project.calibration import (
        CalibrationRecord,
        calibration_payload,
        restore_calibration,
    )
    from asymmetry.core.workflow.calibration import CalibrationOptions, calibrate_run
    from asymmetry.core.workflow.calibration_results import QuickCalibrationResult
    from tests.core.test_tf_calibration_fit import _raw_tf
    from tests.project.test_tf_calibration_records import candidate

    template = _raw_tf(beta=1.0)
    path = tmp_path / "tf.nxs"
    write_nexus_v1(template.run, path)
    ds = load(str(path))
    ds = replace(ds, run=replace(ds.run, grouping=template.run.grouping))
    identity = ds.run.source_binding.observed
    outcome = calibrate_run(
        ds,
        ReductionSettings(),
        CalibrationOptions(
            method, envelope=TFEnvelope.EXPONENTIAL if method == "tf_calibration_fit" else None
        ),
    )
    if isinstance(outcome.result, QuickCalibrationResult):
        codec = outcome.result.to_dict()
        assert QuickCalibrationResult.from_dict(codec).content_identity == identity
        codec.pop("content_identity")
        assert QuickCalibrationResult.from_dict(codec).content_identity is None
    else:
        codec = calibration_payload(outcome.result)
        assert restore_calibration(codec).result.content_identity == identity
        codec.pop("content_identity")
        assert restore_calibration(codec).result.content_identity is None
    historical = replace(candidate(), content_identity=identity)
    record = CalibrationRecord("history", historical, computation_revision="older")
    restored = restore_calibration(calibration_payload(record))
    assert restored == record
    assert (
        verify_sources(
            [SourceDeclaration("calibration", str(path), restored.result.content_identity)]
        )[0].status
        == "matched"
    )


def test_prepared_sources_keep_reference_calibration_and_combined_roles(tmp_path):
    from asymmetry.core.project.calibration import CalibrationRecord, apply_calibration
    from asymmetry.core.project.profiles import (
        BackgroundPolicy,
        GroupingProfile,
        ProfileFingerprint,
    )
    from tests.project.test_tf_calibration_records import candidate

    identity = ContentIdentity("c" * 64)
    profile = GroupingProfile(
        "owned",
        ProfileFingerprint("test", 2),
        background_policy=BackgroundPolicy(
            "reference_run",
            {
                "background_run": {
                    "source_file": "reference.nxs",
                    "run_number": 5,
                    "period_number": 2,
                    "content_identity": identity.to_dict(),
                }
            },
        ),
    )
    apply_calibration(
        profile, CalibrationRecord("cal", replace(candidate(), content_identity=identity))
    )
    state = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "datasets": [],
        "grouping_profiles": [profile.to_dict()],
        "combined_datasets": [
            {
                "combined_run_number": -1,
                "source_run_numbers": [3, 4],
                "constituent_sources": [
                    {
                        "source_file": "same.nxs",
                        "source_run_number": 3,
                        "data_period_number": p,
                        "coefficient": c,
                        "source_binding": SourceBinding(identity, identity).to_dict(),
                    }
                    for p, c in ((1, 1.0), (2, -2.0))
                ],
            }
        ],
    }
    prepared = prepare_project(state)
    assert [(d.role, d.period_number, d.coefficient) for d in prepared.sources] == [
        ("constituent", 1, 1.0),
        ("constituent", 2, -2.0),
        ("reference", 2, 1.0),
        ("calibration", 2, 1.0),
    ]
    assert all(d.expected == identity for d in prepared.sources)
    assert prepared.state == state


def test_result_recipe_inventory_retains_unknown_and_ordered_signed_terms():
    from asymmetry.core.data.reduction_recipe import EffectiveReductionRecipe

    recipe = EffectiveReductionRecipe.from_dict(
        {
            "version": 1,
            "operation": "prepared_counts",
            "constituents": [
                {
                    "coefficient": c,
                    "input_number": n,
                    "source": {
                        "source_file": "same.nxs",
                        "run_number": 3,
                        "data_period_number": n,
                        "content_identity": None,
                    },
                }
                for n, c in ((1, 1.0), (2, -2.0))
            ],
        }
    )
    declarations = recipe.sources()
    assert [(d.input_number, d.period_number, d.coefficient, d.expected) for d in declarations] == [
        (1, 1, 1.0, None),
        (2, 2, -2.0, None),
    ]


@pytest.mark.parametrize("candidate", ["loop", "bad\x00name"])
def test_unresolvable_locator_is_unreadable(tmp_path, candidate):
    path = tmp_path / candidate
    if candidate == "loop":
        path.symlink_to(path)
    fact = verify_sources([SourceDeclaration("sample", str(path), None)])[0]
    assert fact.status == "unreadable" and fact.observed is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_file", []),
        ("coefficient", "1"),
        ("coefficient", float("nan")),
        ("coefficient", True),
        ("period_number", True),
        ("data_period_number", 0),
        ("input_ordinal", -1),
        ("dae_period_number", "2"),
    ],
)
def test_new_constituent_codec_rejects_malformed_owned_values(field, value):
    from asymmetry.core.project import ProjectValidationError

    row = {"source_file": "missing.nxs", "source_binding": SourceBinding().to_dict(), field: value}
    with pytest.raises(ProjectValidationError, match="constituent_sources"):
        prepare_project(
            {
                "schema_version": CURRENT_SCHEMA_VERSION,
                "datasets": [],
                "combined_datasets": [
                    {
                        "combined_run_number": -1,
                        "source_run_numbers": [1, 2],
                        "constituent_sources": [row],
                    }
                ],
            }
        )


def test_prepared_inventory_keeps_saved_fit_and_series_expectations():
    from asymmetry.core.representation import FitSeries, FitSlot, RepresentationType
    from asymmetry.core.representation.project_model import ProjectModel

    identity = ContentIdentity("d" * 64)
    recipe = {
        "version": 1,
        "operation": "prepared_counts",
        "source": {
            "source_file": "old.nxs",
            "run_number": 1,
            "content_identity": identity.to_dict(),
        },
    }
    result = {"input_attribution": {"effective_recipe": recipe}}
    model = ProjectModel()
    rep = model.ensure_dataset(1).ensure(RepresentationType.TIME_FB_ASYMMETRY)
    rep.fit = FitSlot(fit_id="history", result=result)
    model.add_batch(
        FitSeries(
            "series",
            RepresentationType.TIME_FB_ASYMMETRY,
            member_run_numbers=[1],
            results_by_run={1: result},
        )
    )
    state = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "datasets": [{"run_number": 1, "source_file": "current.nxs"}],
    }
    model.write_to_project_state(state)
    prepared = prepare_project(state)
    assert [(d.source_file, d.expected) for d in prepared.sources] == [
        ("current.nxs", None),
        ("old.nxs", identity),
        ("old.nxs", identity),
    ]
    assert prepared.state == state

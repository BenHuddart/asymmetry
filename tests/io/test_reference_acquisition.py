"""A reference denotes one acquisition and one acquired period."""

import numpy as np
import pytest

from asymmetry.core.io import load_background_run, resolve_background_reference, select_period
from asymmetry.core.transform.reduce import corrected_grouped_counts
from asymmetry.core.transform.reference_scale import ReferenceExposure
from tests.io.test_background_reference_resolution import _dataset, _run
from tests.io.test_period_selection import _combined_two_period


def test_declared_source_wins_over_run_number_collision():
    a, b = [_run(42, good_frames=100, level=level) for level in (2, 9)]
    a.source_file, b.source_file = "a.nxs", "b.nxs"
    result = resolve_background_reference(
        {"source_file": b.source_file, "run_number": 42}, datasets=[_dataset(a), _dataset(b)]
    )
    np.testing.assert_array_equal(result.histograms[0].counts, 9)


def test_number_only_collision_is_ambiguous():
    a, b = [_run(42, good_frames=100, level=level) for level in (2, 9)]
    a.source_file, b.source_file = "a.nxs", "b.nxs"
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_background_reference({"run_number": 42}, datasets=[_dataset(a), _dataset(b)])


def test_live_selected_input_wins_after_disk_cache_fill(monkeypatch):
    disk, live = [_run(42, good_frames=100, level=level) for level in (3, 13)]
    disk.source_file = live.source_file = "ref.nxs"
    monkeypatch.setattr("asymmetry.core.io.load_background_run", lambda _: _dataset(disk))
    payload, cache = {"source_file": "ref.nxs", "run_number": 42}, {}
    resolve_background_reference(payload, cache=cache)
    result = resolve_background_reference(payload, cache=cache, datasets=[_dataset(live)])
    np.testing.assert_array_equal(result.histograms[0].counts, 13)


@pytest.mark.parametrize("identity_mapping", [False, True])
def test_disk_and_loaded_selected_period_agree(monkeypatch, identity_mapping):
    combined = _combined_two_period()
    if identity_mapping:
        for facts in (combined.run.grouping, combined.run.metadata):
            facts["period_mapping"] = {"1": "red", "2": "green"}
    green = select_period(combined, 2)
    monkeypatch.setattr("asymmetry.core.io.load", lambda _: combined)
    payload = {"source_file": combined.run.source_file, "run_number": 12345, "period_number": 2}
    disk = resolve_background_reference(payload, sample_exposure=ReferenceExposure(40, "user"))
    live = resolve_background_reference(
        payload, sample_exposure=ReferenceExposure(40, "user"), datasets=[green]
    )
    np.testing.assert_array_equal(disk.histograms[0].counts, green.run.histograms[0].counts)
    np.testing.assert_array_equal(disk.histograms[0].counts, live.histograms[0].counts)
    assert disk.deadtime_normaliser == live.deadtime_normaliser == 20
    assert disk.run_number == live.run_number == 12345
    assert disk.period_number == live.period_number == 2


def test_disk_wrong_acquisition_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "asymmetry.core.io.load", lambda _: _dataset(_run(8, good_frames=1, level=4))
    )
    with pytest.raises(ValueError, match="acquisition"):
        load_background_run({"source_file": "ref.nxs", "run_number": 7})


def test_identical_observations_cannot_be_independent_reference():
    sample = _run(42, good_frames=100, level=50)
    reference = resolve_background_reference({"run_number": 42}, datasets=[_dataset(sample)])
    with pytest.raises(ValueError, match="identical observations"):
        corrected_grouped_counts(
            histograms=sample.histograms,
            grouping={"background_mode": "reference_run"},
            forward_idx=[0],
            backward_idx=[1],
            use_deadtime=False,
            deadtime_mode="off",
            use_background=True,
            reference_resolver=lambda _: reference,
        )


def test_released_project_encoded_reference_is_recovered_from_source_facts(monkeypatch):
    from asymmetry.core.project.schema import migrate_to_current

    combined = _combined_two_period()
    old = {
        "schema_version": 31,
        "datasets": [
            {
                "grouping_overrides": {
                    "background_run": {"source_file": "synthetic.nxs", "run_number": 12345002},
                }
            }
        ],
    }
    migrated = migrate_to_current(old)
    payload = migrated["datasets"][0]["grouping_overrides"]["background_run"]
    monkeypatch.setattr("asymmetry.core.io.load", lambda _: combined)
    actual = resolve_background_reference(payload)
    assert actual.period_number == 2
    assert payload == {
        "source_file": "synthetic.nxs",
        "run_number": 12345,
        "period_number": 2,
        "timing": None,
        "scale_request": {"mode": "automatic"},
    }
    assert actual.run_number == 12345
    assert actual.deadtime_normaliser == 20  # DT normalisation follows the selected reference.
    np.testing.assert_array_equal(
        actual.histograms[0].counts, select_period(combined, 2).run.histograms[0].counts
    )


def test_distinct_acquired_periods_in_one_source_are_eligible(monkeypatch):
    combined = _combined_two_period()
    sample = select_period(combined, 1).run
    sample.metadata["source_file"] = sample.source_file
    monkeypatch.setattr("asymmetry.core.io.load", lambda _: combined)
    reference = resolve_background_reference(
        {"source_file": "synthetic.nxs", "run_number": 12345, "period_number": 2}
    )
    result = corrected_grouped_counts(
        histograms=sample.histograms,
        grouping={"background_mode": "reference_run"},
        forward_idx=[0],
        backward_idx=[1],
        use_deadtime=False,
        deadtime_mode="off",
        use_background=True,
        reference_resolver=lambda _: reference,
        metadata=sample.metadata,
    )
    assert result.corrections.background.effective["reference"]["period_number"] == 2


def test_file_replacement_invalidates_cached_reference(tmp_path, monkeypatch):
    source = tmp_path / "ref.nxs"
    source.write_text("first")
    loads = []

    def load_reference(_):
        loads.append(source.read_text())
        run = _run(42, good_frames=100, level=len(loads))
        run.source_file = str(source)
        return _dataset(run)

    monkeypatch.setattr("asymmetry.core.io.load_background_run", load_reference)
    payload, cache = {"source_file": str(source), "run_number": 42}, {}
    first = resolve_background_reference(payload, cache=cache)
    replacement = tmp_path / "replacement.nxs"
    replacement.write_text("other")
    replacement.replace(source)
    second = resolve_background_reference(payload, cache=cache)
    np.testing.assert_array_equal(first.histograms[0].counts, 1)
    np.testing.assert_array_equal(second.histograms[0].counts, 2)
    assert loads == ["first", "other"]


@pytest.mark.parametrize("number,period", [(None, 1), (12345, 1), (12345001, 1), (12345002, 2)])
@pytest.mark.parametrize("shape", ["grouping_overrides", "profile"])
def test_historical_reference_shapes_round_trip(number, period, shape, tmp_path, monkeypatch):
    from asymmetry.core.project.schema import load_project, save_project

    payload = {"run_number": number, "source_file": "synthetic.nxs"}
    state = {"schema_version": 31, "datasets": [], "grouping_profiles": []}
    if shape == "grouping_overrides":
        state["datasets"] = [
            {
                "run_number": 1,
                "source_file": "sample.nxs",
                "grouping_overrides": {"background_run": payload},
            }
        ]
    else:
        state["grouping_profiles"] = [
            {"background_policy": {"mode": "reference_run", "details": {"background_run": payload}}}
        ]
    path = tmp_path / "old.asymp"
    import json

    path.write_text(json.dumps(state))
    restored = load_project(path)
    save_project(restored, path)
    reopened = load_project(path)
    if shape == "grouping_overrides":
        request = reopened["datasets"][0]["grouping_overrides"]["background_run"]
    else:
        request = reopened["grouping_profiles"][0]["background_policy"]["details"]["background_run"]
    combined = _combined_two_period()
    monkeypatch.setattr("asymmetry.core.io.load", lambda _: combined)
    result = resolve_background_reference(request, sample_exposure=ReferenceExposure(40, "user"))
    assert (result.run_number, result.period_number) == (12345, period)
    assert result.deadtime_normaliser == (10 if period == 1 else 20)


def test_legacy_key_is_not_divided_without_acquisition_evidence(monkeypatch):
    from asymmetry.core.project.schema import migrate_to_current

    legacy = migrate_to_current(
        {
            "schema_version": 31,
            "background_run": {
                "run_number": 12345002,
                "source_file": "synthetic.nxs",
            },
        }
    )["background_run"]
    actual = _run(12345002, good_frames=100, level=7)
    actual.source_file = "synthetic.nxs"
    monkeypatch.setattr("asymmetry.core.io.load", lambda _: _dataset(actual))
    selected = resolve_background_reference(legacy)
    assert selected.run_number == 12345002
    assert selected.period_number == 1
    actual.run_number = 99
    with pytest.raises(ValueError, match="cannot be established|unavailable"):
        resolve_background_reference(legacy)


def test_same_source_same_period_from_disk_is_rejected(monkeypatch):
    combined = _combined_two_period()
    sample = select_period(combined, 2).run
    sample.metadata["source_file"] = sample.source_file
    monkeypatch.setattr("asymmetry.core.io.load", lambda _: combined)
    reference = resolve_background_reference(
        {"source_file": "synthetic.nxs", "run_number": 12345, "period_number": 2}
    )
    with pytest.raises(ValueError, match="identical observations"):
        corrected_grouped_counts(
            histograms=sample.histograms,
            grouping={"background_mode": "reference_run"},
            forward_idx=[0],
            backward_idx=[1],
            use_deadtime=False,
            deadtime_mode="off",
            use_background=True,
            reference_resolver=lambda _: reference,
            metadata=sample.metadata,
        )


def test_number_only_request_does_not_guess_a_period():
    combined = _combined_two_period()
    with pytest.raises(ValueError, match="period.*ambiguous|ambiguous.*period"):
        resolve_background_reference({"run_number": 12345}, datasets=[combined])


def test_number_only_request_accepts_one_loaded_selected_period():
    green = select_period(_combined_two_period(), 2)
    result = resolve_background_reference({"run_number": 12345}, datasets=[green])
    assert result.period_number == 2


def test_parent_view_cannot_hide_a_different_acquisition_collision():
    combined = _combined_two_period()
    other = select_period(combined, 1)
    other.run.source_file = "another.nxs"
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_background_reference(
            {"run_number": 12345, "period_number": 1}, datasets=[combined, other]
        )


def test_live_selected_period_overrides_parent_view_and_cached_disk(monkeypatch):
    combined = _combined_two_period()
    payload = {"source_file": "synthetic.nxs", "run_number": 12345, "period_number": 2}
    monkeypatch.setattr("asymmetry.core.io.load", lambda _: combined)
    cache = {}
    resolve_background_reference(payload, cache=cache)
    green = select_period(combined, 2)
    green.run.histograms[0].counts += 17
    actual = resolve_background_reference(payload, cache=cache, datasets=[combined, green])
    np.testing.assert_array_equal(actual.histograms[0].counts, green.run.histograms[0].counts)
    green.run.histograms[0].counts += 11
    updated = resolve_background_reference(payload, cache=cache, datasets=[combined, green])
    np.testing.assert_array_equal(updated.histograms[0].counts, green.run.histograms[0].counts)
    assert len(cache) == 1
    key = next(iter(cache))
    assert key[1:3] == (12345, 2)


def test_unused_reference_detectors_cannot_hide_identical_selected_observations():
    sample = _run(42, good_frames=100, level=50)
    reference = resolve_background_reference({"run_number": 42}, datasets=[_dataset(sample)])
    with pytest.raises(ValueError, match="identical observations"):
        corrected_grouped_counts(
            histograms=sample.histograms[:2],
            grouping={"background_mode": "reference_run"},
            forward_idx=[0],
            backward_idx=[1],
            use_deadtime=False,
            deadtime_mode="off",
            use_background=True,
            reference_resolver=lambda _: reference,
        )


def test_mapped_reference_is_not_relabelled_as_one_acquired_period():
    mapped = _combined_two_period()
    mapped.run.grouping["period_mapping"] = {"1": "red", "2": "green", "3": "red"}
    with pytest.raises(ValueError, match="mapped.*acquired period"):
        resolve_background_reference(
            {"source_file": "synthetic.nxs", "run_number": 12345, "period_number": 1},
            datasets=[mapped],
            allow_file=False,
        )


def test_released_mapped_reference_remains_explicitly_unavailable():
    from asymmetry.core.project.schema import migrate_to_current

    state = {
        "schema_version": 31,
        "datasets": [
            {
                "run_number": 12345,
                "source_file": "synthetic.nxs",
                "grouping_overrides": {"period_mapping": {"1": "red", "2": "green", "3": "red"}},
            }
        ],
        "background_run": {"run_number": 12345, "source_file": "synthetic.nxs"},
    }
    request = migrate_to_current(state)["background_run"]
    with pytest.raises(ValueError, match="mapped.*acquired period"):
        resolve_background_reference(request, datasets=[_combined_two_period()], allow_file=False)


def test_released_identity_mapping_keeps_acquired_reference_available(monkeypatch):
    from asymmetry.core.project.schema import migrate_to_current

    combined = _combined_two_period()
    state = {
        "schema_version": 31,
        "datasets": [
            {
                "run_number": 12345,
                "source_file": "synthetic.nxs",
                "grouping_overrides": {"period_mapping": {"1": "red", "2": "green"}},
            }
        ],
        "background_run": {"run_number": 12345002, "source_file": "synthetic.nxs"},
    }
    request = migrate_to_current(state)["background_run"]
    monkeypatch.setattr("asymmetry.core.io.load", lambda _: combined)
    reference = resolve_background_reference(request)
    assert reference.period_number == 2


@pytest.mark.parametrize("consumer", ["group", "fb", "automatic_alpha"])
def test_scripted_run_identity_does_not_depend_on_optional_metadata(tmp_path, consumer):
    from dataclasses import replace

    from asymmetry.core.fourier.fb_input import build_fb_signal_dataset
    from asymmetry.core.fourier.grouped import build_group_signal_dataset
    from asymmetry.core.io.nexus_writer import write_nexus_v1
    from asymmetry.core.project.profiles import (
        AlphaPolicy,
        ProfileFingerprint,
        profile_from_payload,
        resolve_effective_grouping,
    )

    sample = _run(42, good_frames=100, level=50)
    sample.metadata = {}
    sample.grouping.update(
        groups={1: [1], 2: [2]},
        forward_group=1,
        backward_group=2,
        alpha=1,
        first_good_bin=8,
        last_good_bin=63,
    )
    source = tmp_path / "sample.nxs"
    sample.source_file = str(source)
    write_nexus_v1(sample, source, effective=True)
    sample.grouping.update(
        background_correction=True,
        background_mode="reference_run",
        background_run={"source_file": str(source), "run_number": 42, "period_number": 1},
    )
    with pytest.raises(ValueError, match="identical observations"):
        if consumer == "group":
            build_group_signal_dataset(
                sample, 1, center_signal=False, apply_lifetime_correction=False
            )
        elif consumer == "fb":
            build_fb_signal_dataset(sample)
        else:
            profile = profile_from_payload(sample.grouping, "Own", ProfileFingerprint("", 4))
            profile = replace(profile, alpha_policy=AlphaPolicy(mode="per_run_estimate"))
            resolve_effective_grouping(
                profile,
                sample,
                reference_resolver=lambda g: resolve_background_reference(g["background_run"]),
            )


@pytest.mark.parametrize("same_source", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_mapped_match_cannot_hide_another_acquisition_collision(same_source, reverse):
    mapped = _combined_two_period()
    mapped.run.grouping["period_mapping"] = {"1": "red", "2": "green", "3": "red"}
    ordinary = select_period(_combined_two_period(), 1)
    if not same_source:
        ordinary.run.source_file = "different.nxs"
    datasets = [mapped, ordinary]
    if reverse:
        datasets.reverse()
    payload = {"run_number": 12345, "period_number": 1}
    if same_source:
        result = resolve_background_reference(payload, datasets=datasets, allow_file=False)
        np.testing.assert_array_equal(
            result.histograms[0].counts, ordinary.run.histograms[0].counts
        )
    else:
        with pytest.raises(ValueError, match="ambiguous"):
            resolve_background_reference(payload, datasets=datasets, allow_file=False)


def test_wrapped_combined_file_reference_selects_the_acquired_period(monkeypatch):
    combined = _combined_two_period()
    monkeypatch.setattr("asymmetry.core.io.load", lambda _: [combined])
    result = load_background_run(
        {"source_file": "synthetic.nxs", "run_number": 12345, "period_number": 2}
    )
    np.testing.assert_array_equal(
        result.run.histograms[0].counts, select_period(combined, 2).run.histograms[0].counts
    )


@pytest.mark.parametrize("consumer", ["integral", "period_integral", "workflow"])
def test_run_reduction_boundaries_preserve_canonical_identity(monkeypatch, consumer):
    from asymmetry.core.transform import integral
    from asymmetry.core.workflow import reduction

    sample = _combined_two_period().run
    sample.metadata = {}
    sample.grouping["period_mode"] = "green_plus_red" if consumer == "period_integral" else "red"
    captured = []
    owner = reduction if consumer == "workflow" else integral
    name = {
        "workflow": "reduce_period_asymmetry",
        "integral": "corrected_period_counts",
        "period_integral": "corrected_period_counts",
    }[consumer]
    original = getattr(owner, name)

    def record(*args, **kwargs):
        captured.append(kwargs["metadata"])
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, record)
    if consumer == "workflow":
        reduction.reduce_run(sample, reduction.ReductionSettings(pair=("1", "2")))
    else:
        integral.reduce_period_integral(sample)
    assert captured
    assert all(
        item.get("source_file") == sample.source_file
        and item.get("run_number") == sample.run_number
        for item in captured
    )

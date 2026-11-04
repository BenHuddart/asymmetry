"""Signed windows remain editable and share preview, final and project consumers."""

from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.project.profiles import (
    GoodWindowPolicy,
    profile_fingerprint_for_run,
    resolve_effective_grouping,
)
from asymmetry.core.project.schema import load_project, save_project
from asymmetry.core.simulate import reduce_run_to_dataset
from asymmetry.gui.mainwindow import MainWindow
from asymmetry.gui.tasks import TaskWorker
from asymmetry.gui.windows.grouping.alpha_section import build_alpha_request, run_alpha_estimate
from asymmetry.gui.windows.grouping.dialog import GroupingDialog
from tests.gui.test_grouping_preview_pane import _wait_until
from tests.project.test_raw_window_policy import acquired_run, request

pytestmark = pytest.mark.gui


def dataset_and_profile(policy):
    run = acquired_run(anchors=(10, 15))
    run.grouping.update(groups={1: [1], 2: [2]}, forward_group=1, backward_group=2)
    profile = replace(policy, fingerprint=profile_fingerprint_for_run(run), active=True)
    return reduce_run_to_dataset(run), profile


def test_from_file_controls_rebase_signed_display_after_common_timing_edit(qapp):
    dataset, profile = dataset_and_profile(request())
    dialog = GroupingDialog([dataset], profiles=[profile])
    dialog._set_t0_mode_combo("manual")
    dialog._on_t0_mode_changed()
    dialog._t0_spin.setValue(30)
    assert dialog._resolve_good_bin_limits_from_controls() == (30, -5, 25, 80)
    combo = dialog._good_window_mode_combo
    combo.setCurrentIndex(combo.findData("manual"))
    assert dialog._current_good_window_policy() == GoodWindowPolicy(
        mode="manual", first_offset_bins=-5, last_offset_bins=50
    )
    dialog._clear_dirty()
    dialog.close()


@pytest.mark.parametrize("first,last", [(None, 70), (-10, None)])
def test_partial_manual_endpoints_survive_form_and_t0_edits(qapp, first, last):
    dataset, profile = dataset_and_profile(request(0, first, last, manual=True))
    dialog = GroupingDialog([dataset], profiles=[profile])
    dialog._t0_spin.setValue(20)
    policy = dialog._current_good_window_policy()
    assert (policy.first_offset_bins, policy.last_offset_bins) == (first, last)
    file_first, file_last = 25, 80
    assert dialog._resolve_good_bin_limits_from_controls()[2:] == (
        file_first if first is None else 20 + first,
        file_last if last is None else 20 + last,
    )
    dialog._clear_dirty()
    dialog.close()


def test_signed_preview_final_and_project_reopening(qapp, tmp_path):
    from asymmetry.core.io.nexus import NexusLoader
    from tests.io.test_nexus_good_times import write_good_times

    source = tmp_path / "source.nxs"
    write_good_times(source)
    dataset = NexusLoader().load(source)
    profile = replace(
        request(5, -10, 10, manual=True),
        fingerprint=profile_fingerprint_for_run(dataset.run),
        active=True,
    )
    dialog = GroupingDialog([dataset], profiles=[profile])
    pane = dialog._preview_pane
    pane.flush()
    _wait_until(lambda: pane.corrections_complete)
    assert dialog._t_good_offset_spin.value() == -10
    expected = pane._last_result
    payload = resolve_effective_grouping(profile, dataset.run)
    window = MainWindow()
    assert window._apply_grouping_settings_to_dataset(dataset, payload)[0]
    np.testing.assert_allclose(dataset.time, expected.time)
    np.testing.assert_allclose(dataset.asymmetry, expected.asymmetry)
    window._data_browser.add_dataset(dataset)
    path = tmp_path / "signed.asymp"
    save_project(window.collect_project_state(), path)
    restored = MainWindow()
    restored.restore_project_state(load_project(path), str(path))
    reopened = restored._data_browser.get_dataset(dataset.run_number)
    np.testing.assert_allclose(reopened.time, dataset.time)
    assert reopened.run.grouping["t_good_offset"] == -10
    for view in (window, restored):
        view._clear_all_state()
        view.close()
    dialog._clear_dirty()
    dialog.close()


def test_invalid_saved_manual_request_stays_editable_and_repairs(qapp):
    dataset, profile = dataset_and_profile(request(0, 5000, 6000, manual=True))
    dialog = GroupingDialog([dataset], profiles=[profile])
    assert dialog._t_good_offset_spin.value() == 5000
    assert dialog._last_good_spin.value() == 6015
    pane = dialog._preview_pane
    pane.flush()
    _wait_until(lambda: pane._status.text().startswith("Preview unavailable"))
    assert not pane.corrections_complete
    assert not dialog._apply_btn.isEnabled()
    dialog._t_good_offset_spin.setValue(-5)
    dialog._last_good_spin.setValue(80)
    pane.flush()
    _wait_until(lambda: pane.corrections_complete)
    assert dialog._current_good_window_policy() == GoodWindowPolicy(
        mode="manual", first_offset_bins=-5, last_offset_bins=65
    )
    dialog._clear_dirty()
    dialog.close()


def test_alpha_worker_uses_the_resolved_local_window_after_leading_crop(qapp):
    dataset, profile = dataset_and_profile(request(5, -10, 70, manual=True))
    grouping = resolve_effective_grouping(profile, dataset.run)
    snapshot = build_alpha_request(
        token=1,
        dataset=dataset,
        groups={1: [0], 2: [1]},
        forward_gid=1,
        backward_gid=2,
        excluded_detectors=[],
        method="ratio",
        correction_provider=lambda _: grouping,
        reference_resolver=None,
        facility="",
    )
    result = run_alpha_estimate(TaskWorker(lambda _: None), snapshot)
    # Common t0 20, start 10 and end 90, detector 1 leads by five bins.
    oracle = (
        dataset.run.histograms[0].counts[5:86].sum() / dataset.run.histograms[1].counts[10:91].sum()
    )
    assert result.estimate.alpha == pytest.approx(oracle, rel=1e-15)


def test_released_run_preview_and_calibration_keep_its_explicit_window(qapp):
    dataset, profile = dataset_and_profile(request(0, -10, 75, manual=True))
    dataset.run.grouping = resolve_effective_grouping(profile, dataset.run)
    dataset.run.grouping.pop("good_window_policy")
    dialog = GroupingDialog([dataset], overridden_run_numbers=[dataset.run_number])
    assert dialog._resolve_good_bin_limits_from_controls()[2:] == (5, 90)
    assert dialog._current_good_window_policy() == GoodWindowPolicy(
        mode="manual", first_offset_bins=-10, last_offset_bins=75
    )
    assert "good_window_policy" not in dialog._current_grouping_payload()
    pane = dialog._preview_pane
    pane.flush()
    _wait_until(lambda: pane.corrections_complete)
    expected = reduce_run_to_dataset(dataset.run)
    np.testing.assert_allclose(pane._last_result.time, expected.time)
    np.testing.assert_allclose(pane._last_result.asymmetry, expected.asymmetry)
    grouping = dialog._calibration_correction_provider()(dataset)
    assert (grouping["first_good_bin"], grouping["last_good_bin"]) == (5, 90)
    dialog._clear_dirty()
    dialog.close()


def test_selected_period_controls_use_fresh_detector_file_limits(qapp):
    dataset, profile = dataset_and_profile(request())
    green = acquired_run(anchors=(10, 15), bounds=((40, 60), (40, 60)))
    dataset.run.grouping["period_histograms"] = [dataset.run.histograms, green.histograms]
    dialog = GroupingDialog([dataset], profiles=[replace(profile, period_mode="green")])
    assert dialog._resolve_good_bin_limits_from_controls()[2:] == (45, 60)
    dialog._clear_dirty()
    dialog.close()


@pytest.mark.parametrize("key", ["first_offset_bins", "last_offset_bins"])
@pytest.mark.parametrize("value", [True, 1.9, "2", "bad"])
def test_malformed_project_window_reopens_and_repairs(qapp, tmp_path, key, value):
    from asymmetry.core.io.nexus import NexusLoader
    from tests.io.test_nexus_good_times import write_good_times

    source = tmp_path / "malformed.nxs"
    raw = write_good_times(source)
    dataset = NexusLoader().load(source)
    window = MainWindow()
    window._data_browser.add_dataset(dataset)
    profile = replace(request(), fingerprint=profile_fingerprint_for_run(dataset.run), active=True)
    window._grouping_profiles = [profile]
    state = window.collect_project_state()
    state["grouping_profiles"][0]["good_window_policy"] = {"mode": "manual", key: value}
    project = tmp_path / "malformed.asymp"
    save_project(state, project)
    window.restore_project_state(load_project(project), str(project))
    reopened = window._data_browser.get_dataset(dataset.run_number)
    assert reopened is not None
    for histogram, original in zip(reopened.run.histograms, raw):
        np.testing.assert_array_equal(histogram.counts, original)
    assert "signed integer" in reopened.metadata["correction_error"]
    restored = window._grouping_profiles[0]
    assert restored.to_dict()["good_window_policy"] == {"mode": "manual", key: value}
    dialog = GroupingDialog([reopened], profiles=[restored])
    assert dialog._current_good_window_policy().to_dict() == {"mode": "manual", key: value}
    pane = dialog._preview_pane
    pane.flush()
    _wait_until(lambda: pane._status.text().startswith("Preview unavailable"))
    assert not dialog._apply_btn.isEnabled()
    spin = dialog._t_good_offset_spin if key == "first_offset_bins" else dialog._last_good_spin
    assert repr(value) in spin.lineEdit().text()
    from PySide6.QtTest import QTest

    spin.lineEdit().selectAll()
    QTest.keyClicks(
        spin.lineEdit(), str(1 if key == "first_offset_bins" else 14 + dialog._bin_index_base())
    )
    pane.flush()
    _wait_until(lambda: pane.corrections_complete)
    assert dialog._apply_btn.isEnabled()
    dialog._clear_dirty()
    dialog.close()
    window._clear_all_state()
    window.close()


def test_selected_period_applied_metadata_good_events_and_reopening(qapp, tmp_path):
    import h5py

    from asymmetry.core.io.nexus import NexusLoader
    from asymmetry.core.transform.grouping import good_event_count
    from tests.io.test_nexus_good_times import write_good_times

    source = tmp_path / "selected.nxs"
    red = write_good_times(source)
    green = red + 100
    with h5py.File(source, "a") as handle:
        detector = handle["raw_data_1/instrument/detector_1"]
        for name in ("counts", "first_good_time", "last_good_time"):
            del detector[name]
        handle["raw_data_1"].create_dataset("good_frames", data=[100, 200])
        detector.create_dataset("counts", data=np.stack([red, green]))
        detector.create_dataset("first_good_time", data=[[0.22] * 2, [0.24] * 2])
        detector.create_dataset("last_good_time", data=[[0.30] * 2, [0.32] * 2])
    dataset = NexusLoader().load(source)
    originals = [histogram.counts.copy() for histogram in dataset.run.histograms]
    profile = replace(
        request(),
        fingerprint=profile_fingerprint_for_run(dataset.run),
        active=True,
        period_mode="green",
    )
    window = MainWindow()
    window._grouping_profiles = [profile]
    window._data_browser.add_dataset(dataset)
    payload = dict(
        dataset.run.grouping, period_mode="green", good_window_policy={"mode": "from_file"}
    )
    assert window._apply_grouping_settings_to_dataset(dataset, payload)[0]
    grouping = dataset.run.grouping
    assert (grouping["first_good_bin"], grouping["last_good_bin"]) == (12, 15)
    assert grouping["detector_first_good_bins"] == [12, 12]
    assert grouping["detector_last_good_bins"] == [15, 15]
    assert good_event_count(dataset.run.histograms, grouping) == float(green[:, 12:16].sum())
    assert window._data_browser._resolve_run_info_value(
        dataset, "run_info.events_per_frame"
    ) == pytest.approx(green[:, 12:16].sum() / 200)
    for histogram, original in zip(dataset.run.histograms, originals):
        np.testing.assert_array_equal(histogram.counts, original)
    project = tmp_path / "selected.asymp"
    save_project(window.collect_project_state(), project)
    window.restore_project_state(load_project(project), str(project))
    restored = window._data_browser.get_dataset(dataset.run_number)
    assert (restored.run.grouping["first_good_bin"], restored.run.grouping["last_good_bin"]) == (
        12,
        15,
    )
    assert good_event_count(restored.run.histograms, restored.run.grouping) == float(
        green[:, 12:16].sum()
    )
    window._clear_all_state()
    window.close()


def test_applied_period_metadata_keeps_selected_timing_and_detector_stagger(qapp):
    from asymmetry.core.transform.grouping import good_event_count

    dataset, profile = dataset_and_profile(request(5))
    green = acquired_run(anchors=(20, 25), bounds=((40, 60), (40, 60)))
    dataset.run.grouping["period_histograms"] = [dataset.run.histograms, green.histograms]
    window = MainWindow()
    for mode, anchors, first, last in [
        ("green", (20, 25), 45, 60),
        ("red", (10, 15), 25, 80),
        ("green", (20, 25), 45, 60),
    ]:
        payload = resolve_effective_grouping(replace(profile, period_mode=mode), dataset.run)
        assert window._apply_grouping_settings_to_dataset(dataset, payload)[0]
        grouping = dataset.run.grouping
        assert grouping["detector_t0_bins"] == list(anchors)
        assert grouping["effective_detector_t0_bins"] == [anchor + 5 for anchor in anchors]
        assert (grouping["first_good_bin"], grouping["last_good_bin"]) == (first, last)
        assert grouping["t0_time_us"] == pytest.approx((max(anchors) + 5.24) * 0.01)
        selected = green.histograms if mode == "green" else dataset.run.histograms
        expected = (
            selected[0].counts[first - 5 : last - 4].sum()
            + selected[1].counts[first : last + 1].sum()
        )
        assert good_event_count(dataset.run.histograms, grouping) == expected
    window._clear_all_state()
    window.close()


@pytest.mark.parametrize("mismatched", [False, True])
@pytest.mark.parametrize("mode", ["green_minus_red", "green_plus_red"])
def test_combined_preview_shares_final_period_calculation_and_grid_checks(qapp, mismatched, mode):
    from asymmetry.core.transform.rebin import binning_spec
    from asymmetry.core.transform.reduce import reduce_period_asymmetry

    dataset, profile = dataset_and_profile(request())
    bounds = ((40, 60), (40, 60)) if mismatched else ((20, 80), (20, 80))
    green = acquired_run(anchors=(10, 15), bounds=bounds)
    green.histograms[0].counts *= 2
    dataset.run.grouping["period_histograms"] = [dataset.run.histograms, green.histograms]
    profile = replace(profile, period_mode=mode)
    dialog = GroupingDialog([dataset], profiles=[profile])
    pane = dialog._preview_pane
    pane.flush()
    if mismatched:
        _wait_until(lambda: pane._status.text().startswith("Preview unavailable"))
        assert not pane.corrections_complete
        assert not dialog._apply_btn.isEnabled()
    else:
        _wait_until(lambda: pane.corrections_complete)
        grouping = resolve_effective_grouping(profile, dataset.run)
        result = reduce_period_asymmetry(
            dataset.run.histograms,
            grouping,
            mode,
            binning=binning_spec(grouping),
            forward_idx=[0],
            backward_idx=[1],
            alpha=1,
            use_deadtime=False,
            deadtime_mode="off",
            use_background=False,
        )
        np.testing.assert_allclose(pane._last_result.time, result.time)
        np.testing.assert_allclose(pane._last_result.asymmetry, result.asymmetry)
        np.testing.assert_allclose(pane._last_result.error, result.error)
        assert pane._last_result.counts is None
        pane.set_view("counts")
        assert any("single selected period" in text.get_text() for text in pane._axes.texts)
    dialog._clear_dirty()
    dialog.close()


@pytest.mark.parametrize("source", ["live", "saved", "explicit", "unity"])
def test_combined_preview_reference_exposure_matches_final_and_analytic_counts(qapp, source):
    from asymmetry.core.io import resolve_background_reference
    from asymmetry.core.project.profiles import BackgroundPolicy
    from asymmetry.core.transform.rebin import binning_spec
    from asymmetry.core.transform.reduce import reduce_period_asymmetry

    dataset, profile = dataset_and_profile(request())
    green = acquired_run(anchors=(10, 15))
    reference, _ = dataset_and_profile(request())
    reference.run.run_number = 99
    reference.run.metadata["run_number"] = 99
    reference.metadata["run_number"] = 99
    for histograms, levels in [
        (dataset.run.histograms, (100, 200)),
        (green.histograms, (300, 400)),
        (reference.run.histograms, (10, 20)),
    ]:
        for histogram, level in zip(histograms, levels):
            histogram.counts[:] = level
    payload = {"run_number": 99}
    if source == "live":
        reference.run.grouping["good_frames"] = 50
    elif source == "saved":
        payload.update(good_frames_sample=300, good_frames_reference=50)
    elif source == "explicit":
        payload["scale"] = 3
    dataset.run.grouping.update(
        good_frames=300,
        period_good_frames=[100, 200],
        period_histograms=[dataset.run.histograms, green.histograms],
    )
    profile = replace(
        profile,
        period_mode="green_minus_red",
        background_policy=BackgroundPolicy(
            mode="reference_run", details={"background_run": payload}
        ),
    )
    dialog = GroupingDialog(
        [dataset, reference], profiles=[profile], selected_run_number=dataset.run_number
    )
    pane = dialog._preview_pane
    pane.flush()
    _wait_until(lambda: pane.corrections_complete)
    grouping = resolve_effective_grouping(profile, dataset.run)

    def resolver(selected):
        return resolve_background_reference(
            payload,
            sample_good_frames=selected.get("good_frames"),
            datasets=[reference],
            allow_file=False,
        )

    final = reduce_period_asymmetry(
        dataset.run.histograms,
        grouping,
        "green_minus_red",
        binning=binning_spec(grouping),
        forward_idx=[0],
        backward_idx=[1],
        alpha=1,
        use_deadtime=False,
        deadtime_mode="off",
        use_background=True,
        reference_resolver=resolver,
    )
    scales = (2, 4) if source == "live" else ({"saved": 6, "explicit": 3, "unity": 1}[source],) * 2
    red_f, red_b = 100 - scales[0] * 10, 200 - scales[0] * 20
    green_f, green_b = 300 - scales[1] * 10, 400 - scales[1] * 20
    expected = 100 * ((green_f - green_b) / (green_f + green_b) - (red_f - red_b) / (red_f + red_b))
    np.testing.assert_allclose(pane._last_result.asymmetry, expected, atol=1e-12)
    np.testing.assert_allclose(pane._last_result.asymmetry, final.asymmetry)
    np.testing.assert_allclose(pane._last_result.error, final.error)
    dialog._clear_dirty()
    dialog.close()

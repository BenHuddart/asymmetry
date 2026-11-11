"""Shared calibration staging, conditional presentation and request lifetime."""

from dataclasses import replace
from threading import Event

import pytest
from PySide6.QtCore import QTimer

from asymmetry.core.project.profiles import profile_fingerprint_for_run, profile_from_payload
from asymmetry.core.workflow.calibration import CalibrationOptions, calibrate_run
from asymmetry.core.workflow.reduction import ReductionSettings
from asymmetry.gui.windows.grouping import alpha_section as module
from asymmetry.gui.windows.grouping.alpha_section import AlphaSectionWidget
from tests.core.test_tf_calibration_fit import _raw_tf
from tests.gui.test_grouping_dialog import _wait_until

pytestmark = pytest.mark.gui


def configured(qapp, method="tf_calibration_fit", beta=1):
    data = _raw_tf(beta=beta)
    profile = profile_from_payload(data.run.grouping, "test", profile_fingerprint_for_run(data.run))
    section = AlphaSectionWidget()
    section.configure(
        datasets=[data],
        selected_run_number=data.run_number,
        context_provider=lambda: lambda slot, dataset: (profile, None),
    )
    section._method_combo.setCurrentIndex(section._method_combo.findData(method))
    return section, data, profile


@pytest.mark.parametrize("method", ["tf_calibration_fit", "diamagnetic", "general", "ratio"])
@pytest.mark.parametrize("vector", [False, True])
def test_method_beta_layout(qapp, method, vector):
    section, _, _ = configured(qapp)
    section._fit_beta.setChecked(True)
    section.set_vector(vector, ["pair"])
    section._method_combo.setCurrentIndex(section._method_combo.findData(method))
    assert section._time_response.isEnabled() == (method == "tf_calibration_fit")
    assert section._fit_beta.isEnabled() == (method == "tf_calibration_fit" and not vector)
    if not section._fit_beta.isEnabled():
        assert not section._fit_beta.isChecked()
    section.shutdown()


def test_tf_quick_tf_does_not_restore_free_beta(qapp):
    section, _, _ = configured(qapp)
    section._fit_beta.setChecked(True)
    section._method_combo.setCurrentIndex(section._method_combo.findData("general"))
    assert section._method_combo.currentText() == "Decay balance"
    section._method_combo.setCurrentIndex(0)
    assert section._fit_beta.isEnabled() and not section._fit_beta.isChecked()
    section.shutdown()


@pytest.mark.parametrize("free_beta", [False, True])
def test_result_staged_with_model_and_conditional_errors(qapp, free_beta):
    section, data, _ = configured(qapp, beta=1.13 if free_beta else 1)
    section._fit_beta.setChecked(free_beta)
    section._on_estimate()
    _wait_until(lambda: section._tasks.active_count == 0)
    outcome = section.pending()["single"]
    direct = calibrate_run(
        data, ReductionSettings(), CalibrationOptions("tf_calibration_fit", free_beta)
    )
    assert outcome.record.result == direct.record.result
    assert outcome.selected.envelope.value.title() in section._result_label.text()
    assert "conditional" in section._note_label.text()
    assert ("fixed" in section._result_label.text()) == (not free_beta)
    section.shutdown()


@pytest.mark.parametrize(
    "change", ["invalidate", "method", "beta", "pulse", "separation", "source", "stop", "shutdown"]
)
def test_held_worker_obsolete_results_never_publish_and_heartbeat(qapp, monkeypatch, change):
    section, _, _ = configured(qapp)
    entered, release = Event(), Event()
    original = module.run_calibration

    def held(worker, request):
        entered.set()
        assert release.wait(5)
        return original(worker, request)

    monkeypatch.setattr(module, "run_calibration", held)
    heartbeat = []
    timer = QTimer()
    timer.timeout.connect(lambda: heartbeat.append(1))
    timer.start(5)
    section._on_estimate()
    _wait_until(entered.is_set)
    _wait_until(lambda: len(heartbeat) >= 2)
    if change == "method":
        section._method_combo.setCurrentIndex(1)
    elif change == "beta":
        section._fit_beta.setChecked(True)
    elif change == "pulse":
        section._time_response.mode.setCurrentIndex(1)
    elif change == "separation":
        section._time_response.separation.setText("0.4")
    elif change == "source":
        section.configure(datasets=[], selected_run_number=None, context_provider=lambda: None)
    elif change == "stop":
        section._stop_btn.click()
    elif change == "shutdown":
        section.shutdown()
    else:
        section.invalidate()
    release.set()
    _wait_until(lambda: section._tasks.active_count == 0)
    assert section.pending() == {}
    assert "Computing" not in section._result_label.text()
    assert not section._stop_btn.isEnabled()
    timer.stop()
    section.shutdown()


@pytest.mark.parametrize(
    "dependency",
    [
        "group",
        "projection",
        "exclusion",
        "period",
        "mapping",
        "mapping_draft",
        "window",
        "binning",
        "timing",
        "deadtime",
        "background",
        "reference",
        "manual",
    ],
)
def test_dialog_dependency_edits_invalidate_held_result(qapp, monkeypatch, dependency):
    from asymmetry.gui.windows.grouping.dialog import GroupingDialog
    from docs.screenshots.data.calibration import calibration_control
    from tests.gui.test_grouping_dialog import _wait_for_t0_detection

    dialog = GroupingDialog([calibration_control()])
    _wait_for_t0_detection(dialog)
    section = dialog._alpha_section
    entered, release = Event(), Event()
    original = module.run_calibration

    def held(worker, request):
        entered.set()
        assert release.wait(5)
        return original(worker, request)

    monkeypatch.setattr(module, "run_calibration", held)
    section._on_estimate()
    _wait_until(entered.is_set)
    if dependency == "group":
        dialog._groups[1] = [1]
    elif dependency == "projection":
        dialog._projection_specs = [{"label": "other", "forward_group": 2, "backward_group": 1}]
    elif dependency == "exclusion":
        dialog._exclude_edit.setText("1")
    elif dependency == "period":
        dialog._set_period_mode("green")
    elif dependency == "mapping":
        dialog._run.grouping["period_mapping"] = {1: "green"}
    elif dependency == "mapping_draft":
        dialog.period_mapping_request = {"mapping": {1: "red", 2: "green"}}
    elif dependency == "window":
        dialog._last_good_spin.setValue(dialog._last_good_spin.value() - 1)
    elif dependency == "binning":
        dialog._bunch_spin.setValue(2)
    elif dependency == "timing":
        dialog._t0_spin.setValue(dialog._t0_spin.value() + 1)
    elif dependency == "deadtime":
        dialog._deadtime_section._mode_buttons["manual"].click()
    elif dependency == "background":
        dialog._background_mode = "fixed"
    elif dependency == "reference":
        dialog._background_run_payload = {"run_number": 123}
    else:
        dialog._alpha_spin.setValue(1.7)
    dialog._refresh_preview()
    release.set()
    _wait_until(lambda: section._tasks.active_count == 0)
    assert section.pending() == {}
    assert "Computing calibration" not in section._result_label.text()
    if dependency == "manual":
        assert dialog._alpha_spin.value() == 1.7
    dialog._clear_dirty()
    dialog.close()


@pytest.mark.parametrize("mode", ["off", "from_file", "manual", "estimate"])
@pytest.mark.parametrize("method", ["ratio", "tf_calibration_fit"])
def test_resolved_gui_corrections_match_direct_core(qapp, mode, method):
    from asymmetry.core.project.profiles import BackgroundPolicy, DeadtimePolicy
    from asymmetry.core.workflow.calibration import calibrate_source
    from asymmetry.gui.tasks import TaskWorker
    from tests.core.test_tf_deadtime_calibration import _deadtime_tf

    data = _deadtime_tf(mode="manual")
    profile = profile_from_payload(
        data.run.grouping, "source", profile_fingerprint_for_run(data.run)
    )
    profile.deadtime_policy = DeadtimePolicy(mode=mode, values=[0.02, 0.02], estimated_us=0.02)
    profile.background_policy = BackgroundPolicy(
        mode="fixed", details={"background_fixed_values": [0.0, 0.0]}
    )
    options = CalibrationOptions(
        method, envelope="exponential" if method == "tf_calibration_fit" else None
    )
    reply = module.run_calibration(
        TaskWorker(lambda _: None), module.CalibrationRequest(1, data, profile, options)
    )
    grouping = module.resolve_effective_grouping(profile, data.run)
    direct = calibrate_source(replace(data, run=replace(data.run, grouping=grouping)), options)
    if direct.record:
        assert reply.outcome.record.result == direct.record.result
        assert direct.selected.fit.domains[0].deadtime.requested == mode
    else:
        assert reply.outcome.result == direct.result
        assert direct.source.corrections.deadtime.requested == mode
        assert direct.source.corrections.background.requested == "fixed"


@pytest.mark.parametrize("status", ["unavailable", "unreliable"])
@pytest.mark.parametrize(
    "objective,label", [("poisson", "Poisson deviance/ν"), ("gaussian", "χ²/ν")]
)
def test_usable_alpha_only_fit_without_reliable_error_stays_applicable(
    qapp, monkeypatch, status, objective, label
):
    from asymmetry.core.fitting.likelihood import Objective
    from asymmetry.core.project.calibration import CalibrationRecord
    from asymmetry.core.workflow.calibration_results import CalibrationOutcome
    from tests.project.test_tf_calibration_records import candidate

    result = candidate(fit_beta=False)
    result = replace(
        result,
        fit=replace(
            result.fit,
            objective=Objective(objective),
            covariance=None if status == "unavailable" else result.fit.covariance,
            covariance_accurate=False,
        ),
    )

    def captured(worker, request):
        return module.CalibrationReply(
            request.generation,
            request.slot,
            CalibrationOutcome(CalibrationRecord("captured", result)),
        )

    monkeypatch.setattr(module, "run_calibration", captured)
    section, _, _ = configured(qapp)
    section._on_estimate()
    _wait_until(lambda: section._tasks.active_count == 0)
    assert section.pending()["single"].alpha == result.alpha
    assert "error unavailable" in section._result_label.text()
    assert "1 (fixed)" in section._result_label.text()
    assert status in section._note_label.text()
    assert f"{label} 1.92 (12 dof)" in section._note_label.text()
    assert "Period 2" in section._note_label.text()
    assert "F1/B2" in section._note_label.text()
    section.shutdown()


@pytest.mark.parametrize("period", [1, 2])
@pytest.mark.parametrize("mapped", [False, True])
def test_gui_source_keeps_acquired_or_mapped_set_identity(qapp, period, mapped):
    from asymmetry.core.io.periods import combine_mapped_periods
    from asymmetry.gui.tasks import TaskWorker
    from tests.io.test_tf_calibration_period import _calibration_periods

    if mapped:
        parts = [_raw_tf(beta=1) for _ in range(4)]
        parts[3].run.histograms[1].counts *= 2
        data = combine_mapped_periods(
            parts, {1: "ignore", 2: "red", 3: "red", 4: "green"}, source_run_number=7101
        )
    else:
        data, _ = _calibration_periods()
    profile = profile_from_payload(
        data.run.grouping, "source", profile_fingerprint_for_run(data.run)
    )
    profile.period_mode = "green" if period == 2 else "red"
    options = CalibrationOptions("ratio")
    gui = module.run_calibration(
        TaskWorker(lambda _: None), module.CalibrationRequest(1, data, profile, options)
    ).outcome
    direct = calibrate_run(data, ReductionSettings(), replace(options, period=period))
    assert gui.result == direct.result
    assert gui.period_number == (None if mapped else period)
    assert gui.source.mapped_period_number == (period if mapped else None)
    section = AlphaSectionWidget()
    section.configure(
        datasets=[data],
        selected_run_number=data.run_number,
        context_provider=lambda: lambda slot, dataset: (profile, None),
    )
    section._method_combo.setCurrentIndex(section._method_combo.findData("ratio"))
    section._on_estimate()
    _wait_until(lambda: section._tasks.active_count == 0)
    identity = f"Mapped set {period}" if mapped else f"Period {period}"
    assert section._note_label.text().startswith(identity)
    assert "first-order errors" in section._note_label.text()
    section.shutdown()


@pytest.mark.parametrize("method", ["general", "count_fit"])
def test_historical_values_and_method_survive_open_without_fit(qapp, method):
    from asymmetry.core.project.profiles import AlphaPolicy
    from asymmetry.gui.windows.grouping.dialog import GroupingDialog

    data = _raw_tf(beta=1)
    profile = profile_from_payload(
        data.run.grouping, "saved", profile_fingerprint_for_run(data.run)
    )
    profile.alpha_policy = AlphaPolicy(mode="calibrated", value=1.37, method=method, source_run=42)
    dialog = GroupingDialog([data], profiles=[profile])
    assert dialog._draft.alpha_policy.to_dict() == profile.alpha_policy.to_dict()
    assert dialog._current_alpha_method() == method
    assert dialog._alpha_section._tasks.active_count == 0
    assert dialog._alpha_section._method_combo.currentData() == "tf_calibration_fit"
    dialog._clear_dirty()
    dialog.close()


def test_quick_difference_selects_red_once_without_reapplying_target_period_mode(qapp):
    from asymmetry.core.workflow.reduction import GREEN_MINUS_RED
    from asymmetry.gui.tasks import TaskWorker
    from tests.io.test_tf_calibration_period import _calibration_periods

    data, _ = _calibration_periods()
    profile = profile_from_payload(
        data.run.grouping, "source", profile_fingerprint_for_run(data.run)
    )
    profile.period_mode = GREEN_MINUS_RED
    options = CalibrationOptions("ratio")
    gui = module.run_calibration(
        TaskWorker(lambda _: None), module.CalibrationRequest(1, data, profile, options)
    ).outcome
    direct = calibrate_run(data, ReductionSettings(), replace(options, period=GREEN_MINUS_RED))
    assert gui.result == direct.result


def test_vector_manual_edit_invalidates_held_calibration(qapp, monkeypatch):
    from asymmetry.gui.windows.grouping.dialog import GroupingDialog
    from docs.screenshots.data.calibration import calibration_control
    from tests.gui.test_grouping_dialog import _wait_for_t0_detection

    dialog = GroupingDialog([calibration_control(vector=True)])
    _wait_for_t0_detection(dialog)
    entered, release = Event(), Event()
    original = module.run_calibration

    def held(worker, request):
        entered.set()
        assert release.wait(5)
        return original(worker, request)

    monkeypatch.setattr(module, "run_calibration", held)
    dialog._estimate_alpha_for_axis("P_x")
    _wait_until(entered.is_set)
    dialog._vector_alpha_spins["P_x"].setValue(1.7)
    release.set()
    _wait_until(lambda: dialog._alpha_section._tasks.active_count == 0)
    assert dialog._alpha_section.pending() == {}
    assert dialog._vector_alpha_spins["P_x"].value() == 1.7
    dialog._clear_dirty()
    dialog.close()


@pytest.mark.parametrize("fit_beta", [False, True])
def test_fixed_pulse_gui_worker_matches_shared_selection_and_invalidates(qapp, fit_beta):
    from asymmetry.core.utils.constants import GAUSS_TO_TESLA, MUON_GYROMAGNETIC_RATIO_MHZ_PER_T
    from tests.core.test_tf_pulse_calibration import _acquisition

    data = _acquisition(0.8 if fit_beta else 1)
    data.run.metadata["field"] = 1.3 / MUON_GYROMAGNETIC_RATIO_MHZ_PER_T / GAUSS_TO_TESLA
    profile = profile_from_payload(
        data.run.grouping, "pulse", profile_fingerprint_for_run(data.run)
    )
    section = AlphaSectionWidget()
    section.configure(
        datasets=[data],
        selected_run_number=data.run_number,
        context_provider=lambda: lambda slot, dataset: (profile, None),
    )
    section._fit_beta.setChecked(fit_beta)
    assert section._time_response.value().mode == "off"
    section._time_response.mode.setCurrentIndex(1)
    section._on_estimate()
    _wait_until(lambda: section._tasks.active_count == 0)
    outcome = section.pending()["single"]
    direct = calibrate_run(
        data,
        ReductionSettings(),
        CalibrationOptions("tf_calibration_fit", fit_beta, pulse_separation_us=0.324),
    )
    assert outcome.record.result == direct.record.result
    assert len(outcome.record.result.candidates) == 4
    section._time_response.separation.setText("0.4")
    assert not section.pending()
    section._method_combo.setCurrentIndex(section._method_combo.findData("general"))
    assert not section._time_response.isEnabled()
    section.shutdown()

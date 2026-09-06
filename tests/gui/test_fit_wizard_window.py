"""Tests for the fit wizard window UI."""

from __future__ import annotations

import dataclasses
import os
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.gui, pytest.mark.slow, pytest.mark.integration]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

import asymmetry.gui.windows.fit_wizard_window as wizard_window_module
from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.fit_wizard import (
    CandidateAssessment,
    CandidateTemplate,
    ConfidenceTier,
    FitWizardRecommendation,
    RecommendationVerdict,
    SelectionMetric,
    SpectrumFingerprint,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fitting.peak_detection import (
    DetectedPeak,
    MultipletMatch,
    PeakAnalysis,
)
from asymmetry.gui.windows.fit_wizard_window import (
    _PAGE_RESULT,
    _PAGE_RUNNING,
    _PAGE_WELCOME,
    FitWizardWindow,
)
from tests._qt_helpers import wait_for


def _canned_peak_analysis() -> PeakAnalysis:
    return PeakAnalysis(
        peaks=(
            DetectedPeak(
                frequency_mhz=5.0,
                amplitude=1.0,
                snr=12.0,
                width_mhz=0.3,
                prominence=0.5,
                source="fft",
            ),
            DetectedPeak(
                frequency_mhz=8.0,
                amplitude=0.6,
                snr=6.0,
                width_mhz=0.4,
                prominence=0.3,
                source="fft",
            ),
        ),
        noise_floor=0.05,
        resolution_mhz=0.5,
        nyquist_mhz=50.0,
        detrended=True,
    )


def _canned_multiplet_match() -> MultipletMatch:
    return MultipletMatch(
        kind="fmuf_linear",
        family_key="fmuf",
        peak_indices=(0, 1),
        quality=0.9,
        derived_values=(("r_muF_angstrom", 1.17),),
        note="lines match the collinear F-mu-F signature",
    )


def _recommendation_with_peaks(dataset: MuonDataset) -> FitWizardRecommendation:
    return dataclasses.replace(
        _fake_recommendation(dataset),
        peak_analysis=_canned_peak_analysis(),
        multiplet_matches=(_canned_multiplet_match(),),
    )


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def dataset() -> MuonDataset:
    t = np.linspace(0.0, 8.0, 120)
    y = 0.2 * np.exp(-0.4 * t) + 0.01
    e = np.full_like(t, 0.01)
    return MuonDataset(time=t, asymmetry=y, error=e, metadata={"run_number": 101})


def _fake_recommendation(dataset: MuonDataset) -> FitWizardRecommendation:
    exp_model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    gauss_model = CompositeModel(["Gaussian", "Constant"], operators=["+"])

    exp_params = ParameterSet(
        [
            Parameter("A_1", value=0.2, min=0.0, max=1.0),
            Parameter("Lambda", value=0.4, min=0.0, max=5.0),
            Parameter("A_bg", value=0.01, min=0.0, max=0.5),
        ]
    )
    exp_curve = exp_model.function(dataset.time, A_1=0.2, Lambda=0.4, A_bg=0.01)
    exp_result = FitResult(
        success=True,
        chi_squared=5.0,
        reduced_chi_squared=0.05,
        parameters=exp_params,
        uncertainties={"A_1": 0.01, "Lambda": 0.02, "A_bg": 0.001},
        residuals=np.asarray(dataset.asymmetry - exp_curve, dtype=float),
        message="ok",
    )

    gauss_params = ParameterSet(
        [
            Parameter("A_1", value=0.18, min=0.0, max=1.0),
            Parameter("sigma", value=0.6, min=0.0, max=5.0),
            Parameter("A_bg", value=0.02, min=0.0, max=0.5),
        ]
    )
    gauss_curve = gauss_model.function(dataset.time, A_1=0.18, sigma=0.6, A_bg=0.02)
    gauss_result = FitResult(
        success=True,
        chi_squared=9.0,
        reduced_chi_squared=0.09,
        parameters=gauss_params,
        uncertainties={"A_1": 0.02, "sigma": 0.03, "A_bg": 0.002},
        residuals=np.asarray(dataset.asymmetry - gauss_curve, dtype=float),
        message="ok",
    )

    fingerprint = SpectrumFingerprint(
        tail_estimate=0.01,
        initial_amplitude_estimate=0.2,
        zero_crossings=0,
        smoothed_zero_crossings=0,
        smoothed_turning_points=0,
        dominant_fft_frequency_mhz=0.0,
        dominant_fft_snr=0.0,
        dominant_fft_cycles_in_window=0.0,
        monotonic_decay_fraction=1.0,
        early_time_curvature=-0.1,
        semilog_slope_ratio=1.0,
        late_time_dip_recovery_score=0.0,
        oscillatory_hint=False,
        kt_like_hint=False,
        multi_rate_hint=False,
    )

    exp_template = CandidateTemplate(
        key="exp_constant",
        title="Exponential + Constant",
        category="General",
        rationale="Baseline single-relaxation model.",
        model=exp_model,
    )
    gauss_template = CandidateTemplate(
        key="gaussian_constant",
        title="Gaussian + Constant",
        category="General",
        rationale="Alternative Gaussian envelope.",
        model=gauss_model,
    )

    exp_assessment = CandidateAssessment(
        template=exp_template,
        fit_result=exp_result,
        aic=8.0,
        aicc=8.2,
        bic=10.0,
        selected_score=8.2,
        residual_rms=0.9,
        runs_z_score=0.2,
        max_abs_autocorrelation=0.1,
        residual_fft_peak_snr=1.2,
        residual_gate_passed=True,
        residual_gate_reasons=(),
        bound_hits=(),
        fitted_time=np.asarray(dataset.time, dtype=float).copy(),
        fitted_curve=np.asarray(exp_curve, dtype=float),
        component_curves=tuple(
            exp_model.evaluate_components(
                dataset.time, additive_only=True, A_1=0.2, Lambda=0.4, A_bg=0.01
            )
        ),
    )
    gauss_assessment = CandidateAssessment(
        template=gauss_template,
        fit_result=gauss_result,
        aic=12.0,
        aicc=12.2,
        bic=14.0,
        selected_score=12.2,
        residual_rms=1.5,
        runs_z_score=2.5,
        max_abs_autocorrelation=0.4,
        residual_fft_peak_snr=7.0,
        residual_gate_passed=False,
        residual_gate_reasons=("runs-test z score suggests structure (2.50)",),
        bound_hits=(),
        fitted_time=np.asarray(dataset.time, dtype=float).copy(),
        fitted_curve=np.asarray(gauss_curve, dtype=float),
        component_curves=tuple(
            gauss_model.evaluate_components(
                dataset.time, additive_only=True, A_1=0.18, sigma=0.6, A_bg=0.02
            )
        ),
    )
    return FitWizardRecommendation(
        fingerprint=fingerprint,
        templates=(exp_template, gauss_template),
        assessments=(exp_assessment, gauss_assessment),
        metric=SelectionMetric.AICC,
        recommended_key="exp_constant",
        comparable_keys=(),
        summary="Recommended: Exponential + Constant by AICc.",
    )


def _analysis_complete(window: FitWizardWindow) -> bool:
    return window._recommendation is not None and window._tasks.active_count == 0


def test_fit_wizard_window_populates_banners_and_tables(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        wizard_window_module,
        "build_fit_wizard_recommendation",
        lambda dataset, current_model=None, metric=SelectionMetric.AICC, **kwargs: (
            _fake_recommendation(dataset)
        ),
    )
    window = FitWizardWindow()

    window.set_analysis_context(dataset)
    # Welcome state: no result populated yet.
    assert window._stack.currentIndex() == _PAGE_WELCOME
    assert window._compare_table.rowCount() == 0

    window._start_analysis()
    wait_for(lambda: _analysis_complete(window), qapp)

    # Result state: card + deep panels populated.
    assert window._stack.currentIndex() == _PAGE_RESULT
    assert window._answer_card._verdict_label.text()
    assert window._fingerprint_banner.text()
    assert window._compare_table.rowCount() == 2
    # The six-step decision trail is rendered below the card.
    assert window._result_trail.step_keys() == (
        "conditions",
        "families",
        "spectrum",
        "candidates",
        "verdict",
        "confidence",
    )


def test_fit_wizard_window_failed_refresh_clears_recommendation(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refresh that fails must clear the stale recommendation and prefix the error.

    Regression: after the WizardWindowBase unification the single wizard briefly
    inherited the base default failure hook, which set only the raw message —
    dropping the "Fit wizard analysis failed:" prefix (leaving it inconsistent
    with the global wizard) and leaving the previous successful recommendation
    live, so the still-enabled metric combo could resurrect stale results.
    """
    monkeypatch.setattr(
        wizard_window_module,
        "build_fit_wizard_recommendation",
        lambda dataset, current_model=None, metric=SelectionMetric.AICC, **kwargs: (
            _fake_recommendation(dataset)
        ),
    )
    window = FitWizardWindow()
    window.set_analysis_context(dataset)
    window._start_analysis()
    wait_for(lambda: _analysis_complete(window), qapp)
    assert window._recommendation is not None
    assert window._metric_combo.isEnabled()

    def _boom(dataset, current_model=None, metric=SelectionMetric.AICC, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(wizard_window_module, "build_fit_wizard_recommendation", _boom)
    window._start_analysis()
    wait_for(lambda: not window._analysis_in_progress and window._tasks.active_count == 0, qapp)

    assert window._recommendation is None
    assert window._status_label.text() == "Fit wizard analysis failed: boom"
    assert not window._metric_combo.isEnabled()


def test_fit_wizard_window_metric_info_dialog_contains_expected_text(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    def _fake_information(_parent, title: str, text: str) -> None:
        captured["title"] = title
        captured["text"] = text

    monkeypatch.setattr(QMessageBox, "information", _fake_information)
    window = FitWizardWindow()
    window._show_metric_info()

    assert captured["title"] == "Fit Wizard Metrics"
    assert "AICc" in captured["text"]
    assert "BIC" in captured["text"]


def test_fit_wizard_window_selection_updates_apply_page(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        wizard_window_module,
        "build_fit_wizard_recommendation",
        lambda dataset, current_model=None, metric=SelectionMetric.AICC, **kwargs: (
            _fake_recommendation(dataset)
        ),
    )
    window = FitWizardWindow()
    window.set_analysis_context(dataset)
    window._start_analysis()
    wait_for(lambda: _analysis_complete(window), qapp)

    window._compare_table.selectRow(1)
    qapp.processEvents()

    # Selecting a candidate in the compare table swaps the card's selection and
    # updates the residual-warning panel for that candidate.
    assert window._selected_key == "gaussian_constant"
    assert window._answer_card.selected_key() == "gaussian_constant"
    assert "Residual gate warning" in window._compare_warning_text.toPlainText()


def test_fit_wizard_window_apply_recommended_emits_assessment(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        wizard_window_module,
        "build_fit_wizard_recommendation",
        lambda dataset, current_model=None, metric=SelectionMetric.AICC, **kwargs: (
            _fake_recommendation(dataset)
        ),
    )
    window = FitWizardWindow()
    window.set_analysis_context(dataset)
    window._start_analysis()
    wait_for(lambda: _analysis_complete(window), qapp)

    emitted: dict[str, object] = {}
    window.apply_assessment_requested.connect(
        lambda assessment, recommendation: emitted.update(
            {"assessment": assessment, "recommendation": recommendation}
        )
    )

    # The answer card's "Apply this fit" applies the selected (default =
    # recommended) assessment; the window relays it with the recommendation.
    window._answer_card._on_apply_clicked()

    assert emitted["assessment"].template.key == "exp_constant"
    assert emitted["recommendation"].recommended_key == "exp_constant"


def test_fit_wizard_window_shows_progress_while_analysis_runs(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _slow_recommendation(dataset, current_model=None, metric=SelectionMetric.AICC, **kwargs):
        time.sleep(0.05)
        return _fake_recommendation(dataset)

    monkeypatch.setattr(
        wizard_window_module,
        "build_fit_wizard_recommendation",
        _slow_recommendation,
    )
    window = FitWizardWindow()
    window.set_analysis_context(dataset)

    window._start_analysis()

    assert window._analysis_in_progress is True
    assert window._progress_bar.isHidden() is False
    wait_for(
        lambda: window._analysis_in_progress is False and window._tasks.active_count == 0, qapp
    )
    assert window._progress_bar.isHidden() is True


def test_fit_wizard_window_emits_cached_analysis_payload(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        wizard_window_module,
        "build_fit_wizard_recommendation",
        lambda dataset, current_model=None, metric=SelectionMetric.AICC, **kwargs: (
            _fake_recommendation(dataset)
        ),
    )
    window = FitWizardWindow()
    payload: dict[str, object] = {}
    window.analysis_cached.connect(
        lambda recommendation, log_text, signature: payload.update(
            {
                "recommendation": recommendation,
                "log_text": log_text,
                "signature": signature,
            }
        )
    )

    window.set_analysis_context(dataset)
    window._start_analysis()
    wait_for(lambda: _analysis_complete(window), qapp)

    assert isinstance(payload.get("recommendation"), FitWizardRecommendation)
    assert payload.get("log_text") == ""
    assert payload.get("signature") == {
        "run_number": int(dataset.run_number),
        "model": None,
        "scope": {"version": 1, "preset": "auto", "include": [], "exclude": []},
        "user_peaks": [],
    }


def test_fit_wizard_window_accepts_cached_recommendation(
    qapp: QApplication,
    dataset: MuonDataset,
) -> None:
    window = FitWizardWindow()
    recommendation = _fake_recommendation(dataset)

    window.set_analysis_context(dataset)
    window.set_cached_recommendation(
        recommendation,
        signature={"run_number": int(dataset.run_number), "model": None},
        log_text="cached log",
    )

    assert window.current_recommendation() is recommendation
    assert window.current_log_text() == "cached log"
    # Cached reopen goes straight to the Result state.
    assert window._stack.currentIndex() == _PAGE_RESULT
    assert window._compare_table.rowCount() == 2
    assert window._answer_card.selected_key() == "exp_constant"
    # Legacy signature (no scope/user_peaks keys) restores Auto and is not stale.
    assert window._scope_selector.current_scope()["preset"] == "auto"
    assert window._user_peaks == []
    # The result page is a scroll area (see test_fit_wizard_window_result_page_is_scrollable)
    # so an expanded trail step can never push content past the window unreachably.
    assert window._stack.widget(_PAGE_RESULT) is window._result_scroll
    assert window._analysis_stale is False
    assert window._stale_banner.isHidden() is True


def test_fit_wizard_window_result_page_is_scrollable(
    qapp: QApplication,
    dataset: MuonDataset,
) -> None:
    """The Result page is a QScrollArea so an expanded trail step (scope
    selector, FFT panel, compare table) can always be reached even if it pushes
    content past the window's height."""
    from PySide6.QtWidgets import QFrame, QScrollArea

    window = FitWizardWindow()
    recommendation = _fake_recommendation(dataset)
    window.set_analysis_context(dataset)
    window.set_cached_recommendation(recommendation)

    scroll = window._stack.widget(_PAGE_RESULT)
    assert isinstance(scroll, QScrollArea)
    assert scroll is window._result_scroll
    assert scroll.widgetResizable() is True
    assert scroll.frameShape() == QFrame.Shape.NoFrame
    assert scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    # The answer card and trail live inside the scroll area's content widget.
    content = scroll.widget()
    assert content is not None
    assert content.isAncestorOf(window._answer_card)
    assert content.isAncestorOf(window._result_trail)


# ── Resolver adapter shape ───────────────────────────────────────────────────


def test_fit_wizard_window_resolver_groups_time_domain_by_category(
    qapp: QApplication,
    dataset: MuonDataset,
) -> None:
    from asymmetry.core.fitting.composite import COMPONENTS

    window = FitWizardWindow()
    window.set_analysis_context(dataset)

    result = window._resolve_scope("auto", {"include": [], "exclude": []})
    families = result["families"]
    names = [c["name"] for f in families for c in f["components"]]

    assert names
    # Frequency-domain components are skipped entirely.
    assert all(COMPONENTS[name].domain == "time" for name in names)
    # Each family is titled by the component category it groups.
    for family in families:
        first = family["components"][0]["name"]
        assert family["title"] == COMPONENTS[first].category
        assert all(COMPONENTS[c["name"]].category == family["title"] for c in family["components"])
    # Estimate is a two-element (candidates, fits) pair.
    assert len(result["estimate"]) == 2


def test_fit_wizard_window_resolver_guards_without_dataset(qapp: QApplication) -> None:
    window = FitWizardWindow()
    result = window._resolve_scope("auto", {"include": [], "exclude": []})
    assert result["families"] == []
    assert result["note"] == "Load a dataset first"


# ── Scope tab: ordering + selection ──────────────────────────────────────────


def test_fit_wizard_window_opens_on_welcome_with_collapsed_guidance(
    qapp: QApplication,
    dataset: MuonDataset,
) -> None:
    window = FitWizardWindow()
    # No tabs on the new window: the base tab scaffolding is not built.
    assert window._tabs is None
    window.set_analysis_context(dataset)
    # Opens on the Welcome state with the guidance section collapsed by default.
    assert window._stack.currentIndex() == _PAGE_WELCOME
    assert window._guidance_section.isExpanded() is False
    # The scope selector lives inside the (collapsed) guidance section.
    assert window._scope_panel.parent() is window._guidance_scope_slot


# ── Fingerprint plot decimation (GUI-thread stall guard) ─────────────────────


def test_fit_wizard_window_fingerprint_plot_bounds_point_count(qapp: QApplication) -> None:
    """A huge dataset's errorbar draw stays bounded, not O(points).

    Regression guard for the diagnosed stall: opening the wizard on a long
    high-resolution run drew a matplotlib ``errorbar`` over every raw point
    synchronously on the GUI thread, freezing it for seconds (same disease
    PR #229 fixed for the grouping preview pane). ``set_analysis_context`` ->
    ``_populate_fingerprint_plot`` must decimate before drawing.
    """
    n = 50_000
    t = np.linspace(0.0, 8.0, n)
    y = 0.2 * np.exp(-0.4 * t) + 0.01
    e = np.full_like(t, 0.01)
    big_dataset = MuonDataset(time=t, asymmetry=y, error=e, metadata={"run_number": 202})

    window = FitWizardWindow()
    window.set_analysis_context(big_dataset)

    ax_time = window._fingerprint_plot_widget._figure.axes[0]
    # errorbar(fmt=".") draws the markers as a Line2D on the axes.
    (data_line,) = ax_time.lines
    plotted_points = len(data_line.get_xdata())
    assert plotted_points <= wizard_window_module._MAX_FINGERPRINT_POINTS
    assert plotted_points < n


# ── Scope + user peaks forwarded to the builder ──────────────────────────────


def test_fit_wizard_window_forwards_scope_and_user_peaks(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from asymmetry.core.fitting.wizard_scope import WizardScope, WizardScopePreset

    captured: dict[str, object] = {}

    def _capture(dataset, current_model=None, metric=SelectionMetric.AICC, **kwargs):
        captured.update(kwargs)
        return _fake_recommendation(dataset)

    monkeypatch.setattr(wizard_window_module, "build_fit_wizard_recommendation", _capture)
    window = FitWizardWindow()
    window.set_analysis_context(dataset)

    window._scope_selector.set_scope(
        {"version": 1, "preset": "lf-dynamics", "include": [], "exclude": []}
    )
    window._user_peaks = [{"freq_mhz": 3.5}, {"freq_mhz": 12.0}]

    window._start_analysis()
    wait_for(lambda: _analysis_complete(window), qapp)

    scope = captured.get("scope")
    assert isinstance(scope, WizardScope)
    assert scope.preset is WizardScopePreset.LF_DYNAMICS
    assert captured.get("user_frequencies_mhz") == [3.5, 12.0]
    # The cooperative cancel_callback is threaded through to the engine.
    assert callable(captured.get("cancel_callback"))


# ── Staleness after a completed analysis ─────────────────────────────────────


def test_fit_wizard_window_scope_change_marks_stale(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        wizard_window_module,
        "build_fit_wizard_recommendation",
        lambda dataset, current_model=None, metric=SelectionMetric.AICC, **kwargs: (
            _fake_recommendation(dataset)
        ),
    )
    window = FitWizardWindow()
    window.set_analysis_context(dataset)
    window._start_analysis()
    wait_for(lambda: _analysis_complete(window), qapp)

    previous = window.current_recommendation()
    assert window._stale_banner.isHidden() is True

    # Toggle scope via the selector's scope_changed emission.
    window._scope_selector.set_scope(None)
    window._scope_selector._preset_combo.setCurrentIndex(
        window._scope_selector._preset_combo.findData("lf-dynamics")
    )
    qapp.processEvents()

    assert window._analysis_stale is True
    assert window._stale_banner.isHidden() is False
    assert window._refresh_btn.text() == "Re-run Analysis"
    # Old recommendation still displayed.
    assert window.current_recommendation() is previous
    assert window._compare_table.rowCount() == 2


# ── Mid-run scope change discards the result ─────────────────────────────────


def test_fit_wizard_window_mid_run_scope_change_discards_result(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = threading.Event()

    def _blocking(dataset, current_model=None, metric=SelectionMetric.AICC, **kwargs):
        release.wait(timeout=5.0)
        return _fake_recommendation(dataset)

    monkeypatch.setattr(wizard_window_module, "build_fit_wizard_recommendation", _blocking)
    window = FitWizardWindow()
    window.set_analysis_context(dataset)

    window._start_analysis()
    assert window._analysis_in_progress is True

    # Change scope mid-run: orphans the in-flight request and clears busy.
    window._mark_analysis_stale("Scope changed")
    assert window._analysis_in_progress is False

    release.set()
    wait_for(lambda: window._tasks.active_count == 0, qapp)

    # The blocked result was discarded — no recommendation was applied.
    assert window.current_recommendation() is None
    assert window._analysis_in_progress is False


# ── Start disabled when scope is invalid ─────────────────────────────────────


def test_fit_wizard_window_start_disabled_when_scope_invalid(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = FitWizardWindow()
    window.set_analysis_context(dataset)
    assert window._refresh_btn.isEnabled() is True

    monkeypatch.setattr(window._scope_selector, "is_valid", lambda: False)
    window._on_scope_validity_changed(False)

    assert window._refresh_btn.isEnabled() is False
    assert "at least one candidate family" in window._status_label.text()


# ── Cached restore with scope + peaks ────────────────────────────────────────


def test_fit_wizard_window_cached_restore_with_scope_and_peaks(
    qapp: QApplication,
    dataset: MuonDataset,
) -> None:
    window = FitWizardWindow()
    recommendation = _fake_recommendation(dataset)
    window.set_analysis_context(dataset)

    window.set_cached_recommendation(
        recommendation,
        signature={
            "run_number": int(dataset.run_number),
            "model": None,
            "scope": {"version": 1, "preset": "lf-dynamics", "include": [], "exclude": []},
            "user_peaks": [{"freq_mhz": 4.0}],
        },
        log_text="cached",
    )

    assert window._scope_selector.current_scope()["preset"] == "lf-dynamics"
    assert window._user_peaks == [{"freq_mhz": 4.0}]
    assert window._analysis_stale is False
    assert window._stale_banner.isHidden() is True
    # The restored user peak repopulates the peaks table as a user row.
    sources = [window._peaks_table.item(r, 5).text() for r in range(window._peaks_table.rowCount())]
    assert "user" in sources
    freqs = [
        window._peaks_table.item(r, 0).data(_user_role())
        for r in range(window._peaks_table.rowCount())
    ]
    assert any(abs(f - 4.0) < 1e-9 for f in freqs)


# ── Peak table + interactive FFT editing ─────────────────────────────────────


def _user_role():
    return Qt.ItemDataRole.UserRole


def _display_role():
    return Qt.ItemDataRole.DisplayRole


def _press(window: FitWizardWindow, *, xdata: float, x: float = 100.0, y: float = 100.0):
    return SimpleNamespace(inaxes=window._fft_ax, button=1, x=x, y=y, xdata=xdata)


def _release(button: int = 1):
    return SimpleNamespace(inaxes=None, button=button, x=0.0, y=0.0, xdata=None)


def test_fit_wizard_window_fft_click_adds_and_removes_user_peak(
    qapp: QApplication,
    dataset: MuonDataset,
) -> None:
    window = FitWizardWindow()
    window.set_analysis_context(dataset)
    window._fingerprint_plot_widget._canvas.draw()

    # A click inside the FFT axes at 2.2 MHz adds a user peak.
    px = float(window._fft_ax.transData.transform((2.2, 0.0))[0])
    window._on_fft_press(_press(window, xdata=2.2, x=px))
    window._on_fft_release(_release())

    assert any(abs(p["freq_mhz"] - 2.2) < 1e-9 for p in window._user_peaks)
    sources = [window._peaks_table.item(r, 5).text() for r in range(window._peaks_table.rowCount())]
    assert "user" in sources

    # A second click at the same frequency removes it (within ~12 device px).
    px2 = float(window._fft_ax.transData.transform((2.2, 0.0))[0])
    window._on_fft_press(_press(window, xdata=2.2, x=px2))
    window._on_fft_release(_release())

    assert not any(abs(p["freq_mhz"] - 2.2) < 1e-9 for p in window._user_peaks)


def test_fit_wizard_window_fft_click_marks_stale_with_recommendation(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        wizard_window_module,
        "build_fit_wizard_recommendation",
        lambda dataset, current_model=None, metric=SelectionMetric.AICC, **kwargs: (
            _recommendation_with_peaks(dataset)
        ),
    )
    window = FitWizardWindow()
    window.set_analysis_context(dataset)
    window._start_analysis()
    wait_for(lambda: _analysis_complete(window), qapp)
    window._fingerprint_plot_widget._canvas.draw()

    assert window._stale_banner.isHidden() is True

    px = float(window._fft_ax.transData.transform((2.2, 0.0))[0])
    window._on_fft_press(_press(window, xdata=2.2, x=px))
    window._on_fft_release(_release())

    assert window._analysis_stale is True
    assert window._stale_banner.isHidden() is False


def test_fit_wizard_window_fft_click_motion_disarms(
    qapp: QApplication,
    dataset: MuonDataset,
) -> None:
    window = FitWizardWindow()
    window.set_analysis_context(dataset)
    window._fingerprint_plot_widget._canvas.draw()

    window._on_fft_press(_press(window, xdata=2.2, x=100.0, y=100.0))
    window._on_fft_motion(SimpleNamespace(inaxes=window._fft_ax, button=1, x=140.0, y=100.0))
    window._on_fft_release(_release())

    assert window._user_peaks == []


def test_fit_wizard_window_peaks_table_shows_pattern_and_source(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        wizard_window_module,
        "build_fit_wizard_recommendation",
        lambda dataset, current_model=None, metric=SelectionMetric.AICC, **kwargs: (
            _recommendation_with_peaks(dataset)
        ),
    )
    window = FitWizardWindow()
    window.set_analysis_context(dataset)
    window._start_analysis()
    wait_for(lambda: _analysis_complete(window), qapp)

    assert window._peaks_table.rowCount() == 2
    # First peak (index 0) carries the F-mu-F pattern label; source is "fft".
    assert "F-mu-F triplet" in window._peaks_table.item(0, 4).text()
    assert window._peaks_table.item(0, 5).text() == "fft"
    # Auto rows are not removable.
    window._peaks_table.selectRow(0)
    qapp.processEvents()
    assert window._remove_peak_btn.isEnabled() is False
    # The best match note is appended to the fingerprint banner.
    assert "collinear F-mu-F" in window._fingerprint_banner.text()


def test_fit_wizard_window_remove_button_removes_user_peak(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        wizard_window_module,
        "build_fit_wizard_recommendation",
        lambda dataset, current_model=None, metric=SelectionMetric.AICC, **kwargs: (
            _recommendation_with_peaks(dataset)
        ),
    )
    window = FitWizardWindow()
    window.set_analysis_context(dataset)
    window._start_analysis()
    wait_for(lambda: _analysis_complete(window), qapp)
    window._fingerprint_plot_widget._canvas.draw()

    # Add a distinct user peak well away from the detected lines.
    px = float(window._fft_ax.transData.transform((15.0, 0.0))[0])
    window._on_fft_press(_press(window, xdata=15.0, x=px))
    window._on_fft_release(_release())
    assert any(abs(p["freq_mhz"] - 15.0) < 1e-9 for p in window._user_peaks)

    # Select the user row and remove it via the button.
    user_row = next(
        r
        for r in range(window._peaks_table.rowCount())
        if window._peaks_table.item(r, 5).text() == "user"
    )
    window._peaks_table.selectRow(user_row)
    qapp.processEvents()
    assert window._remove_peak_btn.isEnabled() is True
    window._remove_selected_peak()

    assert not any(abs(p["freq_mhz"] - 15.0) < 1e-9 for p in window._user_peaks)


def test_fit_wizard_window_pre_analysis_click_works(
    qapp: QApplication,
    dataset: MuonDataset,
) -> None:
    window = FitWizardWindow()
    window.set_analysis_context(dataset)
    window._fingerprint_plot_widget._canvas.draw()

    # No recommendation yet; the FFT axes still render and accept clicks.
    assert window.current_recommendation() is None
    assert window._fft_ax is not None

    px = float(window._fft_ax.transData.transform((3.3, 0.0))[0])
    window._on_fft_press(_press(window, xdata=3.3, x=px))
    window._on_fft_release(_release())

    assert any(abs(p["freq_mhz"] - 3.3) < 1e-9 for p in window._user_peaks)
    sources = [window._peaks_table.item(r, 5).text() for r in range(window._peaks_table.rowCount())]
    assert sources == ["user"]
    # Pre-analysis: no recommendation, so no stale banner (correct behaviour).
    assert window._stale_banner.isHidden() is True


def test_fit_wizard_window_click_ignored_without_dataset(
    qapp: QApplication,
) -> None:
    window = FitWizardWindow()
    # No dataset loaded: _fft_ax is None and clicks are ignored.
    window._on_fft_press(SimpleNamespace(inaxes=None, button=1, x=1.0, y=1.0, xdata=2.0))
    window._on_fft_release(_release())
    assert window._user_peaks == []


# ── Cooperative cancel ───────────────────────────────────────────────────────


def test_cancel_button_aborts_analysis(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from asymmetry.core.fitting.engine import FitCancelledError

    def blocking_builder(_dataset, **kwargs):
        cancel = kwargs.get("cancel_callback")
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if cancel is not None and cancel():
                raise FitCancelledError("cancelled")
            time.sleep(0.01)
        raise AssertionError("cancel_callback was never observed")

    monkeypatch.setattr(wizard_window_module, "build_fit_wizard_recommendation", blocking_builder)
    window = FitWizardWindow()
    window.set_analysis_context(dataset, None)
    window._start_analysis()
    wait_for(lambda: window._analysis_in_progress, qapp)

    # Cancel is cooperative + asynchronous now: the base only flags the worker;
    # busy/status clear later when the engine observes the flag, raises
    # FitCancelledError, and the base's cancelled-slot runs. So wait, don't assert
    # synchronously (unlike the old bespoke-QThread window).
    window._cancel_current_analysis()
    wait_for(lambda: not window._analysis_in_progress and window._tasks.active_count == 0, qapp)
    assert "cancelled" in window._status_label.text().lower()
    assert window._recommendation is None
    window.close()


def test_cancel_button_visibility_tracks_busy(qapp: QApplication, dataset: MuonDataset) -> None:
    window = FitWizardWindow()
    window.set_analysis_context(dataset, None)
    assert not window._cancel_btn.isVisibleTo(window)
    window._set_busy(True)
    assert window._cancel_btn.isVisibleTo(window)
    window._set_busy(False)
    assert not window._cancel_btn.isVisibleTo(window)
    window.close()


# ── Confidence tier / verdict / caveat display ───────────────────────────────


def _recommendation_with_confidence(
    dataset: MuonDataset,
    *,
    confidence: ConfidenceTier,
    verdict: RecommendationVerdict,
    caveat: str,
) -> FitWizardRecommendation:
    return dataclasses.replace(
        _fake_recommendation(dataset),
        confidence=confidence,
        verdict=verdict,
        caveat=caveat,
    )


def test_fit_wizard_window_high_confidence_reads_as_confident(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        wizard_window_module,
        "build_fit_wizard_recommendation",
        lambda dataset, current_model=None, metric=SelectionMetric.AICC, **kwargs: (
            _recommendation_with_confidence(
                dataset,
                confidence=ConfidenceTier.HIGH,
                verdict=RecommendationVerdict.STRUCTURED,
                caveat="",
            )
        ),
    )
    window = FitWizardWindow()
    window.set_analysis_context(dataset)
    window._start_analysis()
    wait_for(lambda: _analysis_complete(window), qapp)

    # Confidence lives on the answer card, in words, never buried.
    assert "High confidence" in window._answer_card._confidence_label.text()
    # The verdict reads as a plain-physics result, not a failure.
    assert "failed" not in window._answer_card._verdict_label.text().lower()


def test_fit_wizard_window_medium_confidence_shows_caveat_not_error(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        wizard_window_module,
        "build_fit_wizard_recommendation",
        lambda dataset, current_model=None, metric=SelectionMetric.AICC, **kwargs: (
            _recommendation_with_confidence(
                dataset,
                confidence=ConfidenceTier.MEDIUM,
                verdict=RecommendationVerdict.STRUCTURED,
                caveat="Structured residuals remain: runs-test flagged this fit.",
            )
        ),
    )
    window = FitWizardWindow()
    window.set_analysis_context(dataset)
    window._start_analysis()
    wait_for(lambda: _analysis_complete(window), qapp)

    # The medium-confidence caveat is on the card, in plain words, never buried.
    text = window._answer_card._confidence_label.text()
    assert "Medium confidence" in text
    assert "Structured residuals remain" in text
    # Framed as usable-with-caveat, not a failure.
    assert "failed" not in text.lower()


def test_fit_wizard_window_no_significant_structure_is_unmissable(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        wizard_window_module,
        "build_fit_wizard_recommendation",
        lambda dataset, current_model=None, metric=SelectionMetric.AICC, **kwargs: (
            _recommendation_with_confidence(
                dataset,
                confidence=ConfidenceTier.NONE,
                verdict=RecommendationVerdict.NO_SIGNIFICANT_STRUCTURE,
                caveat="The data show no significant structure beyond the null.",
            )
        ),
    )
    window = FitWizardWindow()
    window.set_analysis_context(dataset)
    window._start_analysis()
    wait_for(lambda: _analysis_complete(window), qapp)

    # The null verdict is presented as a RESULT on the card, not a failure.
    verdict = window._answer_card._verdict_label.text().lower()
    confidence = window._answer_card._confidence_label.text().lower()
    assert "simple decay" in verdict
    assert "no oscillation" in verdict
    assert "failed" not in verdict
    assert "failed" not in confidence
    assert "simple decay" in confidence


def test_fit_wizard_window_marks_disqualified_and_null_in_compare_table(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _fake_recommendation(dataset)
    exp, gauss = base.assessments
    # Turn the exp assessment into a null baseline, and disqualify the gauss one.
    exp = dataclasses.replace(exp, is_null_baseline=True)
    gauss = dataclasses.replace(
        gauss, disqualification_reasons=("oscillation amplitude consistent with zero",)
    )
    rec = dataclasses.replace(base, assessments=(exp, gauss), recommended_key="exp_constant")

    monkeypatch.setattr(
        wizard_window_module,
        "build_fit_wizard_recommendation",
        lambda dataset, current_model=None, metric=SelectionMetric.AICC, **kwargs: rec,
    )
    window = FitWizardWindow()
    window.set_analysis_context(dataset)
    window._start_analysis()
    wait_for(lambda: _analysis_complete(window), qapp)

    titles = {
        window._compare_table.item(r, 0).text(): r for r in range(window._compare_table.rowCount())
    }
    null_row = next(r for t, r in titles.items() if "baseline" in t)
    disq_row = next(r for t, r in titles.items() if "disqualified" in t)
    # The disqualified row carries its reasons as a tooltip.
    assert "oscillation amplitude" in window._compare_table.item(disq_row, 0).toolTip()
    assert window._compare_table.item(null_row, 0).text().endswith("(baseline)")


def test_fit_wizard_window_default_tier_shows_model_not_no_recommendation(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recommendation with the default NONE tier + a set key must NOT contradict
    itself with a spurious 'no confident recommendation' confidence line — the
    card shows the best model and leaves the confidence line empty instead."""
    monkeypatch.setattr(
        wizard_window_module,
        "build_fit_wizard_recommendation",
        lambda dataset, current_model=None, metric=SelectionMetric.AICC, **kwargs: (
            _fake_recommendation(dataset)  # confidence=NONE, verdict=NONE, caveat="", key set
        ),
    )
    window = FitWizardWindow()
    window.set_analysis_context(dataset)
    window._start_analysis()
    wait_for(lambda: _analysis_complete(window), qapp)

    # The card names the best model rather than claiming no recommendation.
    assert "Exponential + Constant" in window._answer_card._verdict_label.text()
    # The bare NONE-fallback line is suppressed (empty + hidden).
    assert window._answer_card._confidence_label.text() == ""
    assert window._answer_card._confidence_label.isHidden() is True


# ── New-structure behaviours (welcome / running / result states) ─────────────


def test_fit_wizard_window_two_click_path_analyze_then_apply(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The core two-click novice path: Analyze on Welcome → Apply on Result.

    Expanding the optional guidance is never required to reach a recommendation.
    """
    monkeypatch.setattr(
        wizard_window_module,
        "build_fit_wizard_recommendation",
        lambda dataset, current_model=None, metric=SelectionMetric.AICC, **kwargs: (
            _fake_recommendation(dataset)
        ),
    )
    window = FitWizardWindow()
    window.set_analysis_context(dataset)

    # Click 1: Analyze (from the Welcome page, guidance still collapsed).
    assert window._guidance_section.isExpanded() is False
    window._analyze_btn.click()
    wait_for(lambda: _analysis_complete(window), qapp)

    assert window._stack.currentIndex() == _PAGE_RESULT

    # Click 2: Apply this fit.
    emitted: list[object] = []
    window.apply_assessment_requested.connect(lambda a, r: emitted.append(a))
    window._answer_card._apply_btn.click()
    assert emitted and emitted[0].template.key == "exp_constant"


def test_fit_wizard_window_running_state_streams_trail(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fake progress messages light the mapped trail steps; unknown text only
    updates the status line."""
    monkeypatch.setattr(
        wizard_window_module,
        "build_fit_wizard_recommendation",
        lambda dataset, current_model=None, metric=SelectionMetric.AICC, **kwargs: (
            _fake_recommendation(dataset)
        ),
    )
    window = FitWizardWindow()
    window.set_analysis_context(dataset)
    window._show_running()
    assert window._stack.currentIndex() == _PAGE_RUNNING

    # Placeholder trail is shown immediately with the six known keys.
    assert window._running_trail.step_keys() == (
        "conditions",
        "families",
        "spectrum",
        "candidates",
        "verdict",
        "confidence",
    )

    # Recognised progress prefixes advance the mapped step (these are the real
    # strings the core's build_fit_wizard_recommendation emits).
    window._on_progress(0, 0, "Stage 1: screening 5 candidate families")
    assert window._running_trail.active_step_key() == "families"
    window._on_progress(0, 0, "Spectral search: 3 line(s), 1 pattern match(es)")
    assert window._running_trail.active_step_key() == "spectrum"
    assert "Spectral search" in window._running_trail._status_label.text()
    window._on_progress(0, 0, "Stage 2: fitting 4 expanded candidates")
    assert window._running_trail.active_step_key() == "candidates"

    # An unrecognised message only updates the status line, leaving the active
    # step unchanged (no raise).
    window._on_progress(0, 0, "some novel stage nobody mapped")
    assert window._running_trail.active_step_key() == "candidates"
    assert "novel stage" in window._running_trail._status_label.text()


def test_fit_wizard_window_alternatives_swap_changes_applied_key(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        wizard_window_module,
        "build_fit_wizard_recommendation",
        lambda dataset, current_model=None, metric=SelectionMetric.AICC, **kwargs: (
            _fake_recommendation(dataset)
        ),
    )
    window = FitWizardWindow()
    window.set_analysis_context(dataset)
    window._start_analysis()
    wait_for(lambda: _analysis_complete(window), qapp)

    # gauss is offered as an alternative on the card.
    assert "gaussian_constant" in window._answer_card._alt_buttons
    window._answer_card._alt_buttons["gaussian_constant"].click()
    qapp.processEvents()

    # Selecting the alternative retargets Apply and syncs the compare table.
    assert window._answer_card.selected_key() == "gaussian_constant"
    assert window._selected_key == "gaussian_constant"
    emitted: list[object] = []
    window.apply_assessment_requested.connect(lambda a, r: emitted.append(a))
    window._answer_card._apply_btn.click()
    assert emitted[0].template.key == "gaussian_constant"


def test_fit_wizard_window_copy_log_uses_render_log_text(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PySide6.QtGui import QGuiApplication

    from asymmetry.core.fitting.wizard_narrative import render_log_text

    monkeypatch.setattr(
        wizard_window_module,
        "build_fit_wizard_recommendation",
        lambda dataset, current_model=None, metric=SelectionMetric.AICC, **kwargs: (
            _fake_recommendation(dataset)
        ),
    )
    window = FitWizardWindow()
    window.set_analysis_context(dataset)
    window._start_analysis()
    wait_for(lambda: _analysis_complete(window), qapp)

    window._copy_log_btn.click()
    clipboard = QGuiApplication.clipboard()
    expected = render_log_text(window.current_recommendation())
    assert clipboard.text() == expected


def test_fit_wizard_window_trail_expansion_reveals_reparented_panels(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Result trail embeds the existing deep panels, re-parented in place."""
    monkeypatch.setattr(
        wizard_window_module,
        "build_fit_wizard_recommendation",
        lambda dataset, current_model=None, metric=SelectionMetric.AICC, **kwargs: (
            _recommendation_with_peaks(dataset)
        ),
    )
    window = FitWizardWindow()
    window.set_analysis_context(dataset)
    window._start_analysis()
    wait_for(lambda: _analysis_complete(window), qapp)

    # The scope, fingerprint (FFT+peaks) and compare panels are re-parented into
    # the trail's conditions / spectrum / candidates expansions.
    assert window._scope_panel.parent() is not window._guidance_scope_slot
    trail = window._result_trail
    trail.set_step_expanded("spectrum", True)
    qapp.processEvents()
    assert window._fingerprint_panel.isVisibleTo(window) is True
    # The re-parented compare table keeps its baseline/disqualified marks intact.
    trail.set_step_expanded("candidates", True)
    qapp.processEvents()
    assert window._compare_table.rowCount() == 2


def test_fit_wizard_window_reanalyze_returns_to_welcome(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        wizard_window_module,
        "build_fit_wizard_recommendation",
        lambda dataset, current_model=None, metric=SelectionMetric.AICC, **kwargs: (
            _fake_recommendation(dataset)
        ),
    )
    window = FitWizardWindow()
    window.set_analysis_context(dataset)
    window._start_analysis()
    wait_for(lambda: _analysis_complete(window), qapp)
    assert window._stack.currentIndex() == _PAGE_RESULT

    window._reanalyze_btn.click()
    assert window._stack.currentIndex() == _PAGE_WELCOME
    # The guidance panels are back in the Welcome expander for steering.
    assert window._scope_panel.parent() is window._guidance_scope_slot


def test_fit_wizard_window_progress_callback_is_wired_through(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worker task must hand the core a progress_callback that reaches the
    trail — a mock builder that emits a stage message lights the mapped step."""
    seen: list[str] = []

    def _builder_with_progress(dataset, current_model=None, metric=SelectionMetric.AICC, **kwargs):
        progress = kwargs.get("progress_callback")
        assert callable(progress), "the task closure must pass a progress_callback"
        progress("Stage 1: screening 5 candidate families")
        progress("Stage 2: fitting 4 expanded candidates")
        seen.append("emitted")
        return _fake_recommendation(dataset)

    monkeypatch.setattr(
        wizard_window_module, "build_fit_wizard_recommendation", _builder_with_progress
    )
    window = FitWizardWindow()
    window.set_analysis_context(dataset)
    window._start_analysis()
    wait_for(lambda: _analysis_complete(window), qapp)

    # The core emitted progress and it reached the running trail before Result.
    assert seen == ["emitted"]
    # On completion the trail is rebuilt from the recommendation (six steps).
    assert window._result_trail.step_keys() == (
        "conditions",
        "families",
        "spectrum",
        "candidates",
        "verdict",
        "confidence",
    )


# ── Matched-apodisation scan lines in the fingerprint panel ──────────────────


def _damped_scan_peak_analysis() -> PeakAnalysis:
    """One accepted scan line (240 MHz, λ = 44 µs⁻¹) plus a Hann-pass line."""
    return PeakAnalysis(
        peaks=(
            DetectedPeak(
                frequency_mhz=240.0,
                amplitude=4.7,
                snr=8.5,
                width_mhz=44.0 / np.pi,
                prominence=0.0,
                source="damped_scan",
                damping_rate_per_us=44.0,
                amplitude_percent=4.7,
                phase_rad=0.3,
                delta_chi_squared=612.0,
            ),
            DetectedPeak(
                frequency_mhz=5.0,
                amplitude=1.0,
                snr=12.0,
                width_mhz=0.3,
                prominence=0.5,
                source="fft",
            ),
        ),
        noise_floor=0.0,
        resolution_mhz=0.5,
        nyquist_mhz=5000.0,
        detrended=False,
    )


def _damped_scan_recommendation(
    dataset: MuonDataset, *, rebin_factor: int = 1, analysed_points: int = 0
) -> FitWizardRecommendation:
    base = _fake_recommendation(dataset)
    return dataclasses.replace(
        base,
        peak_analysis=_damped_scan_peak_analysis(),
        fingerprint=dataclasses.replace(
            base.fingerprint,
            damped_line_frequency_mhz=240.0,
            damped_line_snr=8.5,
            damped_line_rate_per_us=44.0,
        ),
        rebin_factor=rebin_factor,
        analysed_points=analysed_points,
    )


def _show_damped_result(
    window: FitWizardWindow,
    dataset: MuonDataset,
    *,
    rebin_factor: int = 1,
    analysed_points: int = 0,
    fit_range: tuple[float | None, float | None] | None = None,
) -> None:
    """Drive the window to the Result page from a canned scan recommendation."""
    window.set_analysis_context(dataset, fit_range=fit_range)
    window.set_cached_recommendation(
        _damped_scan_recommendation(
            dataset, rebin_factor=rebin_factor, analysed_points=analysed_points
        )
    )


def test_fit_wizard_window_peaks_table_shows_the_measured_damping(
    qapp: QApplication,
    dataset: MuonDataset,
) -> None:
    window = FitWizardWindow()
    _show_damped_result(window, dataset)

    assert window._peaks_table.columnCount() == 6
    assert window._peaks_table.horizontalHeaderItem(3).text() == "Damping (µs⁻¹)"

    sources = [window._peaks_table.item(r, 5).text() for r in range(window._peaks_table.rowCount())]
    scan_row = sources.index("damped_scan")
    hann_row = sources.index("fft")
    assert window._peaks_table.item(scan_row, 3).data(_display_role()) == pytest.approx(44.0)
    # A windowed-pass line has no measured envelope, and says so rather than
    # borrowing a number from somewhere else.
    assert window._peaks_table.item(hann_row, 3).text() == "—"

    tooltip = window._peaks_table.item(scan_row, 3).toolTip()
    assert "matched-apodisation scan" in tooltip
    assert "4.7 %" in tooltip and "0.3 rad" in tooltip and "612" in tooltip
    assert window._peaks_table.item(hann_row, 3).toolTip() == ""


def test_fit_wizard_window_draws_scan_lines_dashed_on_the_windowed_fft(
    qapp: QApplication,
    dataset: MuonDataset,
) -> None:
    window = FitWizardWindow()
    _show_damped_result(window, dataset)

    def _marker_at(frequency: float):
        for line in window._fft_ax.lines:
            xdata = np.asarray(line.get_xdata(), dtype=float)
            if xdata.size == 2 and abs(xdata[0] - frequency) < 1e-9:
                return line
        raise AssertionError(f"no marker drawn at {frequency} MHz")

    # Dashed for the scan line; the Hann-pass line keeps the solid marker.
    assert _marker_at(240.0).get_linestyle() == "--"
    assert _marker_at(5.0).get_linestyle() == "-"
    legend = window._fft_ax.get_legend()
    assert legend is not None
    labels = [text.get_text() for text in legend.get_texts()]
    assert labels == [wizard_window_module._DAMPED_SCAN_LEGEND_LABEL]
    assert "not expected to be visible in this windowed transform" in labels[0]


def test_fit_wizard_window_fingerprint_rows_appear_only_with_a_damped_line(
    qapp: QApplication,
    dataset: MuonDataset,
) -> None:
    window = FitWizardWindow()
    _show_damped_result(window, dataset)

    def _labels() -> list[str]:
        table = window._fingerprint_table
        return [table.item(r, 0).text() for r in range(table.rowCount())]

    def _value(label: str) -> str:
        table = window._fingerprint_table
        return table.item(_labels().index(label), 1).text()

    assert "Damped line frequency" in _labels()
    assert _value("Damped line frequency").endswith("MHz")
    assert _value("Damped line damping rate").startswith("44.")
    assert _value("Damped line scan SNR") == "8.500"
    # The interpretation banner names the scan line as the evidence, ahead of
    # the windowed-FFT features (which describe only the slow background here).
    banner = window._fingerprint_banner.text()
    assert banner.startswith("The matched-apodisation scan found a damped line at ")
    assert "envelope rate 44" in banner
    assert "scan SNR 8.5" in banner

    # A fingerprint with no accepted scan line does not carry the rows at all.
    window.set_cached_recommendation(_fake_recommendation(dataset))
    assert "Damped line frequency" not in _labels()
    assert "matched-apodisation scan found" not in window._fingerprint_banner.text()


def test_fit_wizard_window_notes_a_fit_range_that_excludes_the_damped_line(
    qapp: QApplication,
    dataset: MuonDataset,
) -> None:
    window = FitWizardWindow()
    # 1/λ is 23 ns, so a range opening at 1 µs starts long after the line is gone.
    _show_damped_result(window, dataset, fit_range=(1.0, 8.0))

    assert window._notes_banner.isHidden() is False
    note = window._notes_banner.text()
    assert note.startswith("The plot's fit range starts at 1 µs, after the 240 MHz line")
    assert "widen the range before applying this fit." in note


def test_fit_wizard_window_says_nothing_when_the_fit_range_keeps_the_line(
    qapp: QApplication,
    dataset: MuonDataset,
) -> None:
    window = FitWizardWindow()
    _show_damped_result(window, dataset, fit_range=(0.0, 8.0))

    assert window._notes_banner.text() == ""
    assert window._notes_banner.isHidden() is True


def test_fit_wizard_window_notes_rebinned_candidate_fitting(
    qapp: QApplication,
    dataset: MuonDataset,
) -> None:
    window = FitWizardWindow()
    _show_damped_result(window, dataset, rebin_factor=8, analysed_points=11250)

    assert window._notes_banner.text() == (
        "Candidates were fitted on a ×8 rebinned copy (11250 points); "
        "information criteria refer to that record."
    )


def test_fit_wizard_window_analyses_the_uncropped_record(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wizard fingerprints the whole record, not the plot's fit-range crop."""
    cropped = dataset.time_range(2.0, 6.0)
    seen: list[MuonDataset] = []

    def _capture(dataset, current_model=None, metric=SelectionMetric.AICC, **kwargs):
        seen.append(dataset)
        return _fake_recommendation(dataset)

    monkeypatch.setattr(wizard_window_module, "build_fit_wizard_recommendation", _capture)
    window = FitWizardWindow()
    window.set_analysis_context(cropped, full_dataset=dataset, fit_range=(2.0, 6.0))

    assert window._dataset is dataset
    window._start_analysis()
    wait_for(lambda: _analysis_complete(window), qapp)

    assert len(seen) == 1
    assert seen[0] is dataset
    assert seen[0].n_points > cropped.n_points


def test_fit_wizard_window_click_seeded_frequency_carries_a_damping_estimate(
    qapp: QApplication,
) -> None:
    """A click on a heavily damped line seeds λ as well as the frequency."""
    rng = np.random.default_rng(5)
    time = np.arange(8000, dtype=float) * 1e-4
    error = np.minimum(1.5 * np.exp(time / 4.4), 25.0)
    asymmetry = (
        4.7 * np.exp(-44.0 * time) * np.cos(2.0 * np.pi * 240.0 * time + 0.3)
        + 2.6 * np.exp(-0.19 * time)
        + 4.6
        + rng.normal(0.0, error)
    )
    damped = MuonDataset(time=time, asymmetry=asymmetry, error=error, metadata={"run_number": 202})

    window = FitWizardWindow()
    window.set_analysis_context(damped)
    window._fingerprint_plot_widget._canvas.draw()

    px = float(window._fft_ax.transData.transform((240.0, 0.0))[0])
    window._on_fft_press(_press(window, xdata=240.0, x=px))
    window._on_fft_release(_release())

    # The click seeds the bare frequency at once; the envelope is measured on a
    # worker and attached when it returns.
    seed = window._user_peaks[0]
    assert seed["source"] == "user"
    assert "damping_rate_per_us" not in seed
    wait_for(lambda: window._tasks.active_count == 0, qapp)
    assert seed["damping_rate_per_us"] == pytest.approx(44.0, rel=0.35)
    # ...and the peaks table shows the estimate it just gained.
    row = [
        r
        for r in range(window._peaks_table.rowCount())
        if window._peaks_table.item(r, 5).text() == "user"
    ][0]
    assert window._peaks_table.item(row, 3).data(_display_role()) == pytest.approx(
        seed["damping_rate_per_us"]
    )


# --------------------------------------------------------------------------- #
# On-demand dense curves. A build materialises curves only for the rows it
# exposes as its answer; the window builds any other row the user selects on a
# worker thread and hands it to the card.
# --------------------------------------------------------------------------- #


def _recommendation_with_bare_alternative(dataset: MuonDataset) -> FitWizardRecommendation:
    """The shape a real build returns: the answer drawn, the alternative bare."""
    recommendation = _fake_recommendation(dataset)
    winner, alternative = recommendation.assessments
    empty = np.array([], dtype=float)
    return dataclasses.replace(
        recommendation,
        assessments=(
            winner,
            dataclasses.replace(
                alternative, fitted_time=empty, fitted_curve=empty, component_curves=()
            ),
        ),
    )


def _fit_line_lengths(window: FitWizardWindow) -> list[int]:
    figure = window._answer_card._plot_widget._figure
    return [
        len(line.get_xdata())
        for axes in figure.axes
        for line in axes.lines
        if line.get_label() == "Fit"
    ]


def test_recommended_row_draws_without_building_any_curve(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The answer is materialised by the build, so showing it costs nothing here."""
    monkeypatch.setattr(
        wizard_window_module,
        "build_fit_wizard_recommendation",
        lambda dataset, current_model=None, metric=SelectionMetric.AICC, **kwargs: (
            _recommendation_with_bare_alternative(dataset)
        ),
    )
    builds: list[str] = []
    monkeypatch.setattr(
        wizard_window_module,
        "assessment_with_curves",
        lambda recommendation, key, data: builds.append(key),
    )
    window = FitWizardWindow()
    window.set_analysis_context(dataset)
    window._start_analysis()
    wait_for(lambda: _analysis_complete(window), qapp)

    assert window._answer_card.selected_key() == "exp_constant"
    assert builds == []
    assert window._requested_curve_keys == set()
    assert _fit_line_lengths(window) == [dataset.n_points]


def test_selecting_a_bare_row_materialises_its_curves_off_the_gui_thread(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dense multi-component curve is far too expensive to build on a click.

    The card asks for the curves it lacks, the window builds them through its
    TaskRunner (never on the GUI thread), and the result is folded back into the
    recommendation and drawn.
    """
    monkeypatch.setattr(
        wizard_window_module,
        "build_fit_wizard_recommendation",
        lambda dataset, current_model=None, metric=SelectionMetric.AICC, **kwargs: (
            _recommendation_with_bare_alternative(dataset)
        ),
    )
    window = FitWizardWindow()
    window.set_analysis_context(dataset)
    window._start_analysis()
    wait_for(lambda: _analysis_complete(window), qapp)

    real_build = wizard_window_module.assessment_with_curves
    build_threads: list[int] = []

    def _recorded(recommendation, key, data):
        build_threads.append(threading.get_ident())
        return real_build(recommendation, key, data)

    monkeypatch.setattr(wizard_window_module, "assessment_with_curves", _recorded)

    window._answer_card.set_selected_key("gaussian_constant")
    # The click itself only queues work — nothing is built on the GUI thread.
    assert window._requested_curve_keys == {"gaussian_constant"}

    def _materialised() -> bool:
        assessment = window._recommendation.assessment_for_key("gaussian_constant")
        return assessment is not None and assessment.fitted_time.size > 0

    wait_for(_materialised, qapp)

    assert build_threads and threading.get_ident() not in build_threads
    materialised = window._recommendation.assessment_for_key("gaussian_constant")
    assert _fit_line_lengths(window) == [materialised.fitted_time.size]
    # The rest of the recommendation is untouched: same ranking, same selection.
    assert window._recommendation.recommended_key == "exp_constant"
    assert window._answer_card.selected_key() == "gaussian_constant"


def test_one_curve_build_per_row_while_it_is_in_flight(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A redraw while a build runs must not queue a second worker for that row.

    The card re-asks on every redraw it cannot draw a fit line for — a residuals
    toggle, a re-rank — so the window tracks the keys it already has in flight.
    """
    monkeypatch.setattr(
        wizard_window_module,
        "build_fit_wizard_recommendation",
        lambda dataset, current_model=None, metric=SelectionMetric.AICC, **kwargs: (
            _recommendation_with_bare_alternative(dataset)
        ),
    )
    window = FitWizardWindow()
    window.set_analysis_context(dataset)
    window._start_analysis()
    wait_for(lambda: _analysis_complete(window), qapp)

    release = threading.Event()
    real_build = wizard_window_module.assessment_with_curves
    starts: list[str] = []

    def _blocking(recommendation, key, data):
        starts.append(key)
        release.wait(timeout=5.0)
        return real_build(recommendation, key, data)

    monkeypatch.setattr(wizard_window_module, "assessment_with_curves", _blocking)

    window._answer_card.set_selected_key("gaussian_constant")
    # Force further redraws of the same still-bare row.
    window._answer_card._residuals_toggle.setChecked(True)
    window._answer_card._residuals_toggle.setChecked(False)
    release.set()

    def _materialised() -> bool:
        assessment = window._recommendation.assessment_for_key("gaussian_constant")
        return assessment is not None and assessment.fitted_time.size > 0

    wait_for(_materialised, qapp)
    assert starts == ["gaussian_constant"]


def test_a_curve_result_for_a_replaced_recommendation_is_dropped(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-analysing while a curve build runs must not resurrect the old result."""
    monkeypatch.setattr(
        wizard_window_module,
        "build_fit_wizard_recommendation",
        lambda dataset, current_model=None, metric=SelectionMetric.AICC, **kwargs: (
            _recommendation_with_bare_alternative(dataset)
        ),
    )
    window = FitWizardWindow()
    window.set_analysis_context(dataset)
    window._start_analysis()
    wait_for(lambda: _analysis_complete(window), qapp)

    release = threading.Event()
    real_build = wizard_window_module.assessment_with_curves

    def _blocking(recommendation, key, data):
        release.wait(timeout=5.0)
        return real_build(recommendation, key, data)

    monkeypatch.setattr(wizard_window_module, "assessment_with_curves", _blocking)
    window._answer_card.set_selected_key("gaussian_constant")

    replacement = _fake_recommendation(dataset)
    window._recommendation = replacement
    window._invalidate_pending_curves()
    release.set()

    wait_for(lambda: window._tasks.active_count == 0, qapp)
    assert window._recommendation is replacement

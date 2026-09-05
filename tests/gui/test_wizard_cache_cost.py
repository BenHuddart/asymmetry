"""Cost pins for the single-fit tab's cached wizard recommendation.

A fit-wizard recommendation holds a dense fitted curve, its component curves
and the fit residuals for every candidate it assessed. Embedding a *serialised*
copy of it in the single-fit tab's session state made every run switch pay two
serialisations, a deserialisation and a couple of deep copies of a
hundreds-of-MB payload (3–5 s per switch on a ~90k-bin run, plus GC pauses),
and wrote that payload into the project file for every fitted run.

These tests pin the fix from both ends: the session form carries the
recommendation *by reference* (no serialisation, no deep copy, on the real
run-switch path), and the persistence boundaries emit a bounded, JSON-safe
payload that still restores.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import asymmetry.core.fitting.fit_wizard as fit_wizard_module  # noqa: E402
import asymmetry.gui.mainwindow as mw_module  # noqa: E402
import asymmetry.gui.panels.fit.wizard_cache as wizard_cache_module  # noqa: E402
from asymmetry.core.data.dataset import Histogram, MuonDataset, Run  # noqa: E402
from asymmetry.core.fitting.composite import CompositeModel  # noqa: E402
from asymmetry.core.fitting.engine import FitResult  # noqa: E402
from asymmetry.core.fitting.fit_wizard import (  # noqa: E402
    CandidateAssessment,
    CandidateTemplate,
    FitWizardRecommendation,
    SelectionMetric,
    SpectrumFingerprint,
    serialize_fit_wizard_recommendation,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet  # noqa: E402
from asymmetry.gui.mainwindow import MainWindow  # noqa: E402
from asymmetry.gui.panels.fit.panel import FitPanel  # noqa: E402
from asymmetry.gui.panels.fit.wizard_cache import WizardCacheEntry  # noqa: E402
from asymmetry.gui.windows.fit_wizard_window import FitWizardWindow  # noqa: E402

pytestmark = [pytest.mark.gui]


# ── fixtures and synthetic payloads ────────────────────────────────────────


@pytest.fixture
def mainwindow(qapp: QApplication) -> MainWindow:
    settings = QSettings()
    settings.setValue(mw_module._UI_SCALE_SETTINGS_KEY, 1.0)
    window = MainWindow()
    qapp.processEvents()
    yield window
    window.close()
    window.deleteLater()


def _grouped_dataset(run_number: int, n_bins: int = 64) -> MuonDataset:
    """A two-detector run the data browser can select and reduce."""
    t = np.arange(n_bins, dtype=float) * 0.05
    counts_f = 1000.0 * np.exp(-t / 2.2) + 5.0
    counts_b = 900.0 * np.exp(-t / 2.2) + 5.0
    run = Run(
        run_number=run_number,
        histograms=[
            Histogram(counts=counts_f, bin_width=0.05),
            Histogram(counts=counts_b, bin_width=0.05),
        ],
        metadata={"run_number": run_number, "field": 100.0, "temperature": 5.0},
        grouping={
            "groups": {1: [1], 2: [2]},
            "group_names": {1: "F", 2: "B"},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": n_bins - 1,
            "bunching_factor": 1,
            "deadtime_correction": False,
        },
    )
    asym = 100.0 * (counts_f - counts_b) / (counts_f + counts_b)
    return MuonDataset(
        time=t,
        asymmetry=asym,
        error=np.full_like(t, 0.5),
        metadata={"run_number": run_number, "field": 100.0, "temperature": 5.0},
        run=run,
    )


def _plain_dataset(run_number: int = 101, n_points: int = 256) -> MuonDataset:
    t = np.linspace(0.0, 10.0, n_points)
    a = 0.2 * np.exp(-0.4 * t) + 0.01
    return MuonDataset(
        time=t,
        asymmetry=a,
        error=np.full_like(t, 0.01),
        metadata={"run_number": run_number},
    )


def _recommendation_for(dataset: MuonDataset, *, candidates: int = 1) -> FitWizardRecommendation:
    """A recommendation whose arrays are as dense as the record, as in real use."""
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    values = {"A_1": 0.2, "Lambda": 0.4, "A_bg": 0.01}
    time = np.asarray(dataset.time, dtype=float)
    curve = np.asarray(model.function(time, **values), dtype=float)
    components = tuple(model.evaluate_components(time, additive_only=True, **values))

    assessments: list[CandidateAssessment] = []
    templates: list[CandidateTemplate] = []
    for index in range(candidates):
        template = CandidateTemplate(
            key=f"exp_constant_{index}" if index else "exp_constant",
            title="Exponential + Constant",
            category="General",
            rationale="Baseline candidate",
            model=model,
        )
        result = FitResult(
            success=True,
            chi_squared=5.0,
            reduced_chi_squared=0.1,
            parameters=ParameterSet(
                [
                    Parameter("A_1", 0.2, min=0.0, max=1.0),
                    Parameter("Lambda", 0.4, min=0.0, max=5.0),
                    Parameter("A_bg", 0.01, min=-0.5, max=0.5),
                ]
            ),
            uncertainties={"A_1": 0.01, "Lambda": 0.02, "A_bg": 0.001},
            residuals=np.asarray(dataset.asymmetry, dtype=float) - curve,
        )
        templates.append(template)
        assessments.append(
            CandidateAssessment(
                template=template,
                fit_result=result,
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
                fitted_time=time.copy(),
                fitted_curve=curve.copy(),
                component_curves=components,
            )
        )

    return FitWizardRecommendation(
        fingerprint=SpectrumFingerprint(
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
        ),
        templates=tuple(templates),
        assessments=tuple(assessments),
        metric=SelectionMetric.AICC,
        recommended_key="exp_constant",
        comparable_keys=(),
        summary="Recommended: Exponential + Constant by AICc.",
    )


class _SerialisationCounter:
    """Counts every (de)serialisation of a single-fit wizard recommendation."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.serialised = 0
        self.deserialised = 0
        for module in (wizard_cache_module, fit_wizard_module):
            monkeypatch.setattr(
                module,
                "serialize_fit_wizard_recommendation",
                self._counting(module.serialize_fit_wizard_recommendation, "serialised"),
            )
            monkeypatch.setattr(
                module,
                "deserialize_fit_wizard_recommendation",
                self._counting(module.deserialize_fit_wizard_recommendation, "deserialised"),
            )

    def _counting(self, original, field: str):
        def _wrapped(*args, **kwargs):
            setattr(self, field, getattr(self, field) + 1)
            return original(*args, **kwargs)

        return _wrapped


def _cache_on_single_tab(panel: FitPanel, recommendation: FitWizardRecommendation) -> None:
    panel._single_tab._cache_wizard_analysis(
        recommendation,
        signature={
            "run_number": panel._active_single_run_number,
            "model": panel._single_tab._composite_model.to_dict(),
        },
        log_text="cached log",
    )


# ── what a run switch may cost ─────────────────────────────────────────────


class TestRunSwitchCost:
    def test_switching_runs_never_serialises_the_cached_recommendation(
        self, mainwindow: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        a, b = _grouped_dataset(9301), _grouped_dataset(9302)
        mainwindow._data_browser.add_dataset(a)
        mainwindow._data_browser.add_dataset(b)
        mainwindow._data_browser.select_runs({9301})
        QApplication.processEvents()

        panel = mainwindow._fit_panel
        recommendation = _recommendation_for(panel._single_tab._current_dataset or a)
        _cache_on_single_tab(panel, recommendation)

        counter = _SerialisationCounter(monkeypatch)
        mainwindow._data_browser.select_runs({9302})
        QApplication.processEvents()
        mainwindow._data_browser.select_runs({9301})
        QApplication.processEvents()

        assert (counter.serialised, counter.deserialised) == (0, 0)

    def test_switching_runs_carries_the_same_recommendation_object(
        self, mainwindow: MainWindow
    ) -> None:
        a, b = _grouped_dataset(9311), _grouped_dataset(9312)
        mainwindow._data_browser.add_dataset(a)
        mainwindow._data_browser.add_dataset(b)
        mainwindow._data_browser.select_runs({9311})
        QApplication.processEvents()

        panel = mainwindow._fit_panel
        recommendation = _recommendation_for(panel._single_tab._current_dataset or a)
        _cache_on_single_tab(panel, recommendation)

        mainwindow._data_browser.select_runs({9312})
        QApplication.processEvents()

        # Carried forward by reference — never a rebuilt copy.
        assert panel._single_tab._cached_wizard_recommendation is recommendation

    def test_deep_copying_the_form_state_shares_the_recommendation(
        self, qapp: QApplication
    ) -> None:
        panel = FitPanel()
        dataset = _plain_dataset()
        panel.set_dataset(dataset)
        recommendation = _recommendation_for(dataset)
        _cache_on_single_tab(panel, recommendation)

        state = panel._single_tab.get_state()
        entry = state["wizard_state"]
        assert isinstance(entry, WizardCacheEntry)

        # Identity equality: comparing two entries field-by-field would compare
        # numpy arrays and raise, so the handle never grows a value __eq__.
        assert entry != WizardCacheEntry(recommendation=recommendation, signature={}, log_text="")

        copied = copy.deepcopy(state)
        # The handle is immutable, so a deep copy of the state that carries it
        # is O(form size): this is what keeps FitPanel's per-switch copies cheap.
        assert copied["wizard_state"] is entry
        assert copy.copy(entry) is entry
        assert copied["wizard_state"].recommendation is recommendation

    def test_applying_a_wizard_assessment_never_serialises(
        self, qapp: QApplication, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        panel = FitPanel()
        dataset = _plain_dataset()
        panel.set_dataset(dataset)
        recommendation = _recommendation_for(dataset)
        _cache_on_single_tab(panel, recommendation)
        assessment = recommendation.assessments[0]

        counter = _SerialisationCounter(monkeypatch)
        panel._single_tab._apply_fit_wizard_assessment(assessment, recommendation)

        assert (counter.serialised, counter.deserialised) == (0, 0)
        assert panel._single_tab._cached_wizard_recommendation is recommendation


# ── what leaves the session ────────────────────────────────────────────────


class TestPersistenceBoundary:
    def test_form_state_for_a_slot_is_json_safe_and_bounded(self, qapp: QApplication) -> None:
        panel = FitPanel()
        dataset = _plain_dataset(n_points=40_000)
        panel.set_dataset(dataset)
        recommendation = _recommendation_for(dataset, candidates=4)
        _cache_on_single_tab(panel, recommendation)

        payload = panel.get_single_form_state()
        encoded = json.dumps(payload)

        # Four candidates × (40k-point fitted curve + time axis + two component
        # curves + a 40k-point residual series) is tens of MB unbounded; the
        # stored form is a few tens of kB per candidate.
        assert len(encoded) < 200_000
        stored = payload["wizard_state"]["recommendation"]
        for assessment in stored["assessments"]:
            assert len(assessment["fitted_time"]) <= fit_wizard_module.PERSISTED_CURVE_MAX_POINTS
            assert len(assessment["fitted_curve"]) == len(assessment["fitted_time"])
            for component in assessment["component_curves"]:
                assert len(component["values"]) == len(assessment["fitted_time"])
            # Residual *series* is dropped; the numbers derived from it stay.
            assert assessment["fit_result"]["residuals"] is None
            assert assessment["residual_rms"] == pytest.approx(0.9)

    def test_recorded_fit_slot_ui_state_holds_no_live_handle(
        self, mainwindow: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        a = _grouped_dataset(9321)
        mainwindow._data_browser.add_dataset(a)
        mainwindow._data_browser.select_runs({9321})
        QApplication.processEvents()

        panel = mainwindow._fit_panel
        dataset = panel._single_tab._current_dataset or a
        recommendation = _recommendation_for(dataset, candidates=2)
        _cache_on_single_tab(panel, recommendation)

        # A panel that hands back the session-shaped form (a stub, or a future
        # caller) must still not put a live handle into the slot.
        session_state = panel._single_tab.get_state()
        monkeypatch.setattr(panel, "get_single_form_state", lambda: session_state)

        counter = _SerialisationCounter(monkeypatch)
        mainwindow._record_single_fit_slot(
            FitResult(success=True, chi_squared=1.0, reduced_chi_squared=0.5)
        )
        # Exactly one serialisation, at the boundary — and this is also what
        # proves the counter used by the run-switch pins above is wired to the
        # code path that would serialise.
        assert (counter.serialised, counter.deserialised) == (1, 0)

        rep_type = mainwindow._active_representation_type()
        representation = mainwindow._project_model.ensure_dataset(9321).ensure(rep_type)
        slot = representation.fit_for(mainwindow._current_single_fit_projection())
        assert slot.ui_state
        assert isinstance(slot.ui_state["wizard_state"], dict)
        json.dumps(slot.to_dict())

    def test_project_state_carries_no_live_wizard_handle(self, mainwindow: MainWindow) -> None:
        a = _grouped_dataset(9331)
        mainwindow._data_browser.add_dataset(a)
        mainwindow._data_browser.select_runs({9331})
        QApplication.processEvents()

        panel = mainwindow._fit_panel
        _cache_on_single_tab(panel, _recommendation_for(panel._single_tab._current_dataset or a))

        state = mainwindow.collect_project_state()

        # The panel's own blocks drop the (regenerable) wizard cache entirely;
        # whatever remains must be plain JSON.
        json.dumps(state, default=str)
        single_state = state["fit_states"]["time"]["single_fit_state"]
        assert "wizard_state" not in single_state
        for run_state in single_state.get("states_by_run", {}).values():
            assert "wizard_state" not in run_state


class TestPersistedCacheRestores:
    def test_compact_payload_restores_and_still_renders_the_answer_card(
        self, qapp: QApplication
    ) -> None:
        panel = FitPanel()
        dataset = _plain_dataset(n_points=5_000)
        panel.set_dataset(dataset)
        recommendation = _recommendation_for(dataset)
        _cache_on_single_tab(panel, recommendation)

        persisted = panel.get_single_form_state()

        restored_panel = FitPanel()
        restored_panel.set_dataset(dataset)
        restored_panel.restore_single_fit_ui(persisted)
        cached = restored_panel._single_tab._cached_wizard_recommendation
        assert cached is not None
        assert cached.summary == recommendation.summary
        assert restored_panel._single_tab._cached_wizard_log_text == "cached log"

        window = FitWizardWindow()
        window.set_analysis_context(dataset)
        window.set_cached_recommendation(
            cached,
            signature=restored_panel._single_tab._cached_wizard_signature,
            log_text="cached log",
        )
        assert window._answer_card.selected_key() == "exp_constant"
        assert window.current_recommendation() is cached
        window.close()
        window.deleteLater()

    def test_legacy_full_resolution_payload_still_restores(self, qapp: QApplication) -> None:
        panel = FitPanel()
        dataset = _plain_dataset(n_points=3_000)
        panel.set_dataset(dataset)
        recommendation = _recommendation_for(dataset)

        legacy_state = {
            "model_name": "Composite",
            "composite_model": panel._single_tab._composite_model.to_dict(),
            "parameters": [],
            "result_html": "",
            "wizard_state": {
                "signature": {
                    "run_number": int(dataset.run_number),
                    "model": panel._single_tab._composite_model.to_dict(),
                },
                # Pre-this-change files stored the full-resolution payload.
                "recommendation": serialize_fit_wizard_recommendation(recommendation),
                "log_text": "legacy log",
            },
        }

        panel._single_tab.restore_state(legacy_state)

        cached = panel._single_tab._cached_wizard_recommendation
        assert cached is not None
        assert cached.assessments[0].fitted_time.size == dataset.time.size
        assert cached.assessments[0].fit_result.residuals is not None
        assert panel._single_tab._cached_wizard_log_text == "legacy log"

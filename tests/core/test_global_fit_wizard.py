"""Tests for the global fit wizard core service."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration]

import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest

import asymmetry.core.fitting.fit_wizard as fit_wizard_module
import asymmetry.core.fitting.global_fit_wizard as global_fit_wizard_module
from asymmetry.core import fitting as fitting_api
from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitCancelledError, FitEngine, FitResult
from asymmetry.core.fitting.fit_wizard import (
    CandidateAssessment,
    CandidateTemplate,
    FitWizardRecommendation,
    SelectionMetric,
    SpectrumFingerprint,
    build_fit_wizard_recommendation_for_templates,
    fingerprint_spectrum,
)
from asymmetry.core.fitting.global_fit_wizard import (
    GlobalCandidateAssessment,
    GlobalFitWizardRecommendation,
    RunResidualDiagnostic,
    _build_parameter_recommendations_from_exact_cache,
    _canonicalize_parameter_sets,
    _deserialize_global_candidate_assessment,
    _fit_exact_assignment,
    _globalization_candidate_order,
    _layer_parameter_count,
    _localisation_penalty,
    _metric_penalty,
    _single_run_prefit_parameter_sets,
    _staged_assignment_seed,
    _staged_globalization_assignment,
    _staged_multi_local_assignment,
    _supported_oscillatory_run_numbers,
    _warm_start_parameter_sets,
    build_global_fit_wizard_candidate_portfolio,
    build_global_fit_wizard_recommendation,
    build_global_fit_wizard_screening_recommendation,
    build_or_complete_single_fit_wizard_recommendations_for_global_portfolio,
    merge_global_fit_wizard_recommendations,
    rerank_global_fit_wizard_recommendation,
    single_fit_table_covers_portfolio,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fitting.wizard_scope import WizardScope, WizardScopePreset


def _dataset_for(
    run_number: int,
    *,
    field: float,
    temperature: float,
    model: CompositeModel,
    params: dict[str, float],
) -> MuonDataset:
    time = np.linspace(0.0, 8.0, 120)
    asymmetry = model.function(time, **params)
    error = np.full_like(time, 0.01)
    return MuonDataset(
        time=time,
        asymmetry=asymmetry,
        error=error,
        metadata={
            "run_number": run_number,
            "field": field,
            "temperature": temperature,
            "run_label": str(run_number),
        },
    )


def _dense_dataset_for(
    run_number: int,
    *,
    field: float,
    temperature: float,
    model: CompositeModel,
    params: dict[str, float],
    error_level: float = 0.005,
) -> MuonDataset:
    time = np.linspace(0.0, 8.0, 80)
    asymmetry = model.function(time, **params)
    error = np.full_like(time, error_level)
    return MuonDataset(
        time=time,
        asymmetry=asymmetry,
        error=error,
        metadata={
            "run_number": run_number,
            "field": field,
            "temperature": temperature,
            "run_label": str(run_number),
        },
    )


def _fingerprint(
    *,
    dominant_fft_snr: float = 0.0,
    cycles: float = 0.0,
    turning_points: int = 0,
    oscillatory_hint: bool = False,
) -> fitting_api.SpectrumFingerprint:
    return fitting_api.SpectrumFingerprint(
        tail_estimate=0.01,
        initial_amplitude_estimate=0.2,
        zero_crossings=0,
        smoothed_zero_crossings=0,
        smoothed_turning_points=turning_points,
        dominant_fft_frequency_mhz=0.35 if dominant_fft_snr > 0.0 else 0.0,
        dominant_fft_snr=dominant_fft_snr,
        dominant_fft_cycles_in_window=cycles,
        monotonic_decay_fraction=0.7,
        early_time_curvature=-0.1,
        semilog_slope_ratio=1.2,
        late_time_dip_recovery_score=0.0,
        oscillatory_hint=oscillatory_hint,
        kt_like_hint=False,
        multi_rate_hint=False,
    )


def _assessment_with_diagnostics(
    datasets: list[MuonDataset],
    diagnostics: list[RunResidualDiagnostic],
) -> GlobalCandidateAssessment:
    template = CandidateTemplate(
        key="exp_constant",
        title="Exponential + Constant",
        category="General",
        rationale="test",
        model=CompositeModel(["Exponential", "Constant"], operators=["+"]),
    )
    fit_results = {
        int(dataset.run_number): FitResult(
            success=True,
            chi_squared=1.0,
            reduced_chi_squared=0.1,
            parameters=ParameterSet(
                [
                    Parameter("A_1", value=0.2, min=0.0, max=1.0),
                    Parameter("Lambda", value=0.3, min=0.0, max=5.0),
                    Parameter("A_bg", value=0.01, min=-0.2, max=0.2),
                ]
            ),
            message="ok",
        )
        for dataset in datasets
    }
    return GlobalCandidateAssessment(
        template=template,
        fit_results_by_run=fit_results,
        global_parameters=fit_results[int(datasets[0].run_number)].parameters,
        global_param_names=("A_1", "Lambda", "A_bg"),
        local_param_names=(),
        fixed_param_names=(),
        parameter_recommendations=(),
        run_diagnostics=tuple(diagnostics),
        series_warnings=(),
        aic=10.0,
        aicc=10.0,
        bic=12.0,
        selected_score=10.0,
        fitted_curves_by_run={},
        component_curves_by_run={},
    )


def _restrict_to_exp_constant_template(
    monkeypatch: pytest.MonkeyPatch,
    model: CompositeModel,
) -> CandidateTemplate:
    template = CandidateTemplate(
        key="exp_constant",
        title="Exponential + Constant",
        category="General",
        rationale="test",
        model=model,
    )
    monkeypatch.setattr(
        global_fit_wizard_module,
        "build_candidate_templates",
        lambda fingerprint, current_model=None: (template,),
    )
    return template


def test_global_fit_wizard_prefers_shared_exponential_for_uniform_series(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    _restrict_to_exp_constant_template(monkeypatch, model)
    datasets = [
        _dataset_for(
            run_number=100 + idx,
            field=50.0 * idx,
            temperature=5.0,
            model=model,
            params={"A_1": 0.2, "Lambda": 0.35, "A_bg": 0.01},
        )
        for idx in range(1, 5)
    ]

    recommendation = build_global_fit_wizard_recommendation(datasets)

    assert recommendation.recommended_assessment is not None
    assert recommendation.recommended_assessment.template.key == "exp_constant"
    assert recommendation.recommended_assessment.local_param_names == ()


def test_global_fit_wizard_localizes_lambda_when_series_rate_varies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    _restrict_to_exp_constant_template(monkeypatch, model)
    lambdas = [0.15, 0.25, 0.55, 0.9]
    datasets = [
        _dataset_for(
            run_number=200 + idx,
            field=100.0 * idx,
            temperature=10.0,
            model=model,
            params={"A_1": 0.2, "Lambda": lambdas[idx - 1], "A_bg": 0.01},
        )
        for idx in range(1, 5)
    ]

    recommendation = build_global_fit_wizard_recommendation(datasets)
    assessment = recommendation.recommended_assessment

    assert assessment is not None
    assert assessment.template.key == "exp_constant"
    assert "Lambda" in assessment.local_param_names
    assert "A_1" not in assessment.local_param_names


def test_global_fit_wizard_records_consolidated_search_instrumentation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    _restrict_to_exp_constant_template(monkeypatch, model)
    lambdas = [0.15, 0.25, 0.55, 0.9]
    datasets = [
        _dataset_for(
            run_number=260 + idx,
            field=100.0 * idx,
            temperature=10.0,
            model=model,
            params={"A_1": 0.2, "Lambda": lambdas[idx - 1], "A_bg": 0.01},
        )
        for idx in range(1, 5)
    ]
    instrumentation: dict[str, object] = {}

    recommendation = build_global_fit_wizard_recommendation(
        datasets,
        metric=SelectionMetric.BIC,
        instrumentation=instrumentation,
    )
    assessment = recommendation.recommended_assessment

    assert assessment is not None
    assert assessment.template.key == "exp_constant"
    counters = instrumentation.get("counters")
    assert isinstance(counters, dict)
    assert counters.get("exact_fit_invocations", 0) > 0
    assert counters.get("global_fit_calls", 0) > 0
    assert counters.get("curvature_hint_applications", 0) > 0
    assert counters.get("minuit_function_calls", 0) > 0
    assert instrumentation.get("strategy") == "consolidated"
    assert isinstance(instrumentation.get("staged_frontier_widths"), list)
    assert isinstance(instrumentation.get("relaxed_penalties"), list)
    assert isinstance(instrumentation.get("curvature_hint_sizes"), list)
    assert isinstance(instrumentation.get("minuit_edm"), list)


def test_global_fit_wizard_screening_recommendation_stays_prescreen_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    datasets = [
        _dataset_for(
            run_number=290 + idx,
            field=40.0 * idx,
            temperature=8.0,
            model=model,
            params={"A_1": 0.2, "Lambda": 0.2 + (0.08 * idx), "A_bg": 0.01},
        )
        for idx in range(1, 4)
    ]
    template = _restrict_to_exp_constant_template(monkeypatch, model)
    _stub_single_run_wizard(monkeypatch, (template,))

    recommendation = build_global_fit_wizard_screening_recommendation(datasets, current_model=model)

    assert recommendation.recommended_assessment is None
    assert recommendation.assessments
    assert all(assessment.prescreen_only for assessment in recommendation.assessments)
    assert "have not yet been optimized" in recommendation.summary


def _stub_single_run_wizard(
    monkeypatch: pytest.MonkeyPatch,
    templates: tuple[CandidateTemplate, ...],
    *,
    calls: list[int] | None = None,
) -> None:
    """Make phase 1's per-run single-run wizard cheap and deterministic.

    Phase 1 now runs the *full* single-run Fit Wizard per dataset — minutes of
    fitting on a real series, seconds even on a synthetic one. Every test below
    is about what phase 1 does with those analyses, not about the analyses
    themselves, so the per-run build is replaced by an explicit-template table
    stamped with the signature a real one would carry.
    """
    from asymmetry.core.fitting.fit_wizard import single_fit_build_signature

    def _fake_build(dataset, current_model=None, **kwargs):
        if calls is not None:
            calls.append(int(dataset.run_number))
        recommendation = build_fit_wizard_recommendation_for_templates(
            dataset,
            templates,
            metric=kwargs.get("metric", SelectionMetric.AICC),
        )
        return replace(
            recommendation,
            build_signature=single_fit_build_signature(
                kwargs.get("scope"), kwargs.get("user_frequencies_mhz")
            ),
        )

    monkeypatch.setattr(global_fit_wizard_module, "build_fit_wizard_recommendation", _fake_build)


def test_build_or_complete_single_fit_tables_reuses_a_matching_single_run_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reuse is keyed on the question asked, not on the template list returned.

    The old gate demanded that a cached recommendation's templates equal the
    series portfolio's exactly, which no genuine single-run analysis ever does —
    so every run was re-analysed. The rule is now the scope and user frequencies
    the cached analysis was built for.
    """
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    datasets = [
        _dataset_for(
            run_number=280 + idx,
            field=40.0 * idx,
            temperature=8.0,
            model=model,
            params={"A_1": 0.2, "Lambda": 0.2 + (0.1 * idx), "A_bg": 0.01},
        )
        for idx in range(1, 4)
    ]
    template = _restrict_to_exp_constant_template(monkeypatch, model)
    generated_calls: list[int] = []
    _stub_single_run_wizard(monkeypatch, (template,), calls=generated_calls)
    monkeypatch.setattr(
        global_fit_wizard_module, "_single_fit_table_worker_count", lambda _count: 1
    )

    existing_run = int(datasets[0].run_number)
    existing_recommendation = replace(
        build_fit_wizard_recommendation_for_templates(datasets[0], (template,)),
        build_signature=fit_wizard_module.single_fit_build_signature(None, None),
    )
    store = {existing_run: existing_recommendation}

    table = build_or_complete_single_fit_wizard_recommendations_for_global_portfolio(
        datasets,
        current_model=model,
        existing_recommendations_by_run=store,
    )

    assert table.portfolio.dataset_order == tuple(int(d.run_number) for d in datasets)
    assert table.single_fit_recommendations_by_run[existing_run] is existing_recommendation
    assert generated_calls == [int(datasets[1].run_number), int(datasets[2].run_number)]
    assert set(table.generated_run_numbers) == set(generated_calls)
    # The caller's store keeps the runs' own analyses, never the derived table.
    assert store[existing_run] is existing_recommendation
    assert set(store) == {int(dataset.run_number) for dataset in datasets}
    # ...and every run is scored against every alphabet template.
    for dataset in datasets:
        completed = table.recommendations_by_run[int(dataset.run_number)]
        assert [t.key for t in completed.templates] == [t.key for t in table.portfolio.templates]


def test_build_or_complete_single_fit_tables_does_not_reuse_another_scopes_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    datasets = [
        _dataset_for(
            run_number=286 + idx,
            field=40.0 * idx,
            temperature=8.0,
            model=model,
            params={"A_1": 0.2, "Lambda": 0.2 + (0.1 * idx), "A_bg": 0.01},
        )
        for idx in range(1, 3)
    ]
    template = _restrict_to_exp_constant_template(monkeypatch, model)
    generated_calls: list[int] = []
    _stub_single_run_wizard(monkeypatch, (template,), calls=generated_calls)
    monkeypatch.setattr(
        global_fit_wizard_module, "_single_fit_table_worker_count", lambda _count: 1
    )

    other_scope = WizardScope(preset=WizardScopePreset.FLUORIDE_FMUF)
    stale_run = int(datasets[0].run_number)
    stale = replace(
        build_fit_wizard_recommendation_for_templates(datasets[0], (template,)),
        build_signature=fit_wizard_module.single_fit_build_signature(other_scope, None),
    )

    build_or_complete_single_fit_wizard_recommendations_for_global_portfolio(
        datasets,
        current_model=model,
        existing_recommendations_by_run={stale_run: stale},
    )

    assert stale_run in generated_calls


#: Muon lifetime (us), shaping the invented error growth of the records below.
_TAU_MU = 2.197


def _damped_series_record(
    run_number: int,
    temperature: float,
    lines: tuple[tuple[float, float, float], ...],
    *,
    n_points: int = 600,
    t_max: float = 4.0,
) -> MuonDataset:
    """A zero-field record: optional damped cosines on a slowly relaxing tail.

    Every value is invented. The lines are heavily damped — dead inside the
    first few hundred nanoseconds — which is the regime only the single-run
    wizard's matched-apodisation scan finds.
    """
    dt = t_max / n_points
    time = np.arange(dt, t_max + 0.5 * dt, dt)[:n_points]
    sigma = np.minimum(0.9 * np.exp(time / (2.0 * _TAU_MU)), 100.0)
    asymmetry = 6.0 * np.exp(-0.25 * time) + 0.5
    for amplitude, frequency, rate in lines:
        asymmetry = asymmetry + amplitude * np.exp(-rate * time) * np.cos(
            2.0 * np.pi * frequency * time
        )
    rng = np.random.default_rng(20260906 + run_number)
    return MuonDataset(
        time=time,
        asymmetry=asymmetry + rng.normal(0.0, sigma),
        error=sigma,
        metadata={
            "run_number": run_number,
            "temperature": temperature,
            "field": 0.0,
            "field_direction": "ZF",
            "run_label": str(run_number),
        },
    )


def test_series_alphabet_holds_a_multiplet_only_a_minority_of_runs_show() -> None:
    """Two of five runs carry a damped pair; the series must still consider it.

    This is the defect Phase A exists for. The old portfolio was built from the
    *median* fingerprint plus a half-of-the-runs vote, so a template describing
    only the low-temperature phase of a series never entered it — and the global
    wizard could not recommend the model that fits half the data. Going through
    the single-run wizard per run and taking the union puts it first.
    """
    two_lines = ((24.0, 60.0, 10.0), (16.0, 30.0, 6.0))
    datasets = [
        _damped_series_record(700, 2.0, two_lines),
        _damped_series_record(701, 4.0, two_lines),
        _damped_series_record(702, 20.0, ()),
        _damped_series_record(703, 24.0, ()),
        _damped_series_record(704, 28.0, ()),
    ]
    instrumentation: dict[str, object] = {}

    table = build_or_complete_single_fit_wizard_recommendations_for_global_portfolio(
        datasets,
        scope=WizardScope(preset=WizardScopePreset.ZF_STATIC_MAGNETISM),
        instrumentation=instrumentation,
    )

    alphabet_keys = [template.key for template in table.portfolio.templates]
    multiplets = [key for key in alphabet_keys if key.startswith("oscillatory2_")]
    assert multiplets, alphabet_keys
    # ...and the runs that carry the pair chose it.
    chosen = {
        run_number: recommendation.recommended_key
        for run_number, recommendation in table.single_fit_recommendations_by_run.items()
    }
    assert chosen[700] in multiplets
    assert chosen[701] in multiplets
    # A multiplet template is evidence, so it is protected from tier trimming.
    assert set(multiplets) <= set(table.portfolio.pattern_template_keys)

    # Every (run, template) cell exists, at the one series resolution.
    for dataset in datasets:
        completed = table.recommendations_by_run[int(dataset.run_number)]
        assert completed.rebin_factor == table.series_rebin_factor
        for template in table.portfolio.templates:
            assert completed.assessment_for_key(template.key) is not None, (
                dataset.run_number,
                template.key,
            )

    assert instrumentation["alphabet_size"] == len(table.portfolio.templates)
    assert instrumentation["series_rebin_factor"] == table.series_rebin_factor
    assert int(instrumentation["completion_fits"]) >= 0


def test_series_rebin_factor_is_the_smallest_per_run_factor() -> None:
    """One resolution for the series, and it is the one every run's bandwidth allows."""
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    datasets = [
        _dataset_for(
            run_number=810 + idx,
            field=0.0,
            temperature=5.0 + idx,
            model=model,
            params={"A_1": 0.2, "Lambda": 0.3, "A_bg": 0.01},
        )
        for idx in range(3)
    ]
    template = CandidateTemplate(
        key="exp_constant", title="e", category="General", rationale="t", model=model
    )
    factors = (4, 2, 7)
    recommendations = {
        int(dataset.run_number): replace(
            build_fit_wizard_recommendation_for_templates(dataset, (template,)),
            rebin_factor=factor,
        )
        for dataset, factor in zip(datasets, factors)
    }

    assert global_fit_wizard_module.series_rebin_factor(datasets, recommendations) == 2


def test_completion_refits_a_cell_recorded_at_another_rebin_factor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run analysed at a coarser resolution is re-scored, not copied across.

    Information criteria are only summable when every run refers to a record of
    the same construction, so a cell carried over at the wrong factor would make
    the whole series table incomparable.
    """
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    datasets = [
        _dataset_for(
            run_number=820 + idx,
            field=0.0,
            temperature=5.0 + idx,
            model=model,
            params={"A_1": 0.2, "Lambda": 0.3 + 0.1 * idx, "A_bg": 0.01},
        )
        for idx in range(2)
    ]
    template = _restrict_to_exp_constant_template(monkeypatch, model)

    coarse_run = int(datasets[0].run_number)

    def _fake_build(dataset, current_model=None, **kwargs):
        recommendation = build_fit_wizard_recommendation_for_templates(dataset, (template,))
        return replace(
            recommendation,
            build_signature=fit_wizard_module.single_fit_build_signature(None, None),
            rebin_factor=3 if int(dataset.run_number) == coarse_run else 1,
        )

    monkeypatch.setattr(global_fit_wizard_module, "build_fit_wizard_recommendation", _fake_build)
    monkeypatch.setattr(
        global_fit_wizard_module, "_single_fit_table_worker_count", lambda _count: 1
    )

    table = build_or_complete_single_fit_wizard_recommendations_for_global_portfolio(datasets)

    assert table.series_rebin_factor == 1
    coarse_source = table.single_fit_recommendations_by_run[coarse_run]
    coarse_completed = table.recommendations_by_run[coarse_run]
    assert coarse_completed.rebin_factor == 1
    # Re-fitted, not carried over.
    assert coarse_completed.assessment_for_key(
        "exp_constant"
    ) is not coarse_source.assessment_for_key("exp_constant")
    # The run whose factor already matched keeps its own assessment object.
    fresh_run = int(datasets[1].run_number)
    assert table.recommendations_by_run[fresh_run].assessment_for_key(
        "exp_constant"
    ) is table.single_fit_recommendations_by_run[fresh_run].assessment_for_key("exp_constant")


def test_build_or_complete_single_fit_tables_uses_spawn_safe_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The completion fits — one task per run — go over the spawn pool."""
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    datasets = [
        _dataset_for(
            run_number=380 + idx,
            field=20.0 * idx,
            temperature=8.0,
            model=model,
            params={"A_1": 0.2, "Lambda": 0.2 + (0.05 * idx), "A_bg": 0.01},
        )
        for idx in range(1, 4)
    ]
    template = _restrict_to_exp_constant_template(monkeypatch, model)
    _stub_single_run_wizard(monkeypatch, (template,))

    created_with: dict[str, object] = {}

    class _FakeExecutor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args):
            # A real, already-resolved Future: the drain waits on these through
            # ``concurrent.futures.wait``, which reaches into Future internals.
            future: Future = Future()
            future.set_result(fn(*args))
            return future

        def shutdown(self, *args, **kwargs):
            return None

    def _fake_open(max_workers):
        created_with["max_workers"] = max_workers
        return _FakeExecutor()

    # Pool creation is delegated to the shared spawn-safe helper; intercept it there.
    monkeypatch.setattr(global_fit_wizard_module, "open_spawn_pool", _fake_open)
    # Pin the host width so this asserts the per-run cap, not the runner's cores.
    monkeypatch.setattr(global_fit_wizard_module.os, "cpu_count", lambda: 8)

    table = build_or_complete_single_fit_wizard_recommendations_for_global_portfolio(
        datasets,
        current_model=model,
    )

    assert created_with["max_workers"] == 3
    assert set(table.recommendations_by_run) == {int(dataset.run_number) for dataset in datasets}
    assert set(table.generated_run_numbers) == {int(dataset.run_number) for dataset in datasets}


class _Phase1Recorder:
    """What stage 1 asked of the single-run wizard, and when it asked it."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0
        self.started: list[int] = []
        self.finished: list[int] = []
        self.max_workers: list[int | None] = []
        self.warm_starts: dict[int, dict[str, ParameterSet]] = {}


def _install_recording_single_run_wizard(
    monkeypatch: pytest.MonkeyPatch,
    datasets: list[MuonDataset],
    template: CandidateTemplate,
    recorder: _Phase1Recorder,
    *,
    delays: dict[int, float],
    on_start: Callable[[_Phase1Recorder], None] | None = None,
) -> dict[int, FitWizardRecommendation]:
    """Replace the per-run analysis with a timed stand-in of known duration.

    Every run's recommendation is fitted once up front and simply handed back
    after the run's ``delays`` sleep, so the stage's timing is the sleep and
    nothing else — which is what lets a test assert *completion* order.
    """
    from asymmetry.core.fitting.fit_wizard import single_fit_build_signature

    prepared = {
        int(dataset.run_number): replace(
            build_fit_wizard_recommendation_for_templates(dataset, (template,)),
            build_signature=single_fit_build_signature(None, None),
        )
        for dataset in datasets
    }

    def _fake_build(dataset, current_model=None, **kwargs):
        run_number = int(dataset.run_number)
        with recorder.lock:
            recorder.started.append(run_number)
            recorder.in_flight += 1
            recorder.max_in_flight = max(recorder.max_in_flight, recorder.in_flight)
            recorder.max_workers.append(kwargs.get("max_workers"))
            recorder.warm_starts[run_number] = dict(kwargs.get("warm_start_by_template") or {})
        if on_start is not None:
            on_start(recorder)
        time.sleep(delays[run_number])
        with recorder.lock:
            recorder.in_flight -= 1
            recorder.finished.append(run_number)
        return prepared[run_number]

    monkeypatch.setattr(global_fit_wizard_module, "build_fit_wizard_recommendation", _fake_build)
    return prepared


def _phase1_datasets(model: CompositeModel, count: int, *, first_run: int) -> list[MuonDataset]:
    return [
        _dataset_for(
            run_number=first_run + idx,
            field=0.0,
            temperature=5.0 + idx,
            model=model,
            params={"A_1": 0.2, "Lambda": 0.2 + 0.05 * idx, "A_bg": 0.01},
        )
        for idx in range(count)
    ]


def test_phase_one_analyses_several_runs_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stage 1 overlaps per-run analyses and reports them as they finish.

    Each analysis leaves most of a machine idle through its serial stages
    (detection, pattern search, tier gating), so a few run side by side with the
    worker count divided between them. Progress stays per completed run.
    """
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    datasets = _phase1_datasets(model, 4, first_run=900)
    template = _restrict_to_exp_constant_template(monkeypatch, model)
    recorder = _Phase1Recorder()
    # The first-submitted run is the slowest, so completion order is not
    # submission order; the gaps are wide against the drain's poll interval.
    delays = {900: 0.6, 901: 0.4, 902: 0.2, 903: 0.3}
    _install_recording_single_run_wizard(monkeypatch, datasets, template, recorder, delays=delays)
    monkeypatch.setattr(global_fit_wizard_module.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(global_fit_wizard_module, "_PHASE_ONE_CANCEL_POLL_SECONDS", 0.01)
    monkeypatch.setattr(
        global_fit_wizard_module, "_single_fit_table_worker_count", lambda _count: 1
    )
    progress: list[str] = []
    instrumentation: dict[str, object] = {}

    table = build_or_complete_single_fit_wizard_recommendations_for_global_portfolio(
        datasets,
        current_model=model,
        phase1_concurrency=3,
        progress_callback=progress.append,
        instrumentation=instrumentation,
    )

    assert recorder.max_in_flight > 1
    # Three analyses at once on an eight-core host: 8 // 3 workers each, so the
    # total worker count stays at the machine's width.
    assert set(recorder.max_workers) == {8 // 3}
    assert instrumentation["phase1_concurrency"] == 3
    # Every run but the first-started gets a finished neighbour's values.
    assert 1 <= int(instrumentation["phase1_warm_seeded"]) <= 3
    # Reported per completed run, in completion order.
    assert recorder.finished == [902, 901, 903, 900]
    assert table.generated_run_numbers == tuple(recorder.finished)
    reported = [
        message.split()[2].rstrip(":")
        for message in progress
        if message.startswith("Single-fit analysis") and ": done" in message
    ]
    assert reported == [str(run_number) for run_number in recorder.finished]


def test_phase_one_cancel_during_stage_one_raises_promptly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancel while the per-run analyses are running comes back at once."""
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    datasets = _phase1_datasets(model, 6, first_run=910)
    template = _restrict_to_exp_constant_template(monkeypatch, model)
    recorder = _Phase1Recorder()
    cancelled = threading.Event()
    _install_recording_single_run_wizard(
        monkeypatch,
        datasets,
        template,
        recorder,
        delays={int(dataset.run_number): 0.2 for dataset in datasets},
        on_start=lambda _recorder: cancelled.set(),
    )
    monkeypatch.setattr(global_fit_wizard_module, "_PHASE_ONE_CANCEL_POLL_SECONDS", 0.01)

    started_at = time.monotonic()
    with pytest.raises(FitCancelledError):
        build_or_complete_single_fit_wizard_recommendations_for_global_portfolio(
            datasets,
            current_model=model,
            phase1_concurrency=2,
            cancel_callback=cancelled.is_set,
        )

    assert time.monotonic() - started_at < 5.0
    # The queued analyses were dropped rather than run to completion.
    assert len(recorder.started) < len(datasets)


def test_phase_one_warm_seeds_each_run_from_the_nearest_finished_neighbour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run's analysis starts from the values of the run beside it on the axis.

    Serial here so "nearest finished neighbour" is exactly the previous run and
    the assertion can name the values; with concurrency the same rule picks
    whichever neighbour has finished.
    """
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    datasets = _phase1_datasets(model, 4, first_run=920)
    template = _restrict_to_exp_constant_template(monkeypatch, model)
    recorder = _Phase1Recorder()
    prepared = _install_recording_single_run_wizard(
        monkeypatch,
        datasets,
        template,
        recorder,
        delays={int(dataset.run_number): 0.0 for dataset in datasets},
    )
    monkeypatch.setattr(
        global_fit_wizard_module, "_single_fit_table_worker_count", lambda _count: 1
    )
    instrumentation: dict[str, object] = {}

    build_or_complete_single_fit_wizard_recommendations_for_global_portfolio(
        datasets,
        current_model=model,
        phase1_concurrency=1,
        instrumentation=instrumentation,
    )

    order = [int(dataset.run_number) for dataset in datasets]
    assert recorder.started == order
    assert recorder.warm_starts[order[0]] == {}
    for previous, current in zip(order, order[1:]):
        neighbour = {
            assessment.template.key: {
                parameter.name: parameter.value for parameter in assessment.fit_result.parameters
            }
            for assessment in prepared[previous].assessments
            if assessment.is_successful
        }
        seeded = {
            key: {parameter.name: parameter.value for parameter in parameters}
            for key, parameters in recorder.warm_starts[current].items()
        }
        assert seeded == neighbour
        assert seeded
    assert int(instrumentation["phase1_warm_seeded"]) == len(datasets) - 1


def test_phase_one_concurrency_splits_the_host_between_analyses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The derived width never puts more fit workers on the host than cores."""
    monkeypatch.setattr(global_fit_wizard_module.os, "cpu_count", lambda: 8)
    assert global_fit_wizard_module._phase_one_concurrency(12) == 4
    assert global_fit_wizard_module._phase_one_analysis_workers(4) == 2
    monkeypatch.setattr(global_fit_wizard_module.os, "cpu_count", lambda: 2)
    assert global_fit_wizard_module._phase_one_concurrency(12) == 1
    assert global_fit_wizard_module._phase_one_analysis_workers(1) == 2
    monkeypatch.setattr(global_fit_wizard_module.os, "cpu_count", lambda: 16)
    assert global_fit_wizard_module._phase_one_concurrency(3) == 3
    assert global_fit_wizard_module._phase_one_analysis_workers(3) == 5


def _template_named(key: str, *, model: CompositeModel | None = None) -> CandidateTemplate:
    return CandidateTemplate(
        key=key,
        title=key,
        category="General",
        rationale="test",
        model=model if model is not None else CompositeModel(["Exponential"], operators=[]),
    )


@lru_cache(maxsize=1)
def _synthetic_fingerprint() -> SpectrumFingerprint:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    return fingerprint_spectrum(
        _dataset_for(
            run_number=1,
            field=0.0,
            temperature=5.0,
            model=model,
            params={"A_1": 0.2, "Lambda": 0.3, "A_bg": 0.01},
        )
    )


def _scored_assessment(
    template: CandidateTemplate, *, chi_squared: float, criterion: float
) -> CandidateAssessment:
    return CandidateAssessment(
        template=template,
        fit_result=FitResult(
            success=True,
            chi_squared=chi_squared,
            reduced_chi_squared=chi_squared / 10.0,
            parameters=ParameterSet([Parameter("A_1", value=0.2)]),
            message="ok",
        ),
        aic=criterion,
        aicc=criterion,
        bic=criterion,
        selected_score=criterion,
        residual_rms=0.001,
        runs_z_score=0.1,
        max_abs_autocorrelation=0.1,
        residual_fft_peak_snr=1.0,
        residual_gate_passed=True,
        residual_gate_reasons=(),
        bound_hits=(),
        fitted_time=np.array([0.0, 1.0]),
        fitted_curve=np.array([0.2, 0.1]),
        component_curves=(),
    )


def _scored_recommendation(
    scores: dict[CandidateTemplate, tuple[float, float]],
    *,
    recommended_key: str | None = None,
    comparable_keys: tuple[str, ...] = (),
) -> FitWizardRecommendation:
    templates = tuple(scores)
    return FitWizardRecommendation(
        fingerprint=_synthetic_fingerprint(),
        templates=templates,
        assessments=tuple(
            _scored_assessment(template, chi_squared=chi_squared, criterion=criterion)
            for template, (chi_squared, criterion) in scores.items()
        ),
        metric=SelectionMetric.AICC,
        recommended_key=recommended_key,
        comparable_keys=comparable_keys,
        summary="",
    )


def test_alphabet_bound_drops_only_templates_one_rival_beats_on_every_run() -> None:
    """The exact bound: a fixed rival's IC below this template's bare χ² everywhere."""
    winner = _template_named("winner")
    loser = _template_named("loser")
    single_win = _template_named("single_win")
    comparable_once = _template_named("comparable_once")
    borderline = _template_named("borderline")
    scores = {
        winner: (10.0, 20.0),
        loser: (40.0, 60.0),
        single_win: (40.0, 60.0),
        comparable_once: (40.0, 60.0),
        borderline: (25.0, 45.0),
    }
    recommendations = {
        1: _scored_recommendation(scores, recommended_key="winner"),
        2: _scored_recommendation(
            scores, recommended_key="single_win", comparable_keys=("comparable_once",)
        ),
        3: _scored_recommendation({**scores, borderline: (15.0, 35.0)}, recommended_key="winner"),
    }

    dropped = global_fit_wizard_module.alphabet_bound_dropped_keys(recommendations)

    # Beaten by ``winner`` on every run and chosen by none: it cannot win any
    # segment, whatever the sharing pattern.
    assert "loser" in dropped
    # A run's own verdict is never bounded away, whether it won or merely tied.
    assert "single_win" not in dropped
    assert "comparable_once" not in dropped
    # χ² below the best criterion on run 3, so no rival dominates it everywhere.
    assert "borderline" not in dropped
    assert "winner" not in dropped
    # ...and the ranked alphabet keeps its order once the dropped key is out.
    alphabet = [
        template.key
        for template in global_fit_wizard_module.series_template_alphabet(recommendations)
        if template.key not in dropped
    ]
    assert set(alphabet) == {"winner", "single_win", "comparable_once", "borderline"}
    assert alphabet[0] == "winner"


def test_alphabet_bound_needs_one_rival_not_a_per_run_best() -> None:
    """Comparing against each run's *best* criterion would not be exact.

    Two runs where the leader swaps: ``rival_a`` at (10, 100) and ``rival_b`` at
    (100, 10) both clear ``middling``'s χ² of 11 on one run and lose badly on
    the other, so ``middling`` beats either of them over the two runs (24
    against 110) despite exceeding the per-run minimum criterion on both.
    """
    rival_a = _template_named("rival_a")
    rival_b = _template_named("rival_b")
    middling = _template_named("middling")
    recommendations = {
        1: _scored_recommendation(
            {rival_a: (5.0, 10.0), rival_b: (95.0, 100.0), middling: (11.0, 12.0)},
            recommended_key="rival_a",
        ),
        2: _scored_recommendation(
            {rival_a: (95.0, 100.0), rival_b: (5.0, 10.0), middling: (11.0, 12.0)},
            recommended_key="rival_b",
        ),
    }

    assert global_fit_wizard_module.alphabet_bound_dropped_keys(recommendations) == frozenset()


def test_completion_fits_a_cold_cell_with_a_short_ladder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cell no run has fitted gets two rungs; one with values to start from, one."""
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    dataset = _dataset_for(
        run_number=940,
        field=0.0,
        temperature=5.0,
        model=model,
        params={"A_1": 0.2, "Lambda": 0.3, "A_bg": 0.01},
    )
    fitted = _template_named("fitted", model=model)
    cold = _template_named("cold", model=CompositeModel(["Exponential"], operators=[]))
    source = build_fit_wizard_recommendation_for_templates(dataset, (fitted,))
    budgets: list[tuple[tuple[str, ...], int]] = []
    real = global_fit_wizard_module.build_fit_wizard_recommendation_for_templates

    def _record(target, templates, **kwargs):
        budgets.append((tuple(t.key for t in templates), int(kwargs["variant_budget"])))
        return real(target, templates, **kwargs)

    monkeypatch.setattr(
        global_fit_wizard_module, "build_fit_wizard_recommendation_for_templates", _record
    )

    global_fit_wizard_module._single_fit_completion_task(
        dataset, (fitted, cold), source, 1, SelectionMetric.AICC, {}
    )
    assert budgets == [(("cold",), 2)]

    budgets.clear()
    sibling = source.assessment_for_key("fitted")
    assert sibling is not None
    global_fit_wizard_module._single_fit_completion_task(
        dataset,
        (fitted, cold),
        source,
        1,
        SelectionMetric.AICC,
        {"cold": sibling.fit_result.parameters},
    )
    assert budgets == [(("cold",), 1)]


def test_global_fit_wizard_uses_single_fit_prescreen_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    _restrict_to_exp_constant_template(monkeypatch, model)
    lambdas = [0.18, 0.26, 0.44]
    datasets = [
        _dataset_for(
            run_number=320 + idx,
            field=60.0 * idx,
            temperature=6.0,
            model=model,
            params={"A_1": 0.2, "Lambda": lambdas[idx - 1], "A_bg": 0.01},
        )
        for idx in range(1, 4)
    ]
    portfolio = build_global_fit_wizard_candidate_portfolio(datasets, current_model=model)
    single_fit_recommendations_by_run = {
        int(dataset.run_number): build_fit_wizard_recommendation_for_templates(
            dataset,
            portfolio.templates,
        )
        for dataset in datasets
    }

    original = global_fit_wizard_module._build_single_fit_prescreen_assessments
    observed: dict[str, bool] = {"called": False}

    def _wrapped(*args, **kwargs):
        observed["called"] = True
        return original(*args, **kwargs)

    monkeypatch.setattr(
        global_fit_wizard_module,
        "_build_single_fit_prescreen_assessments",
        _wrapped,
    )

    recommendation = build_global_fit_wizard_recommendation(
        datasets,
        current_model=model,
        single_fit_recommendations_by_run=single_fit_recommendations_by_run,
    )

    assert observed["called"] is True
    assert recommendation.recommended_assessment is not None
    assert recommendation.recommended_assessment.template.key == "exp_constant"


def test_global_fit_wizard_selected_candidate_optimisation_merges_into_screening() -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    datasets = [
        _dataset_for(
            run_number=340 + idx,
            field=55.0 * idx,
            temperature=7.0,
            model=model,
            params={"A_1": 0.2, "Lambda": 0.22 + (0.12 * idx), "A_bg": 0.01},
        )
        for idx in range(1, 4)
    ]
    portfolio = build_global_fit_wizard_candidate_portfolio(datasets, current_model=model)
    single_fit_recommendations_by_run = {
        int(dataset.run_number): build_fit_wizard_recommendation_for_templates(
            dataset,
            portfolio.templates,
        )
        for dataset in datasets
    }

    screening = build_global_fit_wizard_screening_recommendation(
        datasets,
        current_model=model,
        single_fit_recommendations_by_run=single_fit_recommendations_by_run,
    )
    optimized = build_global_fit_wizard_recommendation(
        datasets,
        current_model=model,
        single_fit_recommendations_by_run=single_fit_recommendations_by_run,
        selected_template_keys=("exp_constant",),
    )
    merged = merge_global_fit_wizard_recommendations(screening, optimized)

    assert screening.assessment_for_key("exp_constant") is not None
    assert screening.assessment_for_key("exp_constant").prescreen_only is True
    assert merged.assessment_for_key("exp_constant") is not None
    assert merged.assessment_for_key("exp_constant").prescreen_only is True
    assert any(
        not assessment.prescreen_only
        for assessment in merged.assessments_for_template_key("exp_constant")
    )
    assert merged.recommended_assessment is not None
    assert merged.recommended_assessment.template.key == "exp_constant"
    assert len({assessment.selection_key for assessment in merged.optimized_assessments()}) == len(
        merged.optimized_assessments()
    )


def test_filtered_gate_reasons_drops_bound_hit_when_fit_is_good() -> None:
    """Fix C1: a parameter resting at its bound is the valid answer (e.g. a
    relaxation rate -> 0) when the fit is otherwise good, so it is not a gate
    failure; it only vetoes when the fit is also poor."""
    result = FitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        parameters=ParameterSet(),
        message="ok",
    )
    good = global_fit_wizard_module._filtered_gate_reasons(
        fit_result=result,
        residual_rms=1.0,
        runs_z_score=0.0,
        max_abs_autocorrelation=0.0,
        residual_fft_peak_snr=0.0,
        bound_hits=["Lambda at lower bound"],
    )
    assert good == []
    poor = global_fit_wizard_module._filtered_gate_reasons(
        fit_result=result,
        residual_rms=2.0,
        runs_z_score=0.0,
        max_abs_autocorrelation=0.0,
        residual_fft_peak_snr=0.0,
        bound_hits=["Lambda at lower bound"],
    )
    assert "Lambda at lower bound" in poor


def test_rerank_recommends_tentatively_when_only_series_warnings_block() -> None:
    """Fix C2: when every run clears its per-run residual gate (a demonstrably
    strong coupled fit) but a heuristic series-consistency warning fires, the
    wizard recommends the candidate tentatively with a caveat rather than
    returning None. A per-run gate failure still blocks the recommendation."""
    datasets = [
        _dataset_for(
            run_number=700 + idx,
            field=30.0 * idx,
            temperature=6.0,
            model=CompositeModel(["Exponential", "Constant"], operators=["+"]),
            params={"A_1": 0.2, "Lambda": 0.3 + (0.05 * idx), "A_bg": 0.01},
        )
        for idx in range(1, 4)
    ]
    diagnostics = [
        RunResidualDiagnostic(
            run_number=int(dataset.run_number),
            run_label=dataset.run_label,
            axis_value=float(dataset.metadata["field"]),
            residual_rms=0.1,
            runs_z_score=0.0,
            max_abs_autocorrelation=0.0,
            residual_fft_peak_snr=0.0,
            gate_passed=True,
            gate_reasons=(),
        )
        for dataset in datasets
    ]
    assessment = replace(
        _assessment_with_diagnostics(datasets, diagnostics),
        global_param_names=("A_1", "A_bg"),
        local_param_names=("Lambda",),
        series_warnings=("Fingerprint features change abruptly between 2 and 3.",),
    )

    def _recommendation(assessments):
        return GlobalFitWizardRecommendation(
            series_axis_key="field",
            series_axis_label="Field (G)",
            mixed_axes_warning=None,
            fingerprints_by_run={},
            dataset_order=tuple(int(d.run_number) for d in datasets),
            templates=(assessment.template,),
            assessments=assessments,
            metric=SelectionMetric.AICC,
            recommended_key=None,
            comparable_keys=(),
            summary="",
        )

    reranked = rerank_global_fit_wizard_recommendation(
        _recommendation((assessment,)), SelectionMetric.AICC
    )
    assert reranked.recommended_key == assessment.selection_key
    assert "tentative" in reranked.summary.lower()
    assert "Fingerprint features change abruptly" in reranked.summary

    # A genuine per-run gate failure must still block (stays None).
    failed = list(diagnostics)
    failed[0] = replace(failed[0], gate_passed=False, gate_reasons=("bad fit",))
    blocked = replace(assessment, run_diagnostics=tuple(failed))
    reranked_blocked = rerank_global_fit_wizard_recommendation(
        _recommendation((blocked,)), SelectionMetric.AICC
    )
    assert reranked_blocked.recommended_key is None


def test_merge_global_fit_wizard_recommendations_keeps_all_optimized_variants() -> None:
    datasets = [
        _dataset_for(
            run_number=440 + idx,
            field=30.0 * idx,
            temperature=6.0,
            model=CompositeModel(["Exponential", "Constant"], operators=["+"]),
            params={"A_1": 0.2, "Lambda": 0.3 + (0.05 * idx), "A_bg": 0.01},
        )
        for idx in range(1, 4)
    ]
    diagnostics = [
        RunResidualDiagnostic(
            run_number=int(dataset.run_number),
            run_label=dataset.run_label,
            axis_value=float(dataset.metadata["field"]),
            residual_rms=0.1,
            runs_z_score=0.0,
            max_abs_autocorrelation=0.0,
            residual_fft_peak_snr=0.0,
            gate_passed=True,
            gate_reasons=(),
        )
        for dataset in datasets
    ]
    screening_assessment = replace(
        _assessment_with_diagnostics(datasets, diagnostics),
        prescreen_only=True,
        global_parameters=ParameterSet(),
    )
    local_variant = replace(
        _assessment_with_diagnostics(datasets, diagnostics),
        global_param_names=("A_1", "A_bg"),
        local_param_names=("Lambda",),
        aic=8.0,
        aicc=8.0,
        bic=10.0,
        selected_score=8.0,
        assessment_key="exp_constant|g=A_1,A_bg|l=Lambda",
    )
    shared_variant = replace(
        _assessment_with_diagnostics(datasets, diagnostics),
        global_param_names=("A_1", "Lambda", "A_bg"),
        local_param_names=(),
        aic=9.0,
        aicc=9.0,
        bic=11.0,
        selected_score=9.0,
        assessment_key="exp_constant|g=A_1,Lambda,A_bg|l=none",
    )
    screening = GlobalFitWizardRecommendation(
        series_axis_key="field",
        series_axis_label="Field (G)",
        mixed_axes_warning=None,
        fingerprints_by_run={},
        dataset_order=tuple(int(dataset.run_number) for dataset in datasets),
        templates=(screening_assessment.template,),
        assessments=(screening_assessment,),
        metric=SelectionMetric.AICC,
        recommended_key=None,
        comparable_keys=(),
        summary="screening",
    )
    optimized = GlobalFitWizardRecommendation(
        series_axis_key="field",
        series_axis_label="Field (G)",
        mixed_axes_warning=None,
        fingerprints_by_run={},
        dataset_order=tuple(int(dataset.run_number) for dataset in datasets),
        templates=(screening_assessment.template,),
        assessments=(screening_assessment, local_variant, shared_variant),
        metric=SelectionMetric.AICC,
        recommended_key=local_variant.selection_key,
        comparable_keys=(local_variant.selection_key, shared_variant.selection_key),
        summary="optimized",
    )

    merged = merge_global_fit_wizard_recommendations(screening, optimized)

    assert len(merged.sorted_prescreen_assessments()) == 1
    assert len(merged.sorted_optimized_assessments()) == 2
    assert {assessment.selection_key for assessment in merged.sorted_optimized_assessments()} == {
        local_variant.selection_key,
        shared_variant.selection_key,
    }
    assert merged.assessment_for_key(shared_variant.selection_key) is not None


def test_merge_global_fit_wizard_recommendations_prefers_simpler_comparable_variant() -> None:
    datasets = [
        _dataset_for(
            run_number=440 + idx,
            field=30.0 * idx,
            temperature=6.0,
            model=CompositeModel(["Exponential", "Constant"], operators=["+"]),
            params={"A_1": 0.2, "Lambda": 0.3 + (0.05 * idx), "A_bg": 0.01},
        )
        for idx in range(1, 4)
    ]
    diagnostics = [
        RunResidualDiagnostic(
            run_number=int(dataset.run_number),
            run_label=dataset.run_label,
            axis_value=float(dataset.metadata["field"]),
            residual_rms=0.1,
            runs_z_score=0.0,
            max_abs_autocorrelation=0.0,
            residual_fft_peak_snr=0.0,
            gate_passed=True,
            gate_reasons=(),
        )
        for dataset in datasets
    ]
    screening_assessment = replace(
        _assessment_with_diagnostics(datasets, diagnostics),
        prescreen_only=True,
        global_parameters=ParameterSet(),
    )
    local_variant = replace(
        _assessment_with_diagnostics(datasets, diagnostics),
        global_param_names=("A_1", "A_bg"),
        local_param_names=("Lambda",),
        aic=8.0,
        aicc=8.0,
        bic=10.0,
        selected_score=8.0,
        assessment_key="exp_constant|g=A_1,A_bg|l=Lambda",
    )
    shared_variant = replace(
        _assessment_with_diagnostics(datasets, diagnostics),
        global_param_names=("A_1", "Lambda", "A_bg"),
        local_param_names=(),
        aic=9.0,
        aicc=9.0,
        bic=11.0,
        selected_score=9.0,
        assessment_key="exp_constant|g=A_1,Lambda,A_bg|l=none",
    )
    screening = GlobalFitWizardRecommendation(
        series_axis_key="field",
        series_axis_label="Field (G)",
        mixed_axes_warning=None,
        fingerprints_by_run={},
        dataset_order=tuple(int(dataset.run_number) for dataset in datasets),
        templates=(screening_assessment.template,),
        assessments=(screening_assessment,),
        metric=SelectionMetric.AICC,
        recommended_key=None,
        comparable_keys=(),
        summary="screening",
    )
    optimized = GlobalFitWizardRecommendation(
        series_axis_key="field",
        series_axis_label="Field (G)",
        mixed_axes_warning=None,
        fingerprints_by_run={},
        dataset_order=tuple(int(dataset.run_number) for dataset in datasets),
        templates=(screening_assessment.template,),
        assessments=(screening_assessment, local_variant, shared_variant),
        metric=SelectionMetric.AICC,
        recommended_key=local_variant.selection_key,
        comparable_keys=(local_variant.selection_key, shared_variant.selection_key),
        summary="optimized",
    )

    merged = merge_global_fit_wizard_recommendations(screening, optimized)

    assert len(merged.sorted_prescreen_assessments()) == 1
    assert len(merged.sorted_optimized_assessments()) == 2
    assert merged.recommended_assessment is not None
    assert merged.recommended_assessment.selection_key == shared_variant.selection_key
    assert set(merged.comparable_keys) == {
        local_variant.selection_key,
        shared_variant.selection_key,
    }
    assert merged.assessment_for_key(shared_variant.selection_key) is not None


def test_selected_candidate_optimisation_skips_prescreen_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    datasets = [
        _dataset_for(
            run_number=350 + idx,
            field=30.0 * idx,
            temperature=7.0,
            model=model,
            params={"A_1": 0.2, "Lambda": 0.2 + (0.08 * idx), "A_bg": 0.01},
        )
        for idx in range(1, 4)
    ]
    portfolio = build_global_fit_wizard_candidate_portfolio(datasets, current_model=model)
    single_fit_recommendations_by_run = {
        int(dataset.run_number): build_fit_wizard_recommendation_for_templates(
            dataset,
            portfolio.templates,
        )
        for dataset in datasets
    }
    repair_calls: list[str] = []

    original_repair = global_fit_wizard_module._repair_partial_single_fit_prescreen_assessments

    def _wrapped(*args, **kwargs):
        repair_calls.append(kwargs["template"].key if "template" in kwargs else args[2].key)
        return original_repair(*args, **kwargs)

    monkeypatch.setattr(
        global_fit_wizard_module,
        "_repair_partial_single_fit_prescreen_assessments",
        _wrapped,
    )

    recommendation = build_global_fit_wizard_recommendation(
        datasets,
        current_model=model,
        single_fit_recommendations_by_run=single_fit_recommendations_by_run,
        selected_template_keys=("exp_constant",),
    )

    assert repair_calls == []
    assert recommendation.assessment_for_key("exp_constant") is not None


def test_global_fit_wizard_screening_completes_a_failed_cell_from_a_sibling(
    monkeypatch,
) -> None:
    """A failed cell in a run's own analysis is retried by the completion pass.

    The per-run analyses the fit tabs cache are what the global wizard receives;
    phase 1 completes them into the series table, refitting a failed cell from
    the best sibling run's values, so no separate serial repair pass is needed.
    """
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    datasets = [
        _dataset_for(
            run_number=360 + idx,
            field=40.0 * idx,
            temperature=6.0,
            model=model,
            params={"A_1": 0.2, "Lambda": lambda_value, "A_bg": 0.01},
        )
        for idx, lambda_value in enumerate((0.18, 0.42, 0.55), start=1)
    ]
    portfolio = build_global_fit_wizard_candidate_portfolio(datasets, current_model=model)
    own_analyses = {
        int(dataset.run_number): build_fit_wizard_recommendation_for_templates(
            dataset, portfolio.templates
        )
        for dataset in datasets
    }
    # Stand in for the per-run single-fit wizard: the derived tables above are
    # every run's "own analysis" here, so stage 1 costs nothing and no pool
    # nests under the test runner's.
    monkeypatch.setattr(
        global_fit_wizard_module,
        "build_fit_wizard_recommendation",
        lambda dataset, *_args, **_kwargs: own_analyses[int(dataset.run_number)],
    )
    table = build_or_complete_single_fit_wizard_recommendations_for_global_portfolio(
        datasets, current_model=model
    )
    failed_run = int(datasets[1].run_number)

    def _with_failed_cell(recommendation):
        return replace(
            recommendation,
            assessments=tuple(
                replace(
                    assessment,
                    fit_result=FitResult(
                        success=False,
                        chi_squared=float("inf"),
                        reduced_chi_squared=float("inf"),
                        parameters=assessment.fit_result.parameters,
                        message="forced single-fit failure",
                    ),
                    aic=float("inf"),
                    aicc=None,
                    bic=float("inf"),
                    selected_score=float("inf"),
                )
                if assessment.template.key == "exp_constant"
                else assessment
                for assessment in recommendation.assessments
            ),
        )

    forced = _with_failed_cell(table.recommendations_by_run[failed_run])
    # The run's *own analysis* carries the failure: this is the fit tab's cache
    # a user hands the global wizard, and phase 1's completion pass — not a
    # separate repair — is what puts a converged fit back in that cell.
    own_analyses[failed_run] = forced
    instrumentation: dict[str, object] = {}
    completed = build_or_complete_single_fit_wizard_recommendations_for_global_portfolio(
        datasets,
        current_model=model,
        existing_recommendations_by_run=dict(own_analyses),
        instrumentation=instrumentation,
    )
    assert int(instrumentation["completion_fits"]) >= 1
    retried = completed.recommendations_by_run[failed_run].assessment_for_key("exp_constant")
    assert retried is not None and retried.fit_result.success is True
    # A completed table covers its portfolio even where a cell failed: the
    # completion pass already retried it, so a failure there is the answer.
    # The failure is stamped on *this* run's completed table, because the
    # alphabet is derived from the analyses handed in and the two calls above
    # were handed different ones.
    supplied = dict(completed.recommendations_by_run)
    supplied[failed_run] = _with_failed_cell(completed.recommendations_by_run[failed_run])
    assert single_fit_table_covers_portfolio(datasets, completed.portfolio.templates, supplied)

    progress_messages: list[str] = []
    recommendation = build_global_fit_wizard_screening_recommendation(
        datasets,
        current_model=model,
        single_fit_recommendations_by_run=completed.recommendations_by_run,
        portfolio=completed.portfolio,
        progress_callback=progress_messages.append,
    )

    assessment = recommendation.assessment_for_key("exp_constant")
    assert assessment is not None
    assert assessment.fit_results_by_run[failed_run].success is True
    assert set(assessment.fit_results_by_run) == {int(dataset.run_number) for dataset in datasets}
    assert np.isfinite(assessment.aic)
    assert not any("repairing partial" in message for message in progress_messages)


def test_single_run_prefit_parameter_sets_reuses_cache() -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    template = CandidateTemplate(
        key="exp_constant",
        title="Exponential + Constant",
        category="General",
        rationale="test",
        model=model,
    )
    datasets = [
        _dataset_for(
            run_number=610 + idx,
            field=50.0 * idx,
            temperature=5.0,
            model=model,
            params={"A_1": 0.2, "Lambda": 0.35 + (0.05 * idx), "A_bg": 0.01},
        )
        for idx in range(2)
    ]
    base_by_run = {
        int(dataset.run_number): ParameterSet(
            [
                Parameter("A_1", 0.2, min=0.0, max=1.0),
                Parameter("Lambda", 0.4, min=0.0, max=5.0),
                Parameter("A_bg", 0.01, min=-0.5, max=0.5),
            ]
        )
        for dataset in datasets
    }
    fit_engine = FitEngine()
    fit_calls: list[tuple[int, str]] = []

    def _fake_fit(dataset, _function, params, *, method="migrad"):
        fit_calls.append((int(dataset.run_number), method))
        cloned = ParameterSet(
            [
                Parameter(parameter.name, parameter.value, min=parameter.min, max=parameter.max)
                for parameter in params
            ]
        )
        return FitResult(
            success=True,
            chi_squared=1.0,
            reduced_chi_squared=0.1,
            parameters=cloned,
            message="ok",
        )

    fit_engine.fit = _fake_fit  # type: ignore[method-assign]
    cache: dict[object, dict[int, ParameterSet]] = {}

    first = _single_run_prefit_parameter_sets(
        datasets,
        template,
        fit_engine=fit_engine,
        base_by_run=base_by_run,
        fixed_param_names=(),
        cache=cache,
    )
    calls_after_first = len(fit_calls)
    second = _single_run_prefit_parameter_sets(
        datasets,
        template,
        fit_engine=fit_engine,
        base_by_run=base_by_run,
        fixed_param_names=(),
        cache=cache,
    )

    assert calls_after_first > 0
    assert len(fit_calls) == calls_after_first
    assert sorted(first) == sorted(second)


def test_warm_start_parameter_sets_reuses_cache() -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    template = CandidateTemplate(
        key="exp_constant",
        title="Exponential + Constant",
        category="General",
        rationale="test",
        model=model,
    )
    datasets = [
        _dataset_for(
            run_number=710 + idx,
            field=25.0 * idx,
            temperature=5.0,
            model=model,
            params={"A_1": 0.2, "Lambda": 0.2 + (0.1 * idx), "A_bg": 0.01},
        )
        for idx in range(2)
    ]
    base_by_run = {
        int(dataset.run_number): ParameterSet(
            [
                Parameter("A_1", 0.2, min=0.0, max=1.0),
                Parameter("Lambda", 0.4, min=0.0, max=5.0),
                Parameter("A_bg", 0.01, min=-0.5, max=0.5),
            ]
        )
        for dataset in datasets
    }
    fit_results = {
        int(dataset.run_number): FitResult(
            success=True,
            chi_squared=1.0,
            reduced_chi_squared=0.1,
            parameters=ParameterSet(
                [
                    Parameter("A_1", 0.2, min=0.0, max=1.0),
                    Parameter("Lambda", 0.25 + (0.1 * idx), min=0.0, max=5.0),
                    Parameter("A_bg", 0.01, min=-0.5, max=0.5),
                ]
            ),
            message="ok",
        )
        for idx, dataset in enumerate(datasets)
    }
    assessment = GlobalCandidateAssessment(
        template=template,
        fit_results_by_run=fit_results,
        global_parameters=fit_results[int(datasets[0].run_number)].parameters,
        global_param_names=("A_1", "Lambda", "A_bg"),
        local_param_names=(),
        fixed_param_names=(),
        parameter_recommendations=(),
        run_diagnostics=(),
        series_warnings=(),
        aic=10.0,
        aicc=10.0,
        bic=10.0,
        selected_score=10.0,
        fitted_curves_by_run={},
        component_curves_by_run={},
    )
    fit_engine = FitEngine()
    fit_calls: list[int] = []

    def _fake_fit(dataset, _function, params, *, method="migrad"):
        fit_calls.append(int(dataset.run_number))
        cloned = ParameterSet(
            [
                Parameter(parameter.name, parameter.value, min=parameter.min, max=parameter.max)
                for parameter in params
            ]
        )
        return FitResult(
            success=True,
            chi_squared=1.0,
            reduced_chi_squared=0.1,
            parameters=cloned,
            message="ok",
        )

    fit_engine.fit = _fake_fit  # type: ignore[method-assign]
    cache: dict[object, dict[int, ParameterSet]] = {}

    first = _warm_start_parameter_sets(
        datasets,
        assessment=assessment,
        base_by_run=base_by_run,
        target_global_names=("A_1", "A_bg"),
        target_local_names=("Lambda",),
        fit_engine=fit_engine,
        template=template,
        cache=cache,
    )
    calls_after_first = len(fit_calls)
    second = _warm_start_parameter_sets(
        datasets,
        assessment=assessment,
        base_by_run=base_by_run,
        target_global_names=("A_1", "A_bg"),
        target_local_names=("Lambda",),
        fit_engine=fit_engine,
        template=template,
        cache=cache,
    )

    assert calls_after_first == len(datasets)
    assert len(fit_calls) == calls_after_first
    assert sorted(first) == sorted(second)


def test_cache_derived_role_recommendations_reuse_wavefront_assignments() -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    datasets = [
        _dataset_for(
            run_number=760 + idx,
            field=100.0 * idx,
            temperature=10.0,
            model=model,
            params={"A_1": 0.2, "Lambda": 0.15 + (0.2 * idx), "A_bg": 0.01},
        )
        for idx in range(1, 5)
    ]
    template = CandidateTemplate(
        key="exp_constant",
        title="Exponential + Constant",
        category="General",
        rationale="test",
        model=model,
    )
    diagnostics = tuple(
        RunResidualDiagnostic(
            run_number=int(dataset.run_number),
            run_label=dataset.run_label,
            axis_value=float(dataset.metadata["field"]),
            residual_rms=0.1,
            runs_z_score=0.0,
            max_abs_autocorrelation=0.0,
            residual_fft_peak_snr=0.0,
            gate_passed=True,
            gate_reasons=(),
        )
        for dataset in datasets
    )

    def _make_assessment(*, local: tuple[str, ...], score: float) -> GlobalCandidateAssessment:
        fit_results = {
            int(dataset.run_number): FitResult(
                success=True,
                chi_squared=score / len(datasets),
                reduced_chi_squared=0.1,
                parameters=ParameterSet(
                    [
                        Parameter("A_1", value=0.2, min=0.0, max=1.0),
                        Parameter(
                            "Lambda",
                            value=0.15 + 0.2 * idx if "Lambda" in local else 0.45,
                            min=0.0,
                            max=5.0,
                        ),
                        Parameter("A_bg", value=0.01, min=-0.2, max=0.2),
                    ]
                ),
                message="ok",
            )
            for idx, dataset in enumerate(datasets, start=1)
        }
        return GlobalCandidateAssessment(
            template=template,
            fit_results_by_run=fit_results,
            global_parameters=next(iter(fit_results.values())).parameters,
            global_param_names=tuple(
                name for name in template.model.param_names if name not in set(local)
            ),
            local_param_names=local,
            fixed_param_names=(),
            parameter_recommendations=(),
            run_diagnostics=diagnostics,
            series_warnings=(),
            aic=score,
            aicc=score,
            bic=score,
            selected_score=score,
            fitted_curves_by_run={},
            component_curves_by_run={},
        )

    local_assessment = _make_assessment(local=("Lambda",), score=10.0)
    shared_assessment = _make_assessment(local=(), score=16.0)
    recommendations = _build_parameter_recommendations_from_exact_cache(
        datasets,
        local_assessment,
        template=template,
        fixed_param_names=(),
        metric=SelectionMetric.AICC,
        cache={
            (
                local_assessment.global_param_names,
                local_assessment.local_param_names,
            ): local_assessment,
            (
                shared_assessment.global_param_names,
                shared_assessment.local_param_names,
            ): shared_assessment,
        },
        names_to_test={"Lambda"},
    )

    by_name = {recommendation.name: recommendation for recommendation in recommendations}
    assert by_name["Lambda"].recommended_role == "Local"
    assert by_name["Lambda"].local_score == pytest.approx(10.0)
    assert by_name["Lambda"].global_score == pytest.approx(16.0)
    assert "penalized score" in by_name["Lambda"].rationale


def test_global_fit_wizard_threads_selected_template_keys(monkeypatch) -> None:
    captured: list[tuple[str, ...] | None] = []
    sentinel = object()

    def _fake_staged(*args, **kwargs):
        captured.append(kwargs.get("selected_template_keys"))
        return sentinel

    monkeypatch.setattr(
        global_fit_wizard_module,
        "_build_global_fit_wizard_recommendation_staged",
        _fake_staged,
    )

    assert build_global_fit_wizard_recommendation([]) is sentinel
    assert (
        build_global_fit_wizard_recommendation([], selected_template_keys=("exp_constant",))
        is sentinel
    )
    assert captured == [None, ("exp_constant",)]


def test_global_fit_wizard_warns_for_mixed_field_temperature_grid() -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    datasets = [
        _dataset_for(
            run_number=300 + idx,
            field=50.0 * idx,
            temperature=5.0 * idx,
            model=model,
            params={"A_1": 0.2, "Lambda": 0.35, "A_bg": 0.01},
        )
        for idx in range(1, 5)
    ]

    recommendation = build_global_fit_wizard_recommendation(datasets)

    assert recommendation.mixed_axes_warning is not None
    assert recommendation.recommended_key is None


def test_global_fit_wizard_flags_abrupt_regime_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exp_model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    _restrict_to_exp_constant_template(monkeypatch, exp_model)
    osc_model = CompositeModel(["Oscillatory", "Exponential", "Constant"], operators=["*", "+"])
    datasets = [
        _dataset_for(
            run_number=400,
            field=0.0,
            temperature=2.0,
            model=exp_model,
            params={"A_1": 0.2, "Lambda": 0.25, "A_bg": 0.01},
        ),
        _dataset_for(
            run_number=401,
            field=100.0,
            temperature=2.0,
            model=exp_model,
            params={"A_1": 0.2, "Lambda": 0.3, "A_bg": 0.01},
        ),
        _dataset_for(
            run_number=402,
            field=200.0,
            temperature=2.0,
            model=osc_model,
            params={"A_1": 0.18, "frequency": 0.6, "phase": 0.0, "Lambda": 0.2, "A_bg": 0.01},
        ),
        _dataset_for(
            run_number=403,
            field=300.0,
            temperature=2.0,
            model=osc_model,
            params={"A_1": 0.18, "frequency": 0.7, "phase": 0.0, "Lambda": 0.2, "A_bg": 0.01},
        ),
    ]

    recommendation = build_global_fit_wizard_recommendation(datasets)

    assert any(
        "Fingerprint features change abruptly" in warning
        for assessment in recommendation.assessments
        for warning in assessment.series_warnings
    )


def test_global_fit_wizard_reranks_existing_assessments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    _restrict_to_exp_constant_template(monkeypatch, model)
    datasets = [
        _dataset_for(
            run_number=500 + idx,
            field=25.0 * idx,
            temperature=8.0,
            model=model,
            params={"A_1": 0.2, "Lambda": 0.2 + 0.05 * idx, "A_bg": 0.01},
        )
        for idx in range(1, 5)
    ]

    recommendation = build_global_fit_wizard_recommendation(datasets, metric=SelectionMetric.AICC)
    reranked = rerank_global_fit_wizard_recommendation(recommendation, SelectionMetric.BIC)

    assert reranked.metric == SelectionMetric.BIC
    assert reranked.assessments == recommendation.assessments


def test_global_fit_wizard_symbols_are_exported() -> None:
    assert fitting_api.build_global_fit_wizard_recommendation is not None
    assert fitting_api.rerank_global_fit_wizard_recommendation is not None
    assert fitting_api.GlobalFitWizardRecommendation is not None
    assert fitting_api.GlobalCandidateAssessment is not None
    assert fitting_api.GlobalParameterRecommendation is not None


def test_deserialize_global_candidate_assessment_migrates_legacy_fraction_params() -> None:
    """A global-wizard assessment cached before the fraction rework migrates on load.

    The per-run fitted ``fraction_<k>`` params, the standalone ``global_parameters``
    set, and the ``global_param_names``/``local_param_names``/``fixed_param_names``
    role tuples all carry the legacy positional names in an old cache entry; all
    four are renamed to the n-1 free scheme against the template's model, mirroring
    fit_wizard._deserialize_candidate_assessment's single-fit treatment.
    """
    model = CompositeModel.from_expression("( Exponential + Gaussian + Constant ){frac}")
    template_payload = {
        "key": "cand",
        "title": "",
        "category": "",
        "rationale": "",
        "model": model.to_dict(),
    }
    fit_result_payload = {
        "success": True,
        "chi_squared": 1.0,
        "reduced_chi_squared": 1.0,
        "parameters": [
            {"name": "A_1", "value": 20.0, "min": 0.0, "max": 1e9, "fixed": False},
            {"name": "Lambda", "value": 0.5, "min": 0.0, "max": 1e9, "fixed": False},
            {"name": "fraction_1", "value": 2.0, "min": 0.0, "max": 1.0, "fixed": False},
            {"name": "sigma", "value": 0.3, "min": 0.0, "max": 1e9, "fixed": False},
            {"name": "fraction_2", "value": 1.0, "min": 0.0, "max": 1.0, "fixed": False},
            {"name": "fraction_3", "value": 1.0, "min": 0.0, "max": 1.0, "fixed": False},
        ],
        "uncertainties": {"fraction_1": 0.01, "fraction_2": 0.02, "fraction_3": 0.03},
    }
    payload = {
        "template": template_payload,
        "fit_results_by_run": {"1001": fit_result_payload},
        "global_parameters": [
            {"name": "A_1", "value": 20.0, "min": 0.0, "max": 1e9, "fixed": False},
            {"name": "fraction_1", "value": 2.0, "min": 0.0, "max": 1.0, "fixed": False},
            {"name": "fraction_2", "value": 1.0, "min": 0.0, "max": 1.0, "fixed": False},
            {"name": "fraction_3", "value": 1.0, "min": 0.0, "max": 1.0, "fixed": False},
        ],
        "global_param_names": ["A_1", "fraction_1", "fraction_2", "fraction_3"],
        "local_param_names": ["Lambda"],
        "fixed_param_names": [],
        "aic": 1.0,
        "bic": 1.0,
        "selected_score": 1.0,
    }

    assessment = _deserialize_global_candidate_assessment(payload)

    assert assessment is not None

    # Per-run fit result parameters migrated.
    run_result = assessment.fit_results_by_run[1001]
    run_names = {parameter.name for parameter in run_result.parameters}
    assert "f_Exponential" in run_names
    assert "f_Gaussian" in run_names
    assert not any(name.startswith("fraction_") for name in run_names)
    assert "f_Exponential" in run_result.uncertainties
    assert "f_Gaussian" in run_result.uncertainties
    assert not any(key.startswith("fraction_") for key in run_result.uncertainties)

    # Standalone global_parameters ParameterSet migrated.
    global_names = set(assessment.global_parameters.names)
    assert "f_Exponential" in global_names
    assert "f_Gaussian" in global_names
    assert not any(name.startswith("fraction_") for name in global_names)

    # Parameter-role tuples migrated (legacy names renamed/dropped in place).
    assert "f_Exponential" in assessment.global_param_names
    assert "f_Gaussian" in assessment.global_param_names
    assert not any(name.startswith("fraction_") for name in assessment.global_param_names)
    assert not any(name.startswith("fraction_") for name in assessment.local_param_names)
    assert not any(name.startswith("fraction_") for name in assessment.fixed_param_names)


def test_localisation_penalty_prefers_rate_parameters_over_amplitudes() -> None:
    assert _localisation_penalty(("Lambda",)) < _localisation_penalty(("A_1",))
    assert _localisation_penalty(("A_1",)) < _localisation_penalty(("A_bg",))


def test_component_canonicalization_orders_biexponential_components_by_rate() -> None:
    template = CandidateTemplate(
        key="biexp_constant",
        title="Exponential + Exponential + Constant",
        category="General",
        rationale="test",
        model=CompositeModel(["Exponential", "Exponential", "Constant"], operators=["+", "+"]),
    )
    params = ParameterSet(
        [
            Parameter("A_1", value=0.1, min=0.0, max=1.0),
            Parameter("Lambda_1", value=0.2, min=0.0, max=10.0),
            Parameter("A_2", value=0.3, min=0.0, max=1.0),
            Parameter("Lambda_2", value=1.4, min=0.0, max=10.0),
            Parameter("A_bg", value=0.01, min=-0.2, max=0.2),
        ]
    )

    canonical = _canonicalize_parameter_sets(
        {1: params},
        template=template,
        global_param_names=("A_1", "Lambda_1", "A_2", "Lambda_2", "A_bg"),
        local_param_names=(),
        fixed_param_names=(),
    )

    assert canonical[1]["Lambda_1"].value == 1.4
    assert canonical[1]["Lambda_2"].value == 0.2
    assert canonical[1]["A_1"].value == 0.3
    assert canonical[1]["A_2"].value == 0.1


def test_globalization_candidate_order_prefers_stable_amplitudes_over_rates() -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    template = CandidateTemplate(
        key="exp_constant",
        title="Exponential + Constant",
        category="General",
        rationale="test",
        model=model,
    )
    datasets = [
        _dataset_for(
            run_number=720 + idx,
            field=25.0 * idx,
            temperature=5.0,
            model=model,
            params={"A_1": 0.2, "Lambda": 0.2 + (0.2 * idx), "A_bg": 0.01},
        )
        for idx in range(3)
    ]
    fit_results = {
        int(dataset.run_number): FitResult(
            success=True,
            chi_squared=1.0,
            reduced_chi_squared=0.1,
            parameters=ParameterSet(
                [
                    Parameter("A_1", value=0.2 + (0.005 * idx), min=0.0, max=1.0),
                    Parameter("Lambda", value=0.15 + (0.35 * idx), min=0.0, max=5.0),
                    Parameter("A_bg", value=0.01, min=-0.2, max=0.2),
                ]
            ),
            message="ok",
        )
        for idx, dataset in enumerate(datasets)
    }
    diagnostics = tuple(
        RunResidualDiagnostic(
            run_number=int(dataset.run_number),
            run_label=dataset.run_label,
            axis_value=float(dataset.metadata["field"]),
            residual_rms=0.05,
            runs_z_score=0.0,
            max_abs_autocorrelation=0.0,
            residual_fft_peak_snr=0.0,
            gate_passed=True,
            gate_reasons=(),
        )
        for dataset in datasets
    )
    assessment = GlobalCandidateAssessment(
        template=template,
        fit_results_by_run=fit_results,
        global_parameters=fit_results[int(datasets[0].run_number)].parameters,
        global_param_names=(),
        local_param_names=("A_1", "Lambda", "A_bg"),
        fixed_param_names=(),
        parameter_recommendations=(),
        run_diagnostics=diagnostics,
        series_warnings=(),
        aic=10.0,
        aicc=10.0,
        bic=10.0,
        selected_score=10.0,
        fitted_curves_by_run={},
        component_curves_by_run={},
    )

    ordered = _globalization_candidate_order(
        datasets,
        assessment,
        remaining=assessment.local_param_names,
    )

    assert ordered.index("A_1") < ordered.index("Lambda")
    assert ordered.index("A_bg") < ordered.index("Lambda")


def test_staged_assignment_seed_completes_when_local_only_pinned_on_one_run() -> None:
    """A stage must not hand Minuit a run with zero free parameters.

    Regression for the docs-build crash traced to 45e5110 (PR #301, "pin B_L
    from run metadata"): on a longitudinal-field Kubo-Toyabe series, B_L
    legitimately gets a Local role (it varies run to run) even though it is
    *not* uniformly free -- a ZF run pins B_L at 0 (measured, not fitted)
    while an LF run fits it. The local-only refinement half of the staged
    seed holds every Global parameter fixed for *every* run while unlocking
    only the Local names; for the ZF run that left B_L (already pinned) as
    the sole "active" name fixed too, so every parameter of that run's
    ``ParameterSet`` came out fixed and Minuit raised "starting value(s) are
    required". The fix (``FitEngine._fit_core``) treats a fully-fixed
    parameter set as a valid degenerate fit rather than an error.
    """
    model = CompositeModel(["LongitudinalFieldKT", "Constant"], operators=["+"])
    template = CandidateTemplate(
        key="lf_kt_constant",
        title="Longitudinal-field KT + Constant",
        category="KT-like",
        rationale="test",
        model=model,
    )
    zf_dataset = _dataset_for(
        run_number=901,
        field=0.0,
        temperature=20.0,
        model=model,
        params={"A_1": 24.0, "Delta": 0.4, "B_L": 0.0, "A_bg": 2.0},
    )
    lf_dataset = _dataset_for(
        run_number=902,
        field=50.0,
        temperature=20.0,
        model=model,
        params={"A_1": 24.0, "Delta": 0.4, "B_L": 50.0, "A_bg": 2.0},
    )

    def _params(b_l_value: float, *, b_l_fixed: bool) -> ParameterSet:
        return ParameterSet(
            [
                Parameter("A_1", value=24.0, min=0.0, max=100.0),
                Parameter("Delta", value=0.4, min=0.0, max=4.0),
                Parameter("B_L", value=b_l_value, min=0.0, max=200.0, fixed=b_l_fixed),
                Parameter("A_bg", value=2.0, min=-10.0, max=10.0),
            ]
        )

    initial_params = {
        # ZF run: B_L pinned from run metadata, per _pinned_longitudinal_field.
        901: _params(0.0, b_l_fixed=True),
        # LF run: B_L unrecorded/free, seeded away from the true value.
        902: _params(40.0, b_l_fixed=False),
    }

    result = _staged_assignment_seed(
        [zf_dataset, lf_dataset],
        template,
        fit_engine=FitEngine(),
        global_param_names=("A_1", "Delta", "A_bg"),
        local_param_names=("B_L",),
        initial_params=initial_params,
    )

    assert result[901]["B_L"].fixed
    assert result[901]["B_L"].value == 0.0
    assert not result[902]["B_L"].fixed


def test_staged_globalization_assignment_keeps_varying_lambda_local(
    monkeypatch,
) -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    template = CandidateTemplate(
        key="exp_constant",
        title="Exponential + Constant",
        category="General",
        rationale="test",
        model=model,
    )
    lambda_values = {801: 0.15, 802: 0.30, 803: 0.60, 804: 0.90}
    datasets = [
        _dataset_for(
            run_number=run_number,
            field=40.0 * index,
            temperature=4.0,
            model=model,
            params={"A_1": 0.2, "Lambda": lambda_value, "A_bg": 0.01},
        )
        for index, (run_number, lambda_value) in enumerate(sorted(lambda_values.items()), start=1)
    ]
    base_by_run = {
        run_number: ParameterSet(
            [
                Parameter("A_1", value=0.28, min=0.0, max=1.0),
                Parameter("Lambda", value=0.45, min=0.0, max=2.0),
                Parameter("A_bg", value=0.04, min=-0.2, max=0.2),
            ]
        )
        for run_number in lambda_values
    }

    trace_values = {
        "A_1": {801: 0.20, 802: 0.205, 803: 0.195, 804: 0.20},
        "Lambda": {801: 0.15, 802: 0.30, 803: 0.60, 804: 0.90},
        "A_bg": {801: 0.01, 802: 0.01, 803: 0.01, 804: 0.01},
    }
    score_by_local_names = {
        ("A_1", "A_bg", "Lambda"): 10.0,
        ("A_bg", "Lambda"): 8.0,
        ("A_1", "Lambda"): 9.0,
        ("A_1", "A_bg"): 12.0,
        ("Lambda",): 7.5,
        ("A_bg",): 11.0,
        ("A_1",): 11.0,
        (): 13.0,
    }

    def _make_assessment(local_param_names: tuple[str, ...]) -> GlobalCandidateAssessment:
        local_set = set(local_param_names)
        global_param_names = tuple(
            name for name in template.model.param_names if name not in local_set
        )
        fit_results = {}
        for run_number in lambda_values:
            params = []
            for name in template.model.param_names:
                values = trace_values[name]
                if name in local_set:
                    value = values[run_number]
                else:
                    value = float(np.mean(list(values.values())))
                bounds = (0.0, 1.0)
                if name == "Lambda":
                    bounds = (0.0, 2.0)
                elif name == "A_bg":
                    bounds = (-0.2, 0.2)
                params.append(Parameter(name, value=value, min=bounds[0], max=bounds[1]))
            fit_results[run_number] = FitResult(
                success=True,
                chi_squared=score_by_local_names[tuple(sorted(local_param_names))],
                reduced_chi_squared=0.1,
                parameters=ParameterSet(params),
                message="ok",
            )
        diagnostics = tuple(
            RunResidualDiagnostic(
                run_number=run_number,
                run_label=str(run_number),
                axis_value=float(run_number),
                residual_rms=0.05,
                runs_z_score=0.0,
                max_abs_autocorrelation=0.0,
                residual_fft_peak_snr=0.0,
                gate_passed=True,
                gate_reasons=(),
            )
            for run_number in lambda_values
        )
        score = score_by_local_names[tuple(sorted(local_param_names))]
        return GlobalCandidateAssessment(
            template=template,
            fit_results_by_run=fit_results,
            global_parameters=fit_results[801].parameters,
            global_param_names=global_param_names,
            local_param_names=tuple(sorted(local_param_names)),
            fixed_param_names=(),
            parameter_recommendations=(),
            run_diagnostics=diagnostics,
            series_warnings=(),
            aic=score,
            aicc=score,
            bic=score,
            selected_score=score,
            fitted_curves_by_run={},
            component_curves_by_run={},
        )

    def _fake_fit_exact_assignment(
        datasets,
        template,
        *,
        fit_engine,
        base_by_run,
        global_param_names,
        local_param_names,
        fixed_param_names,
        axis_key,
        metric,
        cache,
        warm_start_by_run=None,
        progress_callback=None,
        search_strategy="legacy",
        instrumentation=None,
        initial_step_sizes=None,
    ):
        del (
            fit_engine,
            base_by_run,
            fixed_param_names,
            axis_key,
            metric,
            cache,
            warm_start_by_run,
            progress_callback,
            search_strategy,
            instrumentation,
            initial_step_sizes,
            global_param_names,
        )
        return _make_assessment(tuple(sorted(local_param_names)))

    monkeypatch.setattr(
        global_fit_wizard_module,
        "_fit_exact_assignment",
        _fake_fit_exact_assignment,
    )
    monkeypatch.setattr(
        global_fit_wizard_module,
        "_warm_start_parameter_sets",
        lambda *args, **kwargs: base_by_run,
    )
    monkeypatch.setattr(
        global_fit_wizard_module,
        "_step_hints_from_assessment",
        lambda *args, **kwargs: {},
    )

    assessment = _staged_globalization_assignment(
        datasets,
        template,
        fit_engine=FitEngine(),
        base_by_run=base_by_run,
        fixed_param_names=(),
        axis_key="field",
        metric=SelectionMetric.AICC,
        cache={},
        warm_start_cache={},
    )

    assert assessment is not None
    assert assessment.is_successful
    assert "Lambda" in assessment.local_param_names
    assert "A_1" in assessment.global_param_names
    assert "A_bg" in assessment.global_param_names


def test_staged_globalization_assignment_attempts_high_dimension_all_local_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    template = CandidateTemplate(
        key="exp_constant",
        title="Exponential + Constant",
        category="General",
        rationale="test",
        model=model,
    )
    datasets = [
        _dataset_for(
            run_number=900 + idx,
            field=10.0 * idx,
            temperature=5.0,
            model=model,
            params={"A_1": 0.2, "Lambda": 0.3, "A_bg": 0.01},
        )
        for idx in range(1, 4)
    ]
    base_by_run = {
        int(dataset.run_number): ParameterSet(
            [
                Parameter("A_1", value=0.2, min=0.0, max=1.0),
                Parameter("Lambda", value=0.3, min=0.0, max=2.0),
                Parameter("A_bg", value=0.01, min=-0.2, max=0.2),
            ]
        )
        for dataset in datasets
    }
    diagnostics = tuple(
        RunResidualDiagnostic(
            run_number=int(dataset.run_number),
            run_label=dataset.run_label,
            axis_value=float(dataset.metadata["field"]),
            residual_rms=0.05,
            runs_z_score=0.0,
            max_abs_autocorrelation=0.0,
            residual_fft_peak_snr=0.0,
            gate_passed=True,
            gate_reasons=(),
        )
        for dataset in datasets
    )
    baseline_calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def _fake_fit_exact_assignment(
        datasets,
        template,
        *,
        fit_engine,
        base_by_run,
        global_param_names,
        local_param_names,
        fixed_param_names,
        axis_key,
        metric,
        cache,
        warm_start_by_run=None,
        progress_callback=None,
        search_strategy="legacy",
        instrumentation=None,
        initial_step_sizes=None,
    ):
        del (
            fit_engine,
            base_by_run,
            fixed_param_names,
            axis_key,
            metric,
            cache,
            warm_start_by_run,
            progress_callback,
            search_strategy,
            instrumentation,
            initial_step_sizes,
        )
        baseline_calls.append((tuple(global_param_names), tuple(local_param_names)))
        fit_results = {
            int(dataset.run_number): FitResult(
                success=True,
                chi_squared=1.0,
                reduced_chi_squared=0.1,
                parameters=ParameterSet(
                    [
                        Parameter("A_1", value=0.2, min=0.0, max=1.0),
                        Parameter("Lambda", value=0.3, min=0.0, max=2.0),
                        Parameter("A_bg", value=0.01, min=-0.2, max=0.2),
                    ]
                ),
                message="ok",
            )
            for dataset in datasets
        }
        return GlobalCandidateAssessment(
            template=template,
            fit_results_by_run=fit_results,
            global_parameters=next(iter(fit_results.values())).parameters,
            global_param_names=tuple(global_param_names),
            local_param_names=tuple(local_param_names),
            fixed_param_names=(),
            parameter_recommendations=(),
            run_diagnostics=diagnostics,
            series_warnings=(),
            aic=10.0,
            aicc=10.0,
            bic=10.0,
            selected_score=10.0,
            fitted_curves_by_run={},
            component_curves_by_run={},
        )

    monkeypatch.setattr(
        global_fit_wizard_module,
        "_fit_exact_assignment",
        _fake_fit_exact_assignment,
    )
    monkeypatch.setattr(
        global_fit_wizard_module,
        "_free_parameter_count",
        lambda *args, **kwargs: 133,
    )
    monkeypatch.setattr(
        global_fit_wizard_module,
        "_globalization_candidate_order",
        lambda *args, **kwargs: (),
    )

    assessment = _staged_globalization_assignment(
        datasets,
        template,
        fit_engine=FitEngine(),
        base_by_run=base_by_run,
        fixed_param_names=(),
        axis_key="field",
        metric=SelectionMetric.AICC,
        cache={},
        warm_start_cache={},
    )

    assert assessment is not None
    assert assessment.is_successful
    assert baseline_calls == [((), ("A_1", "Lambda", "A_bg"))]


def test_single_run_prefits_improve_staged_seed_parameters() -> None:
    template = CandidateTemplate(
        key="exp_constant",
        title="Exponential + Constant",
        category="General",
        rationale="test",
        model=CompositeModel(["Exponential", "Constant"], operators=["+"]),
    )
    lambdas = {701: 0.15, 702: 0.30, 703: 0.55, 704: 0.85}
    datasets = [
        _dataset_for(
            run_number=run_number,
            field=25.0 * index,
            temperature=6.0,
            model=template.model,
            params={"A_1": 0.2, "Lambda": lambda_value, "A_bg": 0.01},
        )
        for index, (run_number, lambda_value) in enumerate(sorted(lambdas.items()), start=1)
    ]
    base_by_run = {
        run_number: ParameterSet(
            [
                Parameter("A_1", value=0.5, min=0.0, max=1.0),
                Parameter("Lambda", value=1.5, min=0.0, max=2.0),
                Parameter("A_bg", value=0.1, min=-0.2, max=0.2),
            ]
        )
        for run_number in lambdas
    }

    prefitted = _single_run_prefit_parameter_sets(
        datasets,
        template,
        fit_engine=FitEngine(),
        base_by_run=base_by_run,
        fixed_param_names=(),
    )

    initial_error = sum(
        abs(base_by_run[run_number]["Lambda"].value - truth)
        for run_number, truth in lambdas.items()
    )
    fitted_error = sum(
        abs(prefitted[run_number]["Lambda"].value - truth) for run_number, truth in lambdas.items()
    )

    assert fitted_error < initial_error


def test_oscillatory_rescue_requires_repeated_aligned_evidence() -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    datasets = [
        _dataset_for(
            run_number=600 + idx,
            field=50.0 * idx,
            temperature=5.0,
            model=model,
            params={"A_1": 0.2, "Lambda": 0.3, "A_bg": 0.01},
        )
        for idx in range(6)
    ]
    diagnostics = [
        RunResidualDiagnostic(
            run_number=int(dataset.run_number),
            run_label=dataset.run_label,
            axis_value=float(dataset.metadata["field"]),
            residual_rms=0.3,
            runs_z_score=-3.1 if idx < 3 else 0.2,
            max_abs_autocorrelation=0.2,
            residual_fft_peak_snr=7.4 if idx < 3 else 1.2,
            gate_passed=idx >= 3,
            gate_reasons=(),
        )
        for idx, dataset in enumerate(datasets)
    ]
    fingerprints = {
        int(dataset.run_number): (
            _fingerprint(dominant_fft_snr=5.2, cycles=2.0, turning_points=2)
            if idx < 3
            else _fingerprint()
        )
        for idx, dataset in enumerate(datasets)
    }

    supported_runs = _supported_oscillatory_run_numbers(
        datasets,
        assessment=_assessment_with_diagnostics(datasets, diagnostics),
        fingerprints_by_run=fingerprints,
    )

    assert supported_runs == tuple(int(dataset.run_number) for dataset in datasets[:3])


def test_oscillatory_rescue_ignores_isolated_fft_peaks() -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    datasets = [
        _dataset_for(
            run_number=700 + idx,
            field=50.0 * idx,
            temperature=5.0,
            model=model,
            params={"A_1": 0.2, "Lambda": 0.3, "A_bg": 0.01},
        )
        for idx in range(6)
    ]
    diagnostics = [
        RunResidualDiagnostic(
            run_number=int(dataset.run_number),
            run_label=dataset.run_label,
            axis_value=float(dataset.metadata["field"]),
            residual_rms=0.3,
            runs_z_score=-3.3 if idx == 2 else 0.1,
            max_abs_autocorrelation=0.2,
            residual_fft_peak_snr=7.8 if idx == 2 else 1.0,
            gate_passed=idx != 2,
            gate_reasons=(),
        )
        for idx, dataset in enumerate(datasets)
    ]
    fingerprints = {
        int(dataset.run_number): (
            _fingerprint(dominant_fft_snr=5.4, cycles=2.1, turning_points=2)
            if idx == 2
            else _fingerprint()
        )
        for idx, dataset in enumerate(datasets)
    }

    supported_runs = _supported_oscillatory_run_numbers(
        datasets,
        assessment=_assessment_with_diagnostics(datasets, diagnostics),
        fingerprints_by_run=fingerprints,
    )

    assert supported_runs == ()


def test_staged_multi_local_assignment_succeeds_for_large_biexp_gaussian_series() -> None:
    model = CompositeModel(
        ["Exponential", "Exponential", "Gaussian", "Constant"],
        operators=["+", "+", "+"],
    )
    template = CandidateTemplate(
        key="double_exp_gaussian_constant",
        title="Exponential + Exponential + Gaussian + Constant",
        category="General",
        rationale="test",
        model=model,
    )
    datasets = []
    for index in range(4):
        run_number = 8000 + index
        datasets.append(
            _dense_dataset_for(
                run_number=run_number,
                field=10.0 * index,
                temperature=5.0,
                model=model,
                params={
                    "A_1": 0.11,
                    "Lambda_1": 0.10 + 0.012 * index,
                    "A_2": 0.06,
                    "Lambda_2": 0.58 + 0.035 * index,
                    "A_3": 0.05,
                    "sigma": 0.18 + 0.022 * index,
                    "A_bg": 0.01,
                },
            )
        )

    base_by_run = {
        int(dataset.run_number): ParameterSet(
            [
                Parameter("A_1", value=0.14, min=0.0, max=1.0),
                Parameter("Lambda_1", value=0.18, min=0.0, max=2.0),
                Parameter("A_2", value=0.04, min=0.0, max=1.0),
                Parameter("Lambda_2", value=0.95, min=0.0, max=2.0),
                Parameter("A_3", value=0.04, min=0.0, max=1.0),
                Parameter("sigma", value=0.38, min=0.0, max=2.0),
                Parameter("A_bg", value=0.015, min=-0.2, max=0.2),
            ]
        )
        for dataset in datasets
    }
    fit_engine = FitEngine()
    baseline = _fit_exact_assignment(
        datasets,
        template,
        fit_engine=fit_engine,
        base_by_run=base_by_run,
        global_param_names=tuple(model.param_names),
        local_param_names=(),
        fixed_param_names=(),
        axis_key="field",
        metric=SelectionMetric.AICC,
        cache={},
    )

    assessment, best_partial = _staged_multi_local_assignment(
        datasets,
        template,
        fit_engine=fit_engine,
        base_by_run=base_by_run,
        baseline_assessment=baseline,
        target_local_names=("Lambda_1", "Lambda_2", "sigma"),
        fixed_param_names=(),
        axis_key="field",
        metric=SelectionMetric.AICC,
        cache={(baseline.global_param_names, baseline.local_param_names): baseline},
    )

    assert baseline.is_successful
    assert best_partial.is_successful
    assert assessment is not None
    assert assessment.is_successful
    assert assessment.local_param_names == ("Lambda_1", "Lambda_2", "sigma")


def test_staged_multi_local_assignment_succeeds_for_large_exp_double_gaussian_series() -> None:
    model = CompositeModel(
        ["Exponential", "Gaussian", "Gaussian", "Constant"],
        operators=["+", "+", "+"],
    )
    template = CandidateTemplate(
        key="exp_double_gaussian_constant",
        title="Exponential + Gaussian + Gaussian + Constant",
        category="General",
        rationale="test",
        model=model,
    )
    datasets = []
    for index in range(4):
        run_number = 8100 + index
        datasets.append(
            _dense_dataset_for(
                run_number=run_number,
                field=12.5 * index,
                temperature=5.0,
                model=model,
                params={
                    "A_1": 0.10,
                    "Lambda": 0.08 + 0.010 * index,
                    "A_2": 0.05,
                    "sigma_2": 0.15 + 0.018 * index,
                    "A_3": 0.05,
                    "sigma_3": 0.42 + 0.028 * index,
                    "A_bg": 0.012,
                },
            )
        )

    base_by_run = {
        int(dataset.run_number): ParameterSet(
            [
                Parameter("A_1", value=0.13, min=0.0, max=1.0),
                Parameter("Lambda", value=0.16, min=0.0, max=2.0),
                Parameter("A_2", value=0.04, min=0.0, max=1.0),
                Parameter("sigma_2", value=0.24, min=0.0, max=2.0),
                Parameter("A_3", value=0.04, min=0.0, max=1.0),
                Parameter("sigma_3", value=0.62, min=0.0, max=2.0),
                Parameter("A_bg", value=0.018, min=-0.2, max=0.2),
            ]
        )
        for dataset in datasets
    }
    fit_engine = FitEngine()
    baseline = _fit_exact_assignment(
        datasets,
        template,
        fit_engine=fit_engine,
        base_by_run=base_by_run,
        global_param_names=tuple(model.param_names),
        local_param_names=(),
        fixed_param_names=(),
        axis_key="field",
        metric=SelectionMetric.AICC,
        cache={},
    )

    assessment, best_partial = _staged_multi_local_assignment(
        datasets,
        template,
        fit_engine=fit_engine,
        base_by_run=base_by_run,
        baseline_assessment=baseline,
        target_local_names=("Lambda", "sigma_2", "sigma_3"),
        fixed_param_names=(),
        axis_key="field",
        metric=SelectionMetric.AICC,
        cache={(baseline.global_param_names, baseline.local_param_names): baseline},
    )

    assert baseline.is_successful
    assert best_partial.is_successful
    assert assessment is not None
    assert assessment.is_successful
    assert assessment.local_param_names == ("Lambda", "sigma_2", "sigma_3")


@pytest.mark.parametrize(
    "metric",
    [SelectionMetric.AIC, SelectionMetric.AICC, SelectionMetric.BIC],
)
def test_metric_penalty_matches_information_criteria(metric) -> None:
    """The layer-bound penalty (technique A/B) must equal IC - chi2 exactly.

    The bound's admissibility rests on penalty(k) being the *same* additive term
    the winning assessment's IC uses. Verify _metric_penalty reproduces
    compute_information_criteria's penalty for a range of (k, n), including the
    AICc small-sample-correction boundary (n <= k + 1 falls back to AIC).
    """
    from asymmetry.core.fitting.fit_wizard import compute_information_criteria

    for k in (1, 2, 5, 12, 40):
        for n in (3, 6, 50, 500):
            aic, aicc, bic = compute_information_criteria(0.0, k, n)
            expected = {
                SelectionMetric.AIC: aic,
                SelectionMetric.BIC: bic,
                SelectionMetric.AICC: aicc if aicc is not None else aic,
            }[metric]
            assert _metric_penalty(k, sample_count=n, metric=metric) == pytest.approx(expected)


def test_metric_penalty_is_monotone_in_layer() -> None:
    """penalty(k(m)) must be non-decreasing in the Hamming layer m.

    This monotonicity is what makes the layer bound admissible: the anchor floor
    plus penalty(m) lower-bounds every assignment at layer >= m. k(m) grows with
    m for G >= 2, so the penalty must too, for all three metrics.
    """
    free_param_count = 5
    n_datasets = 3
    sample_count = 360
    for metric in (SelectionMetric.AIC, SelectionMetric.AICC, SelectionMetric.BIC):
        penalties = [
            _metric_penalty(
                _layer_parameter_count(m, free_param_count=free_param_count, n_datasets=n_datasets),
                sample_count=sample_count,
                metric=metric,
            )
            for m in range(free_param_count + 1)
        ]
        assert penalties == sorted(penalties)
        # k(m) = (P - m) + m*G, so k(0) = P and k(P) = P*G.
        assert (
            _layer_parameter_count(0, free_param_count=free_param_count, n_datasets=n_datasets)
            == free_param_count
        )
        assert (
            _layer_parameter_count(
                free_param_count, free_param_count=free_param_count, n_datasets=n_datasets
            )
            == free_param_count * n_datasets
        )


def test_warm_certificate_failure_escalates_to_full_battery() -> None:
    """A failed monotonicity certificate (technique D) must escalate, not accept.

    Force the certificate to fail by passing an impossibly-low parent chi2
    (warm_start_chi2 = -1e9): no converged child can satisfy
    child <= parent + epsilon, so _fit_exact_assignment must fall through to the
    full multi-start battery and still return a valid, well-converged fit whose
    chi2 matches the battery-only run. This proves the certificate branch is not
    dead code and that escalation, not silent acceptance, guards a bad warm fit.
    """
    import copy

    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    template = CandidateTemplate(
        key="exp_constant",
        title="Exponential + Constant",
        category="General",
        rationale="test",
        model=model,
    )
    datasets = [
        _dense_dataset_for(
            run_number=9200 + index,
            field=25.0 * (index + 1),
            temperature=5.0,
            model=model,
            params={"A_1": 0.20, "Lambda": 0.20 + 0.25 * index, "A_bg": 0.010},
            error_level=0.004,
        )
        for index in range(3)
    ]
    base_by_run = {
        int(dataset.run_number): ParameterSet(
            [
                Parameter("A_1", value=0.18, min=0.0, max=1.0),
                Parameter("Lambda", value=0.30, min=0.0, max=3.0),
                Parameter("A_bg", value=0.012, min=-0.2, max=0.2),
            ]
        )
        for dataset in datasets
    }
    # Two locals so the fast path (len(local_param_names) >= 2) is active.
    global_names: tuple[str, ...] = ("A_bg",)
    local_names = ("A_1", "Lambda")

    warm_start_by_run = {
        int(dataset.run_number): copy.deepcopy(base_by_run[int(dataset.run_number)])
        for dataset in datasets
    }

    # Battery-only reference (no warm start at all).
    reference = _fit_exact_assignment(
        datasets,
        template,
        fit_engine=FitEngine(),
        base_by_run=base_by_run,
        global_param_names=global_names,
        local_param_names=local_names,
        fixed_param_names=(),
        axis_key="field",
        metric=SelectionMetric.AICC,
        cache={},
    )
    assert reference.is_successful

    # Impossible certificate → must escalate to the battery and still converge.
    escalated = _fit_exact_assignment(
        datasets,
        template,
        fit_engine=FitEngine(),
        base_by_run=base_by_run,
        global_param_names=global_names,
        local_param_names=local_names,
        fixed_param_names=(),
        axis_key="field",
        metric=SelectionMetric.AICC,
        cache={},
        warm_start_by_run=warm_start_by_run,
        warm_start_chi2=-1.0e9,
    )
    assert escalated.is_successful
    ref_chi2 = sum(r.chi_squared for r in reference.fit_results_by_run.values())
    esc_chi2 = sum(r.chi_squared for r in escalated.fit_results_by_run.values())
    # Escalation reached the same minimum the battery finds from cold start.
    assert esc_chi2 == pytest.approx(ref_chi2, rel=0.05)


# --------------------------------------------------------------------------- #
# Non-exhaustive search engines (PR 4, techniques E/F/G/H)
# --------------------------------------------------------------------------- #


def _uniform_series(model: CompositeModel) -> list[MuonDataset]:
    return [
        _dataset_for(
            run_number=700 + idx,
            field=50.0 * idx,
            temperature=5.0,
            model=model,
            params={"A_1": 0.2, "Lambda": 0.35, "A_bg": 0.01},
        )
        for idx in range(1, 5)
    ]


def _varying_lambda_series(model: CompositeModel) -> list[MuonDataset]:
    lambdas = [0.15, 0.25, 0.55, 0.9]
    return [
        _dataset_for(
            run_number=750 + idx,
            field=100.0 * idx,
            temperature=10.0,
            model=model,
            params={"A_1": 0.2, "Lambda": lambdas[idx - 1], "A_bg": 0.01},
        )
        for idx in range(1, 5)
    ]


def _winner_flip_neighbours_present(recommendation) -> bool:
    """Every single-role-flip of the winner exists as a returned assessment.

    The verdict layer (``_build_parameter_recommendations_from_exact_cache`` +
    rerank) compares the winner against its flip-neighbourhood; a sparse search
    must have explicitly fitted those P neighbours, else the verdict is starved.
    """

    winner = recommendation.recommended_assessment
    assert winner is not None
    template_key = winner.template.key
    free = tuple(
        name for name in winner.template.model.param_names if name not in winner.fixed_param_names
    )
    present = {
        (a.global_param_names, a.local_param_names)
        for a in recommendation.assessments
        if a.template.key == template_key and a.is_successful
    }
    winner_local = set(winner.local_param_names)
    for name in free:
        if name in winner_local:
            flipped_local = tuple(sorted(winner_local - {name}))
        else:
            flipped_local = tuple(sorted(winner_local | {name}))
        flipped_global = tuple(n for n in free if n not in set(flipped_local))
        if (flipped_global, flipped_local) not in present:
            return False
    return True


@pytest.mark.parametrize("engine", ["low", "balanced"])
def test_heuristic_engines_share_uniform_series(
    monkeypatch: pytest.MonkeyPatch, engine: str
) -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    _restrict_to_exp_constant_template(monkeypatch, model)
    datasets = _uniform_series(model)

    recommendation = build_global_fit_wizard_recommendation(datasets, search_engine=engine)

    assessment = recommendation.recommended_assessment
    assert assessment is not None
    assert assessment.template.key == "exp_constant"
    assert assessment.local_param_names == ()


@pytest.mark.parametrize("engine", ["low", "balanced"])
def test_heuristic_engines_localize_varying_lambda(
    monkeypatch: pytest.MonkeyPatch, engine: str
) -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    _restrict_to_exp_constant_template(monkeypatch, model)
    datasets = _varying_lambda_series(model)

    recommendation = build_global_fit_wizard_recommendation(datasets, search_engine=engine)

    assessment = recommendation.recommended_assessment
    assert assessment is not None
    assert assessment.template.key == "exp_constant"
    assert "Lambda" in assessment.local_param_names
    assert "A_1" not in assessment.local_param_names


@pytest.mark.parametrize("engine", ["low", "balanced"])
def test_heuristic_winner_flip_neighbourhood_is_fitted(
    monkeypatch: pytest.MonkeyPatch, engine: str
) -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    _restrict_to_exp_constant_template(monkeypatch, model)
    # Mixed truth: one global amplitude/background, one local rate — the winner
    # has a non-trivial flip-neighbourhood spanning all three free params.
    datasets = _varying_lambda_series(model)

    recommendation = build_global_fit_wizard_recommendation(datasets, search_engine=engine)

    assert _winner_flip_neighbours_present(recommendation), (
        "sparse search left the winner's flip-neighbourhood incomplete; the "
        "verdict/robustness layer would be starved"
    )


def test_heuristic_engine_records_q_pretest_instrumentation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    _restrict_to_exp_constant_template(monkeypatch, model)
    datasets = _varying_lambda_series(model)
    instrumentation: dict[str, object] = {}

    build_global_fit_wizard_recommendation(
        datasets, search_engine="balanced", instrumentation=instrumentation
    )

    counters = instrumentation.get("counters")
    assert isinstance(counters, dict)
    # Q pre-tests ran (some params classified) and the flip-neighbourhood filled.
    q_total = (
        int(counters.get("q_pretest_fixed_local", 0))
        + int(counters.get("q_pretest_fixed_global", 0))
        + int(counters.get("q_pretest_ambiguous", 0))
    )
    assert q_total > 0
    assert instrumentation.get("search_engine") == "balanced"


def test_separable_engine_is_the_default_and_agrees_with_the_referee(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default engine is separable, and it reaches the referee's verdict.

    ``search_engine="exhaustive"`` is the frozen-baseline wavefront and stays
    reachable through the seam; the default no longer routes there, which the
    instrumentation metric states outright.
    """
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    _restrict_to_exp_constant_template(monkeypatch, model)
    datasets = _varying_lambda_series(model)
    default_instrumentation: dict[str, object] = {}

    default = build_global_fit_wizard_recommendation(
        datasets, instrumentation=default_instrumentation
    )
    explicit = build_global_fit_wizard_recommendation(datasets, search_engine="exhaustive")

    assert default_instrumentation.get("search_engine") == "separable"
    assert default.recommended_assessment is not None
    assert explicit.recommended_assessment is not None
    assert (
        default.recommended_assessment.local_param_names
        == explicit.recommended_assessment.local_param_names
    )


def test_unknown_search_engine_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    _restrict_to_exp_constant_template(monkeypatch, model)
    datasets = _uniform_series(model)

    with pytest.raises(ValueError, match="Unknown search_engine"):
        build_global_fit_wizard_recommendation(datasets, search_engine="turbo")


# --------------------------------------------------------------------------- #
# Effort tiers (PR 5, revised): every EffortTier collapses to the exact engine.
# The heuristic engines + I/J/K knobs are retained ONLY behind the low-level
# ``search_engine`` string (the PR 4 seam), never via ``effort_tier``.
# --------------------------------------------------------------------------- #


def _restrict_to_templates(
    monkeypatch: pytest.MonkeyPatch,
    templates: tuple[CandidateTemplate, ...],
) -> None:
    monkeypatch.setattr(
        global_fit_wizard_module,
        "build_candidate_templates",
        lambda fingerprint, current_model=None: templates,
    )


@pytest.mark.parametrize("tier_value", ["low", "balanced", "thorough", "exhaustive"])
def test_effort_tier_always_resolves_to_the_separable_engine(
    monkeypatch: pytest.MonkeyPatch, tier_value: str
) -> None:
    """Every user-facing EffortTier runs the separable role search.

    The engine name is recorded on the instrumentation, so a tier that silently
    routed somewhere else would show it.
    """
    from asymmetry.core.fitting.wizard_scope import EffortTier

    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    _restrict_to_exp_constant_template(monkeypatch, model)
    datasets = _uniform_series(model)
    instrumentation: dict[str, object] = {}

    build_global_fit_wizard_recommendation(
        datasets, instrumentation=instrumentation, effort_tier=EffortTier(tier_value)
    )

    assert instrumentation.get("search_engine") == "separable"


def test_effort_tier_search_engine_map_collapses_to_separable() -> None:
    from asymmetry.core.fitting.global_fit_wizard import (
        _DEFAULT_SEARCH_ENGINE,
        _EFFORT_TIER_SEARCH_ENGINE,
        SEARCH_ENGINE_SEPARABLE,
    )
    from asymmetry.core.fitting.wizard_scope import EffortTier

    assert _DEFAULT_SEARCH_ENGINE == SEARCH_ENGINE_SEPARABLE
    assert set(_EFFORT_TIER_SEARCH_ENGINE) == set(EffortTier)
    assert all(engine == SEARCH_ENGINE_SEPARABLE for engine in _EFFORT_TIER_SEARCH_ENGINE.values())


def test_explicit_search_engine_overrides_effort_tier_engine_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from asymmetry.core.fitting.wizard_scope import EffortTier

    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    _restrict_to_exp_constant_template(monkeypatch, model)
    datasets = _uniform_series(model)
    instrumentation: dict[str, object] = {}

    # effort_tier says Balanced, but an explicit search_engine="low" must win for
    # *engine selection* (backward compatibility with PR 4 callers/tests).
    build_global_fit_wizard_recommendation(
        datasets,
        instrumentation=instrumentation,
        effort_tier=EffortTier.BALANCED,
        search_engine="low",
    )

    assert instrumentation.get("search_engine") == "low"


def _triple_exp_template() -> CandidateTemplate:
    model = CompositeModel(
        ["Exponential", "Exponential", "Exponential", "Constant"],
        operators=["+", "+", "+"],
    )
    return CandidateTemplate(
        key="triple_exp_constant",
        title="Exponential + Exponential + Exponential + Constant",
        category="General",
        rationale="test",
        model=model,
    )


def _searched_role_split_count(recommendation, template_key: str) -> int:
    """How many distinct role-split assessments a template earned.

    Every template gets one cheap initial-screen assessment (``assessment_key
    is None``) regardless of tier; a template that reached the expensive
    coupled role search additionally earns assessments with a real
    ``assessment_key`` (its all-global fit, flip-neighbourhood, etc.). Zero
    extra role splits is the observable signature of "never reached the
    search" — this is what technique I's cap must produce for an over-budget
    template.
    """
    return sum(
        1
        for assessment in recommendation.optimized_assessments()
        if assessment.template.key == template_key and assessment.assessment_key is not None
    )


def test_low_portfolio_cap_skips_over_budget_templates_via_search_engine_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Technique I is retained behind the low-level ``search_engine="low"`` seam.

    An explicit user selection on the retained Low heuristic engine still narrows
    to the cap: a >5-parameter template never reaches the expensive coupled role
    search, even though the user selected it alongside a small template that fits
    inside the budget. This knob is NO LONGER reachable via ``effort_tier`` (which
    always resolves to the exact engine) — only through the ``search_engine``
    override.
    """
    small_model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    small_template = CandidateTemplate(
        key="exp_constant",
        title="Exponential + Constant",
        category="General",
        rationale="test",
        model=small_model,
    )
    big_template = _triple_exp_template()
    _restrict_to_templates(monkeypatch, (small_template, big_template))
    datasets = _uniform_series(small_model)

    recommendation = build_global_fit_wizard_recommendation(
        datasets,
        search_engine="low",
        selected_template_keys=(small_template.key, big_template.key),
    )

    assert _searched_role_split_count(recommendation, small_template.key) > 0
    assert _searched_role_split_count(recommendation, big_template.key) == 0


def test_low_portfolio_cap_is_inert_via_effort_tier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passing ``effort_tier=LOW`` no longer applies the cap.

    Every user-facing tier resolves to the separable engine, so an over-budget
    template still reaches the coupled search — the I/J/K knobs are only reachable
    through the ``search_engine`` override now.
    """
    from asymmetry.core.fitting.wizard_scope import EffortTier

    small_model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    small_template = CandidateTemplate(
        key="exp_constant",
        title="Exponential + Constant",
        category="General",
        rationale="test",
        model=small_model,
    )
    big_template = _triple_exp_template()
    _restrict_to_templates(monkeypatch, (small_template, big_template))
    datasets = _uniform_series(small_model)

    recommendation = build_global_fit_wizard_recommendation(
        datasets,
        effort_tier=EffortTier.LOW,
        selected_template_keys=(small_template.key, big_template.key),
    )

    # The separable engine (what LOW now resolves to) applies no cap: both
    # explicitly selected templates reach the coupled search.
    assert _searched_role_split_count(recommendation, small_template.key) > 0
    assert _searched_role_split_count(recommendation, big_template.key) > 0


def test_low_complexity_prior_prefers_fewer_additive_terms() -> None:
    from asymmetry.core.fitting.global_fit_wizard import _low_complexity_prior_penalty

    exp_constant = CandidateTemplate(
        key="exp_constant",
        title="Exponential + Constant",
        category="General",
        rationale="test",
        model=CompositeModel(["Exponential", "Constant"], operators=["+"]),
    )
    triple_exp = _triple_exp_template()

    assert _low_complexity_prior_penalty(exp_constant) == 0.0
    assert _low_complexity_prior_penalty(triple_exp) > _low_complexity_prior_penalty(exp_constant)


def test_identifiability_demotion_flags_highly_correlated_covariance() -> None:
    from asymmetry.core.fitting.global_fit_wizard import (
        _initial_assessment_is_identifiability_degenerate,
    )

    template = CandidateTemplate(
        key="exp_constant",
        title="Exponential + Constant",
        category="General",
        rationale="test",
        model=CompositeModel(["Exponential", "Constant"], operators=["+"]),
    )
    covariance = np.array([[1.0, 0.999], [0.999, 1.0]])
    result = FitResult(
        success=True,
        chi_squared=1.0,
        parameters=ParameterSet(
            [Parameter(name="A_1", value=0.2), Parameter(name="Lambda", value=0.3)]
        ),
        uncertainties={"A_1": 0.01, "Lambda": 0.02},
        covariance=covariance,
        covariance_parameters=["A_1", "Lambda"],
    )
    assessment = GlobalCandidateAssessment(
        template=template,
        fit_results_by_run={1: result},
        global_parameters=ParameterSet(),
        global_param_names=(),
        local_param_names=(),
        fixed_param_names=(),
        parameter_recommendations=(),
        run_diagnostics=(),
        series_warnings=(),
        aic=10.0,
        aicc=10.0,
        bic=12.0,
        selected_score=10.0,
        fitted_curves_by_run={},
        component_curves_by_run={},
    )

    assert _initial_assessment_is_identifiability_degenerate(assessment)


def test_identifiability_demotion_ignores_well_conditioned_covariance() -> None:
    from asymmetry.core.fitting.global_fit_wizard import (
        _initial_assessment_is_identifiability_degenerate,
    )

    template = CandidateTemplate(
        key="exp_constant",
        title="Exponential + Constant",
        category="General",
        rationale="test",
        model=CompositeModel(["Exponential", "Constant"], operators=["+"]),
    )
    covariance = np.array([[1.0, 0.05], [0.05, 1.0]])
    result = FitResult(
        success=True,
        chi_squared=1.0,
        parameters=ParameterSet(
            [Parameter(name="A_1", value=0.2), Parameter(name="Lambda", value=0.3)]
        ),
        uncertainties={"A_1": 0.01, "Lambda": 0.02},
        covariance=covariance,
        covariance_parameters=["A_1", "Lambda"],
    )
    assessment = GlobalCandidateAssessment(
        template=template,
        fit_results_by_run={1: result},
        global_parameters=ParameterSet(),
        global_param_names=(),
        local_param_names=(),
        fixed_param_names=(),
        parameter_recommendations=(),
        run_diagnostics=(),
        series_warnings=(),
        aic=10.0,
        aicc=10.0,
        bic=12.0,
        selected_score=10.0,
        fitted_curves_by_run={},
        component_curves_by_run={},
    )

    assert not _initial_assessment_is_identifiability_degenerate(assessment)


@pytest.mark.parametrize("engine", ["low", "balanced"])
def test_screening_decimation_fires_and_leaves_full_resolution_leaderboard(
    monkeypatch: pytest.MonkeyPatch, engine: str
) -> None:
    """Technique K: decimation applies during search but never on the returned ICs.

    Every returned assessment's per-run fit results carry the full dataset
    point count (``dof`` reflects it), proving the winner + flip-neighbourhood
    were refitted at native resolution and no decimated assessment leaked into
    the leaderboard the verdict layer reranks over.
    """
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    _restrict_to_exp_constant_template(monkeypatch, model)
    datasets = _varying_lambda_series(model)
    full_n_points = datasets[0].n_points
    instrumentation: dict[str, object] = {}

    recommendation = build_global_fit_wizard_recommendation(
        datasets, search_engine=engine, instrumentation=instrumentation
    )

    counters = instrumentation.get("counters")
    assert isinstance(counters, dict)
    assert counters.get("decimation_applied", 0) >= 1

    for assessment in recommendation.optimized_assessments():
        for run_number, result in assessment.fit_results_by_run.items():
            dataset = next(d for d in datasets if int(d.run_number) == run_number)
            assert dataset.n_points == full_n_points
            n_free = len(assessment.global_param_names) + len(assessment.local_param_names)
            # dof = N_data - N_free (per-run free count); a decimated leftover
            # would show a much smaller dof for the same free-parameter count.
            assert result.dof == pytest.approx(full_n_points - n_free, abs=2)


def test_screening_decimation_skips_when_nyquist_gate_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from asymmetry.core.fitting.fit_wizard import SpectrumFingerprint
    from asymmetry.core.fitting.global_fit_wizard import (
        _DECIMATION_FACTOR_BALANCED,
        _decimated_datasets_for_search,
    )

    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    datasets = _uniform_series(model)
    # A high dominant frequency relative to the coarse-binned Nyquist limit —
    # decimating would alias this content, so the gate must refuse.
    dt = float(np.mean(np.diff(datasets[0].time)))
    aliasing_fingerprint = SpectrumFingerprint(
        tail_estimate=0.0,
        initial_amplitude_estimate=0.2,
        zero_crossings=10,
        smoothed_zero_crossings=10,
        smoothed_turning_points=10,
        dominant_fft_frequency_mhz=0.5 / (dt * _DECIMATION_FACTOR_BALANCED),
        dominant_fft_snr=10.0,
        dominant_fft_cycles_in_window=5.0,
        monotonic_decay_fraction=0.1,
        early_time_curvature=0.0,
        semilog_slope_ratio=0.0,
        late_time_dip_recovery_score=0.0,
        oscillatory_hint=True,
        kt_like_hint=False,
        multi_rate_hint=False,
    )

    search_datasets, factor = _decimated_datasets_for_search(
        datasets,
        engine="balanced",
        aggregate_fingerprint=aliasing_fingerprint,
        instrumentation=None,
    )

    assert factor == 1
    assert search_datasets is datasets


def test_thorough_and_exhaustive_engines_stay_byte_identical() -> None:
    """PR 5 tier policy: Thorough is a faithful alias of the exact wavefront.

    The tier table calls for "full wavefront, exact bounds A/B only, generous
    margins" at Thorough — but there is no independent acceptance bar for it
    (only Exhaustive is the referee), and building a third hybrid enumerator
    would risk the one path that must stay byte-identical. Thorough therefore
    resolves to exactly the same ``SEARCH_ENGINE_THOROUGH`` string as before,
    which ``_EXACT_SEARCH_ENGINES`` already routes to the untouched wavefront.
    """
    from asymmetry.core.fitting.global_fit_wizard import (
        _EXACT_SEARCH_ENGINES,
        SEARCH_ENGINE_EXHAUSTIVE,
        SEARCH_ENGINE_THOROUGH,
    )

    assert SEARCH_ENGINE_THOROUGH in _EXACT_SEARCH_ENGINES
    assert SEARCH_ENGINE_EXHAUSTIVE in _EXACT_SEARCH_ENGINES


# --------------------------------------------------------------------------- #
# Cooperative cancel (PR 5 rework): a cancel_callback aborts the exact search
# promptly with FitCancelledError, before the full role search runs.
# --------------------------------------------------------------------------- #


def test_cancel_callback_aborts_before_running_full_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancel_callback that returns True aborts quickly and runs no fits.

    Cancel is checked at the top of the builder (before any screening/anchor
    fit), so a callback that is already truthy raises ``FitCancelledError``
    without dispatching the exhaustive role search — proven by the absence of any
    ``exact_fit_invocations`` / ``global_fit_calls`` instrumentation.
    """
    from asymmetry.core.fitting import FitCancelledError

    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    _restrict_to_exp_constant_template(monkeypatch, model)
    datasets = _uniform_series(model)
    instrumentation: dict[str, object] = {}

    with pytest.raises(FitCancelledError):
        build_global_fit_wizard_recommendation(
            datasets,
            instrumentation=instrumentation,
            cancel_callback=lambda: True,
        )

    counters = instrumentation.get("counters")
    if isinstance(counters, dict):
        assert counters.get("exact_fit_invocations", 0) == 0
        assert counters.get("global_fit_calls", 0) == 0


def test_cancel_callback_aborts_mid_search_past_the_top_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancel is honoured *inside* the exact search, not only at the top guard.

    A callback that returns False first (letting execution past the staged-top
    guard) and True afterwards must still raise ``FitCancelledError`` — proving
    the between-templates / between-layers / before-dispatch checks inside
    ``_run_exhaustive_wavefront_search`` are wired, which is the whole point of
    the cancel fix (a cancel that only fires before the search starts is
    useless). We assert the callback advanced past its first poll, so the abort
    came from a later internal check rather than the top guard.
    """
    from asymmetry.core.fitting import FitCancelledError

    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    _restrict_to_exp_constant_template(monkeypatch, model)
    datasets = _uniform_series(model)

    state = {"calls": 0}

    def _cancel_after_first_poll() -> bool:
        state["calls"] += 1
        # First poll (the staged-top guard) passes; every later poll cancels.
        return state["calls"] > 1

    with pytest.raises(FitCancelledError):
        build_global_fit_wizard_recommendation(
            datasets,
            cancel_callback=_cancel_after_first_poll,
        )

    assert state["calls"] > 1


def test_cancel_callback_false_completes_normally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancel_callback that never fires does not perturb the exact result."""
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    _restrict_to_exp_constant_template(monkeypatch, model)
    datasets = _uniform_series(model)

    recommendation = build_global_fit_wizard_recommendation(
        datasets,
        cancel_callback=lambda: False,
    )

    assert recommendation.recommended_key is not None


# --------------------------------------------------------------------------- #
# Phase-1 single-fit table fan-out. Reported from a downstream continuous-source
# ZF/TF analysis as "screening never returns": the fits themselves are minutes
# long, and the drain used to block in ``as_completed`` with no cancel poll and
# then tear the pool down with a *blocking* shutdown, so neither a Cancel nor a
# KeyboardInterrupt could get the call back — indistinguishable from a deadlock,
# and a killed parent left the spawn workers orphaned.
# --------------------------------------------------------------------------- #


def _exponential_decay_datasets(count: int = 4) -> list[MuonDataset]:
    """Synthetic exponentially relaxing ZF runs built from bare arrays."""
    time_us = np.linspace(0.0, 8.0, 200)
    datasets: list[MuonDataset] = []
    for index in range(count):
        rate = 0.4 + 0.05 * index
        datasets.append(
            MuonDataset(
                time=time_us,
                asymmetry=20.0 * np.exp(-rate * time_us),
                error=np.full_like(time_us, 0.3),
                metadata={
                    "run_number": 900 + index,
                    "temperature": 5.0 + index,
                    "field": 0.0,
                    "field_direction": "ZF",
                },
            )
        )
    return datasets


class _NeverCompletingPool:
    """Stand-in for a spawn pool whose submitted fits never finish.

    Models the real regime — a phase-1 single-fit table takes minutes — without
    paying for it. ``shutdown`` records its arguments so the test can prove the
    teardown does not wait on the in-flight work.
    """

    def __init__(self) -> None:
        self.shutdown_calls: list[dict[str, object]] = []
        self.submitted = 0

    def submit(self, *_args: object, **_kwargs: object) -> Future:
        self.submitted += 1
        return Future()  # deliberately never resolved

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        self.shutdown_calls.append({"wait": wait, "cancel_futures": cancel_futures})


@pytest.mark.timeout(90)
def test_phase_one_screening_cancel_does_not_block_on_the_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancel is polled *while draining* phase 1, and teardown never blocks.

    On the pre-fix drain this test fails by timeout: ``as_completed`` waits for
    futures that never resolve, and the ``finally`` shutdown then waits again.
    """
    pool = _NeverCompletingPool()
    monkeypatch.setattr(
        global_fit_wizard_module,
        "_try_open_process_pool",
        lambda **_kwargs: pool,
    )
    datasets = _exponential_decay_datasets(4)
    _stub_single_run_wizard(
        monkeypatch,
        (
            _restrict_to_exp_constant_template(
                monkeypatch, CompositeModel(["Exponential", "Constant"], operators=["+"])
            ),
        ),
    )

    def _cancel_once_dispatched() -> bool:
        # Stays false through the guards that precede the fan-out, so only a
        # poll from *inside* the drain loop can see it fire.
        return pool.submitted >= len(datasets)

    started = time.monotonic()
    with pytest.raises(FitCancelledError):
        build_global_fit_wizard_screening_recommendation(
            datasets,
            cancel_callback=_cancel_once_dispatched,
        )
    elapsed = time.monotonic() - started

    assert pool.submitted == len(datasets)
    assert elapsed < 30.0
    # Non-blocking teardown: queued work dropped, no wait on the in-flight fits.
    assert pool.shutdown_calls == [{"wait": False, "cancel_futures": True}]


class _ImmediatePool:
    """Spawn-pool stand-in that runs each task inline and resolves at once."""

    def __init__(self) -> None:
        self.shutdown_calls: list[dict[str, object]] = []

    def submit(self, fn, /, *args: object, **kwargs: object) -> Future:
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as error:  # pragma: no cover - defensive
            future.set_exception(error)
        return future

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        self.shutdown_calls.append({"wait": wait, "cancel_futures": cancel_futures})


@pytest.mark.timeout(300)
def test_phase_one_screening_reports_progress_per_completed_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The uncancelled drain still completes, and each finished table is logged.

    Without a per-completion message the stage is silent for its whole (many
    minute) duration, which is what made the downstream report read as a hang.
    """
    pool = _ImmediatePool()
    monkeypatch.setattr(
        global_fit_wizard_module,
        "_try_open_process_pool",
        lambda **_kwargs: pool,
    )
    datasets = _exponential_decay_datasets(2)
    _stub_single_run_wizard(
        monkeypatch,
        (
            _restrict_to_exp_constant_template(
                monkeypatch, CompositeModel(["Exponential", "Constant"], operators=["+"])
            ),
        ),
    )
    messages: list[str] = []

    table = build_or_complete_single_fit_wizard_recommendations_for_global_portfolio(
        datasets,
        progress_callback=messages.append,
    )

    assert sorted(table.generated_run_numbers) == [900, 901]
    assert set(table.recommendations_by_run) == {900, 901}
    analysed = [text for text in messages if "single-fit analysis" in text.lower() and "/" in text]
    assert len(analysed) == len(datasets)
    scored = [text for text in messages if "score table" in text.lower() and "/" in text]
    assert len(scored) == len(datasets)
    assert pool.shutdown_calls == [{"wait": True, "cancel_futures": False}]


def test_single_fit_table_worker_count_never_oversubscribes_the_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase-1 width is the CPU count, bounded only by how many tables there are.

    A fixed width of four on a two-core host runs four minutes-long fits over
    two cores, which is slower than two workers *and* looks stalled — and the
    same fixed four on a large host left a 14-dataset series running at well
    under one core while the stage looked hung from outside. Phase-1 tables are
    independent whole-dataset jobs, so the host's core count is the right bound.
    """
    monkeypatch.setattr(global_fit_wizard_module.os, "cpu_count", lambda: 2)
    assert global_fit_wizard_module._single_fit_table_worker_count(37) == 2

    monkeypatch.setattr(global_fit_wizard_module.os, "cpu_count", lambda: 16)
    assert global_fit_wizard_module._single_fit_table_worker_count(37) == 16
    assert global_fit_wizard_module._single_fit_table_worker_count(9) == 9
    assert global_fit_wizard_module._single_fit_table_worker_count(1) == 1
    assert global_fit_wizard_module._single_fit_table_worker_count(0) == 1

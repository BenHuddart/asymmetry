"""Scaffolding tests for the tiered fit-wizard screening substrate.

These cover the mechanical pieces added ahead of the tiered-screening flow:
``build_wizard_families``, the ``FamilyScreeningReport`` serializer, the
additive recommendation/assessment serializer keys, and the
``_run_template_assessments`` process/thread fan-out helper. The orchestrating
flow tests land later in the same file.
"""

from __future__ import annotations

import concurrent.futures
import math
import pickle
import time
from dataclasses import replace

import numpy as np
import pytest

import asymmetry.core.fitting.fit_wizard as fit_wizard_module
from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.component_tags import ComputationalCost
from asymmetry.core.fitting.composite import COMPONENTS, CompositeModel
from asymmetry.core.fitting.engine import FitCancelledError, FitResult
from asymmetry.core.fitting.fit_wizard import (
    _FIT_WIZARD_TITLES,
    _FMUF_R_LADDER,
    CandidateAssessment,
    CandidateTemplate,
    ConfidenceTier,
    FamilyScreeningReport,
    FitWizardRecommendation,
    RecommendationVerdict,
    SelectionMetric,
    SpectrumFingerprint,
    WizardFamily,
    _apply_frequency_support_disqualifiers,
    _AssessmentTask,
    _decide_family_promotions,
    _effective_hint_keys,
    _fmuf_r_ladder_variants,
    _initial_parameters_for_template,
    _parameter_variants,
    _run_template_assessments,
    _stage2_variant_budget,
    build_fit_wizard_recommendation,
    build_null_baseline_templates,
    build_wizard_families,
    deserialize_family_screening_report,
    deserialize_fit_wizard_recommendation,
    fingerprint_spectrum,
    rerank_fit_wizard_recommendation,
    serialize_family_screening_report,
    serialize_fit_wizard_recommendation,
)
from asymmetry.core.fitting.models import longitudinal_field_kubo_toyabe
from asymmetry.core.fitting.muon_fluorine.polarization import linear_fmuf_polarization
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fitting.peak_detection import (
    DetectedPeak,
    MultipletMatch,
    PeakAnalysis,
)
from asymmetry.core.fitting.wizard_scope import (
    WizardScope,
    WizardScopePreset,
    resolve_scope,
)


def _plain_fingerprint(**overrides: object) -> SpectrumFingerprint:
    base = dict(
        tail_estimate=0.0,
        initial_amplitude_estimate=0.2,
        zero_crossings=0,
        smoothed_zero_crossings=0,
        smoothed_turning_points=0,
        dominant_fft_frequency_mhz=0.0,
        dominant_fft_snr=0.0,
        dominant_fft_cycles_in_window=0.0,
        monotonic_decay_fraction=1.0,
        early_time_curvature=0.0,
        semilog_slope_ratio=1.0,
        late_time_dip_recovery_score=0.0,
        oscillatory_hint=False,
        kt_like_hint=False,
        multi_rate_hint=False,
    )
    base.update(overrides)
    return SpectrumFingerprint(**base)  # type: ignore[arg-type]


_EXPECTED_REPS = {
    "relaxation": "exp_constant",
    "multi_rate": "biexp_constant",
    "kt": "static_gkt_constant",
    "oscillatory": "oscillatory_exp_constant",
    "muonium": "muonium_low_tf_constant",
    "fmuf": "fmuf_linear_exp_constant",
}

_CANONICAL_ORDER = ("relaxation", "multi_rate", "kt", "oscillatory", "muonium", "fmuf")


# --------------------------------------------------------------------------- #
# build_wizard_families
# --------------------------------------------------------------------------- #


def test_plain_fingerprint_yields_all_six_families_in_canonical_order() -> None:
    families = build_wizard_families(_plain_fingerprint())
    assert tuple(f.key for f in families) == _CANONICAL_ORDER
    for family in families:
        assert family.stage1_rep.key == _EXPECTED_REPS[family.key]
        assert family.priority == 0.0
        # The representative is never repeated among stage-2 members.
        member_keys = [m.key for m in family.stage2_members]
        assert family.stage1_rep.key not in member_keys


def test_no_duplicate_template_keys_across_the_table() -> None:
    families = build_wizard_families(_plain_fingerprint())
    all_keys: list[str] = []
    for family in families:
        all_keys.append(family.stage1_rep.key)
        all_keys.extend(m.key for m in family.stage2_members)
    assert len(all_keys) == len(set(all_keys))


def test_every_template_key_is_registered_in_titles() -> None:
    families = build_wizard_families(
        _plain_fingerprint(),
        current_model=CompositeModel(["Exponential", "Constant"], operators=["+"]),
    )
    for family in families:
        for template in (family.stage1_rep, *family.stage2_members):
            assert template.key in _FIT_WIZARD_TITLES
            assert template.title == _FIT_WIZARD_TITLES[template.key]


def test_multi_rate_hint_raises_priority_ordering() -> None:
    families = build_wizard_families(_plain_fingerprint(multi_rate_hint=True))
    assert families[0].key == "multi_rate"
    assert families[0].priority == 1.0
    # Ties (the remaining zero-priority families) keep canonical order.
    rest = tuple(f.key for f in families[1:])
    assert rest == ("relaxation", "kt", "oscillatory", "muonium", "fmuf")


def test_kt_and_oscillatory_hints_raise_their_priority() -> None:
    families = build_wizard_families(_plain_fingerprint(kt_like_hint=True, oscillatory_hint=True))
    prioritised = [f.key for f in families if f.priority == 1.0]
    assert set(prioritised) == {"kt", "oscillatory"}
    # Between the two priority-1 families, canonical order (kt before oscillatory).
    assert prioritised == ["kt", "oscillatory"]


# --------------------------------------------------------------------------- #
# Scope filtering
# --------------------------------------------------------------------------- #


def test_fluoride_fmuf_scope_reduces_families() -> None:
    resolution = resolve_scope(WizardScope(preset=WizardScopePreset.FLUORIDE_FMUF))
    families = build_wizard_families(_plain_fingerprint(), scope_resolution=resolution)
    keys = {f.key for f in families}
    # No transverse-precession or Kubo-Toyabe families survive the ZF/LF molecular scope.
    assert "oscillatory" not in keys
    assert "kt" not in keys
    # The fmuf family survives with its collinear representative.
    fmuf = next(f for f in families if f.key == "fmuf")
    assert fmuf.stage1_rep.key == "fmuf_linear_exp_constant"


def test_scope_rep_fallback_promotes_cheapest_surviving_member() -> None:
    # Excluding MuoniumLowTF (the rep) while keeping the other muonium forms
    # forces promotion of the cheapest surviving member; all muonium members are
    # CHEAP, so the tie breaks alphabetically -> muonium_high_tf_constant.
    resolution = resolve_scope(
        WizardScope(
            preset=WizardScopePreset.MUONIUM_RADICAL,
            exclude_components=frozenset({"MuoniumLowTF"}),
        )
    )
    assert "MuoniumLowTF" not in resolution.included_set
    assert "MuoniumTF" in resolution.included_set
    families = build_wizard_families(_plain_fingerprint(), scope_resolution=resolution)
    muonium = next(f for f in families if f.key == "muonium")
    assert muonium.stage1_rep.key == "muonium_high_tf_constant"
    assert "muonium_high_tf_constant" not in {m.key for m in muonium.stage2_members}


def test_scope_omits_family_with_nothing_surviving() -> None:
    resolution = resolve_scope(WizardScope(preset=WizardScopePreset.FLUORIDE_FMUF))
    families = build_wizard_families(_plain_fingerprint(), scope_resolution=resolution)
    # Muonium has no ZF/LF molecular component in scope, so the family is omitted.
    assert "muonium" not in {f.key for f in families}


def test_baseline_family_is_last_and_never_scope_filtered() -> None:
    # A narrow scope that would drop the current model's components must still
    # keep the baseline family (a Bessel oscillatory model has no ZF-molecular
    # component, yet the baseline family is exempt from scope filtering).
    current_model = CompositeModel(["Bessel", "Exponential", "Constant"], operators=["*", "+"])
    resolution = resolve_scope(WizardScope(preset=WizardScopePreset.FLUORIDE_FMUF))
    families = build_wizard_families(
        _plain_fingerprint(), current_model=current_model, scope_resolution=resolution
    )
    assert families[-1].key == "baseline"
    baseline = families[-1]
    assert baseline.stage1_rep.is_current_model_baseline is True
    assert baseline.stage2_members == ()
    assert baseline.must_run_stage1 is True


# --------------------------------------------------------------------------- #
# FamilyScreeningReport serialization
# --------------------------------------------------------------------------- #


def test_family_screening_report_round_trip_with_inf_metric() -> None:
    report = FamilyScreeningReport(
        family_key="kt",
        title="Kubo-Toyabe",
        stage1_template_key="static_gkt_constant",
        stage1_metric_value=math.inf,
        stage1_gate_passed=False,
        promoted=False,
        reason="representative fit failed",
        stage2_template_keys=("dynamic_gkt_constant",),
    )
    payload = serialize_family_screening_report(report)
    # inf cannot live in JSON: it is stored as None.
    assert payload["stage1_metric_value"] is None
    restored = deserialize_family_screening_report(payload)
    assert restored is not None
    assert math.isinf(restored.stage1_metric_value)
    assert restored == report


def test_family_screening_report_round_trip_finite_metric() -> None:
    report = FamilyScreeningReport(
        family_key="relaxation",
        title="Relaxation",
        stage1_template_key="exp_constant",
        stage1_metric_value=12.5,
        stage1_gate_passed=True,
        promoted=True,
        reason="best Stage-1 family",
        stage2_template_keys=("gaussian_constant", "stretched_constant"),
    )
    restored = deserialize_family_screening_report(serialize_family_screening_report(report))
    assert restored == report


def test_family_screening_report_legacy_payload_tolerance() -> None:
    # A sparse/legacy dict deserializes with defaults; missing metric -> inf.
    restored = deserialize_family_screening_report({"family_key": "muonium"})
    assert restored is not None
    assert restored.family_key == "muonium"
    assert restored.title == ""
    assert restored.stage1_template_key == ""
    assert math.isinf(restored.stage1_metric_value)
    assert restored.stage1_gate_passed is False
    assert restored.promoted is False
    assert restored.stage2_template_keys == ()
    assert deserialize_family_screening_report("not-a-dict") is None


# --------------------------------------------------------------------------- #
# Recommendation / assessment serializer additions
# --------------------------------------------------------------------------- #


def _dummy_assessment(key: str, *, stage: int = 2) -> CandidateAssessment:
    template = CandidateTemplate(
        key=key,
        title=key,
        category="General",
        rationale="",
        model=CompositeModel(["Exponential", "Constant"], operators=["+"]),
    )
    fit_result = FitResult(success=True, chi_squared=1.0, reduced_chi_squared=1.0)
    empty = np.array([], dtype=float)
    return CandidateAssessment(
        template=template,
        fit_result=fit_result,
        aic=1.0,
        aicc=1.0,
        bic=1.0,
        selected_score=1.0,
        residual_rms=1.0,
        runs_z_score=0.0,
        max_abs_autocorrelation=0.0,
        residual_fft_peak_snr=0.0,
        residual_gate_passed=True,
        residual_gate_reasons=(),
        bound_hits=(),
        fitted_time=empty,
        fitted_curve=empty,
        component_curves=(),
        stage=stage,
    )


def _recommendation_with_extras() -> FitWizardRecommendation:
    analysis = PeakAnalysis(
        peaks=(
            DetectedPeak(
                frequency_mhz=1.5,
                amplitude=2.0,
                snr=8.0,
                width_mhz=0.1,
                prominence=1.0,
                source="fft",
                burg_confirmed=True,
            ),
        ),
        noise_floor=0.25,
        resolution_mhz=0.1,
        nyquist_mhz=50.0,
        detrended=True,
        detrend_template_key="exp_constant",
        burg_order=8,
    )
    match = MultipletMatch(
        kind="larmor",
        family_key="oscillatory",
        peak_indices=(0,),
        quality=0.9,
        derived_values=(("field_gauss", 110.0),),
        note="line matches Larmor frequency",
    )
    report = FamilyScreeningReport(
        family_key="oscillatory",
        title="Precession",
        stage1_template_key="oscillatory_exp_constant",
        stage1_metric_value=math.inf,
        stage1_gate_passed=False,
        promoted=False,
        reason="rep failed",
    )
    return FitWizardRecommendation(
        fingerprint=_plain_fingerprint(),
        templates=(),
        assessments=(_dummy_assessment("exp_constant"),),
        metric=SelectionMetric.AICC,
        recommended_key="exp_constant",
        comparable_keys=(),
        summary="ok",
        peak_analysis=analysis,
        multiplet_matches=(match,),
        family_reports=(report,),
    )


def test_recommendation_round_trip_with_new_fields() -> None:
    recommendation = _recommendation_with_extras()
    payload = serialize_fit_wizard_recommendation(recommendation)
    restored = deserialize_fit_wizard_recommendation(payload)
    assert restored is not None

    assert restored.peak_analysis is not None
    assert len(restored.peak_analysis.peaks) == 1
    assert restored.peak_analysis.peaks[0].frequency_mhz == 1.5
    assert restored.peak_analysis.detrend_template_key == "exp_constant"

    assert len(restored.multiplet_matches) == 1
    assert restored.multiplet_matches[0].kind == "larmor"
    assert restored.multiplet_matches[0].derived("field_gauss") == 110.0

    assert len(restored.family_reports) == 1
    assert restored.family_reports[0].family_key == "oscillatory"
    assert math.isinf(restored.family_reports[0].stage1_metric_value)


def test_recommendation_legacy_payload_defaults_new_fields() -> None:
    recommendation = _recommendation_with_extras()
    payload = serialize_fit_wizard_recommendation(recommendation)
    # Simulate an older persisted payload predating the tiered fields.
    for key in ("peak_analysis", "multiplet_matches", "family_reports"):
        payload.pop(key, None)
    restored = deserialize_fit_wizard_recommendation(payload)
    assert restored is not None
    assert restored.peak_analysis is None
    assert restored.multiplet_matches == ()
    assert restored.family_reports == ()


def test_candidate_assessment_stage_default_on_legacy_payload() -> None:
    assessment = _dummy_assessment("exp_constant", stage=1)
    recommendation = FitWizardRecommendation(
        fingerprint=_plain_fingerprint(),
        templates=(),
        assessments=(assessment,),
        metric=SelectionMetric.AICC,
        recommended_key=None,
        comparable_keys=(),
        summary="",
    )
    payload = serialize_fit_wizard_recommendation(recommendation)
    assert payload["assessments"][0]["stage"] == 1

    # Drop the stage key to emulate a legacy assessment payload -> defaults to 2.
    payload["assessments"][0].pop("stage")
    restored = deserialize_fit_wizard_recommendation(payload)
    assert restored is not None
    assert restored.assessments[0].stage == 2


# --------------------------------------------------------------------------- #
# _AssessmentTask / _run_template_assessments fan-out helper
# --------------------------------------------------------------------------- #


def _dummy_task(key: str) -> _AssessmentTask:
    template = CandidateTemplate(
        key=key,
        title=key,
        category="General",
        rationale="",
        model=CompositeModel(["Exponential", "Constant"], operators=["+"]),
    )
    return _AssessmentTask(
        dataset=MuonDataset(
            time=np.linspace(0.0, 1.0, 4),
            asymmetry=np.zeros(4),
            error=np.ones(4),
            metadata={"run_number": 1},
        ),
        fingerprint=_plain_fingerprint(),
        template=template,
        metric=SelectionMetric.AICC,
        seed_context=None,
        variant_budget=1,
        stage=1,
    )


def _order_preserving_tasks() -> tuple[list[_AssessmentTask], list[str]]:
    keys = ["a", "b", "c", "d", "e"]
    return [_dummy_task(k) for k in keys], keys


class _FakeProcessPool:
    """A stand-in for a spawn ``ProcessPoolExecutor`` that runs tasks inline.

    Mirrors the pattern in ``test_global_fit_wizard.py``: submitting a task
    just calls it synchronously. Real :class:`concurrent.futures.Future`
    objects are used (not a bespoke fake) because ``_run_template_assessments``
    drives completion through ``concurrent.futures.as_completed``, which
    reaches into a future's internal condition/state — a duck-typed fake with
    only ``.result()`` cannot satisfy it. Running synchronously (rather than on
    a real process) keeps the test deterministic and keeps test-only
    monkeypatches visible (a real ``spawn`` worker cannot see them).
    """

    def __init__(self) -> None:
        self.shutdown_calls: list[dict] = []

    def submit(self, fn, *args):
        future: concurrent.futures.Future = concurrent.futures.Future()
        try:
            future.set_result(fn(*args))
        except BaseException as exc:  # noqa: BLE001 - mirror a real future's behavior
            future.set_exception(exc)
        return future

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        self.shutdown_calls.append({"wait": wait, "cancel_futures": cancel_futures})


def _monkeypatch_dummy_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the real fit worker with one that maps a task to ``_dummy_assessment``.

    ``_execute_assessment_task`` normally drives a real fit; these tests only
    care about fan-out mechanics (ordering, cancellation, pool lifecycle), so a
    fake keeps them fast and deterministic on both the serial and process-pool
    paths (a monkeypatch on the plain function is visible to
    ``_FakeProcessPool.submit`` because it calls it in-process, unlike a real
    spawned worker).
    """

    def _fake_execute(task, cancel_callback=None):
        return _dummy_assessment(task.template.key, stage=task.stage)

    monkeypatch.setattr(fit_wizard_module, "_execute_assessment_task", _fake_execute)


def test_run_template_assessments_preserves_order_serial() -> None:
    tasks, keys = _order_preserving_tasks()
    results = _run_template_assessments(tasks, max_workers=1)
    assert [a.template.key for a in results] == keys


def test_run_template_assessments_empty_returns_empty() -> None:
    assert _run_template_assessments([]) == []


def test_run_template_assessments_preserves_order_process_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _monkeypatch_dummy_worker(monkeypatch)
    tasks, keys = _order_preserving_tasks()
    pool = _FakeProcessPool()
    monkeypatch.setattr(fit_wizard_module, "open_spawn_pool", lambda workers: pool)

    results = _run_template_assessments(tasks, max_workers=4)

    assert [a.template.key for a in results] == keys
    # A pool passed in by the caller (or, here, opened by this call because no
    # executor= was supplied) is closed by this call in its finally.
    assert pool.shutdown_calls == [{"wait": True, "cancel_futures": False}]


def test_run_template_assessments_reuses_caller_supplied_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _monkeypatch_dummy_worker(monkeypatch)
    tasks, keys = _order_preserving_tasks()
    pool = _FakeProcessPool()

    def _fail_open(_workers):
        raise AssertionError("must not open a new pool when executor= is given")

    monkeypatch.setattr(fit_wizard_module, "open_spawn_pool", _fail_open)

    results = _run_template_assessments(tasks, max_workers=4, executor=pool)

    assert [a.template.key for a in results] == keys
    # A caller-supplied pool is the caller's to shut down, not this call's.
    assert pool.shutdown_calls == []


def test_run_template_assessments_falls_back_to_threads_when_spawn_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _monkeypatch_dummy_worker(monkeypatch)
    tasks, keys = _order_preserving_tasks()
    monkeypatch.setattr(fit_wizard_module, "open_spawn_pool", lambda workers: None)

    results = _run_template_assessments(tasks, max_workers=4)

    assert [a.template.key for a in results] == keys


def test_run_template_assessments_process_path_cancels_promptly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _monkeypatch_dummy_worker(monkeypatch)
    tasks, _keys = _order_preserving_tasks()
    pool = _FakeProcessPool()
    monkeypatch.setattr(fit_wizard_module, "open_spawn_pool", lambda workers: pool)

    calls = {"n": 0}

    def _cancel_after_first() -> bool:
        calls["n"] += 1
        return calls["n"] > 1

    with pytest.raises(FitCancelledError):
        _run_template_assessments(tasks, max_workers=4, cancel_callback=_cancel_after_first)

    # The pool this call opened is torn down with cancel_futures on abort.
    assert pool.shutdown_calls == [{"wait": False, "cancel_futures": True}]


def test_run_template_assessments_retries_failed_future_serially(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks, keys = _order_preserving_tasks()
    pool = _FakeProcessPool()
    monkeypatch.setattr(fit_wizard_module, "open_spawn_pool", lambda workers: pool)

    call_count = {"c": 0}

    def _fake_execute(task, cancel_callback=None):
        # Fail the *first* call for task "c" (the submitted, process-pool
        # call) and succeed on the retry (the serial in-parent fallback).
        if task.template.key == "c":
            call_count["c"] += 1
            if call_count["c"] == 1:
                raise RuntimeError("boom")
        return _dummy_assessment(task.template.key, stage=task.stage)

    monkeypatch.setattr(fit_wizard_module, "_execute_assessment_task", _fake_execute)

    results = _run_template_assessments(tasks, max_workers=4)

    assert [a.template.key for a in results] == keys
    assert call_count["c"] == 2


def test_run_template_assessments_propagates_cancelled_error_from_future(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks, _keys = _order_preserving_tasks()
    pool = _FakeProcessPool()
    monkeypatch.setattr(fit_wizard_module, "open_spawn_pool", lambda workers: pool)

    def _fake_execute(task, cancel_callback=None):
        raise FitCancelledError("cancelled inside worker")

    monkeypatch.setattr(fit_wizard_module, "_execute_assessment_task", _fake_execute)

    with pytest.raises(FitCancelledError):
        _run_template_assessments(tasks, max_workers=4)


def test_assessment_task_pickle_round_trip() -> None:
    task = _dummy_task("exp_constant")
    restored = pickle.loads(pickle.dumps(task))
    assert restored.template.key == task.template.key
    assert restored.stage == task.stage
    assert restored.variant_budget == task.variant_budget
    assert restored.screening_cap == task.screening_cap
    assert np.array_equal(restored.dataset.time, task.dataset.time)


# --------------------------------------------------------------------------- #
# Support-gated Stage-2 fit effort (per-member variant budget)
# --------------------------------------------------------------------------- #


def test_stage2_budget_unsupported_family_is_trimmed() -> None:
    # A cheap member whose family reached Stage 2 with no pattern/hint/sniff
    # support runs the reduced ladder.
    assert (
        _stage2_variant_budget(
            is_expensive=False,
            is_peak_seeded=False,
            family_key="muonium",
            supported=False,
        )
        == 2
    )


def test_stage2_budget_supported_family_keeps_full_ladder() -> None:
    # The same family, now positively identified (pattern/hint/sniff), keeps 5.
    assert (
        _stage2_variant_budget(
            is_expensive=False,
            is_peak_seeded=False,
            family_key="muonium",
            supported=True,
        )
        == 5
    )


@pytest.mark.parametrize("family_key", ["relaxation", "baseline"])
def test_stage2_budget_never_trim_families_keep_full_ladder(family_key: str) -> None:
    # Guard-rail families are never trimmed even when unsupported.
    assert (
        _stage2_variant_budget(
            is_expensive=False,
            is_peak_seeded=False,
            family_key=family_key,
            supported=False,
        )
        == 5
    )


def test_stage2_budget_peak_seeded_multiplet_keeps_full_ladder() -> None:
    # Peak-seeded multiplet members keep the full ladder even inside an
    # unsupported family (their frequencies are already tight from detected peaks).
    assert (
        _stage2_variant_budget(
            is_expensive=False,
            is_peak_seeded=True,
            family_key="oscillatory",
            supported=False,
        )
        == 5
    )


def test_stage2_budget_expensive_member_always_reduced() -> None:
    # EXPENSIVE members get the reduced ladder regardless of support / seeding
    # (unchanged from the prior behaviour).
    for supported in (True, False):
        for is_peak_seeded in (True, False):
            assert (
                _stage2_variant_budget(
                    is_expensive=True,
                    is_peak_seeded=is_peak_seeded,
                    family_key="fmuf",
                    supported=supported,
                )
                == 2
            )


def test_execute_assessment_task_caps_only_screening_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Stage-1 screening tasks carry the migrad call cap; Stage-2 tasks (and the
    # uncapped null baselines) run the engine's adaptive drive with no cap, so
    # their final AICc is exact. This pins the "no EXPENSIVE Stage-2 cap" decision.
    captured: dict[str, object] = {}

    def _fake_assess(*_args: object, migrad_ncall: int | None = None, **_kwargs: object):
        captured["migrad_ncall"] = migrad_ncall
        return object()

    monkeypatch.setattr(fit_wizard_module, "_assess_candidate_template", _fake_assess)

    task = _dummy_task("exp_constant")

    # A screening task is call-capped.
    fit_wizard_module._execute_assessment_task(replace(task, screening_cap=True))
    assert captured["migrad_ncall"] == fit_wizard_module._SCREENING_MIGRAD_NCALL

    # A Stage-2 task is not.
    fit_wizard_module._execute_assessment_task(replace(task, screening_cap=False))
    assert captured["migrad_ncall"] is None


def _real_spawn_tasks() -> list[_AssessmentTask]:
    """Two cheap, real (picklable) tasks: flat + exp templates, budget 1.

    Small enough that a real ``spawn`` pool's process-startup cost stays a
    few seconds — this is the one test in the suite that exercises an actual
    subprocess rather than a monkeypatched fan-out mechanic.
    """
    rng = np.random.default_rng(7)
    t = np.linspace(0.1, 6.0, 80)
    y = 0.18 * np.exp(-0.5 * t) + 0.02 + rng.normal(0.0, 0.003, t.size)
    dataset = MuonDataset(
        time=t,
        asymmetry=y,
        error=np.full_like(t, 0.003),
        metadata={"run_number": 1},
    )
    fingerprint = fingerprint_spectrum(dataset)
    flat_template = CandidateTemplate(
        key="null_constant",
        title="Null baseline: constant",
        category="Baseline",
        rationale="",
        model=CompositeModel(["Constant"], operators=[]),
    )
    exp_template = CandidateTemplate(
        key="exp_constant",
        title="Exponential + Constant",
        category="General",
        rationale="",
        model=CompositeModel(["Exponential", "Constant"], operators=["+"]),
    )
    return [
        _AssessmentTask(
            dataset=dataset,
            fingerprint=fingerprint,
            template=template,
            metric=SelectionMetric.AICC,
            seed_context=None,
            variant_budget=1,
            stage=1,
            screening_cap=True,
        )
        for template in (flat_template, exp_template)
    ]


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_run_template_assessments_process_path_matches_serial() -> None:
    """Real spawn pool vs. serial: same template keys/order, metrics agree to 1e-6.

    This is the actual process-vs-serial equivalence check (no monkeypatched
    worker) — it proves ``_run_template_assessments`` gives the same answer
    whichever path executes the fits, not just that the fan-out plumbing works.
    """
    probe_pool = fit_wizard_module.open_spawn_pool(2)
    if probe_pool is None:
        pytest.skip("spawn pool unavailable in this environment.")
    probe_pool.shutdown()

    serial_tasks = _real_spawn_tasks()
    process_tasks = _real_spawn_tasks()

    start = time.monotonic()
    serial_results = _run_template_assessments(serial_tasks, max_workers=1)
    process_results = _run_template_assessments(process_tasks, max_workers=2)
    elapsed = time.monotonic() - start

    serial_keys = [a.template.key for a in serial_results]
    process_keys = [a.template.key for a in process_results]
    assert process_keys == serial_keys

    for serial_assessment, process_assessment in zip(serial_results, process_results):
        assert process_assessment.selected_score == pytest.approx(
            serial_assessment.selected_score, abs=1e-6
        )
        assert process_assessment.fit_result.chi_squared == pytest.approx(
            serial_assessment.fit_result.chi_squared, abs=1e-6
        )

    print(f"\nprocess-equivalence wall time: {elapsed:.2f}s")


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_run_template_assessments_process_path_cancellation_is_prompt() -> None:
    """Real spawn pool: cancel_callback flipping True after one completion aborts."""
    probe_pool = fit_wizard_module.open_spawn_pool(2)
    if probe_pool is None:
        pytest.skip("spawn pool unavailable in this environment.")
    probe_pool.shutdown()

    tasks = _real_spawn_tasks()
    seen = {"n": 0}

    def _cancel_after_first() -> bool:
        seen["n"] += 1
        return seen["n"] > 1

    with pytest.raises(FitCancelledError):
        _run_template_assessments(tasks, max_workers=2, cancel_callback=_cancel_after_first)


# --------------------------------------------------------------------------- #
# _effective_hint_keys
# --------------------------------------------------------------------------- #


def test_effective_hint_keys_kept_when_no_pattern_or_sniff() -> None:
    hints = frozenset({"kt", "multi_rate"})
    assert _effective_hint_keys(hints, frozenset(), frozenset()) == hints


def test_effective_hint_keys_dropped_when_pattern_present() -> None:
    hints = frozenset({"kt", "multi_rate"})
    assert _effective_hint_keys(hints, frozenset({"oscillatory"}), frozenset()) == frozenset()


def test_effective_hint_keys_dropped_when_sniff_present() -> None:
    hints = frozenset({"kt"})
    assert _effective_hint_keys(hints, frozenset(), frozenset({"fmuf"})) == frozenset()


# --------------------------------------------------------------------------- #
# Tiered orchestrator (end-to-end)
# --------------------------------------------------------------------------- #


def _tiered_dataset(
    t: np.ndarray, y: np.ndarray, *, error: float = 0.01, metadata: dict | None = None
) -> MuonDataset:
    payload = {"run_number": 1}
    payload.update(metadata or {})
    return MuonDataset(
        time=np.asarray(t, dtype=float),
        asymmetry=np.asarray(y, dtype=float),
        error=np.full_like(np.asarray(t, dtype=float), error),
        metadata=payload,
    )


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_tiered_flow_screens_all_families_and_reports() -> None:
    rng = np.random.default_rng(21)
    t = np.linspace(0.02, 10.0, 220)
    y = 0.22 * np.exp(-0.8 * t) + 0.03 + rng.normal(0.0, 0.004, t.size)
    dataset = _tiered_dataset(t, y, error=0.004)

    recommendation = build_fit_wizard_recommendation(dataset, max_workers=1)

    report_keys = {report.family_key for report in recommendation.family_reports}
    assert report_keys == {"relaxation", "multi_rate", "kt", "oscillatory", "muonium", "fmuf"}
    assert all(report.reason for report in recommendation.family_reports)
    assert recommendation.peak_analysis is not None
    stage1 = [a for a in recommendation.assessments if a.stage == 1]
    # Every family fitted at least its representative in Stage 1.
    assert len(stage1) >= len(report_keys)
    assert recommendation.recommended_key == "exp_constant"


def _strip_expensive_members(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop EXPENSIVE Stage-2 members (numerical powder averages, strong-collision
    solvers) so end-to-end orchestrator tests stay inside the CI per-test timeout
    (same precedent as the global wizard's template-restriction helpers)."""
    from asymmetry.core.fitting import fit_wizard as fw

    original = fw.build_wizard_families

    def _cheap(*args: object, **kwargs: object) -> tuple:
        families = original(*args, **kwargs)
        threshold = fw._COST_RANK[ComputationalCost.EXPENSIVE]
        return tuple(
            replace(
                family,
                stage2_members=tuple(
                    member
                    for member in family.stage2_members
                    if fw._template_cost_rank(member) < threshold
                ),
            )
            for family in families
        )

    monkeypatch.setattr(fw, "build_wizard_families", _cheap)


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_pattern_promotion_expands_fmuf_family(monkeypatch: pytest.MonkeyPatch) -> None:
    _strip_expensive_members(monkeypatch)
    rng = np.random.default_rng(22)
    t = np.linspace(0.02, 24.0, 480)
    y = 0.25 * linear_fmuf_polarization(t, 1.17) + 0.02
    y = y + rng.normal(0.0, 0.004, t.size)
    dataset = _tiered_dataset(t, y, error=0.004, metadata={"field_direction": "Zero field"})

    recommendation = build_fit_wizard_recommendation(dataset, max_workers=1)

    fmuf_report = next(
        report for report in recommendation.family_reports if report.family_key == "fmuf"
    )
    assert fmuf_report.promoted
    assert fmuf_report.reason.startswith(("pattern match", "best", "residual gates"))
    assert fmuf_report.reason.startswith("pattern match")
    matches = [m for m in recommendation.multiplet_matches if m.kind == "fmuf_linear"]
    assert matches and matches[0].quality > 0.8
    # With EXPENSIVE members stripped for CI, a 3-cosine multiplet seeded at the
    # F-mu-F line frequencies can legitimately out-score the cheap fmuf members,
    # so accept either description of the triplet.
    recommended = recommendation.recommended_key or ""
    assert recommended.startswith(("fmuf", "muf", "dynamic_fmuf", "dipolar", "oscillatory"))


def _exploding_tiered_dataset(signal_fn, *, seed: int, metadata: dict | None = None) -> MuonDataset:
    """Percent-scale record with realistic dying-muon (exploding, capped) errors.

    The envelope matcher's whole point is the data FFT peak detection misses; the
    exploding-error model (σ grows exp with t, capped at 100 %) is where the
    SNR-truncated window and the surrogate-null significance test actually matter.
    """
    t = np.linspace(0.15, 32.6, 2000)
    sigma = np.minimum(0.7 * np.exp(t / (2.0 * 2.2)), 100.0)
    rng = np.random.default_rng(seed)
    payload = {"run_number": 1}
    payload.update(metadata or {})
    return MuonDataset(
        time=t,
        asymmetry=signal_fn(t) + rng.normal(0.0, sigma),
        error=sigma,
        metadata=payload,
    )


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_envelope_matcher_promotes_and_recommends_fmuf_on_exploding_s4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # S4: linear F-mu-F r = 1.17 A under exp(-0.2 t), ZF metadata, exploding
    # errors. FFT peak detection finds no lines; the time-domain matcher must
    # match the fmuf bank (r within 10 %), promote the family via the pattern
    # exemption, and let a fmuf-family template be recommended.
    _strip_expensive_members(monkeypatch)
    dataset = _exploding_tiered_dataset(
        lambda t: 20.0 * np.exp(-0.2 * t) * linear_fmuf_polarization(t, 1.17) + 4.0,
        seed=101,
        metadata={"field_direction": "Zero field"},
    )

    recommendation = build_fit_wizard_recommendation(dataset, max_workers=1)

    envelope = [m for m in recommendation.multiplet_matches if m.kind == "fmuf_envelope"]
    assert envelope, "the fmuf envelope bank should match the F-mu-F signature"
    r = envelope[0].derived("r_muF_angstrom")
    assert r is not None and abs(r - 1.17) <= 0.10 * 1.17

    fmuf_report = next(
        report for report in recommendation.family_reports if report.family_key == "fmuf"
    )
    assert fmuf_report.promoted
    assert fmuf_report.reason.startswith("pattern match")
    recommended = recommendation.recommended_key or ""
    assert recommended.startswith(("fmuf", "muf", "dynamic_fmuf", "dipolar"))


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_envelope_matcher_matches_kt_on_exploding_gkt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Static Gaussian KT, Delta = 0.3, ZF metadata, exploding errors -> kt_envelope
    # match with Delta within ~15 %, and the kt family promoted via the pattern
    # exemption.
    _strip_expensive_members(monkeypatch)
    dataset = _exploding_tiered_dataset(
        lambda t: 20.0 * longitudinal_field_kubo_toyabe(t, 1.0, 0.3, 0.0, 0.0) + 4.0,
        seed=102,
        metadata={"field_direction": "Zero field"},
    )

    recommendation = build_fit_wizard_recommendation(dataset, max_workers=1)

    envelope = [m for m in recommendation.multiplet_matches if m.kind == "kt_envelope"]
    assert envelope, "the KT envelope bank should match the dip + 1/3 tail"
    delta = envelope[0].derived("Delta")
    assert delta is not None and abs(delta - 0.3) <= 0.15 * 0.3
    kt_report = next(
        report for report in recommendation.family_reports if report.family_key == "kt"
    )
    assert kt_report.promoted
    assert kt_report.reason.startswith("pattern match")


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_envelope_matcher_no_match_on_exploding_pure_noise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression control extension: pure noise on the exploding-error model (ZF
    # metadata, so fmuf/kt banks are in scope) must produce NO envelope match.
    _strip_expensive_members(monkeypatch)
    dataset = _exploding_tiered_dataset(
        lambda t: np.full_like(t, 4.0),
        seed=105,
        metadata={"field_direction": "Zero field"},
    )

    recommendation = build_fit_wizard_recommendation(dataset, max_workers=1)

    envelope = [m for m in recommendation.multiplet_matches if "envelope" in m.kind]
    assert envelope == []


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_fluorine_sniff_promotes_fmuf_family(monkeypatch: pytest.MonkeyPatch) -> None:
    # A chemical-formula fluorine in the sample name promotes the fmuf family even
    # when no pattern matches (relaxation-only data). ZF metadata keeps fmuf in
    # scope; the promotion reason names the sniff.
    _strip_expensive_members(monkeypatch)
    rng = np.random.default_rng(31)
    t = np.linspace(0.02, 10.0, 300)
    y = 0.2 * np.exp(-0.5 * t) + 0.02 + rng.normal(0.0, 0.004, t.size)
    dataset = _tiered_dataset(
        t, y, error=0.004, metadata={"field_direction": "Zero field", "sample": "CaF2 powder"}
    )

    recommendation = build_fit_wizard_recommendation(dataset, max_workers=1)

    fmuf_report = next(
        report for report in recommendation.family_reports if report.family_key == "fmuf"
    )
    assert fmuf_report.promoted
    assert "fluorine" in fmuf_report.reason


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_multiplet_templates_generated_for_two_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    _strip_expensive_members(monkeypatch)
    rng = np.random.default_rng(23)
    t = np.linspace(0.0, 16.0, 2048)
    envelope = np.exp(-0.15 * t)
    y = (
        0.15 * np.cos(2.0 * np.pi * 1.3 * t) + 0.10 * np.cos(2.0 * np.pi * 3.7 * t)
    ) * envelope + 0.02
    y = y + rng.normal(0.0, 0.005, t.size)
    dataset = _tiered_dataset(t, y, error=0.005)

    recommendation = build_fit_wizard_recommendation(dataset, max_workers=1)

    assert recommendation.peak_analysis is not None
    assert len(recommendation.peak_analysis.peaks) >= 2
    template_keys = {template.key for template in recommendation.templates}
    assert "oscillatory2_exp_constant" in template_keys


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_scope_restricts_screened_families() -> None:
    rng = np.random.default_rng(24)
    t = np.linspace(0.02, 10.0, 200)
    y = 0.22 * np.exp(-0.8 * t) + 0.03 + rng.normal(0.0, 0.004, t.size)
    dataset = _tiered_dataset(t, y, error=0.004)

    scope = WizardScope(preset=WizardScopePreset.FLUORIDE_FMUF)
    recommendation = build_fit_wizard_recommendation(dataset, scope=scope, max_workers=1)

    report_keys = {report.family_key for report in recommendation.family_reports}
    assert "fmuf" in report_keys
    assert "kt" not in report_keys
    assert "oscillatory" not in report_keys
    assert "muonium" not in report_keys


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_user_frequencies_merge_into_peaks() -> None:
    rng = np.random.default_rng(25)
    t = np.linspace(0.02, 10.0, 200)
    y = 0.22 * np.exp(-0.8 * t) + 0.03 + rng.normal(0.0, 0.004, t.size)
    dataset = _tiered_dataset(t, y, error=0.004)

    recommendation = build_fit_wizard_recommendation(
        dataset, user_frequencies_mhz=[2.5], max_workers=1
    )

    assert recommendation.peak_analysis is not None
    user_peaks = [peak for peak in recommendation.peak_analysis.peaks if peak.source == "user"]
    assert user_peaks
    assert user_peaks[0].frequency_mhz == 2.5


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_empty_scope_reports_no_candidates() -> None:
    t = np.linspace(0.02, 10.0, 50)
    dataset = _tiered_dataset(t, np.exp(-t))
    scope = WizardScope(preset=WizardScopePreset.ALL, exclude_components=frozenset(COMPONENTS))
    recommendation = build_fit_wizard_recommendation(dataset, scope=scope, max_workers=1)
    assert recommendation.recommended_key is None
    assert recommendation.templates == ()
    assert "scope" in recommendation.summary


# --------------------------------------------------------------------------- #
# Regression controls (end-to-end): structureless data must NOT yield a
# confident oscillatory recommendation. Each control is one row of the failure
# taxonomy (F6 / spurious oscillation) and must stay suppressed.
# --------------------------------------------------------------------------- #


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_control_flat_zf_noise_yields_no_significant_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Control (a): Ag-ZF-like flat spectrum + noise. Previously a spurious
    # 0.28 MHz cosine won; the null-baseline test must catch it — no oscillatory
    # model beats the flat null by ΔAICc ≥ 10, so the verdict is null.
    _strip_expensive_members(monkeypatch)
    rng = np.random.default_rng(3)
    t = np.linspace(0.08, 10.0, 500)
    y = 0.05 + rng.normal(0.0, 0.006, t.size)
    dataset = _tiered_dataset(t, y, error=0.006, metadata={"field_direction": "Zero field"})

    recommendation = build_fit_wizard_recommendation(dataset, max_workers=1)

    assert recommendation.verdict is RecommendationVerdict.NO_SIGNIFICANT_STRUCTURE
    assert recommendation.confidence is ConfidenceTier.NONE
    # The recommendation, if any, points at a null baseline — never a confident
    # oscillatory template.
    assert recommendation.recommended_key in ("null_constant", "null_exp")


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_control_pure_noise_yields_no_significant_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Control (c): pure white noise (F6). Must NOT produce a confident (High)
    # structured recommendation — expect "no significant structure".
    _strip_expensive_members(monkeypatch)
    rng = np.random.default_rng(99)
    t = np.linspace(0.1, 12.0, 400)
    y = rng.normal(0.0, 0.01, t.size)
    dataset = _tiered_dataset(t, y, error=0.01)

    recommendation = build_fit_wizard_recommendation(dataset, max_workers=1)

    assert recommendation.confidence is not ConfidenceTier.HIGH
    assert recommendation.verdict is RecommendationVerdict.NO_SIGNIFICANT_STRUCTURE
    assert recommendation.recommended_key in ("null_constant", "null_exp")


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_control_flat_lf_oscillatory_frequency_floor_disqualified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Control (b): flat LF-like data where oscillatory/bessel templates fit with
    # their frequency driven down to the 1/T resolution floor (a smooth envelope
    # masquerading as a barely-one-cycle cosine). The resolution-floor
    # disqualifier must FIRE on every such fit, and none of them may be the
    # confident (High) recommendation.
    _strip_expensive_members(monkeypatch)
    rng = np.random.default_rng(11)
    t = np.linspace(0.05, 8.0, 400)
    # Slowly decaying, essentially non-oscillatory LF-like envelope + noise.
    y = 0.18 * np.exp(-0.05 * t) + 0.02 + rng.normal(0.0, 0.005, t.size)
    dataset = _tiered_dataset(t, y, error=0.005, metadata={"field_direction": "Longitudinal"})

    recommendation = build_fit_wizard_recommendation(dataset, max_workers=1)

    def _frequency_params(assessment: CandidateAssessment) -> list[float]:
        return [
            float(p.value)
            for p in assessment.fit_result.parameters
            if p.name.split("_")[0] == "frequency"
        ]

    floor = 1.0 / float(t.max() - t.min())
    at_floor = [
        assessment
        for assessment in recommendation.assessments
        if assessment.is_successful
        and any(abs(value) <= floor * 1.05 for value in _frequency_params(assessment))
    ]
    # The construction must actually exercise the disqualifier — an oscillatory
    # candidate must have collapsed to the floor for this control to mean anything.
    assert at_floor, "expected an oscillatory candidate to fit at the resolution floor"
    for assessment in at_floor:
        assert any("resolution floor" in reason for reason in assessment.disqualification_reasons)

    # No floor-frequency oscillation is the confident recommendation.
    recommended = recommendation.recommended_assessment
    if recommended is not None and _frequency_params(recommended):
        assert recommendation.confidence is not ConfidenceTier.HIGH


# --------------------------------------------------------------------------- #
# Frequency-support disqualifiers (unit + control)
# --------------------------------------------------------------------------- #


def _exploding_error_dataset_for_support(n: int = 1200) -> MuonDataset:
    """Percent-scale flat record whose σ explodes with t (effective window ≪ full)."""
    t = np.linspace(0.15, 32.6, n)
    sigma = np.minimum(0.7 * np.exp(t / (2.0 * 2.2)), 100.0)
    return MuonDataset(
        time=t, asymmetry=np.full(t.size, 4.0), error=sigma, metadata={"run_number": 1}
    )


def _assessment_with_frequency(frequency: float) -> CandidateAssessment:
    base = _dummy_assessment("oscillatory_exp_constant")
    params = ParameterSet([Parameter(name="frequency", value=frequency)])
    return replace(
        base,
        fit_result=FitResult(
            success=True, chi_squared=1.0, reduced_chi_squared=1.0, parameters=params
        ),
    )


def _peaks_at(*frequencies: float) -> PeakAnalysis:
    return PeakAnalysis(
        peaks=tuple(
            DetectedPeak(
                frequency_mhz=f, amplitude=1.0, snr=5.0, width_mhz=0.1, prominence=1.0, source="fft"
            )
            for f in frequencies
        ),
        noise_floor=1.0,
        resolution_mhz=0.1,
        nyquist_mhz=31.0,
        detrended=False,
    )


def test_frequency_below_effective_window_floor_is_disqualified() -> None:
    # σ explodes with t, so the informative window is ~8 µs of the 32.5 µs
    # record; a 0.1 MHz claim (<1 effective cycle) is unsupportable even though
    # it clears the 1/T_full floor with ~3 cycles.
    dataset = _exploding_error_dataset_for_support()
    (updated,) = _apply_frequency_support_disqualifiers(
        dataset, (_assessment_with_frequency(0.1),), _peaks_at(0.1)
    )
    assert any("informative window" in reason for reason in updated.disqualification_reasons)


def test_resolvable_frequency_with_matching_peak_is_not_disqualified() -> None:
    dataset = _exploding_error_dataset_for_support()
    (updated,) = _apply_frequency_support_disqualifiers(
        dataset, (_assessment_with_frequency(1.30),), _peaks_at(1.32)
    )
    assert updated.disqualification_reasons == ()


def test_resolvable_frequency_without_any_peak_is_disqualified() -> None:
    dataset = _exploding_error_dataset_for_support()
    (updated,) = _apply_frequency_support_disqualifiers(
        dataset, (_assessment_with_frequency(1.30),), _peaks_at()
    )
    assert any("no supporting detected" in r for r in updated.disqualification_reasons)


def test_flat_error_record_keeps_full_window_for_the_floor() -> None:
    # Constant σ ⇒ no truncation ⇒ the effective floor reduces to ~2/T_full;
    # a 0.5 MHz line over 10 µs (5 cycles) with a matching peak passes.
    t = np.linspace(0.1, 10.0, 400)
    dataset = MuonDataset(
        time=t, asymmetry=np.zeros(t.size), error=np.full(t.size, 0.01), metadata={}
    )
    (updated,) = _apply_frequency_support_disqualifiers(
        dataset, (_assessment_with_frequency(0.5),), _peaks_at(0.5)
    )
    assert updated.disqualification_reasons == ()


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_control_kt_plus_drift_suppresses_uncorroborated_cosine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Validation-programme control (S5c-class): static KT + a slow smooth
    # systematic on exploding-error data. Pre-fix, a ~0.08 MHz damped cosine
    # absorbed drift + KT dip and won by AICc at HIGH confidence even though
    # the kt family was correctly promoted. The fitted frequency completes <2
    # cycles inside the SNR-truncated informative window, so the
    # effective-window floor must disqualify every free-frequency oscillator
    # and let the kt family win.
    _strip_expensive_members(monkeypatch)

    def signal(t: np.ndarray) -> np.ndarray:
        kt = longitudinal_field_kubo_toyabe(t, 20.0, 0.3, 0.0, 0.0) + 4.0
        drift = 0.8 * np.sin(2 * np.pi * t / 40.0) + 1.5 * np.exp(-t / 0.4)
        return kt + drift

    dataset = _exploding_tiered_dataset(signal, seed=77, metadata={"field_direction": "Zero field"})

    recommendation = build_fit_wizard_recommendation(dataset, max_workers=1)

    free_frequency_keys = {
        "oscillatory_exp_constant",
        "oscillatory_gaussian_constant",
        "bessel_exp_constant",
    }
    assert recommendation.recommended_key not in free_frequency_keys
    disqualified = [
        assessment
        for assessment in recommendation.assessments
        if assessment.template.key in free_frequency_keys and assessment.is_successful
    ]
    assert disqualified, "expected free-frequency oscillators to have been fitted"
    for assessment in disqualified:
        assert any(
            "informative window" in reason or "no supporting detected" in reason
            for reason in assessment.disqualification_reasons
        )


# --------------------------------------------------------------------------- #
# Promotion decisions (unit)
# --------------------------------------------------------------------------- #


def _scored_assessment(
    key: str, value: float, *, gate: bool = False, success: bool = True
) -> CandidateAssessment:
    template = CandidateTemplate(
        key=key,
        title=key,
        category="General",
        rationale="",
        model=CompositeModel(["Exponential", "Constant"], operators=["+"]),
    )
    fit_result = FitResult(success=success, chi_squared=value, reduced_chi_squared=1.0)
    empty = np.array([], dtype=float)
    return CandidateAssessment(
        template=template,
        fit_result=fit_result,
        aic=value,
        aicc=value,
        bic=value,
        selected_score=value,
        residual_rms=1.0,
        runs_z_score=0.0,
        max_abs_autocorrelation=0.0,
        residual_fft_peak_snr=0.0,
        residual_gate_passed=gate,
        residual_gate_reasons=() if gate else ("standardized residual RMS is high",),
        bound_hits=(),
        fitted_time=empty,
        fitted_curve=empty,
        component_curves=(),
        stage=1,
    )


def _unit_family(key: str) -> WizardFamily:
    return WizardFamily(
        key=key,
        title=key,
        stage1_rep=CandidateTemplate(
            key=f"{key}_rep",
            title=key,
            category="General",
            rationale="",
            model=CompositeModel(["Exponential", "Constant"], operators=["+"]),
        ),
        stage2_members=(),
    )


def test_promotion_margin_boundary() -> None:
    families = [_unit_family(k) for k in ("a", "b", "c")]
    reps = [
        _scored_assessment("a_rep", 100.0),
        _scored_assessment("b_rep", 109.0),
        _scored_assessment("c_rep", 111.0),
    ]
    decisions = _decide_family_promotions(families, reps, frozenset(), SelectionMetric.AICC)
    by_key = {family.key: (promoted, reason) for family, _a, promoted, reason in decisions}
    assert by_key["a"][0] is True  # best
    assert by_key["b"][0] is True  # within delta 10
    assert by_key["c"][0] is False
    assert "not promoted" in by_key["c"][1]


def test_promotion_gate_beats_delta() -> None:
    families = [_unit_family(k) for k in ("a", "b")]
    reps = [
        _scored_assessment("a_rep", 100.0),
        _scored_assessment("b_rep", 500.0, gate=True),
    ]
    decisions = _decide_family_promotions(families, reps, frozenset(), SelectionMetric.AICC)
    by_key = {family.key: reason for family, _a, promoted, reason in decisions if promoted}
    assert "b" in by_key
    assert "gates" in by_key["b"]


def test_promotion_cap_and_pattern_exemption() -> None:
    keys = ["f1", "f2", "f3", "f4", "f5", "f6", "fmuf"]
    families = [_unit_family(k) for k in keys]
    reps = [_scored_assessment(f"{k}_rep", 100.0 + i, gate=True) for i, k in enumerate(keys)]
    # fmuf scores worst but is pattern-matched.
    reps[-1] = _scored_assessment("fmuf_rep", 1000.0)
    decisions = _decide_family_promotions(families, reps, frozenset({"fmuf"}), SelectionMetric.AICC)
    promoted = {family.key for family, _a, ok, _r in decisions if ok}
    assert "fmuf" in promoted
    assert len(promoted - {"fmuf"}) == 4  # score promotions capped
    demoted_reasons = [r for family, _a, ok, r in decisions if not ok]
    assert any("cap" in reason for reason in demoted_reasons)


def test_promotion_hint_rescues_shape_mismatched_family() -> None:
    families = [_unit_family(k) for k in ("a", "kt")]
    reps = [
        _scored_assessment("a_rep", 100.0),
        _scored_assessment("kt_rep", 500.0),
    ]
    decisions = _decide_family_promotions(
        families,
        reps,
        frozenset(),
        SelectionMetric.AICC,
        hint_family_keys=frozenset({"kt"}),
    )
    by_key = {family.key: (ok, reason) for family, _a, ok, reason in decisions}
    assert by_key["kt"][0] is True
    assert "hint" in by_key["kt"][1]


def test_promotion_fluorine_sniff_exempt_from_cap() -> None:
    # The fluorine sniff promotes fmuf even when it scores worst and the score
    # promotions are already at the cap — same exemption class as a pattern match.
    keys = ["f1", "f2", "f3", "f4", "f5", "f6", "fmuf"]
    families = [_unit_family(k) for k in keys]
    reps = [_scored_assessment(f"{k}_rep", 100.0 + i, gate=True) for i, k in enumerate(keys)]
    reps[-1] = _scored_assessment("fmuf_rep", 1000.0)  # fmuf scores worst
    decisions = _decide_family_promotions(
        families,
        reps,
        frozenset(),
        SelectionMetric.AICC,
        sniff_family_keys=frozenset({"fmuf"}),
    )
    by_key = {family.key: (ok, reason) for family, _a, ok, reason in decisions}
    assert by_key["fmuf"][0] is True
    assert "fluorine" in by_key["fmuf"][1]
    promoted = {key for key, (ok, _r) in by_key.items() if ok}
    assert len(promoted - {"fmuf"}) == 4  # score promotions still capped at 4


def _fmuf_rep_template() -> CandidateTemplate:
    return CandidateTemplate(
        key="fmuf_linear_exp_constant",
        title="F-mu-F",
        category="Nuclear dipolar",
        rationale="",
        model=CompositeModel(["FmuF_Linear", "Exponential", "Constant"], operators=["*", "+"]),
    )


def test_fmuf_r_ladder_spans_the_physical_range() -> None:
    # Belt-and-braces: independent of any match, the fmuf rep's variants must step
    # r_muF across the physical ladder rather than sit at one default that falls
    # into the ~0.53 A minimum.
    dataset = _tiered_dataset(
        np.linspace(0.02, 24.0, 400),
        0.2 * linear_fmuf_polarization(np.linspace(0.02, 24.0, 400), 1.17) + 0.02,
        metadata={"field_direction": "Zero field"},
    )
    template = _fmuf_rep_template()
    from asymmetry.core.fitting.fit_wizard import fingerprint_spectrum

    base = _initial_parameters_for_template(dataset, fingerprint_spectrum(dataset), template)
    variants = _parameter_variants(base, template=template, variant_budget=3)
    assert len(variants) == 3
    seeds = [next(p.value for p in variant if p.name.split("_")[0] == "r") for variant in variants]
    # First rung keeps the base seed (default 1.17); the rest bracket it.
    assert seeds[0] == pytest.approx(1.17, abs=1e-6)
    assert set(round(s, 2) for s in seeds) == {round(r, 2) for r in _FMUF_R_LADDER}


def test_fmuf_r_ladder_first_rung_honours_match_derived_r() -> None:
    # When a match supplies r_muF (via seed context), it seeds the base params and
    # thus becomes the first ladder rung.
    from asymmetry.core.fitting.fit_wizard import TemplateSeedContext, fingerprint_spectrum

    dataset = _tiered_dataset(
        np.linspace(0.02, 24.0, 400),
        0.2 * linear_fmuf_polarization(np.linspace(0.02, 24.0, 400), 1.24) + 0.02,
        metadata={"field_direction": "Zero field"},
    )
    template = _fmuf_rep_template()
    match = MultipletMatch(
        kind="fmuf_envelope",
        family_key="fmuf",
        peak_indices=(),
        quality=0.9,
        derived_values=(("r_muF_angstrom", 1.24),),
        note="",
    )
    ctx = TemplateSeedContext(multiplet_matches=(match,))
    base = _initial_parameters_for_template(
        dataset, fingerprint_spectrum(dataset), template, seed_context=ctx
    )
    variants = _fmuf_r_ladder_variants(base)
    first_r = next(p.value for p in variants[0] if p.name.split("_")[0] == "r")
    assert first_r == pytest.approx(1.24, abs=1e-6)


def test_promotion_failed_rep_not_promoted() -> None:
    families = [_unit_family(k) for k in ("a", "b")]
    reps = [
        _scored_assessment("a_rep", 100.0),
        _scored_assessment("b_rep", 90.0, success=False),
    ]
    decisions = _decide_family_promotions(families, reps, frozenset(), SelectionMetric.AICC)
    by_key = {family.key: (ok, reason) for family, _a, ok, reason in decisions}
    assert by_key["b"][0] is False
    assert "failed" in by_key["b"][1]


def test_cancel_callback_aborts_analysis() -> None:
    from asymmetry.core.fitting.engine import FitCancelledError

    t = np.linspace(0.02, 10.0, 120)
    dataset = _tiered_dataset(t, 0.2 * np.exp(-0.8 * t) + 0.02)
    with pytest.raises(FitCancelledError):
        build_fit_wizard_recommendation(dataset, max_workers=1, cancel_callback=lambda: True)


# --------------------------------------------------------------------------- #
# Recommendation policy: confidence tiers, null baselines, disqualifiers.
#
# These exercise ``rerank_fit_wizard_recommendation`` — the single policy
# engine — on hand-built assessments, so they run without the expensive
# orchestrator and stay in the fast (non-integration) tier.
# --------------------------------------------------------------------------- #


def _policy_assessment(
    key: str,
    value: float,
    *,
    parameters: ParameterSet | None = None,
    param_count: int | None = None,
    uncertainties: dict[str, float] | None = None,
    gate: bool = True,
    success: bool = True,
    disqualified: tuple[str, ...] = (),
    null: bool = False,
    model: CompositeModel | None = None,
) -> CandidateAssessment:
    """Build a fully-formed assessment for policy-engine tests.

    ``value`` is used as the AICc/AIC/BIC so metric ranking is deterministic.
    ``param_count`` (via a padded ParameterSet) drives the strictly-simpler null
    comparison; ``gate`` maps to ``residual_gate_passed`` (High vs Medium tier).
    """
    if parameters is None:
        parameters = ParameterSet()
        for i in range(param_count if param_count is not None else 1):
            parameters.add(Parameter(name=f"p{i}", value=1.0))
    fit_result = FitResult(
        success=success,
        chi_squared=value,
        reduced_chi_squared=1.0,
        parameters=parameters,
        uncertainties=uncertainties or {},
    )
    empty = np.array([], dtype=float)
    return CandidateAssessment(
        template=CandidateTemplate(
            key=key,
            title=key,
            category="Baseline" if null else "General",
            rationale="",
            model=model or CompositeModel(["Exponential", "Constant"], operators=["+"]),
        ),
        fit_result=fit_result,
        aic=value,
        aicc=value,
        bic=value,
        selected_score=value,
        residual_rms=1.0,
        runs_z_score=0.0 if gate else 5.0,
        max_abs_autocorrelation=0.0,
        residual_fft_peak_snr=0.0,
        residual_gate_passed=gate,
        residual_gate_reasons=() if gate else ("runs-test z score suggests structure (5.00)",),
        bound_hits=(),
        fitted_time=empty,
        fitted_curve=empty,
        component_curves=(),
        stage=2,
        disqualification_reasons=disqualified,
        is_null_baseline=null,
    )


def _policy_recommendation(*assessments: CandidateAssessment) -> FitWizardRecommendation:
    return FitWizardRecommendation(
        fingerprint=_plain_fingerprint(),
        templates=tuple(a.template for a in assessments),
        assessments=tuple(assessments),
        metric=SelectionMetric.AICC,
        recommended_key=None,
        comparable_keys=(),
        summary="",
    )


def test_policy_recommends_best_metric_even_when_gates_fail() -> None:
    # F1 fix: gate failure no longer vetoes. A gate-failing metric winner is
    # still recommended, at Medium confidence, with the gate reasons as caveat.
    rec = _policy_recommendation(
        _policy_assessment("winner", 100.0, gate=False, param_count=4),
        _policy_assessment("null_constant", 400.0, param_count=1, null=True),
        _policy_assessment("null_exp", 250.0, param_count=3, null=True),
    )
    out = rerank_fit_wizard_recommendation(rec, SelectionMetric.AICC)
    assert out.recommended_key == "winner"
    assert out.confidence is ConfidenceTier.MEDIUM
    assert out.verdict is RecommendationVerdict.STRUCTURED
    assert "residual" in out.caveat.lower() or "structured" in out.caveat.lower()


def test_policy_high_confidence_when_gates_pass() -> None:
    rec = _policy_recommendation(
        _policy_assessment("winner", 100.0, gate=True, param_count=4),
        _policy_assessment("null_constant", 400.0, param_count=1, null=True),
    )
    out = rerank_fit_wizard_recommendation(rec, SelectionMetric.AICC)
    assert out.recommended_key == "winner"
    assert out.confidence is ConfidenceTier.HIGH
    assert out.verdict is RecommendationVerdict.STRUCTURED
    assert out.caveat == ""


def test_policy_null_verdict_when_winner_ties_simpler_null() -> None:
    # Control (c): pure noise. A 4-param winner that barely beats the flat null
    # (Δ < 10) must fall back to "no significant structure" -> null, not High.
    rec = _policy_recommendation(
        _policy_assessment("winner", 395.0, gate=True, param_count=4),
        _policy_assessment("null_constant", 400.0, param_count=1, null=True),
        _policy_assessment("null_exp", 402.0, param_count=3, null=True),
    )
    out = rerank_fit_wizard_recommendation(rec, SelectionMetric.AICC)
    assert out.recommended_key == "null_constant"
    assert out.confidence is ConfidenceTier.NONE
    assert out.verdict is RecommendationVerdict.NO_SIGNIFICANT_STRUCTURE
    assert "no significant structure" in out.caveat.lower()


def test_policy_exponential_need_not_beat_equal_complexity_null() -> None:
    # The strictly-simpler rule: an exp candidate (3 params) that equals the
    # exp null (3 params) but beats the flat null (1 param) IS structured — it
    # must not be vetoed by the equal-complexity null.
    rec = _policy_recommendation(
        _policy_assessment("exp_constant", 300.0, gate=True, param_count=3),
        _policy_assessment("null_constant", 400.0, param_count=1, null=True),
        _policy_assessment("null_exp", 300.5, param_count=3, null=True),
    )
    out = rerank_fit_wizard_recommendation(rec, SelectionMetric.AICC)
    assert out.recommended_key == "exp_constant"
    assert out.verdict is RecommendationVerdict.STRUCTURED


def test_policy_disqualified_candidate_drops_to_next_survivor() -> None:
    # Control (a)/(b): a spurious oscillation at the resolution floor is
    # disqualified even though it wins by metric, dropping to the next model.
    rec = _policy_recommendation(
        _policy_assessment(
            "spurious_cosine",
            100.0,
            gate=True,
            param_count=4,
            disqualified=("frequency at the 1/T resolution floor (0.28 ≤ 0.30 MHz)",),
        ),
        _policy_assessment("exp_constant", 150.0, gate=True, param_count=3),
        _policy_assessment("null_constant", 400.0, param_count=1, null=True),
    )
    out = rerank_fit_wizard_recommendation(rec, SelectionMetric.AICC)
    assert out.recommended_key == "exp_constant"
    assert out.verdict is RecommendationVerdict.STRUCTURED


def test_policy_all_disqualified_falls_back_to_null() -> None:
    rec = _policy_recommendation(
        _policy_assessment(
            "spurious_cosine",
            100.0,
            gate=True,
            param_count=4,
            disqualified=("oscillation amplitude A consistent with zero",),
        ),
        _policy_assessment("null_constant", 400.0, param_count=1, null=True),
    )
    out = rerank_fit_wizard_recommendation(rec, SelectionMetric.AICC)
    assert out.recommended_key == "null_constant"
    assert out.verdict is RecommendationVerdict.NO_SIGNIFICANT_STRUCTURE
    assert out.confidence is ConfidenceTier.NONE


def test_policy_tolerates_missing_nulls() -> None:
    # The explicit-template path and old payloads carry no nulls; the null test
    # is skipped and a normal structured recommendation is returned.
    rec = _policy_recommendation(
        _policy_assessment("exp_constant", 100.0, gate=True, param_count=3),
    )
    out = rerank_fit_wizard_recommendation(rec, SelectionMetric.AICC)
    assert out.recommended_key == "exp_constant"
    assert out.verdict is RecommendationVerdict.STRUCTURED
    assert out.confidence is ConfidenceTier.HIGH


def test_policy_no_successful_candidate_yields_none_verdict() -> None:
    rec = _policy_recommendation(
        _policy_assessment("failed", 100.0, success=False, param_count=3),
    )
    out = rerank_fit_wizard_recommendation(rec, SelectionMetric.AICC)
    assert out.recommended_key is None
    assert out.verdict is RecommendationVerdict.NONE
    assert out.confidence is ConfidenceTier.NONE


def test_policy_holds_when_reranking_by_a_different_metric() -> None:
    # Requirement 6: rerank by BIC (not the AICc the recommendation was built
    # with) must recompute tier/verdict coherently. Disqualifiers are stored on
    # the assessment (metric-independent) and still suppress; the null-baseline
    # verdict is re-derived against the BIC-best strictly-simpler null.
    rec = _policy_recommendation(
        _policy_assessment(
            "spurious_cosine",
            50.0,
            gate=True,
            param_count=4,
            disqualified=("frequency at the 1/T resolution floor",),
        ),
        _policy_assessment("exp_constant", 120.0, gate=False, param_count=3),
        _policy_assessment("null_constant", 400.0, param_count=1, null=True),
    )
    out = rerank_fit_wizard_recommendation(rec, SelectionMetric.BIC)
    assert out.metric is SelectionMetric.BIC
    # The disqualified metric winner is skipped; exp_constant survives (Medium,
    # since its gates fail) and beats the flat null by ΔBIC = 280 ≥ 10.
    assert out.recommended_key == "exp_constant"
    assert out.confidence is ConfidenceTier.MEDIUM
    assert out.verdict is RecommendationVerdict.STRUCTURED


def _family_report(
    family_key: str,
    stage1_key: str,
    *,
    stage2_keys: tuple[str, ...] = (),
) -> FamilyScreeningReport:
    return FamilyScreeningReport(
        family_key=family_key,
        title=family_key,
        stage1_template_key=stage1_key,
        stage1_metric_value=0.0,
        stage1_gate_passed=True,
        promoted=True,
        reason="",
        stage2_template_keys=stage2_keys,
    )


def test_tie_break_same_family_still_prefers_simpler() -> None:
    # (a) Both templates screened under the same family: the ±2-AICc
    # simpler-preferred swap still applies, as before this change.
    rec = _policy_recommendation(
        _policy_assessment("biexp_constant", 100.0, gate=True, param_count=5),
        _policy_assessment("risch_kehr_constant", 101.0, gate=True, param_count=3),
        _policy_assessment("null_constant", 400.0, param_count=1, null=True),
    )
    rec = replace(
        rec,
        family_reports=(
            _family_report(
                "multi_rate",
                "biexp_constant",
                stage2_keys=("risch_kehr_constant",),
            ),
        ),
    )
    out = rerank_fit_wizard_recommendation(rec, SelectionMetric.AICC)
    assert out.recommended_key == "risch_kehr_constant"
    assert set(out.comparable_keys) == {"biexp_constant", "risch_kehr_constant"}


def test_tie_break_cross_family_keeps_metric_winner_primary() -> None:
    # (b) The S7 flip regression: stretched_constant (relaxation) is the
    # metric winner within 2 AICc of risch_kehr_constant (multi_rate) — a
    # different family. The metric winner must stay primary; the simpler
    # cross-family template only appears in comparable_keys.
    rec = _policy_recommendation(
        _policy_assessment("stretched_constant", 100.0, gate=True, param_count=4),
        _policy_assessment("risch_kehr_constant", 101.0, gate=True, param_count=3),
        _policy_assessment("null_constant", 400.0, param_count=1, null=True),
    )
    rec = replace(
        rec,
        family_reports=(
            _family_report("relaxation", "exp_constant", stage2_keys=("stretched_constant",)),
            _family_report("multi_rate", "biexp_constant", stage2_keys=("risch_kehr_constant",)),
        ),
    )
    out = rerank_fit_wizard_recommendation(rec, SelectionMetric.AICC)
    assert out.recommended_key == "stretched_constant"
    assert set(out.comparable_keys) == {"stretched_constant", "risch_kehr_constant"}

    # (h, tie-break half) Reranking the same payload by BIC must not change
    # the cross-family behavior — the metric winner (by BIC score) still
    # stays primary rather than flipping to the simpler template.
    out_bic = rerank_fit_wizard_recommendation(rec, SelectionMetric.BIC)
    assert out_bic.recommended_key == "stretched_constant"
    assert set(out_bic.comparable_keys) == {"stretched_constant", "risch_kehr_constant"}


def test_tie_break_old_payload_without_family_reports_does_not_flip() -> None:
    # (c) No family_reports at all (old payload / explicit-template path):
    # every key is "unknown family", so the swap never fires and the metric
    # winner stays primary — the safe direction.
    rec = _policy_recommendation(
        _policy_assessment("stretched_constant", 100.0, gate=True, param_count=4),
        _policy_assessment("risch_kehr_constant", 101.0, gate=True, param_count=3),
        _policy_assessment("null_constant", 400.0, param_count=1, null=True),
    )
    assert rec.family_reports == ()
    out = rerank_fit_wizard_recommendation(rec, SelectionMetric.AICC)
    assert out.recommended_key == "stretched_constant"
    assert set(out.comparable_keys) == {"stretched_constant", "risch_kehr_constant"}


def _oscillatory_assessment(
    key: str,
    value: float,
    *,
    frequency: float,
    param_count: int = 4,
    gate: bool = True,
) -> CandidateAssessment:
    parameters = ParameterSet()
    parameters.add(Parameter(name="frequency", value=frequency))
    for i in range(max(param_count - 1, 0)):
        parameters.add(Parameter(name=f"p{i}", value=1.0))
    return _policy_assessment(
        key,
        value,
        parameters=parameters,
        gate=gate,
        model=CompositeModel(["Oscillatory", "Exponential", "Constant"], operators=["*", "+"]),
    )


def _peak_analysis_with(frequency_mhz: float, snr: float, *, resolution_mhz: float) -> PeakAnalysis:
    return PeakAnalysis(
        peaks=(
            DetectedPeak(
                frequency_mhz=frequency_mhz,
                amplitude=1.0,
                snr=snr,
                width_mhz=resolution_mhz,
                prominence=1.0,
                source="fft",
            ),
        ),
        noise_floor=0.1,
        resolution_mhz=resolution_mhz,
        nyquist_mhz=50.0,
        detrended=False,
    )


def test_edge_caveat_fires_at_ag_candlestick_anchor() -> None:
    # (d) Ag-candlestick real-data calibration anchor: ~2.25 effective cycles,
    # peak SNR 4.4 -> caveat fires.
    resolution_mhz = 1.0
    frequency = 2.25
    winner = _oscillatory_assessment("oscillatory_exp_constant", 100.0, frequency=frequency)
    rec = replace(
        _policy_recommendation(winner),
        peak_analysis=_peak_analysis_with(frequency, 4.4, resolution_mhz=resolution_mhz),
    )
    out = rerank_fit_wizard_recommendation(rec, SelectionMetric.AICC)
    assert out.recommended_key == "oscillatory_exp_constant"
    assert "edge" not in out.caveat.lower()  # sanity: caveat text doesn't use that word
    assert "spectral support is weak" in out.caveat.lower()
    assert "2.2" in out.caveat or "2.3" in out.caveat
    # This is the HIGH-confidence path (gate passes -> empty base caveat), so
    # the edge caveat alone determines the final string: pin exact casing
    # (regression guard — a naive ``str.capitalize()`` would lowercase "MHz"
    # and "SNR" while upcasing the leading letter).
    assert "MHz" in out.caveat
    assert "SNR" in out.caveat
    assert out.caveat[0].isupper()


def test_edge_caveat_absent_at_s1_evidence_anchor() -> None:
    # (e) S1 Larmor evidence calibration anchor: ~2.1 cycles but SNR ~37 ->
    # the strong spectral line saves it, no caveat.
    resolution_mhz = 1.0
    frequency = 2.1
    winner = _oscillatory_assessment("oscillatory_exp_constant", 100.0, frequency=frequency)
    rec = replace(
        _policy_recommendation(winner),
        peak_analysis=_peak_analysis_with(frequency, 37.0, resolution_mhz=resolution_mhz),
    )
    out = rerank_fit_wizard_recommendation(rec, SelectionMetric.AICC)
    assert out.recommended_key == "oscillatory_exp_constant"
    assert "spectral support is weak" not in out.caveat.lower()


def test_edge_caveat_absent_well_inside_window_even_with_weak_snr() -> None:
    # (f) 4 effective cycles is well past the edge band -> no caveat even
    # though the peak SNR (4.0) would count as weak.
    resolution_mhz = 1.0
    frequency = 4.0
    winner = _oscillatory_assessment("oscillatory_exp_constant", 100.0, frequency=frequency)
    rec = replace(
        _policy_recommendation(winner),
        peak_analysis=_peak_analysis_with(frequency, 4.0, resolution_mhz=resolution_mhz),
    )
    out = rerank_fit_wizard_recommendation(rec, SelectionMetric.AICC)
    assert out.recommended_key == "oscillatory_exp_constant"
    assert "spectral support is weak" not in out.caveat.lower()


def test_edge_caveat_appends_to_existing_medium_caveat() -> None:
    # (g) The residual-gate Medium caveat and the edge-of-window caveat must
    # both be visible — appended, never replacing one another.
    resolution_mhz = 1.0
    frequency = 2.25
    winner = _oscillatory_assessment(
        "oscillatory_exp_constant", 100.0, frequency=frequency, gate=False
    )
    rec = replace(
        _policy_recommendation(winner),
        peak_analysis=_peak_analysis_with(frequency, 4.4, resolution_mhz=resolution_mhz),
    )
    out = rerank_fit_wizard_recommendation(rec, SelectionMetric.AICC)
    assert out.confidence is ConfidenceTier.MEDIUM
    assert "structured residuals" in out.caveat.lower()
    assert "spectral support is weak" in out.caveat.lower()


def test_edge_caveat_behavior_identical_when_reranking_by_bic() -> None:
    # (h) Rerank by BIC exercises the same caveat path as AICc; the trigger is
    # metric-independent (it only reads peak_analysis + fitted frequency).
    resolution_mhz = 1.0
    frequency = 2.25
    winner = _oscillatory_assessment("oscillatory_exp_constant", 100.0, frequency=frequency)
    rec = replace(
        _policy_recommendation(winner),
        peak_analysis=_peak_analysis_with(frequency, 4.4, resolution_mhz=resolution_mhz),
    )
    out = rerank_fit_wizard_recommendation(rec, SelectionMetric.BIC)
    assert out.metric is SelectionMetric.BIC
    assert out.recommended_key == "oscillatory_exp_constant"
    assert "spectral support is weak" in out.caveat.lower()


def test_null_baseline_templates_are_simple_and_tagged() -> None:
    nulls = build_null_baseline_templates()
    keys = {t.key for t in nulls}
    assert keys == {"null_constant", "null_exp"}
    by_key = {t.key: t for t in nulls}
    assert by_key["null_constant"].parameter_count == 1
    # Constant is strictly simpler than exp+constant.
    assert by_key["null_constant"].parameter_count < by_key["null_exp"].parameter_count


# --------------------------------------------------------------------------- #
# Frequency-floor / zero-amplitude disqualifiers (unit).
# --------------------------------------------------------------------------- #


def test_disqualifier_flags_frequency_at_resolution_floor() -> None:
    from asymmetry.core.fitting.fit_wizard import _disqualification_reasons

    # Window T = 10 µs -> 1/T = 0.1 MHz floor. A 0.05 MHz cosine is below it.
    t = np.linspace(0.0, 10.0, 200)
    dataset = _tiered_dataset(t, np.zeros_like(t))
    params = ParameterSet()
    params.add(Parameter(name="A", value=0.2))
    params.add(Parameter(name="frequency", value=0.05))
    params.add(Parameter(name="phase", value=0.0))
    fit_result = FitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        parameters=params,
        uncertainties={"A": 0.01},
    )
    template = CandidateTemplate(
        key="oscillatory_exp_constant",
        title="osc",
        category="Oscillatory",
        rationale="",
        model=CompositeModel(["Oscillatory", "Constant"], operators=["+"]),
    )
    reasons = _disqualification_reasons(dataset, template, fit_result, bound_hits=[])
    assert any("resolution floor" in reason for reason in reasons)


def test_disqualifier_flags_zero_consistent_amplitude() -> None:
    from asymmetry.core.fitting.fit_wizard import _disqualification_reasons

    t = np.linspace(0.0, 10.0, 200)
    dataset = _tiered_dataset(t, np.zeros_like(t))
    params = ParameterSet()
    params.add(Parameter(name="A", value=0.01))  # tiny amplitude
    params.add(Parameter(name="frequency", value=1.5))  # well above the floor
    params.add(Parameter(name="phase", value=0.0))
    fit_result = FitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        parameters=params,
        uncertainties={"A": 0.02},  # |A| = 0.01 < 2*0.02 -> consistent with zero
    )
    template = CandidateTemplate(
        key="oscillatory_exp_constant",
        title="osc",
        category="Oscillatory",
        rationale="",
        model=CompositeModel(["Oscillatory", "Constant"], operators=["+"]),
    )
    reasons = _disqualification_reasons(dataset, template, fit_result, bound_hits=[])
    assert any("consistent with zero" in reason for reason in reasons)


def test_disqualifier_skips_amplitude_when_error_missing() -> None:
    from asymmetry.core.fitting.fit_wizard import _disqualification_reasons

    t = np.linspace(0.0, 10.0, 200)
    dataset = _tiered_dataset(t, np.zeros_like(t))
    params = ParameterSet()
    params.add(Parameter(name="A", value=0.01))
    params.add(Parameter(name="frequency", value=1.5))
    fit_result = FitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        parameters=params,
        uncertainties={},  # no error -> do not suppress on unknown uncertainty
    )
    template = CandidateTemplate(
        key="oscillatory_exp_constant",
        title="osc",
        category="Oscillatory",
        rationale="",
        model=CompositeModel(["Oscillatory", "Constant"], operators=["+"]),
    )
    reasons = _disqualification_reasons(dataset, template, fit_result, bound_hits=[])
    assert not any("consistent with zero" in reason for reason in reasons)


def test_disqualifier_ignores_non_oscillatory_templates() -> None:
    from asymmetry.core.fitting.fit_wizard import _disqualification_reasons

    t = np.linspace(0.0, 10.0, 200)
    dataset = _tiered_dataset(t, np.zeros_like(t))
    params = ParameterSet()
    params.add(Parameter(name="A_1", value=0.001))  # tiny, but no frequency here
    params.add(Parameter(name="Lambda", value=0.4))
    fit_result = FitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        parameters=params,
        uncertainties={"A_1": 0.02},
    )
    template = CandidateTemplate(
        key="exp_constant",
        title="exp",
        category="Relaxation",
        rationale="",
        model=CompositeModel(["Exponential", "Constant"], operators=["+"]),
    )
    reasons = _disqualification_reasons(dataset, template, fit_result, bound_hits=[])
    assert reasons == []


# --------------------------------------------------------------------------- #
# Serialization round-trip of the new additive policy fields.
# --------------------------------------------------------------------------- #


def test_policy_fields_round_trip_through_serialization() -> None:
    winner = _policy_assessment(
        "winner",
        100.0,
        gate=False,
        param_count=4,
        disqualified=("frequency at the 1/T resolution floor",),
    )
    null = _policy_assessment("null_constant", 400.0, param_count=1, null=True)
    rec = replace(
        _policy_recommendation(winner, null),
        confidence=ConfidenceTier.MEDIUM,
        verdict=RecommendationVerdict.STRUCTURED,
        caveat="structured residuals remain",
    )
    restored = deserialize_fit_wizard_recommendation(serialize_fit_wizard_recommendation(rec))
    assert restored is not None
    assert restored.confidence is ConfidenceTier.MEDIUM
    assert restored.verdict is RecommendationVerdict.STRUCTURED
    assert restored.caveat == "structured residuals remain"
    restored_winner = restored.assessment_for_key("winner")
    assert restored_winner is not None
    assert restored_winner.disqualification_reasons == ("frequency at the 1/T resolution floor",)
    restored_null = restored.assessment_for_key("null_constant")
    assert restored_null is not None
    assert restored_null.is_null_baseline is True


def test_old_payload_without_policy_fields_deserializes_with_defaults() -> None:
    # Tolerant deserializer: a payload predating the new fields loads with
    # sensible defaults (NONE confidence, STRUCTURED verdict, empty caveat).
    payload = serialize_fit_wizard_recommendation(
        _policy_recommendation(_policy_assessment("exp_constant", 100.0, param_count=3))
    )
    payload.pop("confidence", None)
    payload.pop("verdict", None)
    payload.pop("caveat", None)
    for assessment in payload.get("assessments", []):
        assessment.pop("disqualification_reasons", None)
        assessment.pop("is_null_baseline", None)
    restored = deserialize_fit_wizard_recommendation(payload)
    assert restored is not None
    assert restored.confidence is ConfidenceTier.NONE
    assert restored.verdict is RecommendationVerdict.STRUCTURED
    assert restored.caveat == ""
    restored_exp = restored.assessment_for_key("exp_constant")
    assert restored_exp is not None
    assert restored_exp.disqualification_reasons == ()
    assert restored_exp.is_null_baseline is False

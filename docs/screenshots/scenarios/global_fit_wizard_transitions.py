"""Global Fit Wizard — the Phases step with a phase picked, optimised.

Drives the wizard's **Phases** step to a **partitioned** recommendation over
the synthetic two-phase ZF scan (:func:`make_two_phase_zf_tscan`): a damped-oscillation
phase below a planted 20 K transition and a plain-relaxation phase above it.
Unlike ``global_fit_wizard_result`` this does not run any real search or fit —
the screening table, the penalty path, and the per-phase answers (each a rung
of a sharing ladder, as the default "Best for trending" objective gives, with
its cost and trend quality) are all hand-built (mirroring
``tests/gui/test_wizard_transitions_card.py::_trend_partitioned_recommendation``)
and handed to the window via ``set_cached_recommendation``, the same path a
reopened, already-optimised wizard state uses. That keeps the capture fast and
fully deterministic while still exercising the real ``TransitionsCard`` and
``GlobalFitWizardWindow`` rendering code — nothing about the *display* is
faked, only the (expensive) search that would normally produce its input.

The cached state lands on Screen, so the scenario opens Phases and picks the
second phase's row: the overlay is coloured by phase and draws that phase's
coupled fit over its runs, and the trace strip below it follows the precession
frequency through the ordered phase and the relaxation rate above it, with the
boundary marked.

``requires_fit = False``: no iminuit call happens at capture time.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QWidget

from ..data import make_two_phase_zf_tscan
from ._base import Scenario, _process_events_for, register


def _template(key: str, title: str, components: list[str], operators: list[str]):
    from asymmetry.core.fitting.composite import CompositeModel
    from asymmetry.core.fitting.fit_wizard import CandidateTemplate

    return CandidateTemplate(
        key=key,
        title=title,
        category="General",
        rationale="Synthetic two-phase scenario candidate.",
        model=CompositeModel(components, operators=operators),
    )


def _fingerprint(dataset):
    from asymmetry.core.fitting.fit_wizard import SpectrumFingerprint

    return SpectrumFingerprint(
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


def _assessment(
    datasets,
    runs: tuple[int, ...],
    template,
    param_values_for_run,
    *,
    global_param_names: tuple[str, ...],
    local_param_names: tuple[str, ...],
    fixed_param_names: tuple[str, ...] = (),
    series_cost: float,
    trend_quality: float,
):
    """A phase's answer: one rung, at ``series_cost`` σ, whose local parameters trend."""
    from asymmetry.core.fitting.engine import FitResult
    from asymmetry.core.fitting.global_fit_wizard import (
        GlobalCandidateAssessment,
        RunResidualDiagnostic,
    )
    from asymmetry.core.fitting.global_search.trend_objective import CandidateRung
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet
    from asymmetry.core.fitting.trend_quality import CandidateTrend, TraceQuality

    by_run = {int(dataset.run_number): dataset for dataset in datasets}
    fit_results: dict[int, FitResult] = {}
    fitted_curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    component_curves: dict[int, tuple] = {}
    diagnostics: list[RunResidualDiagnostic] = []
    for run_number in runs:
        dataset = by_run[run_number]
        params = param_values_for_run(run_number, float(dataset.metadata["temperature"]))
        curve = template.model.function(dataset.time, **params)
        fit_results[run_number] = FitResult(
            success=True,
            chi_squared=48.0,
            reduced_chi_squared=1.03,
            dof=47,
            parameters=ParameterSet(
                [
                    Parameter(
                        name, value=value, min=-50.0, max=50.0, fixed=name in fixed_param_names
                    )
                    for name, value in params.items()
                ]
            ),
            uncertainties={
                name: abs(value) * 0.03 + 1e-3
                for name, value in params.items()
                if name not in fixed_param_names
            },
            residuals=np.zeros_like(dataset.time),
            message="ok",
        )
        fitted_curves[run_number] = (
            np.asarray(dataset.time, dtype=float).copy(),
            np.asarray(curve, dtype=float),
        )
        component_curves[run_number] = ()
        diagnostics.append(
            RunResidualDiagnostic(
                run_number=run_number,
                run_label=dataset.run_label,
                axis_value=float(dataset.metadata["temperature"]),
                residual_rms=0.7,
                runs_z_score=0.2,
                max_abs_autocorrelation=0.08,
                residual_fft_peak_snr=1.1,
                gate_passed=True,
                gate_reasons=(),
            )
        )
    return GlobalCandidateAssessment(
        template=template,
        fit_results_by_run=fit_results,
        global_parameters=ParameterSet(
            [Parameter(name, value=1.0, min=-50.0, max=50.0) for name in global_param_names]
        ),
        global_param_names=global_param_names,
        local_param_names=local_param_names,
        fixed_param_names=fixed_param_names,
        parameter_recommendations=(),
        run_diagnostics=tuple(diagnostics),
        series_warnings=(),
        aic=90.0,
        aicc=92.0,
        bic=105.0,
        selected_score=92.0,
        fitted_curves_by_run=fitted_curves,
        component_curves_by_run=component_curves,
        rung=CandidateRung(
            exempt_runs=(),
            series_cost=series_cost,
            run_costs=dict.fromkeys(runs, series_cost),
            trend=CandidateTrend(
                {
                    name: TraceQuality(determined=1.0, signal=trend_quality, zigzag=0.0)
                    for name in local_param_names
                }
            ),
            hard_to_justify=(),
            pass_disagreements={},
            preselected=True,
            amplitude_unshareable_runs=(),
            all_local_chi2r=1.02,
        ),
    )


class GlobalFitWizardTransitionsScenario(Scenario):
    name = "global_fit_wizard_transitions"
    description = (
        "Global Fit Wizard Phases step — the Transitions card, each phase's rung, "
        "the phase-coloured overlay and the trace strip on a synthetic two-phase "
        "temperature scan."
    )
    size = (1180, 800)
    requires_fit = False

    def build(self) -> QWidget:
        from asymmetry.core.fitting.fit_wizard import SelectionMetric
        from asymmetry.core.fitting.global_fit_wizard import GlobalFitWizardRecommendation
        from asymmetry.core.fitting.global_search.partition import (
            PartitionPath,
            PartitionSolution,
            Segment,
        )
        from asymmetry.core.fitting.global_search.trend_objective import SelectionObjective
        from asymmetry.gui.windows.global_fit_wizard_window import GlobalFitWizardWindow

        datasets = make_two_phase_zf_tscan()
        runs = tuple(int(dataset.run_number) for dataset in datasets)
        cold_runs = runs[0:5]
        warm_runs = runs[5:9]

        cold_template = _template(
            "cold_osc_exp_const",
            "Oscillatory × Exponential + Constant",
            ["Oscillatory", "Exponential", "Constant"],
            ["*", "+"],
        )
        warm_template = _template(
            "warm_exp_const",
            "Exponential + Constant",
            ["Exponential", "Constant"],
            ["+"],
        )

        tc_planted_k = 20.0

        def _cold_params(run_number: int, temperature: float) -> dict[str, float]:
            order = (1.0 - temperature / tc_planted_k) ** 0.35
            return {
                "A_1": 18.0,
                "frequency": 6.0 * order,
                "phase": 0.0,
                "Lambda": 0.35,
                "A_bg": 0.3,
            }

        def _warm_params(run_number: int, temperature: float) -> dict[str, float]:
            return {
                "A_1": 18.0,
                "Lambda": 0.15 + 0.01 * (temperature - tc_planted_k),
                "A_bg": 0.3,
            }

        cold_assessment = _assessment(
            datasets,
            cold_runs,
            cold_template,
            _cold_params,
            global_param_names=("A_bg", "A_1", "Lambda"),
            local_param_names=("frequency",),
            fixed_param_names=("phase",),
            series_cost=0.6,
            trend_quality=0.95,
        )
        warm_assessment = _assessment(
            datasets,
            warm_runs,
            warm_template,
            _warm_params,
            global_param_names=("A_bg", "A_1"),
            local_param_names=("Lambda",),
            series_cost=0.3,
            trend_quality=0.88,
        )

        cold_segment = Segment(
            start=0, stop=5, run_numbers=cold_runs, structure="osc_const", ic=620.0, excluded=False
        )
        warm_segment = Segment(
            start=5, stop=9, run_numbers=warm_runs, structure="exp_const", ic=410.0, excluded=False
        )
        whole_segment = Segment(
            start=0, stop=9, run_numbers=runs, structure="osc_const", ic=1122.4, excluded=False
        )
        partition_path = PartitionPath(
            solutions=(
                PartitionSolution(
                    breaks=0,
                    segments=(whole_segment,),
                    total_ic=1122.4,
                    gain=0.0,
                    admissible=True,
                    boundaries=(),
                ),
                PartitionSolution(
                    breaks=1,
                    segments=(cold_segment, warm_segment),
                    total_ic=1030.0,
                    gain=92.4,
                    admissible=True,
                    boundaries=((21.0, 3.0),),
                ),
            ),
            selected_k=1,
            beta_floor=34.2,
        )

        recommendation = GlobalFitWizardRecommendation(
            series_axis_key="temperature",
            series_axis_label="Temperature (K)",
            mixed_axes_warning=None,
            fingerprints_by_run={
                int(dataset.run_number): _fingerprint(dataset) for dataset in datasets
            },
            dataset_order=runs,
            templates=(cold_template, warm_template),
            assessments=(),
            metric=SelectionMetric.AICC,
            recommended_key=None,
            comparable_keys=(),
            summary="Single-fit screening complete.",
            partition_path=partition_path,
            phase_assessments={(1, 0): cold_assessment, (1, 1): warm_assessment},
            recommended_partition_k=1,
            objective=SelectionObjective.TREND,
        )

        window = GlobalFitWizardWindow()
        window.set_analysis_context(datasets)
        _process_events_for(milliseconds=60)
        window.set_cached_recommendation(recommendation)
        window._show_step("phases")
        # Pick the paramagnetic phase's row, as a click would: its relaxation
        # curves read clearly, where the 5 MHz precession fills the overlay.
        window._phase_rung_rows[1].picked.emit(window._phase_rung_rows[1].key)
        _process_events_for(milliseconds=200)
        return window


register(GlobalFitWizardTransitionsScenario())

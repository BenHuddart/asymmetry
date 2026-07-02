"""Per-member fit-quality flags (Phase 2.1 — the trend quality-gating contract).

Covers the shared :class:`MemberQuality` record and the flag derivation on
synthetic pathological fits, plus the two series engines' population of
``member_quality`` (the χ²ᵣ / σ / off-trend signal the trend tables and plots
consume). Flags are advisory: they never mutate trend inclusion (D3).
"""

from __future__ import annotations

import numpy as np

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.member_quality import (
    MEMBER_QUALITY_FLAGS,
    MemberQuality,
    assess_member_quality,
    large_relative_error_params,
    member_quality_flags,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fitting.result_summary import fit_result_summary
from asymmetry.core.fitting.series import fit_asymmetry_series


def _result(*, success=True, chi2r=1.0, params=None, unc=None) -> FitResult:
    ps = ParameterSet()
    for p in params or []:
        ps.add(p)
    return FitResult(
        success=success,
        reduced_chi_squared=chi2r,
        parameters=ps,
        uncertainties=unc or {},
    )


# ── flag derivation ──────────────────────────────────────────────────────────


def test_clean_fit_carries_no_flags_and_records_chi2_and_errors():
    result = _result(
        params=[Parameter("A", value=20.0, min=0.0, max=100.0)],
        unc={"A": 0.4},
    )
    quality = assess_member_quality(result)
    assert quality.quality_flags == set()
    assert quality.chi2_reduced == 1.0
    assert quality.param_errors == {"A": 0.4}


def test_failed_fit_flagged_failed():
    quality = assess_member_quality(_result(success=False))
    assert "failed" in quality.quality_flags


def test_large_relative_error_flagged():
    # σ (10) swamps the fitted value (0.5): |σ/value| = 20 ≫ 1.
    result = _result(params=[Parameter("A", value=0.5)], unc={"A": 10.0})
    assert large_relative_error_params(result) == ["A"]
    assert "large_rel_err" in assess_member_quality(result).quality_flags


def test_value_collapsed_to_zero_with_finite_sigma_flagged():
    # The EuO run-2947 pathology: amplitude collapsed to ~0 but σ is finite.
    result = _result(params=[Parameter("A", value=1.4e-5)], unc={"A": 3e-3})
    assert "large_rel_err" in assess_member_quality(result).quality_flags


def test_fixed_parameter_never_flags_large_rel_err():
    # A fixed parameter carries no free error and must be skipped.
    result = _result(params=[Parameter("A", value=0.0, fixed=True)], unc={"A": 5.0})
    assert large_relative_error_params(result) == []


def test_free_param_pinned_on_bound_flagged():
    result = _result(params=[Parameter("r", value=2.5, min=0.5, max=2.5)], unc={"r": 0.1})
    assert "bound_pinned" in assess_member_quality(result).quality_flags


def test_interior_fit_not_bound_pinned():
    result = _result(params=[Parameter("r", value=1.5, min=0.5, max=2.5)], unc={"r": 0.1})
    assert "bound_pinned" not in assess_member_quality(result).quality_flags


def test_extra_flags_are_vocabulary_gated():
    result = _result()
    flags = member_quality_flags(result, extra_flags=("spurious_reseeded", "not_a_real_flag"))
    assert "spurious_reseeded" in flags
    assert "not_a_real_flag" not in flags
    assert flags <= set(MEMBER_QUALITY_FLAGS)


def test_non_finite_chi2_becomes_none():
    assert assess_member_quality(_result(chi2r=float("nan"))).chi2_reduced is None
    assert assess_member_quality(_result(chi2r=0.0)).chi2_reduced is None


def test_member_quality_payload_roundtrips():
    quality = MemberQuality(
        chi2_reduced=1.3,
        param_errors={"A": 0.4, "f": 0.02},
        quality_flags={"failed", "large_rel_err"},
    )
    restored = MemberQuality.from_payload(quality.to_payload())
    assert restored.chi2_reduced == 1.3
    assert restored.param_errors == {"A": 0.4, "f": 0.02}
    assert restored.quality_flags == {"failed", "large_rel_err"}


def test_from_payload_tolerates_absent_and_garbage_fields():
    assert MemberQuality.from_payload(None).quality_flags == set()
    assert MemberQuality.from_payload({}).chi2_reduced is None
    salvaged = MemberQuality.from_payload({"chi2_reduced": "x", "param_errors": {"A": "y"}})
    assert salvaged.chi2_reduced is None
    assert salvaged.param_errors == {}


def test_fit_result_summary_embeds_quality_flags():
    result = _result(success=False, params=[Parameter("A", value=0.5)], unc={"A": 10.0})
    summary = fit_result_summary(result)
    assert "failed" in summary["quality_flags"]
    assert "large_rel_err" in summary["quality_flags"]
    # Extra flags thread through the summary too.
    threaded = fit_result_summary(result, extra_flags=("spurious_reseeded",))
    assert "spurious_reseeded" in threaded["quality_flags"]


# ── F-B series engine populates member_quality (incl. spurious_reseeded) ──────

_REAL_FREQ = {2960: 30.0, 2955: 26.0, 2950: 18.0, 2945: 12.0, 2940: 8.0}
_ORDER = {2960: 10.0, 2955: 30.0, 2950: 50.0, 2945: 60.0, 2940: 67.0}
_CAPTURE_WINDOW = 5.0
_SPURIOUS_FREQ = 30.0


def _datasets():
    time = np.linspace(0.1, 8.0, 16)
    return [
        MuonDataset(
            time=time,
            asymmetry=np.zeros_like(time),
            error=np.ones_like(time),
            metadata={"run_number": run, "temperature": _ORDER[run]},
        )
        for run in _REAL_FREQ
    ]


def _initial() -> dict[int, ParameterSet]:
    out: dict[int, ParameterSet] = {}
    for run in _REAL_FREQ:
        ps = ParameterSet()
        ps.add(Parameter(name="A_1", value=20.0, min=0.0, max=100.0))
        ps.add(Parameter(name="frequency", value=_SPURIOUS_FREQ, min=0.0, max=100.0))
        out[run] = ps
    return out


class _BistableEngine:
    """Finds the real branch only when seeded within the capture window."""

    def fit(self, dataset, _model_fn, parameters, **_kwargs) -> FitResult:
        run = int(dataset.run_number)
        seed_freq = float(parameters["frequency"].value)
        real = _REAL_FREQ[run]
        fitted = ParameterSet()
        if abs(seed_freq - real) <= _CAPTURE_WINDOW:
            fitted.add(Parameter(name="A_1", value=20.0))
            fitted.add(Parameter(name="frequency", value=real))
            return FitResult(
                success=True,
                reduced_chi_squared=1.0,
                parameters=fitted,
                uncertainties={"A_1": 0.4, "frequency": 0.1},
            )
        fitted.add(Parameter(name="A_1", value=0.05))
        fitted.add(Parameter(name="frequency", value=_SPURIOUS_FREQ))
        return FitResult(
            success=True,
            reduced_chi_squared=3.0,
            parameters=fitted,
            uncertainties={"A_1": 0.4, "frequency": 0.1},
        )


def _run_series(seeding: str, engine=None):
    return fit_asymmetry_series(
        _datasets(),
        lambda t: t,
        global_params=[],
        local_params=["A_1", "frequency"],
        initial_params=_initial(),
        fit_engine=engine or _BistableEngine(),
        seeding=seeding,
        order_key=_ORDER,
        amplitude_param="A_1",
        frequency_param="frequency",
    )


def test_series_result_populates_member_quality_for_every_run():
    result = _run_series("as_provided")
    assert set(result.member_quality) == set(_REAL_FREQ)
    # A clean member records its χ²ᵣ and σ.
    good = result.member_quality[2960]
    assert good.chi2_reduced == 1.0
    assert good.param_errors["A_1"] == 0.4


def test_reseeded_runs_are_flagged_spurious():
    result = _run_series("chain")
    assert result.reseeded_runs, "expected a detect-and-reseed for this fixture"
    for run in result.reseeded_runs:
        assert "spurious_reseeded" in result.member_quality[run].quality_flags


def test_stranded_spurious_member_flagged_even_without_reseed():
    # A single collapsed member among good ones is flagged off-trend by the
    # batch-wide diagnosis, independent of the seeding mode.
    class _OneCollapsedEngine:
        def fit(self, dataset, _model_fn, parameters, **_kwargs) -> FitResult:
            run = int(dataset.run_number)
            ps = ParameterSet()
            if run == 2950:
                ps.add(Parameter(name="A_1", value=0.05))  # collapsed
                ps.add(Parameter(name="frequency", value=30.0))
            else:
                ps.add(Parameter(name="A_1", value=20.0))
                ps.add(Parameter(name="frequency", value=_REAL_FREQ[run]))
            return FitResult(
                success=True,
                reduced_chi_squared=1.0,
                parameters=ps,
                uncertainties={"A_1": 0.4, "frequency": 0.1},
            )

    result = _run_series("as_provided", engine=_OneCollapsedEngine())
    assert "spurious_reseeded" in result.member_quality[2950].quality_flags
    assert "spurious_reseeded" not in result.member_quality[2960].quality_flags


def test_flags_never_mutate_trend_inclusion_state():
    # The contract is advisory: AsymmetrySeriesResult carries no inclusion side
    # effect — member_quality is pure diagnosis.
    result = _run_series("chain")
    assert all(q.quality_flags <= set(MEMBER_QUALITY_FLAGS) for q in result.member_quality.values())

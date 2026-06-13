"""Tests for capture_ratio_report / fb_capture_ratio_report (WP2.2).

Verification-plan §3: Fixtures A (zero covariance → 2.00(10)) and
B (positive covariance → 2.00(9)) reproduced exactly. FitResults are built
directly from known values — no actual fitting.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset, Run
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.negmu.fit import CaptureModelSpec, fit_capture_fb_alpha
from asymmetry.core.negmu.model import CaptureComponent
from asymmetry.core.negmu.ratio import (
    CaptureRatio,
    CaptureRatioReport,
    capture_ratio_report,
    fb_capture_ratio_report,
)
from asymmetry.core.simulate import simulate_capture_run
from tests.negmu.helpers import combine_groups as _combine_groups
from tests.negmu.helpers import make_fb_template

# ---------------------------------------------------------------------------
# Helpers for building synthetic FitResults
# ---------------------------------------------------------------------------


def _build_fit_result(
    amplitudes: dict[str, float],
    uncertainties: dict[str, float],
    *,
    cov_matrix: np.ndarray | None = None,
    cov_params: list[str] | None = None,
) -> FitResult:
    """Build a minimal FitResult from known amplitudes and uncertainties."""
    ps = ParameterSet()
    for label, amp in amplitudes.items():
        ps.add(Parameter(name=f"amp_{label}", value=amp))
    unc = {f"amp_{lbl}": sigma for lbl, sigma in uncertainties.items()}
    return FitResult(
        success=True,
        parameters=ps,
        uncertainties=unc,
        covariance=cov_matrix,
        covariance_parameters=list(cov_params) if cov_params else [],
    )


# ---------------------------------------------------------------------------
# Fixture A — zero covariance: ratio C/O = 2.00(10)
# ---------------------------------------------------------------------------

_AMP_C = 1000.0
_AMP_O = 500.0
_SIG_C = 30.0
_SIG_O = 20.0

_SPEC_CO = CaptureModelSpec(elements=("C", "O"), include_decay_background=False)

# Hand-computed expected values (verification-plan §3).
_R_A = _AMP_C / _AMP_O  # = 2.0
_SIGMA_A = _R_A * math.sqrt((_SIG_C / _AMP_C) ** 2 + (_SIG_O / _AMP_O) ** 2)
# = 2 * sqrt(0.0009 + 0.0016) = 2 * 0.05 = 0.100


@pytest.fixture(scope="module")
def fixture_a_report():
    fit = _build_fit_result(
        amplitudes={"C": _AMP_C, "O": _AMP_O},
        uncertainties={"C": _SIG_C, "O": _SIG_O},
    )
    return capture_ratio_report(fit, _SPEC_CO, reference="O")


def test_fixture_a_ratio(fixture_a_report):
    assert fixture_a_report.ratios[0].ratio == pytest.approx(_R_A, rel=1e-6)


def test_fixture_a_sigma(fixture_a_report):
    assert fixture_a_report.ratios[0].sigma == pytest.approx(_SIGMA_A, rel=1e-6)


def test_fixture_a_numerator_denominator(fixture_a_report):
    r = fixture_a_report.ratios[0]
    assert r.numerator == "C"
    assert r.denominator == "O"


def test_fixture_a_side(fixture_a_report):
    assert fixture_a_report.side == "forward"


def test_fixture_a_amplitudes_reported(fixture_a_report):
    assert fixture_a_report.amplitudes["C"] == pytest.approx(_AMP_C, rel=1e-12)
    assert fixture_a_report.amplitudes["O"] == pytest.approx(_AMP_O, rel=1e-12)


# ---------------------------------------------------------------------------
# Fixture B — covariance-aware: ratio C/O = 2.00, σ = 2*sqrt(0.0019)
# ---------------------------------------------------------------------------

_COV_CO = 150.0  # cov(amp_C, amp_O) — positive correlation reduces σ_R

# Covariance matrix in the order [amp_C, amp_O].
_COV_MATRIX_B = np.array([[_SIG_C**2, _COV_CO], [_COV_CO, _SIG_O**2]])
_COV_PARAMS_B = ["amp_C", "amp_O"]

# Hand-computed:
# σ_R = R * sqrt((σ_C/amp_C)² + (σ_O/amp_O)² − 2*cov/(amp_C*amp_O))
#      = 2 * sqrt(0.0009 + 0.0016 − 0.0006) = 2 * sqrt(0.0019)
_VAR_B = (_SIG_C / _AMP_C) ** 2 + (_SIG_O / _AMP_O) ** 2 - 2 * _COV_CO / (_AMP_C * _AMP_O)
_SIGMA_B = _R_A * math.sqrt(_VAR_B)


@pytest.fixture(scope="module")
def fixture_b_report():
    fit = _build_fit_result(
        amplitudes={"C": _AMP_C, "O": _AMP_O},
        uncertainties={"C": _SIG_C, "O": _SIG_O},
        cov_matrix=_COV_MATRIX_B,
        cov_params=_COV_PARAMS_B,
    )
    return capture_ratio_report(fit, _SPEC_CO, reference="O")


def test_fixture_b_ratio(fixture_b_report):
    assert fixture_b_report.ratios[0].ratio == pytest.approx(_R_A, rel=1e-6)


def test_fixture_b_sigma(fixture_b_report):
    assert fixture_b_report.ratios[0].sigma == pytest.approx(_SIGMA_B, rel=1e-6)


def test_fixture_b_sigma_less_than_a(fixture_b_report):
    """Positive covariance reduces the ratio uncertainty (Fixture B < Fixture A)."""
    assert fixture_b_report.ratios[0].sigma < _SIGMA_A


def test_fixture_b_side_override():
    fit = _build_fit_result(
        amplitudes={"C": _AMP_C, "O": _AMP_O},
        uncertainties={"C": _SIG_C, "O": _SIG_O},
    )
    report = capture_ratio_report(fit, _SPEC_CO, reference="O", side="backward")
    assert report.side == "backward"


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------


def test_reference_excluded_from_ratios():
    """The reference element does not appear as a numerator."""
    fit = _build_fit_result(
        amplitudes={"C": 500.0, "O": 300.0},
        uncertainties={"C": 10.0, "O": 8.0},
    )
    report = capture_ratio_report(fit, _SPEC_CO, reference="O")
    numerators = [r.numerator for r in report.ratios]
    assert "O" not in numerators


def test_decaybg_excluded_from_ratios():
    """decayBG is excluded from ratios by default even when in spec."""
    spec = CaptureModelSpec(elements=("C", "O"), include_decay_background=True)
    fit = _build_fit_result(
        amplitudes={"C": 500.0, "O": 300.0, "decayBG": 100.0},
        uncertainties={"C": 10.0, "O": 8.0, "decayBG": 5.0},
    )
    report = capture_ratio_report(fit, spec, reference="O")
    numerators = [r.numerator for r in report.ratios]
    assert "decayBG" not in numerators


def test_ratio_returns_captureratiorepport_type():
    fit = _build_fit_result(
        amplitudes={"C": 1000.0, "O": 500.0},
        uncertainties={"C": 30.0, "O": 20.0},
    )
    report = capture_ratio_report(fit, _SPEC_CO, reference="O")
    assert isinstance(report, CaptureRatioReport)
    assert all(isinstance(r, CaptureRatio) for r in report.ratios)


# ---------------------------------------------------------------------------
# fb_capture_ratio_report smoke test
# ---------------------------------------------------------------------------

N_BINS = 512
BIN_WIDTH = 0.016


def _make_fb_template() -> Run:
    return make_fb_template(n_bins=N_BINS, bin_width=BIN_WIDTH)


_COMPS_SMOKE = [
    CaptureComponent(label="C", tau_us=2.030),
    CaptureComponent(label="O", tau_us=1.795),
]
_WEIGHTS_SMOKE = {"C": 5.0, "O": 3.0}
_SPEC_SMOKE = CaptureModelSpec(elements=("C", "O"), include_decay_background=False)


@pytest.fixture(scope="module")
def fb_grouped_result():
    template = _make_fb_template()
    run_f = simulate_capture_run(
        template,
        _COMPS_SMOKE,
        _WEIGHTS_SMOKE,
        total_events=5.0e6,
        group_id=1,
        seed=20,
    )
    run_b = simulate_capture_run(
        template,
        _COMPS_SMOKE,
        _WEIGHTS_SMOKE,
        total_events=5.0e6,
        group_id=2,
        seed=21,
    )
    combined = _combine_groups(run_f, run_b, template)
    ds = MuonDataset(
        time=np.array([]), asymmetry=np.array([]), error=np.array([]), metadata={}, run=combined
    )
    return fit_capture_fb_alpha(ds, forward_group=1, backward_group=2, spec=_SPEC_SMOKE)


def test_fb_report_has_both_sides(fb_grouped_result):
    reports = fb_capture_ratio_report(fb_grouped_result, _SPEC_SMOKE, 1, 2, reference="O")
    assert "forward" in reports
    assert "backward" in reports


def test_fb_report_forward_side_label(fb_grouped_result):
    reports = fb_capture_ratio_report(fb_grouped_result, _SPEC_SMOKE, 1, 2, reference="O")
    assert reports["forward"].side == "forward"
    assert reports["backward"].side == "backward"


def test_fb_report_shared_ratios_identical(fb_grouped_result):
    """Shared amplitudes → identical ratios on both sides."""
    reports = fb_capture_ratio_report(fb_grouped_result, _SPEC_SMOKE, 1, 2, reference="O")
    r_fwd = reports["forward"].ratios[0].ratio
    r_bwd = reports["backward"].ratios[0].ratio
    assert r_fwd == pytest.approx(r_bwd, rel=1e-12)


def test_fb_report_type(fb_grouped_result):
    reports = fb_capture_ratio_report(fb_grouped_result, _SPEC_SMOKE, 1, 2, reference="O")
    assert isinstance(reports["forward"], CaptureRatioReport)
    assert isinstance(reports["backward"], CaptureRatioReport)

"""Backward-amplitude balance (β) calibration from a weak-TF run.

The scalar β of musrfit fit type 2 scales the **backward** polarization
amplitude relative to the forward one, so a detector pair sees

    N_f(t) = N0·√α · e^(−t/τ_μ) · [1 + A·P(t)]        + b_f
    N_b(t) = N0/√α · e^(−t/τ_μ) · [1 − β·A·P(t)]      + b_b

with P(t) the (damped-cosine) polarization of a weak transverse-field
calibration run — the same run type used to calibrate α. β is measured, not
assumed; a value below 1 means the backward detector under-reports the
asymmetry amplitude. This module is a thin orchestrator over the count-domain
primitives in :mod:`asymmetry.core.fitting.count_domain`, offering two protocols:

- ``"count_fit"`` (Protocol A, default) — one simultaneous forward+backward
  TF count fit (Poisson for raw counts, weighted after applied deadtime) that
  shares the physics between the two histograms and floats a single ``beta`` on
  the backward amplitude. This is the endorsed measurement protocol and the only
  one that yields an α–β correlation.
- ``"single_histogram"`` (Protocol B, cross-check) — two independent
  single-histogram fits (:func:`fit_single_histogram`), one per side, each
  floating its own physics; β̂ = Â₀,b/Â₀,f and α̂ = N̂₀,f/N̂₀,b by ratio, with no
  cross-parameter correlation available. Statistically weaker (the physics is
  not shared) but a useful independent check.

β is degenerate without precession, so failure modes — no oscillation found, a
forward amplitude consistent with zero, or a fit that did not converge — return
``ok=False`` with an informative message rather than a garbage number, mirroring
:class:`~asymmetry.core.transform.asymmetry.AlphaEstimate`.

This is a pure-core module: no Qt, matplotlib, or GUI imports.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.count_domain import fit_single_histogram
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.grouped_time_domain import build_count_group
from asymmetry.core.fitting.models import oscillatory
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fitting.tf_calibration import (
    TFCalibrationRequest,
    _amplitude_is_significant,
    _frequency_is_resolved,
    _seed_frequency_mhz,
    fit_tf_calibration,
)

#: Method identifiers accepted by :func:`estimate_beta_detailed`.
BETA_ESTIMATION_METHODS: tuple[str, ...] = ("count_fit", "single_histogram")

#: Typical seed amplitude for a transverse-field calibration run (percent), the
#: convention shared with the count-fit seeder in the GUI.
_SEED_AMPLITUDE_PERCENT = 20.0


@dataclass(frozen=True)
class BetaEstimate:
    """Result of a backward-amplitude balance (β) estimation.

    ``beta_error`` / ``alpha_error`` are the symmetric (HESSE) standard errors,
    or ``None`` when unavailable. ``alpha_beta_correlation`` is the Pearson
    correlation of the fitted α and β from the joint fit's covariance block for
    the ``"count_fit"`` method, and ``None`` for ``"single_histogram"`` (two
    independent fits carry no cross-parameter correlation). ``reduced_chi2`` is
    Poisson deviance or weighted chi-square per joint dof for ``count_fit``
    (unique free parameters),
    or pooled per-side deviance/dof for ``single_histogram``.
    """

    beta: float
    beta_error: float | None
    alpha: float
    alpha_error: float | None
    alpha_beta_correlation: float | None
    method: str
    n_bins_used: int
    reduced_chi2: float | None
    ok: bool
    message: str = ""


def _fail(method: str, message: str, *, n_bins_used: int = 0) -> BetaEstimate:
    """Build a failed :class:`BetaEstimate` carrying an informative message."""
    return BetaEstimate(
        beta=float("nan"),
        beta_error=None,
        alpha=float("nan"),
        alpha_error=None,
        alpha_beta_correlation=None,
        method=method,
        n_bins_used=int(n_bins_used),
        reduced_chi2=None,
        ok=False,
        message=message,
    )


def _n0_seed(dataset: MuonDataset, group_id: int) -> float:
    """First-good-bin count as the ``N0`` seed for a group (mirrors the GUI seeder)."""
    try:
        group = build_count_group(dataset, int(group_id), lifetime_corrected=False)
    except ValueError:
        return 1.0
    counts = np.asarray(group.counts, dtype=float)
    return float(counts[0]) if counts.size else 1.0


def _oscillatory_seed_params(
    dataset: MuonDataset, group_id: int, *, freq_mhz: float
) -> ParameterSet:
    """Seed set for one single-histogram oscillatory count fit.

    ``A0`` carries a ``min=0`` floor: the fit floats the phase, so the amplitude
    stays non-negative and β̂ = Â₀,b/Â₀,f is a ratio of positive amplitudes.
    """
    return ParameterSet(
        [
            Parameter("N0", _n0_seed(dataset, group_id), min=0.0),
            Parameter("background", 0.0, min=0.0),
            Parameter("A0", _SEED_AMPLITUDE_PERCENT, min=0.0, max=100.0),
            Parameter("frequency", float(freq_mhz), min=0.0),
            Parameter("phase", 0.0),
            Parameter("Lambda", 0.1, min=0.0),
            # A weak-TF calibration asymmetry has no static offset; the flat count
            # background already carries the DC level. Fixing baseline keeps it
            # from trading against the background.
            Parameter("baseline", 0.0, fixed=True),
        ]
    )


def _combined_reduced_chi2(*results: FitResult) -> float | None:
    """Pooled χ²/dof over the supplied side fits (``None`` when dof ≤ 0)."""
    chi2 = 0.0
    dof = 0
    for r in results:
        chi2 += float(r.chi_squared)
        dof += int(r.dof)
    if dof <= 0:
        return None
    return chi2 / dof


def _estimate_beta_count_fit(
    dataset: MuonDataset,
    forward_group: int,
    backward_group: int,
    *,
    t_min: float | None,
    t_max: float | None,
    exclude: tuple[float, float] | None,
    freq_mhz: float,
    minos: bool,
    pulse_separation_us: float | None,
    cancel_callback: Callable[[], bool] | None,
) -> BetaEstimate:
    """Keep the historical count_fit result vocabulary over the shared TF fit."""
    result = fit_tf_calibration(
        dataset,
        TFCalibrationRequest(
            forward_group,
            backward_group,
            fit_beta=True,
            t_min=t_min,
            t_max=t_max,
            exclude=exclude,
            frequency_mhz=freq_mhz,
            minos=minos,
            pulse_separation_us=pulse_separation_us,
        ),
        cancel_callback=cancel_callback,
    )
    if not result.ok:
        return _fail("count_fit", result.message, n_bins_used=result.n_bins_used)
    return BetaEstimate(
        beta=result.beta,
        beta_error=result.beta_error,
        alpha=result.alpha,
        alpha_error=result.alpha_error,
        alpha_beta_correlation=result.alpha_beta_correlation,
        method="count_fit",
        n_bins_used=result.n_bins_used,
        reduced_chi2=result.fit.statistic_per_dof,
        ok=True,
        message="β estimated by simultaneous forward/backward count fit.",
    )


def _ratio_error(value: float, num: float, num_err: float, den: float, den_err: float) -> float:
    """Standard error of ``value = num/den`` from independent relative errors."""
    rel_sq = (num_err / num) ** 2 + (den_err / den) ** 2
    return float(abs(value) * np.sqrt(rel_sq))


def _estimate_beta_single_histogram(
    dataset: MuonDataset,
    forward_group: int,
    backward_group: int,
    *,
    t_min: float | None,
    t_max: float | None,
    exclude: tuple[float, float] | None,
    freq_mhz: float,
    minos: bool,
    cancel_callback: Callable[[], bool] | None,
) -> BetaEstimate:
    """Protocol B: independent single-histogram fits, β̂ = Â₀,b/Â₀,f by ratio."""
    method = "single_histogram"
    fit_fwd = fit_single_histogram(
        dataset,
        forward_group,
        oscillatory,
        _oscillatory_seed_params(dataset, forward_group, freq_mhz=freq_mhz),
        side="forward",
        t_min=t_min,
        t_max=t_max,
        exclude=exclude,
        minos=minos,
        cancel_callback=cancel_callback,
    )
    fit_bwd = fit_single_histogram(
        dataset,
        backward_group,
        oscillatory,
        _oscillatory_seed_params(dataset, backward_group, freq_mhz=freq_mhz),
        side="backward",
        t_min=t_min,
        t_max=t_max,
        exclude=exclude,
        minos=minos,
        cancel_callback=cancel_callback,
    )
    n_bins = int(np.size(fit_fwd.residuals))
    if not fit_fwd.success or not fit_bwd.success:
        return _fail(
            method, "One or both single-histogram fits did not converge.", n_bins_used=n_bins
        )

    a_f = fit_fwd.parameters["A0"].value
    a_b = fit_bwd.parameters["A0"].value
    a_f_err = fit_fwd.uncertainties.get("A0")
    a_b_err = fit_bwd.uncertainties.get("A0")
    freq_f = (
        fit_fwd.parameters["frequency"].value if "frequency" in fit_fwd.parameters.names else None
    )
    freq_f_err = fit_fwd.uncertainties.get("frequency")
    if not _amplitude_is_significant(a_f, a_f_err) or not _frequency_is_resolved(
        freq_f, freq_f_err
    ):
        return _fail(
            method,
            "No resolved precession signal (forward amplitude or frequency consistent "
            "with zero); β is degenerate without precession.",
            n_bins_used=n_bins,
        )
    if abs(a_f) <= 1.0e-9:
        return _fail(method, "Forward amplitude is degenerate (near zero).", n_bins_used=n_bins)

    # Both amplitudes float their own phase against a min=0 bound, so each comes
    # out non-negative; β̂ is then a positive ratio for physical data.
    beta = float(a_b / a_f)
    # Backward N0 balances against forward as α = N0_f / N0_b (the fgFB convention:
    # forward scale √α, backward 1/√α). NOTE this is forward/backward — the
    # opposite direction to β = backward/forward; do not mirror β's ratio here.
    n0_f = fit_fwd.parameters["N0"].value
    n0_b = fit_bwd.parameters["N0"].value
    n0_f_err = fit_fwd.uncertainties.get("N0")
    n0_b_err = fit_bwd.uncertainties.get("N0")
    alpha = float(n0_f / n0_b) if abs(n0_b) > 0.0 else float("nan")

    beta_err: float | None = None
    if (
        a_f_err is not None
        and a_b_err is not None
        and np.isfinite(a_f_err)
        and np.isfinite(a_b_err)
    ):
        beta_err = _ratio_error(beta, a_b, a_b_err, a_f, a_f_err)
    alpha_err: float | None = None
    if (
        n0_f_err is not None
        and n0_b_err is not None
        and np.isfinite(n0_f_err)
        and np.isfinite(n0_b_err)
        and abs(n0_b) > 0.0
        and abs(n0_f) > 0.0
    ):
        alpha_err = _ratio_error(alpha, n0_f, n0_f_err, n0_b, n0_b_err)

    return BetaEstimate(
        beta=beta,
        beta_error=beta_err,
        alpha=alpha,
        alpha_error=alpha_err,
        alpha_beta_correlation=None,
        method=method,
        n_bins_used=n_bins,
        reduced_chi2=_combined_reduced_chi2(fit_fwd, fit_bwd),
        ok=True,
        message="β estimated by paired single-histogram fits (α–β correlation unavailable).",
    )


def estimate_beta_detailed(
    dataset: MuonDataset,
    forward_group: int,
    backward_group: int,
    *,
    method: str = "count_fit",
    t_min: float | None = None,
    t_max: float | None = None,
    exclude: tuple[float, float] | None = None,
    field_tesla: float | None = None,
    minos: bool = False,
    pulse_separation_us: float | None = None,
    cancel_callback: Callable[[], bool] | None = None,
) -> BetaEstimate:
    """Estimate the backward-amplitude balance β from a weak-TF calibration run.

    Parameters
    ----------
    dataset
        The loaded run (must carry detector histograms and a forward/backward
        detector grouping).
    forward_group, backward_group
        The detector-group ids of the balanced pair.
    method
        ``"count_fit"`` (default, Protocol A — simultaneous fit) or
        ``"single_histogram"`` (Protocol B — paired independent fits). See the
        module docstring.
    t_min, t_max, exclude
        Fit window and an optional interior exclusion, passed straight to the
        count-domain primitives.
    field_tesla
        Applied transverse field (Tesla) used to seed the precession frequency
        (γ_μ·B). When ``None`` the dataset's recorded field is used.
    pulse_separation_us
        Fixed equal-incident double-pulse separation in µs for ``count_fit``;
        omitted means off. The independent single-histogram protocol does not
        consume this setting.
    minos
        Run MINOS for asymmetric intervals on the underlying fit(s) (symmetric
        HESSE errors are reported regardless).
    cancel_callback
        Cooperative cancellation predicate threaded into the fit.

    Returns
    -------
    BetaEstimate
        On success, ``beta``/``alpha`` with symmetric errors, the α–β correlation
        (Protocol A only), the bins used, and deviance per joint dof for Protocol A
        or pooled per-side dof for Protocol B. On any
        data-quality failure (no precession, degenerate amplitude, fit did not
        converge) ``ok`` is ``False`` and ``message`` explains why; this function
        does not raise for such cases.
    """
    if method not in BETA_ESTIMATION_METHODS:
        raise ValueError(
            f"Unknown β-estimation method {method!r}; expected one of {BETA_ESTIMATION_METHODS}"
        )
    if pulse_separation_us is not None and method != "count_fit":
        raise ValueError("Pulse separation requires the count_fit TF calibration method")
    if dataset.run is None or not dataset.run.histograms:
        return _fail(method, "β estimation needs a run with detector histograms.")
    if int(forward_group) == int(backward_group):
        return _fail(method, "β estimation needs two distinct detector groups.")

    freq_mhz = _seed_frequency_mhz(dataset, field_tesla)
    kwargs = dict(
        t_min=t_min,
        t_max=t_max,
        exclude=exclude,
        freq_mhz=freq_mhz,
        minos=minos,
        cancel_callback=cancel_callback,
    )
    if method == "count_fit":
        return _estimate_beta_count_fit(
            dataset,
            forward_group,
            backward_group,
            pulse_separation_us=pulse_separation_us,
            **kwargs,
        )
    return _estimate_beta_single_histogram(dataset, forward_group, backward_group, **kwargs)

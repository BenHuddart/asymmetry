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
  Poisson count fit (:func:`fit_fb_alpha` with ``estimate_beta=True``) that
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
from asymmetry.core.fitting.count_domain import fit_fb_alpha, fit_single_histogram
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.grouped_time_domain import build_count_group
from asymmetry.core.fitting.models import oscillatory
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.utils.constants import (
    GAUSS_TO_TESLA,
    MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
)

#: Method identifiers accepted by :func:`estimate_beta_detailed`.
BETA_ESTIMATION_METHODS: tuple[str, ...] = ("count_fit", "single_histogram")

#: Amplitude / frequency significance below which the precession signal is
#: treated as absent (β is degenerate without a resolved oscillation — the design
#: note's §4). Both gates are needed: a flat (non-precessing) asymmetry fits a
#: *significant* constant amplitude at a frequency that collapses to zero, so the
#: frequency gate is what actually rejects the degenerate case.
_MIN_AMPLITUDE_SIGNIFICANCE = 3.0
_MIN_FREQUENCY_SIGNIFICANCE = 3.0

#: Fallback precession-frequency seed (MHz) when no field is known — a weak-TF
#: calibration run should always carry field metadata, so this is a last resort.
_FALLBACK_FREQUENCY_MHZ = 1.0

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
    the combined per-side χ²/dof summary of the underlying fit(s).
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


def _seed_frequency_mhz(dataset: MuonDataset, field_tesla: float | None) -> float:
    """Precession-frequency seed (MHz) for the oscillatory model.

    Prefers an explicit ``field_tesla``; otherwise falls back to the dataset's
    recorded field (Gauss). A missing or zero field yields
    :data:`_FALLBACK_FREQUENCY_MHZ` — a calibration run normally carries the
    field, so this is only a floor to keep the fit runnable.
    """
    if field_tesla is not None and np.isfinite(field_tesla) and field_tesla != 0.0:
        return float(MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * abs(float(field_tesla)))
    field_gauss = float(dataset.run.field) if dataset.run is not None else 0.0
    if np.isfinite(field_gauss) and field_gauss != 0.0:
        return float(MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA * abs(field_gauss))
    return _FALLBACK_FREQUENCY_MHZ


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


def _amplitude_is_significant(value: float | None, error: float | None) -> bool:
    """Whether a fitted amplitude is resolved above zero (precession present)."""
    if value is None or not np.isfinite(value):
        return False
    if abs(value) <= 1.0e-9:
        return False
    if error is None or not np.isfinite(error) or error <= 0.0:
        # No usable error: accept only a clearly non-zero amplitude.
        return abs(value) > 1.0e-6
    return abs(value) >= _MIN_AMPLITUDE_SIGNIFICANCE * error


def _frequency_is_resolved(value: float | None, error: float | None) -> bool:
    """Whether a fitted precession frequency is resolved above zero.

    A flat / non-precessing asymmetry drives the fitted frequency to zero; when
    it is consistent with zero (or has no usable error) there is no resolved
    precession and β is degenerate.
    """
    if value is None or not np.isfinite(value):
        return False
    if error is None or not np.isfinite(error) or error <= 0.0:
        return False
    return abs(value) >= _MIN_FREQUENCY_SIGNIFICANCE * error


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


def _alpha_beta_correlation(result: FitResult) -> float | None:
    """Pearson α–β correlation from a joint fit's covariance block, or ``None``."""
    cov = result.covariance
    names = result.covariance_parameters
    if cov is None or not names or "alpha" not in names or "beta" not in names:
        return None
    i = names.index("alpha")
    j = names.index("beta")
    denom = np.sqrt(cov[i, i] * cov[j, j])
    if not np.isfinite(denom) or denom <= 0.0:
        return None
    rho = float(cov[i, j] / denom)
    if not np.isfinite(rho):
        return None
    return float(np.clip(rho, -1.0, 1.0))


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
    cancel_callback: Callable[[], bool] | None,
) -> BetaEstimate:
    """Protocol A: simultaneous forward/backward count fit with β free."""
    method = "count_fit"
    params = ParameterSet(
        [
            Parameter("alpha", 1.0, min=1.0e-6, max=100.0),
            Parameter("N0", _n0_seed(dataset, forward_group), min=0.0),
            Parameter("background", 0.0, min=0.0),
            Parameter("background_b", 0.0, min=0.0),
            Parameter("A0", _SEED_AMPLITUDE_PERCENT, min=0.0, max=100.0),
            Parameter("frequency", float(freq_mhz), min=0.0),
            Parameter("phase", 0.0),
            Parameter("Lambda", 0.1, min=0.0),
            Parameter("baseline", 0.0, fixed=True),
        ]
    )
    result = fit_fb_alpha(
        dataset,
        forward_group,
        backward_group,
        oscillatory,
        params,
        t_min=t_min,
        t_max=t_max,
        exclude=exclude,
        estimate_beta=True,
        minos=minos,
        cancel_callback=cancel_callback,
    )
    fwd = result.group_results.get(int(forward_group))
    bwd = result.group_results.get(int(backward_group))
    n_bins = int(np.size(fwd.residuals)) if fwd is not None else 0
    if not result.success or fwd is None or bwd is None:
        return _fail(method, "Forward/backward count fit did not converge.", n_bins_used=n_bins)

    amplitude = fwd.parameters["A0"].value if "A0" in fwd.parameters.names else None
    amplitude_err = fwd.uncertainties.get("A0")
    frequency = fwd.parameters["frequency"].value if "frequency" in fwd.parameters.names else None
    frequency_err = fwd.uncertainties.get("frequency")
    if not _amplitude_is_significant(amplitude, amplitude_err) or not _frequency_is_resolved(
        frequency, frequency_err
    ):
        return _fail(
            method,
            "No resolved precession signal (amplitude or frequency consistent with "
            "zero); β is degenerate without precession.",
            n_bins_used=n_bins,
        )

    beta = float(result.shared_parameters["beta"].value)
    alpha = float(result.shared_parameters["alpha"].value)
    beta_err = fwd.uncertainties.get("beta")
    alpha_err = fwd.uncertainties.get("alpha")
    if beta_err is None or not np.isfinite(beta_err):
        return _fail(
            method,
            "β could not be constrained (no finite error); the run may lack a "
            "resolved precession signal.",
            n_bins_used=n_bins,
        )

    return BetaEstimate(
        beta=beta,
        beta_error=float(beta_err),
        alpha=alpha,
        alpha_error=float(alpha_err) if alpha_err is not None else None,
        alpha_beta_correlation=_alpha_beta_correlation(fwd),
        method=method,
        n_bins_used=n_bins,
        reduced_chi2=_combined_reduced_chi2(fwd, bwd),
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
    minos
        Run MINOS for asymmetric intervals on the underlying fit(s) (symmetric
        HESSE errors are reported regardless).
    cancel_callback
        Cooperative cancellation predicate threaded into the fit.

    Returns
    -------
    BetaEstimate
        On success, ``beta``/``alpha`` with symmetric errors, the α–β correlation
        (Protocol A only), the bins used and the pooled reduced χ². On any
        data-quality failure (no precession, degenerate amplitude, fit did not
        converge) ``ok`` is ``False`` and ``message`` explains why; this function
        does not raise for such cases.
    """
    if method not in BETA_ESTIMATION_METHODS:
        raise ValueError(
            f"Unknown β-estimation method {method!r}; expected one of {BETA_ESTIMATION_METHODS}"
        )
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
        return _estimate_beta_count_fit(dataset, forward_group, backward_group, **kwargs)
    return _estimate_beta_single_histogram(dataset, forward_group, backward_group, **kwargs)

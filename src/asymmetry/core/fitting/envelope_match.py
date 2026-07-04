"""Time-domain matched-filter matching for damped-envelope signatures.

FFT peak detection resolves *sharp spectral lines*, but the fluorine-dipolar
(F-mu-F, mu-F) and Kubo-Toyabe families are **damped envelopes** — a sum of a
few low-frequency cosines under a decay (F-mu-F), or the static-KT dip-and-1/3-
tail — whose power is smeared across a leakage-elevated low-frequency floor.  The
line-based :func:`~asymmetry.core.fitting.peak_detection.match_multiplets`
therefore almost never fires on exactly the data these families exist for (the
"circular dependency" failure): no detected peaks, no multiplet match.

This module recognises those shapes directly in the time domain by correlating
the (detrended, inverse-variance-weighted) signal against **template banks**
generated with the branch's own component functions — the same functions the
fitter uses, so the matcher can never disagree with the fit about shape:

* ``fmuf`` — collinear F-mu-F polarization over a physical ``r_muF`` grid,
* ``muF``  — single-fluorine mu-F polarization over the same grid,
* ``kt``   — static Gaussian Kubo-Toyabe (``B_L = 0``) over a ``Delta`` grid.

The F-mu-F beat structure and the KT dip survive multiplication by an unknown
smooth envelope, so both template and signal are **monotonically detrended** (a
single ``A e^{-lambda t} + c`` is removed) and then zero-mean/unit-norm
normalised over the SNR-truncated effective window; the score is the normalized
cross-correlation at zero lag, maximised over the grid.  The monotonic detrend is
the crux of the discrimination: a *plain* (or stretched) exponential decay — the
dangerous smooth false-positive for the KT bank — is annihilated by it, while the
non-monotonic KT dip and the F-mu-F oscillation survive.

**Significance.**  A match must clear a null threshold so that pure noise, flat
data, and smooth relaxations never match.  The null is the distribution of the
grid-max score under **phase-randomised surrogates** of the detrended signal
(magnitudes preserved, phases scrambled — this keeps the residual's
autocorrelation, so a smooth residual's surrogates stay smooth rather than being
whitened by a permutation).  Taking the max over the grid *inside* every
surrogate draw is the look-elsewhere correction for the grid scan; the threshold
is the 99th percentile of those maxes.  The surrogate RNG is seeded from the data
so the match/no-match boundary is reproducible.

The module is GUI-free (numpy + scipy + core component functions only).
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.models import longitudinal_field_kubo_toyabe
from asymmetry.core.fitting.muon_fluorine.polarization import (
    linear_fmuf_polarization,
    mu_f_polarization,
)
from asymmetry.core.fitting.peak_detection import (
    MultipletMatch,
    effective_analysis_window,
)

_EPS = 1e-12

#: Physical mu-F bond-length grid (Angstrom). Typical F-mu bond lengths in ionic
#: fluorides are ~1.14-1.20 A; the range is padded either side, at 0.01 A spacing
#: so the recovered r is within a few % (omega_d ~ r^-3, so the shape is a smooth
#: function of r and 0.01 A resolves the beat structure finely enough).
_R_GRID: NDArray[np.float64] = np.round(np.arange(1.00, 1.351, 0.01), 4)

#: Static Gaussian KT width grid (us^-1), covering typical nuclear dipolar widths.
_DELTA_GRID: NDArray[np.float64] = np.round(np.arange(0.05, 1.501, 0.02), 4)

#: Fixed decay rate for the monotonic-detrend seed (us^-1); the fit refines it.
_DETREND_LAMBDA_SEED = 0.2

#: Templates whose in-window variance (before normalisation) is below this share
#: of the bank's peak variance are degenerate (near-flat over the window) and are
#: dropped — normalising a flat template amplifies numerical junk into a spurious
#: "shape" and lets a meaningless grid point win.
_TEMPLATE_VARIANCE_FLOOR = 1e-4

#: Phase-randomised surrogate count and null percentile for the significance test.
_N_SURROGATES = 400
_NULL_PERCENTILE = 99.0

#: Minimum detrended-signal power below which no match is attempted (a genuinely
#: flat record carries no envelope to match and its FFT is numerical dust).
_MIN_SIGNAL_NORM = 1e-9

#: Score below which a match is never reported regardless of the surrogate null
#: (guards the small-sample regime where the null percentile can dip low).
_MIN_ABSOLUTE_SCORE = 0.35


@dataclass(frozen=True)
class _Bank:
    """A normalised template bank: rows are unit-norm detrended templates."""

    kind: str
    family_key: str
    param_name: str  # "r_muF_angstrom" or "Delta"
    grid: NDArray[np.float64]  # physical parameter per surviving row
    matrix: NDArray[np.float64]  # (n_rows, n_win) normalised templates


def _monotonic_detrend(
    t: NDArray[np.float64], y: NDArray[np.float64], weights: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return ``y`` minus its best weighted ``A e^{-lambda t} + c`` fit.

    A single monotonic exponential (plus constant) is removed so that a plain or
    stretched relaxation — which *is* essentially one monotonic decay — collapses
    to ~zero residual, while the non-monotonic KT dip/recovery and the F-mu-F
    oscillation survive.  The fit is nonlinear but runs once per signal and once
    per template (templates are cached), so it is off the per-call hot path.  A
    failed/ill-conditioned fit falls back to weighted-mean subtraction.
    """
    import warnings

    from scipy.optimize import OptimizeWarning, curve_fit

    def _model(tt: NDArray[np.float64], amp: float, lam: float, base: float) -> NDArray[np.float64]:
        # Clip the exponent so the least-squares search never overflows exploring
        # large negative lambda (a growing exponential); the fit stays bounded.
        return amp * np.exp(np.clip(-lam * tt, -700.0, 700.0)) + base

    sigma = 1.0 / np.sqrt(np.clip(weights, _EPS, None))
    p0 = [float(y[0] - y[-1]), _DETREND_LAMBDA_SEED, float(y[-1])]
    try:
        with warnings.catch_warnings():
            # Near-flat data (a genuine null) leaves the covariance unestimable;
            # the fit still yields the best monotonic curve, which is all we need.
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _ = curve_fit(_model, t, y, p0=p0, sigma=sigma, maxfev=2000)
        return y - _model(t, *popt)
    except Exception:
        return y - float(np.average(y, weights=weights))


def _normalise(
    vec: NDArray[np.float64], weights: NDArray[np.float64]
) -> NDArray[np.float64] | None:
    """Weighted zero-mean, unit-norm transform; ``None`` if degenerate.

    Subtracting the weighted mean and scaling by ``sqrt(weights)`` makes the
    zero-lag inner product of two such vectors the inverse-variance-weighted
    normalized cross-correlation, invariant to the amplitude and DC offset of the
    input (an unknown envelope scale multiplies both template and signal).
    """
    w = weights
    mean = float(np.sum(vec * w) / np.sum(w))
    centred = (vec - mean) * np.sqrt(w)
    norm = float(np.linalg.norm(centred))
    if norm <= _MIN_SIGNAL_NORM:
        return None
    return centred / norm


def _template_values(kind: str, t: NDArray[np.float64], param: float) -> NDArray[np.float64]:
    if kind == "fmuf_envelope":
        return np.asarray(linear_fmuf_polarization(t, param), dtype=float)
    if kind == "muF_envelope":
        return np.asarray(mu_f_polarization(t, param), dtype=float)
    # Static Gaussian KT is the B_L = 0 limit of the longitudinal-field KT.
    return np.asarray(longitudinal_field_kubo_toyabe(t, 1.0, param, 0.0, 0.0), dtype=float)


def _build_bank(
    kind: str,
    family_key: str,
    param_name: str,
    grid: NDArray[np.float64],
    t: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> _Bank:
    """Detrend + normalise every grid template, dropping degenerate ones."""
    detrended: list[NDArray[np.float64]] = []
    variances: list[float] = []
    for param in grid:
        raw = _template_values(kind, t, float(param))
        dt = _monotonic_detrend(t, raw, weights)
        detrended.append(dt)
        variances.append(float(np.var(dt)))

    peak_var = max(variances) if variances else 0.0
    floor = _TEMPLATE_VARIANCE_FLOOR * peak_var
    rows: list[NDArray[np.float64]] = []
    params: list[float] = []
    for param, dt, var in zip(grid, detrended, variances, strict=True):
        if var < floor:
            continue
        normed = _normalise(dt, weights)
        if normed is None:
            continue
        rows.append(normed)
        params.append(float(param))

    matrix = np.asarray(rows, dtype=float) if rows else np.zeros((0, t.size), dtype=float)
    return _Bank(
        kind=kind,
        family_key=family_key,
        param_name=param_name,
        grid=np.asarray(params, dtype=float),
        matrix=matrix,
    )


# Bank construction depends only on the (time grid, weights) of the effective
# window, which are identical across the banks of one analysis and repeat across
# analyses of same-shaped records; cache on a cheap fingerprint of both.
_BANK_CACHE: dict[tuple, tuple[_Bank, ...]] = {}
_BANK_CACHE_MAX = 16


def _window_key(t: NDArray[np.float64], weights: NDArray[np.float64]) -> tuple:
    return (
        int(t.size),
        round(float(t[0]), 6),
        round(float(t[-1]), 6),
        round(float(weights[0]), 9),
        round(float(weights[-1]), 9),
        round(float(weights[t.size // 2]), 9),
    )


def _banks_for_window(t: NDArray[np.float64], weights: NDArray[np.float64]) -> tuple[_Bank, ...]:
    key = _window_key(t, weights)
    cached = _BANK_CACHE.get(key)
    if cached is not None:
        return cached
    banks = (
        _build_bank("fmuf_envelope", "fmuf", "r_muF_angstrom", _R_GRID, t, weights),
        _build_bank("muF_envelope", "fmuf", "r_muF_angstrom", _R_GRID, t, weights),
        _build_bank("kt_envelope", "kt", "Delta", _DELTA_GRID, t, weights),
    )
    if len(_BANK_CACHE) >= _BANK_CACHE_MAX:
        _BANK_CACHE.pop(next(iter(_BANK_CACHE)))
    _BANK_CACHE[key] = banks
    return banks


def _surrogate_threshold(
    signal_normed: NDArray[np.float64],
    signal_detrended: NDArray[np.float64],
    weights: NDArray[np.float64],
    matrix: NDArray[np.float64],
    *,
    seed: int,
) -> float:
    """99th-percentile grid-max score under phase-randomised surrogates.

    Surrogates are drawn from the **already-detrended** signal (its power spectrum
    carries the dip/beat structure with the monotonic decay removed), so no
    per-surrogate detrend is needed.  Randomising phases while keeping magnitudes
    preserves the residual autocorrelation — the null for a smooth residual stays
    smooth, which is exactly the regime a permutation null would (wrongly) whiten.
    """
    rng = np.random.default_rng(seed)
    magnitude = np.abs(np.fft.rfft(signal_detrended))
    n = signal_detrended.size
    even = n % 2 == 0
    maxes: list[float] = []
    for _ in range(_N_SURROGATES):
        phases = rng.uniform(0.0, 2.0 * np.pi, size=magnitude.shape)
        phases[0] = 0.0  # keep the (removed) DC real
        if even:
            phases[-1] = 0.0  # Nyquist bin is real for even n
        surrogate = np.fft.irfft(magnitude * np.exp(1j * phases), n=n)
        normed = _normalise(surrogate, weights)
        if normed is None:
            continue
        maxes.append(float(np.max(np.abs(matrix @ normed))))
    if not maxes:
        return 1.0
    return float(np.percentile(maxes, _NULL_PERCENTILE))


def _seed_from_signal(signal_detrended: NDArray[np.float64]) -> int:
    """Deterministic surrogate seed derived from the data (reproducible boundary).

    Uses a stable CRC of the quantised signal — *not* the builtin ``hash()``,
    which is salted per interpreter (``PYTHONHASHSEED``) and would give a
    different surrogate null (hence a different threshold) in every fresh process,
    flaking the match/no-match boundary across CI runs.
    """
    quantised = np.round(signal_detrended, 6).tobytes()
    return int(zlib.crc32(quantised) & 0xFFFFFFFF)


def match_envelope_banks(
    dataset: MuonDataset,
    *,
    field_gauss: float | None = None,
    include_families: frozenset[str] | None = None,
) -> tuple[MultipletMatch, ...]:
    """Match the F-mu-F / mu-F / KT envelope banks against ``dataset``.

    The tail-centred **raw** asymmetry over the SNR-truncated effective window is
    the signal (these envelopes are the dominant component, not a weak line on a
    large relaxation, so the Peak-pass-B residual — from which the KT family is
    itself the detrend model — is *not* used: matching the KT bank against a
    KT-detrended residual would subtract the very shape it looks for).

    Parameters
    ----------
    field_gauss
        Applied field; a KT match records it as the ``B_L`` at which the static
        KT was recognised (informational only — the bank is the ``B_L = 0`` form).
    include_families
        When given, only banks whose ``family_key`` is in this set are evaluated
        (skip out-of-scope work).  ``None`` evaluates every bank.

    Returns
    -------
    tuple[MultipletMatch, ...]
        The best-scoring significant match per bank (empty when none clear the
        surrogate null), quality = the normalized cross-correlation score, with a
        derived ``r_muF_angstrom`` (fmuf/muF) or ``Delta`` (KT).
    """
    t_full = np.asarray(dataset.time, dtype=float)
    y_full = np.asarray(dataset.asymmetry, dtype=float)
    err_full = np.asarray(dataset.error, dtype=float)
    n_full = t_full.size
    if n_full < 3 or y_full.size != n_full or err_full.size != n_full:
        return ()

    end = effective_analysis_window(t_full, err_full)
    t = t_full[:end]
    y = y_full[:end]
    err = err_full[:end]
    if t.size < 3:
        return ()

    # Tail-centre the raw signal (mean of the last ~20 %), mirroring the
    # peak-detection centering convention.
    late = min(t.size, max(5, t.size // 5))
    tail = float(np.mean(y[-late:]))
    signal = y - tail
    weights = 1.0 / np.clip(err, _EPS, None) ** 2

    signal_detrended = _monotonic_detrend(t, signal, weights)
    signal_normed = _normalise(signal_detrended, weights)
    if signal_normed is None:
        return ()

    banks = _banks_for_window(t, weights)
    seed = _seed_from_signal(signal_detrended)

    matches: list[MultipletMatch] = []
    for bank in banks:
        if include_families is not None and bank.family_key not in include_families:
            continue
        if bank.matrix.shape[0] == 0:
            continue
        scores = np.abs(bank.matrix @ signal_normed)
        best_index = int(np.argmax(scores))
        best_score = float(scores[best_index])
        if best_score < _MIN_ABSOLUTE_SCORE:
            continue
        threshold = _surrogate_threshold(
            signal_normed, signal_detrended, weights, bank.matrix, seed=seed
        )
        if best_score <= threshold:
            continue
        best_param = float(bank.grid[best_index])
        matches.append(_build_match(bank, best_index, best_score, best_param, field_gauss))
    return tuple(matches)


def _build_match(
    bank: _Bank,
    index: int,
    score: float,
    param: float,
    field_gauss: float | None,
) -> MultipletMatch:
    if bank.kind == "kt_envelope":
        derived: tuple[tuple[str, float], ...] = (("Delta", param),)
        if field_gauss is not None:
            derived = (*derived, ("B_L", float(field_gauss)))
        note = (
            f"time-domain envelope matches a static Gaussian Kubo-Toyabe with "
            f"Delta = {param:.3g} us^-1 (dip + 1/3 tail; matched-filter score "
            f"{score:.2f})"
        )
    else:
        derived = (("r_muF_angstrom", param),)
        shape = "collinear F-mu-F" if bank.kind == "fmuf_envelope" else "single-fluorine mu-F"
        note = (
            f"time-domain envelope matches a {shape} dipolar signature with "
            f"r_muF = {param:.3g} A (matched-filter score {score:.2f})"
        )
    return MultipletMatch(
        kind=bank.kind,
        family_key=bank.family_key,
        peak_indices=(),  # matched in the time domain — no constituent FFT peaks
        quality=float(np.clip(score, 0.0, 1.0)),
        derived_values=derived,
        note=note,
    )


__all__ = ["match_envelope_banks"]

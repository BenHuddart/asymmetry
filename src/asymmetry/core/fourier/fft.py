"""FFT helpers for μSR asymmetry and grouped detector signals."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fourier.apodisation import (
    ApodisationEarlySignalWarning,
    early_signal_apodisation_loss,
)
from asymmetry.core.fourier.window import apply_fft_filter, apply_window

_DISPLAY_ALIASES = {
    "(power)^1/2": "power_sqrt",
    "(power)1/2": "power_sqrt",
    "(power)½": "power_sqrt",
    "cos": "cos",
    "imaginary": "imaginary",
    "magnitude": "magnitude",
    "phase": "phase_corrected",
    "phase spectrum": "phase_spectrum",
    "phase_opt_real": "phase_opt_real",
    "phaseoptreal": "phase_opt_real",
    "power": "power",
    "real": "real",
    "real+imag": "real_imag",
    "real_imag": "real_imag",
    "resolution (burg)": "burg",
    "burg": "burg",
    "sin": "sin",
    "correlation": "correlation",
    "correlation (radical)": "correlation",
}
_DISPLAY_MODES = frozenset(_DISPLAY_ALIASES.values())


def canonical_fourier_display_mode(display: str) -> str:
    """Return the canonical Fourier display-mode key for *display*."""
    mode = _DISPLAY_ALIASES.get(str(display).strip().lower())
    if mode is None:
        raise ValueError(
            f"Unknown Fourier display mode {display!r}. Expected one of "
            "'(Power)^1/2', 'Phase Spectrum', 'Cos', 'Sin', 'Phase', 'phaseOptReal', "
            "or the legacy modes 'Real', 'Imaginary', 'Magnitude', 'Power'."
        )
    return mode


def fourier_mode_uses_phase_correction(display: str) -> bool:
    """Return whether *display* consumes the phase-corrected spectrum."""
    return canonical_fourier_display_mode(display) == "phase_corrected"


def fourier_mode_uses_entropy_optimizer(display: str) -> bool:
    """Return whether *display* requires the entropy-based phase optimizer.

    The ``phaseOptReal`` mode runs musrfit's ``PFTPhaseCorrection`` algorithm
    (entropy + penalty minimisation) on the averaged complex spectrum rather
    than consuming a manually supplied or table-driven phase correction.
    """
    return canonical_fourier_display_mode(display) == "phase_opt_real"


def _normalize_phase_degrees(value: float) -> float:
    """Wrap an angle into the half-open interval [-180, 180)."""
    wrapped = (float(value) + 180.0) % 360.0 - 180.0
    if np.isclose(wrapped, -180.0):
        return 180.0
    return wrapped


def _weighted_average_signal(
    signal: NDArray[np.float64],
    error: NDArray[np.float64] | None = None,
) -> float:
    """Return the WiMDA-style error-weighted mean of *signal* over finite bins.

    WiMDA weights each bin by ``1 / error`` (proportional to counts) when usable
    errors are available; otherwise it falls back to the finite arithmetic mean.
    Returns NaN only when the signal has no finite samples at all.
    """
    values = np.asarray(signal, dtype=np.float64)
    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return float("nan")

    if error is not None:
        err = np.asarray(error, dtype=np.float64)
        if err.shape == values.shape:
            weights = np.zeros_like(values, dtype=np.float64)
            valid_weights = finite_mask & np.isfinite(err) & (err > 0.0)
            weights[valid_weights] = 1.0 / err[valid_weights]
            weight_sum = float(np.sum(weights[valid_weights], dtype=np.float64))
            if weight_sum > 0.0:
                return float(np.sum(weights * values, dtype=np.float64) / weight_sum)
    return float(np.mean(values[finite_mask], dtype=np.float64))


def _subtract_average_signal(
    signal: NDArray[np.float64],
    error: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Return ``signal`` with its WiMDA-style average removed.

    WiMDA subtracts an error-weighted average before filtering and FFT (see
    :func:`_weighted_average_signal`).
    """
    values = np.asarray(signal, dtype=np.float64).copy()
    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return values
    values[finite_mask] -= _weighted_average_signal(values, error)
    return values


#: Highest polynomial order :func:`_detrend_baseline` will fit. A baseline
#: removal is meant to take out a *trend* the oscillation rides on; past cubic a
#: polynomial starts fitting the oscillation itself and quietly deletes the line
#: the transform exists to find, so the cap is enforced rather than advised.
_MAX_DETREND_ORDER = 3


def _detrend_baseline(
    time: NDArray[np.float64],
    signal: NDArray[np.float64],
    error: NDArray[np.float64] | None,
    detrend: int | Callable[[NDArray[np.float64]], NDArray[np.float64]],
    *,
    caller: str,
) -> NDArray[np.float64]:
    """Return the baseline ``detrend`` asks to be subtracted from *signal*.

    An ``int`` fits an error-weighted least-squares polynomial of that order
    (weights ``1/σ``, so the fit minimises ``Σ (residual/σ)²`` — the standard
    least-squares weighting, *not* the ``1/σ`` WiMDA average weighting of
    :func:`_weighted_average_signal`) and returns it evaluated on *time*. A
    callable is invoked as ``f(time)`` and its return used as-is, letting a
    caller subtract a slow model they fitted elsewhere.
    """
    times = np.asarray(time, dtype=np.float64)
    if callable(detrend):
        baseline = np.asarray(detrend(times), dtype=np.float64)
        if baseline.shape != signal.shape:
            raise ValueError(
                f"{caller}: the detrend callable returned shape {baseline.shape}, "
                f"but the cropped signal has shape {signal.shape}."
            )
        if not np.all(np.isfinite(baseline)):
            raise ValueError(
                f"{caller}: the detrend callable returned non-finite values (NaN or inf)."
            )
        return baseline

    if isinstance(detrend, bool) or not isinstance(detrend, (int, np.integer)):
        raise ValueError(
            f"{caller}: detrend must be None, an integer polynomial order, or a "
            f"callable f(time) -> baseline; got {detrend!r}."
        )
    order = int(detrend)
    if order < 0 or order > _MAX_DETREND_ORDER:
        raise ValueError(
            f"{caller}: detrend={order} is outside the supported polynomial "
            f"orders 0–{_MAX_DETREND_ORDER}. A higher-order polynomial fits the "
            "oscillation itself rather than the trend it rides on; crop the "
            "record or subtract a fitted model through the callable form instead."
        )

    usable = np.isfinite(times) & np.isfinite(signal)
    weights: NDArray[np.float64] | None = None
    if error is not None and error.shape == signal.shape:
        weighted = usable & np.isfinite(error) & (error > 0.0)
        if int(np.count_nonzero(weighted)) > order:
            usable = weighted
            weights = 1.0 / error[usable]
    if int(np.count_nonzero(usable)) <= order:
        raise ValueError(
            f"{caller}: detrend={order} needs more than {order} usable point(s); "
            f"the cropped signal has {int(np.count_nonzero(usable))}."
        )
    fitted = np.polynomial.Polynomial.fit(times[usable], signal[usable], deg=order, w=weights)
    return np.asarray(fitted(times), dtype=np.float64)


def _resolve_baseline_removal(
    subtract_average_signal: bool | None,
    detrend: int | Callable[[NDArray[np.float64]], NDArray[np.float64]] | None,
    *,
    caller: str,
) -> bool:
    """Return whether the WiMDA average subtraction runs, rejecting ambiguity.

    ``subtract_average_signal`` defaults to ``None`` meaning "the historical
    ``True``, unless ``detrend`` takes over". A fitted baseline subsumes the
    mean, so the two are alternatives, and asking for both explicitly is
    ambiguous (a callable baseline need not be mean-free) — that combination
    raises rather than being guessed at.
    """
    if detrend is None:
        return True if subtract_average_signal is None else bool(subtract_average_signal)
    if subtract_average_signal:
        raise ValueError(
            f"{caller}: detrend= and subtract_average_signal=True are alternatives, "
            "not a sequence — a fitted baseline already removes the mean, and a "
            "callable baseline need not be mean-free. Pass detrend= alone (leave "
            "subtract_average_signal unset) or pass subtract_average_signal=False "
            "explicitly."
        )
    return False


@dataclass
class PreparedFFTSignal:
    """Preprocessed FFT time-domain input plus the calibration it enables.

    ``window_sum`` is the coherent gain of the apodisation actually applied over
    the ``n`` populated (unpadded) samples — ``Σ wₙ`` — used by the amplitude
    calibration (``2 / Σw``); with no apodisation it is exactly ``n``. When
    fractional footing runs, ``fractional_baseline`` records the error-weighted
    baseline ``N₀`` the signal was divided by, and ``fractional_applied`` is
    true; the degenerate guard (non-positive/non-finite baseline) leaves the
    signal on its raw footing with ``fractional_applied`` false.
    """

    signal: NDArray[np.float64]
    dt: float
    window_sum: float
    fractional_baseline: float | None = None
    fractional_applied: bool = False


def _clip_fft_window(
    time: NDArray[np.float64],
    asymmetry: NDArray[np.float64],
    error: NDArray[np.float64] | None,
    t_min: float | None,
    t_max: float | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
    """Restrict a ``(t, A, σ)`` triple to ``[t_min, t_max]`` inclusive.

    The single crop seam shared by :func:`prepare_fft_time_signal` and
    :func:`prepare_fft_arrays`, reproducing
    :meth:`asymmetry.core.data.dataset.MuonDataset.time_range` exactly: the
    bounds are inclusive, ``None`` for both is a no-op that passes the arrays
    through untouched, and a clipped result is a copy. ``error`` may be ``None``
    (a bare array caller need not supply one).
    """
    if t_min is None and t_max is None:
        return time, asymmetry, error
    mask = np.ones(len(time), dtype=bool)
    if t_min is not None:
        mask &= time >= t_min
    if t_max is not None:
        mask &= time <= t_max
    return (
        time[mask].copy(),
        asymmetry[mask].copy(),
        None if error is None else error[mask].copy(),
    )


def _validated_fft_arrays(
    time: NDArray[np.float64] | Sequence[float],
    asymmetry: NDArray[np.float64] | Sequence[float],
    error: NDArray[np.float64] | Sequence[float] | None,
    *,
    caller: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
    """Coerce and validate a raw ``(t, A, σ)`` triple for the array front doors.

    Returns the arrays as ``float64`` (``error`` stays ``None`` when omitted).
    Raises :class:`ValueError` with a message naming the offending array for a
    non-1-D shape, a length mismatch, an empty triple, or any non-finite entry —
    the validation domain a
    :class:`~asymmetry.core.data.dataset.MuonDataset` would otherwise have
    carried implicitly. Mirrors ``FitEngine.fit_arrays``' guards; *caller* names
    the public entry point in the message.
    """
    coerced: dict[str, NDArray[np.float64]] = {}
    named: list[tuple[str, object]] = [("time", time), ("asymmetry", asymmetry)]
    if error is not None:
        named.append(("error", error))
    for name, values in named:
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(
                f"{caller}: {name} must be a one-dimensional array, got shape {arr.shape}."
            )
        coerced[name] = arr

    n_time = coerced["time"].size
    for name in ("asymmetry", "error"):
        arr = coerced.get(name)
        if arr is not None and arr.size != n_time:
            raise ValueError(
                f"{caller}: time, asymmetry and error must have the same length; "
                f"time has {n_time} point(s) but {name} has {arr.size}."
            )
    if n_time == 0:
        raise ValueError(f"{caller}: time and asymmetry are empty — nothing to transform.")

    for name, arr in coerced.items():
        n_bad = int(np.count_nonzero(~np.isfinite(arr)))
        if n_bad:
            raise ValueError(
                f"{caller}: {name} contains {n_bad} non-finite value(s) (NaN or inf). "
                "Drop or interpolate those points before transforming."
            )
    return coerced["time"], coerced["asymmetry"], coerced.get("error")


def _warn_on_early_signal_suppression(
    time: NDArray[np.float64],
    signal: NDArray[np.float64],
    weights: NDArray[np.float64],
    error: NDArray[np.float64] | None,
    *,
    window: str,
) -> None:
    """Warn when the apodisation in force has deleted the early-time signal.

    Advisory only — nothing about the returned spectrum changes. The failure
    mode this exists for is concrete: a symmetric taper (``hann``/``cosine``) is
    exactly zero at the first sample, so a heavily damped oscillation whose
    lifetime is a small fraction of the record can vanish into the noise floor
    under Hann while being many-sigma without it. The metric and its two
    thresholds live in
    :func:`~asymmetry.core.fourier.apodisation.early_signal_apodisation_loss`.

    Requirements the calibration satisfies (pinned by the synthetic study in
    ``tests/core/test_fourier_apodisation_guard.py``):

    * a heavily damped cosine (λ ≫ 1/record length) under ``hann`` or
      ``cosine`` **must** warn;
    * the same signal with ``window="none"``, or with the matched exponential
      filter (``window="lorentzian"``, ``filter_time_constant_us ≈ 1/λ``),
      **must not**;
    * a conventional transverse-field record whose oscillation is alive across
      the whole window **must not**, under any apodisation.
    """
    loss = early_signal_apodisation_loss(time, signal, weights, error)
    if loss is None or not loss.triggered:
        return
    warnings.warn(
        f"Apodisation window {window!r} has deleted the early-time signal: "
        f"{loss.early_power_fraction:.1%} of this record's error-weighted power "
        f"lies in its first {loss.early_window_fraction:.0%}, and the window keeps "
        f"only {loss.early_retained_fraction:.2%} of it. A symmetric taper is zero "
        "at t = 0, so a heavily damped oscillation can vanish entirely under it "
        "while being many-sigma without it. For early-time work use "
        'window="none" with a t_max crop, or the exponential filter '
        '(window="lorentzian", filter_time_constant_us ~ 1/lambda), which is '
        "weight-1 at the start and is the matched apodisation for a Lorentzian "
        "line.",
        ApodisationEarlySignalWarning,
        stacklevel=4,
    )


def _prepare_fft_core(
    time: NDArray[np.float64],
    asymmetry: NDArray[np.float64],
    error: NDArray[np.float64] | None,
    *,
    caller: str,
    window: str,
    subtract_average_signal: bool | None,
    filter_start_us: float,
    filter_time_constant_us: float,
    fractional: bool,
    detrend: int | Callable[[NDArray[np.float64]], NDArray[np.float64]] | None,
) -> PreparedFFTSignal:
    """Preprocess an already-cropped ``(t, A, σ)`` triple for the FFT.

    The one implementation behind :func:`prepare_fft_time_signal` and
    :func:`prepare_fft_arrays`; the arrays arrive pre-cropped to the transform
    window by :func:`_clip_fft_window`, so the two front doors cannot drift, and
    the baseline-removal and apodisation-safety behaviour reaches both paths
    from one place.
    """
    subtract_average = _resolve_baseline_removal(subtract_average_signal, detrend, caller=caller)
    times = np.asarray(time, dtype=np.float64)
    signal = np.asarray(asymmetry, dtype=np.float64).copy()
    err = np.asarray(error, dtype=np.float64) if error is not None else None
    dt = np.mean(np.diff(times)) if times.size > 1 else 1.0

    fractional_baseline: float | None = None
    fractional_applied = False
    if fractional:
        baseline = _weighted_average_signal(signal, err)
        if np.isfinite(baseline) and baseline > 0.0:
            signal = signal / baseline
            if err is not None:
                err = err / baseline
            fractional_baseline = float(baseline)
            fractional_applied = True
        # Degenerate baseline (empty/zero/negative mean): fall back to the raw
        # footing rather than dividing by ~0. The caller stamps a note.

    if subtract_average:
        signal = _subtract_average_signal(signal, err)
    elif detrend is not None:
        signal = signal - _detrend_baseline(times, signal, err, detrend, caller=caller)

    window_key = str(window).strip().lower()
    if window_key in {"none", "gaussian", "lorentzian"}:
        weights = apply_fft_filter(
            np.ones_like(signal),
            times,
            mode=window_key,
            start_time_us=float(filter_start_us),
            time_constant_us=float(filter_time_constant_us),
        )
    else:
        weights = apply_window(np.ones_like(signal), window_key)
    _warn_on_early_signal_suppression(times, signal, weights, err, window=window_key)
    windowed = signal * weights

    window_sum = float(np.sum(weights, dtype=np.float64))
    return PreparedFFTSignal(
        signal=windowed,
        dt=float(dt),
        window_sum=window_sum,
        fractional_baseline=fractional_baseline,
        fractional_applied=fractional_applied,
    )


def prepare_fft_time_signal(
    dataset: MuonDataset,
    *,
    window: str = "none",
    t_min: float | None = None,
    t_max: float | None = None,
    subtract_average_signal: bool | None = None,
    filter_start_us: float = 0.0,
    filter_time_constant_us: float = 1.5,
    fractional: bool = False,
    detrend: int | Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
) -> PreparedFFTSignal:
    """Return the preprocessed real time-domain signal and its calibration.

    Applies (in order) the time crop, optional fractional footing, the baseline
    removal (``subtract_average_signal`` or ``detrend``), and the apodisation
    filter/window — the exact preprocessing :func:`fft_complex_asymmetry` feeds
    to the FFT, so an all-poles (Burg) estimate can share the same input.

    ``window`` accepts the WiMDA *filters* (``"none"``, ``"gaussian"``,
    ``"lorentzian"`` — weight-1 at ``t = 0``) and the symmetric *tapers*
    (``"hann"``, ``"cosine"`` — zero at the record's ends). A taper deletes
    early-time signal and is for late-time / narrow-line work only; when it
    detects that a taper has erased a front-loaded signal this function raises
    :class:`~asymmetry.core.fourier.apodisation.ApodisationEarlySignalWarning`,
    which names the numbers and the two alternatives (no window with a crop, or
    the exponential filter at ``filter_time_constant_us ≈ 1/λ``). The warning is
    advisory: nothing about the returned value changes.

    To preprocess a bare ``(t, A, σ)`` triple with no dataset in hand, use
    :func:`prepare_fft_arrays`, which shares this function's entire
    implementation.

    When ``fractional`` is set the signal (a lifetime-corrected count scale) and
    its error are divided by the error-weighted baseline ``N₀`` so a
    ``N₀·(1 + A·cos)`` signal becomes ``1 + A·cos``; the subsequent average
    subtraction then removes the residual DC, leaving ``A·cos``.  This puts the
    spectrum on a fractional-asymmetry footing invariant to counting statistics.

    Baseline removal
    ----------------
    ``subtract_average_signal`` removes a *constant* — WiMDA's error-weighted
    average. A signal that relaxes across the record does not ride on a
    constant, and the residual trend dumps power into the low-frequency bins,
    where it can swamp a genuine line. ``detrend`` removes a trend instead:

    * ``None`` (the default) — no detrend; ``subtract_average_signal`` behaves
      exactly as it always has.
    * an ``int`` ``n`` (``0 ≤ n ≤ 3``) — an **error-weighted least-squares
      polynomial** of order ``n`` is fitted to the cropped, post-fractional
      signal and subtracted. Keep the order low: a high-order polynomial fits
      the oscillation itself, so orders above 3 are rejected rather than
      silently deleting the line.
    * a **callable** ``f(time) -> baseline`` — subtracted as given, for
      removing a slow model fitted elsewhere. It receives the cropped time axis
      and must return an array of the same shape.

    ``detrend`` is applied **instead of** the average subtraction, not after it
    (a fitted polynomial already subsumes the mean). ``subtract_average_signal``
    therefore defaults to ``None``, meaning "the historical ``True``, unless
    ``detrend`` takes over"; passing ``detrend=`` together with an explicit
    ``subtract_average_signal=True`` raises :class:`ValueError` rather than
    guessing an order for the two.
    """
    time, asymmetry, error = _clip_fft_window(
        dataset.time,
        dataset.asymmetry,
        np.asarray(dataset.error, dtype=np.float64) if dataset.error is not None else None,
        t_min,
        t_max,
    )
    return _prepare_fft_core(
        time,
        asymmetry,
        error,
        caller="prepare_fft_time_signal",
        window=window,
        subtract_average_signal=subtract_average_signal,
        filter_start_us=filter_start_us,
        filter_time_constant_us=filter_time_constant_us,
        fractional=fractional,
        detrend=detrend,
    )


def prepare_fft_arrays(
    time: NDArray[np.float64] | Sequence[float],
    asymmetry: NDArray[np.float64] | Sequence[float],
    error: NDArray[np.float64] | Sequence[float] | None = None,
    *,
    window: str = "none",
    t_min: float | None = None,
    t_max: float | None = None,
    subtract_average_signal: bool | None = None,
    filter_start_us: float = 0.0,
    filter_time_constant_us: float = 1.5,
    fractional: bool = False,
    detrend: int | Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
) -> PreparedFFTSignal:
    """Preprocess a bare ``(t, A, σ)`` triple — no :class:`MuonDataset` required.

    The array-taking front door onto the same preprocessing as
    :func:`prepare_fft_time_signal`: both crop through one seam and then run one
    shared internal core, so
    ``prepare_fft_arrays(ds.time, ds.asymmetry, ds.error, …)`` returns exactly
    what ``prepare_fft_time_signal(ds, …)`` returns.

    Parameters
    ----------
    time : array_like
        Time axis in microseconds, one dimension.
    asymmetry : array_like
        Signal values to transform. The FFT is scale-linear, so whatever scale
        goes in comes out: a percent-scale curve (the
        :class:`~asymmetry.core.data.dataset.MuonDataset` convention) yields a
        percent-scaled spectrum, a fraction-scale curve a fraction-scaled one.
        Nothing here rescales the input — see "Asymmetry units across the API".
    error : array_like, optional
        1σ uncertainty on ``asymmetry``, on the same scale. Used only for the
        error-weighted average subtraction and the fractional baseline; omit it
        (``None``) and those fall back to their unweighted forms.
    window, t_min, t_max, subtract_average_signal, filter_start_us, \
filter_time_constant_us, fractional, detrend
        Exactly as documented on :func:`prepare_fft_time_signal` (including its
        "Baseline removal" section for ``detrend``).
        ``t_min``/``t_max`` crop the arrays with the same inclusive bounds
        :meth:`~asymmetry.core.data.dataset.MuonDataset.time_range` applies.

    Raises
    ------
    ValueError
        When the arrays are not one-dimensional, have unequal lengths, are
        empty, contain non-finite values, or when ``t_min``/``t_max`` leave no
        points. :func:`prepare_fft_time_signal` does not validate its dataset
        this way (its behaviour is unchanged); these guards exist because a raw
        triple has no container to have checked them earlier.
    """
    return _prepare_fft_arrays(
        time,
        asymmetry,
        error,
        caller="prepare_fft_arrays",
        window=window,
        t_min=t_min,
        t_max=t_max,
        subtract_average_signal=subtract_average_signal,
        filter_start_us=filter_start_us,
        filter_time_constant_us=filter_time_constant_us,
        fractional=fractional,
        detrend=detrend,
    )


def _prepare_fft_arrays(
    time: NDArray[np.float64] | Sequence[float],
    asymmetry: NDArray[np.float64] | Sequence[float],
    error: NDArray[np.float64] | Sequence[float] | None,
    *,
    caller: str,
    window: str,
    t_min: float | None,
    t_max: float | None,
    subtract_average_signal: bool | None,
    filter_start_us: float,
    filter_time_constant_us: float,
    fractional: bool,
    detrend: int | Callable[[NDArray[np.float64]], NDArray[np.float64]] | None,
) -> PreparedFFTSignal:
    """Validate, crop and preprocess a bare triple on behalf of *caller*.

    Shared by every array front door so they report the same guard messages
    under their own names.
    """
    time_arr, asym_arr, err_arr = _validated_fft_arrays(time, asymmetry, error, caller=caller)
    clipped_time, clipped_asym, clipped_err = _clip_fft_window(
        time_arr, asym_arr, err_arr, t_min, t_max
    )
    if clipped_time.size == 0:
        raise ValueError(
            f"{caller}: no data points remain in the transform window "
            f"t_min={t_min!r}, t_max={t_max!r} (the input spans "
            f"{float(time_arr.min()):g}–{float(time_arr.max()):g} µs)."
        )
    return _prepare_fft_core(
        clipped_time,
        clipped_asym,
        clipped_err,
        caller=caller,
        window=window,
        subtract_average_signal=subtract_average_signal,
        filter_start_us=filter_start_us,
        filter_time_constant_us=filter_time_constant_us,
        fractional=fractional,
        detrend=detrend,
    )


def fft_complex_asymmetry(
    dataset: MuonDataset,
    window: str = "none",
    padding_factor: int = 1,
    t_min: float | None = None,
    t_max: float | None = None,
    phase_degrees: float = 0.0,
    t0_offset_us: float = 0.0,
    subtract_average_signal: bool | None = None,
    filter_start_us: float = 0.0,
    filter_time_constant_us: float = 1.5,
    fractional: bool = False,
    amplitude_calibration: bool = False,
    diagnostics: dict | None = None,
    detrend: int | Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Compute the phase-rotated complex FFT of the asymmetry signal.

    Parameters
    ----------
    dataset
        The time-domain data.
    window
        Apodization window name (``"none"``, ``"gaussian"``, ``"hann"``,
        ``"cosine"``, ``"lorentzian"``). ``"gaussian"``/``"lorentzian"`` are the
        WiMDA *filters* — weight-1 at ``t = 0``, decaying with
        ``filter_time_constant_us`` — while ``"hann"``/``"cosine"`` are
        **symmetric tapers**, zero at the record's ends. A taper is for
        late-time / narrow-line work and DELETES early-time signal: a heavily
        damped oscillation (lifetime ≪ record length) can vanish entirely under
        Hann while being many-sigma with ``"none"`` and a crop, or with the
        exponential filter at ``filter_time_constant_us ≈ 1/λ`` (the matched
        apodisation for a Lorentzian line). This path warns when it detects
        that case — see
        :class:`~asymmetry.core.fourier.apodisation.ApodisationEarlySignalWarning`.
    padding_factor
        Zero-pad the signal to ``padding_factor × N``.
    t_min, t_max
        Restrict the time range before transforming.
    phase_degrees
        Manual phase correction in degrees. A positive value applies the usual
        phase-correction rotation ``exp(-i * phi)`` to the complex spectrum.
    t0_offset_us
        Additional WiMDA-style time-zero offset in microseconds. This applies a
        frequency-dependent phase term ``exp(-i * 2π f t0)`` after the FFT.
    subtract_average_signal
        When true, subtract the WiMDA-style pre-FFT average signal before any
        window is applied. Defaults to ``None``, which means ``True`` unless
        ``detrend`` takes the baseline removal over.
    filter_start_us
        WiMDA-style filter start time in microseconds for ``"gaussian"`` and
        ``"lorentzian"`` FFT filtering.
    filter_time_constant_us
        WiMDA-style filter time constant in microseconds for ``"gaussian"`` and
        ``"lorentzian"`` FFT filtering.
    fractional
        When true, put the signal on a fractional-asymmetry footing by dividing
        it (and its error) by the error-weighted baseline before averaging and
        the FFT (see :func:`prepare_fft_time_signal`).
    amplitude_calibration
        When true, multiply the complex spectrum by the coherent-gain / percent
        factor ``100 · 2 / Σw`` (``Σw`` the apodisation coherent gain over the
        unpadded samples), so a pure cosine of fractional amplitude ``A`` peaks
        at ``100·A`` in the magnitude spectrum — invariant to counting
        statistics, window length, apodisation, and zero padding.
    diagnostics
        Optional mutable dict; when supplied it is populated with the
        ``window_sum``, ``fractional_baseline``, ``fractional_applied`` and
        ``amplitude_calibrated`` used, so callers can stamp provenance/guard
        notes without a second pass.
    detrend
        Baseline removal for a signal that relaxes across the record:
        ``None``, an integer polynomial order (``0``–``3``, fitted
        error-weighted), or a callable ``f(time) -> baseline``. Applied
        *instead of* ``subtract_average_signal`` — see "Baseline removal" in
        :func:`prepare_fft_time_signal`.

    Returns
    -------
    frequencies, spectrum
        Frequency axis (MHz) and the complex, optionally phase-rotated spectrum.
    """
    prepared = prepare_fft_time_signal(
        dataset,
        window=window,
        t_min=t_min,
        t_max=t_max,
        subtract_average_signal=subtract_average_signal,
        filter_start_us=filter_start_us,
        filter_time_constant_us=filter_time_constant_us,
        fractional=fractional,
        detrend=detrend,
    )
    return _fft_complex_core(
        prepared,
        padding_factor=padding_factor,
        phase_degrees=phase_degrees,
        t0_offset_us=t0_offset_us,
        amplitude_calibration=amplitude_calibration,
        diagnostics=diagnostics,
    )


def _fft_complex_core(
    prepared: PreparedFFTSignal,
    *,
    padding_factor: int,
    phase_degrees: float,
    t0_offset_us: float,
    amplitude_calibration: bool,
    diagnostics: dict | None,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Transform an already-preprocessed signal into a calibrated complex spectrum.

    The one implementation behind :func:`fft_complex_asymmetry` and
    :func:`fft_complex_arrays`: everything downstream of
    :class:`PreparedFFTSignal` — padding, the coherent-gain calibration, the
    phase rotation, and the diagnostics stamp — lives here once.
    """
    signal, dt = prepared.signal, prepared.dt

    n = len(signal)
    n_padded = n * max(padding_factor, 1)

    spectrum = np.fft.rfft(signal, n=n_padded)
    freqs = np.fft.rfftfreq(n_padded, d=dt)  # MHz (since dt is in µs)

    # Coherent-gain / length / percent calibration is a single positive real
    # scalar, so it commutes with the phase rotation and leaves every phase and
    # angle-derived display mode correct.  Σw is taken over the unpadded samples
    # (zero-padding contributes nothing), so the calibrated peak height is
    # independent of the padding factor and the time-window length.  The DC and
    # Nyquist bins strictly carry a 1/Σw one-sided gain rather than 2/Σw, but DC
    # is removed by the average subtraction and test tones avoid Nyquist.
    if amplitude_calibration and prepared.window_sum > 0.0:
        spectrum = spectrum * (100.0 * 2.0 / prepared.window_sum)

    if phase_degrees or t0_offset_us:
        phase = np.deg2rad(float(phase_degrees)) + 2.0 * np.pi * freqs * float(t0_offset_us)
        spectrum = spectrum * np.exp(-1j * phase)

    if diagnostics is not None:
        diagnostics["window_sum"] = prepared.window_sum
        diagnostics["fractional_baseline"] = prepared.fractional_baseline
        diagnostics["fractional_applied"] = prepared.fractional_applied
        diagnostics["amplitude_calibrated"] = bool(amplitude_calibration)

    return freqs, spectrum


def fft_complex_arrays(
    time: NDArray[np.float64] | Sequence[float],
    asymmetry: NDArray[np.float64] | Sequence[float],
    error: NDArray[np.float64] | Sequence[float] | None = None,
    *,
    window: str = "none",
    padding_factor: int = 1,
    t_min: float | None = None,
    t_max: float | None = None,
    phase_degrees: float = 0.0,
    t0_offset_us: float = 0.0,
    subtract_average_signal: bool | None = None,
    filter_start_us: float = 0.0,
    filter_time_constant_us: float = 1.5,
    fractional: bool = False,
    amplitude_calibration: bool = False,
    diagnostics: dict | None = None,
    detrend: int | Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Complex FFT of a bare ``(t, A, σ)`` triple — no :class:`MuonDataset` required.

    The array-taking front door onto the same transform as
    :func:`fft_complex_asymmetry`: both preprocess through one shared core and
    then run one shared spectral core, so
    ``fft_complex_arrays(ds.time, ds.asymmetry, ds.error, …)`` returns exactly
    the frequencies and spectrum ``fft_complex_asymmetry(ds, …)`` returns —
    equal bit for bit, with no tolerance. Use it for any curve that is not a
    dataset's default reduction (a custom detector pairing, a detrended or
    background-corrected export, a hand-built model curve) instead of
    synthesising a ``MuonDataset`` to carry it.

    Parameters
    ----------
    time, asymmetry, error
        As documented on :func:`prepare_fft_arrays`. The FFT is *scale-linear*:
        percent in, percent-scaled spectrum out.
    window, padding_factor, t_min, t_max, phase_degrees, t0_offset_us, \
subtract_average_signal, filter_start_us, filter_time_constant_us, fractional, \
amplitude_calibration, diagnostics, detrend
        Exactly as documented on :func:`fft_complex_asymmetry`, keyword-only
        here because ``error`` is optional.

    Raises
    ------
    ValueError
        As :func:`prepare_fft_arrays`.
    """
    return _fft_complex_arrays(
        time,
        asymmetry,
        error,
        caller="fft_complex_arrays",
        window=window,
        padding_factor=padding_factor,
        t_min=t_min,
        t_max=t_max,
        phase_degrees=phase_degrees,
        t0_offset_us=t0_offset_us,
        subtract_average_signal=subtract_average_signal,
        filter_start_us=filter_start_us,
        filter_time_constant_us=filter_time_constant_us,
        fractional=fractional,
        amplitude_calibration=amplitude_calibration,
        diagnostics=diagnostics,
        detrend=detrend,
    )


def _fft_complex_arrays(
    time: NDArray[np.float64] | Sequence[float],
    asymmetry: NDArray[np.float64] | Sequence[float],
    error: NDArray[np.float64] | Sequence[float] | None,
    *,
    caller: str,
    window: str,
    padding_factor: int,
    t_min: float | None,
    t_max: float | None,
    phase_degrees: float,
    t0_offset_us: float,
    subtract_average_signal: bool | None,
    filter_start_us: float,
    filter_time_constant_us: float,
    fractional: bool,
    amplitude_calibration: bool,
    diagnostics: dict | None,
    detrend: int | Callable[[NDArray[np.float64]], NDArray[np.float64]] | None,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Transform a bare triple on behalf of *caller* (whose name the guards use)."""
    prepared = _prepare_fft_arrays(
        time,
        asymmetry,
        error,
        caller=caller,
        window=window,
        t_min=t_min,
        t_max=t_max,
        subtract_average_signal=subtract_average_signal,
        filter_start_us=filter_start_us,
        filter_time_constant_us=filter_time_constant_us,
        fractional=fractional,
        detrend=detrend,
    )
    return _fft_complex_core(
        prepared,
        padding_factor=padding_factor,
        phase_degrees=phase_degrees,
        t0_offset_us=t0_offset_us,
        amplitude_calibration=amplitude_calibration,
        diagnostics=diagnostics,
    )


def fft_asymmetry(
    dataset: MuonDataset,
    window: str = "none",
    padding_factor: int = 1,
    t_min: float | None = None,
    t_max: float | None = None,
    phase_degrees: float = 0.0,
    t0_offset_us: float = 0.0,
    subtract_average_signal: bool | None = None,
    filter_start_us: float = 0.0,
    filter_time_constant_us: float = 1.5,
    fractional: bool = False,
    amplitude_calibration: bool = False,
    detrend: int | Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute the FFT of the asymmetry signal.

    Parameters
    ----------
    dataset
        The time-domain data.
    window
        Apodization window name (``"none"``, ``"gaussian"``, ``"hann"``,
        ``"cosine"``, ``"lorentzian"``). ``"gaussian"``/``"lorentzian"`` are the
        WiMDA *filters* — weight-1 at ``t = 0``, decaying with
        ``filter_time_constant_us`` — while ``"hann"``/``"cosine"`` are
        **symmetric tapers**, zero at the record's ends. A taper is for
        late-time / narrow-line work and DELETES early-time signal: a heavily
        damped oscillation (lifetime ≪ record length) can vanish entirely under
        Hann while being many-sigma with ``"none"`` and a crop, or with the
        exponential filter at ``filter_time_constant_us ≈ 1/λ`` (the matched
        apodisation for a Lorentzian line). This path warns when it detects
        that case — see
        :class:`~asymmetry.core.fourier.apodisation.ApodisationEarlySignalWarning`.
    padding_factor
        Zero-pad the signal to ``padding_factor × N``.
    t_min, t_max
        Restrict the time range before transforming.
    phase_degrees
        Manual phase correction in degrees applied to the complex spectrum
        before the real and magnitude outputs are derived.
    t0_offset_us
        Additional WiMDA-style time-zero offset in microseconds. This adds the
        frequency-dependent phase term before the real and magnitude outputs are
        derived.
    subtract_average_signal
        When true, subtract the WiMDA-style pre-FFT average signal before any
        window is applied. Defaults to ``None``, which means ``True`` unless
        ``detrend`` takes the baseline removal over.
    filter_start_us
        WiMDA-style filter start time in microseconds for ``"gaussian"`` and
        ``"lorentzian"`` FFT filtering.
    filter_time_constant_us
        WiMDA-style filter time constant in microseconds for ``"gaussian"`` and
        ``"lorentzian"`` FFT filtering.
    detrend
        Baseline removal for a signal that relaxes across the record:
        ``None``, an integer polynomial order (``0``–``3``, fitted
        error-weighted), or a callable ``f(time) -> baseline``. Applied
        *instead of* ``subtract_average_signal`` — see "Baseline removal" in
        :func:`prepare_fft_time_signal`.

    Returns
    -------
    frequencies, real_part, magnitude
        Frequency axis (MHz) and the real and magnitude spectra.
    """
    freqs, spectrum = fft_complex_asymmetry(
        dataset,
        window=window,
        padding_factor=padding_factor,
        t_min=t_min,
        t_max=t_max,
        phase_degrees=phase_degrees,
        t0_offset_us=t0_offset_us,
        subtract_average_signal=subtract_average_signal,
        filter_start_us=filter_start_us,
        filter_time_constant_us=filter_time_constant_us,
        fractional=fractional,
        amplitude_calibration=amplitude_calibration,
        detrend=detrend,
    )

    return freqs, spectrum.real, np.abs(spectrum)


def fft_arrays(
    time: NDArray[np.float64] | Sequence[float],
    asymmetry: NDArray[np.float64] | Sequence[float],
    error: NDArray[np.float64] | Sequence[float] | None = None,
    *,
    window: str = "none",
    padding_factor: int = 1,
    t_min: float | None = None,
    t_max: float | None = None,
    phase_degrees: float = 0.0,
    t0_offset_us: float = 0.0,
    subtract_average_signal: bool | None = None,
    filter_start_us: float = 0.0,
    filter_time_constant_us: float = 1.5,
    fractional: bool = False,
    amplitude_calibration: bool = False,
    detrend: int | Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """FFT a bare ``(t, A, σ)`` triple — no :class:`MuonDataset` required.

    The array-taking front door matching :func:`fft_asymmetry` kwarg for kwarg,
    running the same complex transform as :func:`fft_complex_arrays` and
    deriving the same two channels :func:`fft_asymmetry` derives from
    :func:`fft_complex_asymmetry`. ``fft_arrays(ds.time, ds.asymmetry,
    ds.error, …)`` returns exactly what ``fft_asymmetry(ds, …)`` returns — the
    same frequencies, real part and magnitude, bit for bit.

    Parameters
    ----------
    time, asymmetry, error
        As documented on :func:`prepare_fft_arrays`. The FFT is *scale-linear*:
        percent in, percent-scaled spectrum out.
    window, padding_factor, t_min, t_max, phase_degrees, t0_offset_us, \
subtract_average_signal, filter_start_us, filter_time_constant_us, fractional, \
amplitude_calibration, detrend
        Exactly as documented on :func:`fft_asymmetry`, keyword-only here
        because ``error`` is optional.

    Returns
    -------
    frequencies, real_part, magnitude
        Frequency axis (MHz) and the real and magnitude spectra.

    Raises
    ------
    ValueError
        As :func:`prepare_fft_arrays`.
    """
    freqs, spectrum = _fft_complex_arrays(
        time,
        asymmetry,
        error,
        caller="fft_arrays",
        window=window,
        padding_factor=padding_factor,
        t_min=t_min,
        t_max=t_max,
        phase_degrees=phase_degrees,
        t0_offset_us=t0_offset_us,
        subtract_average_signal=subtract_average_signal,
        filter_start_us=filter_start_us,
        filter_time_constant_us=filter_time_constant_us,
        fractional=fractional,
        amplitude_calibration=amplitude_calibration,
        diagnostics=None,
        detrend=detrend,
    )

    return freqs, spectrum.real, np.abs(spectrum)


def estimate_fft_phase(
    freqs: NDArray[np.float64],
    spectrum: NDArray[np.complex128],
    *,
    method: str = "peak",
    min_frequency: float = 0.0,
    max_frequency: float | None = None,
) -> float:
    """Estimate the phase, in degrees, that best projects a spectrum onto the real axis.

    Parameters
    ----------
    freqs
        Frequency axis in MHz.
    spectrum
        Complex FFT spectrum.
    method
        ``"peak"`` uses the dominant non-zero frequency bin. ``"average"`` uses a
        power-weighted circular mean across the selected spectrum.
    min_frequency
        Ignore bins below this frequency threshold when estimating the phase.
    max_frequency
        Ignore bins above this frequency threshold when estimating the phase.
    """
    frequencies = np.asarray(freqs, dtype=float)
    values = np.asarray(spectrum, dtype=np.complex128)
    if frequencies.size == 0 or values.size == 0:
        return 0.0

    mask = np.isfinite(frequencies) & np.isfinite(values.real) & np.isfinite(values.imag)
    mask &= frequencies > float(min_frequency)
    if max_frequency is not None:
        mask &= frequencies <= float(max_frequency)
    if not np.any(mask):
        return 0.0

    selected = values[mask]
    method_key = str(method).strip().lower()
    if method_key == "peak":
        idx = int(np.argmax(np.abs(selected)))
        return _normalize_phase_degrees(np.rad2deg(np.angle(selected[idx])))
    if method_key == "average":
        weights = np.abs(selected) ** 2
        if not np.any(weights > 0.0):
            return 0.0
        phasor = np.sum(weights * np.exp(1j * np.angle(selected)))
        if np.isclose(np.abs(phasor), 0.0):
            return 0.0
        return _normalize_phase_degrees(np.rad2deg(np.angle(phasor)))

    raise ValueError(
        f"Unknown phase-estimation method {method!r}. Expected one of 'peak', 'average'."
    )


def optimize_phase_entropy(
    spectrum: NDArray[np.complex128],
    *,
    min_bin: int = 1,
    max_bin: int | None = None,
    gamma: float = 1.0,
) -> tuple[NDArray[np.float64], float, float]:
    """Find the linear phase (c₀ + c₁·i/N) that maximises spectral compactness.

    Implements musrfit's ``PFTPhaseCorrection`` algorithm: minimise an entropy
    + penalty functional over the phase-corrected real spectrum using iminuit.

    The phase model is ``φ(i) = c₀ + c₁ · (i − min_bin) / span`` where
    ``span = max_bin − min_bin``.  The optimisation targets the bins in the
    range ``[min_bin, max_bin]`` but the returned real spectrum covers all
    input bins.

    Parameters
    ----------
    spectrum
        Complex FFT spectrum, **not** pre-rotated.
    min_bin
        First bin index to include in the optimisation (default 1 to skip DC).
    max_bin
        Last bin index to include (default: last bin).
    gamma
        Weight of the negativity penalty relative to the entropy term.

    Returns
    -------
    real_values, c0_rad, c1_rad
        Optimised real-valued display spectrum and the phase parameters in
        radians.
    """
    values = np.asarray(spectrum, dtype=np.complex128)
    n = values.size
    if n == 0:
        return np.zeros(0, dtype=np.float64), 0.0, 0.0

    lo = max(0, int(min_bin))
    hi = int(max_bin) if max_bin is not None else n - 1
    hi = min(hi, n - 1)
    span = float(max(1, hi - lo))

    re = values.real
    im = values.imag

    def _apply(c0: float, c1: float) -> NDArray[np.float64]:
        weights = (np.arange(n, dtype=np.float64) - lo) / span
        angles = float(c0) + float(c1) * weights
        return re * np.cos(angles) - im * np.sin(angles)

    def _cost(c0: float, c1: float) -> float:
        real_part = _apply(c0, c1)[lo : hi + 1]
        delta = np.abs(np.diff(real_part))
        total = float(np.sum(delta))
        if total <= 0.0:
            # Mirrors musrfit PFTPhaseCorrection: return a large cost for the
            # trivial zero-spectrum solution so the optimizer avoids it.
            return 1.0e10
        p = delta / total
        p_pos = p[p > 0.0]
        entropy = float(-np.sum(p_pos * np.log(p_pos)))
        neg = real_part[real_part < 0.0]
        penalty = float(np.sum(neg * neg))
        return entropy + float(gamma) * penalty

    c0_opt = 0.0
    c1_opt = 0.0
    try:
        from iminuit import Minuit  # noqa: PLC0415

        # Coarse grid scan over c0 to find a good initial point before
        # Minuit refines.  The entropy cost has a complex landscape, so a
        # gradient-only start from c0=0 frequently misses the true minimum.
        grid = np.linspace(-np.pi, np.pi, 37, endpoint=False)
        best_c0 = 0.0
        best_val = _cost(0.0, 0.0)
        for c0_trial in grid:
            val = _cost(float(c0_trial), 0.0)
            if val < best_val:
                best_val = val
                best_c0 = float(c0_trial)

        m = Minuit(_cost, c0=best_c0, c1=0.0)
        m.errordef = 1.0
        m.errors = (np.deg2rad(10.0), np.deg2rad(5.0))
        m.migrad()
        c0_opt = float(m.values["c0"])
        c1_opt = float(m.values["c1"])
    except Exception:
        pass

    return _apply(c0_opt, c1_opt).astype(np.float64), c0_opt, c1_opt


def fourier_display_values(
    spectrum: NDArray[np.complex128],
    *,
    display: str = "Real",
) -> NDArray[np.float64]:
    """Return one real-valued display channel derived from a complex spectrum.

    For ``phaseOptReal`` mode the caller must first run
    :func:`optimize_phase_entropy` and pass the resulting already-optimised
    real spectrum wrapped in a complex array (imaginary part zero), or pass
    the real array directly as a complex dtype.  The function simply returns
    the real part in that case.
    """
    values = np.asarray(spectrum, dtype=np.complex128)
    mode = canonical_fourier_display_mode(display)
    if mode in {"real", "cos", "phase_corrected", "phase_opt_real", "real_imag"}:
        return values.real.astype(np.float64, copy=False)
    if mode in {"imaginary", "sin"}:
        return values.imag.astype(np.float64, copy=False)
    magnitude = np.abs(values)
    if mode in {"magnitude", "power_sqrt"}:
        return magnitude.astype(np.float64, copy=False)
    if mode == "power":
        return np.square(magnitude, dtype=np.float64)

    phase = np.rad2deg(np.angle(values))
    phase = np.where(magnitude > 0.0, phase, 0.0)
    return phase.astype(np.float64, copy=False)


def exclude_frequency_ranges(
    freqs: NDArray[np.float64],
    values: NDArray[np.float64],
    exclusion_ranges: list[tuple[float, float]] | tuple[tuple[float, float], ...],
) -> NDArray[np.float64]:
    """Zero spectral values inside one or more symmetric exclusion ranges.

    Parameters
    ----------
    freqs
        Frequency axis in MHz.
    values
        Real-valued Fourier display channel to be filtered.
    exclusion_ranges
        Iterable of ``(center_mhz, half_width_mhz)`` pairs.
    """
    frequencies = np.asarray(freqs, dtype=float)
    filtered = np.asarray(values, dtype=float).copy()
    if frequencies.size == 0 or filtered.size == 0 or not exclusion_ranges:
        return filtered

    mask = np.zeros(frequencies.shape, dtype=bool)
    for center, half_width in exclusion_ranges:
        width = max(0.0, float(half_width))
        if width <= 0.0 or not np.isfinite(center):
            continue
        mask |= np.abs(frequencies - float(center)) <= width
    filtered[mask] = 0.0
    return filtered


def average_fourier_display_values(
    display_channels: list[NDArray[np.float64]] | tuple[NDArray[np.float64], ...],
    *,
    estimate_error: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Average real-valued Fourier display channels across groups.

    When ``estimate_error`` is true, the returned error follows the same
    WiMDA-style grouped-average estimate used for averaged FFT spectra,
    computed from the per-bin second moment and mean across the selected
    groups.
    """
    if not display_channels:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    stacked = np.asarray(display_channels, dtype=np.float64)
    averaged = np.mean(stacked, axis=0, dtype=np.float64)
    if not estimate_error or stacked.shape[0] <= 1:
        return averaged, np.zeros_like(averaged)

    mean_square = np.mean(np.square(stacked, dtype=np.float64), axis=0, dtype=np.float64)
    variance = np.clip(mean_square - np.square(averaged, dtype=np.float64), 0.0, None)
    error = np.sqrt(variance / float(stacked.shape[0]), dtype=np.float64)
    return averaged, error

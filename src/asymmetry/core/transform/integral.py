r"""Time-integral asymmetry and field-scan assembly.

The *integral-counting* method reduces a whole muon time spectrum to a **single
number per run** — the asymmetry over a time window — and a *field scan*
collects that number across a series of runs ordered by a run variable (field,
temperature, or run number).  This is the model-free observable behind ALC
(avoided level crossing), LF repolarisation, and QLCR analysis.

This module is the porting target documented in
``docs/porting/time-integral-asymmetry/``.  The behavioural contract follows
Mantid's ``PlotAsymmetryByLogValue`` (alpha-aware, two reduction *methods*,
single time window, ordering by a run log) with WiMDA's count-integral as the
default ``"integral"`` formula (``alpha = 1.0`` reproduces WiMDA exactly).
musrfit has no native integral observable, so nothing is ported from it.

Two reduction methods are provided:

* ``"integral"`` (default) — **sum the counts** over ``[t_min, t_max]`` and then
  form ``(F_int - α B_int) / (F_int + α B_int)``.  This is the WiMDA ALC method
  and Mantid's ``Type=Integral``.
* ``"differential"`` — form the per-bin asymmetry first and then take its
  **mean** over the window.  This is Mantid's ``Type=Differential`` normalised by
  the window so the result is independent of window width and bin size (Mantid's
  single-period path returns an un-normalised time-integral; the normalised mean
  is the more useful observable and matches Mantid's dual-period normalisation).

The two methods agree only when the asymmetry is flat across the window.

Layering note: this is a pure transform.  It deliberately does **not** import
``asymmetry.core.io`` (``io`` depends on ``transform``, not the reverse).  For
multi-period runs, select the period upstream with
:func:`asymmetry.core.io.periods.select_period` and pass the resulting dataset
in.

This module must stay free of Qt / matplotlib / ``asymmetry.gui`` imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from asymmetry.core.data.dataset import MuonDataset, Run
from asymmetry.core.transform.asymmetry import compute_asymmetry
from asymmetry.core.transform.grouping import apply_grouping_aligned, common_t0_for_groups

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import NDArray

__all__ = [
    "METHODS",
    "ORDER_KEYS",
    "integrate_asymmetry",
    "integrate_curve",
    "integrate_run",
    "build_field_scan",
    "differentiate_scan",
    "FieldScan",
    "FieldScanPoint",
]

#: Supported per-run reduction methods.
METHODS = ("integral", "differential")

#: Supported scan ordering variables (mirrors
#: :data:`asymmetry.core.representation.series.ORDER_KEYS`).
ORDER_KEYS = ("field", "temperature", "run")

#: Human-facing axis labels for each ordering variable.
_X_LABELS = {"field": "B (G)", "temperature": "T (K)", "run": "Run"}


# --- per-run reduction --------------------------------------------------------


def integrate_asymmetry(
    forward: NDArray[np.float64],
    backward: NDArray[np.float64],
    *,
    alpha: float = 1.0,
    time: NDArray[np.float64] | None = None,
    t_min: float | None = None,
    t_max: float | None = None,
    method: str = "integral",
) -> tuple[float, float]:
    r"""Reduce grouped forward/backward counts to a single ``(value, error)``.

    Parameters
    ----------
    forward, backward
        Grouped forward and backward count arrays (same convention as
        :func:`asymmetry.core.transform.compute_asymmetry`).
    alpha
        Detector balance parameter (must be > 0).
    time
        Optional time axis (μs) aligned with the count arrays.  Required when
        ``t_min``/``t_max`` are supplied; otherwise the whole array is used.
    t_min, t_max
        Inclusive integration window in μs.
    method
        ``"integral"`` (sum counts, then form asymmetry — WiMDA / Mantid
        Integral) or ``"differential"`` (per-bin asymmetry, then mean over the
        window — Mantid Differential, window-normalised).

    Returns
    -------
    (value, error)
        The integral asymmetry (dimensionless, fractional) and its error.  The
        error uses the same Mantid-compatible model as
        :func:`compute_asymmetry`, so the integral and time-domain observables
        share one error formula.
    """
    _validate_method(method)
    _validate_alpha(alpha)

    f = np.asarray(forward, dtype=np.float64)
    b = np.asarray(backward, dtype=np.float64)
    n = min(f.size, b.size)
    if n == 0:
        raise ValueError("integrate_asymmetry requires non-empty count arrays.")
    f = f[:n]
    b = b[:n]

    mask = _window_mask(time, t_min, t_max, n)
    if not np.any(mask):
        raise ValueError("Integration window selects no bins.")

    if method == "integral":
        f_int = np.array([float(np.sum(f[mask]))], dtype=np.float64)
        b_int = np.array([float(np.sum(b[mask]))], dtype=np.float64)
        asym, err = compute_asymmetry(f_int, b_int, alpha)
        return float(asym[0]), float(err[0])

    # "differential": per-bin asymmetry, then mean over the window. Exclude
    # zero-denominator bins, where compute_asymmetry returns a sentinel
    # (asym=0, err=1.0); including them would bias the mean toward zero and
    # inflate the error.
    asym, err = compute_asymmetry(f, b, alpha)
    valid = mask & ((f + alpha * b) != 0.0)
    if not np.any(valid):
        raise ValueError("Integration window selects no bins with non-zero counts.")
    return _mean_over_window(asym, err, valid)


def integrate_curve(
    time: NDArray[np.float64],
    asymmetry: NDArray[np.float64],
    error: NDArray[np.float64],
    *,
    t_min: float | None = None,
    t_max: float | None = None,
) -> tuple[float, float]:
    """Mean of an already-formed asymmetry curve over ``[t_min, t_max]``.

    Useful when only the reduced curve is available (for example a combined
    green∓red spectrum) rather than the forward/backward counts.  The error is
    the error on the mean, ``sqrt(Σ eᵢ²) / N``.  Output units match the input
    asymmetry units (the loader's reduced datasets are in percent).
    """
    t = np.asarray(time, dtype=np.float64)
    a = np.asarray(asymmetry, dtype=np.float64)
    e = np.asarray(error, dtype=np.float64)
    n = min(t.size, a.size, e.size)
    if n == 0:
        raise ValueError("integrate_curve requires non-empty arrays.")
    mask = _window_mask(t[:n], t_min, t_max, n)
    if not np.any(mask):
        raise ValueError("Integration window selects no bins.")
    return _mean_over_window(a[:n], e[:n], mask)


def integrate_run(
    data: MuonDataset | Run,
    *,
    t_min: float | None = None,
    t_max: float | None = None,
    method: str = "integral",
    alpha: float | None = None,
) -> tuple[float, float]:
    """Reduce a loaded run to a single integral-asymmetry ``(value, error)``.

    The run's forward/backward groups are formed from ``run.grouping`` exactly
    as :class:`asymmetry.core.representation.time.TimeFBAsymmetry` does, so the
    integral observable is consistent with the time-domain asymmetry.  When the
    window is unspecified it defaults to the run's good-bin range (matching the
    WiMDA ALC / Mantid full-range behaviour).

    Parameters
    ----------
    data
        A :class:`MuonDataset` (its ``.run`` is used) or a :class:`Run`.  For
        multi-period runs, select the period first with
        :func:`asymmetry.core.io.periods.select_period`.
    t_min, t_max
        Inclusive window in μs.  Default: the run's good-bin range.
    method
        See :func:`integrate_asymmetry`.
    alpha
        Balance parameter; when ``None`` it is read from ``run.grouping['alpha']``
        (default 1.0).
    """
    _validate_method(method)
    run = _resolve_run(data)
    time, forward, backward, alpha_used, good = _reduce_run_to_fb(run, alpha)

    if t_min is None:
        t_min = float(time[good[0]])
    if t_max is None:
        t_max = float(time[good[1]])
    _validate_window(t_min, t_max)

    return integrate_asymmetry(
        forward,
        backward,
        alpha=alpha_used,
        time=time,
        t_min=t_min,
        t_max=t_max,
        method=method,
    )


# --- field-scan assembly ------------------------------------------------------


@dataclass
class FieldScanPoint:
    """One run's contribution to a field scan."""

    run_number: int
    x: float
    value: float
    error: float
    label: str = ""


@dataclass
class FieldScan:
    """An ordered series of integral-asymmetry values vs a run variable.

    The arrays are parallel and sorted by ``x`` ascending.  ``excluded`` lists
    ``(run_number, reason)`` for runs that could not contribute (for example a
    missing field log — Mantid skips such runs rather than failing the scan).
    """

    x: NDArray[np.float64]
    value: NDArray[np.float64]
    error: NDArray[np.float64]
    run_numbers: list[int]
    order_key: str
    method: str
    derivative: bool = False
    x_label: str = ""
    y_label: str = "Integral asymmetry"
    excluded: list[tuple[int, str]] = field(default_factory=list)

    @property
    def n_points(self) -> int:
        return int(self.x.size)

    @property
    def points(self) -> list[FieldScanPoint]:
        return [
            FieldScanPoint(
                run_number=self.run_numbers[i],
                x=float(self.x[i]),
                value=float(self.value[i]),
                error=float(self.error[i]),
            )
            for i in range(self.n_points)
        ]

    def __repr__(self) -> str:
        kind = "derivative" if self.derivative else self.method
        return (
            f"FieldScan({kind}, n={self.n_points}, order_key={self.order_key!r}, "
            f"excluded={len(self.excluded)})"
        )


def build_field_scan(
    runs,
    *,
    t_min: float | None = None,
    t_max: float | None = None,
    method: str = "integral",
    alpha: float | None = None,
    order_key: str = "field",
) -> FieldScan:
    """Assemble a field scan from a series of loaded runs.

    Parameters
    ----------
    runs
        Iterable of loaded objects — each a :class:`MuonDataset` or :class:`Run`.
        For multi-period runs, select the period upstream and pass the per-period
        datasets in.
    t_min, t_max, method, alpha
        Passed through to :func:`integrate_run`.
    order_key
        ``"field"``, ``"temperature"``, or ``"run"`` — the x-axis variable the
        points are ordered by (mirrors ``FitSeries.order_key``).

    Returns
    -------
    FieldScan
        Sorted parallel arrays plus the list of excluded runs.
    """
    _validate_method(method)
    if order_key not in ORDER_KEYS:
        raise ValueError(f"order_key must be one of {ORDER_KEYS}, got {order_key!r}")

    points: list[FieldScanPoint] = []
    excluded: list[tuple[int, str]] = []

    for item in runs:
        try:
            run = _resolve_run(item)
        except (TypeError, ValueError) as exc:
            # One un-resolvable item must not abort the whole scan; exclude it
            # with a reason, like a run missing its log value.
            excluded.append((_excluded_run_number(item), str(exc)))
            continue
        run_number = int(run.run_number)
        x_value = _order_value(run, order_key)
        if x_value is None:
            excluded.append((run_number, f"no {order_key} value"))
            continue
        try:
            value, error = integrate_run(run, t_min=t_min, t_max=t_max, method=method, alpha=alpha)
        except ValueError as exc:
            excluded.append((run_number, str(exc)))
            continue
        points.append(FieldScanPoint(run_number=run_number, x=x_value, value=value, error=error))

    points.sort(key=lambda p: (p.x, p.run_number))
    return FieldScan(
        x=np.array([p.x for p in points], dtype=np.float64),
        value=np.array([p.value for p in points], dtype=np.float64),
        error=np.array([p.error for p in points], dtype=np.float64),
        run_numbers=[p.run_number for p in points],
        order_key=order_key,
        method=method,
        x_label=_X_LABELS[order_key],
        excluded=excluded,
    )


def differentiate_scan(scan: FieldScan, *, max_gap: float | None = None) -> FieldScan:
    """Forward-difference derivative ``dA/dx`` of a scan (WiMDA ``dA/dB``).

    Each output point sits at the midpoint of an adjacent pair, with
    ``dA/dx = (A₂ - A₁) / (x₂ - x₁)`` and error ``sqrt(e₁² + e₂²) / |Δx|``.
    Pairs with non-positive spacing, or spacing greater than ``max_gap`` (when
    given), are skipped — mirroring WiMDA's adjacent-field threshold.
    """
    if scan.derivative:
        raise ValueError("differentiate_scan expects an integral scan, not a derivative.")

    xs, ys, es = scan.x, scan.value, scan.error
    mid_x: list[float] = []
    deriv: list[float] = []
    deriv_err: list[float] = []
    run_pairs: list[int] = []

    for i in range(scan.n_points - 1):
        dx = float(xs[i + 1] - xs[i])
        if dx <= 0.0 or (max_gap is not None and dx > max_gap):
            continue
        mid_x.append(0.5 * (float(xs[i]) + float(xs[i + 1])))
        deriv.append((float(ys[i + 1]) - float(ys[i])) / dx)
        deriv_err.append(float(np.hypot(es[i], es[i + 1])) / dx)
        run_pairs.append(scan.run_numbers[i + 1])

    return FieldScan(
        x=np.array(mid_x, dtype=np.float64),
        value=np.array(deriv, dtype=np.float64),
        error=np.array(deriv_err, dtype=np.float64),
        run_numbers=run_pairs,
        order_key=scan.order_key,
        method=scan.method,
        derivative=True,
        x_label=scan.x_label,
        y_label=f"d({scan.y_label})/d{scan.order_key}",
        excluded=list(scan.excluded),
    )


# --- internal -----------------------------------------------------------------


def _validate_method(method: str) -> None:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")


def _validate_alpha(alpha: float) -> None:
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError(f"alpha must be a positive, finite number, got {alpha!r}")


def _validate_window(t_min: float, t_max: float) -> None:
    # Equal bounds are allowed: they select a single bin (for example a run
    # whose good-bin range is one bin). Only an inverted window is rejected.
    if t_min > t_max:
        raise ValueError(f"t_min must be <= t_max, got t_min={t_min}, t_max={t_max}")


def _window_mask(
    time: NDArray[np.float64] | None,
    t_min: float | None,
    t_max: float | None,
    n: int,
) -> NDArray[np.bool_]:
    if t_min is None and t_max is None:
        return np.ones(n, dtype=bool)
    if time is None:
        raise ValueError("A time axis is required when t_min/t_max are supplied.")
    if t_min is not None and t_max is not None:
        _validate_window(float(t_min), float(t_max))
    t = np.asarray(time, dtype=np.float64)[:n]
    mask = np.ones(n, dtype=bool)
    if t_min is not None:
        mask &= t >= t_min
    if t_max is not None:
        mask &= t <= t_max
    return mask


def _mean_over_window(
    asym: NDArray[np.float64],
    err: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> tuple[float, float]:
    selected = asym[mask]
    selected_err = err[mask]
    count = selected.size
    value = float(np.mean(selected))
    error = float(np.sqrt(np.sum(np.square(selected_err))) / count)
    return value, error


def _resolve_run(data: MuonDataset | Run) -> Run:
    if isinstance(data, Run):
        return data
    if isinstance(data, MuonDataset):
        if data.run is None:
            raise ValueError(
                "This MuonDataset has no source run; integral reduction needs the "
                "run's histograms and grouping."
            )
        return data.run
    if isinstance(data, list):
        raise TypeError(
            "Got a list of per-period datasets; select one period first with "
            "asymmetry.core.io.periods.select_period(...)."
        )
    raise TypeError(f"Expected a MuonDataset or Run, got {type(data).__name__}")


def _reduce_run_to_fb(
    run: Run,
    alpha: float | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float, tuple[int, int]]:
    """Form forward/backward groups + time axis + good-bin range from a run.

    Mirrors :class:`asymmetry.core.representation.time.TimeFBAsymmetry` so the
    integral observable agrees with the time-domain asymmetry by construction.
    Returns ``(time, forward, backward, alpha_used, (good_start, good_end))``
    where the good indices are 0-based into the returned full-length arrays.
    """
    histograms = list(run.histograms)
    if not histograms:
        raise ValueError("Integral reduction requires detector histograms.")
    grouping = run.grouping if isinstance(run.grouping, dict) else {}
    groups = grouping.get("groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError(
            "Integral reduction requires a detector grouping definition (run.grouping['groups'])."
        )

    forward_gid = int(grouping.get("forward_group", 1))
    backward_gid = int(grouping.get("backward_group", 2))
    forward_indices = _resolve_group_indices(groups, forward_gid)
    backward_indices = _resolve_group_indices(groups, backward_gid)
    if not forward_indices or not backward_indices:
        raise ValueError("Forward/backward groups do not reference any detectors.")

    if alpha is None:
        try:
            alpha_used = float(grouping.get("alpha", 1.0))
        except (TypeError, ValueError):
            alpha_used = 1.0
    else:
        alpha_used = float(alpha)
    _validate_alpha(alpha_used)

    common_t0 = common_t0_for_groups(histograms, forward_indices, backward_indices)
    forward = apply_grouping_aligned(histograms, forward_indices, common_t0_bin=common_t0)
    backward = apply_grouping_aligned(histograms, backward_indices, common_t0_bin=common_t0)
    n = min(forward.size, backward.size)
    forward = forward[:n]
    backward = backward[:n]
    if n == 0:
        raise ValueError("Forward/backward grouping produced empty arrays.")

    try:
        first_good = max(0, int(grouping.get("first_good_bin", 0)))
    except (TypeError, ValueError):
        first_good = 0
    try:
        last_good = int(grouping.get("last_good_bin", n - 1))
    except (TypeError, ValueError):
        last_good = n - 1
    last_good = min(last_good, n - 1)
    if first_good > last_good:
        first_good, last_good = 0, n - 1

    bin_width = float(histograms[0].bin_width)
    time = (np.arange(n, dtype=np.float64) - float(common_t0)) * bin_width
    return time, forward, backward, alpha_used, (first_good, last_good)


def _resolve_group_indices(groups: dict, group_id: int) -> list[int]:
    """Return zero-based detector indices for *group_id* (1-based detector nos.).

    Mirrors ``asymmetry.core.representation.time._resolve_group_indices``.
    """
    entries = groups.get(group_id)
    if entries is None:
        entries = groups.get(str(group_id))
    if not isinstance(entries, list):
        return []
    indices: list[int] = []
    for value in entries:
        detector = value[0] if isinstance(value, (list, tuple)) and value else value
        try:
            indices.append(max(0, int(detector) - 1))
        except (TypeError, ValueError):
            continue
    return indices


def _order_value(run: Run, order_key: str) -> float | None:
    if order_key == "run":
        return float(run.run_number)
    raw = run.metadata.get(order_key)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    # Reject non-finite logs (NaN/inf) so they cannot corrupt the scan order or
    # produce NaN derivative points downstream.
    if not np.isfinite(value):
        return None
    return value


def _excluded_run_number(item: object) -> int:
    """Best-effort run number for an item that could not be resolved to a Run.

    ``MuonDataset.run_number`` and ``Run.run_number`` both work via ``getattr``;
    anything else (a stray ``list``/``None``) falls back to ``0``.
    """
    try:
        return int(getattr(item, "run_number", None))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0

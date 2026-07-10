"""Shared "suggest next point" utility-band overlay rendering.

The Bayesian-experimental-design overlay (normalised utility band anchored to
the bottom of the axes, an extrapolated-candidate style, and a best-x marker +
annotation) was born inside :class:`~asymmetry.gui.widgets.trend_preview.
TrendPreviewCanvas`. Both that canvas *and* the Knight-shift window's plain
``_redraw`` need to draw it, so it lives here as a free function operating on a
matplotlib ``Axes`` passed in by the caller — no widget state, no matplotlib
import at module load (the caller owns the figure/axes).

:class:`SuggestionOverlay` is re-exported from
:mod:`asymmetry.gui.widgets.trend_preview` so existing imports keep working.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from asymmetry.gui.styles import tokens

#: Fraction of the axes height the utility band's peak spans.
_SUGGESTION_BAND_FRACTION = 0.18


@dataclass
class SuggestionOverlay:
    """Bayesian-experimental-design "next point" overlay (see
    ``docs/studies/bed-next-point-suggestion.md`` §5.4).

    ``utility`` is the RAW (unnormalised) per-candidate utility from
    ``NextPointSuggestion`` — the drawing code normalises it for display so
    callers never need to rescale. ``best_x`` is NaN when no suggestion is
    available (draw everything except the marker/annotation). ``risk_mask`` is an
    optional per-candidate boolean flag (multi-series acquisition only — §3.1):
    ``True`` where two predicted curves approach within the misassignment
    threshold, so a new datum there might attach to the wrong curve. ``None`` for
    single-series overlays (which have no crossing to worry about).
    """

    x: NDArray
    utility: NDArray
    extrapolated: NDArray[np.bool_]
    best_x: float
    risk_mask: NDArray[np.bool_] | None = None


def _contiguous_runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """``[start, stop)`` index spans of each contiguous ``True`` run in *mask*."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i in range(len(mask) + 1):
        flagged = i < len(mask) and bool(mask[i])
        if flagged and start is None:
            start = i
        elif not flagged and start is not None:
            runs.append((start, i))
            start = None
    return runs


def _draw_risk_shading(
    ax: object,
    transform: object,
    xs: NDArray[np.float64],
    risk_mask: NDArray[np.bool_],
) -> None:
    """Shade the candidate spans flagged as at-risk of misassignment (§3.1).

    A muted warm hatch band over each flagged run, deliberately distinct from the
    extrapolation styling (which reuses the utility colour at reduced alpha) so
    the two never read as the same thing. Drawn with the blended x-data/y-axes
    transform so it spans the axes height as chrome and never perturbs the data
    limits.
    """
    for start, stop in _contiguous_runs(risk_mask):
        lo = float(xs[start])
        hi = float(xs[stop - 1])
        if hi <= lo:
            # A single-candidate run has no width to shade; nudge it to a hairline
            # span so the flag is still visible rather than collapsing to nothing.
            hi = lo + (float(xs[1] - xs[0]) if xs.size > 1 else 0.0)
        try:
            ax.axvspan(  # type: ignore[union-attr]
                lo,
                hi,
                facecolor=tokens.WARN,
                edgecolor=tokens.WARN,
                alpha=0.10,
                hatch="xx",
                linewidth=0.0,
                zorder=0.9,
            )
        except Exception:
            pass


def _fill_suggestion_run(
    ax: object,
    transform: object,
    xs: NDArray[np.float64],
    heights: NDArray[np.float64],
    extrapolated: bool,
    colour: str,
) -> None:
    """Fill one contiguous run of the utility band at the given alpha."""
    if xs.size == 0:
        return
    try:
        ax.fill_between(  # type: ignore[union-attr]
            xs,
            0.0,
            heights,
            transform=transform,
            facecolor=colour,
            edgecolor="none",
            alpha=0.10 if extrapolated else 0.28,
            zorder=1.2,
        )
    except Exception:
        pass


def draw_suggestion_overlay(
    ax: object,
    overlay: SuggestionOverlay | None,
    colour: str,
    label: str,
    *,
    band_fraction: float = _SUGGESTION_BAND_FRACTION,
) -> None:
    """Draw one BED "next point" overlay band on *ax* (Phase 2 §5.4 / Phase 3 §8.1).

    Utility is drawn as a translucent filled band anchored to the BOTTOM of the
    axes via the blended x-data/y-axes-fraction transform, so it reads as chrome
    rather than a second data series. Normalised so the band's peak spans roughly
    ``band_fraction`` of the axes height; extrapolated candidates are drawn at
    reduced alpha. Candidate spans carrying ``overlay.risk_mask`` are shaded with
    a distinct warm hatch (misassignment risk). A vertical marker + annotation
    mark ``best_x`` (skipped when NaN). *colour* and *label* let this same routine
    draw the refinement suggestion band and the model-discrimination band in
    distinct styles; both may be drawn on the same axes.

    INVARIANT: this overlay must never change the main axes' data limits/
    autoscale. ``axvspan``/``fill_between`` with a blended transform can still
    update the x-datalim (and, on some matplotlib versions, the y-datalim) as a
    side effect of adding the artist, so the pre-overlay ylim is captured and
    forcibly restored after every artist is added.
    """
    if overlay is None:
        return
    x = np.asarray(overlay.x, dtype=float)
    utility = np.asarray(overlay.utility, dtype=float)
    if x.size == 0 or utility.size == 0 or x.size != utility.size:
        return

    # Capture the axes' current y-limits BEFORE adding any overlay artist, and
    # restore them after — the overlay must never perturb autoscale.
    try:
        saved_ylim = ax.get_ylim()  # type: ignore[union-attr]
    except Exception:
        saved_ylim = None

    try:
        extrapolated = np.asarray(overlay.extrapolated, dtype=bool)
        if extrapolated.shape != x.shape:
            extrapolated = np.zeros_like(x, dtype=bool)

        max_u = float(np.nanmax(utility)) if np.any(np.isfinite(utility)) else 0.0
        order = np.argsort(x)
        xs = x[order]
        us = utility[order]
        extra = extrapolated[order]
        if max_u > 0.0:
            normalised = np.clip(us / max_u, 0.0, None) * band_fraction
        else:
            normalised = np.zeros_like(us)

        transform = ax.get_xaxis_transform()  # type: ignore[union-attr]

        # Risk shading is drawn first so the utility band and best-x marker layer
        # on top of it.
        risk_mask = overlay.risk_mask
        if risk_mask is not None:
            risk = np.asarray(risk_mask, dtype=bool)
            if risk.shape == x.shape and np.any(risk):
                _draw_risk_shading(ax, transform, xs, risk[order])

        # Split into contiguous extrapolated / in-range runs so each run gets its
        # own alpha via a separate fill_between call.
        for is_extra in (False, True):
            mask = extra == is_extra
            if not np.any(mask):
                continue
            run_start = None
            for i in range(len(xs) + 1):
                in_run = i < len(xs) and mask[i]
                if in_run and run_start is None:
                    run_start = i
                elif not in_run and run_start is not None:
                    _fill_suggestion_run(
                        ax,
                        transform,
                        xs[run_start:i],
                        normalised[run_start:i],
                        is_extra,
                        colour,
                    )
                    run_start = None

        if np.isfinite(overlay.best_x):
            try:
                ax.axvline(  # type: ignore[union-attr]
                    overlay.best_x,
                    color=colour,
                    linestyle=":",
                    linewidth=1.4,
                    zorder=7,
                )
                ax.annotate(  # type: ignore[union-attr]
                    label,
                    xy=(overlay.best_x, 1.0),
                    xycoords=transform,
                    xytext=(3, -10),
                    textcoords="offset points",
                    ha="left",
                    va="top",
                    color=colour,
                    fontsize=8,
                    zorder=7,
                )
            except Exception:
                pass
    finally:
        if saved_ylim is not None:
            try:
                ax.set_ylim(saved_ylim)  # type: ignore[union-attr]
            except Exception:
                pass

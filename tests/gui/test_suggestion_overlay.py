"""Shared BED overlay helper (:mod:`asymmetry.gui.widgets.suggestion_overlay`)."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

pytest.importorskip("matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from asymmetry.gui.styles import tokens
from asymmetry.gui.widgets.suggestion_overlay import (
    SuggestionOverlay,
    draw_suggestion_overlay,
)


def _overlay(*, best_x: float = 2.0, risk_mask=None) -> SuggestionOverlay:
    x = np.linspace(0.0, 4.0, 41)
    utility = np.exp(-((x - best_x) ** 2))
    extrapolated = (x < 0.5) | (x > 3.5)
    return SuggestionOverlay(
        x=x, utility=utility, extrapolated=extrapolated, best_x=best_x, risk_mask=risk_mask
    )


def test_draw_overlay_does_not_perturb_ylim() -> None:
    fig, ax = plt.subplots()
    ax.plot([0.0, 4.0], [10.0, 20.0])
    before = ax.get_ylim()
    draw_suggestion_overlay(ax, _overlay(), tokens.ACCENT, "suggested")
    assert ax.get_ylim() == before
    plt.close(fig)


def test_draw_overlay_none_is_noop() -> None:
    fig, ax = plt.subplots()
    ax.plot([0.0, 4.0], [10.0, 20.0])
    n_before = len(ax.collections) + len(ax.lines)
    draw_suggestion_overlay(ax, None, tokens.ACCENT, "suggested")
    assert len(ax.collections) + len(ax.lines) == n_before
    plt.close(fig)


def test_draw_overlay_marks_best_x_when_finite() -> None:
    fig, ax = plt.subplots()
    ax.plot([0.0, 4.0], [10.0, 20.0])
    n_lines_before = len(ax.lines)
    draw_suggestion_overlay(ax, _overlay(best_x=2.0), tokens.ACCENT, "suggested")
    # A vertical marker line is added at best_x.
    assert len(ax.lines) > n_lines_before
    plt.close(fig)


def test_draw_overlay_skips_marker_when_best_x_nan() -> None:
    fig, ax = plt.subplots()
    ax.plot([0.0, 4.0], [10.0, 20.0])
    n_lines_before = len(ax.lines)
    draw_suggestion_overlay(ax, _overlay(best_x=float("nan")), tokens.ACCENT, "suggested")
    # No marker line for a NaN best_x (the band may still be drawn).
    assert len(ax.lines) == n_lines_before
    plt.close(fig)


def test_risk_mask_adds_shading_spans() -> None:
    fig, ax = plt.subplots()
    ax.plot([0.0, 4.0], [10.0, 20.0])
    x = np.linspace(0.0, 4.0, 41)
    # Flag a contiguous middle run as at-risk.
    risk = (x > 1.5) & (x < 2.5)
    before = ax.get_ylim()

    n_patches_plain = len(ax.patches)
    draw_suggestion_overlay(ax, _overlay(risk_mask=None), tokens.ACCENT, "suggested")
    n_patches_no_risk = len(ax.patches)

    draw_suggestion_overlay(ax, _overlay(risk_mask=risk), tokens.ACCENT, "suggested")
    n_patches_with_risk = len(ax.patches)

    # The risk overlay adds at least one extra axvspan patch beyond the plain one.
    assert n_patches_with_risk - n_patches_no_risk > n_patches_no_risk - n_patches_plain
    # And it still never disturbs the data limits.
    assert ax.get_ylim() == before
    plt.close(fig)

"""Preview count reductions without modifying runs.

Resolution and reduction run in TaskRunner; the GUI snapshots inputs and draws.
Time axes are in µs and asymmetry is in percent. Generation ids discard obsolete
outputs. One result supplies both views; incomplete corrections remain labelled.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from functools import partial
from html import escape
from typing import Any, NamedTuple

import numpy as np
from PySide6.QtCore import QSize, QTimer, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.io import background_reference_for_grouping
from asymmetry.core.project.profiles import GroupingProfile, resolve_effective_grouping
from asymmetry.core.transform import (
    ForwardBackwardCounts,
    binned_fb_asymmetry,
    corrected_grouped_counts,
    correction_flags_from_grouping,
    effective_group_indices,
    resolve_background_mode,
    resolve_facility,
)
from asymmetry.core.transform.corrections import (
    BackgroundReference,
    CorrectionOutcomes,
)
from asymmetry.core.transform.rebin import binning_spec
from asymmetry.core.transform.reduce import reduce_period_asymmetry
from asymmetry.core.transform.t0 import RunT0Search
from asymmetry.core.utils.constants import PeriodMode
from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.widgets import make_warning_banner
from asymmetry.gui.tasks import TaskCancelledError, TaskRunner, TaskWorker, check_task_cancelled
from asymmetry.gui.utils.plot_decimation import decimate_for_preview as _decimate_for_preview
from asymmetry.gui.utils.plot_decimation import preview_stride as _preview_stride

#: Debounce window for coalescing rapid edits before a recompute.
_DEBOUNCE_MS = 300

#: Cap on points drawn in the preview. The pane is only a few hundred pixels
#: wide, so plotting the full reduced curve (which can be ~1M points for a
#: long high-resolution run) is wasted work — and matplotlib ``errorbar`` over
#: ~1M points freezes the GUI thread for ~12 s (rendering cannot leave the GUI
#: thread). Uniformly striding down to this many points before drawing keeps
#: the draw ~30 ms; the preview is advisory so exact sampling does not matter.
_MAX_PREVIEW_POINTS = 2000

#: Above this many drawn points the markers overlap into an opaque band that
#: hides both the error bars and the ghost, so the solid becomes a line with a
#: ±σ fill instead; at or below it, markers with error bars still read as data.
_LINE_MODE_POINTS = 400

#: Fixed height of the preview section so it never fights the form for space.
_PANE_HEIGHT = 300

#: Identity colour per compare stage — the same colour the stage's pipeline chip
#: outline and correction-card stripe wear. It names what the stage *produced*,
#: so while a stage is focused the as-reduced curve wears it and chip, card and
#: curve read as one thing.
_STAGE_COLORS: dict[str, str] = {
    "deadtime": tokens.STAGE_DEADTIME,
    "background": tokens.STAGE_BACKGROUND,
    "alpha": tokens.STAGE_ALPHA,
    "beta": tokens.STAGE_BETA,
}

#: The ghost — the curve *without* the focused correction — is always grey, in
#: both views. Colour is the correction, grey is its absence.
_GHOST_COLOR = tokens.TEXT_MUTED

#: Counts view: colour is already spent on with/without the correction, so the
#: backward group is told from the forward one by being a lighter tint of the
#: same colour AND dashed, in both pairs — the dash alone merged with F at full
#: scale, the tint alone would read as a third colour.
_BACKWARD_DASH = (0, (5, 2))
_BACKWARD_TINT = 0.45


def _lighter(color: str, amount: float = _BACKWARD_TINT) -> str:
    """*color* (``#rrggbb``) mixed *amount* of the way towards white."""
    r, g, b = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    mix = (round(c + (255 - c) * amount) for c in (r, g, b))
    return "#" + "".join(f"{c:02x}" for c in mix)


class _CaptionRow(NamedTuple):
    color: str
    text: str
    dashed: bool = False


def _solid_color(compare_stage: str | None) -> str:
    """Colour of the as-reduced curve: the focused stage's, or plain accent."""
    return tokens.ACCENT if compare_stage is None else _STAGE_COLORS[compare_stage]


#: What each compare's ghost removes. The plot caption, the pager label and the
#: focused card's indicator all word the same comparison, so it is named once
#: here and imported by the dialog.
COMPARE_STAGE_LABELS: dict[str, str] = {
    "deadtime": "without deadtime",
    "background": "without background",
    "alpha": "α = 1",
    "beta": "β = 1",
}

#: The stages that act when the asymmetry is *formed* rather than on the counts,
#: keyed to the symbol the Counts caption names them by. Membership is the one
#: fact that decides both: these stages leave the F/B spectra untouched, so the
#: Counts view has no ghost to draw for them and says so instead.
_ASYMMETRY_STAGE_SYMBOLS: dict[str, str] = {"alpha": "α", "beta": "β"}

#: Constant-level subtraction draws a rule; a reference spectrum has no level.
_BACKGROUND_MODE_LABELS: dict[str, str] = {
    "fixed": "fixed",
    "range": "range average",
    "tail_fit": "tail fit",
}

#: Caption geometry in axes fractions: swatch from x to x+width, text after it,
#: rows stepping down from the top. Fixed placement (never data-dependent) so an
#: off-scale ghost is still named at a stable spot. ``_CAPTION_ROW_DY`` is the
#: single row step: it places the rows AND sizes the autoscale headroom that
#: keeps the curve from climbing under them, so the two cannot drift apart.
_CAPTION_X = 0.012
_CAPTION_SWATCH_W = 0.033
_CAPTION_TOP = 0.96
_CAPTION_ROW_DY = 0.09
#: Clearance between the lowest caption row and the top of the solid curve.
_CAPTION_HEADROOM_PAD = 0.04

#: Backing for any text drawn over the curve (caption rows, the ⟨A⟩ value), so
#: it stays readable whatever the data does behind it.
_TEXT_BBOX = {
    "boxstyle": "round,pad=0.25",
    "facecolor": tokens.SURFACE,
    "edgecolor": "none",
    "alpha": 0.85,
}


@dataclass(frozen=True)
class PreviewFacts:
    """Immutable run, pair, and binning labels carried with the worker request."""

    run_label: str
    forward_name: str
    backward_name: str
    bunch: int
    t0_mode_label: str
    corrections: tuple[str, ...] = ()
    vector_pair: str | None = None
    member_counts: tuple[int, int] | None = None


@dataclass(frozen=True)
class _PreviewRequest:
    """Immutable plain inputs for worker resolution and reduction."""

    generation: int
    histograms: list[Histogram]
    facility: str
    facts: PreviewFacts
    grouping: dict[str, Any] | None = None
    profile: GroupingProfile | None = None
    run: Run | None = None
    #: The ghost omits one stage; the solid always uses the full correction path.
    compare_stage: str | None = None
    resolved_reference: BackgroundReference | None = None
    timing_searches: dict[int, RunT0Search] = field(default_factory=dict)


@dataclass(frozen=True)
class CountsCurves:
    """Decimated observed bin-centre spectra in μs, with undecimated marker edges."""

    #: Bin-centre times of the drawn bins, in µs after t0.
    time: np.ndarray
    forward: np.ndarray
    backward: np.ndarray
    #: Bin width in µs (undecimated — one histogram bin, not one drawn sample).
    bin_width: float
    #: Undecimated length; the view ends at the trailing acquisition-bin edge.
    n_bins: int
    #: Resolved common t0 bin, and the good window's inclusive bin bounds.
    t0_bin: int
    first_good: int
    last_good: int
    #: The constant level subtracted from (F, B), for the modes that subtract
    #: one; ``None`` for ``reference_run`` (a spectrum) and for no background.
    background_level: tuple[float, float] | None
    background_mode: str
    #: The compare pass' spectra, on the same stride — the deadtime and
    #: background compares only, whose second corrected pass already ran for the
    #: asymmetry ghost. ``None`` for α/β (they do not touch the counts) and off.
    ghost_forward: np.ndarray | None = None
    ghost_backward: np.ndarray | None = None

    def bin_edge_time(self, bin_index: int) -> float:
        """Leading edge of bin *bin_index*, on the same axis as :attr:`time`.

        ``time[0]`` is bin 0's centre (the stride always keeps sample 0), so the
        histogram's origin sits half a bin before it.
        """
        return (bin_index - 0.5) * self.bin_width + float(self.time[0])


@dataclass(frozen=True)
class _PreviewResult:
    """Plain-array reduction result marshalled back to the GUI thread."""

    generation: int
    facts: PreviewFacts
    time: np.ndarray
    asymmetry: np.ndarray
    error: np.ndarray
    counts: CountsCurves | None
    alpha: float
    beta: float
    baseline: np.ndarray | None = None
    compare_stage: str | None = None
    centre: tuple[float, float] | None = None
    corrections: tuple[CorrectionOutcomes, ...] = (CorrectionOutcomes(),)


class GroupingPreviewPane(QWidget):
    """Debounce F/B preview reductions on workers and show explicit failures."""

    preview_changed = Signal()
    preview_completed = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the canvas, status strip, and debounce/worker plumbing."""
        super().__init__(parent)
        self.setFixedHeight(_PANE_HEIGHT)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(2)
        self._incomplete_banner = make_warning_banner(severity="strong")
        self._incomplete_banner.hide()
        layout.addWidget(self._incomplete_banner)

        self._status = QLabel("")
        self._status.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        self._status.setWordWrap(True)

        self._tasks = TaskRunner(self)
        self._generation = 0
        self._pending: _PreviewRequest | None = None
        #: One worker runs; only the latest pending request is dispatched next.
        self._in_flight = False

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(_DEBOUNCE_MS)
        self._debounce.timeout.connect(self._dispatch_pending)

        self._figure = None
        self._canvas = None
        self._axes = None
        self._nav_toolbar = None
        #: Preview-only view selected by the dialog's segmented control.
        self._view = "asymmetry"
        #: Last result retained for Home redraws without a recompute.
        self._state: _PreviewResult | str | None = None
        #: User limits survive axes.clear(); None means solid-only autoscale.
        self._user_view: tuple[tuple[float, float], tuple[float, float]] | None = None
        #: Limits at nav-drag press, to tell a real drag from a mode-on click.
        self._nav_press_view: tuple | None = None
        #: Draws arriving mid-drag wait until release to preserve the rubber band.
        self._deferred_result: _PreviewResult | None = None
        try:
            from asymmetry.gui.widgets.mpl_canvas import create_canvas

            self._figure, self._canvas, self._nav_toolbar = create_canvas(
                layout="tight", toolbar=True, parent=self
            )
            self._axes = self._figure.add_subplot(111)
            self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            layout.addWidget(self._canvas, stretch=1)
            # The full toolbar is far too much chrome for a small advisory pane;
            # keep it hidden and surface just pan/zoom/home as compact buttons on
            # the status strip (QToolButtons bound to the toolbar's own actions,
            # so checked-state and mode plumbing stay matplotlib's).
            self._nav_toolbar.setVisible(False)
            # A pan/zoom drag that MOVED the view means the user chose it —
            # capture the chosen limits on release (the toolbar's own release
            # handlers run first, so the limits are final by then).
            self._canvas.mpl_connect("button_press_event", self._on_canvas_button_press)
            self._canvas.mpl_connect("button_release_event", self._on_canvas_button_release)
        except ImportError:
            fallback = QLabel("matplotlib is not installed — preview unavailable.")
            fallback.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
            fallback.setWordWrap(True)
            layout.addWidget(fallback, stretch=1)

        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(4)
        status_row.addWidget(self._status, stretch=1)
        if self._nav_toolbar is not None:
            for key in ("pan", "zoom", "home"):
                action = self._nav_toolbar._actions.get(key)
                if action is None:
                    continue
                button = QToolButton(self)
                button.setDefaultAction(action)
                button.setAutoRaise(True)
                button.setIconSize(QSize(14, 14))
                if key == "home":
                    # Home = back to the solid-only autoscale contract.
                    action.triggered.connect(self._on_home_triggered)
                status_row.addWidget(button)
        layout.addLayout(status_row)

    # -- public API ------------------------------------------------------

    def request_preview(
        self,
        *,
        histograms: list[Histogram] | None,
        grouping: dict[str, Any],
        facts: PreviewFacts,
        facility: str = "",
        compare_stage: str | None = None,
        resolved_reference: BackgroundReference | None = None,
    ) -> None:
        """Queue a resolved draft and its reference snapshot for worker reduction."""
        if not histograms:
            self.show_unavailable()
            return
        if self._canvas is None:
            return  # matplotlib missing; fallback label already shown

        self.setVisible(True)
        self._queue_request(
            _PreviewRequest(
                generation=self._next_generation(),
                histograms=list(histograms),
                grouping=dict(grouping),
                facility=str(facility or "") or resolve_facility(grouping=grouping),
                facts=facts,
                compare_stage=compare_stage,
                resolved_reference=resolved_reference,
            )
        )

    def request_preview_from_profile(
        self,
        *,
        profile: GroupingProfile,
        run: Run | None,
        facts: PreviewFacts,
        facility: str = "",
        compare_stage: str | None = None,
        resolved_reference: BackgroundReference | None = None,
        timing_searches: dict[int, RunT0Search] | None = None,
    ) -> None:
        """Queue worker resolution of a copied profile and a read-only run snapshot."""
        histograms = list(run.histograms) if run is not None and run.histograms else []
        if not histograms:
            self.show_unavailable()
            return
        if self._canvas is None:
            return  # matplotlib missing; fallback label already shown

        self.setVisible(True)
        self._queue_request(
            _PreviewRequest(
                generation=self._next_generation(),
                histograms=histograms,
                profile=copy.deepcopy(profile),
                run=run,
                facility=str(facility or ""),
                facts=facts,
                compare_stage=compare_stage,
                resolved_reference=resolved_reference,
                timing_searches=dict(timing_searches or {}),
            )
        )

    @property
    def _last_result(self) -> _PreviewResult | None:
        return self._state if isinstance(self._state, _PreviewResult) else None

    @property
    def corrections_complete(self) -> bool:
        return not isinstance(self._state, str) and (
            (self._last_result is None and self._generation == 0)
            or (
                self._last_result is not None
                and self._last_result.generation == self._generation
                and all(outcome.complete for outcome in self._last_result.corrections)
            )
        )

    def show_error(self, message: str) -> None:
        self._generation += 1
        self._pending = self._deferred_result = None
        self._debounce.stop()
        self._state = message
        self._incomplete_banner.setText(f"Incomplete preview — {message}")
        self._incomplete_banner.show()
        self._set_error(message)
        self.preview_changed.emit()

    def set_view(self, view: str) -> None:
        """Draw the last result as *view* (``"asymmetry"`` / ``"counts"``).

        A redraw, never a recompute: one worker pass produces both views (D8).
        The user's pan/zoom is dropped, because a range chosen on a percent
        asymmetry over the good window means nothing on a log counts axis over
        the full histogram.
        """
        self._view = view
        self._user_view = None
        if self._last_result is not None:
            self._draw(self._last_result)

    def _next_generation(self) -> int:
        self._generation += 1
        return self._generation

    def _queue_request(self, request: _PreviewRequest) -> None:
        self._pending = request
        self._state = self._deferred_result = None
        self._incomplete_banner.setText("")
        self._incomplete_banner.hide()
        self._status.setText("Computing preview…")
        self._debounce.start()
        self.preview_changed.emit()

    def flush(self) -> None:
        """Dispatch any pending request immediately (used by tests)."""
        if self._debounce.isActive():
            self._debounce.stop()
            self._dispatch_pending()

    def shutdown(self) -> None:
        """Stop the debounce and tear down the runner (call from closeEvent)."""
        self._debounce.stop()
        self._pending = None
        self._tasks.shutdown()

    def _dispatch_pending(self) -> None:
        if self._pending is None:
            return
        if self._in_flight:
            return
        request = self._pending
        self._pending = None
        self._in_flight = True
        self._tasks.start(
            lambda worker: _run_reduction(worker, request),
            on_finished=self._on_finished,
            on_error=partial(self._on_error, generation=request.generation),
        )

    def _dispatch_next_after_completion(self) -> None:
        self._in_flight = False
        if self._pending is not None and not self._debounce.isActive():
            self._dispatch_pending()

    def _on_finished(self, result: object) -> None:
        self._dispatch_next_after_completion()
        if not isinstance(result, _PreviewResult):
            return
        if result.generation != self._generation:
            return  # superseded by a newer edit
        self._state = result
        items = [
            item
            for outcome in result.corrections
            for item in (outcome.deadtime, outcome.background)
        ]
        missing = [f"{item.name} ({item.reason})" for item in items if not item.usable]
        warnings = dict.fromkeys(item.warning for item in items if item.warning is not None)
        text = (
            "Incomplete preview — not applied: " + ", ".join(missing)
            if missing
            else " ".join(warnings)
        )
        self._incomplete_banner.setText(text)
        self._incomplete_banner.setVisible(bool(text))
        if all(outcome.complete for outcome in result.corrections):
            self.preview_completed.emit(result.alpha)
        self.preview_changed.emit()
        self._draw(result)

    def _nav_active(self) -> bool:
        toolbar = self._nav_toolbar
        return toolbar is not None and bool(str(getattr(toolbar, "mode", "")))

    def _drag_in_progress(self) -> bool:
        """A nav-mode mouse drag is live (press seen, release not yet)."""
        return self._nav_press_view is not None

    def _on_canvas_button_press(self, event: object) -> None:
        if self._nav_active() and self._axes is not None:
            self._nav_press_view = (self._axes.get_xlim(), self._axes.get_ylim())

    def _on_canvas_button_release(self, event: object) -> None:
        """Capture the user's view when a nav drag actually moved it.

        Toolbar release handlers run first, so the limits are final. A click
        that moves nothing does not freeze the autoscale; a deferred draw lands now.
        """
        press_view = self._nav_press_view
        self._nav_press_view = None
        if press_view is not None and self._axes is not None:
            view = (self._axes.get_xlim(), self._axes.get_ylim())
            if view != press_view:
                self._user_view = view
        if self._deferred_result is not None:
            deferred, self._deferred_result = self._deferred_result, None
            self._draw(deferred)

    def _on_home_triggered(self, *args: object) -> None:
        """Home: hand the view back to the solid-only autoscale contract."""
        self._user_view = None
        if self._last_result is not None:
            self._draw(self._last_result)

    def _on_error(self, message: str, *, generation: int) -> None:
        self._dispatch_next_after_completion()
        if generation == self._generation:
            self.show_error(f"Preview unavailable: {message}")

    def _draw(self, result: _PreviewResult) -> None:
        """Redraw the current view, preserving user limits until Home or a view switch."""
        if self._axes is None or self._canvas is None:
            return
        # Never clear the axes under a live pan/zoom rubber band — park the
        # result and draw it on release.
        if self._drag_in_progress():
            self._deferred_result = result
            return
        # User limits come from the drag, not from error/empty redraws.
        preserved = self._user_view
        # The axes rectangle is part of the user's chosen view: while they hold a
        # pan/zoom, tight layout must not re-fit it around the tick and marker
        # labels that moved with the limits.
        self._figure.set_layout_engine("none" if preserved is not None else "tight")
        self._axes.clear()
        if self._view == "counts":
            self._draw_counts(result, preserved)
        else:
            self._draw_asymmetry(result, preserved)
        self._axes.tick_params(labelsize=7)
        self._canvas.draw_idle()
        self._status.setText(_status_html(result.facts, result.time))

    def _draw_asymmetry(
        self,
        result: _PreviewResult,
        preserved: tuple[tuple[float, float], tuple[float, float]] | None,
    ) -> None:
        """Solid = the full reduction, ghost = the compare.

        The y-axis always follows the *solid* curve (its finite ``asymmetry ±
        error`` range, padded ~8%) — the ghost never influences the autoscale,
        because a deadtime-removed ghost can reach ~1e7 % (e.g. a FLAME run) and
        would crush the solid flat. There is no legend: the fixed caption in the
        top-left corner names both curves, at a placement independent of the
        data, so an off-scale ghost is named too.
        """
        # The counts view leaves the axis logarithmic; this one is always linear.
        self._axes.set_yscale("linear")
        if result.time.size == 0:
            self._axes.text(
                0.5,
                0.5,
                "No data in the good-bin window.",
                ha="center",
                va="center",
                transform=self._axes.transAxes,
                color=tokens.TEXT_MUTED,
            )
        else:
            solid_color = _solid_color(result.compare_stage)
            self._draw_solid(result, solid_color)
            self._axes.axhline(0.0, color=tokens.TEXT_MUTED, linewidth=0.5, alpha=0.5)
            # The ghost sits under the solid (zorder 3 under 4) but over the ±σ
            # band (2): where a small correction leaves the two nearly coincident,
            # the coloured as-reduced curve is the one on top.
            if result.baseline is not None:
                self._axes.plot(
                    result.time,
                    result.baseline,
                    color=_GHOST_COLOR,
                    linewidth=1.4,
                    alpha=0.9,
                    zorder=3,
                )
            if result.compare_stage == "alpha" and result.centre is not None:
                self._draw_residual_baseline(*result.centre, color=solid_color)
            reduced = f"as reduced · α = {result.alpha:.3f}"
            if abs(result.beta - 1.0) > 1e-12:
                reduced += f" · β = {result.beta:.3f}"
            rows = [_CaptionRow(solid_color, reduced)]
            if result.baseline is not None:
                stage = result.compare_stage
                rows.append(_CaptionRow(_GHOST_COLOR, COMPARE_STAGE_LABELS[stage]))
            caption_rows = self._draw_caption(rows)
            # Solid-only autoscale, set explicitly AFTER plotting so neither the
            # ghost nor matplotlib's own autoscale can widen the range — unless
            # the user panned/zoomed, in which case their view wins verbatim.
            if preserved is not None:
                self._axes.set_xlim(*preserved[0])
                limits: tuple[float, float] | None = preserved[1]
            else:
                limits = _solid_ylimits(
                    result.asymmetry,
                    result.error,
                    top_headroom=_CAPTION_ROW_DY * caption_rows + _CAPTION_HEADROOM_PAD,
                )
            if limits is not None:
                self._axes.set_ylim(*limits)
        self._axes.set_xlabel("Time (µs)", fontsize=8)
        self._axes.set_ylabel("Asymmetry (%)", fontsize=8)

    def _draw_counts(
        self,
        result: _PreviewResult,
        preserved: tuple[tuple[float, float], tuple[float, float]] | None,
    ) -> None:
        """Draw single-period spectra on their observed bin-centre grid.

        Background-subtracted counts at or below zero use the log-plot floor.
        Combined asymmetries do not define a single count spectrum.
        """
        counts = result.counts
        if counts is None or counts.forward.size == 0:
            self._axes.text(
                0.5,
                0.5,
                "Counts view requires a single selected period."
                if counts is None
                else "No counts in this run's histograms.",
                ha="center",
                va="center",
                transform=self._axes.transAxes,
                color=tokens.TEXT_MUTED,
            )
            return
        self._axes.set_yscale("log")
        time = counts.time
        forward = np.maximum(counts.forward, 1.0)
        backward = np.maximum(counts.backward, 1.0)

        stage = result.compare_stage
        solid_color = _solid_color(stage)
        rows = [
            _CaptionRow(solid_color, f"F: {result.facts.forward_name} · as reduced"),
            _CaptionRow(
                _lighter(solid_color), f"B: {result.facts.backward_name} · as reduced", dashed=True
            ),
        ]
        if counts.ghost_forward is not None:
            rows.append(_CaptionRow(_GHOST_COLOR, COMPARE_STAGE_LABELS[stage]))
        elif stage in _ASYMMETRY_STAGE_SYMBOLS:
            rows.append(
                _CaptionRow(
                    solid_color,
                    f"{_ASYMMETRY_STAGE_SYMBOLS[stage]} acts when the asymmetry is "
                    "formed — see the Asymmetry view",
                )
            )

        # Limits first: the markers below are drawn only where they land inside
        # the view, and axvline/axhline would otherwise pull the range to them.
        if preserved is not None:
            xlimits, ylimits = preserved
        else:
            # The histogram's own edges, not its first and last drawn bin
            # centres: the good window's trailing rule sits on bin
            # ``last_good + 1``'s edge, which for a window ending at the last
            # bin — the default — is half a bin past the last centre and would
            # be culled with its label.
            xlimits = (counts.bin_edge_time(0), counts.bin_edge_time(counts.n_bins))
            # The same headroom rule as the asymmetry view, applied in log₁₀
            # space where the caption rows are laid out: the drawn (clipped)
            # spectra alone set the range, so a compare ghost can never widen it.
            drawn = np.concatenate((forward, backward))
            decades = _solid_ylimits(
                np.log10(drawn),
                np.zeros_like(drawn),
                top_headroom=_CAPTION_ROW_DY * len(rows) + _CAPTION_HEADROOM_PAD,
            )
            ylimits = (10.0 ** decades[0], 10.0 ** decades[1])

        self._draw_counts_regions(counts, result.facts.t0_mode_label, xlimits)
        self._axes.plot(time, forward, color=solid_color, linewidth=1.2, zorder=4)
        self._axes.plot(
            time,
            backward,
            color=_lighter(solid_color),
            linewidth=1.2,
            linestyle=_BACKWARD_DASH,
            zorder=4,
        )
        if counts.ghost_forward is not None:
            self._axes.plot(
                time,
                np.maximum(counts.ghost_forward, 1.0),
                color=_GHOST_COLOR,
                linewidth=1.3,
                alpha=0.9,
                zorder=3,
            )
            self._axes.plot(
                time,
                np.maximum(counts.ghost_backward, 1.0),
                color=_lighter(_GHOST_COLOR),
                linewidth=1.3,
                alpha=0.9,
                linestyle=_BACKWARD_DASH,
                zorder=3,
            )
        level = counts.background_level
        # Drawn only where it lands inside the view, for the same reason the
        # window rules are: a marker pinned to the axis edge would claim the
        # level is somewhere it is not. It sits in range whenever the spectra
        # decay towards it, which is the case the marker exists for.
        if level is not None and ylimits[0] <= level[0] <= ylimits[1]:
            self._draw_background_level(counts)
        self._draw_caption(rows)

        self._axes.set_xlim(*xlimits)
        self._axes.set_ylim(*ylimits)
        self._axes.set_xlabel("Time after t0 (µs)", fontsize=8)
        self._axes.set_ylabel("Counts / bin", fontsize=8)

    def _draw_counts_regions(
        self,
        counts: CountsCurves,
        t0_mode_label: str,
        xlimits: tuple[float, float],
    ) -> None:
        """Shade what the asymmetry never sees, and rule the bins that bound it.

        The good window is the inclusive bin range ``[first_good, last_good]``,
        so in time it spans its first bin's leading edge to its last bin's
        trailing edge — the two rules and the two shaded regions share those
        edges. A rule outside the current x-range is dropped along with its
        label rather than clamped to the axis edge, which would claim a bin sits
        somewhere it does not.
        """
        window_start = counts.bin_edge_time(counts.first_good)
        window_end = counts.bin_edge_time(counts.last_good + 1)
        # Pre-t0 and merely out-of-window bins are different exclusions, so the
        # window shading is lighter than the (opaque) pre-t0 band and the two
        # read apart where they overlap.
        self._axes.axvspan(
            xlimits[0], counts.bin_edge_time(counts.t0_bin), color=tokens.SURFACE_HI, zorder=1
        )
        self._axes.axvspan(xlimits[0], window_start, color=tokens.SURFACE_HI, alpha=0.6, zorder=1)
        self._axes.axvspan(window_end, xlimits[1], color=tokens.SURFACE_HI, alpha=0.6, zorder=1)

        offset = counts.first_good - counts.t0_bin
        # (x, colour, dash, label side, label height in axes fraction, label).
        # The t0 and first-good labels sit at different heights because a small
        # t_good offset puts their rules within a few pixels of each other.
        rules = (
            (
                0.0,
                tokens.PLOT_AXIS,
                (0, (2, 2)),
                "left",
                0.03,
                f"t0 · bin {counts.t0_bin} ({t0_mode_label})",
            ),
            (
                window_start,
                tokens.TEXT_MUTED,
                "solid",
                "left",
                0.14,
                f"good window: t_good offset {offset} bins",
            ),
            (
                window_end,
                tokens.TEXT_MUTED,
                "solid",
                "right",
                0.03,
                f"last good bin {counts.last_good}",
            ),
        )
        for x, color, dash, side, y, label in rules:
            if not xlimits[0] <= x <= xlimits[1]:
                continue
            self._axes.axvline(x, color=color, linewidth=1.0, linestyle=dash, zorder=2)
            self._axes.text(
                x,
                y,
                label,
                transform=self._axes.get_xaxis_transform(),
                ha=side,
                va="bottom",
                fontsize=7,
                color=tokens.TEXT_MUTED,
                clip_on=True,
                zorder=5,
                bbox=dict(_TEXT_BBOX),
            )

    def _draw_background_level(self, counts: CountsCurves) -> None:
        """Rule the constant level subtracted from each group, named with its mode."""
        forward_level, backward_level = counts.background_level
        self._axes.axhline(
            forward_level,
            color=tokens.STAGE_BACKGROUND,
            linewidth=1.0,
            linestyle=(0, (4, 3)),
            zorder=4,
        )
        mode = _BACKGROUND_MODE_LABELS[counts.background_mode]
        self._axes.text(
            0.99,
            forward_level,
            f"background level · F {forward_level:.1f} / B {backward_level:.1f} "
            f"counts per bin ({mode})",
            transform=self._axes.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=7,
            color=tokens.STAGE_BACKGROUND,
            clip_on=True,
            zorder=5,
            bbox=dict(_TEXT_BBOX),
        )

    def _draw_solid(self, result: _PreviewResult, color: str) -> None:
        """The "as reduced" curve: a line with a ±σ band, or markers when sparse."""
        if result.time.size > _LINE_MODE_POINTS:
            self._axes.plot(result.time, result.asymmetry, color=color, linewidth=1.2, zorder=4)
            self._axes.fill_between(
                result.time,
                result.asymmetry - result.error,
                result.asymmetry + result.error,
                color=color,
                alpha=0.18,
                linewidth=0,
                zorder=2,
            )
            return
        self._axes.errorbar(
            result.time,
            result.asymmetry,
            yerr=result.error,
            fmt="o",
            markersize=2.0,
            linewidth=0.0,
            elinewidth=0.5,
            capsize=0.0,
            color=color,
            ecolor=tokens.TEXT_MUTED,
            zorder=4,
        )

    def _draw_residual_baseline(self, mean: float, err: float, *, color: str) -> None:
        """⟨A⟩ drawn where it lives: a dashed line across the curve it describes."""
        self._axes.axhline(mean, color=color, linewidth=1.0, linestyle=(0, (4, 3)), zorder=4)
        self._axes.text(
            0.99,
            mean,
            f"⟨A⟩ = {mean:.3f} ± {err:.3f} % (residual baseline)",
            transform=self._axes.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=7,
            color=color,
            clip_on=True,
            # Above both curves, or they paint over the backing.
            zorder=5,
            bbox=dict(_TEXT_BBOX),
        )

    def _draw_caption(self, rows: list[_CaptionRow]) -> int:
        """Name curves in drawing order; row count sets autoscale headroom."""
        for index, (color, text, dashed) in enumerate(rows):
            y = _CAPTION_TOP - index * _CAPTION_ROW_DY
            # Axes-fraction coordinates leave dataLim untouched, so the caption
            # cannot influence the solid-only autoscale below.
            self._axes.plot(
                [_CAPTION_X, _CAPTION_X + _CAPTION_SWATCH_W],
                [y, y],
                transform=self._axes.transAxes,
                color=color,
                linewidth=1.4,
                linestyle=_BACKWARD_DASH if dashed else "solid",
                solid_capstyle="butt",
                zorder=5,
            )
            self._axes.text(
                _CAPTION_X + _CAPTION_SWATCH_W + 0.012,
                y,
                text,
                transform=self._axes.transAxes,
                va="center",
                ha="left",
                fontsize=7,
                color=color,
                zorder=5,
                bbox=dict(_TEXT_BBOX),
            )
        return len(rows)

    def _set_error(self, message: str) -> None:
        self._status.setText(message)
        if self._axes is not None and self._canvas is not None:
            self._axes.clear()
            self._axes.tick_params(labelsize=7)
            self._canvas.draw_idle()

    def show_unavailable(self) -> None:
        """Hide the pane with a note: this dataset has no histograms to reduce."""
        self._generation += 1
        self._state = self._deferred_result = None
        self._incomplete_banner.hide()
        self.setVisible(False)
        self._status.setText("Preview needs raw detector histograms (none loaded).")
        self.preview_changed.emit()


def _run_reduction(worker: TaskWorker, request: _PreviewRequest) -> _PreviewResult:
    """Reduce immutable request snapshots off-thread, with cooperative cancellation."""
    if worker.is_cancelled():
        raise TaskCancelledError

    def resolve_reference(grouping):
        return background_reference_for_grouping(request.resolved_reference, grouping)

    def search_timing(histograms, metadata, *, period_index=None, **options):
        from asymmetry.core.io.periods import selected_period_index

        if period_index is None:
            period_index = (
                selected_period_index(request.run.grouping, request.profile.period_mode)
                if request.profile is not None
                else 0
            )
        cached = request.timing_searches.get(period_index or 0)
        if cached is not None:
            return cached
        from asymmetry.core.transform.t0 import find_t0_for_run

        return find_t0_for_run(
            histograms, metadata, cancel_check=partial(check_task_cancelled, worker)
        )

    if request.profile is not None:
        grouping = resolve_effective_grouping(
            request.profile,
            request.run,
            reference_resolver=resolve_reference,
            timing_search=search_timing,
        )
    else:
        grouping = request.grouping or {}

    from asymmetry.core.io.periods import select_period_histograms, selected_period_index

    histograms = request.histograms
    period_index = selected_period_index(grouping)
    if period_index is not None:
        histograms, grouping = select_period_histograms(histograms, grouping, period_index)
    metadata = (
        dict(
            request.run.metadata,
            source_file=request.run.source_file,
            run_number=request.run.run_number,
        )
        if request.run is not None
        else {}
    )
    if period_index is not None:
        metadata["period_number"] = period_index + 1
    n_hist = len(histograms)
    forward_gid = _as_int(grouping.get("forward_group"), 1)
    backward_gid = _as_int(grouping.get("backward_group"), 2)
    forward_idx = effective_group_indices(grouping, forward_gid, n_histograms=n_hist)
    backward_idx = effective_group_indices(grouping, backward_gid, n_histograms=n_hist)
    if worker.is_cancelled():
        raise TaskCancelledError
    flags = correction_flags_from_grouping(grouping)
    use_deadtime = flags.use_deadtime
    use_background = flags.use_background
    facility = request.facility or resolve_facility(grouping=grouping)

    mode = grouping.get("period_mode", PeriodMode.RED)
    if mode in (PeriodMode.GREEN_MINUS_RED, PeriodMode.GREEN_PLUS_RED):
        alpha, beta = float(grouping.get("alpha", 1.0)), float(grouping.get("beta", 1.0))
        reduced = reduce_period_asymmetry(
            histograms,
            grouping,
            mode,
            binning=binning_spec(grouping),
            forward_idx=forward_idx,
            backward_idx=backward_idx,
            alpha=alpha,
            beta=beta,
            use_deadtime=use_deadtime,
            deadtime_mode=flags.deadtime_mode,
            use_background=use_background,
            facility=facility,
            metadata=metadata,
            timing_search=search_timing,
            reference_resolver=resolve_reference,
            allow_incomplete=True,
        )
        if worker.is_cancelled():
            raise TaskCancelledError
        time, asymmetry, error = _decimate_for_preview(
            reduced.time, reduced.asymmetry, reduced.error, _MAX_PREVIEW_POINTS
        )
        return _PreviewResult(
            generation=request.generation,
            facts=replace(request.facts, member_counts=(len(forward_idx), len(backward_idx))),
            time=time,
            asymmetry=asymmetry,
            error=error,
            counts=None,
            alpha=alpha,
            beta=beta,
            corrections=tuple(reduced.corrections.values()),
        )

    def _reduce(dt: bool, bg: bool):
        return corrected_grouped_counts(
            histograms=histograms,
            grouping=grouping,
            forward_idx=forward_idx,
            backward_idx=backward_idx,
            use_deadtime=dt,
            deadtime_mode=flags.deadtime_mode,
            use_background=bg,
            facility=facility,
            metadata=metadata,
            reference_resolver=resolve_reference,
            allow_incomplete=True,
            output_binning=binning_spec(grouping),
        )

    corrected = _reduce(use_deadtime, use_background)
    first_good, last_good = corrected.window.first, corrected.window.last
    alpha = _as_float(grouping.get("alpha"), 1.0)
    beta = _as_float(grouping.get("beta"), 1.0)
    if not np.isfinite(beta) or beta <= 0.0:
        beta = 1.0

    time, asymmetry, error = _form_asymmetry(corrected, grouping, alpha, beta=beta)

    compare = request.compare_stage
    baseline = None
    centre: tuple[float, float] | None = None
    ghost_counts = None
    if compare == "alpha":
        centre = _weighted_centre(asymmetry, error)
        if abs(alpha - 1.0) > 1e-12:
            _bt, base_asym, _be = _form_asymmetry(corrected, grouping, 1.0, beta=beta)
            _dt, baseline, _de = _decimate_for_preview(time, base_asym, error, _MAX_PREVIEW_POINTS)
    elif compare == "beta" and abs(beta - 1.0) > 1e-12:
        _bt, base_asym, _be = _form_asymmetry(corrected, grouping, alpha, beta=1.0)
        _dt, baseline, _de = _decimate_for_preview(time, base_asym, error, _MAX_PREVIEW_POINTS)
    elif (compare == "deadtime" and use_deadtime) or (compare == "background" and use_background):
        if worker.is_cancelled():
            raise TaskCancelledError
        ghost_counts = _reduce(
            use_deadtime and compare != "deadtime",
            use_background and compare != "background",
        )
        ghost = _form_asymmetry(ghost_counts, grouping, alpha, beta=beta)
        _dt, baseline, _de = _decimate_for_preview(time, ghost[1], error, _MAX_PREVIEW_POINTS)

    time, asymmetry, error = _decimate_for_preview(time, asymmetry, error, _MAX_PREVIEW_POINTS)
    return _PreviewResult(
        generation=request.generation,
        facts=replace(request.facts, member_counts=(len(forward_idx), len(backward_idx))),
        time=time,
        asymmetry=asymmetry,
        error=error,
        counts=_counts_curves(corrected, ghost_counts, grouping, first_good, last_good),
        alpha=alpha,
        beta=beta,
        baseline=baseline,
        compare_stage=compare,
        centre=centre,
        corrections=(corrected.corrections,),
    )


def _counts_curves(
    corrected: ForwardBackwardCounts,
    ghost: ForwardBackwardCounts | None,
    grouping: dict[str, Any],
    first_good: int,
    last_good: int,
) -> CountsCurves:
    """Keep observed centres (k + ½)w − t0 and undecimated window edges in μs."""
    n = int(corrected.forward.counts.size)
    step = _preview_stride(n, _MAX_PREVIEW_POINTS)
    bins = np.arange(0, n, step, dtype=np.float64)
    ghost_forward = (
        None if ghost is None else np.asarray(ghost.forward.counts[::step], dtype=np.float64)
    )
    ghost_backward = (
        None if ghost is None else np.asarray(ghost.backward.counts[::step], dtype=np.float64)
    )
    return CountsCurves(
        time=(bins + 0.5) * corrected.bin_width_us - corrected.t0_time_us,
        forward=np.asarray(corrected.forward.counts[::step], dtype=np.float64),
        backward=np.asarray(corrected.backward.counts[::step], dtype=np.float64),
        bin_width=float(corrected.bin_width_us),
        n_bins=n,
        t0_bin=int(corrected.common_t0_bin),
        first_good=first_good,
        last_good=last_good,
        background_level=corrected.background_level,
        background_mode=resolve_background_mode(grouping),
        ghost_forward=ghost_forward,
        ghost_backward=ghost_backward,
    )


def _form_asymmetry(
    corrected: ForwardBackwardCounts,
    grouping: dict[str, Any],
    alpha: float,
    *,
    beta: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Form percent asymmetry on the same count edges as final reduction."""
    time, asym, err = binned_fb_asymmetry(
        corrected,
        binning=binning_spec(grouping),
        alpha=alpha,
        beta=beta,
    )
    return (
        np.asarray(time, dtype=np.float64),
        np.asarray(asym, dtype=np.float64) * 100.0,
        np.asarray(err, dtype=np.float64) * 100.0,
    )


def _solid_ylimits(
    asymmetry: np.ndarray,
    error: np.ndarray,
    pad_fraction: float = 0.08,
    *,
    top_headroom: float = 0.0,
) -> tuple[float, float] | None:
    """Y-limits covering the solid curve's finite ``asymmetry ± error``, padded.

    The preview's autoscale contract: only the solid (fully-reduced) curve sets
    the range — a compare ghost, which can sit orders of magnitude away, must
    never crush it. *top_headroom* is an extra band above the symmetric pad, as
    a fraction of the finite range, reserved for the corner caption so the curve
    does not climb under it. Returns ``None`` when nothing is finite (caller
    keeps matplotlib's default limits).
    """
    a = np.asarray(asymmetry, dtype=np.float64)
    e = np.asarray(error, dtype=np.float64)
    e = np.where(np.isfinite(e), np.abs(e), 0.0)
    finite = np.isfinite(a)
    if not finite.any():
        return None
    lo = float(np.min(a[finite] - e[finite]))
    hi = float(np.max(a[finite] + e[finite]))
    span = hi - lo
    pad = pad_fraction * span
    if pad <= 0.0:
        pad = max(1.0, abs(hi) * pad_fraction)  # flat curve: keep a visible band
        span = 2.0 * pad
    return lo - pad, hi + pad + top_headroom * span


def _weighted_centre(asymmetry: np.ndarray, error: np.ndarray) -> tuple[float, float] | None:
    """Inverse-variance weighted mean of the asymmetry and its error.

    This is the residual baseline the α compare draws: for a weak-TF calibration
    run a balanced α drives it to zero, so it is the honest numeric replacement
    for eyeballing whether the oscillation sits on zero. ``None`` when no point
    carries a usable weight.
    """
    a = np.asarray(asymmetry, dtype=np.float64)
    e = np.asarray(error, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(e) & (e > 0.0)
    if not mask.any():
        return None
    weights = 1.0 / np.square(e[mask])
    total = float(np.sum(weights))
    if total <= 0.0:
        return None
    mean = float(np.sum(a[mask] * weights) / total)
    return mean, float(np.sqrt(1.0 / total))


def _status_html(facts: PreviewFacts, time: np.ndarray) -> str:
    """Name the reduced pair, effective channel counts, and actual time window."""
    details = [
        f"F = {facts.forward_name}",
        f"B = {facts.backward_name}",
        f"bin {facts.bunch}",
    ]
    if facts.member_counts is not None:
        details.append(f"{facts.member_counts[0]}/{facts.member_counts[1]} channels")
    if time.size:
        details.append(f"{float(time[0]):.1f} – {float(time[-1]):.1f} µs")
    details.extend(facts.corrections)
    if facts.vector_pair is not None:
        details.append(f"{facts.vector_pair} pair")
    return (
        f'<span style="color:{tokens.TEXT_MUTED}; font-weight:600; letter-spacing:0.04em">'
        f"PREVIEW</span>&nbsp; "
        f'<b style="color:{tokens.TEXT}">{escape(facts.run_label)} (selected run)</b> '
        f'<span style="color:{tokens.TEXT_MUTED}">· {escape(" · ".join(details))}</span>'
    )


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = ["COMPARE_STAGE_LABELS", "CountsCurves", "GroupingPreviewPane", "PreviewFacts"]

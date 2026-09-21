"""Live forward/backward asymmetry preview for the grouping editor.

The grouping dialog edits an in-memory *draft* grouping profile; without a
preview, the effect of a grouping / alpha / binning / exclusion / deadtime /
background change is only visible after Apply. :class:`GroupingPreviewPane`
closes that loop: it reduces the *preview run* under the *current draft* and
draws the resulting asymmetry curve, updating (debounced) as the user edits.

Threading contract (see ``AGENTS.md`` and ``gui/tasks.py``): the reduction — the
only expensive step — runs on a :class:`~asymmetry.gui.tasks.TaskRunner` worker
thread and never on the GUI thread. Edits are debounced with a ~300 ms timer;
each computation carries a generation counter so a result from a superseded edit
is dropped on arrival. Results cross back as plain numpy arrays through the
TaskRunner's GUI-thread relay (never a bare lambda touching widgets). The runner
is shut down in :meth:`shutdown`, called from the dialog's ``closeEvent``.

Resolution happens on the worker thread too: :meth:`request_preview_from_profile`
takes the *unresolved* draft profile plus the preview run, and the worker calls
:func:`resolve_effective_grouping` before reducing. That call can be expensive —
an ``auto_detect`` t0 policy scans every detector's full histogram and a
``per_run_estimate`` alpha policy sums whole groups — so it must never run on
the GUI thread per edit. The profile is deep-copied at request time so later
form edits cannot race the worker; the run is shared read-only (the dialog does
not mutate it while open). Vector-mode drafts are previewed on their primary
forward/backward pair (the resolved ``forward_group`` / ``backward_group``; for
canonical EMU that is the P_z axis).

Drawing contract (see :meth:`GroupingPreviewPane._draw`): the solid curve is
always the full configured reduction and alone sets the y-limits; a compare
ghost is drawn just beneath it — the as-reduced curve stays on top, so where a
small correction leaves the two nearly coincident it is the reduction the eye
follows — and a fixed caption in the axes' top-left names both curves (no legend — the pager label and the focused correction card name
the same comparison). Colour means "with this correction": while a stage is
focused the *solid* wears that stage's identity colour and the ghost is grey
(D11 of ``docs/plans/grouping-preview.md``).

One reduction serves two views (D7/D8 of ``docs/plans/grouping-preview.md``):
the asymmetry over the good window, and the count-domain view of the corrected
F/B group spectra over the *full* histogram with t0, the good window and the
subtracted background level marked. Both are drawn from the same
:class:`_PreviewResult`, so :meth:`GroupingPreviewPane.set_view` is a redraw and
never dispatches the worker.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from html import escape
from typing import Any, NamedTuple

import numpy as np
from PySide6.QtCore import QSize, QTimer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.project.profiles import GroupingProfile, resolve_effective_grouping
from asymmetry.core.transform import (
    binned_fb_asymmetry,
    corrected_grouped_counts,
    correction_flags_from_grouping,
    effective_group_indices,
    resolve_background_mode,
    resolve_facility,
)
from asymmetry.gui.styles import tokens
from asymmetry.gui.tasks import TaskCancelledError, TaskRunner, TaskWorker
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

#: How the Counts view words each constant-level background mode. Only these
#: modes subtract a level (``reference_run`` subtracts a spectrum, so
#: ``background_level`` is ``None`` and no line is drawn).
_BACKGROUND_MODE_LABELS: dict[str, str] = {
    "fixed": "fixed",
    "range": "pre-t0 range",
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
    """What the status strip says the preview is showing (D1).

    Plain values read off the draft on the GUI thread and carried through the
    worker untouched, so the strip can never name a different run, pair or
    binning than the curve beside it.
    """

    #: The reference dataset's user-facing run label ("7101", "3039 + 3040").
    run_label: str
    forward_name: str
    backward_name: str
    #: Bunching factor, as the binning row states it.
    bunch: int
    #: Where the previewed t0 came from — ``"from file"``, ``"manual"`` or
    #: ``"detected"``, as the Counts view's t0 marker names it. A dialog fact:
    #: the resolved grouping carries the bin, not the policy that chose it.
    t0_mode_label: str
    #: Active count-domain corrections, worded as the pipeline chips word them.
    corrections: tuple[str, ...] = ()
    #: In vector mode, the primary projection previewed (e.g. ``"P_z"``); the
    #: other projections are not reduced here. ``None`` outside vector mode.
    vector_pair: str | None = None


@dataclass(frozen=True)
class _PreviewRequest:
    """An immutable snapshot of what to resolve/reduce, built on the GUI thread.

    Everything here is a plain object, so the worker function can run entirely
    off the GUI thread. Exactly one of two shapes is populated: a pre-resolved
    ``grouping`` payload, or an unresolved ``profile`` + ``run`` pair that the
    worker resolves first (the expensive path — auto t0 / per-run alpha scans).
    Index/alpha/policy extraction from the resolved payload happens in the
    worker for both shapes.
    """

    generation: int
    histograms: list[Histogram]
    facility: str
    facts: PreviewFacts
    grouping: dict[str, Any] | None = None
    profile: GroupingProfile | None = None
    run: Run | None = None
    #: Stage-generic before/after compare (one focused stage at a time). The solid
    #: curve is always the full configured reduction; the *ghost* removes one
    #: stage: ``"alpha"``/``"beta"`` ghost that factor at unity from the same
    #: corrected counts (α also reports the residual baseline), and
    #: ``"deadtime"``/``"background"`` ghost a *second* corrected pass with that one
    #: stage dropped. ``None`` draws only the solid curve. Preview-only — it never
    #: touches the persisted reduction, and it never degrades the solid, so the α
    #: residual ⟨A⟩ is always read off the fully-corrected curve.
    compare_stage: str | None = None


@dataclass(frozen=True)
class CountsCurves:
    """The count-domain view of one reduction (D8): spectra plus their markers.

    ``forward``/``backward`` are the *corrected* group counts — deadtime
    corrected, grouped, background subtracted — over the **full** histogram from
    bin 0, not the good window the asymmetry is formed on. They and ``time`` are
    decimated on one shared stride, so a drawn sample keeps its partner sample.
    ``time`` stamps each drawn bin's *centre* relative to t0, matching the
    asymmetry axis' convention; the bin *edges* the window markers sit on come
    from :meth:`bin_edge_time`.
    """

    #: Bin-centre times of the drawn bins, in µs after t0.
    time: np.ndarray
    forward: np.ndarray
    backward: np.ndarray
    #: Bin width in µs (undecimated — one histogram bin, not one drawn sample).
    bin_width: float
    #: Undecimated histogram length. The drawn arrays are strided, so their last
    #: sample is not the last bin; the view's x-range runs to bin ``n_bins``'s
    #: leading edge, which is the only bound every window marker lies inside.
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
    #: The same reduction seen in the count domain — one worker pass feeds both
    #: views, so switching view is a redraw (D8).
    counts: CountsCurves
    #: The (α, β) the solid curve was formed with — the caption quotes them.
    alpha: float
    beta: float
    #: Compare extras (all ``None`` unless the request asked for a compare, or
    #: the focused stage was not applied and so had nothing to remove).
    #: ``baseline`` is the ghost curve (aligned to ``time``), named by
    #: ``COMPARE_STAGE_LABELS[compare_stage]``.
    baseline: np.ndarray | None = None
    compare_stage: str | None = None
    #: Residual baseline ``(⟨A⟩, its error)`` in percent — the α compare only,
    #: and ``None`` when no point of the curve is finite. One field, because a
    #: mean without its error is not a state this can be in.
    centre: tuple[float, float] | None = None


class GroupingPreviewPane(QWidget):
    """Embedded canvas + status strip showing the draft's F/B asymmetry.

    The owning dialog calls :meth:`request_preview` (from any of its refresh
    seams) with the resolved effective grouping and the preview run; the pane
    debounces, reduces off-thread, and redraws. Reduction failures show a muted
    message in the status strip — never a popup or a crash.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the canvas, status strip, and debounce/worker plumbing."""
        super().__init__(parent)
        self.setFixedHeight(_PANE_HEIGHT)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(2)

        self._status = QLabel("")
        self._status.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        self._status.setWordWrap(True)

        self._tasks = TaskRunner(self)
        self._generation = 0
        self._pending: _PreviewRequest | None = None
        #: Coalescing guard: at most one reduction runs at a time. While a task
        #: is in flight, newer requests wait in ``_pending`` (latest wins) and
        #: are dispatched from the finished/error callback instead of spawning
        #: concurrent worker threads.
        self._in_flight = False

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(_DEBOUNCE_MS)
        self._debounce.timeout.connect(self._dispatch_pending)

        self._figure = None
        self._canvas = None
        self._axes = None
        self._nav_toolbar = None
        #: Which view of the result is drawn — ``"asymmetry"`` or ``"counts"``
        #: (D7). Preview-only state, owned here; the dialog's segmented control
        #: drives it through :meth:`set_view`.
        self._view = "asymmetry"
        #: Last drawn result, retained so Home can redraw with fresh autoscale
        #: without a recompute.
        self._last_result: _PreviewResult | None = None
        #: The user's chosen ``(xlim, ylim)`` once a pan/zoom drag actually moved
        #: the view, or ``None`` while the solid-only autoscale owns it. Stored
        #: explicitly — never re-read from the axes at draw time, because error
        #: and empty-data paths ``clear()`` the axes (resetting the limits to
        #: matplotlib's defaults), and capturing those would freeze the preview
        #: on a garbage view. Home resets to ``None``.
        self._user_view: tuple[tuple[float, float], tuple[float, float]] | None = None
        #: Limits at nav-drag press, to tell a real drag from a mode-on click.
        self._nav_press_view: tuple | None = None
        #: A result whose draw arrived mid-drag; drawn on release instead, so a
        #: debounced redraw never clears the axes under the rubber band.
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
    ) -> None:
        """Queue a (debounced) recompute of the preview for the current draft.

        *grouping* is the draft resolved against the preview run — i.e. exactly
        the ``run.grouping`` shape the reduction consumes. When the dataset has no
        histograms (co-added curves) the pane hides itself with a note; nothing is
        scheduled. *facts* is what the status strip states about this preview;
        *compare_stage* draws one stage's before/after ghost. Both are
        preview-only and never touch the persisted reduction.
        """
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
    ) -> None:
        """Queue a (debounced) resolve + recompute for an unresolved draft.

        Unlike :meth:`request_preview`, resolution against the run —
        :func:`resolve_effective_grouping`, which may scan every detector for an
        ``auto_detect`` t0 policy or sum whole groups for a per-run alpha
        estimate — happens on the worker thread. *profile* is deep-copied here
        so subsequent form edits cannot race the in-flight worker; *run* is
        shared read-only.
        """
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
            )
        )

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
        self._status.setText("Computing preview…")
        self._debounce.start()

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

    # -- dispatch + worker ----------------------------------------------

    def _dispatch_pending(self) -> None:
        if self._pending is None:
            return
        if self._in_flight:
            # Coalesce: keep the latest request pending; the finished/error
            # callback of the running task dispatches it. This bounds the
            # worker-thread count at one no matter how fast edits arrive.
            return
        request = self._pending
        self._pending = None
        self._in_flight = True
        # A ``reference_run`` background needs the loaded-dataset registry the
        # dialog does not own; the preview simply skips that subtraction (the
        # resolver returns None) so the curve still renders.
        self._tasks.start(
            lambda worker: _run_reduction(worker, request),
            on_finished=self._on_finished,
            on_error=self._on_error,
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
        self._last_result = result
        self._draw(result)

    # -- pan/zoom --------------------------------------------------------

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

        Runs after the toolbar's own release handlers (registered first), so the
        limits are final. A mode-on click that moved nothing captures nothing —
        otherwise a stray click would freeze the autoscale at its current range.
        Any redraw deferred during the drag lands now.
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

    def _on_error(self, message: str) -> None:
        self._dispatch_next_after_completion()
        self._set_error(f"Preview unavailable: {message}")

    # -- drawing ---------------------------------------------------------

    def _draw(self, result: _PreviewResult) -> None:
        """Redraw the preview in the current view; the status strip is shared.

        Once the user pans/zooms, their view is preserved verbatim across
        redraws until Home (or a view switch) resets it.
        """
        if self._axes is None or self._canvas is None:
            return
        # Never clear the axes under a live pan/zoom rubber band — park the
        # result and draw it on release.
        if self._drag_in_progress():
            self._deferred_result = result
            return
        # A user-chosen pan/zoom view survives the debounced redraws — an edit
        # mid-inspection must not yank the axes back — until Home resets it.
        # `_user_view` is the stored drag result, deliberately NOT re-read from
        # the axes here: the error/empty paths clear() the axes back to default
        # limits, and re-capturing those would freeze the preview on a garbage
        # (0..1) view — the "strange state" this guards against.
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
        """The count domain (D7): corrected F/B spectra with the settings marked.

        Every count-domain setting the dialog edits acts here and nowhere the
        asymmetry can show it — t0 shifts the origin, the good window picks the
        bins the asymmetry is formed from, and the background is a level taken
        off both spectra. The spectra span decades, so the axis is log₁₀; a
        background-subtracted bin at or below zero has no logarithm and is drawn
        on the floor at 1.

        Colour encodes *with* the focused correction and grey its absence (D11);
        line style encodes the group — F solid, B dashed — so both codings are
        readable at once.
        """
        counts = result.counts
        if counts.forward.size == 0:
            self._axes.text(
                0.5,
                0.5,
                "No counts in this run's histograms.",
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
        """Name the curves in the axes' top-left; returns the rows drawn.

        One row per curve, in drawing order, its swatch in the curve's colour
        and line style. The caller sizes the autoscale's top headroom from the
        returned count, so no curve climbs under the rows that name it.
        """
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
        self.setVisible(False)
        self._status.setText("Preview needs raw detector histograms (none loaded).")


def _run_reduction(worker: TaskWorker, request: _PreviewRequest) -> _PreviewResult:
    """Resolve (if needed) and reduce one preview request off the GUI thread.

    Pure numpy work: it touches no widgets and returns plain arrays, so the
    TaskRunner relay can marshal the result back safely. Cooperative cancellation
    is honoured up front (``worker.is_cancelled()``) so a shutdown mid-flight
    stops promptly; a merely *superseded* result is dropped later by generation
    in :meth:`GroupingPreviewPane._on_finished`. Raised errors (including the
    no-detectors case) surface through the pane's error status strip.
    """
    if worker.is_cancelled():
        raise TaskCancelledError
    if request.profile is not None:
        # The expensive step for auto-t0 / per-run-alpha policies; must stay
        # off the GUI thread. The profile is the pane's private deep copy.
        grouping = resolve_effective_grouping(request.profile, request.run)
    else:
        grouping = request.grouping or {}

    n_hist = len(request.histograms)
    forward_gid = _as_int(grouping.get("forward_group"), 1)
    backward_gid = _as_int(grouping.get("backward_group"), 2)
    forward_idx = effective_group_indices(grouping, forward_gid, n_histograms=n_hist)
    backward_idx = effective_group_indices(grouping, backward_gid, n_histograms=n_hist)
    if not forward_idx or not backward_idx:
        raise ValueError("forward/backward groups have no detectors in this run")

    if worker.is_cancelled():
        raise TaskCancelledError
    # Correct once, form the asymmetry per α. corrected_grouped_counts runs the
    # deadtime → grouping → background stages (the expensive part).
    flags = correction_flags_from_grouping(grouping)
    # The solid curve is always the full configured reduction — compares only ever
    # add a *ghost*, never degrade the solid — so the α residual ⟨A⟩ (below) is
    # always read off the fully-corrected curve.
    use_deadtime = flags.use_deadtime
    use_background = flags.use_background
    facility = request.facility or resolve_facility(grouping=grouping)

    def _reduce(dt: bool, bg: bool):
        return corrected_grouped_counts(
            histograms=request.histograms,
            grouping=grouping,
            forward_idx=forward_idx,
            backward_idx=backward_idx,
            use_deadtime=dt,
            deadtime_mode=flags.deadtime_mode,
            use_background=bg,
            facility=facility,
            reference_resolver=None,
        )

    corrected = _reduce(use_deadtime, use_background)
    n_grouped = min(len(corrected.forward), len(corrected.backward))
    try:
        first_good = max(0, int(grouping.get("first_good_bin", 0)))
    except (TypeError, ValueError):
        first_good = 0
    try:
        last_good = int(grouping.get("last_good_bin", n_grouped - 1))
    except (TypeError, ValueError):
        last_good = n_grouped - 1
    alpha = _as_float(grouping.get("alpha"), 1.0)
    beta = _as_float(grouping.get("beta"), 1.0)
    if not np.isfinite(beta) or beta <= 0.0:
        beta = 1.0

    time, asymmetry, error = _form_asymmetry(
        corrected, grouping, alpha, first_good, last_good, beta=beta
    )

    # Stage-generic before/after compare: the solid curve is the reduction above;
    # the *ghost* removes one stage. "alpha" ghosts α=1 from the SAME corrected
    # counts (one reduction, two curves) and reports the residual baseline;
    # "deadtime"/"background" ghost a SECOND corrected pass with that one stage
    # dropped — CorrectedGroupedCounts keeps only post-background arrays, so the
    # pedestal cannot be added back and the extra pass is unavoidable, but it runs
    # behind the pane's debounce + single-flight. All preview-only (`_reduce` reads
    # the same grouping; nothing here touches the persisted reduction). An
    # un-applied stage has nothing to remove, so it draws no ghost.
    compare = request.compare_stage
    baseline = None
    centre: tuple[float, float] | None = None
    # The second corrected pass, when the focused stage acts on the counts — it
    # is the asymmetry ghost AND the Counts view's ghost, never run twice (D8).
    ghost_counts = None
    if compare == "alpha":
        # Residual baseline (inverse-variance weighted ⟨A⟩) on the full-res curve.
        centre = _weighted_centre(asymmetry, error)
        if abs(alpha - 1.0) > 1e-12:
            # The ghost removes only α; a configured β stays applied.
            _bt, base_asym, _be = _form_asymmetry(
                corrected, grouping, 1.0, first_good, last_good, beta=beta
            )
            _dt, baseline, _de = _decimate_for_preview(time, base_asym, error, _MAX_PREVIEW_POINTS)
    elif compare == "beta" and abs(beta - 1.0) > 1e-12:
        # β ghost: same corrected counts, α as configured, β removed — the exact
        # mirror of the α compare (one reduction, two curves).
        _bt, base_asym, _be = _form_asymmetry(
            corrected, grouping, alpha, first_good, last_good, beta=1.0
        )
        _dt, baseline, _de = _decimate_for_preview(time, base_asym, error, _MAX_PREVIEW_POINTS)
    elif (compare == "deadtime" and use_deadtime) or (compare == "background" and use_background):
        # The ghost is a second full reduction — honour cancellation before it, as
        # the first pass does, so a shutdown mid-flight stops promptly on big runs.
        if worker.is_cancelled():
            raise TaskCancelledError
        # The focused stage is dropped; the other keeps its configured state.
        ghost_counts = _reduce(
            use_deadtime and compare != "deadtime",
            use_background and compare != "background",
        )
        ghost = _form_asymmetry(ghost_counts, grouping, alpha, first_good, last_good, beta=beta)
        _dt, baseline, _de = _decimate_for_preview(time, ghost[1], error, _MAX_PREVIEW_POINTS)

    # Decimate here, off the GUI thread: bounds both the marshalled payload and
    # the GUI-thread draw (which is O(points) and the real hang on large runs —
    # see _MAX_PREVIEW_POINTS).
    time, asymmetry, error = _decimate_for_preview(time, asymmetry, error, _MAX_PREVIEW_POINTS)
    return _PreviewResult(
        generation=request.generation,
        facts=request.facts,
        time=time,
        asymmetry=asymmetry,
        error=error,
        counts=_counts_curves(corrected, ghost_counts, grouping, first_good, last_good),
        alpha=alpha,
        beta=beta,
        baseline=baseline,
        compare_stage=compare,
        centre=centre,
    )


def _counts_curves(
    corrected: Any,
    ghost: Any | None,
    grouping: dict[str, Any],
    first_good: int,
    last_good: int,
) -> CountsCurves:
    """The count-domain view of a reduction: full spectra on one shared stride.

    Unlike the asymmetry, this covers the histogram from bin 0 — the pre-t0
    region the ``range`` background reads, and the bins the good window leaves
    out, are exactly what the view exists to show. Stamps are bin centres
    ``(k + ½)·w`` minus the run's exact t0, the same convention the reduction's
    own axis uses.
    """
    n = int(corrected.forward.size)
    step = _preview_stride(n, _MAX_PREVIEW_POINTS)
    bins = np.arange(0, n, step, dtype=np.float64)
    # Dropping a correction stage changes neither the histogram lengths nor the
    # common t0 they are aligned on, so the ghost shares this stride and axis.
    ghost_forward = None if ghost is None else np.asarray(ghost.forward[::step], dtype=np.float64)
    ghost_backward = None if ghost is None else np.asarray(ghost.backward[::step], dtype=np.float64)
    return CountsCurves(
        time=(bins + 0.5) * corrected.bin_width - corrected.t0_time_us,
        forward=np.asarray(corrected.forward[::step], dtype=np.float64),
        backward=np.asarray(corrected.backward[::step], dtype=np.float64),
        bin_width=float(corrected.bin_width),
        n_bins=n,
        t0_bin=int(corrected.common_t0),
        first_good=first_good,
        last_good=last_good,
        background_level=corrected.background_level,
        background_mode=resolve_background_mode(grouping),
        ghost_forward=ghost_forward,
        ghost_backward=ghost_backward,
    )


def _form_asymmetry(
    corrected: Any,
    grouping: dict[str, Any],
    alpha: float,
    first_good: int,
    last_good: int,
    *,
    beta: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin the corrected counts into a percent asymmetry for one (α, β).

    Mirrors :func:`reduce_grouped_asymmetry`'s final step (counts → binned
    asymmetry, scaled to percent) so the preview matches the reduction exactly.
    """
    time, asym, err = binned_fb_asymmetry(
        corrected.forward,
        corrected.backward,
        grouping=grouping,
        common_t0=corrected.common_t0,
        bin_width_us=corrected.bin_width,
        # The run's exact common t0 (D4) — without it the preview axis sits up to
        # half a bin away from the reduction's on ISIS/MusrRoot runs.
        t0_time_us=corrected.t0_time_us,
        alpha=alpha,
        first_good_bin=first_good,
        last_good_bin=last_good,
        forward_error=corrected.forward_error,
        backward_error=corrected.backward_error,
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
    """The status strip (D1): which run, pair, binning and window this preview is.

    Rich text: an uppercase ``PREVIEW`` marker, the run in bold, then the facts
    that decide the curve, muted. The time range is read back off the drawn
    curve rather than from the form, so it states the window actually reduced.
    """
    details = [
        f"F = {facts.forward_name}",
        f"B = {facts.backward_name}",
        f"bin {facts.bunch}",
    ]
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

"""Fit panel container (``FitPanel``) hosting the single and global tabs.

Split out of ``fit_panel.py`` (Phase 2 mechanical split).

What lives here: ``_CurrentPageTabWidget`` (a ``QTabWidget`` that only
size-hints its current page, avoiding oversized minimums from hidden tabs) and
``FitPanel(QWidget)``, the dockable container that owns a ``QTabWidget`` with
one ``SingleFitTab`` and one ``GlobalFitTab`` page plus the top-level
model/formula selection shared by both. Entry points: construct a
``FitPanel`` from ``MainWindow``; ``set_dataset``/``set_datasets`` feed data
in; ``get_single_state``/``restore_single_state``,
``get_global_state``/``restore_global_state``, and ``get_domain_state``/
``restore_domain_state`` (a domain-keyed dispatch over the pair above)
serialize each tab for project persistence.
"""

import copy
from collections.abc import Callable

import numpy as np
from PySide6.QtCore import QSize, Signal
from PySide6.QtWidgets import (
    QLabel,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.domain_library import coerce_domain
from asymmetry.core.fitting.fit_wizard import (
    FitWizardRecommendation,
    deserialize_fit_wizard_recommendation,
    serialize_fit_wizard_recommendation,
)
from asymmetry.core.fitting.parameters import (
    ParameterSet,
    get_param_info,
    split_parameter_name,
)
from asymmetry.core.fitting.spectral import (
    default_frequency_model,
)
from asymmetry.gui.utils.formatting import format_param_label
from asymmetry.gui.widgets.current_page_sizing import CurrentPageSizingMixin

from .global_tab import GlobalFitTab
from .single_tab import SingleFitTab
from .tab_base import _get_file_value_for_parameter


def _parse_bounds_text(bounds_text: object) -> tuple[str, str]:
    """Parse a ``GlobalFitTab``-style ``"<lo>, <hi>"`` bounds cell into (min, max) text.

    Mirrors ``GlobalFitTab._parse_parameter_configuration``'s bounds grammar
    (``"-inf"``/``"inf"`` literals, comma-separated, any parse failure falls
    back to unbounded on *both* sides) so a batch template row's bounds
    survive the trip into a single-fit-tab parameter row (see
    ``FitPanel._single_state_from_slot_result``).
    """
    text = str(bounds_text) if bounds_text is not None else ""
    try:
        lo_text, hi_text = (part.strip() for part in text.split(",", 1))
        min_val = -float("inf") if lo_text == "-inf" else float(lo_text)
        max_val = float("inf") if hi_text == "inf" else float(hi_text)
    except (ValueError, IndexError):
        min_val, max_val = -float("inf"), float("inf")
    min_text = "-inf" if not np.isfinite(min_val) else str(min_val)
    max_text = "inf" if not np.isfinite(max_val) else str(max_val)
    return min_text, max_text


class _CurrentPageTabWidget(CurrentPageSizingMixin, QTabWidget):
    """A QTabWidget sized by its *current* tab, not the maximum over all tabs.

    A plain QTabWidget reports the largest size hint across every page, so the
    wide Batch tab would impose its width on the dock even while the compact
    Single tab is showing — forcing the inspector scroll area to scroll
    horizontally (a second scrollbar on top of the parameter table's own).
    Sizing to the visible tab (plus the tab bar) lets the dock follow it.
    """

    def _page_extra(self) -> QSize:
        return self.tabBar().sizeHint()


class FitPanel(QWidget):
    """Fit setup and results panel with tabbed interface.

    Contains tabs for single dataset fitting and global (multi-dataset) fitting.
    """

    fit_completed = Signal(object, object, object)  # (FitResult, fitted_curve, component_curves)
    preview_requested = Signal(
        object, object, object
    )  # (preview_result, fitted_curve, component_curves)
    # Keep payload generic to preserve Python dict key/value types end-to-end.
    global_fit_started = Signal()  # forwarded from GlobalFitTab at worker launch
    global_fit_completed = Signal(object, object)  # (results_dict, global_params)
    grouped_fit_completed = Signal(object, object)  # (grouped_datasets, results_dict)
    add_single_fit_to_series_requested = Signal()
    fit_range_edit_committed = Signal(float, float)  # forwarded from SingleFitTab
    # Forwarded from the Batch tab's on-tab seeding selector so the main window's
    # Analysis ▸ Batch seeding menu can mirror it (two-way sync).
    batch_seeding_mode_changed = Signal(str)
    # Emitted whenever the Single/Batch tab selection changes, so the main
    # window can re-evaluate fit-block/enable state (F17) — switching tabs
    # does not itself change what is fittable, but nothing else re-runs that
    # check on a tab switch.
    tab_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._single_state_by_run: dict[int, dict] = {}
        self._active_single_run_number: int | None = None
        # Provenance of the single-fit form's current contents (D2/F6, D5):
        # "own_slot" (a recorded fit — single or batch/global member — for
        # this run/projection: PROTECTED, never auto-overwritten),
        # "carried_from_run" (REFRESHABLE: inherited from a specific source
        # run, tracked in ``_single_fit_carry_source_run`` — the session's
        # most recent *fitted* function whenever one exists), "carried_session"
        # (REFRESHABLE, no fit exists anywhere yet: this run's own previously
        # cached-but-never-fit form — no specific source run to name), or
        # "representation_default" (blanked to the domain default — no
        # badge; nothing has ever been shown/fit in this panel, or this exact
        # slot was positively confirmed unfit with no fit anywhere yet).
        # Driven entirely by ``set_dataset``'s restore precedence; see its
        # docstring.
        self._single_fit_provenance: str | None = None
        self._single_fit_carry_source_run: int | None = None
        # D5 "refresh-unless-fitted": the most recent single-tab state that was
        # actually *fitted* in this session (and the run it was fitted for),
        # updated at every point a fit lands in the single form's world —
        # see ``_note_last_fitted_single_state``. ``set_dataset`` sources a
        # non-protected run's refresh from this instead of whatever that run
        # (or the previously active one) happened to be displaying.
        self._last_fitted_single_state: dict | None = None
        self._last_fitted_single_run: int | None = None
        # Optional mediator that supplies a per-(run, representation, projection)
        # single-fit restore payload, installed by the main window.  It keeps the
        # panel decoupled from the project model: ``set_dataset`` asks it for the
        # form payload to show, falling back to the run-keyed blob when unset or
        # when it returns ``None``.  See ``set_single_fit_restore_provider``.
        self._single_fit_restore_provider: Callable[[MuonDataset | None], dict | None] | None = None
        self._all_datasets: list[MuonDataset] = []  # Track all datasets fed to the panel
        # Active single-fit projection (driven by the main window via
        # ``set_active_projection_label``); part of the binding identity that
        # guards the Single↔Batch tab-switch snapshot below.
        self._active_single_projection: str | None = None
        # Snapshot of the single-fit form taken when the user leaves the Single
        # tab, restored when they return to it for the *same* binding. Without
        # this, switching to Batch and back loses a hand-built (unfit) model:
        # once a run has been batched its per-projection slot exists but is
        # empty, so the restore provider blanks the form to the default model.
        self._single_form_snapshot: dict | None = None
        self._domain = "time"
        self._single_state_by_domain: dict[str, dict] = {}
        self._global_state_by_domain: dict[str, dict] = {}
        self._ui_state_by_domain: dict[str, dict] = {}

        # Create tab widget (sized to the visible tab so the wide Batch tab
        # doesn't force the dock — and a window-level horizontal scrollbar —
        # while the compact Single tab is showing).
        self._tabs = _CurrentPageTabWidget()

        # Single fit tab
        self._single_tab = SingleFitTab()
        self._single_tab.fit_completed.connect(self._on_single_fit_completed)
        self._single_tab.preview_requested.connect(self.preview_requested.emit)
        self._single_tab.send_model_to_batch_requested.connect(self._on_send_model_to_batch)
        self._single_tab.add_to_series_requested.connect(
            self.add_single_fit_to_series_requested.emit
        )
        self._single_tab.fit_range_edit_committed.connect(self.fit_range_edit_committed.emit)
        self._tabs.addTab(self._single_tab, "Single")

        # Batch fit tab (a global fit is the special case with shared parameters)
        self._global_tab = GlobalFitTab(member_kind="runs")
        self._global_tab.global_fit_started.connect(self.global_fit_started.emit)
        self._global_tab.global_fit_completed.connect(self.global_fit_completed.emit)
        self._global_tab.grouped_fit_completed.connect(self.grouped_fit_completed.emit)
        self._global_tab.fit_range_edit_committed.connect(self.fit_range_edit_committed.emit)
        self._global_tab.batch_seeding_mode_changed.connect(self.batch_seeding_mode_changed.emit)
        self._tabs.addTab(self._global_tab, "Batch")

        # Preserve the single-fit form across a Single↔Batch view switch (see #3
        # / _single_form_snapshot). Connected last so both tabs exist.
        self._tabs.currentChanged.connect(self._on_fit_tab_changed)

        # Echo of the projection a single fit is currently bound to (vector
        # multi-subplot view); hidden when fitting the default/non-projection
        # asymmetry. Driven by the main window via set_active_projection_label.
        self._projection_echo = QLabel("")
        self._projection_echo.setContentsMargins(6, 2, 6, 2)
        self._projection_echo.hide()
        layout.addWidget(self._projection_echo)

        layout.addWidget(self._tabs)

    def set_active_projection_label(self, projection: str | None, tint: str | None = None) -> None:
        """Show/hide the 'Fitting: <projection>' echo for the bound projection.

        ``tint`` colours the text to match the projection's subplot frame.
        """
        # Track the projection as part of the tab-switch snapshot's binding
        # identity: a snapshot only restores onto the same (run, projection).
        if projection != self._active_single_projection:
            self._single_form_snapshot = None
        self._active_single_projection = projection
        if hasattr(self, "_single_fit_provenance"):
            self._update_single_fit_badge()
        if not hasattr(self, "_projection_echo"):
            return
        if projection:
            self._projection_echo.setText(f"Fitting: {projection}")
            self._projection_echo.setStyleSheet(f"color: {tint}; font-weight: 500;" if tint else "")
            self._projection_echo.show()
        else:
            self._projection_echo.clear()
            self._projection_echo.setStyleSheet("")
            self._projection_echo.hide()

    def _on_fit_tab_changed(self, index: int) -> None:
        """Preserve the single-fit form across a Single↔Batch view switch.

        Leaving the Single tab snapshots its form; returning restores that
        snapshot when the binding (run + projection + domain) is unchanged.
        This keeps a hand-built but unfit model alive across the round trip —
        without it, once a run has been batched its per-projection slot exists
        but is empty, so the restore provider blanks the form to the default
        model on re-bind. The domain is part of the binding identity: switching
        the plot FB→FFT while Batch is active re-establishes the same run
        number, and without the domain check the stale time-domain snapshot
        would replay into the frequency form on the next visit to Single. The
        mismatched snapshot is kept (not dropped) so it still restores after a
        domain round trip back.
        """
        single_index = self._tabs.indexOf(self._single_tab)
        if index == single_index:
            snapshot = self._single_form_snapshot
            if (
                snapshot is not None
                and snapshot.get("run") == self._active_single_run_number
                and snapshot.get("domain") == self._domain
            ):
                self._single_tab.restore_state(snapshot["state"])
        else:
            self._single_form_snapshot = {
                "run": self._active_single_run_number,
                "domain": self._domain,
                "state": self.get_single_form_state(),
            }
        self.tab_changed.emit(index)

    def set_batch_seeding_mode(self, mode: str) -> None:
        """Forward the batch-series seeding mode to the Batch tab."""
        self._global_tab.set_batch_seeding_mode(mode)

    def domain(self) -> str:
        """Return the current fitting domain."""
        return self._domain

    def set_rrf_frequency_provider(self, provider: Callable[[], float | None]) -> None:
        """Forward the rotating-frame ν₀ provider to the single-fit tab."""
        self._single_tab.set_rrf_frequency_provider(provider)

    def set_domain(self, domain: str) -> None:
        """Switch the fit panel between time- and frequency-domain workflows."""
        normalized = coerce_domain(domain)
        if normalized == self._domain:
            self._single_tab.set_domain(normalized)
            self._global_tab.set_domain(normalized)
            return

        old_domain = self._domain
        self._single_state_by_domain[old_domain] = self.get_single_state()
        self._global_state_by_domain[old_domain] = self.get_global_state()
        self._ui_state_by_domain[old_domain] = self.get_ui_state()

        self._domain = normalized
        self._single_state_by_run = {}
        self._active_single_run_number = None
        self._set_single_fit_provenance(None)
        # A different domain fits a structurally different model (frequency
        # peaks vs. time relaxation) -- the other domain's last-fitted state
        # is not a valid D5 refresh source here, so it does not carry across.
        self._last_fitted_single_state = None
        self._last_fitted_single_run = None
        self._single_tab.set_domain(normalized)
        self._global_tab.set_domain(normalized)

        if normalized in self._single_state_by_domain:
            self.restore_single_state(self._single_state_by_domain[normalized])
        if normalized in self._global_state_by_domain:
            self.restore_global_state(self._global_state_by_domain[normalized])
        if normalized in self._ui_state_by_domain:
            self.restore_ui_state(self._ui_state_by_domain[normalized])

    def clear(self) -> None:
        """Reset all fit-panel domain state."""
        self._single_state_by_domain = {}
        self._global_state_by_domain = {}
        self._ui_state_by_domain = {}
        self._single_state_by_run = {}
        self._active_single_run_number = None
        # Drop the tab-switch snapshot too, so a stale form can't be restored
        # onto the cleared panel when setCurrentIndex(0) below re-enters Single.
        self._single_form_snapshot = None
        self._active_single_projection = None
        self._all_datasets = []
        # A new project has no fitted function yet: clear the D5 refresh
        # source, or a closed project's fit would leak into the next one.
        self._last_fitted_single_state = None
        self._last_fitted_single_run = None
        self._domain = "time"
        self._single_tab.set_domain("time")
        self._global_tab.set_domain("time")
        self._single_tab.set_dataset(None)
        self._global_tab.set_datasets([])
        self._global_tab.set_current_dataset(None)
        self._set_single_fit_provenance(None)
        self._tabs.setCurrentIndex(0)

    def _on_single_fit_completed(self, fit_result, fitted_curve, component_curves) -> None:
        """Forward single-fit completion and cache seeds for global fitting."""
        dataset = self._single_tab._current_dataset
        if dataset is not None:
            run_number = int(dataset.run_number)
            self._global_tab.register_single_fit_seed(
                run_number,
                self._single_tab._composite_model,
                fit_result,
            )
            # Keep most recent tab state per run (parameters, function, and result text).
            self._single_state_by_run[run_number] = self._single_tab.get_state()
            # A fit was just recorded for the run currently shown — the form is
            # no longer carried content, regardless of the projection it landed
            # on (D2/F6: clear the badge the moment a fit exists).
            self._set_single_fit_provenance("own_slot")
            # D5: this run just became the session's newest fitted function —
            # the source every non-protected run's refresh will draw from.
            self._note_last_fitted_single_state(run_number)
        self.fit_completed.emit(fit_result, fitted_curve, component_curves)

    def _note_last_fitted_single_state(self, run_number: int, state: dict | None = None) -> None:
        """Record the session's most recent *fitted* single-tab state (D5).

        Called at every point a fit lands in the single form's world: a fresh
        single fit (above), a batch/global write-back
        (``register_global_fit_results`` — the last successful member wins,
        which is fine as "most recent"), and ``set_dataset`` restoring an
        ``own_slot`` payload (a loaded project's fitted run being selected is,
        from here on, the newest fitted function the user is looking at).

        Defaults to the tab's current live state, the common case where the
        run just fitted (or just restored) is also the active one; batch
        write-back passes each member's own reconstructed state explicitly
        since it is not the active tab.
        """
        source_state = state if state is not None else self._single_tab.get_state()
        self._last_fitted_single_state = copy.deepcopy(source_state)
        self._last_fitted_single_run = run_number

    def _run_number_from_dataset(self, dataset: MuonDataset | None) -> int | None:
        if dataset is None:
            return None
        try:
            return int(dataset.run_number)
        except (TypeError, ValueError):
            return None

    def set_dataset(self, dataset: MuonDataset | None) -> None:
        """Set the current dataset for single fitting tab.

        D5 "refresh-unless-fitted": three branches decide what the form shows.

        (A) PROTECTED — the restore mediator (``_single_fit_restore_provider``)
            hands back a non-empty payload: a genuine recorded fit for this
            (run, representation, projection) slot, restored verbatim
            (provenance ``own_slot``, no badge). This covers both a single fit
            and a batch/global write-back — the mediator reconstructs a
            payload from a batch member's persisted ``FitSlot`` when it has no
            ``ui_state`` of its own (see
            ``MainWindow._single_fit_restore_payload``), so protection here is
            driven purely by "did the mediator hand back content", with no
            separate probe needed. Never auto-overwritten.
        (B) An *empty* mediator payload: the mediator has a real opinion this
            exact slot was checked and confirmed unfit (a genuine projection,
            or the default slot once another projection has been fit).
            REFRESHABLE: once any fit exists in the session, refresh onto it
            (D5 — even a "domain default" is refreshable); otherwise blank to
            the domain default exactly as before (this slot was positively
            checked, so there is nothing to carry from a previous run either).
        (C) No mediator opinion at all (``None`` — a legacy/no-mediator run,
            or the default slot with no other projection info): REFRESHABLE
            via ``_refresh_single_fit_form``, which prefers the session's most
            recently *fitted* state, falling back to today's pre-D5
            behaviour (this run's own cached form, or the previously active
            run's displayed form) only once nothing has been fitted yet.

        A dismissable badge (D2/F6) surfaces (B)/(C) whenever a source run is
        named — carry-forward itself is kept (useful for run series), but the
        panel must say so instead of silently presenting another run's values
        as this run's own.
        """
        previous_run_number = self._active_single_run_number
        if self._active_single_run_number is not None:
            self._single_state_by_run[self._active_single_run_number] = self._single_tab.get_state()

        self._single_tab.set_dataset(dataset)
        self._global_tab.set_current_dataset(dataset)

        run_number = self._run_number_from_dataset(dataset)
        self._active_single_run_number = run_number

        if run_number is None:
            self._set_single_fit_provenance(None)
            # Even without a run number the bound spectrum drives the fit, so a
            # frequency form still needs its field-dependent peak seed refreshed
            # (see below); a restored real fit is impossible on this path.
            self._reseed_frequency_peaks_if_default(is_real_fit=False)
            return

        # The main window's restore mediator is authoritative when it has an
        # opinion: a non-empty payload is this run's own recorded fit — the
        # canonical store for single fits, reconstructed for a batch/global
        # member when necessary. Consulting it first avoids restoring twice.
        payload = (
            self._single_fit_restore_provider(dataset)
            if self._single_fit_restore_provider is not None
            else None
        )
        is_real_fit = isinstance(payload, dict) and bool(payload)
        self._single_tab.set_has_recorded_fit(is_real_fit)
        if is_real_fit:
            self._restore_protected_single_fit_form(run_number, payload)
        elif payload is not None:
            self._refresh_confirmed_unfit_slot(run_number)
        else:
            self._refresh_single_fit_form(run_number, previous_run_number)

    def _restore_protected_single_fit_form(self, run_number: int, payload: dict) -> None:
        """PROTECTED (D5): restore this run's own recorded fit verbatim."""
        self.restore_single_fit_ui(payload)
        self._set_single_fit_provenance("own_slot")
        # A loaded project's fitted run being selected is, from here on, the
        # newest fitted function the user is looking at (D5) — count it as
        # the refresh source for the runs that are not protected.
        self._note_last_fitted_single_state(run_number)

    def _refresh_confirmed_unfit_slot(self, run_number: int) -> None:
        """REFRESHABLE (D5): the mediator confirmed this exact slot is unfit.

        An empty payload means the per-(run, representation, projection) slot
        was actually checked (not merely "no opinion") — refresh onto the
        session's latest fitted function once one exists (D5: even a "domain
        default" is refreshable), otherwise blank to the domain default
        exactly as before: this slot has no meaningful "previous run" to carry
        from, since it was positively confirmed unfit rather than unseen.
        """
        if self._last_fitted_single_state is not None:
            self._carry_forward_single_fit_form(source_state=self._last_fitted_single_state)
            self._set_single_fit_provenance("carried_from_run", self._last_fitted_single_run)
            self._single_state_by_run[run_number] = self._single_tab.get_state()
        else:
            self._reset_single_fit_form()
            self._set_single_fit_provenance("representation_default")

    def _refresh_single_fit_form(self, run_number: int, previous_run_number: int | None) -> None:
        """REFRESHABLE (D5): no mediator opinion at all for this run.

        Sources the refresh from ``_last_fitted_single_state`` when a fit
        exists anywhere in the session — this supersedes even this run's own
        stale cached carry, so navigating away and back doesn't resurrect it.
        Only once nothing has been fitted yet does it fall back to today's
        pre-D5 behaviour verbatim: this run's own previously-cached form when
        one exists (``carried_session`` — a legacy/no-mediator run's session
        cache, not a confirmed fit), else the previously active run's
        currently-displayed form (``carried_from_run``), else the untouched
        construction-time default (no badge) when nothing has even been
        *shown* in this panel yet (see ``_carry_forward_single_fit_form``'s
        docstring for why that case must not be badged as "carried").
        """
        if self._last_fitted_single_state is not None:
            self._carry_forward_single_fit_form(source_state=self._last_fitted_single_state)
            self._set_single_fit_provenance("carried_from_run", self._last_fitted_single_run)
        elif run_number in self._single_state_by_run:
            self._single_tab.restore_state(self._single_state_by_run[run_number])
            self._set_single_fit_provenance("carried_session")
        elif previous_run_number is None:
            self._carry_forward_single_fit_form()
            self._set_single_fit_provenance("representation_default")
        else:
            self._carry_forward_single_fit_form()
            self._set_single_fit_provenance("carried_from_run", previous_run_number)
        self._single_state_by_run[run_number] = self._single_tab.get_state()

    def _reseed_frequency_peaks_if_default(self, *, is_real_fit: bool) -> None:
        """Refresh frequency peak seeds unless a real recorded fit is shown."""
        if not is_real_fit and self._single_tab.domain() == "frequency":
            self._single_tab.reseed_frequency_peaks()

    def _set_single_fit_provenance(self, kind: str | None, source_run: int | None = None) -> None:
        """Record why the single-fit form holds its current contents and update the badge.

        ``kind`` is one of ``own_slot`` / ``representation_default`` (no
        badge), ``carried_from_run`` (badge names ``source_run`` when known —
        the session's most recent fitted function), or ``carried_session``
        (generic badge, no fit exists anywhere yet — this run's own
        previously cached but never-fitted form).
        """
        self._single_fit_provenance = kind
        self._single_fit_carry_source_run = source_run if kind == "carried_from_run" else None
        self._update_single_fit_badge()

    def _update_single_fit_badge(self) -> None:
        kind = self._single_fit_provenance
        if kind not in ("carried_from_run", "carried_session"):
            self._single_tab.clear_carry_forward_badge()
            return
        projection = self._active_single_projection
        suffix = f" ({projection})" if projection else ""
        source_run = self._single_fit_carry_source_run
        origin = f" from run {source_run}" if source_run is not None else ""
        self._single_tab.show_carry_forward_badge(
            f"Model carried{origin}{suffix} — not fitted for this run"
        )

    def _carry_forward_single_fit_form(self, source_state: dict | None = None) -> None:
        """Inherit *source_state*'s model + parameter setup, sans result.

        Reuses the seen-dataset restore path (so the composite model, seeds,
        bounds, fixed/free flags and link groups all transfer faithfully) but
        clears the fitted uncertainties and result label first — an unseen run
        has not been fit, so it must not display another run's result.

        ``source_state`` defaults to the form currently displayed (today's
        last-*displayed* carry-forward, used once no fit exists anywhere in
        the session yet); ``_refresh_single_fit_form`` passes the session's
        last-*fitted* state explicitly (D5) so a refresh sources from what was
        actually fit, not from whatever the previously active run happened to
        be showing.
        """
        state = (
            copy.deepcopy(source_state)
            if source_state is not None
            else self._single_tab.get_state()
        )
        for entry in state.get("parameters", []):
            entry["uncertainty"] = None
            entry["uncertainty_asymmetric"] = None
        # Per-target field reseeding (D5): a parameter like B_L tracks the
        # dataset it was seeded from, so a carried/refreshed form must not
        # keep the source run's field-derived value — reseed it from the
        # *target* dataset now bound to the tab (ported from the retired
        # "Share with Group" action's per-peer reseeding).
        self._reseed_file_value_parameters(state, self._single_tab._current_dataset)
        self._single_tab.restore_state(state)
        if not self._single_tab._composite_model.missing_component_names:
            self._single_tab._result_label.setText("No fit performed yet")
        # A frequency peak (nu0/height/fwhm/bg) tracks the run's field, so it
        # must never be carried verbatim from the previous run: re-derive it
        # from the now-current spectrum. (Same-run session restore and project
        # restore go through the seen-dataset / provider paths, not here, so
        # those keep their saved values.)
        self._reseed_frequency_peaks_if_default(is_real_fit=False)

    def _reseed_file_value_parameters(self, state: dict, dataset: MuonDataset | None) -> None:
        """Re-seed field-dependent parameters (e.g. ``B_L``) from *dataset*'s own value.

        Ported from the retired ``share_single_function_state`` (D5): each
        parameter's value is overwritten with the dataset's own file-derived
        value (resolved via :func:`_get_file_value_for_parameter`) when one is
        available, so a carried/refreshed form reflects the *target* run's
        field rather than whatever run the source state was fitted/carried
        from. Mutates ``state`` in place; a no-op without a dataset or a
        ``parameters`` list.
        """
        if dataset is None:
            return
        parameters = state.get("parameters")
        if not isinstance(parameters, list):
            return
        for param_dict in parameters:
            if not isinstance(param_dict, dict):
                continue
            pname = param_dict.get("name")
            if not isinstance(pname, str):
                continue
            base_name, _index = split_parameter_name(pname)
            file_value = _get_file_value_for_parameter(dataset, base_name)
            if file_value is not None:
                param_dict["value"] = file_value

    def _reset_single_fit_form(self) -> None:
        """Blank the single-fit form to its domain default ("No fit yet")."""
        default_model = (
            default_frequency_model()
            if self._domain == "frequency"
            else CompositeModel(["Exponential", "Constant"], operators=["+"])
        )
        self._single_tab._set_composite_model(default_model)
        self._single_tab._result_label.setText("No fit performed yet")

    def set_single_fit_restore_provider(
        self, provider: Callable[[MuonDataset | None], dict | None] | None
    ) -> None:
        """Install the per-projection single-fit restore mediator (or clear it).

        The main window passes a callable that maps the dataset being bound to
        the persisted single-fit form payload for the active ``(run,
        representation, projection)`` slot — or ``None`` to defer to the
        panel's own run-keyed state (the default / legacy-project path).
        """
        self._single_fit_restore_provider = provider

    def get_single_form_state(self) -> dict:
        """Return the single-fit *form* payload (no per-run/domain wrapping).

        This is exactly what :meth:`restore_single_fit_ui` consumes, so it is the
        payload the main window stores as a slot's ``ui_state``.
        """
        return copy.deepcopy(self._single_tab.get_state())

    def restore_single_fit_ui(self, payload: dict | None) -> None:
        """Restore (or blank) the single-fit form from a slot ``ui_state`` payload.

        A populated dict restores the form verbatim; an empty dict (or ``None``)
        blanks it — an unfit projection must never inherit another projection's
        fit. The run-keyed blob is deliberately *not* touched: it stays the
        per-run store that global seeding reads, while the per-projection slot
        is the source of truth for the single-fit form.
        """
        if isinstance(payload, dict) and payload:
            self._single_tab.restore_state(payload)
            # A real persisted fit is now shown; drop any stale tab-switch
            # snapshot so it can't override this fit on the next return to Single.
            self._single_form_snapshot = None
        else:
            self._reset_single_fit_form()

    def set_datasets(self, datasets: list[MuonDataset]) -> None:
        """Set the datasets for the global fitting tab, tracking all datasets fed in."""
        self._all_datasets = datasets
        self._global_tab.set_datasets(datasets)

    def batch_datasets(self) -> list[MuonDataset]:
        """Return the datasets configured for the batch/integral-scan."""
        return self._global_tab.batch_datasets()

    def set_bound_group(self, group_id: str | None, name: str | None = None) -> None:
        """Bind the Batch tab to a data group ("Fit this group…", D1)."""
        self._global_tab.set_bound_group(group_id, name)

    def clear_bound_group(self) -> None:
        """Clear the Batch tab's group binding (ad-hoc selection, D1)."""
        self._global_tab.clear_bound_group()

    def bound_group_id(self) -> str | None:
        """Return the id of the data group the Batch tab is bound to, or ``None``."""
        return self._global_tab.bound_group_id()

    def set_frequency_missing_spectra_status(
        self, missing_run_numbers: list[int], cached_count: int
    ) -> None:
        """Show frequency-domain global fit status for selected uncached runs."""
        self._global_tab.set_frequency_missing_spectra_status(missing_run_numbers, cached_count)

    def is_grouped_time_domain_mode(self) -> bool:
        """Return whether the global tab is in grouped time-domain mode."""
        return self._global_tab.is_grouped_time_domain_mode()

    def set_fit_blocked(self, blocked: bool, reason: str = "") -> None:
        """Apply fit-action blocking to both single and global tabs."""
        self._single_tab.set_fit_blocked(blocked, reason)
        self._global_tab.set_fit_blocked(blocked, reason)

    def set_fit_range_display(self, x_min: float | None, x_max: float | None) -> None:
        """Forward fit-range display update to both single and global tabs."""
        self._single_tab.set_fit_range_display(x_min, x_max)
        self._global_tab.set_fit_range_display(x_min, x_max)

    def single_fit_formula_string(self) -> str | None:
        """Return the active single-fit formula string, if available."""
        model = getattr(self._single_tab, "_composite_model", None)
        if model is None:
            return None
        try:
            return str(model.formula_string())
        except Exception:
            return None

    def single_fit_model_and_seed(self) -> tuple[CompositeModel, ParameterSet]:
        """Return the active single-fit model and seed (for headless re-fits)."""
        return self._single_tab.model_and_seed()

    def global_fit_formula_string(self) -> str | None:
        """Return the active global-fit formula string, if available."""
        model = getattr(self._global_tab, "_composite_model", None)
        if model is None:
            return None
        try:
            return str(model.formula_string())
        except Exception:
            return None

    def clear_fits_for_runs(self, run_numbers: list[int]) -> int:
        """Clear cached single/global fit state for specific dataset runs."""
        normalized_runs: set[int] = set()
        for run_number in run_numbers:
            try:
                normalized_runs.add(int(run_number))
            except (TypeError, ValueError):
                continue

        if not normalized_runs:
            return 0

        changed_runs: set[int] = set()
        for run_number in normalized_runs:
            if self._single_state_by_run.pop(run_number, None) is not None:
                changed_runs.add(run_number)

        changed_runs |= self._global_tab.remove_single_fit_seeds(normalized_runs)

        active_run = self._active_single_run_number
        if active_run is not None and active_run in normalized_runs:
            self._single_tab._result_label.setText("No fit performed yet")

        # D5: the session's refresh source must not survive its own run's fit
        # being cleared -- otherwise every other unprotected run would keep
        # refreshing onto a fit that, as far as the browser is concerned, no
        # longer exists.
        if (
            self._last_fitted_single_run is not None
            and self._last_fitted_single_run in normalized_runs
        ):
            self._last_fitted_single_state = None
            self._last_fitted_single_run = None

        return len(changed_runs)

    def get_single_state_for_run(self, run_number: int) -> dict | None:
        """Return current single-fit state for one run, if available."""
        try:
            run_key = int(run_number)
        except (TypeError, ValueError):
            return None

        if self._active_single_run_number == run_key:
            state = self._single_tab.get_state()
            self._single_state_by_run[run_key] = state
            return copy.deepcopy(state)

        state = self._single_state_by_run.get(run_key)
        if isinstance(state, dict):
            return copy.deepcopy(state)
        return None

    def get_single_fit_wizard_cache_for_run(
        self,
        run_number: int,
    ) -> tuple[FitWizardRecommendation | None, dict[str, object] | None, str]:
        state = self.get_single_state_for_run(run_number)
        if not isinstance(state, dict):
            return None, None, ""
        wizard_state = state.get("wizard_state")
        if not isinstance(wizard_state, dict):
            return None, None, ""
        recommendation = deserialize_fit_wizard_recommendation(wizard_state.get("recommendation"))
        signature = wizard_state.get("signature")
        log_text = str(wizard_state.get("log_text", ""))
        return (
            recommendation,
            signature if isinstance(signature, dict) else None,
            log_text,
        )

    def persist_single_fit_wizard_cache_for_run(
        self,
        run_number: int,
        recommendation: FitWizardRecommendation,
        *,
        signature: dict[str, object] | None = None,
        log_text: str = "",
    ) -> None:
        try:
            run_key = int(run_number)
        except (TypeError, ValueError):
            return

        active_signature = (
            copy.deepcopy(signature)
            if isinstance(signature, dict)
            else {
                "run_number": run_key,
                "model": None,
            }
        )
        wizard_state = {
            "signature": active_signature,
            "recommendation": serialize_fit_wizard_recommendation(recommendation),
            "log_text": str(log_text),
        }

        if (
            self._active_single_run_number is not None
            and int(self._active_single_run_number) == run_key
        ):
            self._single_tab._cache_wizard_analysis(
                recommendation,
                signature=active_signature,
                log_text=log_text,
            )
            self._single_state_by_run[run_key] = self._single_tab.get_state()
            return

        state = self.get_single_state_for_run(run_key)
        if not isinstance(state, dict):
            recommended = recommendation.recommended_assessment
            if recommended is not None and recommended.fit_result.success:
                state = self._single_state_from_fit_result(
                    recommended.template.model,
                    recommended.fit_result,
                    source="Fit Wizard",
                )
            else:
                state = {
                    "model_name": "Composite",
                    "composite_model": (
                        recommendation.recommended_assessment.template.model.to_dict()
                        if recommendation.recommended_assessment is not None
                        else self._single_tab._composite_model.to_dict()
                    ),
                    "parameters": [],
                    "result_html": "No fit performed yet",
                }
        state["wizard_state"] = wizard_state
        self._single_state_by_run[run_key] = copy.deepcopy(state)

    def _result_html_from_fit(self, fit_result: object, source: str) -> str:
        """Build single-fit result HTML from a completed fit result object."""
        if getattr(fit_result, "success", False) is not True:
            message = str(getattr(fit_result, "message", "Fit failed"))
            return f"<b>{source} failed:</b> {message}"

        reduced = float(getattr(fit_result, "reduced_chi_squared", float("nan")))
        chi2 = float(getattr(fit_result, "chi_squared", float("nan")))
        lines = [
            f"<b>{source}</b>",
            f"<b>χ² = {chi2:.4f}</b>",
            f"<b>χ²ᵣ = {reduced:.4f}</b>",
            "<br><b>Parameters:</b>",
        ]

        uncertainties = getattr(fit_result, "uncertainties", {}) or {}
        for param in getattr(fit_result, "parameters", []):
            name = getattr(param, "name", None)
            if not isinstance(name, str):
                continue
            value = float(getattr(param, "value", 0.0))
            unc = float(uncertainties.get(name, 0.0))
            lines.append(f"  {format_param_label(name)} = {value:.6f} ± {unc:.6f}")
        return "<br>".join(lines)

    def _single_state_from_fit_result(
        self,
        model: CompositeModel,
        fit_result: object,
        source: str,
        roles: dict[str, str] | None = None,
    ) -> dict:
        """Return single-tab state populated from a fitted model result.

        ``roles`` maps parameter names to their batch role (global/local/fixed);
        each is recorded on the param entry so the single tab can annotate how the
        parameter was classified in the batch fit.
        """
        roles = roles or {}
        values_by_name: dict[str, object] = {}
        for param in getattr(fit_result, "parameters", []):
            name = getattr(param, "name", None)
            if isinstance(name, str):
                values_by_name[name] = param

        params: list[dict[str, object]] = []
        for pname in model.param_names:
            param = values_by_name.get(pname)
            if param is None:
                value = float(model.param_defaults.get(pname, 0.0))
                fixed = False
                default_min = get_param_info(pname).default_min
                min_text = str(default_min) if default_min is not None else "-inf"
                max_text = "inf"
            else:
                try:
                    value = float(getattr(param, "value", model.param_defaults.get(pname, 0.0)))
                except (TypeError, ValueError):
                    value = float(model.param_defaults.get(pname, 0.0))
                fixed = bool(getattr(param, "fixed", False))

                min_val = getattr(param, "min", -float("inf"))
                max_val = getattr(param, "max", float("inf"))
                min_text = (
                    "-inf"
                    if min_val is None or not np.isfinite(float(min_val))
                    else str(float(min_val))
                )
                max_text = (
                    "inf"
                    if max_val is None or not np.isfinite(float(max_val))
                    else str(float(max_val))
                )

            params.append(
                {
                    "name": pname,
                    "value": value,
                    "fixed": fixed,
                    "min": min_text,
                    "max": max_text,
                    "role": roles.get(pname),
                }
            )

        return {
            "model_name": "Composite",
            "composite_model": model.to_dict(),
            "parameters": params,
            "result_html": self._result_html_from_fit(fit_result, source),
        }

    def build_single_fit_payload_from_slot(
        self,
        model: dict | None,
        template_parameters: list[dict] | None,
        result: dict | None,
    ) -> dict | None:
        """Reconstruct a single-fit-tab restore payload from a persisted ``FitSlot``.

        Batch/global-member ``FitSlot``s are "pointer slots": they carry a
        genuine fitted ``model``/``result`` but never a ``ui_state`` (only the
        single-fit GUI path writes one). D5 protects any run with a recorded
        result, so ``MainWindow._single_fit_restore_payload`` calls this to
        synthesise a payload rather than treating such a run as unfit — most
        relevantly across a project reload, before this session's
        ``register_global_fit_results`` cache exists. Returns ``None`` when
        there is nothing usable to build from.
        """
        if not isinstance(model, dict) or not isinstance(result, dict):
            return None
        try:
            composite_model = CompositeModel.from_dict(model, allow_missing=True)
        except (ValueError, KeyError, TypeError):
            return None
        templates = [entry for entry in (template_parameters or []) if isinstance(entry, dict)]
        return self._single_state_from_slot_result(composite_model, templates, result)

    def _single_state_from_slot_result(
        self,
        model: CompositeModel,
        template_parameters: list[dict],
        result: dict,
    ) -> dict:
        """Reconstruct single-tab state from a persisted ``FitSlot`` (no ``ui_state``).

        Bounds/fixed classification come from the batch's shared template row
        (``template_parameters`` — every member of a batch/global series
        shares one model configuration, the ``GlobalFitTab.get_state()``
        snapshot taken at fit-completion time); the fitted *value* for each
        parameter comes from this run's own persisted ``result`` (the
        ``MainWindow._fit_result_summary`` shape). Mirrors
        ``_single_state_from_fit_result``'s shape but is built from serialised
        dicts rather than a live ``FitResult`` object.
        """
        template_by_name: dict[str, dict] = {
            str(entry.get("name")): entry
            for entry in template_parameters
            if isinstance(entry, dict)
        }
        fitted_values = result.get("parameters")
        fitted_values = fitted_values if isinstance(fitted_values, dict) else {}

        params: list[dict[str, object]] = []
        for pname in model.param_names:
            template = template_by_name.get(pname, {})
            try:
                raw_value = fitted_values.get(
                    pname, template.get("value", model.param_defaults.get(pname, 0.0))
                )
                value = float(raw_value)
            except (TypeError, ValueError):
                value = float(model.param_defaults.get(pname, 0.0))
            role = str(template.get("type") or "Local").strip().lower()
            if role not in ("global", "local", "fixed", "file"):
                role = None
            min_text, max_text = _parse_bounds_text(template.get("bounds"))
            params.append(
                {
                    "name": pname,
                    "value": value,
                    "fixed": role == "fixed",
                    "min": min_text,
                    "max": max_text,
                    "role": role,
                }
            )

        return {
            "model_name": "Composite",
            "composite_model": model.to_dict(),
            "parameters": params,
            "result_html": self._result_html_from_slot_result(result),
        }

    def _result_html_from_slot_result(self, result: dict) -> str:
        """Build single-fit result HTML from a persisted (serialised) ``FitSlot`` result.

        Mirrors ``_result_html_from_fit`` but reads the plain ``dict`` shape
        ``MainWindow._fit_result_summary`` writes into ``FitSlot.result``
        instead of a live ``FitResult`` object — used when reconstructing a
        batch member's form from its persisted slot (no in-session
        ``FitResult`` exists after a reload).
        """
        source = "Batch fit"
        if not result.get("success", False):
            message = str(result.get("message", "Fit failed"))
            return f"<b>{source} failed:</b> {message}"

        reduced = float(result.get("reduced_chi_squared", float("nan")))
        chi2 = float(result.get("chi_squared", float("nan")))
        lines = [
            f"<b>{source}</b>",
            f"<b>χ² = {chi2:.4f}</b>",
            f"<b>χ²ᵣ = {reduced:.4f}</b>",
            "<br><b>Parameters:</b>",
        ]

        parameters = result.get("parameters")
        parameters = parameters if isinstance(parameters, dict) else {}
        uncertainties = result.get("uncertainties")
        uncertainties = uncertainties if isinstance(uncertainties, dict) else {}
        for name, raw_value in parameters.items():
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            try:
                unc = float(uncertainties.get(name, 0.0) or 0.0)
            except (TypeError, ValueError):
                unc = 0.0
            lines.append(f"  {format_param_label(name)} = {value:.6f} ± {unc:.6f}")
        return "<br>".join(lines)

    def register_global_fit_results(
        self, results_by_run: dict[int, tuple[object, object, object]]
    ) -> None:
        """Persist per-run single-tab state using the latest successful global fit."""
        model = self._global_tab._composite_model
        active_run = self._active_single_run_number
        roles = self._global_tab.param_role_map()

        for run_number, payload in results_by_run.items():
            if not isinstance(payload, tuple) or not payload:
                continue
            fit_result = payload[0]
            if getattr(fit_result, "success", False) is not True:
                continue
            self._global_tab.register_single_fit_seed(run_number, model, fit_result)
            run_state = self._single_state_from_fit_result(
                model, fit_result, source="Batch fit", roles=roles
            )
            self._single_state_by_run[int(run_number)] = run_state
            # D5: a batch write-back counts as a fit landing for this member;
            # the last successful one processed is fine as "most recent".
            self._note_last_fitted_single_state(int(run_number), run_state)

            if active_run is not None and int(run_number) == int(active_run):
                self._single_tab.restore_state(run_state)

    # ── project state helpers ──────────────────────────────────────────

    def get_single_state(self) -> dict:
        """Return serialisable state of the single-fit tab."""
        if self._active_single_run_number is not None:
            self._single_state_by_run[self._active_single_run_number] = self._single_tab.get_state()

        active_state = self._single_tab.get_state()
        states_by_run = {
            str(run_number): dict(state)
            for run_number, state in self._single_state_by_run.items()
            if isinstance(state, dict)
        }
        combined_state = dict(active_state)
        combined_state["states_by_run"] = states_by_run
        combined_state["active_run_number"] = self._active_single_run_number
        return combined_state

    def get_domain_state(self, domain: str) -> dict:
        """Return serialisable fit state for one fitting domain."""
        normalized = coerce_domain(domain)
        if normalized == self._domain:
            return {
                "domain": normalized,
                "single_fit_state": self.get_single_state(),
                "global_fit_state": self.get_global_state(),
                "fit_ui_state": self.get_ui_state(),
            }
        return {
            "domain": normalized,
            "single_fit_state": copy.deepcopy(self._single_state_by_domain.get(normalized, {})),
            "global_fit_state": copy.deepcopy(self._global_state_by_domain.get(normalized, {})),
            "fit_ui_state": copy.deepcopy(self._ui_state_by_domain.get(normalized, {})),
        }

    def restore_domain_state(self, domain: str, state: dict | None) -> None:
        """Restore serialisable fit state for one fitting domain.

        The blob's ``domain`` tag (written by :meth:`get_domain_state`) must
        match the requested domain — a mismatch means a stale/mis-routed payload
        and is refused rather than silently applied to the wrong form (F21c).
        """
        normalized = coerce_domain(domain)
        if not isinstance(state, dict):
            state = {}
        tag = state.get("domain")
        # Compare the raw tag against the canonical domain rather than routing it
        # through coerce_domain: coerce_domain maps *any* unrecognised token to
        # "time", which would let a garbage/typo tag (e.g. "freq") silently pass
        # when restoring the time domain. get_domain_state always writes the
        # canonical token, so a correct blob matches exactly.
        if tag is not None and str(tag).strip().lower() != normalized:
            raise ValueError(
                f"restore_domain_state({normalized!r}) received fit state tagged for domain {tag!r}"
            )
        self._single_state_by_domain[normalized] = copy.deepcopy(state.get("single_fit_state", {}))
        self._global_state_by_domain[normalized] = copy.deepcopy(state.get("global_fit_state", {}))
        self._ui_state_by_domain[normalized] = copy.deepcopy(state.get("fit_ui_state", {}))
        if normalized == self._domain:
            self.restore_single_state(self._single_state_by_domain[normalized])
            self.restore_global_state(self._global_state_by_domain[normalized])
            self.restore_ui_state(self._ui_state_by_domain[normalized])

    def restore_single_state(self, state: dict) -> None:
        """Restore single-fit tab state from a saved dict."""
        states_by_run: dict[int, dict] = {}
        raw_states = state.get("states_by_run") if isinstance(state, dict) else None
        if isinstance(raw_states, dict):
            for run_key, run_state in raw_states.items():
                if not isinstance(run_state, dict):
                    continue
                try:
                    run_number = int(run_key)
                except (TypeError, ValueError):
                    continue
                states_by_run[run_number] = dict(run_state)

        self._single_state_by_run = states_by_run

        active_run = self._active_single_run_number
        if active_run is not None and active_run in self._single_state_by_run:
            self._single_tab.restore_state(self._single_state_by_run[active_run])
            return

        # Backward-compatible legacy payloads (single shared state).
        if isinstance(state, dict):
            self._single_tab.restore_state(state)
            if active_run is not None:
                self._single_state_by_run[active_run] = self._single_tab.get_state()

    def get_global_state(self) -> dict:
        """Return serialisable state of the global-fit tab."""
        return self._global_tab.get_state()

    def get_grouped_state(self) -> dict:
        """Return the grouped-fit classification (physics roles + nuisance block)."""
        return self._global_tab.get_grouped_state()

    def single_fit_range_text(self) -> str | None:
        """Active single-fit range as a provenance string (see SingleFitTab)."""
        return self._single_tab.current_fit_range_text()

    def batch_fit_range_text(self) -> str | None:
        """Active batch/global/grouped fit range as a provenance string."""
        return self._global_tab.current_fit_range_text()

    def send_single_model_to_batch(self) -> bool:
        """Copy the single-fit tab's model and current seeds into the Batch tab.

        Returns ``True`` when a model was sent. The Single ⇄ Batch flow: build a
        model in Single, send it to seed a batch over the selected runs. The
        batch parameter seeds are taken from the single tab's current table
        values (which reflect the latest fit once one has run), so the batch
        starts from the values the user just set rather than model defaults or
        stale preserved state (BUG B8c).
        """
        model = getattr(self._single_tab, "_composite_model", None)
        if model is None:
            return False
        seed_values = self._single_tab.current_seed_values()
        seed_bounds = self._single_tab.current_bounds()
        self._global_tab._set_composite_model(
            model, seed_values=seed_values, seed_bounds=seed_bounds
        )
        return True

    def _on_send_model_to_batch(self) -> None:
        """Handle the Single tab's 'Send Model to Batch' action."""
        if self.send_single_model_to_batch():
            self._tabs.setCurrentWidget(self._global_tab)

    def restore_global_state(self, state: dict) -> None:
        """Restore global-fit tab state from a saved dict."""
        self._global_tab.restore_state(state)

    def get_ui_state(self) -> dict:
        """Return serialisable UI state for the fit panel container."""
        return {"active_tab_index": int(self._tabs.currentIndex())}

    def restore_ui_state(self, state: dict) -> None:
        """Restore serialisable UI state for the fit panel container."""
        index = state.get("active_tab_index")
        if isinstance(index, int) and 0 <= index < self._tabs.count():
            self._tabs.setCurrentIndex(index)

    def shutdown_workers(self) -> None:
        """Cancel running fits on both tabs and wait for their threads."""
        self._single_tab.shutdown_workers()
        self._global_tab.shutdown_workers()

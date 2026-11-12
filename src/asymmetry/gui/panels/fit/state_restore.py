"""Restore canonical fit forms while retaining unavailable historical content."""

from __future__ import annotations

from copy import deepcopy


def restore_single_state(self, state: dict) -> None:
    """Restore single-fit tab state from a saved dict."""
    from asymmetry.core.fitting.composite import CompositeModel
    from asymmetry.gui.panels.fit.tab_base import RESTORED_TAG, TONE_BY_TAG
    from asymmetry.gui.panels.fit.wizard_cache import wizard_cache_entry

    self._analysis_history = deepcopy(state.get("analysis_history"))
    self._cached_wizard_recommendation = None
    self._cached_wizard_signature = None
    self._cached_wizard_log_text = ""

    composite_data = state.get("composite_model")
    if isinstance(composite_data, dict):
        restored = CompositeModel.from_dict(composite_data, allow_missing=True)
        self._set_composite_model(restored, seed_from_record=False)
        if restored.missing_component_names:
            names = ", ".join(restored.missing_component_names)
            self._results_card.set_message(
                f"<b>Missing user function(s):</b> {names}.<br>"
                "The saved model is preserved (missing components plot as "
                "zero) but cannot be fitted until they are registered — "
                "see Setup → User functions…",
                tag="Error",
            )

    # The table applies the saved values/fix/bounds/link/tie onto its rows
    # (the model was just rebuilt by _set_composite_model) and re-establishes
    # auxiliary non-model parameters that have no row.
    params_data = {p["name"]: p for p in state.get("parameters", []) if isinstance(p, dict)}
    self._param_table.restore_parameters(params_data)
    self._refresh_saved_fits_row()

    result_html = state.get("result_html")
    if isinstance(result_html, str) and result_html:
        # Captured reports without a tag remain recorded rather than unfitted.
        tag = state.get("result_tag")
        tag = tag if isinstance(tag, str) and tag else RESTORED_TAG
        self._results_card.set_message(result_html, tag=tag, tone=TONE_BY_TAG.get(tag, "neutral"))

    # Accepts both shapes: the session handle (no work — the recommendation
    # is shared by reference) and a persisted dict from a project file or a
    # fit slot's ``ui_state`` (deserialised once, here).
    entry = wizard_cache_entry(state.get("wizard_state"))
    if entry is not None:
        self._cached_wizard_recommendation = entry.recommendation
        self._cached_wizard_signature = entry.signature_copy()
        self._cached_wizard_log_text = entry.log_text

    from asymmetry.core.fitting.analysis_history import analysis_block_reason
    from asymmetry.gui.panels.fit.tab_base import FitParameterTable

    history = self._analysis_history
    table = self._param_table
    if history is not None and history.get("constraint_table") == "group_model_parameters":
        table = self._group_model_table
    rows = (
        table.parameters_state()
        if isinstance(table, FitParameterTable)
        else self._parameter_rows_state()
    )
    reason = analysis_block_reason(history, rows)
    if reason is not None:
        report = str(state.get("result_html") or "")
        content = report if reason in report else f"{reason}<br>{report}"
        self._results_card.set_message(content, tag="Stale", tone="warn")


def restore_global_state(self, state: dict) -> None:
    """Restore global-fit tab state from a saved dict."""
    from asymmetry.core.fitting.composite import CompositeModel
    from asymmetry.gui.panels.fit.global_tab import _COL_MAX, _COL_MIN
    from asymmetry.gui.panels.fit.tab_base import (
        RESTORED_TAG,
        TONE_BY_TAG,
        FitParameterTable,
        _configure_fraction_rows_in_table,
    )
    from asymmetry.gui.panels.fit.wizard_cache import global_wizard_cache_entry

    self._analysis_history = deepcopy(state.get("analysis_history"))
    self._wizard_cache_by_run_set = {}
    self._set_active_wizard_cache(None, signature=None, log_text="")

    # Parameter rows use the canonical fraction and product names.

    composite_data = state.get("composite_model")
    if isinstance(composite_data, dict):
        restored = CompositeModel.from_dict(composite_data, allow_missing=True)
        self._set_composite_model(restored, seed_from_record=False)
        if restored.missing_component_names:
            names = ", ".join(restored.missing_component_names)
            self._results_card.set_message(
                f"Missing user function(s): {names}. The saved model is "
                "preserved (missing components plot as zero) but cannot be "
                "fitted until they are registered — see Setup → User functions…",
                tag="Error",
                tone="error",
            )

    self._apply_parameter_rows(state.get("parameters", []))

    self._restore_group_param_table_state(state.get("group_parameters"))
    self._restore_table_state(self._group_model_table, state.get("group_model_parameters"))
    if isinstance(self._group_model_table, FitParameterTable):
        _configure_fraction_rows_in_table(
            self._group_model_table,
            self._grouped_fit_model(),
            min_column=FitParameterTable.COL_MIN,
            max_column=FitParameterTable.COL_MAX,
        )
    else:
        _configure_fraction_rows_in_table(
            self._group_model_table,
            self._grouped_fit_model(),
            min_column=_COL_MIN,
            max_column=_COL_MAX,
            type_column=2,
        )
    self._synchronize_grouped_model_fraction_rows()

    result_html = state.get("result_html")
    if isinstance(result_html, str) and result_html:
        # Captured reports without a tag remain recorded rather than unfitted.
        tag = state.get("result_tag")
        tag = tag if isinstance(tag, str) and tag else RESTORED_TAG
        self._results_card.set_message(result_html, tag=tag, tone=TONE_BY_TAG.get(tag, "neutral"))

    wizard_state_by_run_set = state.get("wizard_state_by_run_set")
    if isinstance(wizard_state_by_run_set, list):
        self._restore_wizard_cache_store(wizard_state_by_run_set)

    # Both shapes: the session handle (shared by reference) and a persisted
    # dict from a project file / grouped fit slot.
    active_entry = global_wizard_cache_entry(state.get("wizard_state"))
    if active_entry is not None:
        self._cache_wizard_analysis(
            active_entry.recommendation,
            signature=active_entry.signature_copy(),
            log_text=active_entry.log_text,
        )
    self._sync_active_wizard_cache_from_selection()
    self._update_mode_ui(preserve_result=True)
    # The rows are back, but which *series* they describe is the host's to
    # resolve (it holds the project model): the tab is a draft until
    # ``open_series`` lands, and ``saved_open_series_id`` says what to open.
    self._saved_open_series_id = state.get("open_series_id")
    self._become_draft()

    from asymmetry.core.fitting.analysis_history import analysis_block_reason
    from asymmetry.gui.panels.fit.tab_base import FitParameterTable

    history = self._analysis_history
    table = self._param_table
    if history is not None and history.get("constraint_table") == "group_model_parameters":
        table = self._group_model_table
    rows = (
        table.parameters_state()
        if isinstance(table, FitParameterTable)
        else self._parameter_rows_state()
    )
    reason = analysis_block_reason(history, rows)
    if reason is not None:
        report = str(state.get("result_html") or "")
        content = report if reason in report else f"{reason}<br>{report}"
        self._results_card.set_message(content, tag="Stale", tone="warn")

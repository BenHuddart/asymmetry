"""Dialog for cross-group parameter model fitting.

Implemented as an extension of the base ModelFitDialog to keep the
cross-group workflow aligned with the standard model-parameter fit UI.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidgetItem,
    QTextEdit,
    QWidget,
)

from asymmetry.core.fitting.cross_group_roles import (
    CrossGroupRoleRecommendation,
    suggest_cross_group_roles,
)
from asymmetry.core.fitting.parameter_models import (
    CrossGroupFitResult,
    ErrorMode,
    ParameterCompositeModel,
    ParameterGroupData,
    global_fit_parameter_model,
    parse_fit_windows,
    validate_fit_windows,
    windows_mask,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.gui.panels.model_fit_dialog import (
    ModelFitDialog,
    _should_reset_param_on_model_change,
    _show_info,
    _show_warning,
)
from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.widgets import error_html, info_html, success_html
from asymmetry.gui.widgets.trend_preview import PreviewSeries


@dataclass
class CrossGroupDialogOutput:
    model: ParameterCompositeModel
    fit_result: CrossGroupFitResult
    groups: list[ParameterGroupData]
    x_key: str
    fit_x_min: float
    fit_x_max: float
    config: dict[str, object]


class CrossGroupFitDialog(ModelFitDialog):
    """Cross-group fit dialog extending the base model-fit dialog."""

    # global_fit_parameter_model now honours error modes, fit windows, and
    # effective-variance x-uncertainty, so all the inherited controls are shown
    # and wired through _run_fit / config.
    _supports_error_modes = True
    _supports_windows = True
    _supports_x_errors = True

    def __init__(
        self,
        *,
        parameter_name: str,
        x_key: str,
        groups: list[ParameterGroupData],
        existing_config: dict[str, object] | None = None,
        parent=None,
        x_label: str | None = None,
    ) -> None:
        self._parameter_name = parameter_name
        self._x_key = x_key
        self._groups = groups
        self._result: CrossGroupFitResult | None = None
        self._fitted_groups: list[ParameterGroupData] = list(groups)
        self._range_roles: list[dict[str, str]] = []
        self._range_results: dict[int, CrossGroupFitResult] = {}

        all_x = (
            np.concatenate([np.asarray(group.x, dtype=float) for group in groups])
            if groups
            else np.array([], dtype=float)
        )
        all_y = (
            np.concatenate([np.asarray(group.y, dtype=float) for group in groups])
            if groups
            else np.array([], dtype=float)
        )
        all_e = (
            np.concatenate([np.asarray(group.yerr, dtype=float) for group in groups])
            if groups
            else np.array([], dtype=float)
        )
        # Concatenated per-point x-uncertainty (only when every group carries it,
        # i.e. the abscissa is a fitted parameter) so the inherited
        # effective-variance toggle enables; the fit itself reads σ_x per group.
        all_xe = (
            np.concatenate([np.asarray(group.xerr, dtype=float) for group in groups])
            if groups and all(getattr(group, "xerr", None) is not None for group in groups)
            else None
        )

        super().__init__(
            parameter_name=parameter_name,
            x_key=x_key,
            x_values=all_x,
            y_values=all_y,
            y_errors=all_e,
            existing_fit=None,
            parent=parent,
            x_errors=all_xe,
            x_label=x_label,
        )
        self.setWindowTitle(f"Cross-group fit: {parameter_name}")

        banner_text = f"Cross-group mode | Parameter: <b>{parameter_name}</b> | X: <b>{self._x_display}</b> | Groups: <b>{len(groups)}</b>"
        # Tolerant read: the field was renamed source_group_name → trend_group_name
        # (D8); accept the legacy key from any in-flight config built pre-rename.
        source_name = (
            (existing_config.get("trend_group_name") or existing_config.get("source_group_name"))
            if isinstance(existing_config, dict)
            else None
        )
        source_chi2 = (
            existing_config.get("source_reduced_chi_squared")
            if isinstance(existing_config, dict)
            else None
        )
        if isinstance(source_name, str) and source_name.strip():
            if isinstance(source_chi2, (int, float)) and np.isfinite(float(source_chi2)):
                banner_text += (
                    f" | Inherited from: <b>{source_name}</b> (chi2_r={float(source_chi2):.4g})"
                )
            else:
                banner_text += f" | Inherited from: <b>{source_name}</b>"

        banner = QLabel(banner_text)
        banner.setStyleSheet(f"color: {tokens.ACCENT};")
        self._header_slot.addWidget(banner)

        self._param_table.setHorizontalHeaderLabels(
            ["Name", "Value", "Min", "Max", "Type", "Error"]
        )

        # Cross-group mode currently supports a single shared model-range.
        while len(self._fit.ranges) > 1:
            del self._fit.ranges[-1]
        self._range_roles = [{} for _ in self._fit.ranges]

        # Role-suggestion state (Phase 4). A busy suggestion sets a cancel flag
        # polled by the engine's cancel_callback; the last recommendation is kept
        # for the rationale panel. Reuses the base dialog's TaskRunner
        # (self._tasks), shut down in the inherited closeEvent.
        self._suggest_in_progress = False
        self._suggest_cancel_requested = False
        self._suggest_recommendation: CrossGroupRoleRecommendation | None = None

        self._apply_existing_config(existing_config)
        self._post_rebuild_ranges_ui()
        self._select_range(0)

        # Cross-group mode pins exactly one shared range (see above), so "Add
        # Range" has nothing meaningful to do — hide it rather than leave an
        # affordance whose only behaviour is an info popup. The base
        # ``_add_range``/``_remove_range`` stubs below remain as a defensive
        # no-op in case anything else ever triggers them.
        self._add_range_btn.setVisible(False)

        self._build_suggest_roles_ui()

    def _post_rebuild_ranges_ui(self) -> None:
        # Cross-group mode uses a single pinned range, which now renders as one
        # RangeCard (Step 1 of the range-cards redesign). The old per-range
        # "active" checkbox this hook used to hide no longer exists — cards carry
        # no such control — so there is nothing to hide here. Kept as a no-op
        # override so the base rebuild flow stays unchanged; the cross-group
        # card degradation is Step 2.3.
        pass

    def _preview_series(self) -> list[PreviewSeries]:
        """One data trace per group (colour-cycled by draw order in the canvas).

        The base dialog draws a single series from the concatenated x/y; the
        cross-group preview instead shows the N per-group traces plus the one
        candidate curve sampled from the shared range's current params. The
        primary series (index 0) — whose in-mask the active range greys — is the
        first group, matching the base's "series[0].x" mask convention.
        """
        series: list[PreviewSeries] = []
        for group in self._groups:
            xerr = getattr(group, "xerr", None)
            series.append(
                PreviewSeries(
                    label=group.group_name,
                    x=np.asarray(group.x, dtype=float),
                    y=np.asarray(group.y, dtype=float),
                    yerr=np.asarray(group.yerr, dtype=float),
                    xerr=None if xerr is None else np.asarray(xerr, dtype=float),
                )
            )
        return series

    def _on_model_edited(self, idx: int) -> None:
        # The base class's per-range ``fit_range.result`` was already cleared
        # by the caller (``ModelFitDialog._edit_model``), but cross-group mode
        # additionally caches the last run's result/config on the dialog
        # itself (``self._result`` / ``self._last_config``, spanning the
        # single shared range) for ``_on_use_fit``/``output()``. Editing the
        # model — add/remove/edit a component — changes ``param_names``, so a
        # result fitted against the OLD model must not survive: drop both
        # caches so ``_on_use_fit`` reports "no fit result" until the fit is
        # re-run against the new model.
        self._result = None
        self._last_config = None
        self._range_results.pop(idx, None)

    # ── Suggest roles (Phase 4) ──────────────────────────────────────────────

    def _build_suggest_roles_ui(self) -> None:
        """Insert the "Suggest roles…" control row and rationale panel.

        Placed just above the dialog's OK/Cancel button box (the base
        dialog's ``_footer_slot``). The button runs the AICc/AIC/BIC role
        search off-thread; the criterion combo picks the statistic; the
        rationale panel (hidden until first run) shows the per-parameter
        recommendation and the candidate ranking.
        """
        controls = QHBoxLayout()
        self._suggest_btn = QPushButton("Suggest roles…")
        self._suggest_btn.setToolTip(
            "Recommend Global vs Local for each parameter by comparing candidate "
            "fits with an information criterion (AICc/AIC/BIC). Fixed rows are left "
            "as you set them; you can still edit roles before fitting."
        )
        self._suggest_btn.clicked.connect(self._on_suggest_roles_clicked)
        controls.addWidget(self._suggest_btn)

        self._suggest_cancel_btn = QPushButton("Cancel")
        self._suggest_cancel_btn.setVisible(False)
        self._suggest_cancel_btn.clicked.connect(self._on_suggest_cancel_clicked)
        controls.addWidget(self._suggest_cancel_btn)

        controls.addWidget(QLabel("Criterion:"))
        self._criterion_combo = QComboBox()
        for label, key in (("AICc", "aicc"), ("AIC", "aic"), ("BIC", "bic")):
            self._criterion_combo.addItem(label, userData=key)
        controls.addWidget(self._criterion_combo)

        self._suggest_status = QLabel("")
        self._suggest_status.setStyleSheet(f"color: {tokens.ACCENT};")
        controls.addWidget(self._suggest_status, 1)
        controls.addStretch()

        # A container so the whole row can be inserted at a fixed position.
        controls_container = QWidget()
        controls_container.setLayout(controls)

        self._rationale_panel = QTextEdit()
        self._rationale_panel.setReadOnly(True)
        self._rationale_panel.setVisible(False)
        self._rationale_panel.setMaximumHeight(180)

        self._footer_slot.addWidget(controls_container)
        self._footer_slot.addWidget(self._rationale_panel)

    def _selected_criterion(self) -> str:
        data = self._criterion_combo.currentData() if hasattr(self, "_criterion_combo") else None
        return str(data) if data else "aicc"

    def _on_suggest_cancel_clicked(self) -> None:
        self._suggest_cancel_requested = True
        self._suggest_status.setText("Cancelling…")

    def _set_suggest_busy(self, busy: bool) -> None:
        self._suggest_in_progress = busy
        self._suggest_btn.setEnabled(not busy)
        self._criterion_combo.setEnabled(not busy)
        self._suggest_cancel_btn.setVisible(busy)

    def _on_suggest_roles_clicked(self) -> None:
        """Run the role suggestion off-thread over the current fitting setup."""
        if self._suggest_in_progress or self._fit_in_progress:
            return
        if not self._fit.ranges:
            return

        self._commit_param_table(notify_adjustments=False)
        fit_range = self._fit.ranges[0]
        idx = self._active_range_idx if self._active_range_idx is not None else 0
        roles = self._range_roles[idx] if idx < len(self._range_roles) else {}

        fixed_params: dict[str, float] = {}
        initial_params: dict[str, float] = {}
        parameter_bounds: dict[str, tuple[float, float]] = {}
        for p in fit_range.parameters:
            initial_params[p.name] = float(p.value)
            parameter_bounds[p.name] = (float(p.min), float(p.max))
            default_role = "Fixed" if p.fixed else "Global"
            if roles.get(p.name, default_role) == "Fixed":
                fixed_params[p.name] = float(p.value)

        # Snapshot everything the worker needs as plain data (no widget access
        # off-thread). The engine masks to windows/range itself, so pass the full
        # groups plus the shared range/windows.
        model_snapshot = ParameterCompositeModel(
            component_names=list(fit_range.model.component_names),
            operators=list(fit_range.model.operators),
        )
        groups_snapshot = [
            ParameterGroupData(
                group_id=g.group_id,
                group_name=g.group_name,
                x=np.asarray(g.x, dtype=float).copy(),
                y=np.asarray(g.y, dtype=float).copy(),
                yerr=np.asarray(g.yerr, dtype=float).copy(),
                group_variable_value=float(g.group_variable_value),
                xerr=(
                    None
                    if getattr(g, "xerr", None) is None
                    else np.asarray(g.xerr, dtype=float).copy()
                ),
            )
            for g in self._groups
        ]
        windows = list(fit_range.windows) if fit_range.windows else None
        error_mode = self._error_mode()
        error_value = self._error_value()
        criterion = self._selected_criterion()

        xerr_map: dict[str, object] | None = None
        if self._use_x_errors():
            xerr_map = {
                g.group_id: np.asarray(g.xerr, dtype=float)
                for g in groups_snapshot
                if getattr(g, "xerr", None) is not None
            } or None

        self._suggest_cancel_requested = False
        self._set_suggest_busy(True)
        self._suggest_status.setText("Suggesting roles… (this may take a moment)")

        def _cancel_callback() -> bool:
            return self._suggest_cancel_requested

        def _task(_worker):
            return suggest_cross_group_roles(
                groups_snapshot,
                model_snapshot,
                initial_params=initial_params,
                parameter_bounds=parameter_bounds,
                fixed_params=fixed_params,
                error_mode=error_mode,
                error_value=error_value,
                windows=windows,
                xerr=xerr_map,
                criterion=criterion,
                max_fits=40,
                cancel_callback=_cancel_callback,
            )

        self._tasks.start(
            _task,
            on_finished=self._on_suggest_roles_done,
            on_error=self._on_suggest_roles_error,
        )

    def _on_suggest_roles_done(self, recommendation: object) -> None:
        self._set_suggest_busy(False)
        if not isinstance(recommendation, CrossGroupRoleRecommendation):
            self._suggest_status.setText("")
            return
        self._suggest_recommendation = recommendation
        if self._suggest_cancel_requested:
            self._suggest_status.setText("Suggestion cancelled")
            self._render_rationale(recommendation, cancelled=True)
            return
        self._suggest_status.setText("")
        self._apply_recommended_roles(recommendation)
        self._render_rationale(recommendation, cancelled=False)

    def _on_suggest_roles_error(self, message: str) -> None:
        self._set_suggest_busy(False)
        self._suggest_status.setText("Role suggestion failed")
        self._rationale_panel.setVisible(True)
        self._rationale_panel.setPlainText(f"Role suggestion failed:\n\n{message}")

    def _apply_recommended_roles(self, recommendation: CrossGroupRoleRecommendation) -> None:
        """Set the role combo of each recommended parameter; leave Fixed rows."""
        recommended = recommendation.recommended
        if recommended is None:
            return
        recommended_local = set(recommended.local_params)
        for row in range(self._param_table.rowCount()):
            name_item = self._param_table.item(row, 0)
            combo = self._param_table.cellWidget(row, 4)
            if name_item is None or not isinstance(combo, QComboBox):
                continue
            pname_data = name_item.data(Qt.ItemDataRole.UserRole)
            pname = (
                str(pname_data)
                if isinstance(pname_data, str) and pname_data
                else name_item.text().strip()
            )
            # A user-pinned Fixed row is respected (the engine never flips it).
            if combo.currentText() == "Fixed":
                continue
            combo.setCurrentText("Local" if pname in recommended_local else "Global")
        # Persist the applied roles back into the range-role map.
        self._commit_param_table(notify_adjustments=False)

    def _render_rationale(
        self, recommendation: CrossGroupRoleRecommendation, *, cancelled: bool
    ) -> None:
        """Populate the rationale panel with the per-parameter + candidate view."""
        crit = recommendation.criterion.upper()
        lines: list[str] = []
        if cancelled:
            lines.append("Suggestion cancelled — roles left unchanged.")
            lines.append("")
        if recommendation.message:
            lines.append(recommendation.message)
            lines.append("")

        if recommendation.parameters:
            lines.append("Per-parameter recommendation:")
            for rec in recommendation.parameters:
                delta = f"{rec.score_delta:+.2f}" if np.isfinite(rec.score_delta) else "n/a"
                lines.append(
                    f"  • {rec.name}: {rec.recommended_role.capitalize()} "
                    f"(Δ{crit} {delta}) — {rec.rationale}"
                )
            lines.append("")

        if recommendation.candidates:
            best = None
            for cand in recommendation.candidates:
                if cand.success:
                    best = cand.criterion_value(recommendation.criterion)
                    break
            lines.append(f"Top candidates ({crit}):")
            for cand in recommendation.candidates[:5]:
                local_label = ", ".join(cand.local_params) or "none"
                if not cand.success:
                    lines.append(f"  ✗ local = [{local_label}] — did not converge")
                    continue
                value = cand.criterion_value(recommendation.criterion)
                if best is not None and np.isfinite(value) and np.isfinite(best):
                    delta = f"  (Δ {value - best:+.2f})"
                else:
                    delta = ""
                lines.append(f"  local = [{local_label}]: {crit} {value:.2f}{delta}")

        self._rationale_panel.setPlainText("\n".join(lines))
        self._rationale_panel.setVisible(True)

    def _collect_config(self) -> dict[str, object]:
        self._commit_param_table()
        if not self._fit.ranges:
            return {
                "model": {"component_names": ["Linear"], "operators": []},
                "fit_x_min": None,
                "fit_x_max": None,
                "parameter_rows": [],
                "error_mode": self._error_mode().value,
                "error_value": self._error_value(),
                "windows": None,
                "use_x_errors": self._use_x_errors(),
            }

        fit_range = self._fit.ranges[0]
        roles = self._range_roles[0] if self._range_roles else {}
        rows: list[dict[str, object]] = []
        for p in fit_range.parameters:
            default_role = "Fixed" if p.fixed else "Global"
            rows.append(
                {
                    "name": p.name,
                    "initial": float(p.value),
                    "min": float(p.min),
                    "max": float(p.max),
                    "type": roles.get(p.name, default_role),
                }
            )

        return {
            "model": fit_range.model.to_dict(),
            "fit_x_min": float(fit_range.x_min) if fit_range.x_min is not None else None,
            "fit_x_max": float(fit_range.x_max) if fit_range.x_max is not None else None,
            "parameter_rows": rows,
            "error_mode": self._error_mode().value,
            "error_value": self._error_value(),
            "windows": (
                [[float(lo), float(hi)] for lo, hi in fit_range.windows]
                if fit_range.windows
                else None
            ),
            "use_x_errors": self._use_x_errors(),
        }

    @property
    def _model(self) -> ParameterCompositeModel:
        if self._fit.ranges:
            return self._fit.ranges[0].model
        return ParameterCompositeModel(["Linear"], [])

    @_model.setter
    def _model(self, model: ParameterCompositeModel) -> None:
        if not self._fit.ranges:
            return
        self._fit.ranges[0].model = model

    def _rebuild_param_table(self) -> None:
        """Compatibility helper: rebuild table from the current first-range model."""
        if not self._fit.ranges:
            return
        fit_range = self._fit.ranges[0]
        model = fit_range.model
        new_params = ParameterSet()
        for pname in model.param_names:
            if pname in fit_range.parameters and not _should_reset_param_on_model_change(
                model, pname
            ):
                old = fit_range.parameters[pname]
                new_params.add(
                    Parameter(
                        name=pname, value=old.value, min=old.min, max=old.max, fixed=old.fixed
                    )
                )
            else:
                new_params.add(
                    Parameter(
                        name=pname,
                        value=float(model.param_defaults[pname]),
                        fixed=(pname == "shape_factor_a"),
                    )
                )
        fit_range.parameters = new_params
        self._select_range(0)

    def _apply_existing_config(self, config: dict[str, object] | None) -> None:
        if not isinstance(config, dict):
            return

        if not self._fit.ranges:
            return

        fit_range = self._fit.ranges[0]

        model_state = config.get("model")
        if isinstance(model_state, dict):
            try:
                model = ParameterCompositeModel.from_dict(model_state)
            except Exception:
                model = None
            if model is not None:
                fit_range.model = model
                new_params = ParameterSet()
                for pname in model.param_names:
                    if pname in fit_range.parameters and not _should_reset_param_on_model_change(
                        model, pname
                    ):
                        old = fit_range.parameters[pname]
                        new_params.add(
                            Parameter(
                                name=pname,
                                value=old.value,
                                min=old.min,
                                max=old.max,
                                fixed=old.fixed,
                            )
                        )
                    else:
                        new_params.add(
                            Parameter(
                                name=pname,
                                value=float(model.param_defaults[pname]),
                                fixed=(pname == "shape_factor_a"),
                            )
                        )
                fit_range.parameters = new_params

        fit_x_min = config.get("fit_x_min")
        fit_x_max = config.get("fit_x_max")
        if isinstance(fit_x_min, (int, float)):
            fit_range.x_min = float(fit_x_min)
        if isinstance(fit_x_max, (int, float)):
            fit_range.x_max = float(fit_x_max)

        fit_range.windows = parse_fit_windows(config.get("windows"))

        # Restore the error-mode selector + value (legacy config → Column).
        if self._error_mode_combo is not None:
            mode_value = str(config.get("error_mode", ErrorMode.COLUMN.value))
            mode_idx = self._error_mode_combo.findData(mode_value)
            if mode_idx >= 0:
                self._error_mode_combo.setCurrentIndex(mode_idx)
        if self._error_value_spin is not None:
            err_value = config.get("error_value")
            if isinstance(err_value, (int, float)) and float(err_value) > 0:
                self._error_value_spin.setValue(float(err_value))

        # Restore the effective-variance toggle (legacy config → off). The
        # error-mode restore above has already set the checkbox's enabled state,
        # so a box checked under None/Scatter stays inert via _use_x_errors().
        if self._x_error_check is not None:
            self._x_error_check.setChecked(bool(config.get("use_x_errors", False)))

        rows_state = config.get("parameter_rows")
        if not isinstance(rows_state, list):
            return
        row_map = {
            str(entry.get("name", "")): entry for entry in rows_state if isinstance(entry, dict)
        }
        roles: dict[str, str] = {}
        for p in fit_range.parameters:
            pname = p.name
            entry = row_map.get(pname)
            if not isinstance(entry, dict):
                continue
            if isinstance(entry.get("initial"), (int, float)):
                p.value = float(entry.get("initial"))
            if isinstance(entry.get("min"), (int, float)):
                p.min = float(entry.get("min"))
            if isinstance(entry.get("max"), (int, float)):
                p.max = float(entry.get("max"))
            if isinstance(entry.get("type"), str):
                roles[pname] = str(entry.get("type"))
        if not self._range_roles:
            self._range_roles = [{} for _ in self._fit.ranges]
        self._range_roles[0] = roles

    def _connect_plot_range_signals(self) -> None:
        # Cross-group uses one pinned range: dragging out a new range or selecting a
        # different one is meaningless, so the plot's add/select gestures are left
        # unconnected (the canvas still emits them; nothing consumes them). Edge-drag
        # and right-drag-exclude on the single range stay wired via the base.
        return

    def _add_range(self) -> None:
        _show_info(
            self,
            "Cross-group range",
            "Cross-group fitting currently uses one shared fitting range.",
        )

    def _remove_range(self, idx: int) -> None:
        _show_info(
            self,
            "Cross-group range",
            "Cross-group fitting currently uses one shared fitting range.",
        )

    # ── template-method hook overrides (see ModelFitDialog) ───────────────────
    #
    # The shared table/status flow lives in ``ModelFitDialog._select_range`` /
    # ``_commit_param_table``. Cross-group mode differs only in: the per-row
    # control is a Global/Local/Fixed combo (not a Fixed checkbox); results are
    # cached in ``self._range_results`` (not ``fit_range.result``); the error
    # cell summarises per-group uncertainties; and the formula/hint/status text
    # is cross-group specific. Each of those is one small override below.

    def _range_card_title(self, idx: int) -> str:
        # Cross-group mode pins exactly one always-active range, so a "Range 1"
        # label is redundant — the card's formula line already carries the
        # range's identity. Use the trended parameter name instead of an empty
        # title so the card still reads cleanly at a glance.
        return self._parameter_name

    def _range_hint_text(self, idx: int) -> str:
        return f"Editing parameters for Range {idx + 1} (Cross-group mode: Global/Local/Fixed)."

    def _result_for_range(self, idx: int) -> object | None:
        return self._range_results.get(idx)

    def _chi2_status_text(self, result: object | None) -> str:
        if result is None:
            return info_html("Fitting not yet run for selected range")
        if result.success:
            return success_html(
                "Cross-group fit successful",
                detail=(
                    f"chi2 = {result.chi_squared:.6g}, "
                    f"reduced chi2 = {result.reduced_chi_squared:.6g}"
                ),
            )
        return error_html(f"Cross-group fit failed: {result.message or 'No convergence'}")

    def _quality_status_text(self, fit_range, result: object | None) -> str:
        # CrossGroupFitResult does not carry the per-range χ²/dof shape the
        # single-fit quality verdict needs, so cross-group mode shows none.
        return ""

    def _range_status_chip_html(self, fit_range, result: object) -> str:
        # Same reason as _quality_status_text: the single-fit dof (free-parameter
        # count) does not describe a cross-group fit whose local params multiply
        # per group, so the card shows no verdict chip either (only success/fail
        # state via the card's status).
        return ""

    def _current_range_roles(self) -> dict[str, str]:
        """Roles dict for the active range, growing ``_range_roles`` as needed."""
        while len(self._range_roles) < len(self._fit.ranges):
            self._range_roles.append({})
        idx = self._active_range_idx if self._active_range_idx is not None else 0
        return self._range_roles[idx] if idx < len(self._range_roles) else {}

    def _make_param_row_control(self, param, row: int) -> QWidget:
        roles = self._current_range_roles()
        type_combo = QComboBox()
        type_combo.addItems(["Global", "Local", "Fixed"])
        default_role = "Fixed" if param.fixed else "Global"
        type_combo.setCurrentText(roles.get(param.name, default_role))
        type_combo.currentTextChanged.connect(self._on_param_table_edited)
        return type_combo

    def _read_param_row_control(self, widget: QWidget) -> dict[str, object]:
        if isinstance(widget, QComboBox):
            role = widget.currentText() or "Global"
            return {"role": role, "fixed": role == "Fixed"}
        return {"role": "Global", "fixed": False}

    def _error_cell_for_param(
        self, param_name: str, row_control: QWidget, result: object | None
    ) -> QTableWidgetItem:
        role = row_control.currentText() if isinstance(row_control, QComboBox) else "Global"
        return self._build_error_cell(param_name, role, result)

    def _post_select_range(self, idx: int) -> None:
        # Ensure no legacy fixed-checkbox widgets survive in the viewport.
        for row in range(self._param_table.rowCount()):
            name_item = self._param_table.item(row, 0)
            if name_item is None:
                continue
            widget = self._param_table.cellWidget(row, 4)
            if widget is not None and not isinstance(widget, QComboBox):
                self._param_table.removeCellWidget(row, 4)
                widget.deleteLater()

    def _build_error_cell(
        self, param_name: str, role: str, result: CrossGroupFitResult | None
    ) -> QTableWidgetItem:
        """Error-column cell for one parameter row.

        A *Global* parameter has one shared uncertainty. A *Local* parameter has
        a distinct fitted value and uncertainty *per group*, which a single
        number cannot represent — show "varies" and list every group's
        value ± error in the tooltip rather than silently reporting only the
        first group's. *Fixed* parameters carry no uncertainty.
        """
        if result is None:
            return QTableWidgetItem("")

        if role == "Global":
            err = result.global_uncertainties.get(param_name, np.nan)
            return QTableWidgetItem(f"{err:.4g}" if np.isfinite(err) else "")

        if role == "Local":
            lines: list[str] = []
            for group in self._groups:
                pset = result.local_parameters.get(group.group_id)
                if pset is None or param_name not in pset:
                    continue
                value = float(pset[param_name].value)
                gerr = result.local_uncertainties.get(group.group_id, {}).get(param_name)
                if gerr is not None and np.isfinite(gerr):
                    lines.append(f"{group.group_name}: {value:.6g} ± {gerr:.4g}")
                else:
                    lines.append(f"{group.group_name}: {value:.6g}")
            if not lines:
                return QTableWidgetItem("")
            item = QTableWidgetItem("varies")
            item.setToolTip("Per-group fitted value ± error:\n" + "\n".join(lines))
            return item

        # Fixed parameters have no uncertainty.
        return QTableWidgetItem("")

    def _commit_param_table(self, *, notify_adjustments: bool = False) -> None:
        super()._commit_param_table(notify_adjustments=notify_adjustments)
        if self._active_range_idx is None or self._active_range_idx >= len(self._fit.ranges):
            return
        while len(self._range_roles) < len(self._fit.ranges):
            self._range_roles.append({})
        roles: dict[str, str] = {}
        fit_range = self._fit.ranges[self._active_range_idx]
        for row in range(self._param_table.rowCount()):
            name_item = self._param_table.item(row, 0)
            type_combo = self._param_table.cellWidget(row, 4)
            if name_item is None or not isinstance(type_combo, QComboBox):
                continue
            pname_data = name_item.data(Qt.ItemDataRole.UserRole)
            pname = (
                str(pname_data).strip()
                if isinstance(pname_data, str) and str(pname_data).strip()
                else name_item.text().strip()
            )
            role = type_combo.currentText() or "Global"
            roles[pname] = role
            if pname in fit_range.parameters:
                # The base _commit_param_table already set .fixed from
                # _read_param_row_control's {"fixed"}; re-assert it here so the
                # role map stays the single source of truth for Fixed. Redundant
                # but harmless (same value); the roles-map write below is this
                # override's real job.
                fit_range.parameters[pname].fixed = role == "Fixed"
        self._range_roles[self._active_range_idx] = roles

    def _run_fit(self, idx: int | None = None) -> None:
        if self._fit_in_progress:
            _show_info(self, "Fit in progress", "Please wait for the current fit to finish.")
            return
        if idx is None:
            idx = self._active_range_idx if self._active_range_idx is not None else 0
        if idx < 0 or idx >= len(self._fit.ranges):
            return

        self._commit_param_table(notify_adjustments=True)
        fit_range = self._fit.ranges[idx]
        windows = list(fit_range.windows) if fit_range.windows else None
        x_min = float(fit_range.x_min if fit_range.x_min is not None else -float("inf"))
        x_max = float(fit_range.x_max if fit_range.x_max is not None else float("inf"))
        if windows:
            try:
                validate_fit_windows(windows)
            except ValueError as exc:
                _show_warning(self, "Invalid window", str(exc))
                return
        elif np.isfinite(x_min) and np.isfinite(x_max) and x_max <= x_min:
            _show_warning(self, "Invalid range", "x max must be greater than x min.")
            return

        # Slice each group to the window union (or the [x_min, x_max] range) here,
        # so the masking lives in one place; the shared model is then fitted
        # across the surviving points.
        fitted_groups: list[ParameterGroupData] = []
        for group in self._groups:
            x = np.asarray(group.x, dtype=float)
            if windows:
                mask = np.isfinite(x) & windows_mask(x, windows)
            else:
                mask = np.isfinite(x)
                if np.isfinite(x_min):
                    mask &= x >= x_min
                if np.isfinite(x_max):
                    mask &= x <= x_max
            if np.count_nonzero(mask) < 2:
                continue
            group_xe = getattr(group, "xerr", None)
            fitted_groups.append(
                ParameterGroupData(
                    group_id=group.group_id,
                    group_name=group.group_name,
                    x=np.asarray(group.x, dtype=float)[mask],
                    y=np.asarray(group.y, dtype=float)[mask],
                    yerr=np.asarray(group.yerr, dtype=float)[mask],
                    group_variable_value=float(group.group_variable_value),
                    xerr=(None if group_xe is None else np.asarray(group_xe, dtype=float)[mask]),
                )
            )

        if len(fitted_groups) < 2:
            _show_warning(
                self,
                "Insufficient data",
                "Not enough groups have at least two points in the selected fitting range.",
            )
            return

        roles = self._range_roles[idx] if idx < len(self._range_roles) else {}
        global_params: list[str] = []
        local_params: list[str] = []
        fixed_params: dict[str, float] = {}
        initial_params: dict[str, float] = {}
        parameter_bounds: dict[str, tuple[float, float]] = {}

        for p in fit_range.parameters:
            pname = p.name
            initial_params[pname] = float(p.value)
            parameter_bounds[pname] = (float(p.min), float(p.max))
            default_role = "Fixed" if p.fixed else "Global"
            role = roles.get(pname, default_role)
            if role == "Local":
                local_params.append(pname)
            elif role == "Fixed":
                fixed_params[pname] = float(p.value)
            else:
                global_params.append(pname)

        model_snapshot = ParameterCompositeModel(
            component_names=list(fit_range.model.component_names),
            operators=list(fit_range.model.operators),
        )
        groups_snapshot = [
            ParameterGroupData(
                group_id=group.group_id,
                group_name=group.group_name,
                x=np.asarray(group.x, dtype=float).copy(),
                y=np.asarray(group.y, dtype=float).copy(),
                yerr=np.asarray(group.yerr, dtype=float).copy(),
                group_variable_value=float(group.group_variable_value),
                xerr=(
                    None
                    if getattr(group, "xerr", None) is None
                    else np.asarray(group.xerr, dtype=float).copy()
                ),
            )
            for group in fitted_groups
        ]
        self._fit_progress_label.setText("Cross-group fit in progress...")
        error_mode = self._error_mode()
        error_value = self._error_value()
        # Effective-variance x-uncertainty: pass per-group σ_x only when the user
        # opted in (and the toggle is live, i.e. x is a fitted param under a real
        # error mode). The core ignores it under None/Scatter regardless.
        xerr_map: dict[str, object] | None = None
        if self._use_x_errors():
            xerr_map = {
                group.group_id: np.asarray(group.xerr, dtype=float)
                for group in groups_snapshot
                if getattr(group, "xerr", None) is not None
            }
            if not xerr_map:
                xerr_map = None

        def _task():
            # Groups are already sliced to the window/range above, so the window
            # mask is not re-applied here (windows omitted to avoid double-masking).
            return global_fit_parameter_model(
                groups=groups_snapshot,
                model=model_snapshot,
                global_params=list(global_params),
                local_params=list(local_params),
                fixed_params=dict(fixed_params),
                initial_params=dict(initial_params),
                parameter_bounds=dict(parameter_bounds),
                error_mode=error_mode,
                error_value=error_value,
                xerr=xerr_map,
            )

        def _on_done(result: object) -> None:
            fit_result = result
            self._result = fit_result
            self._fitted_groups = fitted_groups
            self._range_results[idx] = fit_result
            self._last_config = self._collect_config()

            # Item 4.3: no per-fit success MODAL. _select_range refreshes the
            # shared χ² label (inline success/failure text) and tints the result
            # box green on success. The failure case is surfaced inline too; the
            # blocking pre-check modals above (invalid window/range, <2 groups)
            # stay modal.
            self._select_range(idx)
            if not fit_result.success:
                _show_warning(
                    self,
                    "Cross-group fit failed",
                    fit_result.message or "Cross-group model fit failed",
                )

        self._start_fit_task(_task, _on_done)

    @staticmethod
    def _result_param_names(result: CrossGroupFitResult) -> set[str]:
        """Every parameter name the stored *result* actually covers.

        Global/fixed parameters are shared across the fit; local parameters
        vary per group but share the same name set, so the first group's
        entry is representative.
        """
        names = set(result.global_parameters.names) | set(result.fixed_parameters.names)
        for pset in result.local_parameters.values():
            names |= set(pset.names)
        return names

    def _on_use_fit(self) -> None:
        if self._fit_in_progress:
            QMessageBox.information(
                self,
                "Fit in progress",
                "Please wait for the current fit to finish before using the fit.",
            )
            return
        if self._result is None:
            QMessageBox.information(
                self,
                "No fit result",
                "Run Cross-Group Fit before using the fit.",
            )
            return

        # Only a result that actually ran classifies parameters (even a
        # non-converged Minuit run populates global/local/fixed with every
        # attempted name); the early-return failures inside
        # global_fit_parameter_model (bad windows, <2 groups, unknown
        # classification, missing iminuit) leave them empty and are not a
        # stale-model mismatch, so skip the check for those rather than
        # mislabel a genuine failure as "out of date".
        if self._result.success or self._result_param_names(self._result):
            current_param_names = set(self._model.param_names)
            if self._result_param_names(self._result) != current_param_names:
                QMessageBox.information(
                    self,
                    "Fit is out of date",
                    "The model changed since the last run — Run the fit again.",
                )
                return

        if not self._result.success:
            choice = QMessageBox.question(
                self,
                "Use failed fit?",
                (
                    "The current cross-group fit is marked as failed.\n\n"
                    "Do you want to use this fit anyway?"
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if choice != QMessageBox.StandardButton.Yes:
                return

        self.accept()

    def output(self) -> CrossGroupDialogOutput | None:
        if self._result is None:
            return None
        config = getattr(self, "_last_config", self._collect_config())
        fit_x_min = float("nan")
        fit_x_max = float("nan")
        if self._fit.ranges:
            r = self._fit.ranges[0]
            fit_x_min = float(r.x_min) if r.x_min is not None else float("nan")
            fit_x_max = float(r.x_max) if r.x_max is not None else float("nan")
        return CrossGroupDialogOutput(
            model=self._fit.ranges[0].model
            if self._fit.ranges
            else ParameterCompositeModel(["Linear"], []),
            fit_result=self._result,
            groups=getattr(self, "_fitted_groups", self._groups),
            x_key=self._x_key,
            fit_x_min=fit_x_min,
            fit_x_max=fit_x_max,
            config=config,
        )

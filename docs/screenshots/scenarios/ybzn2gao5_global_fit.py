r"""YbZn2GaO5 two-level analysis: batch relaxation fits -> global fit of lambda(B, T).

.. note:: **Slow.** This scenario drives the full published-parameter workflow
   from :mod:`asymmetry.examples.ybzn2gao5` (Wu *et al.*, arXiv:2502.00130):
   generate the synthetic 8-temperature x 20-field dataset, batch-fit every
   spectrum for its relaxation rate, assemble the lambda(B) trend per
   temperature, then run TWO cross-group global fits (the paper's full model,
   and an LCR-less duplicate for the Compare shot). Marked ``requires_fit =
   True`` -- both the per-run exponential fits and the two cross-group fits
   use iminuit, which trips on numpy >= 2.3 in dev environments; CI keeps
   numpy < 2.3.

   Wall-clock budget: ~160 fast single-run fits (a few seconds total) + two
   cross-group fits (~30-60 s each on the full 8x20 dataset) = comfortably
   inside the capture watchdog's 8-minute cap.

Screenshot -> file mapping (logical name -> PNG stem), kept here as the
single source of truth for the tutorial page (Phase S3) to reference:

* ``ybzn2gao5_runs_loaded``    -- data browser, 8 temperature groups.
* ``ybzn2gao5_batch_fit``      -- trend panel showing one group's batch of
  per-run ExponentialRelaxation fits (Lambda column populated).
* ``ybzn2gao5_trend_lambda``   -- trend panel, all 8 groups selected,
  Lambda vs field (log X).
* ``ybzn2gao5_setup_dialog``   -- GlobalFitSetupDialog, 8 series checked,
  group-variable table showing the 8 temperatures.
* ``ybzn2gao5_roles_dialog``   -- CrossGroupFitDialog with the paper's roles
  (D_2D/nu/f Local, A/D/lambda_BG/m/B0/Bwid Global, D_perp Fixed).
* ``ybzn2gao5_results_window`` -- GlobalParameterFitWindow: grid + components
  + legend, quality bar, global table (Fig. 3 + Table I money shot).
* ``ybzn2gao5_locals_vs_T``    -- same window's local-parameter pane, D_2D
  (log Y) vs temperature.
* ``ybzn2gao5_compare``        -- GlobalFitCompareDialog: full model vs an
  LCR-less (no GaussianLCR) duplicate study.

Only the two cross-group fits are expensive; everything else (data load,
grouping, per-run exponential fits, dialog construction) is cheap. The
per-run batch fit does not go through the interactive Batch-fit-tab click
path -- it calls :class:`~asymmetry.core.fitting.engine.FitEngine` directly
(the same call the recovery-gate test in
``tests/core/test_ybzn2gao5_example.py`` uses) and feeds the results straight
into :meth:`FitParametersPanel.load_representation_series`, the panel's own
plain-data "pull" entry point (bypassing MainWindow's full per-run-click
batch-fit recording machinery, which would cost nothing physically different
but a great deal of wall-clock and code weight for a screenshot).
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QApplication, QDialog

from ._base import CaptureContext, Scenario, register

#: Logical name -> on-disk stem, kept in one place per the task brief.
_IMAGE_NAMES = {
    "runs_loaded": "ybzn2gao5_runs_loaded",
    "batch_fit": "ybzn2gao5_batch_fit",
    "trend_lambda": "ybzn2gao5_trend_lambda",
    "setup_dialog": "ybzn2gao5_setup_dialog",
    "roles_dialog": "ybzn2gao5_roles_dialog",
    "results_window": "ybzn2gao5_results_window",
    "locals_vs_T": "ybzn2gao5_locals_vs_T",
    "compare": "ybzn2gao5_compare",
}

#: Fields per temperature for the screenshot dataset. The full recovery-gate
#: tolerance (10% / 3 sigma on every Table I global) is pinned at 20 fields/T
#: in the test suite; 20 is cheap enough here too (160 single-run fits) and
#: keeps the synthetic lambda(B) panels visually identical to the tutorial's
#: prose, which quotes the same generator call.
_FIELDS_PER_TEMPERATURE = 20


def _pump_events(milliseconds: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(int(milliseconds), loop.quit)
    loop.exec()
    QApplication.processEvents()


def _wait_until(predicate, timeout_s: float = 90.0, poll_s: float = 0.02) -> None:
    """Poll *predicate* on a live Qt event loop until true or *timeout_s* elapses.

    Used for the off-thread TaskRunner work in ``CrossGroupFitDialog`` (Run
    Cross-Group Fit) and ``GlobalParameterFitWindow`` (fit-curve compute) --
    the same style of wait ``tests/gui/test_cross_group_fit_dialog.py`` and
    ``tests/gui/test_global_parameter_fit_window.py`` use, reimplemented
    locally since ``tests/_qt_helpers.py`` is not on the docs-build path.
    """
    app = QApplication.instance()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if app is not None:
            app.processEvents()
        if predicate():
            return
        time.sleep(poll_s)
    raise TimeoutError(f"Timed out after {timeout_s}s waiting for background work")


class Ybzn2gao5GlobalFitScenario(Scenario):
    """The full YbZn2GaO5 two-level analysis, captured as eight PNGs.

    Overrides :meth:`capture` completely (rather than ``build``/``settle``)
    because a single ``Scenario`` here produces eight distinct screenshots
    from one expensive shared pipeline (generate -> batch-fit -> two global
    fits), and the base class's one-widget-per-scenario contract does not fit
    that. ``self.name`` is used only for registration; the individual PNGs
    are named from ``_IMAGE_NAMES`` above.
    """

    name = "ybzn2gao5_global_fit"
    description = (
        "YbZn2GaO5 two-level analysis (Wu et al. arXiv:2502.00130): batch "
        "relaxation fits, lambda(B) trend, and the cross-temperature global fit."
    )
    size = (1400, 900)
    requires_fit = True

    def capture(self, ctx: CaptureContext) -> Path:  # noqa: D401 -- overrides parent
        from asymmetry.examples.ybzn2gao5 import (
            DEFAULT_SEED,
            FIXED_PARAMS,
            GLOBAL_PARAMS,
            LOCAL_PARAMS,
            MODEL_COMPONENTS,
            generate_ybzn2gao5_runs,
        )

        last_path: Path | None = None
        with tempfile.TemporaryDirectory(prefix="ybzn2gao5_screenshots_") as tmp:
            manifest = generate_ybzn2gao5_runs(
                Path(tmp) / "runs",
                seed=DEFAULT_SEED,
                fields_per_temperature=_FIELDS_PER_TEMPERATURE,
            )

            window = self._build_window_with_runs(manifest)
            try:
                rows_by_temp = self._batch_fit_all_runs(window, manifest)
                last_path = self._capture_runs_loaded(ctx, window)
                last_path = self._capture_batch_fit(ctx, window, rows_by_temp)
                last_path = self._capture_trend_lambda(ctx, window)

                groups = self._assemble_groups(window, rows_by_temp)

                last_path = self._capture_setup_dialog(ctx, window)

                full_model_names = list(MODEL_COMPONENTS)
                roles = {
                    **{p: "Global" for p in GLOBAL_PARAMS},
                    **{p: "Local" for p in LOCAL_PARAMS},
                    **{p: "Fixed" for p in FIXED_PARAMS},
                }

                full_dialog = self._make_cross_group_dialog(groups, full_model_names, roles)
                last_path = self._capture_roles_dialog(ctx, full_dialog)

                full_result, full_model = self._run_cross_group_fit_sync(full_dialog)
                full_dialog.close()
                full_dialog.deleteLater()

                gwindow = self._show_results_window(window, groups, full_model, full_result)
                last_path = self._capture_results_window(ctx, gwindow)
                last_path = self._capture_locals_vs_t(ctx, gwindow)

                last_path = self._capture_compare(ctx, groups, full_model, full_result, roles)
            finally:
                # Suppress the unsaved-changes guard: this MainWindow carries a
                # loaded session and would otherwise block indefinitely on a
                # modal save prompt offscreen (see Scenario.teardown()).
                if hasattr(window, "_dirty"):
                    window._dirty = False
                window.close()
                window.deleteLater()
                _pump_events(60)

        assert last_path is not None
        return last_path

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _build_window_with_runs(self, manifest):
        from asymmetry.core.io import load as io_load
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window.resize(*self.size)
        window.resizeDocks([window._dock_data_browser], [340], Qt.Orientation.Horizontal)

        with window._data_browser.batch_updates():
            for spec in manifest.runs:
                result = io_load(spec.path)
                dataset = result[0] if isinstance(result, list) else result
                window._data_browser.add_dataset(dataset)

        # Group by temperature: one DataGroup per unique T (8 groups). No bulk
        # "group by column" helper exists on DataBrowserPanel -- bucket the
        # manifest's run numbers by temperature ourselves and call
        # create_data_group once per bucket, batched for O(n) table rebuilds.
        runs_by_temp: dict[float, list[int]] = {}
        for spec in manifest.runs:
            runs_by_temp.setdefault(spec.temperature, []).append(spec.run_number)

        with window._data_browser.batch_updates():
            for temperature in manifest.temperatures:
                run_numbers = runs_by_temp[temperature]
                # collapsed=True so the browser shows one group header row per
                # temperature rather than all 160 individual runs expanded --
                # this is the "grouped by temperature (8 groups)" screenshot.
                window._data_browser.create_data_group(
                    run_numbers, name=f"T={temperature:g} K", collapsed=True
                )
        _pump_events(80)
        return window

    def _batch_fit_all_runs(self, window, manifest) -> dict[float, list[tuple]]:
        """Fit every run's exponential relaxation rate; return per-T (field, lambda, err) rows.

        Uses :class:`FitEngine` directly -- the same call
        ``tests/core/test_ybzn2gao5_example.py::_fit_relaxation_rate`` makes --
        rather than driving the interactive Batch-fit-tab click path, which
        would add no new pixels to this screenshot but a great deal of
        wall-clock (one real GUI batch-fit dispatch per group, 8 groups).
        """
        from asymmetry.core.fitting.engine import FitEngine
        from asymmetry.core.fitting.models import MODELS
        from asymmetry.core.fitting.parameters import Parameter, ParameterSet
        from asymmetry.core.io import load as io_load

        engine = FitEngine()
        model = MODELS["ExponentialRelaxation"]

        rows_by_temp: dict[float, list[tuple]] = {}
        for spec in manifest.runs:
            result = io_load(spec.path)
            dataset = result[0] if isinstance(result, list) else result
            params = ParameterSet()
            params.add(Parameter("A0", 20.0, 0.0, 100.0))
            params.add(Parameter("Lambda", max(float(spec.lambda_truth), 0.05), 0.0, 100.0))
            params.add(Parameter("baseline", 3.0, -20.0, 20.0))
            fit_result = engine.fit(dataset, model.function, params)
            lam = float(fit_result.parameters["Lambda"].value)
            err = float(fit_result.uncertainties.get("Lambda", 0.0) or 0.01 * abs(lam))
            rows_by_temp.setdefault(spec.temperature, []).append(
                (spec.run_number, spec.field_gauss, lam, err, fit_result)
            )
        return rows_by_temp

    def _load_series_into_trend_panel(self, window, rows_by_temp) -> None:
        """Feed the batch-fit results into the trend panel via its plain-data API.

        :meth:`FitParametersPanel.load_representation_series` is the panel's
        own "pull" entry point (row dicts keyed by run_number/field/
        temperature/values/errors) -- the same shape MainWindow builds after a
        real batch fit, so this is a faithful shortcut, not a simulation of a
        different data path.
        """
        from asymmetry.core.fitting.result_summary import fit_result_summary

        series_entries: list[tuple[str, str, list[dict]]] = []
        for temperature, rows in rows_by_temp.items():
            batch_id = f"ybzn2gao5-T{temperature:g}"
            row_dicts = []
            for run_number, field_gauss, lam, err, fit_result in rows:
                summary = fit_result_summary(fit_result)
                row_dicts.append(
                    {
                        "run_number": run_number,
                        "run_label": str(run_number),
                        "field": field_gauss,
                        "temperature": temperature,
                        "values": dict(summary.get("parameters") or {}),
                        "errors": dict(summary.get("uncertainties") or {}),
                        "model_name": "ExponentialRelaxation",
                        "batch_id": batch_id,
                    }
                )
            series_entries.append((batch_id, f"T={temperature:g} K", row_dicts))

        window._fit_parameters_panel.load_representation_series(series_entries)
        _pump_events(80)

    def _assemble_groups(self, window, rows_by_temp):
        """Select every temperature's group button and switch to the Batch dock."""
        self._load_series_into_trend_panel(window, rows_by_temp)

        panel = window._fit_parameters_panel
        # The "Parameters" dock (the trend panel) is tabified with the Fit
        # dock (_dock_fit_parameters / _dock_fit share one tab group) -- raise
        # its tab explicitly, since _on_fit() only raises the Fit tab.
        window._show_panel("fit_parameters")
        _pump_events(80)

        all_group_ids = list(panel._group_button_map.keys())
        panel._set_selected_group_ids(all_group_ids, emit=False)
        panel._apply_group_selection_to_view(sync_active=False)

        # Field on a log axis (the synthetic grid spans 10 G - 45000 G).
        panel._x_combo.setCurrentText("𝐵 (G)")
        panel._log_x_check.setChecked(True)
        _pump_events(80)

        groups = panel.assemble_cross_group_groups(
            "Lambda", panel._effective_x_key(), all_group_ids
        )
        assert groups is not None and len(groups) == len(rows_by_temp), (
            "expected all 8 temperature groups to assemble into ParameterGroupData"
        )
        return groups

    # ------------------------------------------------------------------
    # Screenshot 1 -- data browser with 8 temperature groups
    # ------------------------------------------------------------------

    def _capture_runs_loaded(self, ctx: CaptureContext, window) -> Path:
        # Widen the data-browser dock (from the default 340px) so the group
        # header rows ("T=... K (20 members)") are legible rather than
        # elided -- the whole point of this screenshot is showing the
        # grouping-by-temperature outcome clearly.
        window.resizeDocks([window._dock_data_browser], [520], Qt.Orientation.Horizontal)
        window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        window.show()
        _pump_events(150)
        window._data_browser._resize_columns_to_content()
        _pump_events(80)
        return self._save(ctx, "runs_loaded", window)

    # ------------------------------------------------------------------
    # Screenshot 2 -- fit panel after batch-fitting one temperature group
    # ------------------------------------------------------------------

    def _capture_batch_fit(self, ctx: CaptureContext, window, rows_by_temp) -> Path:
        self._load_series_into_trend_panel(window, rows_by_temp)
        window._show_panel("fit_parameters")
        _pump_events(80)

        panel = window._fit_parameters_panel
        # Focus a single mid-temperature group (T=1.6 K) so the trend panel
        # shows one group's batch of per-run ExponentialRelaxation fits, the
        # closest available representation of "the fit panel after batch-
        # fitting one temperature group" without reconstructing the
        # interactive per-run batch-fit click path (see _batch_fit_all_runs).
        target_gid = next(
            gid for gid, group in panel._group_fit_results.items() if group.group_name == "T=1.6 K"
        )
        panel._set_selected_group_ids([target_gid], emit=True)
        panel._apply_group_selection_to_view(sync_active=True)

        # Select the Lambda row in the Y-parameter selector so the panel
        # shows the relaxation rate (the quantity of interest) rather than
        # whatever row the rebuild defaulted to.
        for row in range(panel._y_selector_table.rowCount()):
            item = panel._y_selector_table.item(row, 0)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == "Lambda":
                panel._y_selector_table.selectRow(row)
                break
        # The Y-selection change only starts a 120ms debounce timer
        # (_plot_refresh_timer) rather than redrawing synchronously; call
        # _refresh_plot() directly so the grab is deterministic rather than
        # racing the timer.
        panel._refresh_plot()
        _pump_events(120)
        return self._save(ctx, "batch_fit", window)

    # ------------------------------------------------------------------
    # Screenshot 3 -- trend panel, 8 groups, Lambda vs field (log X)
    # ------------------------------------------------------------------

    def _capture_trend_lambda(self, ctx: CaptureContext, window) -> Path:
        panel = window._fit_parameters_panel
        all_group_ids = list(panel._group_button_map.keys())
        panel._set_selected_group_ids(all_group_ids, emit=True)
        panel._apply_group_selection_to_view(sync_active=False)
        panel._x_combo.setCurrentText("𝐵 (G)")
        panel._log_x_check.setChecked(True)

        for row in range(panel._y_selector_table.rowCount()):
            item = panel._y_selector_table.item(row, 0)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == "Lambda":
                panel._y_selector_table.selectRow(row)
                break
        # See _capture_batch_fit: force the redraw rather than racing the
        # panel's own 120ms debounce timer.
        panel._refresh_plot()
        _pump_events(150)
        return self._save(ctx, "trend_lambda", window)

    # ------------------------------------------------------------------
    # Screenshot 4 -- GlobalFitSetupDialog
    # ------------------------------------------------------------------

    def _capture_setup_dialog(self, ctx: CaptureContext, window) -> Path:
        from asymmetry.gui.panels.global_fit_setup_dialog import GlobalFitSetupDialog

        panel = window._fit_parameters_panel
        setup_data = panel.global_fit_setup_data()
        dialog = GlobalFitSetupDialog(
            setup_data,
            preselected_group_ids=[s.group_id for s in setup_data.series],
            preselected_parameter="Lambda",
            preselected_x_key="field",
            parent=window,
        )
        dialog.resize(720, 640)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        dialog.show()
        _pump_events(120)
        path = self._save(ctx, "setup_dialog", dialog)
        dialog.close()
        dialog.deleteLater()
        _pump_events(40)
        return path

    # ------------------------------------------------------------------
    # Screenshot 5 -- CrossGroupFitDialog with the paper's roles
    # ------------------------------------------------------------------

    def _make_cross_group_dialog(self, groups, model_component_names, roles):
        from asymmetry.core.fitting.parameter_models import ParameterCompositeModel
        from asymmetry.core.fitting.parameters import Parameter, ParameterSet
        from asymmetry.examples.ybzn2gao5 import PARAM_BOUNDS
        from asymmetry.gui.panels.cross_group_fit_dialog import CrossGroupFitDialog

        dialog = CrossGroupFitDialog(
            parameter_name="Lambda",
            x_key="field",
            groups=groups,
            parent=None,
            x_label="Field  B (G)",
        )
        dialog.resize(1100, 780)

        # Swap in the paper's 4-component model (in place of the default
        # Linear stand-in) and seed physically-sensible starting values --
        # the same style as ``_edit_model``'s own post-edit reseed, without
        # driving the modal ParameterModelBuilderDialog for a fixed,
        # already-known composite string.
        model = ParameterCompositeModel(list(model_component_names))
        fit_range = dialog._fit.ranges[0]
        fit_range.model = model
        fit_range.result = None

        seed_values = {
            "A": 61.0,
            "D": 18.0,
            "lambda_BG": 0.065,
            "m": 6.9,
            "B0": 26800.0,
            "Bwid": 12900.0,
            "D_2D": 15.0e3,
            "nu": 350.0,
            "f": 0.09,
            "D_perp": 0.0,
        }
        new_params = ParameterSet()
        for pname in model.param_names:
            bounds = PARAM_BOUNDS.get(pname, (0.0, 1.0e9))
            value = seed_values.get(pname, model.param_defaults.get(pname, 0.0))
            new_params.add(
                Parameter(
                    name=pname,
                    value=float(value),
                    min=float(bounds[0]),
                    max=float(bounds[1]),
                    fixed=(roles.get(pname) == "Fixed"),
                )
            )
        fit_range.parameters = new_params
        dialog._on_model_edited(0)
        dialog._range_roles = [dict(roles)]

        dialog._rebuild_ranges_ui()
        dialog._select_range(0)

        # Apply the paper's Global/Local/Fixed roles to the Type combo per row.
        for row in range(dialog._param_table.rowCount()):
            name_item = dialog._param_table.item(row, 0)
            combo = dialog._param_table.cellWidget(row, 4)
            if name_item is None or combo is None:
                continue
            pname = str(name_item.data(Qt.ItemDataRole.UserRole) or name_item.text())
            role = roles.get(pname)
            if role is not None:
                combo.setCurrentText(role)
        dialog._commit_param_table(notify_adjustments=False)
        return dialog

    def _capture_roles_dialog(self, ctx: CaptureContext, dialog) -> Path:
        dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        dialog.show()
        _pump_events(150)
        return self._save(ctx, "roles_dialog", dialog)

    # ------------------------------------------------------------------
    # Cross-group fit (run for real; ~30-60s on the full 8x20 dataset)
    # ------------------------------------------------------------------

    def _run_cross_group_fit_sync(self, dialog):
        # _run_fit dispatches to the shared TaskRunner (off the GUI thread);
        # PYTEST_CURRENT_TEST suppresses the interactive "Fit complete"/
        # "Fit failed" QMessageBox that _show_info/_show_warning would
        # otherwise pop up (guarded by _in_test_mode() in model_fit_dialog.py)
        # -- there is no display to click it away on in an offscreen capture.
        previous = os.environ.get("PYTEST_CURRENT_TEST")
        os.environ["PYTEST_CURRENT_TEST"] = "docs.screenshots.capture (offscreen)"
        try:
            dialog._run_fit(0)
            _wait_until(lambda: not dialog._fit_in_progress, timeout_s=180.0)
        finally:
            if previous is None:
                os.environ.pop("PYTEST_CURRENT_TEST", None)
            else:
                os.environ["PYTEST_CURRENT_TEST"] = previous

        assert dialog._result is not None and dialog._result.success, (
            f"cross-group fit did not converge: {dialog._result}"
        )
        dialog._on_use_fit()
        assert dialog.result() == QDialog.DialogCode.Accepted
        output = dialog.output()
        assert output is not None
        return output.fit_result, output.model

    # ------------------------------------------------------------------
    # Screenshot 6 + 7 -- GlobalParameterFitWindow (results + locals-vs-T)
    # ------------------------------------------------------------------

    def _show_results_window(self, window, groups, model, result):
        from asymmetry.gui.windows.global_parameter_fit_window import (
            GlobalParameterFitWindow,
        )

        gwindow = GlobalParameterFitWindow(window)
        gwindow.resize(1400, 900)
        gwindow.set_results(
            parameter_name="Lambda",
            x_key="field",
            groups=groups,
            model=model,
            result=result,
            x_label="Field  B (G)",
            group_variable_label="Temperature  T (K)",
            batch_id="ybzn2gao5-full-model",
        )
        _wait_until(lambda: not gwindow._fit_curve_compute_active, timeout_s=60.0)
        return gwindow

    def _capture_results_window(self, ctx: CaptureContext, gwindow) -> Path:
        gwindow._show_components_check.setChecked(True)
        # Toggling "Show components" is a cache miss for the components=True
        # flag, so it kicks a NEW off-thread curve compute (_refresh_plot ->
        # _start_fit_curve_compute) and re-arms the "Computing fit curves..."
        # overlay -- wait for it again, exactly as after the initial
        # set_results, or the grab races a still-rendering panel.
        _wait_until(lambda: not gwindow._fit_curve_compute_active, timeout_s=60.0)
        _pump_events(150)
        gwindow.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        gwindow.show()
        _pump_events(200)
        return self._save(ctx, "results_window", gwindow)

    def _capture_locals_vs_t(self, ctx: CaptureContext, gwindow) -> Path:
        gwindow._local_plot_mode_combo.setCurrentText("Single Axes")
        # Select only D_2D so the local-parameters pane reads cleanly; log Y
        # spans the quantum-plateau-to-classical rise cleanly (paper Fig. 4).
        table = gwindow._local_y_selector_table
        for row in range(table.rowCount()):
            item = table.item(row, 0)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == "D_2D":
                table.selectRow(row)
                break
        controls = gwindow._local_y_controls.get("D_2D")
        if controls is not None:
            controls["log"].setChecked(True)
        gwindow._refresh_local_parameter_plots()
        _pump_events(150)
        return self._save(ctx, "locals_vs_T", gwindow)

    # ------------------------------------------------------------------
    # Screenshot 8 -- Compare dialog: full model vs an LCR-less duplicate
    # ------------------------------------------------------------------

    def _capture_compare(self, ctx: CaptureContext, groups, full_model, full_result, roles) -> Path:
        from asymmetry.core.fitting.parameter_models import (
            ParameterCompositeModel,
            global_fit_parameter_model,
        )
        from asymmetry.core.representation.global_fit_study import (
            GlobalFitStudy,
            compute_group_input_digest,
        )
        from asymmetry.examples.ybzn2gao5 import GLOBAL_PARAMS, LOCAL_PARAMS, PARAM_BOUNDS
        from asymmetry.gui.windows.global_fit_compare_dialog import GlobalFitCompareDialog

        # LCR-less duplicate: drop GaussianLCR (and its role f), keep the rest.
        lcrless_names = [n for n in full_model.component_names if n != "GaussianLCR"]
        lcrless_model = ParameterCompositeModel(lcrless_names)
        lcrless_global = [p for p in GLOBAL_PARAMS if p in lcrless_model.param_names]
        lcrless_local = [p for p in LOCAL_PARAMS if p in lcrless_model.param_names]
        lcrless_fixed = {p: 0.0 for p in lcrless_model.param_names if roles.get(p) == "Fixed"}

        seed_values = {
            "A": full_result.global_parameters["A"].value
            if "A" in full_result.global_parameters.names
            else 61.0,
            "D": 18.0,
            "lambda_BG": 0.065,
            "m": 6.9,
            "B0": 26800.0,
            "Bwid": 12900.0,
            "D_2D": 15.0e3,
            "nu": 350.0,
            "D_perp": 0.0,
        }
        initial_params = {
            p: float(seed_values.get(p, lcrless_model.param_defaults.get(p, 0.0)))
            for p in lcrless_model.param_names
        }
        bounds = {p: PARAM_BOUNDS[p] for p in lcrless_model.param_names if p in PARAM_BOUNDS}

        lcrless_result = global_fit_parameter_model(
            groups,
            lcrless_model,
            lcrless_global,
            lcrless_local,
            lcrless_fixed,
            initial_params=initial_params,
            parameter_bounds=bounds,
        )

        digest = compute_group_input_digest(groups)
        now = "2026-01-01T09:30:00+00:00"  # frozen, matches capture.py's determinism patch
        study_a = GlobalFitStudy(
            study_id="ybzn2gao5-full-model",
            name="Full model (with LCR)",
            parameter_name="Lambda",
            x_key="field",
            x_label="Field  B (G)",
            group_variable_key="temperature",
            group_variable_label="Temperature  T (K)",
            created=now,
            updated=now,
            source_group_ids=[g.group_id for g in groups],
            groups=groups,
            model=full_model,
            result=full_result,
            input_digest=digest,
        )
        study_b = GlobalFitStudy(
            study_id="ybzn2gao5-no-lcr",
            name="No LCR term",
            parameter_name="Lambda",
            x_key="field",
            x_label="Field  B (G)",
            group_variable_key="temperature",
            group_variable_label="Temperature  T (K)",
            created=now,
            updated=now,
            source_group_ids=[g.group_id for g in groups],
            groups=groups,
            model=lcrless_model,
            result=lcrless_result,
            input_digest=digest,
        )

        dialog = GlobalFitCompareDialog(study_a, study_b, parent=None)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        dialog.show()
        _pump_events(200)
        path = self._save(ctx, "compare", dialog)
        dialog.close()
        dialog.deleteLater()
        _pump_events(40)
        return path

    # ------------------------------------------------------------------
    # Shared save helper
    # ------------------------------------------------------------------

    def _save(self, ctx: CaptureContext, key: str, widget) -> Path:
        name = _IMAGE_NAMES[key]
        width = int(widget.width() * ctx.device_pixel_ratio)
        height = int(widget.height() * ctx.device_pixel_ratio)
        from PySide6.QtGui import QPixmap

        pixmap = QPixmap(width, height)
        pixmap.setDevicePixelRatio(ctx.device_pixel_ratio)
        pixmap.fill(Qt.GlobalColor.transparent)
        widget.render(pixmap)

        out_path = ctx.output_dir / f"{name}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not pixmap.save(str(out_path), "PNG"):
            raise RuntimeError(f"Failed to save screenshot to {out_path}")
        print(f"[ybzn2gao5_global_fit] wrote {out_path}", flush=True)
        return out_path


register(Ybzn2gao5GlobalFitScenario())

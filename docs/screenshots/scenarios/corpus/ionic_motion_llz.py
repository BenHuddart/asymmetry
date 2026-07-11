"""Corpus scenarios — Ionic motion in a solid electrolyte (Al-doped LLZ garnet).

Flagship **global-fit** worked example from the WiMDA muon school corpus.
EMU (ISIS) longitudinal-field muon decoupling on the candidate Li-ion battery
electrolyte Al-Li₇La₃Zr₂O₁₂: at each temperature the sample is measured at three
longitudinal fields (0 / 5 / 10 G) and the triplet is fitted *simultaneously*
with a **Keren** dynamic-Gaussian relaxation + flat background, sharing the
static field-distribution width Δ and the fluctuation rate ν across the three
runs while the field B_L is fixed per run to its set value (§4 of
``GROUND_TRUTH.md``). Fitting the ν(T) trend over the 13-temperature series and
Arrhenius-analysing the activated rise recovers the Li⁺ activation energy — the
paper value is E_a = 0.19(1) eV (Amores *et al.*, J. Mater. Chem. A 4, 1729
(2016), the published dataset).

Scenarios registered:

* ``corpus_llz_lf_triplet`` — the raw 0/5/10 G overlay at one temperature (the
  signature of weak LF decoupling).
* ``corpus_llz_global_setup`` — the Batch/global-fit panel with the triplet
  loaded, the Keren + Constant model set, and the parameter roles tied
  (Δ, ν, amplitudes **Global**; B_L read per-run **From File**). The parameter
  tying render — THE global-fit showcase.
* ``corpus_llz_global_result`` — the converged triplet fit with the shared
  global parameters and per-run Keren curves overlaid on the data.
* ``corpus_llz_nu_arrhenius`` — the fluctuation-rate trend ν(T) across the
  series with the Arrhenius + constant-baseline curve overlaid, giving E_a.

The Keren model named by the guide exists in Asymmetry
(``asymmetry.core.fitting.models.keren``, params A, Δ, ν, B_L) so no model
substitution is needed.  See ``NOTES_llz.md`` for run selection, the fitted
values, and the E_a comparison against the paper.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QApplication, QComboBox, QTableWidgetItem, QWidget

from .._base import CaptureContext
from ._corpus import CorpusScenario, load_corpus_datasets, register, _process_events_for

EXAMPLE = "Nuclear magnetism and ionic motion/Ionic motion in a solid electrolyte"
_DATA = EXAMPLE + "/Data"

# ZF run number that opens each temperature's triplet; triplet = (zf, zf+1, zf+2)
# = (0 G, 5 G, 10 G).  Setpoint temperatures from GROUND_TRUTH.md §3.
TRIPLETS: dict[int, int] = {
    160: 51341, 180: 51344, 200: 51347, 220: 51350, 240: 51353, 264: 51356,
    284: 51359, 304: 51362, 324: 51365, 344: 51368, 364: 51371, 384: 51374,
    404: 51377,
}

# Guide seed values at 160 K (GROUND_TRUTH.md §4 — starting values, not results).
SEED_A_SIGNAL = 15.0   # sample-signal amplitude (%)
SEED_A_BG = 5.0        # background amplitude (%)
SEED_DELTA = 0.3       # static field-distribution width Δ (µs⁻¹ / "MHz" seed)
SEED_NU = 0.2          # fluctuation rate ν (MHz)
FIT_TMAX = 12.0        # limit fit window to 12 µs (guide hint)


def _triplet_rel_paths(zf_run: int) -> list[str]:
    return [f"{_DATA}/emu000{zf_run + i}.nxs" for i in range(3)]


def _keren_model():
    from asymmetry.core.fitting.composite import CompositeModel

    return CompositeModel(["Keren", "Constant"])


class LlzCalibrationScenario(CorpusScenario):
    """TF 20 G calibration run 51315 — estimate α before the science fits."""

    name = "corpus_llz_calibration"
    description = (
        "Alpha calibration on the Al-LLZ TF 20 G run 51315: the data-prep step "
        "that fixes the detector balance α before the LF triplet fits."
    )
    example = EXAMPLE
    size = (760, 640)

    def capture(self, ctx: CaptureContext) -> Path:  # noqa: D401
        from asymmetry.gui.windows.grouping.alpha_calibration_dialog import (
            AlphaCalibrationDialog,
        )

        dataset = load_corpus_datasets([f"{_DATA}/emu00051315.nxs"])[0]
        grouping = dataset.run.grouping

        dialog = AlphaCalibrationDialog(
            [dataset],
            groups=grouping["groups"],
            group_names=grouping.get("group_names"),
            forward_group=grouping["forward_group"],
            backward_group=grouping["backward_group"],
            selected_run_number=int(dataset.run_number),
        )
        dialog.resize(*self.size)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        dialog.show()
        _pump(150)

        estimate_btn = getattr(dialog, "_estimate_btn", None)
        if estimate_btn is not None:
            estimate_btn.click()
            _pump_until(
                lambda: dialog._tasks.active_count == 0 and dialog._estimate is not None
            )
            _pump(60)

        pix = dialog.grab()
        out_path = ctx.output_dir / f"{self.name}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not pix.save(str(out_path), "PNG"):
            raise RuntimeError(f"Failed to save screenshot to {out_path}")
        dialog.close()
        dialog.deleteLater()
        _pump(40)
        return out_path


class LlzLfTripletScenario(CorpusScenario):
    """Raw 0/5/10 G overlay at 160 K — the weak-LF-decoupling signature."""

    name = "corpus_llz_lf_triplet"
    description = (
        "Al-LLZ 160 K longitudinal-field triplet (0/5/10 G) overlaid: the raw "
        "signature of weak LF decoupling of the muon from Li nuclear fields."
    )
    example = EXAMPLE
    size = (1400, 880)

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window.resizeDocks([window._dock_data_browser], [300], Qt.Orientation.Horizontal)

        datasets = load_corpus_datasets(_triplet_rel_paths(TRIPLETS[160]))
        self.add_to_browser(window, datasets)

        run_numbers = [int(ds.run_number) for ds in datasets]
        window._data_browser.create_data_group(run_numbers, name="Al-LLZ 160 K (0/5/10 G)")

        window._plot_panel.set_overlay_enabled(True, emit_signal=True)
        window._plot_panel.set_bunch_factor(6, emit_signal=True)
        window._data_browser._table.selectAll()
        window._on_dataset_selected(run_numbers[0])
        _process_events_for(milliseconds=120)
        # Clip to the 12 µs analysis window: past ~13 µs the F–B asymmetry
        # diverges as the counts vanish and swamps the decoupling signature.
        window._plot_panel.set_view_limits(0.0, 12.0, -4.0, 14.0)
        _process_events_for(milliseconds=120)
        return window


class LlzGlobalSetupScenario(CorpusScenario):
    """Batch/global-fit panel with the triplet loaded and parameters tied."""

    name = "corpus_llz_global_setup"
    description = (
        "Global-fit setup on the Al-LLZ 160 K triplet: Keren + Constant model "
        "with Δ, ν and the amplitudes shared (Global) and B_L fixed per run "
        "(From File) — the parameter-tying render."
    )
    example = EXAMPLE
    size = (1680, 1000)

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window._on_fit()
        window.resizeDocks([window._dock_data_browser], [280], Qt.Orientation.Horizontal)

        datasets = load_corpus_datasets(_triplet_rel_paths(TRIPLETS[160]))
        self.add_to_browser(window, datasets)
        run_numbers = [int(ds.run_number) for ds in datasets]
        window._data_browser.create_data_group(run_numbers, name="Al-LLZ 160 K (0/5/10 G)")

        fit_panel = window._fit_panel
        fit_panel._tabs.setCurrentWidget(fit_panel._global_tab)
        fit_panel.set_datasets(datasets)
        global_tab = fit_panel._global_tab
        global_tab.set_current_dataset(datasets[0])

        # Keren + flat background; then tie parameters as the guide prescribes.
        global_tab._set_composite_model(_keren_model())
        _process_events_for(milliseconds=60)
        _configure_triplet_param_table(global_tab)

        window._plot_panel.set_bunch_factor(6, emit_signal=True)
        # Select all three runs so the batch surface reflects the loaded triplet.
        window._data_browser._table.selectAll()
        window._on_dataset_selected(run_numbers[0])
        _process_events_for(milliseconds=120)
        # Limit the fit window to 12 µs (guide hint) — set after the dataset
        # selection so the auto data-range refresh does not overwrite it.
        if getattr(global_tab, "_fit_range_max_spin", None) is not None:
            global_tab._fit_range_max_spin.setValue(FIT_TMAX)
        window._plot_panel.set_view_limits(0.0, 12.0, -4.0, 14.0)
        _process_events_for(milliseconds=120)
        return window

    def settle(self, widget: QWidget) -> None:
        _process_events_for(milliseconds=200)
        # Widen the fit dock *after* show: showEvent applies the adaptive
        # default inspector width, which would clobber a resize done in
        # build(). The extra room lets the parameter-classification table show
        # its bounds column without clipping at the right edge.
        widget.resizeDocks([widget._dock_fit], [560], Qt.Orientation.Horizontal)
        _process_events_for(milliseconds=200)


class LlzGlobalResultScenario(CorpusScenario):
    """Converged simultaneous Keren fit of the triplet with per-run overlays."""

    name = "corpus_llz_global_result"
    description = (
        "Converged global Keren fit of the Al-LLZ 160 K triplet: shared Δ, ν "
        "and amplitudes with per-run field, Keren curves overlaid on the data."
    )
    example = EXAMPLE
    size = (1500, 900)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.core.fitting.engine import FitEngine
        from asymmetry.core.fitting.parameters import Parameter, ParameterSet
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window._on_fit()
        window.resizeDocks([window._dock_data_browser], [280], Qt.Orientation.Horizontal)

        datasets = load_corpus_datasets(_triplet_rel_paths(TRIPLETS[160]))
        self.add_to_browser(window, datasets)
        run_numbers = [int(ds.run_number) for ds in datasets]
        window._data_browser.create_data_group(run_numbers, name="Al-LLZ 160 K (0/5/10 G)")

        fit_panel = window._fit_panel
        fit_panel._tabs.setCurrentWidget(fit_panel._global_tab)
        fit_panel.set_datasets(datasets)
        global_tab = fit_panel._global_tab
        global_tab.set_current_dataset(datasets[0])
        model = _keren_model()
        global_tab._set_composite_model(model)
        _process_events_for(milliseconds=60)
        _configure_triplet_param_table(global_tab)

        # Run the coupled fit synchronously (the Global tab worker runs on a
        # QThread; for a deterministic capture we drive the engine directly and
        # feed the result into the same success-render path).
        engine = FitEngine()
        initial_params: dict[int, ParameterSet] = {}
        for ds in datasets:
            initial_params[int(ds.run_number)] = ParameterSet([
                Parameter("A_1", value=SEED_A_SIGNAL, min=0.0, max=40.0),
                Parameter("Delta", value=SEED_DELTA, min=0.0, max=2.0),
                Parameter("nu", value=SEED_NU, min=0.0, max=8.0),
                Parameter("B_L", value=float(ds.field), fixed=True),
                Parameter("A_bg", value=SEED_A_BG, min=-10.0, max=20.0),
            ])
        global_params = ["A_1", "Delta", "nu", "A_bg"]
        local_params = ["B_L"]
        results_dict, fitted_global = engine.global_fit(
            datasets=datasets,
            model_fn=model.function,
            global_params=global_params,
            local_params=local_params,
            initial_params=initial_params,
            t_min=0.0,
            t_max=FIT_TMAX,
        )
        global_tab._emit_global_fit_success(
            model=model,
            results_dict=results_dict,
            fitted_global=fitted_global,
            global_param_names=global_params,
        )
        _process_events_for(milliseconds=120)
        # The completion signal raises the Parameters dock; bring the Batch fit
        # results (fitted shared Δ, ν and average χ²) back to the front.
        window._dock_fit.raise_()

        window._plot_panel.set_bunch_factor(6, emit_signal=True)
        # Select the ZF (0 G) run — the most strongly relaxing trace.
        window._on_dataset_selected(run_numbers[0])
        _process_events_for(milliseconds=120)
        window._plot_panel.set_view_limits(0.0, 12.0, -4.0, 14.0)
        _process_events_for(milliseconds=120)
        return window


class LlzNuArrheniusScenario(CorpusScenario):
    """ν(T) fluctuation-rate trend across the series with an Arrhenius fit."""

    name = "corpus_llz_nu_arrhenius"
    description = (
        "Al-LLZ fluctuation rate ν(T) across the 13-temperature series: flat "
        "plateau then activated rise above ~290 K, with the Arrhenius + "
        "constant-baseline curve giving the Li⁺ activation energy."
    )
    example = EXAMPLE
    size = (1240, 780)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

        temps, nu, nu_err = _fit_nu_of_t()
        fit = _build_arrhenius_fit(temps, nu, nu_err)

        batch_id = "llz-nu-t"
        row_dicts = [
            {
                "run_number": TRIPLETS[int(t)],
                "run_label": f"{int(t)} K",
                "field": 0.0,
                "temperature": float(t),
                "values": {"nu": float(nu[i])},
                "errors": {"nu": float(nu_err[i])},
            }
            for i, t in enumerate(temps)
        ]

        panel = FitParametersPanel()
        panel.load_representation_series(
            [(batch_id, "ν(T) — Al-LLZ", row_dicts)], select_id=batch_id
        )
        panel._model_fits["nu"] = fit
        panel._sync_active_group_state()
        panel._refresh_model_fit_button_labels()
        _process_events_for(milliseconds=80)
        return panel

    def settle(self, widget: QWidget) -> None:
        _process_events_for(milliseconds=200)
        widget._refresh_plot()
        _wait_until(
            lambda: (
                not widget._trend_curve_compute_active
                and widget._precomputed_trend_curves is not None
            ),
            timeout_ms=20000,
        )
        _process_events_for(milliseconds=200)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _configure_triplet_param_table(global_tab) -> None:
    """Set seed values and Global/File roles on the Keren+Constant param table.

    Δ, ν and the two amplitudes are shared (Global) across the triplet; B_L is
    read per run From File (its 0/5/10 G set value) — GROUND_TRUTH.md §4.
    """
    seeds = {
        "A_1": (SEED_A_SIGNAL, "Global"),
        "Delta": (SEED_DELTA, "Global"),
        "nu": (SEED_NU, "Global"),
        "B_L": (0.0, "File"),
        "A_bg": (SEED_A_BG, "Global"),
    }
    table = global_tab._param_table
    for row in range(table.rowCount()):
        name_item = table.item(row, 0)
        if name_item is None:
            continue
        pname = name_item.data(Qt.ItemDataRole.UserRole) or name_item.text()
        spec = seeds.get(str(pname))
        if spec is None:
            continue
        value, role = spec
        if role != "File":
            table.setItem(row, 1, QTableWidgetItem(f"{value:g}"))
        combo = table.cellWidget(row, 2)
        if isinstance(combo, QComboBox):
            idx = combo.findText(role)
            if idx >= 0:
                combo.setCurrentIndex(idx)


def _fit_nu_of_t():
    """Fit every temperature's Keren triplet; return (T, ν, ν_err) arrays.

    Warm-starts each temperature from the previous fit's globals (the guide's
    "propagate up in temperature" workflow).  All 13 triplets fit in well under
    a second combined, so the whole series is used.
    """
    import numpy as np

    from asymmetry.core.fitting.engine import FitEngine
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    model = _keren_model()
    engine = FitEngine()
    prev = {"A_1": SEED_A_SIGNAL, "Delta": SEED_DELTA, "nu": SEED_NU, "A_bg": SEED_A_BG}
    temps, nus, errs = [], [], []
    for temp in sorted(TRIPLETS):
        datasets = load_corpus_datasets(_triplet_rel_paths(TRIPLETS[temp]))
        init = {
            int(ds.run_number): ParameterSet([
                Parameter("A_1", value=prev["A_1"], min=0.0, max=40.0),
                Parameter("Delta", value=prev["Delta"], min=0.0, max=2.0),
                Parameter("nu", value=prev["nu"], min=0.0, max=8.0),
                Parameter("B_L", value=float(ds.field), fixed=True),
                Parameter("A_bg", value=prev["A_bg"], min=-10.0, max=20.0),
            ])
            for ds in datasets
        }
        results, _fitted = engine.global_fit(
            datasets=datasets,
            model_fn=model.function,
            global_params=["A_1", "Delta", "nu", "A_bg"],
            local_params=["B_L"],
            initial_params=init,
            t_min=0.0,
            t_max=FIT_TMAX,
        )
        r0 = next(iter(results.values()))
        vals = {p.name: p.value for p in r0.parameters}
        unc = r0.uncertainties or {}
        prev = {k: vals[k] for k in ("A_1", "Delta", "nu", "A_bg")}
        temps.append(float(temp))
        nus.append(float(vals["nu"]))
        errs.append(float(unc.get("nu") or 0.02))
    return np.asarray(temps), np.asarray(nus), np.asarray(errs)


def _build_arrhenius_fit(temps, nu, nu_err):
    """Fit ν(T) = a·exp(−E_a/k_BT) + c via the trend minimiser the panel uses."""
    from asymmetry.core.fitting.parameter_models import (
        ModelFitRange,
        ParameterCompositeModel,
        ParameterModelFit,
        fit_parameter_model,
    )
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    model = ParameterCompositeModel(["Arrhenius", "Constant"])
    params = ParameterSet([
        Parameter(name="a", value=400.0, min=0.0, max=1e6),
        Parameter(name="Ea", value=200.0, min=0.0, max=2000.0),  # meV
        Parameter(name="c", value=0.27, min=-1.0, max=5.0),
    ])
    result = fit_parameter_model(
        temps, nu, nu_err, model, params,
        x_min=float(temps.min()), x_max=float(temps.max()),
    )
    if not result.success:
        raise RuntimeError("Al-LLZ ν(T) Arrhenius trend fit did not converge for the screenshot")

    fit_range = ModelFitRange(
        x_min=float(temps.min()),
        x_max=float(temps.max()),
        model=model,
        parameters=result.parameters,
        result=result,
    )
    return ParameterModelFit(
        parameter_name="nu",
        x_key="temperature",
        ranges=[fit_range],
        active=True,
    )


def _wait_until(predicate, *, timeout_ms: int, poll_ms: int = 30) -> None:
    elapsed = 0
    while elapsed < timeout_ms:
        if predicate():
            return
        _process_events_for(milliseconds=poll_ms)
        elapsed += poll_ms


def _pump(milliseconds: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(int(milliseconds), loop.quit)
    loop.exec()
    QApplication.processEvents()


def _pump_until(predicate, timeout_ms: int = 10_000) -> None:
    """Pump a nested event loop until *predicate* holds (or the timeout lapses)."""
    if predicate():
        return
    loop = QEventLoop()
    check = QTimer()
    check.timeout.connect(lambda: loop.quit() if predicate() else None)
    check.start(10)
    guard = QTimer()
    guard.setSingleShot(True)
    guard.timeout.connect(loop.quit)
    guard.start(int(timeout_ms))
    loop.exec()
    check.stop()
    guard.stop()


register(LlzCalibrationScenario())
register(LlzLfTripletScenario())
register(LlzGlobalSetupScenario())
register(LlzGlobalResultScenario())
register(LlzNuArrheniusScenario())

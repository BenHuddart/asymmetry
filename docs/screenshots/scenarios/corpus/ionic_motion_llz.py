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
* ``corpus_llz_nu_arrhenius`` — the fluctuation-rate trend ν(T) rendered as a
  native Arrhenius plot via axis transforms (X→1/T, Y→ln(ν−baseline)) with a
  Linear fit on the activated branch, giving E_a.

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
from ._corpus import CorpusScenario, _process_events_for, load_corpus_datasets, register

EXAMPLE = "Nuclear magnetism and ionic motion/Ionic motion in a solid electrolyte"
_DATA = EXAMPLE + "/Data"

# ZF run number that opens each temperature's triplet; triplet = (zf, zf+1, zf+2)
# = (0 G, 5 G, 10 G).  Setpoint temperatures from GROUND_TRUTH.md §3.
TRIPLETS: dict[int, int] = {
    160: 51341,
    180: 51344,
    200: 51347,
    220: 51350,
    240: 51353,
    264: 51356,
    284: 51359,
    304: 51362,
    324: 51365,
    344: 51368,
    364: 51371,
    384: 51374,
    404: 51377,
}

# Guide seed values at 160 K (GROUND_TRUTH.md §4 — starting values, not results).
SEED_A_SIGNAL = 15.0  # sample-signal amplitude (%)
SEED_A_BG = 5.0  # background amplitude (%)
SEED_DELTA = 0.3  # static field-distribution width Δ (µs⁻¹ / "MHz" seed)
SEED_NU = 0.2  # fluctuation rate ν (MHz)
FIT_TMAX = 12.0  # limit fit window to 12 µs (guide hint)

# Boltzmann constant in eV/K, for the Arrhenius slope → activation energy.
K_B_EV = 8.617333e-5
# Activated branch: fit the Arrhenius line only above this temperature. Below it
# ν sits on the flat plateau (ν ≈ baseline) where ln(ν − baseline) is undefined
# / dominated by noise — see NOTES_llz.md "transform + baseline interplay".
ACTIVATED_MIN_T = 264.0


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
    size = (1220, 760)

    def capture(self, ctx: CaptureContext) -> Path:  # noqa: D401
        # α calibration is inline in the Grouping window's Corrections column
        # (the standalone modal is retired); the shared preview overlays the
        # α = 1 ↔ α̂ before/after with the residual baseline ⟨A⟩.
        from asymmetry.gui.windows.grouping.dialog import GroupingDialog

        dataset = load_corpus_datasets([f"{_DATA}/emu00051315.nxs"])[0]
        dialog = GroupingDialog([dataset], selected_run_number=int(dataset.run_number))
        dialog.resize(*self.size)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        dialog.show()
        _pump(150)

        section = dialog._alpha_section
        section._on_estimate()
        _pump_until(lambda: section._tasks.active_count == 0)
        # Let the shared preview redraw the α = 1 ↔ α̂ overlay before grabbing.
        _pump(500)
        dialog._corrections_scroll.ensureWidgetVisible(section)
        _pump(60)

        pix = dialog.grab()
        out_path = ctx.output_dir / f"{self.name}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not pix.save(str(out_path), "PNG"):
            raise RuntimeError(f"Failed to save screenshot to {out_path}")
        # The applied estimate dirtied the draft: close() would raise the
        # discard-guard modal, which hangs headless — tear down directly.
        dialog._teardown_workers()
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
            initial_params[int(ds.run_number)] = ParameterSet(
                [
                    Parameter("A_1", value=SEED_A_SIGNAL, min=0.0, max=40.0),
                    Parameter("Delta", value=SEED_DELTA, min=0.0, max=2.0),
                    Parameter("nu", value=SEED_NU, min=0.0, max=8.0),
                    Parameter("B_L", value=float(ds.field), fixed=True),
                    Parameter("A_bg", value=SEED_A_BG, min=-10.0, max=20.0),
                ]
            )
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
    """ν(T) fluctuation-rate trend rendered as the native Arrhenius line.

    Axis transforms turn the panel into an Arrhenius plot: X → 1/T (reciprocal)
    and Y → ln(ν − baseline) via a **Custom** transform. The plain ``ln x``
    preset does *not* linearise this trend, because the activated rate sits on a
    ν ≈ 0.27 MHz plateau baseline the log cannot see — see NOTES_llz.md. A Linear
    model fit on the activated branch then has slope −E_a/k_B.
    """

    name = "corpus_llz_nu_arrhenius"
    description = (
        "Al-LLZ fluctuation rate ν(T) as a native Arrhenius plot: X → 1/T "
        "(reciprocal), Y → ln(ν − baseline) (a Custom axis transform), and a "
        "Linear model fit on the activated branch (≥264 K) whose slope gives the "
        "Li⁺ activation energy E_a ≈ 0.22 eV (paper 0.19 eV)."
    )
    example = EXAMPLE
    size = (1240, 780)
    requires_fit = True

    def __init__(self) -> None:
        super().__init__()
        self._summary: dict[str, float] = {}

    def build(self) -> QWidget:
        import numpy as np

        from asymmetry.core.fitting.axis_transforms import AxisTransform
        from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

        temps, nu, nu_err = _fit_nu_of_t()
        # Plateau baseline: the mean ν below the activated onset. The Custom Y
        # transform subtracts it before the log so ln(ν − c) is a straight line
        # in 1/T. (The transform layer has no "subtract fitted baseline" option —
        # the constant is baked into the expression string. API note in NOTES.)
        baseline = float(np.mean(nu[temps < ACTIVATED_MIN_T]))
        fit, self._summary = _build_arrhenius_line_fit(temps, nu, nu_err, baseline)

        y_transform = AxisTransform.custom(f"log(x - {baseline:.6g})")
        x_transform = AxisTransform.preset("reciprocal")

        batch_id = "llz-nu-t"
        row_dicts = [
            {
                "run_number": TRIPLETS[int(t)],
                "run_label": f"{int(t)} K",
                "field": 0.0,
                "temperature": float(t),
                "values": {"nu": float(nu[i])},
                "errors": {"nu": float(nu_err[i])},
                # Only the activated branch pulls the Arrhenius line; the plateau
                # points are shown (where finite) but excluded from the trend.
                "include_in_trend": float(t) >= ACTIVATED_MIN_T,
            }
            for i, t in enumerate(temps)
        ]

        panel = FitParametersPanel()
        panel.load_representation_series(
            [(batch_id, "ν(T) — Al-LLZ", row_dicts)], select_id=batch_id
        )
        # Pin the abscissa to temperature so the reciprocal transform is 1/T
        # (not 1/field, which would be 1/0 → all-NaN).
        idx = panel._x_combo.findData("temperature")
        if idx >= 0:
            panel._x_combo.setCurrentIndex(idx)

        panel._set_axis_transform("y", y_transform)
        panel._set_axis_transform("x", x_transform)
        panel._axis_transform_custom_memory["y"] = y_transform.expression

        panel._model_fits["nu"] = fit
        panel._model_fit_transform_sig["nu"] = panel._transform_signature()
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
            int(ds.run_number): ParameterSet(
                [
                    Parameter("A_1", value=prev["A_1"], min=0.0, max=40.0),
                    Parameter("Delta", value=prev["Delta"], min=0.0, max=2.0),
                    Parameter("nu", value=prev["nu"], min=0.0, max=8.0),
                    Parameter("B_L", value=float(ds.field), fixed=True),
                    Parameter("A_bg", value=prev["A_bg"], min=-10.0, max=20.0),
                ]
            )
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


def _build_arrhenius_line_fit(temps, nu, nu_err, baseline):
    """Fit ``Linear`` to the *transformed* Arrhenius line ln(ν − c) vs 1/T.

    Reproduces the panel's Model-Fit dialog once the axes are set to X→1/T and
    a Custom Y→``log(x - c)``: transform the (T, ν) arrays through the real
    :class:`AxisTransform` machinery — propagating ν's error through the log —
    then fit ``Linear`` on the activated branch (T ≥ ``ACTIVATED_MIN_T``). The
    slope is −E_a/k_B. Returns ``(ParameterModelFit, summary)``.
    """
    import numpy as np

    from asymmetry.core.fitting.axis_transforms import AxisTransform
    from asymmetry.core.fitting.parameter_models import (
        ModelFitRange,
        ParameterCompositeModel,
        ParameterModelFit,
        fit_parameter_model,
    )
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    y_transform = AxisTransform.custom(f"log(x - {baseline:.6g})")
    x_transform = AxisTransform.preset("reciprocal")
    y_vals, y_err = y_transform.apply(np.asarray(nu, float), np.asarray(nu_err, float))
    x_vals, _ = x_transform.apply(np.asarray(temps, float))

    branch = (
        (np.asarray(temps, float) >= ACTIVATED_MIN_T) & np.isfinite(y_vals) & np.isfinite(y_err)
    )
    x_b, y_b, e_b = x_vals[branch], y_vals[branch], y_err[branch]

    model = ParameterCompositeModel(["Linear"])
    seed = ParameterSet(
        [
            Parameter(name="m", value=-2500.0, min=-1.0e6, max=1.0e6),  # −E_a/k_B  [K]
            Parameter(name="b", value=5.0, min=-50.0, max=50.0),
        ]
    )
    result = fit_parameter_model(
        x_b, y_b, e_b, model, seed, x_min=float(x_b.min()), x_max=float(x_b.max())
    )
    if not result.success:
        raise RuntimeError("Al-LLZ ν(T) Arrhenius Linear fit did not converge for the screenshot")

    slope = float(result.parameters["m"].value)
    unc = result.uncertainties or {}
    ea_ev = -slope * K_B_EV
    ea_err_ev = float(unc.get("m", 0.0)) * K_B_EV
    summary = {
        "baseline_MHz": float(baseline),
        "slope_K": slope,
        "Ea_eV": float(ea_ev),
        "Ea_err_eV": float(ea_err_ev),
        "chi2r": float(result.reduced_chi_squared or 0.0),
        "n_branch": int(branch.sum()),
    }
    fit_range = ModelFitRange(
        x_min=float(x_b.min()),
        x_max=float(x_b.max()),
        model=model,
        parameters=result.parameters,
        result=result,
    )
    fit = ParameterModelFit(
        parameter_name="nu", x_key="temperature", ranges=[fit_range], active=True
    )
    return fit, summary


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

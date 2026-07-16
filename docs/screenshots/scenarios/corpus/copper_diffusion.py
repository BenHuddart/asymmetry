"""Corpus scenarios — Muon diffusion and QLCR in copper (Nuclear magnetism & ionic motion).

The classic muon-diffusion teaching example from the WiMDA muon school corpus.
Muons stop at octahedral interstitial sites in fcc Cu and dephase in the static
nuclear-dipolar field of the ⁶³Cu/⁶⁵Cu moments; as temperature changes the muon
hops between sites and the relaxation *motionally narrows*. The famous result is
a **quantum-diffusion hop-rate curve** with a mobility minimum near ~50 K — the
hop rate rises both on warming (thermally activated, over-barrier) and toward the
lowest temperatures (coherent tunnelling; Luke *et al.*, PRB **43**, 3284 (1991)).

Data (``GROUND_TRUTH.md`` §2–3): 74 NeXus ``.nxs`` runs — the 2010 **EMU** set
(20882–20917: 100 G TF triplet, a dense ZF T-scan, a 40 K LF/QLCR field scan) and
the 2024 **ARGUS** set (76924–76961: 20 G TF scan, ZF, a 10–150 G LF/QLCR scan).

Scenarios registered:

* ``corpus_cu_zf_static_kt`` — a base-temperature ARGUS ZF run (76935, 40 K)
  fitted with the static Gaussian Kubo–Toyabe: the textbook KT dip-and-⅓-recovery
  on real copper. Fitted Δ ≈ 0.394 µs⁻¹ (literature anchor 0.38–0.39 µs⁻¹, GT §6).
* ``corpus_cu_zf_quantum_diffusion`` — the guide's Q2/Q3 contrast: EMU ZF 40 K
  (20886, static KT) overlaid with base-T ~5 K (20887). The low-T ⅓ tail relaxes
  instead of staying flat — the departure from static KT that signals low-T
  quantum diffusion (GT §5 Q3).
* ``corpus_cu_tf_abragam`` — one EMU 100 G TF run (20885, 100 K) fitted with the
  **Abragam** relaxation envelope on the precessing signal
  (Oscillatory × Abragam + Constant). Δ ≈ 0.385 µs⁻¹ cross-checks the ZF width;
  ν ≈ 0.27 µs⁻¹ the hop rate (GT §4 TF).
* ``corpus_cu_hop_rate_arrhenius`` — **headline**: the ZF dynamic-KT hop rate
  ν(T) across the EMU ZF series (5–200 K), on a log axis so the two-decade span
  reads. The mobility minimum near ~40–85 K with the low-T rise (quantum
  diffusion) and the thermally activated high-T branch fitted with Arrhenius →
  E_a ≈ 70 meV (GT §4 Arrhenius; §7 program anchor E_a ≈ 62 meV).
* ``corpus_cu_qlcr_scan`` — the LF **quadrupolar level-crossing** field scan by
  integral counting: EMU 40 K LF runs 20888–20900 reduced to integral asymmetry
  vs longitudinal field, showing the QLCR dip near ~78 G (GT §5 Q5; resonance
  field a deliverable).

Every ``requires_fit`` scenario runs a genuine iminuit fit through the same core
:class:`FitEngine` / field-scan machinery the GUI drives. See ``NOTES_copper.md``
for run selection, fitted values vs the GT §6 anchors, and problems hit.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QApplication, QWidget

from .._base import _process_events_for
from ._corpus import CorpusScenario, load_corpus_datasets, register

EXAMPLE = "Nuclear magnetism and ionic motion/Muon diffusion and QLCR in copper"
_DATA = EXAMPLE + "/Data"


def _emu(run: int) -> str:
    return f"{_DATA}/EMU000{run}.nxs"


def _argus(run: int) -> str:
    return f"{_DATA}/ARGUS000{run}.nxs"


# EMU ZF T-scan (GT §3a): run → set temperature (K). Read-T is used where set is
# reliable; 20887 is set 1 K / read 5.8 K (GT §9 read-back artefact) — plotted as
# its set value, the lowest point of the series.
_EMU_ZF_SERIES: dict[int, float] = {
    20887: 5.0,  # set 1 K, read 5.80 K — the low-T quantum-diffusion point
    20886: 40.0,
    20901: 60.0,
    20902: 65.0,
    20903: 70.0,
    20904: 75.0,
    20905: 80.0,
    20906: 85.0,
    20907: 90.0,
    20908: 110.0,
    20909: 120.0,
    20910: 130.0,
    20911: 140.0,
    20912: 150.0,
    20913: 160.0,
    20914: 170.0,
    20915: 180.0,
    20916: 190.0,
    20917: 200.0,
}

# EMU 40 K LF QLCR field scan (GT §3a): densely sampled 75–90 G to catch the dip.
_EMU_QLCR_RUNS = list(range(20888, 20901))  # 40,50,60,70,75,78,80,82,85,90,100,110,120 G

# Static Gaussian Kubo–Toyabe + flat background (ZF nuclear-dipolar KT).
_ZF_STATIC_MODEL = (["StaticGKT_ZF", "Constant"], None)
# Dynamic Gaussian KT + flat background (strong-collision hopping; B_L fixed 0).
_ZF_DYNAMIC_MODEL = (["DynamicGaussianKT", "Constant"], None)
# TF precession damped by the Abragam envelope: A·cos(2πft+φ)·G_Abragam(Δ,ν) + bg.
_TF_ABRAGAM_MODEL = (["Oscillatory", "Abragam", "Constant"], ["*", "+"])

_GAMMA_MU_MHZ_PER_G = 0.0135538  # muon gyromagnetic ratio in MHz/G


def _pump(milliseconds: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(int(milliseconds), loop.quit)
    loop.exec()
    QApplication.processEvents()


def _configure_single_fit(window, components, operators, seeds, bounds, positive):
    """Set up and run one time-domain fit in the main window's single-fit tab.

    Mirrors the pattern the GUI drives: pick the composite model, seed the value
    column, pin lower bounds for the parameters that must stay positive, run, and
    wait. *bounds* maps name → (min, max); *positive* names get Min pinned to 0.
    """
    from asymmetry.core.fitting.composite import CompositeModel
    from asymmetry.gui.panels.fit.tab_base import (
        _param_table_rows_by_name,
        _set_param_table_value,
    )

    single_tab = window._fit_panel._single_tab
    single_tab._set_composite_model(CompositeModel(components, operators=operators))
    _process_events_for(milliseconds=80)

    table = single_tab._param_table
    rows_by_name = _param_table_rows_by_name(table)
    for name, value in seeds.items():
        if name in rows_by_name:
            _set_param_table_value(table, rows_by_name[name], value)
    for name, (lo, hi) in bounds.items():
        if name in rows_by_name:
            row = rows_by_name[name]
            if lo is not None:
                item = table.item(row, table.COL_MIN)
                if item is not None:
                    item.setText(f"{lo:g}")
            if hi is not None:
                item = table.item(row, table.COL_MAX)
                if item is not None:
                    item.setText(f"{hi:g}")
    for name in positive:
        if name in rows_by_name:
            item = table.item(rows_by_name[name], table.COL_MIN)
            if item is not None:
                item.setText("0.0")
    _process_events_for(milliseconds=60)

    single_tab._run_fit()
    single_tab.wait_for_fit()
    _process_events_for(milliseconds=80)
    return single_tab


# --------------------------------------------------------------------------- #
#  1. ZF static Gaussian Kubo–Toyabe — the textbook KT dip on real copper.
# --------------------------------------------------------------------------- #
class CuZfStaticKtScenario(CorpusScenario):
    name = "corpus_cu_zf_static_kt"
    description = (
        "Static Gaussian Kubo–Toyabe fit on the Cu ARGUS 40 K zero-field run "
        "76935: the signature nuclear-dipolar KT dip and ⅓ recovery. Fitted "
        "Δ ≈ 0.394 µs⁻¹ matches the literature 0.38–0.39 µs⁻¹ anchor."
    )
    example = EXAMPLE
    size = (1500, 920)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window._on_fit()
        window.resizeDocks([window._dock_data_browser], [320], Qt.Orientation.Horizontal)

        datasets = load_corpus_datasets([_argus(76935)])
        self.add_to_browser(window, datasets)
        window._on_dataset_selected(datasets[0].run_number)
        _process_events_for(milliseconds=80)

        _configure_single_fit(
            window,
            *_ZF_STATIC_MODEL,
            seeds={"A_1": 21.0, "Delta": 0.4, "A_bg": 5.0},
            bounds={"A_1": (0.0, 40.0), "Delta": (0.0, 1.5), "A_bg": (-10.0, 20.0)},
            positive=("A_1", "Delta"),
        )

        # Frame the first ~13 µs: the KT dip-and-⅓-recovery and the fit overlay
        # both read before the ZF F−B asymmetry error fan (vanishing denominator)
        # swamps the late-time panel.
        x0, x1, y0, y1 = window._plot_panel.get_view_limits()
        window._plot_panel.set_view_limits(0.0, 13.0, y0, y1)
        _process_events_for(milliseconds=80)
        return window


# --------------------------------------------------------------------------- #
#  2. ZF 40 K vs base-T contrast — static KT vs low-T quantum diffusion.
# --------------------------------------------------------------------------- #
class CuZfQuantumDiffusionScenario(CorpusScenario):
    name = "corpus_cu_zf_quantum_diffusion"
    description = (
        "Cu EMU zero-field 40 K (20886) vs base-T ~5 K (20887) overlaid. At 40 K "
        "the muon is static → Kubo–Toyabe with a flat ⅓ tail; at ~5 K the ⅓ tail "
        "relaxes — the departure from static KT that signals low-T quantum "
        "diffusion (guide Q3)."
    )
    example = EXAMPLE
    size = (1500, 900)

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window.resizeDocks([window._dock_data_browser], [320], Qt.Orientation.Horizontal)

        datasets = load_corpus_datasets([_emu(20886), _emu(20887)])
        with window._data_browser.batch_updates():
            for dataset in datasets:
                window._data_browser.add_dataset(dataset)
        run_numbers = [int(ds.run_number) for ds in datasets]
        window._data_browser.create_data_group(run_numbers, name="Cu ZF — 40 K vs ~5 K (EMU)")

        window._plot_panel.set_overlay_enabled(True, emit_signal=True)
        # Bunch the bins ~8× so the late-time tail contrast (5 K relaxing below
        # 40 K) reads through the ZF asymmetry noise instead of being buried in it.
        window._plot_panel.set_bunch_factor(8, emit_signal=True)
        window._data_browser._table.selectAll()
        window._on_dataset_selected(run_numbers[0])
        _process_events_for(milliseconds=120)

        # First ~12 µs: the KT dip is common to both; the contrast is the tail —
        # 40 K stays near its ⅓ plateau while ~5 K relaxes below it. Past ~13 µs
        # the ZF error fan overwhelms the difference.
        window._plot_panel.set_view_limits(0.0, 12.0, -2.0, 24.0)
        _process_events_for(milliseconds=100)
        return window


# --------------------------------------------------------------------------- #
#  3. TF Abragam fit — the precessing signal damped by the Abragam envelope.
# --------------------------------------------------------------------------- #
class CuTfAbragamScenario(CorpusScenario):
    name = "corpus_cu_tf_abragam"
    description = (
        "Abragam relaxation fit on the Cu EMU 100 G transverse-field run 20885 "
        "(100 K): A·cos(2πft+φ) damped by the Abragam envelope. Δ ≈ 0.385 µs⁻¹ "
        "cross-checks the ZF static width; hop rate ν ≈ 0.27 µs⁻¹."
    )
    example = EXAMPLE
    size = (1500, 920)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window._on_fit()
        window.resizeDocks([window._dock_data_browser], [320], Qt.Orientation.Horizontal)

        datasets = load_corpus_datasets([_emu(20885)])
        self.add_to_browser(window, datasets)
        window._on_dataset_selected(datasets[0].run_number)
        _process_events_for(milliseconds=80)

        freq = _GAMMA_MU_MHZ_PER_G * 100.0  # ≈ 1.355 MHz at 100 G
        _configure_single_fit(
            window,
            *_TF_ABRAGAM_MODEL,
            seeds={
                "A_1": 20.0,
                "frequency": freq,
                "phase": 0.2,
                "Delta": 0.38,
                "nu": 0.3,
                "A_bg": 0.0,
            },
            bounds={
                "A_1": (0.0, 40.0),
                "frequency": (freq * 0.8, freq * 1.2),
                "phase": (-2.0, 2.0),
                "Delta": (0.0, 2.0),
                "nu": (0.0, 20.0),
                "A_bg": (-5.0, 5.0),
            },
            positive=("A_1", "Delta", "nu"),
        )

        # Zoom to the first ~6 µs so the ~1.4 MHz precession cycles and the
        # Abragam damping envelope are both resolved.
        x0, x1, y0, y1 = window._plot_panel.get_view_limits()
        window._plot_panel.set_view_limits(0.0, 6.0, y0, y1)
        _process_events_for(milliseconds=80)
        return window


# --------------------------------------------------------------------------- #
#  4. Hop rate ν(T) with the quantum-diffusion minimum + Arrhenius — headline.
# --------------------------------------------------------------------------- #
class CuHopRateArrheniusScenario(CorpusScenario):
    name = "corpus_cu_hop_rate_arrhenius"
    description = (
        "Cu muon hop rate ν(T) from the ZF dynamic Kubo–Toyabe across the EMU "
        "series (5–200 K, log axis). A mobility minimum near ~40–85 K with a "
        "low-T rise (quantum diffusion) and a thermally activated high-T branch "
        "fitted with Arrhenius → E_a ≈ 70 meV."
    )
    example = EXAMPLE
    size = (1240, 800)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

        temps, nu, nu_err = _fit_nu_of_t()
        self._temps, self._nu = temps, nu
        fit = _build_arrhenius_fit(temps, nu, nu_err, x_min=90.0)

        batch_id = "cu-nu-t"
        row_dicts = [
            {
                "run_number": run,
                "run_label": f"{_EMU_ZF_SERIES[run]:.0f} K",
                "field": 0.0,
                "temperature": float(_EMU_ZF_SERIES[run]),
                "values": {"nu": float(nu[i])},
                "errors": {"nu": float(nu_err[i])},
            }
            for i, run in enumerate(sorted(_EMU_ZF_SERIES, key=lambda r: _EMU_ZF_SERIES[r]))
        ]

        panel = FitParametersPanel()
        panel.load_representation_series(
            [(batch_id, "ν(T) — Cu ZF dynamic KT", row_dicts)], select_id=batch_id
        )
        panel._model_fits["nu"] = fit
        panel._sync_active_group_state()
        panel._refresh_model_fit_button_labels()
        _process_events_for(milliseconds=80)
        return panel

    def settle(self, widget: QWidget) -> None:
        _process_events_for(milliseconds=120)
        # ν spans ~0.02 → 2.3 µs⁻¹ (two decades); a log-y axis resolves both the
        # low-T mobility minimum and the activated high-T rise in one frame.
        control = widget._y_controls.get("nu")
        if control is not None and hasattr(control, "log"):
            control.log.setChecked(True)
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
#  5. QLCR field scan — quadrupolar level-crossing by integral counting.
# --------------------------------------------------------------------------- #
class CuQlcrScanScenario(CorpusScenario):
    name = "corpus_cu_qlcr_scan"
    description = (
        "Cu quadrupolar level-crossing resonance by integral counting: the EMU "
        "40 K LF runs 20888–20900 reduced to integral asymmetry vs longitudinal "
        "field. A dip near ~78 G (densely sampled 75–90 G) marks the QLCR."
    )
    example = EXAMPLE
    size = (1500, 900)

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window.resizeDocks([window._dock_data_browser], [340], Qt.Orientation.Horizontal)

        datasets = load_corpus_datasets([_emu(r) for r in _EMU_QLCR_RUNS])
        with window._data_browser.batch_updates():
            for dataset in datasets:
                window._data_browser.add_dataset(dataset)
        window._data_browser.create_data_group(
            [int(ds.run_number) for ds in datasets],
            name="Cu QLCR 40 K — 40–120 G LF (EMU)",
        )
        # Multi-select every run so the fit panel's batch (which the scan build
        # reads) is populated, then enter the integral-scan view and build.
        window._data_browser._table.selectAll()
        _pump(200)
        window._plot_workspace.set_active_view("integral_scan")
        _pump(150)
        window._alc_fit_panel.build_requested.emit()
        _pump(400)
        return window


# --------------------------------------------------------------------------- #
#  Helpers — ZF dynamic-KT batch and the Arrhenius trend fit.
# --------------------------------------------------------------------------- #
def _fit_nu_of_t():
    """Fit every EMU ZF run with the dynamic Gaussian KT; return (T, ν, ν_err).

    Warm-starts each temperature from the previous fit (the guide's "follow the
    relaxation up in temperature" workflow). Δ is free but stays ~0.37 µs⁻¹; the
    hop rate ν is the extracted quantity. B_L is fixed at 0 (zero field).
    """
    from asymmetry.core.fitting.composite import CompositeModel
    from asymmetry.core.fitting.engine import FitEngine
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    model = CompositeModel(*_ZF_DYNAMIC_MODEL)
    engine = FitEngine()
    prev = {"A_1": 20.0, "Delta": 0.38, "nu": 0.05, "A_bg": 0.0}
    temps, nus, errs = [], [], []
    for run in sorted(_EMU_ZF_SERIES, key=lambda r: _EMU_ZF_SERIES[r]):
        dataset = load_corpus_datasets([_emu(run)])[0]
        params = ParameterSet(
            [
                Parameter("A_1", value=prev["A_1"], min=0.0, max=40.0),
                Parameter("Delta", value=prev["Delta"], min=0.1, max=0.8),
                Parameter("nu", value=prev["nu"], min=0.0, max=20.0),
                Parameter("B_L", value=0.0, fixed=True),
                Parameter("A_bg", value=prev["A_bg"], min=-10.0, max=20.0),
            ]
        )
        result = engine.fit(dataset, model.function, params)
        vals = {p.name: p.value for p in result.parameters}
        unc = result.uncertainties or {}
        prev = {k: vals[k] for k in ("A_1", "Delta", "nu", "A_bg")}
        temps.append(float(_EMU_ZF_SERIES[run]))
        nus.append(abs(float(vals["nu"])))
        errs.append(float(unc.get("nu") or max(abs(vals["nu"]) * 0.05, 0.005)))
    return np.asarray(temps), np.asarray(nus), np.asarray(errs)


def _build_arrhenius_fit(temps, nu, nu_err, *, x_min):
    """Fit the activated high-T branch ν(T) = a·exp(−E_a/k_BT) + c above *x_min*.

    Only the thermally activated (over-barrier) branch is Arrhenius; the low-T
    quantum-diffusion rise is left out of the fit window (guide: Arrhenius above
    ~100 K). Returns a ``ParameterModelFit`` for the trend panel to overlay.
    """
    from asymmetry.core.fitting.parameter_models import (
        ModelFitRange,
        ParameterCompositeModel,
        ParameterModelFit,
        fit_parameter_model,
    )
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    model = ParameterCompositeModel(["Arrhenius", "Constant"])
    params = ParameterSet(
        [
            Parameter(name="a", value=500.0, min=0.0, max=1e8),
            Parameter(name="Ea", value=80.0, min=0.0, max=2000.0),  # meV
            Parameter(name="c", value=0.02, min=-1.0, max=5.0),
        ]
    )
    x_max = float(temps.max())
    result = fit_parameter_model(temps, nu, nu_err, model, params, x_min=x_min, x_max=x_max)
    if not result.success:
        raise RuntimeError("Cu ν(T) Arrhenius trend fit did not converge for the screenshot")

    fit_range = ModelFitRange(
        x_min=x_min,
        x_max=x_max,
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


register(CuZfStaticKtScenario())
register(CuZfQuantumDiffusionScenario())
register(CuTfAbragamScenario())
register(CuHopRateArrheniusScenario())
register(CuQlcrScanScenario())

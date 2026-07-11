"""Corpus scenarios — Magnetic ordering in EuO (Magnetism).

Drives the Asymmetry GUI through the WiMDA muon-school EuO example on the
**real PSI GPS ``.bin`` corpus files** (``deltat_pta_gps_2923–2973``, runs
2923–2960 zero-field, a temperature scan through the Curie point). The paper is
the spec: S. J. Blundell *et al.*, "Phase transition in the localized
ferromagnet EuO probed by µSR," Phys. Rev. B **81**, 092407 (2010). See the
example's ``GROUND_TRUTH.md``.

These are the *real-data* counterparts to the synthetic ``euo_fit_oscillatory``
/ ``temperature_trend_fit`` scenarios: below T_C ≈ 69 K a single spontaneous
zero-field precession develops in the transverse Forward/Back detector pair;
its frequency ν(T) is the magnetic order parameter, ν(0) ≈ 30 MHz
(B_µ(0) ≈ 0.22 T). Fitting ν(T) with the phenomenological order-parameter
function ν(T) = ν(0)·[1 − (T/T_C)^α]^β recovers the paper's full-range
numbers (α ≈ 1.5, β ≈ 0.4, T_C ≈ 69 K); the *critical* β = 0.32(1) needs a
log–log restriction the trending panel does not do (GROUND_TRUTH §4/§9).

Scenarios registered:

* ``corpus_euo_load_browse``   — PSI ``.bin`` T-scan in the data browser.
* ``corpus_euo_zf_fit``        — converged ZF oscillation fit on the 1.6 K run.
* ``corpus_euo_fft``           — frequency-domain single precession line (~30 MHz).
* ``corpus_euo_nu_t_trend``    — headline: ν(T) order-parameter trend + fit.
* ``corpus_euo_waterfall``     — ZF spectra stacked across the transition.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from ._corpus import CorpusScenario, _process_events_for, load_corpus_datasets, register

EXAMPLE = "Magnetism/Magnetic ordering in EuO"
_DATA = "Magnetism/Magnetic ordering in EuO/data/deltat_pta_gps_%d.bin"


def _rel(run: int) -> str:
    return _DATA % run


# Zero-field temperature scan, ordered by *measured* sample temperature
# (logbook.rtf, GROUND_TRUTH §3). Runs 2923/2924 dropped (bad thermometry /
# low statistics, §3 note); TF-60 G runs 2961–2973 excluded (paramagnetic, not
# order-parameter data). Values are the sample-thermometer readings.
_ZF_SCAN: list[tuple[int, float]] = [
    (2960, 1.60),
    (2925, 10.05),
    (2928, 17.18),
    (2929, 24.16),
    (2930, 30.14),
    (2931, 36.31),
    (2932, 41.28),
    (2933, 46.23),
    (2934, 50.25),
    (2935, 52.76),
    (2936, 57.78),
    (2937, 61.32),
    (2938, 63.36),
    (2939, 64.86),
    (2940, 65.87),
    (2941, 66.88),
    (2942, 67.90),
    (2943, 68.69),
]

# Runs shown as a coarse waterfall across the transition (base → just below T_C).
_WATERFALL_RUNS = [2960, 2930, 2934, 2937, 2940, 2943]


def _fit_zf_frequency(dataset, nu_seed: float):
    """Fit ``Oscillatory*Exponential + Constant`` to one ZF run via the core engine.

    Returns ``(nu_MHz, nu_err, lambda, fit_result)``. The transverse F/B
    asymmetry the loader builds carries the spontaneous precession on a large
    uncalibrated (α = 1) baseline, absorbed by the additive ``Constant``. Seeding
    the frequency near the expected ν(T) and warm-starting downward as T → T_C
    keeps every fit in the correct minimum (a low seed collapses to a spurious
    low-amplitude solution).
    """
    from asymmetry.core.fitting.composite import CompositeModel
    from asymmetry.core.fitting.engine import FitEngine
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    model = CompositeModel(["Oscillatory", "Exponential", "Constant"], operators=["*", "+"])
    params = ParameterSet(
        [
            Parameter("A_1", 8.0),
            Parameter("frequency", nu_seed, min=0.0, max=40.0),
            Parameter("phase", 0.0),
            Parameter("Lambda", 1.0, min=0.0),
            Parameter("A_bg", float(np.nanmean(dataset.asymmetry))),
        ]
    )
    result = FitEngine().fit(dataset, model.function, params, t_min=0.0, t_max=6.0)
    nu = abs(result.parameters["frequency"].value)
    err = float(result.uncertainties.get("frequency", 0.1)) or 0.1
    lam = abs(result.parameters["Lambda"].value)
    return nu, err, lam, result


def _wait_until(predicate, *, timeout_ms: int, poll_ms: int = 30) -> None:
    elapsed = 0
    while elapsed < timeout_ms:
        if predicate():
            return
        _process_events_for(milliseconds=poll_ms)
        elapsed += poll_ms


# ---------------------------------------------------------------------------
# 1. Load + browse the PSI .bin temperature scan
# ---------------------------------------------------------------------------
class EuoLoadBrowseScenario(CorpusScenario):
    name = "corpus_euo_load_browse"
    description = (
        "PSI GPS .bin EuO zero-field temperature scan loaded into the data "
        "browser (real deltat_pta_gps loader, run numbers + temperatures)."
    )
    example = EXAMPLE
    size = (1400, 860)

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window.resizeDocks([window._dock_data_browser], [430], Qt.Orientation.Horizontal)

        # A representative spread through the transition (base → just above T_C):
        # enough rows to read the scan, few enough to stay legible when cropped.
        runs = [2960, 2925, 2929, 2930, 2933, 2936, 2937, 2940, 2943, 2946, 2958, 2959]
        datasets = load_corpus_datasets([_rel(r) for r in runs])
        with window._data_browser.batch_updates():
            for dataset in datasets:
                window._data_browser.add_dataset(dataset)

        window._data_browser.create_data_group(
            [int(ds.run_number) for ds in datasets],
            name="EuO ZF T-scan — GPS@PSI",
        )
        # Select the base-temperature run so the plot shows a real ordered
        # (oscillating) ZF spectrum next to the browser.
        window._on_dataset_selected(datasets[0].run_number)
        _process_events_for(milliseconds=80)
        window._plot_panel.set_view_limits(0.0, 0.6, *window._plot_panel.get_view_limits()[2:])
        _process_events_for(milliseconds=60)
        return window


# ---------------------------------------------------------------------------
# 2. Converged ZF oscillation fit on the base-temperature run
# ---------------------------------------------------------------------------
class EuoZfFitScenario(CorpusScenario):
    name = "corpus_euo_zf_fit"
    description = (
        "Converged Oscillatory*Exponential+Constant fit on the EuO 1.6 K "
        "zero-field run (ν ≈ 30 MHz), zoomed so the precession resolves."
    )
    example = EXAMPLE
    size = (1500, 920)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.core.fitting.composite import CompositeModel
        from asymmetry.gui.mainwindow import MainWindow
        from asymmetry.gui.panels.fit.tab_base import (
            _param_table_rows_by_name,
            _set_param_table_value,
        )

        window = MainWindow()
        window._on_fit()
        window.resizeDocks([window._dock_data_browser], [320], Qt.Orientation.Horizontal)

        # Run 2960 — the lowest temperature (1.6 K), deep in the ordered phase.
        datasets = load_corpus_datasets([_rel(2960)])
        for dataset in datasets:
            window._data_browser.add_dataset(dataset)
        window._on_dataset_selected(datasets[0].run_number)

        single_tab = window._fit_panel._single_tab
        single_tab._set_composite_model(
            CompositeModel(["Oscillatory", "Exponential", "Constant"], operators=["*", "+"])
        )
        _process_events_for(milliseconds=80)

        # Seed near the expected order parameter at base T: ν(0) ≈ 30 MHz
        # (GROUND_TRUTH §6), heavy damping (λ ≈ 3 µs⁻¹) and the large α = 1
        # baseline (~28 %) that the additive Constant absorbs.
        param_table = single_tab._param_table
        rows = _param_table_rows_by_name(param_table)
        seeds = {
            "A_1": 6.0,
            "frequency": 30.0,
            "phase": 0.0,
            "Lambda": 3.0,
            "A_bg": float(np.nanmean(datasets[0].asymmetry)),
        }
        for name, value in seeds.items():
            if name in rows:
                _set_param_table_value(param_table, rows[name], value)
        _process_events_for(milliseconds=40)

        single_tab._run_fit()
        single_tab.wait_for_fit()

        # ν ≈ 30 MHz → ~0.033 µs period; the full 6 µs range compresses ~190
        # cycles into a solid block. Zoom to the first 0.45 µs (~13 cycles),
        # where the damped oscillation lives, and frame Y to that window so the
        # precession sits large on screen rather than as ripple on the baseline.
        window._plot_panel.set_view_limits(0.0, 0.45, *window._plot_panel.get_view_limits()[2:])
        _process_events_for(milliseconds=60)
        self._frame_y_to_window(window, 0.0, 0.45)
        _process_events_for(milliseconds=80)
        return window

    @staticmethod
    def _frame_y_to_window(window, x_min: float, x_max: float) -> None:
        ds = window._plot_panel
        t = getattr(ds, "_last_plot_time", None)
        a = getattr(ds, "_last_plot_asymmetry", None)
        if t is None or a is None or not len(t):
            return
        t = np.asarray(t, dtype=float)
        a = np.asarray(a, dtype=float)
        m = (t >= x_min) & (t <= x_max)
        if not np.any(m):
            return
        lo, hi = float(np.nanmin(a[m])), float(np.nanmax(a[m]))
        pad = 0.12 * (hi - lo or 1.0)
        window._plot_panel.set_view_limits(x_min, x_max, lo - pad, hi + pad)


# ---------------------------------------------------------------------------
# 3. Frequency-domain view — the single precession line at base T
# ---------------------------------------------------------------------------
class EuoFftScenario(CorpusScenario):
    name = "corpus_euo_fft"
    description = (
        "Frequency-domain view of the EuO 1.6 K ZF run: a single precession "
        "line near 30 MHz (Blundell 2010 Fig. 1(c))."
    )
    example = EXAMPLE
    size = (1500, 920)

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window._on_fourier()
        window.resizeDocks([window._dock_data_browser], [320], Qt.Orientation.Horizontal)

        datasets = load_corpus_datasets([_rel(2960)])
        for dataset in datasets:
            window._data_browser.add_dataset(dataset)
        window._on_dataset_selected(datasets[0].run_number)
        _process_events_for(milliseconds=120)

        window._on_domain_button_clicked("frequency")
        _process_events_for(milliseconds=80)

        # The averaged grouped-FFT the panel computes is, over all five GPS
        # detectors, dominated by each detector's slowly-varying (lifetime-
        # corrected) baseline — low-frequency power that buries the small
        # precession line. Restrict to the transverse Forward/Back pair (the
        # only groups carrying the ZF spontaneous precession) and apply a
        # Lorentzian apodisation (τ = 0.5 µs, matched to the ~0.3 µs coherence
        # of the heavily damped signal) so the single line at ν ≈ 30 MHz stands
        # clear — the frequency-domain analogue of Blundell 2010 Fig. 1(c).
        fp = window._fourier_panel
        fp._filter_lorentzian_radio.setChecked(True)
        fp._filter_time_constant_edit.setText("0.5")
        fp.set_group_enabled({1: True, 2: True, 3: False, 4: False, 5: False})
        _process_events_for(milliseconds=40)

        freq_panel = window._frequency_plot_panel
        # Frame onto the precession line and off the residual low-frequency
        # skirt: ν ≈ 30 MHz at base T.
        x_min, x_max = 20.0, 42.0
        window._on_compute_fourier()

        spectrum_x = spectrum_y = None
        for _ in range(100):  # bounded ~10 s
            _process_events_for(milliseconds=100)
            x = freq_panel._last_plot_time
            y = freq_panel._last_plot_asymmetry
            if x is not None and y is not None and len(x) and float(np.nanmax(x)) >= x_max:
                spectrum_x = np.asarray(x, dtype=float)
                spectrum_y = np.asarray(y, dtype=float)
                break
        if spectrum_x is None:
            raise RuntimeError("EuO Fourier recompute did not render within 10 s")

        in_window = (spectrum_x >= x_min) & (spectrum_x <= x_max)
        peak = float(np.max(spectrum_y[in_window])) if np.any(in_window) else 1.0
        freq_panel.set_view_limits(x_min, x_max, -0.05 * peak, 1.12 * peak)
        _process_events_for(milliseconds=120)
        return window


# ---------------------------------------------------------------------------
# 4. Headline — ν(T) order-parameter trend with the fitted power law
# ---------------------------------------------------------------------------
class EuoNuTTrendScenario(CorpusScenario):
    name = "corpus_euo_nu_t_trend"
    description = (
        "EuO order parameter: spontaneous ZF precession frequency ν(T) from "
        "the real corpus runs, with the fitted OrderParameter power law "
        "ν(T)=ν0·[1−(T/Tc)^α]^β (Tc ≈ 69 K)."
    )
    example = EXAMPLE
    size = (1240, 760)
    requires_fit = True  # real per-run ZF fits + the OrderParameter trend fit

    def __init__(self) -> None:
        super().__init__()
        self._fit_summary: dict[str, float] = {}

    def build(self) -> QWidget:
        from asymmetry.core.fitting.parameter_models import (
            ModelFitRange,
            ParameterCompositeModel,
            ParameterModelFit,
            fit_parameter_model,
            suggest_trend_seeds,
        )
        from asymmetry.core.fitting.parameters import Parameter, ParameterSet
        from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

        # Fit each ZF run's spontaneous frequency in ascending-temperature order,
        # warm-starting ν downward from ν(0) ≈ 30 MHz so every run lands in the
        # correct minimum (GROUND_TRUTH §4 workflow).
        temps: list[float] = []
        nus: list[float] = []
        errs: list[float] = []
        nu_seed = 30.0
        for run, temp in _ZF_SCAN:
            ds = load_corpus_datasets([_rel(run)])[0]
            nu, err, _lam, _res = _fit_zf_frequency(ds, nu_seed)
            temps.append(temp)
            nus.append(nu)
            errs.append(err)
            nu_seed = max(nu * 0.9, 2.0)
        temperature = np.array(temps)
        nu = np.array(nus)
        nu_err = np.array(errs)

        batch_id = "euo-nu-t-corpus"
        row_dicts = [
            {
                "run_number": run,
                "run_label": f"{temperature[i]:.1f} K",
                "field": 0.0,
                "temperature": float(temperature[i]),
                "values": {"frequency": float(nu[i])},
                "errors": {"frequency": float(nu_err[i])},
            }
            for i, (run, _t) in enumerate(_ZF_SCAN)
        ]

        panel = FitParametersPanel()
        panel.load_representation_series(
            [(batch_id, "ν(T) — EuO (corpus)", row_dicts)],
            select_id=batch_id,
        )

        # Full-range phenomenological OrderParameter fit — the machinery the
        # panel's Model Fit dialog runs.
        model = ParameterCompositeModel(["OrderParameter"])
        seeds = suggest_trend_seeds(model, temperature, nu)
        params = ParameterSet(
            [
                Parameter(
                    name=p,
                    value=float(seeds.get(p, model.param_defaults[p])),
                )
                for p in model.param_names
            ]
        )
        result = fit_parameter_model(
            temperature,
            nu,
            nu_err,
            model,
            params,
            x_min=float(temperature.min()),
            x_max=float(temperature.max()),
        )
        if not result.success:
            raise RuntimeError("EuO OrderParameter ν(T) fit did not converge")
        self._fit_summary = {n: float(result.parameters[n].value) for n in model.param_names}

        fit_range = ModelFitRange(
            x_min=float(temperature.min()),
            x_max=float(temperature.max()),
            model=model,
            parameters=result.parameters,
            result=result,
        )
        panel._model_fits["frequency"] = ParameterModelFit(
            parameter_name="frequency",
            x_key="temperature",
            ranges=[fit_range],
            active=True,
        )
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


# ---------------------------------------------------------------------------
# 5. Waterfall — ZF spectra stacked across the transition (optional/distinctive)
# ---------------------------------------------------------------------------
class EuoWaterfallScenario(CorpusScenario):
    name = "corpus_euo_waterfall"
    description = (
        "Waterfall of EuO zero-field spectra across the transition (1.6 → "
        "68.7 K): the precession slows as ν(T) collapses toward T_C."
    )
    example = EXAMPLE
    size = (1500, 920)

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window.resizeDocks([window._dock_data_browser], [320], Qt.Orientation.Horizontal)

        datasets = load_corpus_datasets([_rel(r) for r in _WATERFALL_RUNS])
        with window._data_browser.batch_updates():
            for dataset in datasets:
                window._data_browser.add_dataset(dataset)
        run_numbers = [int(ds.run_number) for ds in datasets]
        window._data_browser.create_data_group(run_numbers, name="EuO ZF across T_C")

        window._plot_panel.set_overlay_enabled(True, emit_signal=True)
        window._plot_panel.set_waterfall_enabled(True, emit_signal=True)
        window._data_browser._table.selectAll()
        window._on_dataset_selected(run_numbers[0])
        _process_events_for(milliseconds=80)
        # Zoom into the first ~0.6 µs so the slowing precession is legible in
        # each stacked trace rather than a dense band.
        window._plot_panel.set_view_limits(0.0, 0.6, *window._plot_panel.get_view_limits()[2:])
        _process_events_for(milliseconds=60)
        return window


register(EuoLoadBrowseScenario())
register(EuoZfFitScenario())
register(EuoFftScenario())
register(EuoNuTTrendScenario())
register(EuoWaterfallScenario())

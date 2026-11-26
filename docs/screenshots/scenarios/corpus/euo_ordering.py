"""Corpus scenarios — Magnetic ordering in EuO (Magnetism).

Drives the Asymmetry GUI through the WiMDA muon-school EuO example on the
**real PSI GPS ``.bin`` corpus files** (``deltat_pta_gps_2923–2973``, runs
2923–2960 zero-field, a temperature scan through the Curie point). The paper is
the spec: S. J. Blundell *et al.*, "Phase transition in the localized
ferromagnet EuO probed by µSR," Phys. Rev. B **81**, 092407 (2010). See the
example's ``GROUND_TRUTH.md``.

The zero-field Forward/Backward asymmetry resolves spontaneous precession
below the transition. The trend uses the logged sample thermometer and the
18 ordered spectra selected for the walkthrough. Its fitted exponent
describes the full selected range. The published critical exponent comes
from Fig. 3(a) of Blundell et al.

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

from examples.euo_order_parameter import EUO_RUNS

from ._corpus import CorpusScenario, _process_events_for, load_corpus_datasets, register

EXAMPLE = "Magnetism/Magnetic ordering in EuO"
_DATA = "Magnetism/Magnetic ordering in EuO/data/deltat_pta_gps_%d.bin"


def _rel(run: int) -> str:
    return _DATA % run


_ZF_FIT_RANGE_US = (0.0, 6.0)

# Runs shown as a coarse waterfall across the transition (base → just below T_C).
_WATERFALL_RUNS = [2960, 2930, 2934, 2937, 2940, 2943]


def _fit_zf_frequency(dataset, nu_seed: float):
    """Fit a ZF damped cosine and a constant detector offset."""
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
    return FitEngine().fit(
        dataset, model.function, params, t_min=_ZF_FIT_RANGE_US[0], t_max=_ZF_FIT_RANGE_US[1]
    )


def _wait_until(predicate, *, timeout_ms: int, poll_ms: int = 30) -> None:
    elapsed = 0
    while elapsed < timeout_ms:
        if predicate():
            return
        _process_events_for(milliseconds=poll_ms)
        elapsed += poll_ms


def _show_logged_temperature(window) -> None:
    """Show the same logged temperature column used by the EuO trend."""
    window._use_temperature_from_log_action.setChecked(True)
    browser = window._data_browser
    header = browser._table.horizontalHeader()
    temperature = browser._COLUMNS.index("T (K)")
    header.moveSection(header.visualIndex(temperature), 1)


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
        _show_logged_temperature(window)
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
        _show_logged_temperature(window)
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

        window._plot_panel.set_fit_range(*_ZF_FIT_RANGE_US)
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
        # precession line. Transform the **F−B asymmetry** itself instead (the
        # #265 "Signal source" radio): the forward−backward difference cancels
        # the common baseline, so the single ZF precession line stands ~5× clearer
        # than the restricted-pair grouped average did (PR #265 measured peak/floor
        # 7.27 vs 1.50 on this exact run). A Lorentzian apodisation (τ = 0.5 µs,
        # matched to the ~0.3 µs coherence of the heavily damped signal) sharpens
        # the line at ν ≈ 30 MHz — the frequency-domain analogue of Blundell 2010
        # Fig. 1(c). (In F−B mode the per-group include table is inert.)
        fp = window._fourier_panel
        fp._filter_lorentzian_radio.setChecked(True)
        fp._filter_time_constant_edit.setText("0.5")
        fp._signal_fb_radio.setChecked(True)
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
        "EuO spontaneous ZF precession frequency versus logged sample temperature. "
        "The phenomenological fit to the 18 ordered runs gives Tc = 70.5 K. "
        "Blundell et al. obtain 69.05(1) K in their separate critical-regime fit."
    )
    example = EXAMPLE
    size = (1240, 760)
    requires_fit = True  # real per-run ZF fits + the OrderParameter trend fit

    def build(self) -> QWidget:
        from asymmetry.gui.panels.data_browser import DataBrowserPanel

        browser = DataBrowserPanel()
        browser.set_use_temperature_from_log(True)
        rows = []
        nu_seed = 30.0
        with browser.batch_updates():
            for run in EUO_RUNS:
                dataset = load_corpus_datasets([_rel(run)])[0]
                browser.add_dataset(dataset.fit_record())
                temperature = browser.displayed_coordinate(run)["temperature"]
                if temperature is None or not browser._temperature_uses_log_for_display(dataset):
                    raise RuntimeError(f"Missing GPS sample thermometer for EuO run {run}")
                result = _fit_zf_frequency(dataset, nu_seed)
                nu = result.parameters["frequency"].value
                error = result.uncertainties.get("frequency")
                if not result.success or error is None or not np.isfinite(error) or error <= 0:
                    raise RuntimeError(f"EuO run {run} has no usable fitted frequency uncertainty")
                rows.append(
                    {
                        "run_number": run,
                        "run_label": dataset.run_label,
                        "field": 0.0,
                        "temperature": temperature,
                        "values": {"frequency": nu},
                        "errors": {"frequency": error},
                        "model_name": "Oscillatory * Exponential + Constant",
                        "chi_squared": result.chi_squared,
                        "reduced_chi_squared": result.reduced_chi_squared,
                    }
                )
                nu_seed = max(nu * 0.9, 2.0)
        browser.deleteLater()
        return self.build_trend_panel(rows)

    @staticmethod
    def build_trend_panel(rows: list[dict]) -> QWidget:
        """Attach the fitted curve to the same Fit Parameters rows used for TSV export."""
        from asymmetry.core.fitting.parameter_models import (
            ModelFitRange,
            ParameterCompositeModel,
            ParameterModelFit,
            fit_parameter_model,
        )
        from asymmetry.core.fitting.parameters import Parameter, ParameterSet
        from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

        batch_id = "euo-nu-t-corpus"
        panel = FitParametersPanel()
        panel.load_representation_series(
            [(batch_id, "ν(T) — EuO ZF (logged sample temperature)", rows)],
            select_id=batch_id,
        )
        panel._x_combo.setCurrentText("𝑇 (K)")
        x_key = panel._effective_x_key()
        selected_rows = panel._included_trend_rows(x_key)
        temperature = np.array([panel._x_value(row, x_key) for row in selected_rows])
        nu, nu_err = panel._series_y_arrays(selected_rows, "frequency")
        t_min, t_max = float(temperature.min()), float(temperature.max())
        model = ParameterCompositeModel(["OrderParameter"])
        params = ParameterSet(
            [
                Parameter("y0", 30.0, min=0.0),
                Parameter("Tc", 70.0, min=t_max),
                Parameter("beta", 0.4, min=0.01),
                Parameter("alpha", 1.5, min=0.01),
            ]
        )
        result = fit_parameter_model(
            temperature, nu, nu_err, model, params, x_min=t_min, x_max=t_max
        )
        if not result.success:
            raise RuntimeError("EuO OrderParameter ν(T) fit did not converge")

        fit_range = ModelFitRange(
            x_min=t_min,
            x_max=t_max,
            model=model,
            parameters=result.parameters,
            result=result,
        )
        panel._model_fits["frequency"] = ParameterModelFit(
            parameter_name="frequency",
            x_key=x_key,
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

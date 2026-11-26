"""Corpus scenarios — Ferromagnetic nickel (Magnetism).

Drives the Asymmetry GUI through the WiMDA muon-school **Ferromagnetic nickel**
example on the real **EMU/ISIS NeXus-v2 HDF5** corpus files
(``emu00124218.nxs`` … ``emu00124278.nxs``, runs 124218–124278). The guide is
the spec (``GROUND_TRUTH.md``); the physics reference is Foy, Heiman, Kossler &
Stronach, "Precession of Positive Muons in Nickel and Iron," Phys. Rev. Lett.
**30**, 1064 (1973).

The spontaneous ZF precession softens towards the Curie temperature. The
furnace controller readings are interpreted as Celsius for the trend. EMU
resolves the precession over a limited interval below the transition because
the source pulse averages out the faster low-temperature oscillation.

Scenarios registered:

* ``corpus_ni_zf_precession_fit`` — the ferromagnet money shot: a converged
  spontaneous ZF precession fit (Oscillatory×Exponential+Constant) on the
  618 K (controller 345 °C) run 124232, **no applied field**, ν ≈ 6.1 MHz —
  matching the 1973 Fig. 2 curve (~5.8 MHz at ~615 K).
* ``corpus_ni_nu_t_order_parameter`` — headline: real per-run ZF fits →
  ν(T) in kelvin with the fitted OrderParameter power law ν=y0·(1−T/Tc)^β;
  shows a range-sensitive effective exponent and T_C near 631 K.
* ``corpus_ni_zf_fft`` — the frequency-domain money shot: the FFT of the 618 K
  ZF run showing a single spontaneous-precession line at ν ≈ 6.1 MHz, inside
  EMU's pulsed-source band (contrast guide Q7/Q9).
* ``corpus_ni_lf_decoupling`` — the 473 K (controller 200 °C) longitudinal-field
  scan (1200 → 4000 G): the recovered (decoupled) asymmetry rises with field as
  the static internal field is decoupled (guide Q8 — internal field after the
  oscillations are gone).
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from asymmetry.core.utils.constants import CELSIUS_ZERO_K

from .._base import _process_events_for
from ._corpus import CorpusScenario, load_corpus_datasets, register

EXAMPLE = "Magnetism/Ferromagnetic nickel"
_DATA = "Magnetism/Ferromagnetic nickel/Data/emu00%d.nxs"

# ZF spontaneous precession = Oscillatory × Exponential relaxation + baseline.
# The EMU asymmetry the loader builds is uncalibrated (α = 1); the large
# additive baseline (~30) is absorbed by the Constant, exactly as the guide's
# extra-relaxation discussion (§5 Q4) anticipates.
_ZF_MODEL = (["Oscillatory", "Exponential", "Constant"], ["*", "+"])
_ZF_FIT_RANGE_US = (0.0, 4.5)

# The in-band order-parameter branch (GROUND_TRUTH §3): runs where the internal
# field has softened enough that ν sits inside EMU's ≲10 MHz band and the
# spontaneous precession is cleanly resolved. Below controller 320 °C (593 K)
# the frequency runs out of band (guide Q7); above 356 °C (629 K) the amplitude
# collapses (approaching T_C = 631 K).
_ZF_OP_RUNS = list(range(124227, 124237))  # controller 320…356 °C = 593…629 K

# The money-shot run: highest spontaneous-precession amplitude in the resolved
# window, ν ≈ 6.1 MHz at 618 K (controller 345 °C).
_ZF_FIT_RUN = 124232

# LF decoupling field scan at fixed 473 K (controller 200 °C; GROUND_TRUTH §3):
# runs 124272–124278 (3600 → 1200 G) plus the 4000 G run 124270, low → high
# field for the recovery curve. The 124271 repeat of 4000 G is dropped to keep
# one point per field.
_LF_RUNS = [124278, 124277, 124276, 124275, 124274, 124273, 124272, 124270]


def _process(ms: int = 80) -> None:
    _process_events_for(milliseconds=ms)


def _fit_zf_frequency(dataset, chained_seed: float, amp_seed: float):
    """Resolve the precession using spectral and chained frequency seeds."""
    from asymmetry.core.fitting.composite import CompositeModel
    from asymmetry.core.fitting.engine import FitEngine
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet
    from asymmetry.core.fourier.fft import fft_asymmetry

    model = CompositeModel(_ZF_MODEL[0], operators=_ZF_MODEL[1])
    bg = float(np.nanmean(dataset.asymmetry))
    t_min, t_max = _ZF_FIT_RANGE_US
    frequency, _, magnitude = fft_asymmetry(dataset, t_min=t_min, t_max=t_max, padding_factor=4)
    # Frequencies above 10 MHz are attenuated by the EMU source pulse.
    in_band = (frequency >= 2.0) & (frequency <= 10.0)
    spectral_seed = frequency[in_band][np.argmax(magnitude[in_band])]
    engine = FitEngine()
    resolved = []
    for nu_seed, amplitude in ((chained_seed, amp_seed), (spectral_seed, 10.0)):
        params = ParameterSet(
            [
                Parameter("A_1", amplitude, min=0.0),
                Parameter("frequency", nu_seed, min=0.0, max=15.0),
                Parameter("phase", 0.0),
                Parameter("Lambda", 0.5, min=0.0),
                Parameter("A_bg", bg),
            ]
        )
        result = engine.fit(dataset, model.function, params, t_min=t_min, t_max=t_max)
        amp = result.parameters["A_1"].value
        amp_error = result.uncertainties.get("A_1", np.inf)
        error = result.uncertainties.get("frequency", np.nan)
        if result.success and amp > 3.0 * amp_error and np.isfinite(error) and error > 0.0:
            resolved.append(result)
    if not resolved:
        raise RuntimeError(f"Ni run {dataset.run_number} has no resolved precession fit")
    return min(resolved, key=lambda result: result.chi_squared)


def _wait_until(predicate, *, timeout_ms: int, poll_ms: int = 30) -> None:
    elapsed = 0
    while elapsed < timeout_ms:
        if predicate():
            return
        _process_events_for(milliseconds=poll_ms)
        elapsed += poll_ms


# ---------------------------------------------------------------------------
# 1. Money shot — converged spontaneous ZF precession fit (no applied field)
# ---------------------------------------------------------------------------
class NiZfPrecessionFitScenario(CorpusScenario):
    name = "corpus_ni_zf_precession_fit"
    description = (
        "Converged Oscillatory×Exponential+Constant fit on the Ni 618 K "
        "(controller 345 °C) zero-field run 124232: a spontaneous "
        "internal-field precession (ν ≈ 6.1 MHz) with NO applied field, "
        "matching the 1973 PRL Fig. 2 curve at 0.98 T_C."
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

        datasets = load_corpus_datasets([_DATA % _ZF_FIT_RUN])
        self.add_to_browser(window, datasets)
        window._on_dataset_selected(datasets[0].run_number)
        _process(80)

        single_tab = window._fit_panel._single_tab
        single_tab._set_composite_model(CompositeModel(_ZF_MODEL[0], operators=_ZF_MODEL[1]))
        _process(80)

        table = single_tab._param_table
        rows = _param_table_rows_by_name(table)
        seeds = {
            "A_1": 10.0,
            "frequency": 6.1,
            "phase": 0.0,
            "Lambda": 0.5,
            "A_bg": float(np.nanmean(datasets[0].asymmetry)),
        }
        for name, value in seeds.items():
            if name in rows:
                _set_param_table_value(table, rows[name], value)
        # Pin ν, amplitude and the relaxation rate non-negative so the fit does
        # not settle in a sign-degenerate mirror minimum.
        for name in ("A_1", "frequency", "Lambda"):
            if name in rows:
                item = table.item(rows[name], table.COL_MIN)
                if item is not None:
                    item.setText("0.0")
        _process(60)

        window._plot_panel.set_fit_range(*_ZF_FIT_RANGE_US)
        single_tab._run_fit()
        single_tab.wait_for_fit()
        _process(80)

        # ν ≈ 6.1 MHz ⇒ ~0.16 µs period; zoom to the first ~2.2 µs (~13 cycles)
        # so the damped spontaneous oscillation resolves, and frame Y to that
        # window so the precession sits large rather than as ripple on the
        # large uncalibrated baseline.
        window._plot_panel.set_view_limits(0.0, 2.2, *window._plot_panel.get_view_limits()[2:])
        _process(60)
        self._frame_y_to_window(window, 0.0, 2.2)
        _process(80)
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
# 2. Headline — ν(T) order parameter with the fitted power law
# ---------------------------------------------------------------------------
class NiNuTOrderParameterScenario(CorpusScenario):
    name = "corpus_ni_nu_t_order_parameter"
    description = (
        "Ni magnetic order parameter: spontaneous ZF precession frequency "
        "ν(T) from real per-run fits (temperatures converted °C → K), with "
        "the fitted OrderParameter power law ν=y0·(1−T/Tc)^β "
        "(T_C ≈ 631 K, with a range-sensitive effective exponent)."
    )
    example = EXAMPLE
    size = (1240, 760)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.core.fitting.parameter_models import (
            ModelFitRange,
            ParameterCompositeModel,
            ParameterModelFit,
            fit_parameter_model,
        )
        from asymmetry.core.fitting.parameters import Parameter, ParameterSet
        from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

        rows = []
        nu_seed, amp_seed = 9.6, 5.0
        for run in _ZF_OP_RUNS:
            dataset = load_corpus_datasets([_DATA % run])[0]
            result = _fit_zf_frequency(dataset, nu_seed, amp_seed)
            nu = result.parameters["frequency"].value
            amp = result.parameters["A_1"].value
            # The transition falls near 358 on the furnace controller scale,
            # matching the 630 K reference after conversion from Celsius.
            temperature = float(dataset.temperature) + CELSIUS_ZERO_K
            rows.append(
                {
                    "run_number": run,
                    "run_label": dataset.run_label,
                    "field": 0.0,
                    "temperature": temperature,
                    "values": {"frequency": nu},
                    "errors": {"frequency": result.uncertainties["frequency"]},
                }
            )
            nu_seed = max(nu * 0.92, 1.0)
            amp_seed = min(amp * 1.05, 15.0)

        batch_id = "ni-nu-t-corpus"
        panel = FitParametersPanel()
        panel.load_representation_series(
            [(batch_id, "ν(T) — Ni ZF", rows)],
            select_id=batch_id,
        )

        panel._x_combo.setCurrentText("𝑇 (K)")
        selected_rows = panel._included_trend_rows("temperature")
        temperature = np.array([row.temperature for row in selected_rows])
        nu, nu_err = panel._series_y_arrays(selected_rows, "frequency")

        # Fix the shape exponent to one for ν(T) ∝ (T_C − T)^β.
        model = ParameterCompositeModel(["OrderParameter"])
        params = ParameterSet(
            [
                Parameter(name="y0", value=28.0, min=0.0),
                Parameter(name="Tc", value=632.0, min=629.5, max=680.0),
                Parameter(name="beta", value=0.4, min=0.1, max=0.8),
                Parameter(name="alpha", value=1.0, fixed=True),
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
            raise RuntimeError("Ni OrderParameter ν(T) fit did not converge")

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
        _process(80)
        return panel

    def settle(self, widget: QWidget) -> None:
        _process(200)
        widget._refresh_plot()
        _wait_until(
            lambda: (
                not widget._trend_curve_compute_active
                and widget._precomputed_trend_curves is not None
            ),
            timeout_ms=20000,
        )
        _process(200)


# ---------------------------------------------------------------------------
# 3. Frequency domain — the spontaneous precession line inside the pulsed band
# ---------------------------------------------------------------------------
class NiZfFftScenario(CorpusScenario):
    name = "corpus_ni_zf_fft"
    description = (
        "Frequency-domain view of the Ni 618 K (controller 345 °C) ZF run: a "
        "single spontaneous internal-field precession line at ν ≈ 6.1 MHz, "
        "inside EMU's pulsed-source band (contrast the lower-T runs where "
        "ν ≈ 18–21 MHz is out of band, guide Q7/Q9)."
    )
    example = EXAMPLE
    size = (1500, 920)

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window._on_fourier()
        window.resizeDocks([window._dock_data_browser], [320], Qt.Orientation.Horizontal)

        datasets = load_corpus_datasets([_DATA % _ZF_FIT_RUN])
        self.add_to_browser(window, datasets)
        window._on_dataset_selected(datasets[0].run_number)
        _process(120)

        window._on_domain_button_clicked("frequency")
        _process(80)

        # Lorentzian apodisation matched to the ~1.9 µs coherence (1/λ) of the
        # damped signal, so the single spontaneous line stands clear of the
        # low-frequency skirt.
        fp = window._fourier_panel
        if hasattr(fp, "_filter_lorentzian_radio"):
            fp._filter_lorentzian_radio.setChecked(True)
        if hasattr(fp, "_filter_time_constant_edit"):
            fp._filter_time_constant_edit.setText("2.0")
        _process(40)

        freq_panel = window._frequency_plot_panel
        x_min, x_max = 0.0, 16.0
        window._on_compute_fourier()

        spectrum_x = spectrum_y = None
        for _ in range(100):  # bounded ~10 s
            _process(100)
            x = getattr(freq_panel, "_last_plot_time", None)
            y = getattr(freq_panel, "_last_plot_asymmetry", None)
            if x is not None and y is not None and len(x) and float(np.nanmax(x)) >= x_max:
                spectrum_x = np.asarray(x, dtype=float)
                spectrum_y = np.asarray(y, dtype=float)
                break
        if spectrum_x is None:
            raise RuntimeError("Ni Fourier recompute did not render within 10 s")

        in_window = (spectrum_x >= 2.0) & (spectrum_x <= x_max)
        peak = float(np.max(spectrum_y[in_window])) if np.any(in_window) else 1.0
        freq_panel.set_view_limits(x_min, x_max, -0.05 * peak, 1.15 * peak)
        _process(120)
        return window


# ---------------------------------------------------------------------------
# 4. LF decoupling at 473 K — static internal field after oscillations are gone
# ---------------------------------------------------------------------------
class NiLfDecouplingScenario(CorpusScenario):
    name = "corpus_ni_lf_decoupling"
    description = (
        "Ni 473 K (controller 200 °C) longitudinal-field decoupling: the "
        "recovered (decoupled) asymmetry rises with applied field "
        "(1200 → 4000 G) as the static internal field is decoupled (guide Q8)."
    )
    example = EXAMPLE
    size = (1240, 760)

    def __init__(self) -> None:
        super().__init__()
        self._recovery: list[tuple[float, float]] = []

    def build(self) -> QWidget:
        from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

        # Recovered (LF-decoupled) asymmetry = the mean early-time polarisation
        # of each run. At 473 K ≈ 0.75 T_C (controller 200 °C — well inside the
        # ordered phase) the muons see a large static internal field; raising
        # the applied longitudinal field decouples it, so the recovered
        # asymmetry rises monotonically with field — the LF decoupling curve
        # the guide's Q8 asks for.
        fields: list[float] = []
        recovered: list[float] = []
        for run in _LF_RUNS:
            ds = load_corpus_datasets([_DATA % run])[0]
            t = np.asarray(ds.time, dtype=float)
            a = np.asarray(ds.asymmetry, dtype=float)
            w = np.isfinite(t) & np.isfinite(a) & (t <= 4.0)
            fields.append(float(ds.field))
            recovered.append(float(np.nanmean(a[w])))
        order = np.argsort(fields)
        self._recovery = [(fields[i], recovered[i]) for i in order]

        row_dicts = [
            {
                "run_number": int(_LF_RUNS[i]),
                "run_label": f"{fields[i]:.0f} G",
                "field": float(fields[i]),
                "temperature": 200.0 + CELSIUS_ZERO_K,
                "values": {"asymmetry": float(recovered[i])},
                "errors": {"asymmetry": 0.15},
            }
            for i in order
        ]
        panel = FitParametersPanel()
        panel.load_representation_series(
            [("ni-lf-decoupling", "recovered asymmetry (473 K) — Ni LF", row_dicts)],
            select_id="ni-lf-decoupling",
        )
        # Plot against the applied field, not the (constant 473 K) temperature.
        for label in ("𝐵 (G)", "B (G)"):
            idx = panel._x_combo.findText(label)
            if idx >= 0:
                panel._x_combo.setCurrentIndex(idx)
                break
        _process(80)
        return panel

    def settle(self, widget: QWidget) -> None:
        _process(150)
        widget._refresh_plot()
        _process(200)
        axes = list(getattr(widget, "_figure").axes)
        if axes:
            ax = axes[0]
            ax.set_xlabel("Applied longitudinal field (G)")
            widget._canvas.draw()
        _process(120)


register(NiZfPrecessionFitScenario())
register(NiNuTOrderParameterScenario())
register(NiZfFftScenario())
register(NiLfDecouplingScenario())

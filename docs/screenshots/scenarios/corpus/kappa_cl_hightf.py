"""Corpus scenarios — AFM transition in high TF / κ-(BEDT-TTF)₂Cu[N(CN)₂]Cl.

Drives the Asymmetry GUI through the WiMDA muon-school "AFM transition in high
TF" example on the **real HAL9500/PSI ``.mdu`` corpus files**
(``tdc_hifi_2020_00686–00693`` at 6 T, ``00730–00739`` at 8 T; sample
``kappa-ETCl``). The reference paper is the spec: B. M. Huddart *et al.*,
"µSR investigation of magnetism in κ-(ET)₂X: Antiferromagnetism,"
Phys. Rev. Research **5**, 013015 (2023). See the example's ``GROUND_TRUTH.md``.

This is the corpus's one **high-field TF frequency-domain** example. Two facts
drive every scenario here:

* The ``.mdu`` (HiFi/HAL PSI) loader delivers the raw octant histograms binned
  at **0.0244 ns** (Nyquist ≈ 20.5 GHz), so the diamagnetic Larmor line —
  γ_µ·B ≈ **813 MHz at 6 T**, **1084 MHz at 8 T** — is fully resolved by a
  plain FFT. No rotating reference frame is needed; the GUI's Fourier/MaxEnt
  pipeline reaches the line directly (``GROUND_TRUTH.md`` §9.6 flagged this as a
  possible Nyquist blocker — it is not, at this binning).
* Below T_N the ordered internal field **broadens and depletes** the central
  precession line (spectral weight moves into the wings/satellites, paper
  §III B 1). Tracking that depletion vs T is an order parameter Â(T) that
  reproduces the paper's Fig 8 transition and T_N (§6/§6b).

Scenarios registered:

* ``corpus_kappacl_load_browse`` — the 6 T + 8 T ``.mdu`` series in the data
  browser (measured temperatures), with an ordered-run time trace zoomed onto
  the fast ~813 MHz precession (format-support / loader render).
* ``corpus_kappacl_tf_fft``      — frequency domain: the 6 T diamagnetic line
  at ~813 MHz for an ordered run (3.2 K) overlaid on a paramagnetic run
  (50.7 K), showing the AFM broadening/depletion.
* ``corpus_kappacl_maxent``      — MaxEnt internal-field distribution ΔB around
  the 6 T applied field (the paper's actual observable, field-centred auto
  window) for the base-T ordered run.
* ``corpus_kappacl_amplitude_t`` — headline: the 6 T order parameter Â(T) from
  the central-line depletion, fitted with the OrderParameter power law to
  recover T_N ≈ 28 K (GT §6b target 28.2(5) K).
* ``corpus_kappacl_field_shift`` — Â(T) for 6 T and 8 T together: the
  field-driven upward shift of the transition (28 → 30 K, GT §6b point iii).

``requires_fit = True`` where a real iminuit/MaxEnt computation runs at capture
time (numba-backed; CI pins numpy < 2.3).
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from ._corpus import CorpusScenario, _process_events_for, load_corpus_datasets, register

EXAMPLE = "Magnetism/AFM transition in high TF"
_DATA = "Magnetism/AFM transition in high TF/data/tdc_hifi_2020_%05d.mdu"

# γ_µ / 2π = 135.5 MHz/T ⇒ diamagnetic Larmor line centre per field.
_CENTRE_6T_MHZ = 813.5
_CENTRE_8T_MHZ = 1084.0

# Run → measured sample temperature (GROUND_TRUTH.md §3; from the .mdu metadata).
_RUNS_6T: list[tuple[int, float]] = [
    (686, 3.24),
    (687, 6.00),
    (688, 12.00),
    (689, 18.00),
    (690, 24.00),
    (691, 27.00),
    (692, 30.00),
    (693, 50.66),
]
_RUNS_8T: list[tuple[int, float]] = [
    (730, 3.12),
    (731, 6.00),
    (732, 10.47),
    (733, 18.00),
    (734, 24.00),
    (735, 27.00),
    (736, 30.00),
    (737, 50.00),
    (738, 75.00),
    (739, 100.00),
]

_ORDERED_6T = 686  # base T, 3.24 K — deep in the ordered phase
_PARA_6T = 693  # 50.66 K — paramagnetic reference


def _rel(run: int) -> str:
    return _DATA % run


# --------------------------------------------------------------------------- #
#  Order-parameter metric.
# --------------------------------------------------------------------------- #
def _line_peak_height(
    dataset, centre_mhz: float, *, t_max_us: float = 4.0, band_mhz: float = 25.0
) -> float:
    """Hann-windowed FFT peak height of the diamagnetic line near *centre_mhz*.

    Below T_N the ordered internal field broadens the precession line, moving
    coherent spectral weight into the wings and *lowering* this central-line
    peak. The peak height is therefore an order-parameter observable — the
    complement of the paper's "integrated spectral area on the wings"
    (GROUND_TRUTH.md §4, §6b): what leaves the sharp central line appears in the
    wings. Computed directly on the loader's reduced asymmetry over the first
    ``t_max_us`` (the coherent part of the signal), with a fixed apodisation so
    the depletion is comparable run-to-run.
    """
    t = np.asarray(dataset.time, dtype=float)
    a = np.asarray(dataset.asymmetry, dtype=float)
    ok = np.isfinite(a)
    t, a = t[ok], a[ok]
    dt = float(t[1] - t[0])
    sel = t < t_max_us
    signal = a[sel] - np.nanmean(a[sel])
    n = signal.size
    freq = np.fft.rfftfreq(n, d=dt)  # MHz (dt in µs)
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(n)))
    band = (freq > centre_mhz - band_mhz) & (freq < centre_mhz + band_mhz)
    return float(spectrum[band].max())


def _order_parameter(runs: list[tuple[int, float]], centre_mhz: float):
    """Return (T, Â, Â_err) for *runs*: normalised central-line depletion 1→0.

    Â = (H_para − H) / (H_para − H_0), with H_para the warm (paramagnetic)
    plateau and H_0 the base-T ordered plateau, so Â runs 1 (fully ordered) →
    0 (paramagnetic) through T_N, matching the GT §6b normalised order
    parameter.
    """
    temps, heights = [], []
    for run, temp in runs:
        ds = load_corpus_datasets([_rel(run)])[0]
        temps.append(float(temp))
        heights.append(_line_peak_height(ds, centre_mhz))
    temps = np.asarray(temps)
    heights = np.asarray(heights)
    order = np.argsort(temps)
    temps, heights = temps[order], heights[order]
    h_para = float(heights[-1])  # warmest run — paramagnetic background
    h0 = float(np.mean(heights[:2]))  # two coldest runs — ordered plateau
    amp = (h_para - heights) / (h_para - h0)
    return temps, amp, np.full_like(amp, 0.05)


def _fit_order_parameter(temps, amp, amp_err):
    """Fit the OrderParameter power law ν=y0·[1−(T/Tc)^α]^β; return the result."""
    from asymmetry.core.fitting.parameter_models import (
        ParameterCompositeModel,
        fit_parameter_model,
        suggest_trend_seeds,
    )
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    model = ParameterCompositeModel(["OrderParameter"])
    seeds = suggest_trend_seeds(model, temps, amp)
    params = ParameterSet(
        [
            Parameter(name=p, value=float(seeds.get(p, model.param_defaults[p])))
            for p in model.param_names
        ]
    )
    result = fit_parameter_model(
        temps,
        amp,
        amp_err,
        model,
        params,
        x_min=float(temps.min()),
        x_max=float(temps.max()),
    )
    return model, result


def _amp_series_rows(temps, amp, amp_err, runs):
    run_by_temp = {float(t): r for r, t in runs}
    return [
        {
            "run_number": int(run_by_temp.get(float(temps[i]), 1000 + i)),
            "run_label": f"{temps[i]:.1f} K",
            "field": 0.0,
            "temperature": float(temps[i]),
            "values": {"amplitude": float(amp[i])},
            "errors": {"amplitude": float(amp_err[i])},
        }
        for i in range(len(temps))
    ]


def _install_model_fit(panel, model, result, temps):
    from asymmetry.core.fitting.parameter_models import ModelFitRange, ParameterModelFit

    fit_range = ModelFitRange(
        x_min=float(temps.min()),
        x_max=float(temps.max()),
        model=model,
        parameters=result.parameters,
        result=result,
    )
    panel._model_fits["amplitude"] = ParameterModelFit(
        parameter_name="amplitude",
        x_key="temperature",
        ranges=[fit_range],
        active=True,
    )
    panel._sync_active_group_state()
    panel._refresh_model_fit_button_labels()


def _wait_until(predicate, *, timeout_ms: int, poll_ms: int = 40) -> None:
    elapsed = 0
    while elapsed < timeout_ms:
        if predicate():
            return
        _process_events_for(milliseconds=poll_ms)
        elapsed += poll_ms


# --------------------------------------------------------------------------- #
#  1. Load + browse the 6 T + 8 T .mdu series.
# --------------------------------------------------------------------------- #
class KappaClLoadBrowseScenario(CorpusScenario):
    name = "corpus_kappacl_load_browse"
    description = (
        "HAL/PSI .mdu κ-Cl 6 T and 8 T TF series in the data browser (measured "
        "temperatures), with the base-T ordered run's ~813 MHz precession in "
        "the time-domain plot."
    )
    example = EXAMPLE
    size = (1440, 880)

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window.resizeDocks([window._dock_data_browser], [440], Qt.Orientation.Horizontal)

        # A representative spread of both fields through the transition.
        runs = [686, 687, 690, 691, 692, 693, 730, 732, 735, 736, 737, 739]
        datasets = load_corpus_datasets([_rel(r) for r in runs])
        with window._data_browser.batch_updates():
            for dataset in datasets:
                window._data_browser.add_dataset(dataset)
        window._data_browser.create_data_group(
            [int(ds.run_number) for ds in datasets],
            name="κ-Cl TF — 6 T + 8 T (HAL@PSI)",
        )

        # Select the base-T ordered 6 T run; zoom the time axis onto the first
        # ~0.03 µs so the fast (813 MHz ⇒ 1.23 ns period) diamagnetic precession
        # is a resolved oscillation rather than a solid band.
        window._on_dataset_selected(_ORDERED_6T)
        _process_events_for(milliseconds=120)
        y = window._plot_panel.get_view_limits()[2:]
        window._plot_panel.set_view_limits(0.0, 0.03, *y)
        _process_events_for(milliseconds=80)
        return window


# --------------------------------------------------------------------------- #
#  2. TF FFT — the 6 T diamagnetic line: ordered vs paramagnetic.
# --------------------------------------------------------------------------- #
class KappaClTfFftScenario(CorpusScenario):
    name = "corpus_kappacl_tf_fft"
    description = (
        "Frequency domain: the 6 T diamagnetic Larmor line at ~813 MHz for the "
        "ordered κ-Cl run (3.2 K, broadened/depleted) overlaid on the "
        "paramagnetic run (50.7 K, sharp)."
    )
    example = EXAMPLE
    size = (1180, 760)

    def build(self) -> QWidget:
        from asymmetry.core.fourier import (
            GroupSpectrumConfig,
            compute_average_group_spectrum,
        )
        from asymmetry.gui.panels.plot_panel import PlotPanel

        # Both spectra through the identical core routine the Fourier panel
        # drives (the average grouped FFT), so ordered and paramagnetic are
        # directly comparable: strong Lorentzian apodisation (τ = 3 µs) matched
        # to the long coherent HAL signal so the ~813 MHz line resolves. The
        # 0.0244 ns binning puts the line well below the 20.5 GHz Nyquist — no
        # rotating reference frame is needed.
        def _spectrum(run_id, label):
            ds = load_corpus_datasets([_rel(run_id)])[0]
            run = ds.run
            gids = list(run.grouping["groups"].keys())
            cfg = GroupSpectrumConfig(
                display="(Power)^1/2",
                window="lorentzian",
                filter_time_constant_us=3.0,
                selected_group_ids=gids,
                group_phase_degrees={g: 0.0 for g in gids},
                exclusion_ranges=[],
                t_min_us=0.0,
                t_max_us=6.0,
                subtract_average_signal=True,
            )
            spectrum = compute_average_group_spectrum(run, cfg)
            # The overlay legend reads ``metadata['run_label']`` (see
            # MuonDataset.run_label), so a physics label replaces "686 Average".
            if isinstance(spectrum.metadata, dict):
                spectrum.metadata["run_label"] = label
            return spectrum

        ordered = _spectrum(_ORDERED_6T, "3.2 K — ordered")
        para = _spectrum(_PARA_6T, "50.7 K — paramagnetic")

        # A real frequency-domain PlotPanel overlays the two spectra (the panel
        # renders one run's FFT at a time inside the main window; a standalone
        # frequency PlotPanel overlays the pair for the comparison figure).
        panel = PlotPanel(domain="frequency")
        panel.set_overlay_enabled(True)
        panel.plot_datasets([para, ordered])
        _process_events_for(milliseconds=150)

        # Peak values (para is the taller, sharper line) set a common Y so the
        # ordered line's depletion + broadening reads on the same scale.
        yd = []
        for line in panel._figure.axes[0].get_lines():
            x = np.asarray(line.get_xdata(), dtype=float)
            y = np.asarray(line.get_ydata(), dtype=float)
            if x.size > 10:
                band = (x > _CENTRE_6T_MHZ - 6) & (x < _CENTRE_6T_MHZ + 6)
                if band.any():
                    yd.append(float(np.nanmax(y[band])))
        peak = max(yd) if yd else 1.0
        panel.set_view_limits(_CENTRE_6T_MHZ - 4.0, _CENTRE_6T_MHZ + 4.0, -0.03 * peak, 1.10 * peak)
        _process_events_for(milliseconds=120)
        return panel


# --------------------------------------------------------------------------- #
#  3. MaxEnt internal-field distribution ΔB around the 6 T applied field.
# --------------------------------------------------------------------------- #
class KappaClMaxEntScenario(CorpusScenario):
    name = "corpus_kappacl_maxent"
    description = (
        "MaxEnt internal-field distribution around the 6 T applied field "
        "(field-centred auto window) for the base-T ordered κ-Cl run: the "
        "internal-field line resolves at ~813.7 MHz, just above the γ_µ·B "
        "applied-field marker — the paper's headline observable."
    )
    example = EXAMPLE
    size = (1500, 920)
    requires_fit = True  # numba-backed MaxEnt solver (numpy < 2.3)

    def build(self) -> QWidget:
        import time

        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window._on_fourier()
        window.resizeDocks([window._dock_data_browser], [300], Qt.Orientation.Horizontal)

        ordered = load_corpus_datasets([_rel(_ORDERED_6T)])[0]
        window._data_browser.add_dataset(ordered)
        window._on_dataset_selected(ordered.run_number)
        _process_events_for(milliseconds=120)

        window._on_domain_button_clicked("maxent")
        _process_events_for(milliseconds=120)

        # Steer the reconstruction to a tractable, field-centred window: auto
        # window (centres on γ_µ·6T ≈ 813 MHz), a short coherent time range with
        # generous binning to keep the projection matrix small (Nyquist after
        # 8× binning is ≈ 2.6 GHz, still well above the line).
        panel = window._maxent_panel
        panel._auto_window_check.setChecked(True)
        panel._points_spin.setValue(256)
        panel._t_max_edit.setText("2.0")
        panel._time_binning_spin.setValue(8)
        _process_events_for(milliseconds=60)

        run_number = int(ordered.run_number)
        window._on_compute_maxent(8)
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            _process_events_for(milliseconds=150)
            if run_number in window._maxent_result_by_run:
                break
        _process_events_for(milliseconds=250)

        # Frame onto the reconstructed line; the field-centred auto window
        # already brackets it (≈809–817 MHz around γ_µ·6T ≈ 813 MHz — the paper's
        # ΔB = internal − applied region), so autoscale Y from that range. (The
        # panel's "X relative to ref. field" toggle re-maps this axis to ΔB, the
        # paper's Fig 8 presentation; it is left in absolute MHz here so the
        # γ_µ·B reference marker and the internal-field line both read cleanly.)
        freq_panel = window._frequency_plot_panel
        x = freq_panel._last_plot_time
        y = freq_panel._last_plot_asymmetry
        if x is not None and y is not None and len(x):
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
            inw = np.isfinite(y)
            peak = float(np.nanmax(y[inw])) if np.any(inw) else 1.0
            freq_panel.set_view_limits(lo, hi, -0.04 * peak, 1.10 * peak)
        _process_events_for(milliseconds=120)
        return window


# --------------------------------------------------------------------------- #
#  4. Headline — the 6 T order parameter Â(T) with the OrderParameter fit.
# --------------------------------------------------------------------------- #
class KappaClAmplitudeTScenario(CorpusScenario):
    name = "corpus_kappacl_amplitude_t"
    description = (
        "κ-Cl 6 T order parameter Â(T) from the central-line depletion, fitted "
        "with the OrderParameter power law to recover T_N ≈ 28 K "
        "(GT §6b target 28.2(5) K)."
    )
    example = EXAMPLE
    size = (1240, 780)
    requires_fit = True

    def __init__(self) -> None:
        super().__init__()
        self._fit_summary: dict[str, float] = {}

    def build(self) -> QWidget:
        from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

        temps, amp, amp_err = _order_parameter(_RUNS_6T, _CENTRE_6T_MHZ)
        model, result = _fit_order_parameter(temps, amp, amp_err)
        if not result.success:
            raise RuntimeError("κ-Cl 6 T OrderParameter Â(T) fit did not converge")
        self._fit_summary = {n: float(result.parameters[n].value) for n in model.param_names}

        panel = FitParametersPanel()
        panel.load_representation_series(
            [("kappacl-6T", "Â(T) — κ-Cl 6 T", _amp_series_rows(temps, amp, amp_err, _RUNS_6T))],
            select_id="kappacl-6T",
        )
        _install_model_fit(panel, model, result, temps)
        self._temps = temps
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
#  5. 8 T order parameter — the field-shifted transition (T_N ≈ 30 K).
# --------------------------------------------------------------------------- #
class KappaClAmplitudeT8TScenario(CorpusScenario):
    name = "corpus_kappacl_amplitude_t_8t"
    description = (
        "κ-Cl 8 T order parameter Â(T) with the OrderParameter fit: the "
        "ordered-state transition near 28–30 K at the higher field "
        "(GT §6b target 30.2(2) K; see NOTES on T_N recovery from this proxy)."
    )
    example = EXAMPLE
    size = (1240, 780)
    requires_fit = True

    def __init__(self) -> None:
        super().__init__()
        self._fit_summary: dict[str, float] = {}

    def build(self) -> QWidget:
        from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

        # Drop the two far-background runs (75, 100 K) that sit off the Fig 8(f)
        # x-axis so the plateau + transition dominate the frame.
        runs8 = [(r, t) for r, t in _RUNS_8T if t <= 55.0]
        temps, amp, amp_err = _order_parameter(runs8, _CENTRE_8T_MHZ)
        model, result = _fit_order_parameter(temps, amp, amp_err)
        if not result.success:
            raise RuntimeError("κ-Cl 8 T OrderParameter Â(T) fit did not converge")
        self._fit_summary = {n: float(result.parameters[n].value) for n in model.param_names}

        panel = FitParametersPanel()
        panel.load_representation_series(
            [("kappacl-8T", "Â(T) — κ-Cl 8 T", _amp_series_rows(temps, amp, amp_err, runs8))],
            select_id="kappacl-8T",
        )
        _install_model_fit(panel, model, result, temps)
        self._temps = temps
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


register(KappaClLoadBrowseScenario())
register(KappaClTfFftScenario())
register(KappaClMaxEntScenario())
register(KappaClAmplitudeTScenario())
register(KappaClAmplitudeT8TScenario())

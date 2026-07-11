"""Corpus scenarios — A molecular antiferromagnet (Magnetism).

Drives the Asymmetry GUI through the WiMDA muon-school **molecular
antiferromagnet** example on the real **ISIS/MUSR NeXus-v1 (HDF4)** corpus
files (``MUSR00017094.nxs`` … ``MUSR00017104.nxs``, runs 17094–17104, April
2008). The teaching guide is the spec (``GROUND_TRUTH.md``); no reference paper
or logbook is present, so every number is graded against the **[M] measured**
values in GROUND_TRUTH §3a / §6.

Sample Ni(9S3)₂[Ni(bdt)₂]₂ is a molecular magnet with two inequivalent Ni
environments (Ni²⁺ S = 1 and Ni⁺ S = ½) whose coexisting ferri-/antiferro-
magnetic chains order antiferromagnetically on cooling. In zero field the
ordered state shows a **coherent internal-field precession** whose frequency
ν(T) is the order parameter. Two facts shape the scenarios and are recorded in
``NOTES_molafm.md``:

* **Low-frequency oscillation.** ν(0) ≈ 1.55 MHz here — ~20× *slower* than the
  EuO ferromagnet's 30 MHz spontaneous line — so the period is ~0.65 µs and the
  precession is legible over several µs rather than needing a sub-µs zoom
  (GROUND_TRUTH §3a). This is the visibly different character of a molecular
  magnet: a slow, weakly-damped wiggle, not a dense band.
* **Sharp T_N.** ν(T) falls monotonically 1.55 → 0.66 MHz over 1.2 → 6 K and is
  **gone by the 7 K run** (statistically indistinguishable from the 8–10 K
  paramagnet, GROUND_TRUTH §3a audit note), giving **T_N ≈ 6–7 K**. The 7 K
  run is *not* a low-frequency oscillation — the earlier "0.35 MHz @ 7 K" was a
  coarse-FFT relaxation-leak artefact and is not reproduced here.

Installed data has low statistics per run (some 4–20 MEv) on a fine 16 ns MUSR
time base, so the per-run ZF fits use a modest fit window and warm-started
seeds; the 6 K run (17099) sits at the edge of resolvability near T_N.

Scenarios registered:

* ``corpus_molafm_alpha``     — α / balance calibration on the 20 G TF run
  17104 (10 K, paramagnetic): the guide's "which data set for α?" data-prep
  step; the Estimate balances F/B and returns α ≈ 1.25.
* ``corpus_molafm_zf_fit``    — the money shot: a converged damped-oscillation
  fit (Oscillatory×Exponential+Constant) on the 1.2 K ZF run 17094,
  ν ≈ 1.55 MHz, zoomed so the slow precession resolves.
* ``corpus_molafm_nu_t``      — headline: real per-run ZF fits → ν(T) with the
  fitted OrderParameter power law ν=ν₀·[1−(T/T_N)^α]^β; recovers T_N ≈ 6.3 K
  (inside the measured 6–7 K bracket).
* ``corpus_molafm_zf_overlay`` — 6 K vs 7 K ZF spectra overlaid: the coherent
  oscillation present at 6 K (17099) is gone at 7 K (17100), fixing T_N ≈ 6–7 K
  directly from the data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from asymmetry.gui.styles import tokens

from .._base import _process_events_for
from ._corpus import CorpusScenario, load_corpus_datasets, register

EXAMPLE = "Magnetism/A molecular antiferromagnet"
_DATA = "Magnetism/A molecular antiferromagnet/Data/MUSR000%d.nxs"

# ZF spontaneous precession = damped cosine + baseline. The MUSR F/B asymmetry
# the loader builds is uncalibrated (α = 1); the large additive baseline (~15)
# is absorbed by the Constant, exactly the "relaxing tail" the guide's fit-model
# question (GROUND_TRUTH §4/§5 Q4) anticipates.
_ZF_MODEL = (["Oscillatory", "Exponential", "Constant"], ["*", "+"])

# ZF order-parameter branch (GROUND_TRUTH §3a): the runs that still show a
# coherent oscillation, 1.2 → 6 K. Above 6 K the spectrum is paramagnetic
# (17100–17103, "gone"), so they are excluded from the ν(T) order parameter.
_ZF_OP_RUNS: list[tuple[int, float]] = [
    (17094, 1.2),
    (17095, 2.0),
    (17096, 3.0),
    (17097, 4.0),
    (17098, 5.0),
    (17099, 6.0),
]

# The money-shot run: base temperature, cleanest low-frequency oscillation.
_ZF_FIT_RUN = 17094

# The α / balance run: 20 G transverse field at 10 K (paramagnetic, above T_N),
# where the full TF precession amplitude calibrates the F/B balance
# (GROUND_TRUTH §4/§5 Q1).
_ALPHA_RUN = 17104

# T_N bracket: the last oscillating run (6 K) vs the first paramagnetic run
# (7 K) — the direct-comparison overlay that fixes T_N ≈ 6–7 K.
_OVERLAY_RUNS = (17099, 17100)  # (6 K ordered, 7 K paramagnetic)


def _process(ms: int = 80) -> None:
    _process_events_for(milliseconds=ms)


def _fit_zf_frequency(dataset, nu_seed: float, amp_seed: float, lam_seed: float):
    """Fit the damped ZF oscillation of one run through the core engine.

    Returns ``(nu_MHz, nu_err, amplitude, lambda)``. Warm-starting ν *downward*
    (and letting the amplitude/damping grow) as temperature rises keeps every
    run in the correct minimum — a cold seed lets the low-amplitude oscillation
    near T_N collapse to zero, the EuO/YMnAl/Ni warm-start lesson. The fit runs
    over t = 0.11–8 µs (the low-statistics late-time tail excluded).
    """
    from asymmetry.core.fitting.composite import CompositeModel
    from asymmetry.core.fitting.engine import FitEngine
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    model = CompositeModel(_ZF_MODEL[0], operators=_ZF_MODEL[1])
    bg = float(np.nanmean(dataset.asymmetry))
    seeds = {"A_1": amp_seed, "frequency": nu_seed, "phase": 0.0, "Lambda": lam_seed, "A_bg": bg}
    # ν pinned to the molecular-magnet band (≲2.5 MHz), amplitude non-negative,
    # damping capped so the near-T_N run cannot escape into a degenerate
    # high-damping "decaying baseline" minimum.
    bounds = {"A_1": (0.3, None), "frequency": (0.3, 2.5), "Lambda": (0.0, 1.5)}
    params = ParameterSet(
        [
            Parameter(
                name=n,
                value=seeds[n],
                min=bounds.get(n, (None, None))[0]
                if bounds.get(n, (None, None))[0] is not None
                else -float("inf"),
                max=bounds.get(n, (None, None))[1]
                if bounds.get(n, (None, None))[1] is not None
                else float("inf"),
            )
            for n in model.param_names
        ]
    )
    result = FitEngine().fit(dataset, model.function, params, t_min=0.11, t_max=8.0)
    by_name = {p.name: p.value for p in result.parameters}
    unc = result.uncertainties or {}
    nu = abs(float(by_name["frequency"]))
    err = float(unc.get("frequency", np.nan))
    return (
        nu,
        (err if err == err and err > 0 else 0.02),
        abs(float(by_name["A_1"])),
        abs(float(by_name["Lambda"])),
    )


def _wait_until(predicate, *, timeout_ms: int, poll_ms: int = 30) -> None:
    elapsed = 0
    while elapsed < timeout_ms:
        if predicate():
            return
        _process_events_for(milliseconds=poll_ms)
        elapsed += poll_ms


def _rebin(t: np.ndarray, a: np.ndarray, factor: int):
    """Block-average time/asymmetry by ``factor`` for a legible overlay trace."""
    n = (len(t) // factor) * factor
    if n == 0:
        return t, a
    t = t[:n].reshape(-1, factor).mean(axis=1)
    a = a[:n].reshape(-1, factor).mean(axis=1)
    return t, a


# ---------------------------------------------------------------------------
# 1. α / balance calibration on the 20 G TF run (data-prep)
# ---------------------------------------------------------------------------
class MolAfmAlphaScenario(CorpusScenario):
    name = "corpus_molafm_alpha"
    description = (
        "Alpha calibration on the 20 G transverse-field run MUSR00017104 "
        "(10 K, paramagnetic, above T_N): the guide's 'which data set for α?' "
        "step — Estimate balances the forward/backward asymmetry (α ≈ 1.25)."
    )
    example = EXAMPLE
    size = (760, 660)

    def capture(self, ctx) -> Path:  # noqa: D401
        from asymmetry.gui.windows.grouping.alpha_calibration_dialog import (
            AlphaCalibrationDialog,
        )

        dataset = load_corpus_datasets([_DATA % _ALPHA_RUN])[0]
        grouping = dataset.run.grouping
        # The dialog wants gid -> 0-based detector indices (it re-adds 1 for the
        # reduction); the corpus payload stores 1-based ids, so shift down.
        groups = {int(gid): [int(i) - 1 for i in idxs] for gid, idxs in grouping["groups"].items()}
        dialog = AlphaCalibrationDialog(
            [dataset],
            groups=groups,
            forward_group=int(grouping["forward_group"]),
            backward_group=int(grouping["backward_group"]),
            selected_run_number=int(dataset.run_number),
        )
        dialog.resize(*self.size)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        dialog.show()
        _process(150)

        # Estimate runs on a worker thread; block until it lands so the capture
        # shows the α value and the balanced "after" curve, never the transient
        # "Computing estimate…" state.
        dialog._estimate_btn.click()
        _wait_until(
            lambda: dialog._tasks.active_count == 0 and dialog._estimate is not None,
            timeout_ms=15000,
        )
        _process(80)

        out_path = ctx.output_dir / f"{self.name}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pix = dialog.grab()
        if not pix.save(str(out_path), "PNG"):
            raise RuntimeError(f"Failed to save screenshot to {out_path}")
        dialog.close()
        dialog.deleteLater()
        _process(40)
        return out_path


# ---------------------------------------------------------------------------
# 2. Money shot — converged low-frequency ZF oscillation fit at base T
# ---------------------------------------------------------------------------
class MolAfmZfFitScenario(CorpusScenario):
    name = "corpus_molafm_zf_fit"
    description = (
        "Converged Oscillatory×Exponential+Constant fit on the molecular-AFM "
        "1.2 K zero-field run MUSR00017094: a slow coherent internal-field "
        "precession (ν ≈ 1.55 MHz, period ~0.65 µs), zoomed so the "
        "low-frequency oscillation resolves."
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
        # Seed at the expected base-T order parameter: ν(0) ≈ 1.55 MHz
        # (GROUND_TRUTH §3a), weak damping (λ ≈ 0.2 µs⁻¹) and the large α = 1
        # baseline (~15) absorbed by the Constant.
        seeds = {
            "A_1": 4.0,
            "frequency": 1.55,
            "phase": 0.0,
            "Lambda": 0.2,
            "A_bg": float(np.nanmean(datasets[0].asymmetry)),
        }
        for name, value in seeds.items():
            if name in rows:
                _set_param_table_value(table, rows[name], value)
        # Pin ν, amplitude and rate non-negative so the fit does not settle in a
        # sign-degenerate mirror minimum.
        for name in ("A_1", "frequency", "Lambda"):
            if name in rows:
                item = table.item(rows[name], table.COL_MIN)
                if item is not None:
                    item.setText("0.0")
        _process(60)

        single_tab._run_fit()
        single_tab.wait_for_fit()
        _process(80)

        # ν ≈ 1.55 MHz ⇒ ~0.65 µs period; zoom to the first ~5 µs (~8 cycles) so
        # the slow, weakly-damped molecular-magnet oscillation resolves, and
        # frame Y to that window so the precession sits large rather than as a
        # ripple on the large uncalibrated baseline.
        window._plot_panel.set_view_limits(0.0, 5.0, *window._plot_panel.get_view_limits()[2:])
        _process(60)
        self._frame_y_to_window(window, 0.0, 5.0)
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
# 3. Headline — ν(T) order parameter with the fitted power law → T_N
# ---------------------------------------------------------------------------
class MolAfmNuTScenario(CorpusScenario):
    name = "corpus_molafm_nu_t"
    description = (
        "Molecular-AFM order parameter: spontaneous ZF precession frequency "
        "ν(T) from real per-run fits (1.2 → 6 K), with the fitted "
        "OrderParameter power law ν=ν₀·[1−(T/T_N)^α]^β → T_N ≈ 6.3 K "
        "(measured bracket 6–7 K)."
    )
    example = EXAMPLE
    size = (1240, 760)
    requires_fit = True

    def __init__(self) -> None:
        super().__init__()
        self._fit_summary: dict[str, float] = {}

    def build(self) -> QWidget:
        from asymmetry.core.fitting.parameter_models import (
            ModelFitRange,
            ParameterCompositeModel,
            ParameterModelFit,
            fit_parameter_model,
        )
        from asymmetry.core.fitting.parameters import Parameter, ParameterSet
        from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

        # Fit each ZF run's spontaneous frequency in ascending-T order,
        # warm-starting ν downward from ν(0) ≈ 1.55 MHz so every run lands in the
        # correct minimum (GROUND_TRUTH §4 workflow — chain-seeding through T).
        temps: list[float] = []
        nus: list[float] = []
        errs: list[float] = []
        nu_seed, amp_seed, lam_seed = 1.55, 4.0, 0.15
        for run, temp in _ZF_OP_RUNS:
            ds = load_corpus_datasets([_DATA % run])[0]
            nu, err, amp, lam = _fit_zf_frequency(ds, nu_seed, amp_seed, lam_seed)
            temps.append(temp)
            nus.append(nu)
            errs.append(max(err, 0.02))
            if amp > 0.3:
                nu_seed = max(nu * 0.9, 0.3)
                amp_seed = min(amp * 1.1, 6.0)
                lam_seed = min(lam * 1.15, 1.2)
        temperature = np.array(temps)
        nu = np.array(nus)
        nu_err = np.array(errs)

        batch_id = "molafm-nu-t-corpus"
        row_dicts = [
            {
                "run_number": run,
                "run_label": f"{temperature[i]:.1f} K",
                "field": 0.0,
                "temperature": float(temperature[i]),
                "values": {"frequency": float(nu[i])},
                "errors": {"frequency": float(nu_err[i])},
            }
            for i, (run, _t) in enumerate(_ZF_OP_RUNS)
        ]

        panel = FitParametersPanel()
        panel.load_representation_series(
            [(batch_id, "ν(T) — molecular AFM ZF (corpus)", row_dicts)],
            select_id=batch_id,
        )

        # Phenomenological order-parameter law ν(T)=ν₀·[1−(T/T_N)^α]^β with α=1
        # (the guide leaves the form open, GROUND_TRUTH §4); fit ν₀, T_N, β. T_N
        # (the ν→0 crossing) is the deliverable, graded against the measured
        # 6–7 K bracket (GROUND_TRUTH §6). Few points, so this is frame-honest.
        model = ParameterCompositeModel(["OrderParameter"])
        params = ParameterSet(
            [
                Parameter(name="y0", value=1.6, min=0.5),
                Parameter(name="Tc", value=6.7, min=6.0, max=9.0),
                Parameter(name="beta", value=0.4, min=0.1, max=0.9),
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
            raise RuntimeError("Molecular-AFM OrderParameter ν(T) fit did not converge")
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
# 4. T_N by direct comparison — 6 K (ordered) vs 7 K (paramagnetic) overlay
# ---------------------------------------------------------------------------
class MolAfmZfOverlayScenario(CorpusScenario):
    name = "corpus_molafm_zf_overlay"
    description = (
        "6 K vs 7 K zero-field spectra overlaid: the coherent internal-field "
        "oscillation present at 6 K (MUSR00017099) is gone at 7 K "
        "(MUSR00017100), fixing T_N ≈ 6–7 K directly from the data."
    )
    example = EXAMPLE
    size = (1120, 640)

    def build(self) -> QWidget:
        from asymmetry.gui.widgets.mpl_canvas import create_canvas

        run_ordered, run_para = _OVERLAY_RUNS
        ds_ord = load_corpus_datasets([_DATA % run_ordered])[0]
        ds_par = load_corpus_datasets([_DATA % run_para])[0]

        figure, canvas = create_canvas(layout="tight")
        axes = figure.add_subplot(111)
        # Block-average the fine 16 ns MUSR base (low statistics per run) so the
        # slow 6 K oscillation is a clean trace rather than a noise band; the
        # same binning applies to both runs, so the comparison is fair.
        for ds, colour, label in (
            (ds_ord, tokens.ACCENT, f"6 K — {run_ordered} (ordered)"),
            (ds_par, tokens.TEXT_DIM, f"7 K — {run_para} (paramagnetic)"),
        ):
            t = np.asarray(ds.time, dtype=float)
            a = np.asarray(ds.asymmetry, dtype=float)
            w = np.isfinite(t) & np.isfinite(a) & (t >= 0.11) & (t <= 10.0)
            tb, ab = _rebin(t[w], a[w], 40)
            axes.plot(tb, ab, color=colour, linewidth=1.5, label=label)

        axes.set_xlabel("Time (µs)")
        axes.set_ylabel("Asymmetry (%)")
        axes.set_title("Molecular AFM ZF spectra across T_N: 6 K oscillates, 7 K does not")
        axes.set_xlim(0.0, 8.0)
        axes.legend(loc="upper right", fontsize="small", framealpha=0.9)
        axes.text(
            0.98,
            0.04,
            "oscillation gone by 7 K ⇒ T_N ≈ 6–7 K",
            transform=axes.transAxes,
            ha="right",
            va="bottom",
            fontsize="small",
            color=tokens.TEXT_MUTED,
        )
        self._canvas = canvas
        return canvas

    def settle(self, widget: QWidget) -> None:
        _process(200)
        widget.draw()
        _process(120)


register(MolAfmAlphaScenario())
register(MolAfmZfFitScenario())
register(MolAfmNuTScenario())
register(MolAfmZfOverlayScenario())

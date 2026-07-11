"""Corpus scenarios for the **Basics** calibration & data-handling primer.

Basics is the data-handling on-ramp of the WiMDA muon school corpus (see its
``GROUND_TRUTH.md``): loading, detector **grouping**, the **α** balance
calibration, **dead-time** correction, **t0 / tgood** timing, and the
**fit-table + manual-column** trend that produces the B1 steering deliverable.
These renders drive the real Asymmetry surfaces on the real corpus ``.nxs``
runs, one figure per concept/worked-example.

Every scenario resolves its data through :mod:`._corpus` (``ASYMMETRY_CORPUS_ROOT``)
so it runs locally and in CI. Only the steering scenario runs an iminuit fit at
capture time (the a₀(I) polynomial trend fit — ``requires_fit = True``); the α
estimate is the algebraic diamagnetic estimator and the dead-time before/after
is a pure-core reduction, so the other four are fit-free.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QApplication, QWidget

from asymmetry.gui.styles import tokens

from ._corpus import CorpusScenario, corpus_path, load_corpus_datasets, register

# ── shared corpus run paths ───────────────────────────────────────────────
_MUSR_GROUPING = "Basics/data/MUSR00044989.nxs"  # A3 grouping: MuSR, 64 dets, ZF
_EMU_ALPHA = "Basics/data/EMU00018854.nxs"  # A4 α: Ag TF 100 G (clean silver TF)
_EMU_DEADTIME = "Basics/data/emu00034998.nxs"  # A2 dead-time: silver, high rate
_EMU_T0 = "Basics/data/EMU00018850.nxs"  # B3/A1 t0: Ag TF, pulsed EMU (t0 ≈ 0.24 µs)


def _pump_events(milliseconds: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(int(milliseconds), loop.quit)
    loop.exec()
    QApplication.processEvents()


def _pump_until(predicate, timeout_ms: int = 15_000) -> None:
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


def _grab_widget(widget: QWidget, name: str, output_dir: Path) -> Path:
    out_path = output_dir / f"{name}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pix = widget.grab()
    if not pix.save(str(out_path), "PNG"):
        raise RuntimeError(f"Failed to save screenshot to {out_path}")
    return out_path


# ══════════════════════════════════════════════════════════════════════════
# A3 — detector grouping (MUSR00044989, 64 detectors → 2 groups)
# ══════════════════════════════════════════════════════════════════════════
class BasicsGroupingScenario(CorpusScenario):
    name = "corpus_basics_grouping"
    description = (
        "Grouping window on MUSR00044989 (64 detectors → 2 groups: 1–32 / 33–64) "
        "with the live forward/backward asymmetry preview."
    )
    example = "Basics"
    size = (1180, 720)

    def capture(self, ctx) -> Path:  # noqa: D401
        from asymmetry.gui.windows.grouping.dialog import GroupingDialog

        datasets = load_corpus_datasets([_MUSR_GROUPING])
        dataset = datasets[0]
        # Bunch the live preview *up front* on the run payload so the fine-binned
        # late-time tail is legible (rather than a wall of exploding error bars)
        # and the preview reduces exactly once at open — setting the spin after
        # show() instead would fire a second worker reduction that can race the
        # grab/teardown. The group table, F/B assignment and t0 controls are the
        # subject; the preview is support.
        if isinstance(dataset.run.grouping, dict):
            dataset.run.grouping["bunching_factor"] = 20
        dialog = GroupingDialog(
            [dataset], selected_run_number=int(dataset.run_number)
        )
        dialog.resize(*self.size)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        dialog.show()
        # The live preview reduces on a worker thread after a debounce timer.
        _pump_events(700)
        out = _grab_widget(dialog, self.name, ctx.output_dir)
        dialog.close()
        dialog.deleteLater()
        _pump_events(40)
        return out


# ══════════════════════════════════════════════════════════════════════════
# A4 — α calibration (EMU00018854, Ag TF 100 G) with before/after preview
# ══════════════════════════════════════════════════════════════════════════
class BasicsAlphaScenario(CorpusScenario):
    name = "corpus_basics_alpha"
    description = (
        "Alpha calibration on the Ag TF 100 G run EMU00018854: Estimate balances "
        "the forward/backward asymmetry about zero (α ≈ 0.885)."
    )
    example = "Basics"
    size = (760, 660)

    def capture(self, ctx) -> Path:  # noqa: D401
        from asymmetry.gui.windows.grouping.alpha_calibration_dialog import (
            AlphaCalibrationDialog,
        )

        dataset = load_corpus_datasets([_EMU_ALPHA])[0]
        grouping = dataset.run.grouping
        # The dialog wants gid -> 0-based detector indices (it re-adds 1 for the
        # reduction); the corpus payload stores 1-based ids, so shift down.
        groups = {
            int(gid): [int(i) - 1 for i in idxs]
            for gid, idxs in grouping["groups"].items()
        }
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
        _pump_events(150)

        # Estimate runs on a worker thread; block (live loop) until it lands so
        # the capture shows the α value and the balanced "after" curve, never the
        # transient "Computing estimate…" state.
        dialog._estimate_btn.click()
        _pump_until(
            lambda: dialog._tasks.active_count == 0 and dialog._estimate is not None
        )
        _pump_events(80)
        out = _grab_widget(dialog, self.name, ctx.output_dir)
        dialog.close()
        dialog.deleteLater()
        _pump_events(40)
        return out


# ══════════════════════════════════════════════════════════════════════════
# A2 — dead-time correction (emu00034998): Off vs Auto-Load before/after plot
# ══════════════════════════════════════════════════════════════════════════
class BasicsDeadtimeScenario(CorpusScenario):
    name = "corpus_basics_deadtime"
    description = (
        "Dead-time correction on the silver run emu00034998: forward/backward "
        "asymmetry with correction Off vs Auto-Load (file, silver-derived)."
    )
    example = "Basics"
    size = (1000, 560)

    def capture(self, ctx) -> Path:  # noqa: D401
        import numpy as np

        from asymmetry.core.transform.reduce import reduce_grouped_asymmetry
        from asymmetry.gui.widgets.mpl_canvas import create_canvas

        run = load_corpus_datasets([_EMU_DEADTIME])[0].run
        grouping = dict(run.grouping)
        groups = {int(k): [int(i) for i in v] for k, v in grouping["groups"].items()}
        forward_idx = groups[int(grouping["forward_group"])]
        backward_idx = groups[int(grouping["backward_group"])]
        # A modest bunching keeps the two overlaid curves legible on the fine
        # (~16 ns) EMU time base without changing the reduction.
        grouping["bunching_factor"] = 25
        common = dict(
            histograms=run.histograms,
            grouping=grouping,
            forward_idx=forward_idx,
            backward_idx=backward_idx,
            alpha=1.0,
            use_background=False,
        )
        off = reduce_grouped_asymmetry(
            use_deadtime=False, deadtime_mode="off", **common
        )
        on = reduce_grouped_asymmetry(
            use_deadtime=True, deadtime_mode="file", **common
        )

        figure, canvas = create_canvas(layout="tight")
        axes = figure.add_subplot(111)
        axes.plot(
            off.time,
            off.asymmetry,
            color=tokens.TEXT_DIM,
            linewidth=1.3,
            label="Dead-time: Off",
        )
        axes.plot(
            on.time,
            on.asymmetry,
            color=tokens.ACCENT,
            linewidth=1.5,
            label="Dead-time: Auto-Load (file)",
        )
        axes.set_xlabel("Time (µs)")
        axes.set_ylabel("Asymmetry (%)")
        axes.set_title(
            "Silver emu00034998 — per-detector silver-derived dead-time correction"
        )
        # Focus on the early time where the high-rate correction bites hardest,
        # and on the asymmetry band the two curves occupy (silver ZF sits near a
        # constant ~18–23 %) so the Off↔Auto-Load gap fills the panel. Scale the
        # y-range and the Δ annotation from the *displayed* window only — the
        # late-time tail (counts → 0) is numerically unstable and would otherwise
        # dominate both the limits and the quoted correction size.
        x_max = 12.0
        axes.set_xlim(float(off.time.min()), x_max)
        window = off.time <= x_max
        lo = float(np.nanmin(off.asymmetry[window]))
        hi = float(np.nanmax(on.asymmetry[window]))
        pad = 0.5 * (hi - lo)
        axes.set_ylim(lo - pad, hi + pad)
        peak = float(np.nanmax(np.abs((on.asymmetry - off.asymmetry)[window])))
        axes.legend(loc="best", fontsize="small", framealpha=0.9)
        axes.text(
            0.98,
            0.04,
            f"peak Δasymmetry ≈ {peak:.1f}%",
            transform=axes.transAxes,
            ha="right",
            va="bottom",
            fontsize="small",
            color=tokens.TEXT_MUTED,
        )

        canvas.resize(*self.size)
        canvas.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        canvas.show()
        _pump_events(200)
        canvas.draw()
        out = _grab_widget(canvas, self.name, ctx.output_dir)
        canvas.close()
        canvas.deleteLater()
        _pump_events(40)
        return out


# ══════════════════════════════════════════════════════════════════════════
# A1 / B3 — t0 & tgood: the muon pulse in the raw counts (EMU00018850)
# ══════════════════════════════════════════════════════════════════════════
class BasicsT0Scenario(CorpusScenario):
    name = "corpus_basics_t0"
    description = (
        "t0 / tgood on EMU00018850: raw summed counts show the muon pulse; "
        "t0 (mid-pulse) and tgood (whole pulse arrived) marked from the file."
    )
    example = "Basics"
    size = (1000, 560)

    def capture(self, ctx) -> Path:  # noqa: D401
        import numpy as np

        from asymmetry.gui.widgets.mpl_canvas import create_canvas

        run = load_corpus_datasets([_EMU_T0])[0].run
        grouping = dict(run.grouping)
        bin_width = float(run.histograms[0].bin_width)
        n_bins = run.histograms[0].n_bins
        total = np.zeros(n_bins, dtype=np.float64)
        for hist in run.histograms:
            total += np.asarray(hist.counts, dtype=np.float64)
        time = np.arange(n_bins, dtype=np.float64) * bin_width

        t0_bin = int(grouping["t0_bin"])
        first_good = int(grouping["first_good_bin"])
        t0_us = t0_bin * bin_width
        tgood_us = first_good * bin_width

        # Window on the pulse: a little before t0 through the early decay.
        window = time <= 1.2
        figure, canvas = create_canvas(layout="tight")
        axes = figure.add_subplot(111)
        axes.plot(
            time[window],
            total[window] / 1e3,
            color=tokens.ACCENT,
            linewidth=1.6,
            label="Σ detector counts",
        )
        axes.axvline(
            t0_us,
            color=tokens.TEXT_DIM,
            linestyle="--",
            linewidth=1.2,
            label=f"t0 = {t0_us:.3f} µs (bin {t0_bin})",
        )
        axes.axvline(
            tgood_us,
            color=tokens.WARN,
            linestyle=":",
            linewidth=1.4,
            label=f"tgood = {tgood_us:.3f} µs (bin {first_good})",
        )
        axes.axvspan(t0_us, tgood_us, color=tokens.WARN, alpha=0.12)
        axes.set_xlabel("Time (µs)")
        axes.set_ylabel("Counts (×10³)")
        axes.set_title(
            "EMU00018850 — muon pulse: t0 mid-pulse, tgood after the whole pulse"
        )
        axes.annotate(
            f"tgood offset ≈ {(tgood_us - t0_us) * 1000:.0f} ns",
            xy=(0.5 * (t0_us + tgood_us), total[window].max() / 1e3 * 0.55),
            ha="center",
            fontsize="small",
            color=tokens.TEXT_MUTED,
        )
        axes.legend(loc="upper right", fontsize="small", framealpha=0.9)

        canvas.resize(*self.size)
        canvas.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        canvas.show()
        _pump_events(200)
        canvas.draw()
        out = _grab_widget(canvas, self.name, ctx.output_dir)
        canvas.close()
        canvas.deleteLater()
        _pump_events(40)
        return out


# ══════════════════════════════════════════════════════════════════════════
# B1 — steering curve: fit table with a manual column (a₀ vs steering current)
# ══════════════════════════════════════════════════════════════════════════
# WiMDA reference output (Basics/data/steering_curve.dat): the initial asymmetry
# `a-relaxin` (≈ a₀) of the non-relaxing Ag-mask signal per run, and the manually
# transcribed steering-magnet current (not logged in the EMU files). The a₀(I)
# curve is a parabola with its minimum near I ≈ −0.06 A → the beam-centred current.
_STEERING_ROWS = [
    # (run, steering current A, a0 (a-relaxin), a0 err, chi^2)
    (44989, 0.00, 5.19, 0.15, 1.059),
    (44990, 0.25, 5.28, 0.12, 1.190),
    (44991, 0.50, 5.89, 0.12, 1.083),
    (44992, 0.75, 6.88, 0.16, 1.046),
    (44993, 1.00, 7.96, 0.14, 1.004),
    (44994, -0.25, 5.17, 0.16, 1.258),
    (44995, -0.50, 5.63, 0.15, 1.192),
    (44996, -0.75, 6.10, 0.15, 1.017),
    (44997, -1.00, 6.98, 0.15, 0.977),
]
_STEERING_CUSTOM_KEY = "custom:steering"


def _build_steering_polynomial_fit():
    """Fit the WiMDA-prescribed polynomial to a₀(I) via the real core minimiser.

    The trending panel's Model Fit ``Cubic`` component reproduces the worked
    WiMDA answer: the reference curve in ``steering_curve_fits.tab`` is itself
    a cubic (refit coefficients give c3 ≈ 0.31), and the weighted iminuit fit
    of the nine a₀ points below puts the curve minimum at **I = −0.060 A** —
    GT §B1's "min 5.18 at −0.06 A" beam-centred current. Deterministic: fixed
    seeds, one least-squares solve, no RNG.
    """
    import numpy as np

    from asymmetry.core.fitting.parameter_models import (
        ModelFitRange,
        ParameterCompositeModel,
        ParameterModelFit,
        fit_parameter_model,
    )
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    x = np.array([row[1] for row in _STEERING_ROWS], dtype=float)
    y = np.array([row[2] for row in _STEERING_ROWS], dtype=float)
    yerr = np.array([row[3] for row in _STEERING_ROWS], dtype=float)

    model = ParameterCompositeModel(["Cubic"])
    seed = ParameterSet(
        [
            Parameter("c0", value=5.2),
            Parameter("c1", value=0.2),
            Parameter("c2", value=2.3),
            Parameter("c3", value=0.3),
        ]
    )
    result = fit_parameter_model(x, y, yerr, model, seed, x_min=-1.0, x_max=1.0)
    if not result.success:
        raise RuntimeError("Steering a₀(I) polynomial fit did not converge")
    fit_range = ModelFitRange(
        x_min=-1.0,
        x_max=1.0,
        model=model,
        parameters=result.parameters,
        result=result,
    )
    return ParameterModelFit(
        parameter_name="a0",
        x_key=_STEERING_CUSTOM_KEY,
        ranges=[fit_range],
        active=True,
    )


class BasicsSteeringScenario(CorpusScenario):
    name = "corpus_basics_steering"
    description = (
        "Fit-table trend with a manual column: Ag-mask initial asymmetry a₀ vs "
        "steering-magnet current (EMU 44989–44997) with the fitted polynomial "
        "overlaid — minimum ≈ −0.06 A, the beam-centred current."
    )
    example = "Basics"
    size = (1180, 720)
    requires_fit = True  # runs the real iminuit-backed polynomial trend fit

    def build(self) -> QWidget:
        from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

        batch_id = "basics-steering"
        row_dicts = [
            {
                "run_number": run,
                "run_label": f"{run}",
                "field": 100.0,
                "temperature": 295.0,
                "values": {"a0": a0},
                "errors": {"a0": a0_err},
                "chi_squared": chi2,
                "model_name": "Oscillatory (Ag mask)",
                "custom_values": {_STEERING_CUSTOM_KEY: f"{current:.2f}"},
            }
            for run, current, a0, a0_err, chi2 in _STEERING_ROWS
        ]

        panel = FitParametersPanel()
        # Register the manual steering-current column as an available trend x-axis
        # (the same route the data browser feeds a user-entered column through),
        # load the per-run a₀ series, then select the manual column as the x-axis.
        panel.set_custom_x_fields([("Steering current (A)", _STEERING_CUSTOM_KEY)])
        panel.load_representation_series(
            [(batch_id, "a₀ vs steering current", row_dicts)],
            select_id=batch_id,
        )
        idx = panel._x_combo.findData(_STEERING_CUSTOM_KEY)
        if idx >= 0:
            panel._x_combo.setCurrentIndex(idx)
        _pump_events(80)

        # Run the real polynomial trend fit and inject it as the active model
        # fit for a₀ (the same injection route parameter_trending_mgb2 uses),
        # so the fitted curve — minimum ≈ −0.06 A — overlays the trend points.
        panel._model_fits["a0"] = _build_steering_polynomial_fit()
        panel._sync_active_group_state()
        panel._refresh_model_fit_button_labels()
        _pump_events(80)
        return panel

    def settle(self, widget: QWidget) -> None:
        # Trigger the trend redraw now the panel is shown, then wait for the
        # off-thread overlay-curve sampler (TaskRunner worker) to populate the
        # cache so the fitted curve is present in the grab, not a blank overlay.
        _pump_events(150)
        widget._refresh_plot()
        _pump_until(
            lambda: (
                not widget._trend_curve_compute_active
                and widget._precomputed_trend_curves is not None
            ),
            timeout_ms=20_000,
        )
        _pump_events(200)


register(BasicsGroupingScenario())
register(BasicsAlphaScenario())
register(BasicsDeadtimeScenario())
register(BasicsT0Scenario())
register(BasicsSteeringScenario())

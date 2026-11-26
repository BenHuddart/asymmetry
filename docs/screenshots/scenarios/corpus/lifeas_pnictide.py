"""Counts-first reduction and fit diagnostics for the LiFeAs 40 mT scan."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import TYPE_CHECKING

import numpy as np

from asymmetry.core.data.dataset import MuonDataset, Run
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fitting.results import FitResult
from asymmetry.core.fitting.sc import GaussianWidth
from asymmetry.core.fitting.sc.constants import GAMMA_MU_RAD_S_PER_T
from asymmetry.core.utils.constants import MUON_GYROMAGNETIC_RATIO_MHZ_PER_T
from asymmetry.core.workflow.reduction import ReductionSettings, load_reduction_source, reduce_run

from ._corpus import CorpusScenario, corpus_path, register

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from .._base import CaptureContext

EXAMPLE = "Superconductivity/LiFeAs"
_SCAN_S1 = (
    (3366, 1.5),
    (3367, 4.0),
    (3368, 7.0),
    (3369, 10.0),
    (3370, 12.0),
    (3371, 14.0),
    (3372, 16.0),
    (3373, 18.0),
)
_S1_BASE = _SCAN_S1[0][0]
_S1_NORMAL = _SCAN_S1[-1][0]
_LARMOR_FREQ_MHZ = MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * 0.040
_FIT_WINDOW_US = (0.05, 6.0)


def _load_lifeas_run(run_number: int) -> Run:
    filename = f"deltat_pta_gps_{run_number}.bin"
    return load_reduction_source([corpus_path(f"{EXAMPLE}/data/{filename}")], None).run


def reduce_lifeas_run(
    run: Run,
    *,
    pair: tuple[str, str] | None = ("Up", "Down"),
    rebin: int = 10,
    time_range: tuple[float, float] = _FIT_WINDOW_US,
) -> MuonDataset:
    """Apply the file's time zero and good bins, then select the merged-bin centres."""
    return reduce_run(
        run,
        ReductionSettings(
            pair=pair,
            alpha=1.0,
            alpha_source="assumed",
            deadtime="off",
            background="none",
            rebin=rebin,
            t_min=time_range[0],
            t_max=time_range[1],
        ),
    )


def vortex_field_width_mt(
    fit: FitResult,
    reference: FitResult,
    *,
    sigma_parameter: str,
    width: GaussianWidth,
) -> float | None:
    """Convert positive quadrature excess to mT (Pratt et al., Eq. 2).

    A numerical excess between converged fits is conditional on both models;
    fit adequacy and uncertainty must be assessed before using it as a result.
    """
    if not (fit.success and reference.success):
        return None
    sigma = fit.parameters[sigma_parameter].value
    sigma_n = reference.parameters["sigma"].value
    if not (isfinite(sigma) and 0.0 <= sigma_n < sigma):
        return None
    gamma_rad_us_mt = GAMMA_MU_RAD_S_PER_T * 1e-9
    return width.value * sqrt((sigma - sigma_n) * (sigma + sigma_n)) / gamma_rad_us_mt


@dataclass(frozen=True)
class _ScanFit:
    run_number: int
    temperature: float
    result: FitResult


def _fit_lifeas_dataset(dataset: MuonDataset, *, normal: bool) -> FitResult:
    """Fit precession with Gaussian damping and a constant detector offset."""
    from asymmetry.core.fitting.engine import FitEngine

    freq = _LARMOR_FREQ_MHZ
    offset = float(np.median(dataset.asymmetry))
    if normal:
        model = CompositeModel(["Oscillatory", "Gaussian", "Constant"], operators=["*", "+"])
        parameters = [
            Parameter("A_1", 12.0, min=0.0, max=40.0),
            Parameter("frequency", freq, min=freq - 0.30, max=freq + 0.30),
            Parameter("phase", 0.0),
            Parameter("sigma", 0.2, min=0.0, max=4.0),
        ]
    else:
        # The narrower precession component models the weakly damped Ag support.
        model = CompositeModel(
            ["Oscillatory", "Gaussian", "Oscillatory", "Gaussian", "Constant"],
            operators=["*", "+", "*", "+"],
        )
        parameters = [
            Parameter("A_1", 12.0, min=0.0, max=40.0),
            Parameter("frequency_1", freq, min=freq - 0.25, max=freq + 0.25),
            Parameter("phase_1", 0.0, min=-3.2, max=3.2),
            Parameter("sigma_2", 1.05, min=0.05, max=4.0),
            Parameter("A_3", 3.0, min=0.0, max=20.0),
            Parameter("frequency_3", freq, min=freq - 0.25, max=freq + 0.25),
            Parameter("phase_3", 0.0, min=-3.2, max=3.2),
            Parameter("sigma_4", 0.12, min=0.0, max=0.5),
        ]
    return FitEngine().fit(
        dataset, model.function, ParameterSet([*parameters, Parameter("A_bg", offset)])
    )


def _fit_sample_scan() -> list[_ScanFit]:
    return [
        _ScanFit(
            run_number,
            temperature,
            _fit_lifeas_dataset(
                reduce_lifeas_run(_load_lifeas_run(run_number)), normal=run_number == _S1_NORMAL
            ),
        )
        for run_number, temperature in _SCAN_S1
    ]


class LifeasPairSelectScenario(CorpusScenario):
    name = "corpus_lifeas_pair_select"
    description = "Counts-first comparison of the beam-axis and Up/Down pair for GPS run 3366."
    example = EXAMPLE

    def capture(self, ctx: CaptureContext):
        from matplotlib.figure import Figure

        run = _load_lifeas_run(_S1_BASE)
        window = (_FIT_WINDOW_US[0], 2.5)
        figure = Figure(figsize=(8.6, 5.4), dpi=132, layout="constrained")
        top, bottom = figure.subplots(2, 1, sharex=True, sharey=True)
        for ax, pair, label, colour in (
            (top, None, "Beam-axis pair (Back and Forw)", "0.4"),
            (bottom, ("Up", "Down"), "Up/Down pair", "C0"),
        ):
            dataset = reduce_lifeas_run(run, pair=pair, time_range=window)
            ax.plot(dataset.time, dataset.asymmetry, color=colour, lw=1)
            ax.set_title(label, loc="left")
            ax.set_ylabel("Asymmetry (%)")
            ax.grid(alpha=0.2)
        bottom.set(xlabel="Time (µs)", xlim=window)
        figure.suptitle("LiFeAs Sample 1 at 1.5 K and 40 mT")
        return _save_figure(figure, ctx, self.name)


class LifeasTfFitScenario(CorpusScenario):
    name = "corpus_lifeas_tf_fit"
    description = "Normalised residuals for the 1.5 K and 18 K fits to LiFeAs Sample 1."
    example = EXAMPLE
    requires_fit = True

    def capture(self, ctx: CaptureContext):
        from matplotlib.figure import Figure

        figure = Figure(figsize=(8.6, 5.4), dpi=132, layout="constrained")
        axes = figure.subplots(2, 1, sharex=True, sharey=True)
        for ax, (run_number, temp), model in zip(
            axes, (_SCAN_S1[0], _SCAN_S1[-1]), ("two Gaussians", "one Gaussian"), strict=True
        ):
            normal = run_number == _S1_NORMAL
            dataset = reduce_lifeas_run(_load_lifeas_run(run_number))
            fit = _fit_lifeas_dataset(dataset, normal=normal)
            if fit.success and fit.residuals is not None:
                ax.plot(fit.residual_time, fit.residuals / fit.residual_sigma, lw=0.8)
                status = rf"$\chi^2_\mathrm{{red}} = {fit.reduced_chi_squared:.2f}$"
            else:
                status = "Fit failed" if not fit.success else "Residuals unavailable"
            ax.axhline(0, color="0.4", lw=0.8)
            ax.set_title(f"Run {run_number}, {temp:g} K ({model})", loc="left")
            ax.set_title(status, loc="right")
            ax.set_ylabel("Normalised residual")
            ax.grid(alpha=0.2)
        axes[-1].set(xlabel="Time (µs)", xlim=_FIT_WINDOW_US)
        figure.suptitle("Fit residuals for LiFeAs Sample 1 at 40 mT")
        return _save_figure(figure, ctx, self.name)


class LifeasFitQualityScenario(CorpusScenario):
    name = "corpus_lifeas_fit_quality"
    description = "Convergence and reduced chi-squared across the 40 mT LiFeAs temperature scan."
    example = EXAMPLE
    requires_fit = True

    def capture(self, ctx: CaptureContext):
        from matplotlib.figure import Figure

        scan = _fit_sample_scan()
        reference = scan[-1].result
        figure = Figure(figsize=(8.6, 3.7), dpi=132, layout="constrained")
        ax = figure.subplots()
        markers = {
            "Positive quadrature excess": ("o", "C0"),
            "No positive excess": ("v", "C1"),
            "Normal reference": ("s", "0.35"),
            "Reference unavailable": ("D", "C3"),
        }
        plotted = set()
        for entry in scan:
            fit = entry.result
            if not fit.success or not isfinite(fit.reduced_chi_squared):
                status = "Fit failed" if not fit.success else "No reduced χ²"
                ax.text(
                    entry.temperature,
                    0.02,
                    status,
                    ha="center",
                    va="bottom",
                    fontsize="small",
                    transform=ax.get_xaxis_transform(),
                )
                continue
            if entry.run_number == _S1_NORMAL:
                label = "Normal reference"
            elif not reference.success:
                label = "Reference unavailable"
            elif (
                vortex_field_width_mt(
                    fit,
                    reference,
                    sigma_parameter="sigma_2",
                    width=GaussianWidth.GAUSSIAN_COMPONENT,
                )
                is None
            ):
                label = "No positive excess"
            else:
                label = "Positive quadrature excess"
            marker, colour = markers[label]
            ax.scatter(
                entry.temperature,
                fit.reduced_chi_squared,
                marker=marker,
                color=colour,
                label=label if label not in plotted else None,
            )
            plotted.add(label)
        ax.axhline(1, color="0.5", lw=0.8, ls="--")
        ax.set(
            title="LiFeAs Sample 1 at 40 mT",
            xlabel="Temperature (K)",
            ylabel=r"Reduced $\chi^2$",
            xticks=[entry.temperature for entry in scan],
            ylim=(0, None),
        )
        ax.legend()
        ax.grid(alpha=0.2)
        return _save_figure(figure, ctx, self.name)


class LifeasVortexLineshapeScenario(CorpusScenario):
    name = "corpus_lifeas_vortex_lineshape"
    description = "Peak-normalised FFT magnitudes of the 1.5 K and 18 K LiFeAs traces."
    example = EXAMPLE

    def capture(self, ctx: CaptureContext):
        from matplotlib.figure import Figure

        from asymmetry.core.fourier.fft import fft_asymmetry

        figure = Figure(figsize=(8.6, 4.0), dpi=132, layout="constrained")
        ax = figure.subplots()
        for (run_number, temperature), colour in zip(
            (_SCAN_S1[-1], _SCAN_S1[0]), ("C1", "C0"), strict=True
        ):
            dataset = reduce_lifeas_run(_load_lifeas_run(run_number))
            frequency, _real, magnitude = fft_asymmetry(dataset, padding_factor=8)
            band = (frequency >= 3) & (frequency <= 8)
            ax.plot(
                frequency[band],
                magnitude[band] / np.max(magnitude[band]),
                color=colour,
                label=f"{temperature:g} K (run {run_number})",
            )
        ax.set(
            title="Transverse spectra of LiFeAs Sample 1 at 40 mT",
            xlabel="Frequency (MHz)",
            ylabel="FFT magnitude (peak-normalised)",
            xlim=(3, 8),
            ylim=(0, 1.08),
        )
        ax.axvline(_LARMOR_FREQ_MHZ, color="0.5", lw=0.8, ls="--", label="40 mT Larmor frequency")
        ax.legend()
        ax.grid(alpha=0.2)
        return _save_figure(figure, ctx, self.name)


def _save_figure(figure: Figure, ctx: CaptureContext, name: str):
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    from .._base import _optimize_png

    out_path = ctx.output_dir / f"{name}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    FigureCanvasAgg(figure).print_png(out_path)
    _optimize_png(out_path)
    return out_path


register(LifeasPairSelectScenario())
register(LifeasTfFitScenario())
register(LifeasFitQualityScenario())
register(LifeasVortexLineshapeScenario())

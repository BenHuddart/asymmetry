"""Fit outcomes retain their objective, selected support and observation report."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.ratio_validity import ObservationReport
from asymmetry.core.fitting.likelihood import Objective
from asymmetry.core.fitting.parameters import ParameterSet
from asymmetry.core.transform.deadtime import DeadtimeCalibration


@dataclass
class FitResult:
    """Container for the outcome of a fit.

    Scale: :attr:`residuals`, and every asymmetry-valued entry of
    :attr:`parameters` / :attr:`uncertainties` (an ``A0``, ``A``, ``baseline``),
    are **on the scale of the data that was fitted** — percent (0–100) for a
    :class:`~asymmetry.core.data.dataset.MuonDataset` and for the built-in models'
    seeds. Nothing here rescales, so a fit run on fraction-scale arrays reports
    fraction-scale amplitudes. See "Asymmetry units across the API" in the
    documentation.
    """

    success: bool
    objective: Objective
    chi_squared: float = 0.0
    reduced_chi_squared: float = 0.0
    parameters: ParameterSet = field(default_factory=ParameterSet)
    uncertainties: dict[str, float] = field(default_factory=dict)
    covariance: NDArray[np.float64] | None = None
    covariance_parameters: list[str] = field(default_factory=list)
    #: ``data − model`` over the fitted window, on the **data's** asymmetry scale.
    residuals: NDArray[np.float64] | None = None
    residual_time: NDArray[np.float64] | None = None
    residual_sigma: NDArray[np.float64] | None = None
    message: str = ""
    function_calls: int = 0
    gradient_calls: int = 0
    hessian_calls: int = 0
    edm: float | None = None
    covariance_accurate: bool = False
    #: Degrees of freedom ν = N_data − N_free for this (sub)fit. Used by the χ²
    #: quality verdict; 0 means "unknown" and callers fall back to inference.
    dof: int = 0
    #: Opt-in MINOS asymmetric 1σ intervals, ``{param: (lower, upper)}`` with
    #: ``lower < 0 < upper`` (iminuit's signed offsets). ``None`` when MINOS was
    #: not requested or every scan failed. A *display-only* overlay — the
    #: symmetric HESSE :attr:`uncertainties` are unchanged and remain the value
    #: every downstream surface (trends, export, propagation, promote) consumes.
    minos_errors: dict[str, tuple[float, float]] | None = None
    parameter_history: dict | None = None
    #: Fit warnings are both retained here and emitted through Python warnings.
    warnings: list[str] = field(default_factory=list)
    correction_metadata: dict = field(default_factory=dict)
    deadtime_calibration: DeadtimeCalibration | None = None
    #: Minimum over the whole minimized objective, shared by coupled members.
    minimum_expected_count: float | None = None
    observation_report: ObservationReport | None = None
    input_attribution: dict | None = None
    computation_revision: str | None = None

    def __post_init__(self):
        self.objective = Objective(self.objective)
        if self.observation_report is not None and self.observation_report.omitted:
            message = self.observation_report.summary()
            if message not in self.warnings:
                self.warnings.append(message)
        if self.minimum_expected_count is not None and (
            self.objective is not Objective.POISSON
            or not math.isfinite(self.minimum_expected_count)
            or self.minimum_expected_count < 0
        ):
            raise ValueError(
                "minimum expected count requires a finite non-negative Poisson expectation"
            )

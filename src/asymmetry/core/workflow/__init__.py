"""Scriptable session façade over the analysis engine.

The layer an automated caller — a CLI, a notebook, an agent — drives instead
of assembling grouping profiles and reduction calls by hand. Every function
takes typed inputs, returns JSON-serialisable results, and mirrors what the
desktop application does for the same step, so a scripted analysis and a
GUI analysis of the same runs agree.

Pure core: no Qt, no matplotlib, no ``asymmetry.gui``.
"""

from asymmetry.core.workflow.fourier import FourierOutcome, FourierSettings, fourier_spectrum
from asymmetry.core.workflow.global_fit import GlobalFitOutcome, fit_global
from asymmetry.core.workflow.integral_scan import (
    build_integral_scan,
    field_scan_payload,
    fit_integral_scan,
)
from asymmetry.core.workflow.recipe import FitRecipe, RecipeParameter
from asymmetry.core.workflow.reduction import (
    BACKGROUND_MODES,
    DEADTIME_MODES,
    GREEN_MINUS_RED,
    AlphaEstimate,
    ReductionSettings,
    estimate_alpha_for_run,
    reduce_run,
    reduction_source,
    resolve_reduction_grouping,
)
from asymmetry.core.workflow.screen import (
    GEOMETRY_SOURCES,
    SCOPE_PRESETS,
    ScreenCandidate,
    ScreenResult,
    resolve_geometry,
    screen_run,
)
from asymmetry.core.workflow.series import (
    ORDER_KEYS,
    ScanAxis,
    SeriesBranch,
    SeriesOutcome,
    TrendTable,
    build_trend_table,
    fit_one,
    fit_series,
    scan_axis,
    supplied_axis,
)
from asymmetry.core.workflow.survey import (
    LARMOR_FREQUENCY_TOLERANCE,
    PRECESSION_SNR_FLOOR,
    PRECESSION_STATES,
    ROW_GEOMETRY_SOURCES,
    AlphaStep,
    CalibrationCandidate,
    FolderSurvey,
    PrecessionEvidence,
    RunRow,
    ScanGroup,
    build_run_row,
    calibration_verdict,
    has_file_deadtime,
    precession_evidence,
    resolve_row_geometry,
    run_facility,
    run_geometry,
    survey_folder,
)
from asymmetry.core.workflow.trend_fit import TrendFitOutcome, fit_trend
from asymmetry.core.workflow.workdir import (
    SCHEMA,
    WORKDIR_NAME,
    ReducedEntry,
    RunSelection,
    WorkDir,
    WorkDirMismatchError,
    file_fingerprint,
    instrument_name,
    reduction_digest,
)

__all__ = [
    "BACKGROUND_MODES",
    "GREEN_MINUS_RED",
    "DEADTIME_MODES",
    "GEOMETRY_SOURCES",
    "LARMOR_FREQUENCY_TOLERANCE",
    "ORDER_KEYS",
    "PRECESSION_SNR_FLOOR",
    "PRECESSION_STATES",
    "ROW_GEOMETRY_SOURCES",
    "SCHEMA",
    "SCOPE_PRESETS",
    "WORKDIR_NAME",
    "AlphaEstimate",
    "AlphaStep",
    "CalibrationCandidate",
    "FitRecipe",
    "FourierOutcome",
    "FourierSettings",
    "GlobalFitOutcome",
    "FolderSurvey",
    "PrecessionEvidence",
    "RecipeParameter",
    "ReducedEntry",
    "ReductionSettings",
    "RunRow",
    "RunSelection",
    "ScanAxis",
    "ScanGroup",
    "ScreenCandidate",
    "ScreenResult",
    "SeriesBranch",
    "SeriesOutcome",
    "TrendFitOutcome",
    "TrendTable",
    "WorkDir",
    "WorkDirMismatchError",
    "build_run_row",
    "build_integral_scan",
    "build_trend_table",
    "calibration_verdict",
    "estimate_alpha_for_run",
    "file_fingerprint",
    "instrument_name",
    "fit_one",
    "field_scan_payload",
    "fit_integral_scan",
    "fit_global",
    "fourier_spectrum",
    "fit_series",
    "fit_trend",
    "has_file_deadtime",
    "precession_evidence",
    "reduce_run",
    "reduction_source",
    "reduction_digest",
    "resolve_geometry",
    "resolve_reduction_grouping",
    "resolve_row_geometry",
    "run_facility",
    "run_geometry",
    "scan_axis",
    "screen_run",
    "supplied_axis",
    "survey_folder",
]

"""Scriptable session façade over the analysis engine.

The layer an automated caller — a CLI, a notebook, an agent — drives instead
of assembling grouping profiles and reduction calls by hand. Every function
takes typed inputs, returns JSON-serialisable results, and mirrors what the
desktop application does for the same step, so a scripted analysis and a
GUI analysis of the same runs agree.

Pure core: no Qt, no matplotlib, no ``asymmetry.gui``.
"""

from asymmetry.core.workflow.recipe import FitRecipe, RecipeParameter
from asymmetry.core.workflow.reduction import (
    BACKGROUND_MODES,
    DEADTIME_MODES,
    AlphaEstimate,
    ReductionSettings,
    estimate_alpha_for_run,
    reduce_run,
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
    SeriesOutcome,
    TrendTable,
    build_trend_table,
    fit_one,
    fit_series,
    order_values,
)
from asymmetry.core.workflow.survey import (
    CalibrationCandidate,
    FolderSurvey,
    RunRow,
    ScanGroup,
    build_run_row,
    has_file_deadtime,
    run_facility,
    run_geometry,
    survey_folder,
)
from asymmetry.core.workflow.workdir import (
    SCHEMA,
    WORKDIR_NAME,
    ReducedEntry,
    WorkDir,
    file_fingerprint,
    reduction_digest,
)

__all__ = [
    "BACKGROUND_MODES",
    "DEADTIME_MODES",
    "GEOMETRY_SOURCES",
    "ORDER_KEYS",
    "SCHEMA",
    "SCOPE_PRESETS",
    "WORKDIR_NAME",
    "AlphaEstimate",
    "CalibrationCandidate",
    "FitRecipe",
    "FolderSurvey",
    "RecipeParameter",
    "ReducedEntry",
    "ReductionSettings",
    "RunRow",
    "ScanGroup",
    "ScreenCandidate",
    "ScreenResult",
    "SeriesOutcome",
    "TrendTable",
    "WorkDir",
    "build_run_row",
    "build_trend_table",
    "estimate_alpha_for_run",
    "file_fingerprint",
    "fit_one",
    "fit_series",
    "has_file_deadtime",
    "order_values",
    "reduce_run",
    "reduction_digest",
    "resolve_geometry",
    "resolve_reduction_grouping",
    "run_facility",
    "run_geometry",
    "screen_run",
    "survey_folder",
]

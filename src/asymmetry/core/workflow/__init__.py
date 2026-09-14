"""Scriptable session façade over the analysis engine.

The layer an automated caller — a CLI, a notebook, an agent — drives instead
of assembling grouping profiles and reduction calls by hand. Every function
takes typed inputs, returns JSON-serialisable results, and mirrors what the
desktop application does for the same step, so a scripted analysis and a
GUI analysis of the same runs agree.

Pure core: no Qt, no matplotlib, no ``asymmetry.gui``.
"""

from asymmetry.core.workflow.reduction import (
    BACKGROUND_MODES,
    DEADTIME_MODES,
    AlphaEstimate,
    ReductionSettings,
    estimate_alpha_for_run,
    reduce_run,
    resolve_reduction_grouping,
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
    "SCHEMA",
    "WORKDIR_NAME",
    "AlphaEstimate",
    "CalibrationCandidate",
    "FolderSurvey",
    "ReducedEntry",
    "ReductionSettings",
    "RunRow",
    "ScanGroup",
    "WorkDir",
    "build_run_row",
    "estimate_alpha_for_run",
    "file_fingerprint",
    "has_file_deadtime",
    "reduce_run",
    "reduction_digest",
    "resolve_reduction_grouping",
    "run_facility",
    "run_geometry",
    "survey_folder",
]

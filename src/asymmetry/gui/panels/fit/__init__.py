"""Fit-panel widgets and shared GUI table controls."""

from __future__ import annotations

# Import submodules in dependency order (leaf first) so the package initializes
# without cycles.
from . import global_tab, panel, single_tab, tab_base  # noqa: F401

# Public classes / constants.
from .global_tab import (
    BATCH_SEEDING_LABELS,
    BATCH_SEEDING_MODES,
    BATCH_SEEDING_TOOLTIP,
    GlobalFitTab,
    SeriesCatalogue,
    SeriesMenuEntry,
)
from .panel import FitPanel
from .single_tab import SavedFitCatalogue, SavedFitEntry, SingleFitTab
from .tab_base import (
    _PARAM_BATCH_ROLE_DATA,
    _SINGLE_PARAM_BATCH_COLUMN,
    _SINGLE_PARAM_LINK_COLUMN,
    _SINGLE_PARAM_TIE_COLUMN,
    AffineTieDialog,
    FitParameterTable,
    FitTabBase,
    QMessageBox,
    _CommitOnTabDelegate,
    _dataset_representation_domain,
    _fit_domain_mismatch_message,
    _format_fit_worker_exception,
    _set_tie_button_value,
    _shift_rrf_parameters,
    _start_fit_call,
    _tie_button_value,
    _ValueUncertaintyDelegate,
)

__all__ = [
    "AffineTieDialog",
    "BATCH_SEEDING_LABELS",
    "BATCH_SEEDING_MODES",
    "BATCH_SEEDING_TOOLTIP",
    "FitPanel",
    "FitParameterTable",
    "FitTabBase",
    "GlobalFitTab",
    "QMessageBox",
    "SeriesCatalogue",
    "SeriesMenuEntry",
    "SavedFitCatalogue",
    "SavedFitEntry",
    "SingleFitTab",
    "_CommitOnTabDelegate",
    "_PARAM_BATCH_ROLE_DATA",
    "_SINGLE_PARAM_BATCH_COLUMN",
    "_SINGLE_PARAM_LINK_COLUMN",
    "_SINGLE_PARAM_TIE_COLUMN",
    "_ValueUncertaintyDelegate",
    "_dataset_representation_domain",
    "_fit_domain_mismatch_message",
    "_format_fit_worker_exception",
    "_set_tie_button_value",
    "_shift_rrf_parameters",
    "_start_fit_call",
    "_tie_button_value",
]

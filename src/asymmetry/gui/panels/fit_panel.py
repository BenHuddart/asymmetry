"""Deprecated shim — ``fit_panel`` was split into the ``fit`` package.

Import from :mod:`asymmetry.gui.panels.fit` (or its submodules) instead. This
module re-exports the public API for backward compatibility and will be retired
once callers migrate (see ``docs/audit/shared-foundations/FOLLOW-UPS.md``).
"""

from __future__ import annotations

from asymmetry.gui.panels.fit import (  # noqa: F401
    _PARAM_BATCH_ROLE_DATA,
    _SINGLE_PARAM_BATCH_COLUMN,
    _SINGLE_PARAM_LINK_COLUMN,
    _SINGLE_PARAM_TIE_COLUMN,
    BATCH_SEEDING_LABELS,
    BATCH_SEEDING_MODES,
    BATCH_SEEDING_TOOLTIP,
    AffineTieDialog,
    FitPanel,
    FitParameterTable,
    GlobalFitTab,
    QMessageBox,
    SingleFitTab,
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
    "GlobalFitTab",
    "QMessageBox",
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

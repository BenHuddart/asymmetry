"""Fit-panel package.

Phase 2 mechanical split of the former monolithic ``fit_panel.py`` into
dependency-ordered submodules:

``seeding`` (leaf) → ``tab_base`` → ``single_tab`` / ``global_tab`` → ``panel``.

No logic changed in the split; see ``docs/audit/shared-foundations``. The public
API is re-exported here (and, for backward compatibility, through the thin
``asymmetry.gui.panels.fit_panel`` shim).

The genuinely public surface most callers need is small: ``FitPanel``
(container), ``SingleFitTab``/``GlobalFitTab`` (the two tabs),
``FitParameterTable``/``FitTabBase`` (shared building blocks for a new tab),
and the ``BATCH_SEEDING_*`` constants. Everything else in ``__all__`` —
underscore-prefixed helpers from ``tab_base``/``seeding`` — is re-exported
only so the ``fit_panel.py`` shim and a handful of existing call sites
(e.g. ``multi_group_fit_window.py``, tests) keep working unchanged after the
split; new code should import those directly from their owning submodule
(``panels.fit.tab_base`` / ``panels.fit.seeding``) rather than from the
package root. See ``docs/audit/shared-foundations/FOLLOW-UPS.md`` for the
tracked follow-up to migrate those call sites and retire the shim.
"""

from __future__ import annotations

# Import submodules in dependency order (leaf first) so the package initializes
# without cycles.
from . import global_tab, panel, seeding, single_tab, tab_base  # noqa: F401

# Public classes / constants.
from .global_tab import (
    BATCH_SEEDING_LABELS,
    BATCH_SEEDING_MODES,
    BATCH_SEEDING_TOOLTIP,
    GlobalFitTab,
)
from .panel import FitPanel
from .seeding import (
    _MAX_PHASE_SEED_FFT_POINTS,
    _bounded_phase_seed_padding,
    _seed_group_absolute_phases,
    _seed_group_background_and_n0,
)
from .single_tab import SingleFitTab
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
    _fit_curve_sample_count,
    _fit_domain_mismatch_message,
    _format_fit_worker_exception,
    _get_file_value_for_parameter,
    _model_without_trailing_background,
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
    "SingleFitTab",
    "_CommitOnTabDelegate",
    "_MAX_PHASE_SEED_FFT_POINTS",
    "_PARAM_BATCH_ROLE_DATA",
    "_SINGLE_PARAM_BATCH_COLUMN",
    "_SINGLE_PARAM_LINK_COLUMN",
    "_SINGLE_PARAM_TIE_COLUMN",
    "_ValueUncertaintyDelegate",
    "_bounded_phase_seed_padding",
    "_dataset_representation_domain",
    "_fit_curve_sample_count",
    "_fit_domain_mismatch_message",
    "_format_fit_worker_exception",
    "_get_file_value_for_parameter",
    "_model_without_trailing_background",
    "_seed_group_absolute_phases",
    "_seed_group_background_and_n0",
    "_set_tie_button_value",
    "_shift_rrf_parameters",
    "_start_fit_call",
    "_tie_button_value",
]

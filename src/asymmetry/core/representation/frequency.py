"""Frequency-domain representations: FFT and (reserved) MaxEnt.

``FrequencyFFT`` generates the averaged grouped-FFT spectrum from the recipe's
Fourier configuration, delegating to the shared
:func:`asymmetry.core.fourier.spectrum.compute_average_group_spectrum` so a
generated spectrum and a recipe-recomputed spectrum are identical by
construction.  ``FrequencyMaxEnt`` is a registered placeholder whose
``compute`` raises until the method is implemented.

The recipe's ``fourier_config`` carries the concrete generation settings
(window/display/padding/phase/t0/filter, the selected groups, and resolved
per-group phases).
"""

from __future__ import annotations

from typing import Any

from asymmetry.core.data.dataset import MuonDataset, Run
from asymmetry.core.fourier.spectrum import (
    GroupSpectrumConfig,
    compute_average_group_spectrum,
)
from asymmetry.core.maxent import MaxEntConfig, apply_maxent_specbg, maxent
from asymmetry.core.representation.base import Representation, RepresentationType

# Re-exported for callers that historically imported it from this module; the
# single implementation now lives beside the SpecBG model in core.maxent.specbg
# and is applied through MaxEntResult.as_dataset (one application point).
__all__ = ["FrequencyFFT", "FrequencyMaxEnt", "apply_maxent_specbg"]


def _resolve_selected_group_ids(run: Run, config: dict) -> list[int] | None:
    """Resolve the included group ids from a recipe ``fourier_config``.

    An explicit ``selected_group_ids`` list wins; otherwise a
    ``group_enabled_table`` (as carried by migrated v5 projects) is reduced
    against the run's groups; otherwise ``None`` (meaning all groups).
    """
    selected = config.get("selected_group_ids")
    if isinstance(selected, list):
        return [int(g) for g in selected]

    enabled = config.get("group_enabled_table")
    if isinstance(enabled, dict):
        grouping = run.grouping if isinstance(run.grouping, dict) else {}
        groups = grouping.get("groups")
        if not isinstance(groups, dict):
            return None
        ids: list[int] = []
        for raw_id in groups:
            try:
                gid = int(raw_id)
            except (TypeError, ValueError):
                continue
            if bool(enabled.get(gid, enabled.get(str(gid), True))):
                ids.append(gid)
        return sorted(ids)

    return None


class FrequencyFFT(Representation):
    """Averaged grouped fast-Fourier-transform spectrum for a run.

    Recipe::

        {"fourier_config": {display, window, padding, phase/t0/filter settings,
                            selected_group_ids, group_phase_degrees}}
    """

    rep_type = RepresentationType.FREQ_FFT

    def fourier_config(self) -> dict[str, Any]:
        """Return the raw ``fourier_config`` recipe block (possibly empty)."""
        config = self.recipe.get("fourier_config")
        return dict(config) if isinstance(config, dict) else {}

    def compute(self, run: Run, *, context: Any = None) -> list[MuonDataset]:
        if not run.histograms:
            raise ValueError("FFT representation requires detector histograms.")
        config_dict = self.fourier_config()
        config = GroupSpectrumConfig.from_dict(config_dict)
        config.selected_group_ids = _resolve_selected_group_ids(run, config_dict)
        spectrum = compute_average_group_spectrum(run, config)
        if spectrum is None:
            raise ValueError("FFT representation produced no spectrum (no enabled groups).")
        return [spectrum]


class FrequencyMaxEnt(Representation):
    """Grouped maximum-entropy spectrum for a run.

    Recipe::

        {"maxent_config": {n_spectrum_points, window, phase/background settings,
                           selected_group_ids, group_phase_degrees}}
    """

    rep_type = RepresentationType.FREQ_MAXENT

    #: MaxEnt is an expensive iterative reconstruction: recomputing it
    #: synchronously during project load would freeze the GUI with no
    #: progress/cancel and bypass the workload confirmation. It is recomputed
    #: on demand (GUI worker thread, or an explicit scripted call) instead.
    recompute_on_load = False

    def maxent_config(self) -> dict[str, Any]:
        """Return the raw ``maxent_config`` recipe block (possibly empty)."""
        config = self.recipe.get("maxent_config")
        return dict(config) if isinstance(config, dict) else {}

    def compute(self, run: Run, *, context: Any = None) -> list[MuonDataset]:
        if not run.histograms:
            raise ValueError("MaxEnt representation requires detector histograms.")
        config_dict = self.maxent_config()
        config = MaxEntConfig.from_dict(config_dict)
        if config.selected_group_ids is None:
            config.selected_group_ids = _resolve_selected_group_ids(run, config_dict)
        result = maxent(run, config)
        self.result_metadata = {
            "cycles": int(result.state.cycle),
            "diagnostics": result.diagnostics.to_dict(),
            "f_min_mhz": float(result.frequencies_mhz[0]) if result.frequencies_mhz.size else None,
            "f_max_mhz": float(result.frequencies_mhz[-1]) if result.frequencies_mhz.size else None,
        }
        return [result.as_dataset(run, config)]

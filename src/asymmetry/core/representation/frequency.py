"""Frequency-domain representations: FFT and (reserved) MaxEnt.

``FrequencyFFT`` generates one spectrum per enabled detector group from the
recipe's Fourier configuration, recomputing on demand so the spectra never need
to be serialised.  ``FrequencyMaxEnt`` is a registered placeholder so its nav
slot and fit pipeline exist; its ``compute`` raises until the method is
implemented.

The recipe's ``fourier_config`` mirrors the existing project ``fourier_state``
block (window/display/padding/phase/t0/filter and per-group enable tables).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from asymmetry.core.data.dataset import MuonDataset, Run
from asymmetry.core.fourier.fft import (
    canonical_fourier_display_mode,
    fft_complex_asymmetry,
    fourier_display_values,
)
from asymmetry.core.fourier.grouped import build_group_signal_dataset
from asymmetry.core.representation.base import Representation, RepresentationType

#: Default Fourier configuration used when the recipe omits a key.
DEFAULT_FOURIER_CONFIG: dict[str, Any] = {
    "window": "none",
    "padding": 1,
    "phase_degrees": 0.0,
    "t0_offset_us": 0.0,
    "display": "(Power)^1/2",
    "filter_start_us": 0.0,
    "filter_time_constant_us": 1.5,
    "subtract_average_signal": True,
    "group_enabled_table": {},
}


def _enabled_group_ids(run: Run, config: dict) -> list[int]:
    """Return the sorted group ids enabled for Fourier analysis."""
    grouping = run.grouping if isinstance(run.grouping, dict) else {}
    groups = grouping.get("groups")
    if not isinstance(groups, dict) or not groups:
        return []
    enabled_table = config.get("group_enabled_table")
    enabled_table = enabled_table if isinstance(enabled_table, dict) else {}

    ids: list[int] = []
    for raw_id in groups:
        try:
            gid = int(raw_id)
        except (TypeError, ValueError):
            continue
        flag = enabled_table.get(gid, enabled_table.get(str(gid), True))
        if bool(flag):
            ids.append(gid)
    return sorted(ids)


class FrequencyFFT(Representation):
    """Fast-Fourier-transform spectra for a run's detector groups.

    Recipe keys::

        {"fourier_config": {window, padding, phase_degrees, t0_offset_us,
                            display, filter_start_us, filter_time_constant_us,
                            subtract_average_signal, group_enabled_table}}
    """

    rep_type = RepresentationType.FREQ_FFT

    def fourier_config(self) -> dict[str, Any]:
        """Return the effective Fourier configuration (defaults merged in)."""
        config = dict(DEFAULT_FOURIER_CONFIG)
        recipe_config = self.recipe.get("fourier_config")
        if isinstance(recipe_config, dict):
            config.update(recipe_config)
        return config

    def compute(self, run: Run, *, context: Any = None) -> list[MuonDataset]:
        if not run.histograms:
            raise ValueError("FFT representation requires detector histograms.")
        config = self.fourier_config()
        group_ids = _enabled_group_ids(run, config)
        if not group_ids:
            raise ValueError("FFT representation has no enabled detector groups.")

        display = str(config.get("display", "(Power)^1/2"))
        spectra: list[MuonDataset] = []
        for group_id in group_ids:
            signal = build_group_signal_dataset(run, group_id)
            frequencies, spectrum = fft_complex_asymmetry(
                signal,
                window=str(config.get("window", "none")),
                padding_factor=max(1, int(config.get("padding", 1))),
                phase_degrees=float(config.get("phase_degrees", 0.0)),
                t0_offset_us=float(config.get("t0_offset_us", 0.0)),
                subtract_average_signal=bool(config.get("subtract_average_signal", True)),
                filter_start_us=float(config.get("filter_start_us", 0.0)),
                filter_time_constant_us=float(config.get("filter_time_constant_us", 1.5)),
            )
            values = fourier_display_values(spectrum, display=display)
            metadata = dict(signal.metadata)
            metadata.update(
                {
                    "plot_domain": "frequency",
                    "x_label": "Frequency (MHz)",
                    "y_label": f"FFT {display}",
                    "fourier_display": display,
                    "fourier_display_mode": canonical_fourier_display_mode(display),
                }
            )
            spectra.append(
                MuonDataset(
                    time=np.asarray(frequencies, dtype=float),
                    asymmetry=np.asarray(values, dtype=float),
                    error=np.zeros_like(np.asarray(values, dtype=float)),
                    metadata=metadata,
                    run=run,
                )
            )
        return spectra


class FrequencyMaxEnt(Representation):
    """Maximum-entropy spectra — reserved, not yet implemented.

    The type is registered so its navigation button and fit pipeline exist.
    Implementing MaxEnt requires only a working :meth:`compute` plus a
    frequency-domain fit-library entry — no schema or orchestration change.
    """

    rep_type = RepresentationType.FREQ_MAXENT

    def compute(self, run: Run, *, context: Any = None) -> list[MuonDataset]:
        raise NotImplementedError("Maximum-entropy spectra are not yet implemented.")

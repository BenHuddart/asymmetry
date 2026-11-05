"""Identity of actual prepared observations for resumable MaxEnt state."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from asymmetry.core.maxent.engine import MaxEntConfig, MaxEntInput


def state_signature(maxent_input: MaxEntInput, config: MaxEntConfig) -> tuple[Any, ...]:
    # ``config.backend`` is deliberately absent: the projection math is
    # backend-independent (results agree to solver tolerance), so a state built
    # on one backend must resume on another without a forced restart.
    data = hashlib.sha256()
    for group in maxent_input.groups:
        data.update(np.asarray([group.time_us.size], dtype="<i8").tobytes())
        mask = group.mask if group.mask is not None else np.ones(group.time_us.size, dtype=bool)
        data.update(np.asarray(mask, dtype=bool).tobytes())
        for values in (group.time_us, group.signal, group.sigma):
            data.update(np.asarray(values, dtype="<f8").tobytes())
    data.update(json.dumps(maxent_input.metadata, sort_keys=True, default=str).encode())
    return (
        data.hexdigest(),
        int(maxent_input.run_number),
        tuple(group.group_id for group in maxent_input.groups),
        int(maxent_input.n_spectrum_points),
        round(float(maxent_input.f_min_mhz), 12),
        round(float(maxent_input.f_max_mhz), 12),
        round(float(config.default_level), 12),
        bool(config.fit_phases),
        bool(config.fit_amplitudes),
        bool(config.fit_backgrounds),
        bool(config.fit_constant_background),
        # Data-preparation settings: changing any of these reshapes or rescales
        # the observed signals, so a resumed state would silently iterate a
        # stale spectrum against incompatible data.
        bool(config.use_deadtime_correction),
        None if config.t_min_us is None else round(float(config.t_min_us), 12),
        None if config.t_max_us is None else round(float(config.t_max_us), 12),
        int(config.time_binning_factor),
        # Pulse-shape response and the interior exclusion window both reshape the
        # forward model / the data weighting, so a resumed state with a changed
        # setting would iterate a stale spectrum against an incompatible model.
        str(config.pulse_mode),
        round(float(config.pulse_half_width_us), 12),
        round(float(config.pulse_separation_us), 12),
        None if config.exclude_t_min_us is None else round(float(config.exclude_t_min_us), 12),
        None if config.exclude_t_max_us is None else round(float(config.exclude_t_max_us), 12),
        # ZF/LF mode ties group amplitudes/backgrounds and pins phases, so it
        # reshapes the fit; a changed mode must invalidate a resumed state.
        str(config.mode),
        # Effective phase seeds: ``state.phases`` is seeded from these at
        # initialisation and a resumed state never re-reads the config, so an
        # edited seed must force a restart or it would be silently ignored
        # (with ``fit_phases`` off, permanently; with it on, the ±4°/cycle
        # refinement cannot follow a large seed change either).
        tuple(round(float(group.phase_degrees), 9) for group in maxent_input.groups),
    )

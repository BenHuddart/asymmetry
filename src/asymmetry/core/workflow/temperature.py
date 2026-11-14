"""Explicit logged thermometer preparation for scripts."""

from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.io.nexus import active_series_mean
from asymmetry.core.io.physical_units import canonical


def select_logged_temperature(dataset: MuonDataset, channel: str) -> MuonDataset:
    """Return a copy labelled by the arithmetic mean of an exact log channel.

    ``channel`` must be a full key in ``metadata['nexus_time_series']``; sensor
    labels and suffixes are never matched. Samples with time >= 0 are included,
    equally weighted, without interpolation or duration weighting. A non-empty
    one-dimensional sample list and a finite, equally sized time axis are required.
    Every included reading must be finite and above 0 K after declared-unit
    conversion. Missing, unknown, or incompatible units raise ``ValueError``.
    No header, other channel, or cached summary substitutes for missing readings.

    The returned dataset and Run expose the selected value through ``temperature``
    and ``sample_temperature_logged``. Raw logs and original scalar metadata remain
    available; ``temperature_selection`` records the channel, statistic, sample
    policy, and original scalar fields. Re-selection retains those original fields.
    Arrays and histograms are shared; metadata is detached. Loader defaults and
    the input are unchanged. This statistic does not establish thermal equilibrium.
    """
    series = dataset.metadata.get("nexus_time_series", {})
    if not channel or channel not in series:
        raise ValueError(
            f"Select an exact channel path; {channel!r} is unavailable. "
            f"Available paths: {', '.join(sorted(series)) or '<none>'}."
        )
    entry = series[channel]
    context = f"Logged temperature {channel!r}"
    values = np.asarray(entry.get("values", []), dtype=float)
    times = np.asarray(entry.get("time", []), dtype=float)
    if values.ndim != 1 or not values.size:
        raise ValueError(f"{context}: requires a non-empty one-dimensional values array.")
    if times.ndim != 1 or times.shape != values.shape:
        raise ValueError(f"{context}: time and values must have the same length.")
    if not np.all(np.isfinite(times)):
        raise ValueError(f"{context}: time must be finite.")
    active = times >= 0
    if not np.any(active):
        raise ValueError(f"{context}: no run-active samples (time >= 0).")
    kelvin, _ = canonical(values[active], entry.get("units"), "temperature", context=context)
    if not np.all(np.isfinite(kelvin)):
        raise ValueError(f"{context}: every run-active reading must be finite.")
    if np.any(kelvin <= 0):
        raise ValueError(f"{context}: every run-active reading must be above 0 K.")
    # Validated active samples prevent the shared summary's missing-data fallback.
    temperature = active_series_mean({"values": kelvin.tolist()})
    if temperature is None or not np.isfinite(temperature):
        raise ValueError(f"{context}: the arithmetic mean must be finite.")
    metadata = _selected_temperature_metadata(dataset.metadata, channel, temperature)
    run = dataset.run
    if run is not None:
        run = replace(
            run, metadata=_selected_temperature_metadata(run.metadata, channel, temperature)
        )
    return replace(dataset, metadata=metadata, run=run)


def _selected_temperature_metadata(
    metadata: dict[str, Any], channel: str, temperature: float
) -> dict[str, Any]:
    """Keep acquired scalar provenance when preparing or reselecting a thermometer."""
    prepared = deepcopy(metadata)
    original = (
        prepared["temperature_selection"]["original"]
        if "temperature_selection" in prepared
        else {
            key: prepared[key]
            for key in ("temperature", "sample_temperature_logged", "sample_temperature_log_source")
            if key in prepared
        }
    )
    prepared.update(
        temperature=temperature,
        sample_temperature_logged=temperature,
        sample_temperature_log_source=channel,
        temperature_selection={
            "statistic": "arithmetic_mean",
            "samples": "time >= 0; finite and above 0 K required",
            "original": original,
        },
    )
    return prepared


__all__ = ["select_logged_temperature"]

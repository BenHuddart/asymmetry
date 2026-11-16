"""Bounded reuse of fingerprints under the actual scientific input identity."""

from __future__ import annotations

from typing import TYPE_CHECKING

from asymmetry.core import computation
from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fourier.diagnostic import DiagnosticFitRecord
from asymmetry.core.transform.preparation import wizard_observation_identity

if TYPE_CHECKING:
    from asymmetry.core.fitting.fit_wizard import SpectrumFingerprint

_SERIES_FINGERPRINT_CACHE: dict[tuple[str, str, str | None], SpectrumFingerprint] = {}


def _cached_series_fingerprint(dataset: MuonDataset) -> SpectrumFingerprint:
    from asymmetry.core.fitting.fit_wizard import fingerprint_spectrum

    key = (
        computation.COMPUTATION_REVISION,
        wizard_observation_identity(dataset),
        dataset.diagnostic_observation.attribution.identity
        if isinstance(dataset, DiagnosticFitRecord)
        else None,
    )
    cached = _SERIES_FINGERPRINT_CACHE.get(key)
    if cached is not None:
        return cached
    fingerprint = fingerprint_spectrum(dataset)
    if len(_SERIES_FINGERPRINT_CACHE) >= 256:
        _SERIES_FINGERPRINT_CACHE.pop(next(iter(_SERIES_FINGERPRINT_CACHE)))
    _SERIES_FINGERPRINT_CACHE[key] = fingerprint
    return fingerprint

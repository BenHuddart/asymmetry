"""Retain historical spectral fit values when weighting is unrecorded."""

from asymmetry.core.fitting.spectral_weights import (
    SpectralWeighting,
    SpectralWeightPolicy,
    spectral_quality_reason,
    spectral_weighting_detail,
)
from asymmetry.core.representation.base import RepresentationType

FREQUENCY_REPRESENTATIONS = (RepresentationType.FREQ_FFT, RepresentationType.FREQ_MAXENT)


def _mark_unknown(record: dict) -> bool:
    attribution = record.get("input_attribution")
    if attribution is None:
        attribution = {}
    if not isinstance(attribution, dict) or "spectral_weighting" in attribution:
        return False
    attribution["spectral_weighting"] = SpectralWeighting(SpectralWeightPolicy.UNKNOWN).to_dict()
    record.update(
        input_attribution=attribution,
        quality=None,
        quality_reason=spectral_quality_reason(attribution),
    )
    return True


def _values(mapping):
    return mapping.values() if isinstance(mapping, dict) else ()


def migrate_historical_spectral_weights(data: dict) -> None:
    """The saved spectrum may have been recomputed after its fit was recorded."""
    for key in ("datasets", "combined_datasets"):
        for entry in data.get(key) or []:
            if not isinstance(entry, dict):
                continue
            representations = entry.get("representations")
            if not isinstance(representations, dict):
                continue
            for rep_type in FREQUENCY_REPRESENTATIONS:
                rep = representations.get(rep_type)
                if not isinstance(rep, dict):
                    continue
                for fit_set in _values(rep.get("single_fits")):
                    if not isinstance(fit_set, dict):
                        continue
                    for slot in fit_set.get("fits") or []:
                        result = slot.get("result") if isinstance(slot, dict) else None
                        if not isinstance(result, dict) or not result or not _mark_unknown(result):
                            continue
                        if historical_html := result.pop("result_html", None):
                            result["historical_result_html"] = historical_html
                        form = slot.get("ui_state")
                        if isinstance(form, dict) and form:
                            historical_html = form.get("result_html")
                            if historical_html:
                                result.setdefault("historical_result_html", historical_html)
                            form["result_html"] = spectral_weighting_detail(
                                result["input_attribution"]
                            )

    for series in data.get("batches") or []:
        if not isinstance(series, dict) or series.get("rep_type") not in FREQUENCY_REPRESENTATIONS:
            continue
        if series.get("canonical_model") is not None:
            for result in _values(series.get("results_by_run")):
                if isinstance(result, dict):
                    _mark_unknown(result)


def runs_with_cached_spectra(spectra_state: object) -> set[int]:
    """Return the run numbers with a generated FFT spectrum in schema v5."""
    runs = set()
    if isinstance(spectra_state, dict):
        for run_key in spectra_state:
            try:
                runs.add(int(run_key))
            except (TypeError, ValueError):
                continue
    return runs

"""Uniform spectral views of captured counts, separate from fitted observations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields

import numpy as np
from numpy.typing import NDArray

from asymmetry.core import computation
from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.data.ratio_validity import RatioValidity
from asymmetry.core.data.reduction_recipe import EffectiveReductionRecipe
from asymmetry.core.fourier.sampling import _uniform_sample_interval, is_uniform_sampling
from asymmetry.core.transform.preparation import (
    prepare_histogram_asymmetry,
    wizard_observation_identity,
)
from asymmetry.core.transform.rebin import NativeDiagnosticBinning
from asymmetry.core.utils.perf import perf_timer

DIAGNOSTIC_POLICY = "captured_native_support_fixed1_v1"


@dataclass(frozen=True)
class DiagnosticObservation:
    """Owned diagnostic arrays in µs/percent, without acquisition replay buffers."""

    time: NDArray[np.float64]
    asymmetry: NDArray[np.float64]
    error: NDArray[np.float64]
    ratio_validity: RatioValidity | None
    attribution: EffectiveReductionRecipe

    def __post_init__(self) -> None:
        for array in (self.time, self.asymmetry, self.error):
            array.setflags(write=False)

    def __reduce__(self):
        return type(self), (
            self.time,
            self.asymmetry,
            self.error,
            self.ratio_validity,
            self.attribution,
        )

    def window_attribution(self, stop: int, window: str) -> dict:
        """Record the actual transformed window and its fixed sampling interval."""
        time = self.time[:stop]
        dt = _uniform_sample_interval(time, caller="Internal spectrum", t_min=None, t_max=None)
        return self.attribution.with_facts(
            window=window,
            padding_factor=4,
            n_points=int(time.size),
            interval_us=dt,
            origin_us=float(time[0]),
            last_time_us=float(time[-1]),
        ).to_dict()


@perf_timer("core.fourier.prepare_diagnostic")
def prepare_diagnostic_observation(
    dataset: MuonDataset,
    *,
    cancel_check: Callable[[], None] | None = None,
) -> DiagnosticObservation:
    """Preserve a uniform curve, or reduce its captured counts on native support.

    A Run or recipe alone grants no replay. Selections resolve at the canonical
    count-binning owner, before the retained native edges become fixed bins.
    This synchronous operation belongs on the existing numerical worker.
    """
    if cancel_check is not None:
        cancel_check()
    if isinstance(dataset, DiagnosticFitRecord):
        return dataset.diagnostic_observation
    time = np.asarray(dataset.time, dtype=float)
    observation = dataset.histogram_observation
    unchanged = is_uniform_sampling(time)
    if unchanged:
        arrays = (dataset.time, dataset.asymmetry, dataset.error)
        validity, recipe = dataset.ratio_validity, dataset.effective_recipe
    else:
        if observation is None or "rotating_frame" in dataset.metadata or time.size < 2:
            _uniform_sample_interval(time, caller="Internal spectrum", t_min=None, t_max=None)
            raise ValueError("Internal spectrum requires captured histogram observations.")
        reduced = prepare_histogram_asymmetry(
            observation.source,
            merge=1,
            facility=observation.facility,
            reference=observation.reference,
            cancel_check=cancel_check,
            binning=NativeDiagnosticBinning(observation.binning),
        )
        arrays = (reduced.time, reduced.asymmetry, reduced.error)
        validity, recipe = reduced.ratio_validity, reduced.effective_recipe
        if validity is not None and validity.report().undefined_ratio:
            raise ValueError(
                "Internal spectrum unavailable: the fixed native diagnostic grid contains undefined ratios; select supported count observations or use a valid Fixed reduction. No bins are filled or omitted for the FFT."
            )
    owned = tuple(np.array(a, dtype=float, copy=True) for a in arrays)
    for array in owned:
        array.setflags(write=False)
    interval = _uniform_sample_interval(
        owned[0], caller="Diagnostic support", t_min=None, t_max=None
    )
    attribution = EffectiveReductionRecipe.from_dict(
        {
            "version": 1,
            "operation": "spectral_diagnostic",
            "policy": DIAGNOSTIC_POLICY,
            "computation_revision": computation.COMPUTATION_REVISION,
            "source_computation_revision": dataset.computation_revision,
            "source_support": {
                "n_points": len(owned[0]),
                "origin_us": float(owned[0][0]),
                "last_time_us": float(owned[0][-1]),
                "interval_us": interval,
            },
            "source_observation": wizard_observation_identity(dataset),
            "ratio_classification": {
                "undefined": validity.report().undefined_ratio,
                "zero_variance": validity.report().zero_variance,
            }
            if validity is not None
            else None,
            "preparation": "original_uniform" if unchanged else "fixed_native_support",
            "recipe": _diagnostic_recipe(recipe.to_dict()) if recipe is not None else None,
            "units": "percent",
            "time_units": "microsecond",
        }
    )
    if cancel_check is not None:
        cancel_check()
    return DiagnosticObservation(*owned, validity, attribution)


def _diagnostic_recipe(value):
    """Encode regular aligned edges as a range in the diagnostic's own recipe.

    A fixed native grid can contain millions of edges. Its exact integer first,
    stop and step carry membership without duplicating those values per model.
    """
    if not isinstance(value, dict):
        return value
    compact = {key: _diagnostic_recipe(child) for key, child in value.items()}
    edges = compact.get("aligned_bin_edges")
    if edges is not None and len(edges) > 1:
        steps = np.diff(edges)
        if np.all(steps == steps[0]):
            compact.pop("aligned_bin_edges")
            compact["aligned_bin_edge_range"] = {
                "first": edges[0],
                "stop": edges[-1],
                "step": int(steps[0]),
            }
    return compact


@dataclass(kw_only=True)
class DiagnosticFitRecord(MuonDataset):
    """A process fit record with its separately owned spectral observation.

    Constructed only on numerical workers before candidate fan-out. Cropping or
    merging uses MuonDataset's constructors and drops this execution context.
    """

    diagnostic_observation: DiagnosticObservation


def diagnostic_fit_record(
    dataset: MuonDataset,
    *,
    diagnostic: DiagnosticObservation | None = None,
) -> MuonDataset:
    """Strip acquisitions after preparing the bounded shared spectral payload."""
    if isinstance(dataset, DiagnosticFitRecord):
        return dataset
    view = diagnostic if diagnostic is not None else prepare_diagnostic_observation(dataset)
    record = dataset.fit_record()
    return DiagnosticFitRecord(
        **{f.name: vars(record)[f.name] for f in fields(MuonDataset)},
        diagnostic_observation=view,
    )


@dataclass(frozen=True)
class SpectralPreview:
    """Captured Hann preview of the same declared diagnostic observation."""

    observation: DiagnosticObservation
    frequency: NDArray[np.float64]
    magnitude: NDArray[np.float64]
    attribution: dict


def prepare_spectral_preview(
    dataset: MuonDataset,
    *,
    cancel_check: Callable[[], None] | None = None,
) -> SpectralPreview:
    """Prepare and transform once on the existing scientific-input worker."""
    from asymmetry.core.fitting.spectral_window import effective_analysis_window
    from asymmetry.core.fourier.fft import fft_arrays

    view = prepare_diagnostic_observation(dataset, cancel_check=cancel_check)
    stop = effective_analysis_window(view.time, view.error)
    frequency, _real, magnitude = fft_arrays(
        view.time[:stop],
        view.asymmetry[:stop],
        view.error[:stop],
        window="hann",
        padding_factor=4,
    )
    if cancel_check is not None:
        cancel_check()
    return SpectralPreview(view, frequency, magnitude, view.window_attribution(stop, "hann"))


def check_diagnostic_cancelled(callback: Callable[[], bool] | None) -> None:
    """Use the existing fit cancellation exception between diagnostic passes."""
    if callback is not None and callback():
        from asymmetry.core.fitting.engine import FitCancelledError

        raise FitCancelledError("Spectral diagnostic cancelled.")

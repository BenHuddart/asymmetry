"""Detached declarative inputs for conditional ideal-count pull experiments."""

from __future__ import annotations

import inspect
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from numbers import Integral
from typing import Any

import numpy as np

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.data.reduction_recipe import EffectiveReductionRecipe
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.result_payload import _deserialize_parameter_set
from asymmetry.core.project.json_values import decode_non_finite, encode_non_finite
from asymmetry.core.simulate import matched_statistics
from asymmetry.core.transform.grouping import (
    independent_pair_indices,
    resolve_forward_backward_layout,
)
from asymmetry.core.transform.rebin import BinningSpec, binning_spec, fb_binning_edges

ASSUMPTIONS = (
    "Conditional ideal independent Poisson detector counts; fitted generating values are plug-in truth.",
    "Signal-event budget and flat background are fixed plug-in settings, not uncertain acquisitions.",
    "Deadtime distortion and deadtime/background correction are omitted; synthetic reduction uses merge=1 unless captured binning specifies otherwise.",
    "Timing, grouping, alpha, constraints and source response are fixed; no calibration or general cross-bin covariance draws.",
    "Pulls use symmetric reported errors; MINOS endpoints are separate and do not enter pulls.",
)


def pull_model_unavailable_reason(model: object) -> str | None:
    """Explain an unsupported count-generating request without restricting fitting."""
    if isinstance(model, CompositeModel) and model.time_response.separation_us > 0:
        return (
            "The count-based Pull Diagnostic has no matching pulse-count generator for "
            "positive ordinary pulse separation. Ordinary fitting and its conditional "
            "parameter errors remain available."
        )
    return None


@dataclass(frozen=True)
class PullRequest:
    """Canonical JSON experiment; executable model/refit are supplied separately.

    Template counts are excluded. The stored detector shapes, resolved timing,
    effective pair and binning reconstruct the supported ideal experiment.
    Custom callable settings are unknown unless explicitly declared by the caller.
    Loading this record never simulates or fits.
    """

    canonical_json: str

    def __post_init__(self) -> None:
        value = json.loads(self.canonical_json)
        if value["version"] != 1:
            raise ValueError("Pull request requires version 1")
        object.__setattr__(
            self, "canonical_json", json.dumps(value, sort_keys=True, allow_nan=False)
        )

    def to_dict(self) -> dict:
        return json.loads(self.canonical_json)

    @classmethod
    def from_dict(cls, value: Mapping) -> PullRequest:
        return cls(json.dumps(dict(value), allow_nan=False))

    def template(self) -> Run:
        """Reconstruct owned zero-count detector shapes, without any acquired buffers."""
        value = self.to_dict()["template"]
        return Run(
            value["run_number"],
            [
                Histogram(
                    counts=np.zeros(h["n_bins"]), **{k: v for k, v in h.items() if k != "n_bins"}
                )
                for h in value["histograms"]
            ],
            grouping=value["grouping"],
        )

    @property
    def requested_seeds(self) -> range:
        value = self.to_dict()
        return range(value["seed_start"], value["seed_start"] + value["n_seeds"])


def make_pull_request(
    template: Run,
    model: Any,
    parameters: Mapping[str, float],
    *,
    total_events: float | None,
    n_seeds: int = 200,
    seed_start: int = 0,
    track: Sequence[str] | None = None,
    alpha: float | None = None,
    background_per_bin: float | None = 0.0,
    time_range: tuple[float | None, float | None] | None = None,
    binning: BinningSpec | None = None,
    source_recipe: EffectiveReductionRecipe | None = None,
    refit_configuration: Mapping | None = None,
) -> PullRequest:
    """Validate an authored request once, using existing layout/binning authorities.

    Non-unit beta, combined periods/acquisitions, fractional clock residuals and
    clipped/nonfinite model means have no matching generator here. Refusal affects
    this diagnostic only. Resolved integer detector origins and exclusions are
    supported; selected channels retain their native detector lengths.
    """
    reason = pull_model_unavailable_reason(model)
    if reason:
        raise ValueError(reason)
    for name, value, minimum in (("n_seeds", n_seeds, 1), ("seed_start", seed_start, 0)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    truth = {k: float(v) for k, v in parameters.items()}
    if not truth or not all(np.isfinite(v) for v in truth.values()):
        raise ValueError("Generating parameters must be nonempty and finite.")
    tracked = list(truth) if track is None else list(track)
    if not tracked or len(set(tracked)) != len(tracked) or any(k not in truth for k in tracked):
        raise ValueError(
            "Track unique supported free parameter names; fixed-only fits have no diagnostic."
        )
    declared = None
    if refit_configuration is not None and refit_configuration.get("identity") == "FitEngine.fit":
        declared = _deserialize_parameter_set(decode_non_finite(refit_configuration["parameters"]))
        if set(truth) != set(declared.names) or not set(tracked) <= {
            p.name for p in declared.free_parameters
        }:
            raise ValueError("Track supported free parameters from the executed fit constraints.")
    if isinstance(model, CompositeModel):
        if not set(model.param_defaults) <= set(truth) or (
            declared is None and set(truth) != set(model.param_defaults)
        ):
            raise ValueError("Generating parameter names must match the model.")
        function = model.function
        descriptor = model.to_dict()
    else:
        function = model
        inspect.signature(function).bind(np.zeros(1), **truth)
        identity = (
            function.__qualname__
            if inspect.isfunction(function) or inspect.ismethod(function)
            else type(function).__qualname__
        )
        descriptor = {"callable": identity, "configuration": None}
    if time_range is not None:
        lo, hi = time_range
        if any(v is not None and not np.isfinite(v) for v in time_range) or (
            lo is not None and hi is not None and lo > hi
        ):
            raise ValueError("Fit time window must be finite and ordered.")
    grouping = template.grouping
    if (
        grouping.get("period_histograms")
        or grouping.get("period_inputs")
        or template.metadata.get("combination")
    ):
        raise ValueError(
            "Pull Diagnostic requires one uncombined detector acquisition, not period/run arithmetic."
        )
    pair = independent_pair_indices(grouping, n_histograms=len(template.histograms))
    layout = resolve_forward_backward_layout(template.histograms, grouping, resolved_indices=pair)
    origins = {i: layout.common_t0_bin - offset for i, offset in layout.alignment.detector_offsets}
    matched = total_events is None or background_per_bin is None
    if matched:
        selected = replace(
            template,
            histograms=[
                replace(template.histograms[i], t0_bin=origins[i]) for bank in pair for i in bank
            ],
        )
        signal, background = matched_statistics(selected)
        total_events = signal if total_events is None else total_events
        background_per_bin = background if background_per_bin is None else background_per_bin
    for name, value, positive in (
        ("total_events", total_events, True),
        ("background_per_bin", background_per_bin, False),
    ):
        if not np.isfinite(value) or (value <= 0 if positive else value < 0):
            raise ValueError(
                f"{name} must be finite and {'positive' if positive else 'non-negative'}."
            )
    if layout.beta != 1.0:
        raise ValueError("Pull Diagnostic has no matching non-unit beta count generator.")
    residual = (layout.common_t0_bin + 0.5) * layout.bin_width_us - layout.t0_time_us
    if not np.isclose(residual, 0.0, rtol=0, atol=8 * np.spacing(layout.bin_width_us)) or any(
        not np.isclose(value, 0.0, rtol=0, atol=8 * np.spacing(layout.bin_width_us))
        for _, value in layout.alignment.timing_residuals_us
    ):
        raise ValueError(
            "Pull Diagnostic requires integer-bin source timing; fractional clock response is unsupported."
        )
    balance = layout.alpha if alpha is None else float(alpha)
    if not np.isfinite(balance) or balance <= 0:
        raise ValueError("alpha must be positive and finite.")
    packing = binning_spec(grouping) if binning is None else binning
    edges, start = fb_binning_edges(layout, packing, time_range=time_range)
    if (
        edges.size < 2
        or start + ((edges[0] + edges[1] - 1) * 0.5 - layout.window.first) * layout.bin_width_us < 0
    ):
        raise ValueError("Pull Diagnostic needs nonempty post-source fit support.")
    indices = [*pair[0], *pair[1]]
    histograms = []
    grids = set()
    for i in indices:
        h = template.histograms[i]
        grid = (h.n_bins, origins[i], h.bin_width)
        if grid not in grids:
            times = np.arange(max(0, h.n_bins - origins[i])) * h.bin_width
            signal = np.asarray(function(times, **truth), dtype=float)
            if np.any(~np.isfinite(signal)) or np.any(np.abs(signal) > 100):
                raise ValueError(
                    "Pull Diagnostic requires finite model asymmetry within ±100 percent on generated support."
                )
            grids.add(grid)
        histograms.append(
            {
                "n_bins": h.n_bins,
                "bin_width": h.bin_width,
                "t0_bin": origins[i],
                "good_bin_start": h.good_bin_start,
                "good_bin_end": h.good_bin_end,
            }
        )
    return PullRequest.from_dict(
        encode_non_finite(
            {
                "version": 1,
                "seed_start": int(seed_start),
                "n_seeds": int(n_seeds),
                "parameters": truth,
                "track": tracked,
                "model": descriptor,
                "total_events": float(total_events),
                "statistics_conditioning": "selected-pair plug-in signal budget/pre-t0 flat estimate"
                if matched
                else "supplied signal budget/flat background",
                "background_per_bin": float(background_per_bin),
                "time_range": time_range,
                "binning": asdict(packing),
                "refit": dict(refit_configuration)
                if refit_configuration is not None
                else {"identity": "supplied callable", "configuration": None},
                "assumptions": ASSUMPTIONS,
                "source_context": {
                    "effective_recipe": source_recipe.to_dict()
                    if source_recipe is not None
                    else None,
                    "run_number": template.run_number,
                    "selected_detector_indices": indices,
                    "deadtime_correction": grouping.get("deadtime_correction"),
                    "dead_time_us": grouping.get("dead_time_us"),
                    "background_correction": grouping.get("background_correction"),
                    "excluded_detectors": grouping.get("excluded_detectors", []),
                },
                "template": {
                    "run_number": template.run_number,
                    "histograms": histograms,
                    "grouping": {
                        "groups": {
                            1: list(range(1, len(pair[0]) + 1)),
                            2: list(range(len(pair[0]) + 1, len(indices) + 1)),
                        },
                        "forward_group": 1,
                        "backward_group": 2,
                        "alpha": balance,
                        "beta": 1.0,
                        "first_good_bin": layout.window.first + layout.alignment.first_bin,
                        "last_good_bin": layout.window.last + layout.alignment.first_bin,
                    },
                },
            }
        )
    )


def request_binning(request: PullRequest) -> BinningSpec:
    """Restore the existing immutable binning value from a diagnostic request."""
    value = decode_non_finite(request.to_dict()["binning"])
    value["selections"] = tuple(tuple(selection) for selection in value["selections"])
    if value["time_range"] is not None:
        value["time_range"] = tuple(value["time_range"])
    return BinningSpec(**value)

"""Qt-free adapter and captured single-fit inputs for conditional pull checks."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, replace

import numpy as np

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitEngine
from asymmetry.core.fitting.likelihood import FitInputError
from asymmetry.core.fitting.parameters import ParameterSet
from asymmetry.core.fitting.result_payload import (
    _deserialize_fit_result,
    _deserialize_parameter_set,
    _serialize_fit_result,
    _serialize_parameter_set,
)
from asymmetry.core.fitting.results import FitResult
from asymmetry.core.project.json_values import decode_non_finite, encode_non_finite
from asymmetry.core.pull_diagnostic import RefitOutcome
from asymmetry.core.pull_request import make_pull_request, pull_model_unavailable_reason


class EngineRefit:
    """Own model, constraints and supported options; propagate numerical cancellation.

    Frequencies and constraints remain in the laboratory frame. Source offset
    parameterisation is recorded, but synthetic refits optimise lab parameters.
    The engine's symmetric
    parabolic errors enter pulls; optional MINOS signed offsets stay separate.
    """

    def __init__(
        self,
        model,
        parameters: ParameterSet,
        *,
        minos=False,
        frequency_offsets=None,
        error_oversampling=1.0,
    ):
        self.model = copy.deepcopy(model)
        self.parameters = copy.deepcopy(parameters)
        self.options_json = json.dumps(
            {
                "minos": minos,
                "frequency_offsets": frequency_offsets,
                "error_oversampling": error_oversampling,
            },
            sort_keys=True,
        )

    def configuration(self) -> dict:
        return encode_non_finite(
            {
                "identity": "FitEngine.fit",
                "objective": "gaussian",
                "method": "migrad",
                "parameters": _serialize_parameter_set(self.parameters),
                "options": json.loads(self.options_json) | {"frequency_offsets": None},
                "original_frequency_offsets": json.loads(self.options_json)["frequency_offsets"],
                "difference": "Synthetic refits optimise laboratory parameters; source frequency-offset parameterisation is omitted.",
                "symmetric_error": "engine parabolic error; no residual scaling",
                "reported_frame": "laboratory",
                "asymmetry_units": "percent",
            }
        )

    @classmethod
    def from_configuration(cls, model, configuration):
        """Replay supported declarative engine settings with a supplied model."""
        config = decode_non_finite(configuration)
        options = config["options"] | {"frequency_offsets": config["original_frequency_offsets"]}
        return cls(model, _deserialize_parameter_set(config["parameters"]), **options)

    def __call__(self, dataset, cancelled):
        options = json.loads(self.options_json)
        options["frequency_offsets"] = None
        parameters = copy.deepcopy(self.parameters)
        function = self.model.function if isinstance(self.model, CompositeModel) else self.model
        try:
            result = FitEngine().fit(
                dataset, function, parameters, cancel_callback=cancelled, **options
            )
        except FitInputError as exc:
            return RefitOutcome(False, message=f"unusable generated observations: {exc}")
        values = {p.name: p.value for p in result.parameters}
        return RefitOutcome(
            result.success,
            values,
            dict(result.uncertainties),
            result.message,
            result.minos_errors,
            json.dumps(
                encode_non_finite(
                    {
                        "computation_revision": result.computation_revision,
                        "input_attribution": {
                            k: v
                            for k, v in result.input_attribution.items()
                            if k != "effective_recipe"
                        }
                        if result.input_attribution is not None
                        else None,
                        "observation_report": result.observation_report.to_dict()
                        if result.observation_report is not None
                        else None,
                    }
                ),
                sort_keys=True,
                allow_nan=False,
            ),
        )


@dataclass(frozen=True)
class PullFitInput:
    """Execution-owner capture; later controls cannot reinterpret a completed fit."""

    model: CompositeModel
    parameters: ParameterSet
    dataset: MuonDataset
    options_json: str

    @classmethod
    def capture(cls, model, parameters, dataset, options):
        offsets = options.get("frequency_offsets") or {}
        laboratory = ParameterSet(
            [
                replace(
                    p,
                    value=p.value + offsets.get(p.name, 0),
                    min=p.min + offsets.get(p.name, 0),
                    max=p.max + offsets.get(p.name, 0),
                )
                for p in parameters
            ]
        )
        return cls(
            copy.deepcopy(model),
            laboratory,
            replace(
                dataset.fit_record(),
                time=dataset.time.copy(),
                histogram_observation=dataset.histogram_observation,
            ),
            json.dumps(options, sort_keys=True),
        )

    def unavailable_reason(self) -> str | None:
        reason = pull_model_unavailable_reason(self.model)
        if reason:
            return reason
        offsets = json.loads(self.options_json).get("frequency_offsets") or {}
        if offsets and (
            any(p.expr is not None or p.tie is not None for p in self.parameters)
            or any(
                offsets.get(follower, 0) != offsets.get(main, 0)
                for follower, main in self.parameters.link_followers().items()
            )
        ):
            return "Pull Diagnostic cannot translate the executed frequency-offset constraints to laboratory parameters; ordinary fitting remains available."
        if not self.parameters.free_parameters:
            return "Fixed-only fits have no free-parameter Pull Diagnostic."
        if self.dataset.histogram_observation is None:
            return "Pull Diagnostic requires captured detector observations; transformed or curve-only fits are unsupported."
        if self.dataset.effective_recipe is not None and self.dataset.effective_recipe.to_dict()[
            "operation"
        ] not in {"fb_asymmetry", "red", "green"}:
            return "Pull Diagnostic requires an ordinary laboratory-frame F/B observation, not transformed data."
        return None

    def complete(self, result: FitResult):
        """Bind fitted plug-in values to launch constraints and selected support."""
        reason = self.unavailable_reason()
        if reason:
            raise ValueError(reason)
        parameters = copy.deepcopy(self.parameters)
        parameters.update_values({p.name: p.value for p in result.parameters})
        options = json.loads(self.options_json)
        refit = EngineRefit(self.model, parameters, **options)
        observation = self.dataset.histogram_observation
        request = make_pull_request(
            observation.source,
            self.model,
            {p.name: p.value for p in parameters},
            total_events=None,
            background_per_bin=None,
            track=[p.name for p in parameters.free_parameters],
            time_range=(float(np.min(self.dataset.time)), float(np.max(self.dataset.time))),
            binning=observation.binning,
            source_recipe=self.dataset.effective_recipe,
            refit_configuration=refit.configuration()
            | {
                "source_fit_input_attribution": result.input_attribution,
                "source_fit_computation_revision": result.computation_revision,
                "source_parameter_history": result.parameter_history,
                "source_parameter_inputs": _serialize_parameter_set(self.parameters),
                "source_parameter_frame": "laboratory; converted from actual engine seeds",
                "source_observation_report": result.observation_report.to_dict()
                if result.observation_report is not None
                else None,
            },
        )
        return request, refit


@dataclass(frozen=True)
class CapturedPullFit:
    """Cheap completed-execution binding; numerical preparation runs on the worker."""

    source: PullFitInput
    result_json: str

    @classmethod
    def capture(cls, source, result):
        owned = replace(
            source, model=copy.deepcopy(source.model), parameters=copy.deepcopy(source.parameters)
        )
        return cls(
            owned,
            json.dumps(
                encode_non_finite(_serialize_fit_result(result, include_residuals=False)),
                allow_nan=False,
            ),
        )

    def presentation(self):
        return {
            "n_seeds": 200,
            "seed_start": 0,
            "source_context": {
                "run_number": self.source.dataset.histogram_observation.source.run_number
            },
            "time_range": [
                float(np.min(self.source.dataset.time)),
                float(np.max(self.source.dataset.time)),
            ],
        }

    def prepare(self):
        return self.source.complete(
            _deserialize_fit_result(decode_non_finite(json.loads(self.result_json)))
        )

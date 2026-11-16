"""Conditional ideal-count simulation/refit summaries, independent of Qt.

Pull means and sample widths describe a common usable subset, not estimator
bias, interval coverage or the adequacy of the acquired-data model.
"""

from __future__ import annotations

import json
import platform
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version

import numpy as np
from numpy.typing import NDArray

from asymmetry.core import computation
from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.data.reduction_recipe import EffectiveReductionRecipe
from asymmetry.core.fitting.engine import FitCancelledError
from asymmetry.core.project.json_values import decode_non_finite, encode_non_finite
from asymmetry.core.pull_request import (
    PullRequest,
    make_pull_request,
    pull_model_unavailable_reason,
    request_binning,
)
from asymmetry.core.simulate import reduce_run_to_dataset, simulate_run

__all__ = [
    "ParameterPull",
    "PullDistribution",
    "PullAttempt",
    "RefitOutcome",
    "PullRequest",
    "make_pull_request",
    "pull_model_unavailable_reason",
    "run_pull_distribution",
    "run_pull_request",
]


@dataclass(frozen=True)
class RefitOutcome:
    """Expected numerical outcome; coding errors and cancellation still propagate.

    ``errors`` contains the symmetric reported errors. Optional MINOS signed
    offsets are separate, never substituted into pulls.
    """

    success: bool
    values: Mapping[str, float] = field(default_factory=dict)
    errors: Mapping[str, float | None] = field(default_factory=dict)
    message: str = ""
    minos: Mapping[str, tuple[float, float]] | None = None
    refit_provenance_json: str | None = None


Refit = Callable[[MuonDataset, Callable[[], bool]], RefitOutcome | None]


@dataclass(frozen=True)
class PullAttempt:
    """One seed's outcome; all tracked pulls are admitted or rejected together."""

    seed: int
    fit_success: bool | None
    reason: str
    values: tuple[tuple[str, float], ...] = ()
    errors: tuple[tuple[str, float], ...] = ()
    pulls: tuple[tuple[str, float], ...] = ()
    minos: tuple[tuple[str, tuple[float, float]], ...] = ()
    recipe_index: int | None = None
    message: str = ""
    refit_provenance_json: str | None = None


@dataclass(frozen=True)
class ParameterPull:
    """Descriptive standardised residuals on the common usable subset."""

    name: str
    truth: float
    pulls: NDArray[np.float64]

    @property
    def n(self) -> int:
        return int(self.pulls.size)

    @property
    def mean(self) -> float:
        if not self.n:
            return float("nan")
        scale = float(np.max(np.abs(self.pulls)))
        return scale * float((self.pulls / scale).mean()) if scale else 0.0

    @property
    def mean_uncertainty(self) -> float:
        """Reference SE 1/√N for independent standard-normal pulls only."""
        return 1.0 / np.sqrt(self.n) if self.n else float("nan")

    @property
    def width(self) -> float:
        """Sample SD; two observations suffice for a descriptive value."""
        if self.n < 2:
            return float("nan")
        scale = float(np.max(np.abs(self.pulls)))
        with np.errstate(over="ignore"):
            return float(scale * np.std(self.pulls / scale, ddof=1)) if scale else 0.0

    @property
    def width_uncertainty(self) -> float:
        """Approximate SD reference SE under independent N(0,1) pulls only."""
        return 1.0 / np.sqrt(2.0 * (self.n - 1)) if self.n > 1 else float("nan")

    def verdict(self) -> str:
        """Measured quantities without directional error or coverage certification."""
        if self.n < 2:
            return f"{self.name}: fewer than two usable pulls; sample width unavailable."
        return f"{self.name}: mean {self.mean:.3g}; sample width {self.width:.3g} (N={self.n})."


@dataclass(frozen=True)
class PullDistribution:
    """Attempts are authoritative; counts and parameter samples are derived."""

    request: PullRequest
    attempts: tuple[PullAttempt, ...]
    stopped: bool
    recipes: tuple[EffectiveReductionRecipe, ...]
    runtime_json: str

    @property
    def n_attempted(self) -> int:
        return len(self.attempts)

    @property
    def n_fit_success(self) -> int:
        return sum(a.fit_success is True for a in self.attempts)

    @property
    def n_usable(self) -> int:
        return sum(a.reason == "usable" for a in self.attempts)

    @property
    def n_failed(self) -> int:
        """Unsuccessful/unavailable refits; interrupted attempts remain separate."""
        return sum(a.fit_success is False for a in self.attempts)

    @property
    def failure_reasons(self) -> dict[str, int]:
        return dict(Counter(a.reason for a in self.attempts if a.reason != "usable"))

    @property
    def parameters(self) -> dict[str, ParameterPull]:
        config = self.request.to_dict()
        return {
            name: ParameterPull(
                name,
                config["parameters"][name],
                np.asarray([dict(a.pulls)[name] for a in self.attempts if a.reason == "usable"]),
            )
            for name in config["track"]
        }

    def verdict(self) -> str:
        config = self.request.to_dict()
        lines = [
            f"{'Stopped — partial' if self.stopped else 'Completed'} conditional simulation: "
            f"{self.n_attempted}/{config['n_seeds']} attempted; {self.n_failed} failed refits; "
            f"{self.n_fit_success} fit successes; {self.n_usable} usable.",
            f"Requested seeds {config['seed_start']}…{config['seed_start'] + config['n_seeds'] - 1}.",
        ]
        lines.extend(f"{reason}: {count}" for reason, count in self.failure_reasons.items())
        lines.extend(p.verdict() for p in self.parameters.values())
        lines.append("Usable-subset summaries do not establish unbiasedness or interval coverage.")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Strict JSON-compatible provenance using the existing nonfinite codec."""
        return encode_non_finite(
            {
                "version": 1,
                "request": self.request.to_dict(),
                "attempts": [asdict(a) for a in self.attempts],
                "stopped": self.stopped,
                "recipes": [r.to_dict() for r in self.recipes],
                "runtime": json.loads(self.runtime_json),
            }
        )

    @classmethod
    def from_dict(cls, value: Mapping) -> PullDistribution:
        """Restore recorded facts only; never execute a model or reconstruct history."""
        data = dict(value)
        if data["version"] != 1:
            raise ValueError("Pull result requires version 1")
        attempts = []
        for item in data["attempts"]:
            fields = decode_non_finite(dict(item))
            for key in ("values", "errors", "pulls", "minos"):
                fields[key] = tuple((k, tuple(v) if key == "minos" else v) for k, v in fields[key])
            attempts.append(PullAttempt(**fields))
        return cls(
            PullRequest.from_dict(data["request"]),
            tuple(attempts),
            data["stopped"],
            tuple(EffectiveReductionRecipe.from_dict(r) for r in data["recipes"]),
            json.dumps(data["runtime"], sort_keys=True),
        )


def assess_pull_attempt(seed, outcome, config, recipe_index) -> PullAttempt:
    """Admit a complete tracked vector only after finite subtraction/division."""
    if outcome is None:
        return PullAttempt(seed, False, "no_refit_result", recipe_index=recipe_index)
    values, errors = dict(outcome.values), dict(outcome.errors)
    reasons = []
    pulls = {}
    if not outcome.success:
        reasons.append("unsuccessful_refit")
    else:
        for name in config["track"]:
            if name not in values:
                reasons.append(f"missing_value:{name}")
            elif not np.isfinite(values[name]):
                reasons.append(f"nonfinite_value:{name}")
            if name not in errors:
                reasons.append(f"missing_error:{name}")
            elif errors[name] is None:
                reasons.append(f"unavailable_uncertainty:{name}")
            elif not np.isfinite(errors[name]) or errors[name] <= 0:
                reasons.append(f"invalid_error:{name}")
            if (
                name in values
                and name in errors
                and errors[name] is not None
                and np.isfinite(values[name])
                and np.isfinite(errors[name])
                and errors[name] > 0
            ):
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    pull = float(
                        (np.float64(values[name]) - config["parameters"][name]) / errors[name]
                    )
                if np.isfinite(pull):
                    pulls[name] = pull
                else:
                    reasons.append(f"nonfinite_pull:{name}")
    return PullAttempt(
        seed,
        outcome.success,
        "; ".join(reasons) if reasons else "usable",
        tuple(values.items()),
        tuple(errors.items()),
        () if reasons else tuple(pulls.items()),
        tuple((outcome.minos or {}).items()),
        recipe_index,
        outcome.message,
        outcome.refit_provenance_json,
    )


def pull_runtime_identity() -> dict:
    """Identify the executed RNG/runtime, including unavailable optional engines."""
    packages = {}
    for name in ("asymmetry", "numpy", "scipy", "iminuit"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "packages": packages,
        "rng": f"numpy.random.default_rng/{type(np.random.default_rng(0).bit_generator).__name__}",
        "computation_revision": computation.COMPUTATION_REVISION,
    }


def run_pull_request(
    request: PullRequest,
    model,
    refit: Refit,
    *,
    progress: Callable[[int, int], None] | None = None,
    should_continue: Callable[[], bool] | None = None,
) -> PullDistribution:
    """Execute a declarative request with the same supplied callable/environment.

    Generation is indivisible. Cancellation is checked between generation,
    reduction and refit, and passed into numerical fitting. An interrupted seed
    stays in the partial record, without a fabricated nonconvergence.
    """
    config = request.to_dict()
    template = request.template()
    packing = request_binning(request)
    attempts = []
    recipes = []
    stopped = False

    def cancelled():
        return should_continue is not None and not should_continue()

    def cancel_check():
        if cancelled():
            raise FitCancelledError("Pull experiment stopped")

    for seed in request.requested_seeds:
        if cancelled():
            stopped = True
            break
        recipe_index = None
        try:
            run = simulate_run(
                template,
                model,
                config["parameters"],
                total_events=config["total_events"],
                seed=seed,
                background_per_bin=config["background_per_bin"],
            )
            cancel_check()
            dataset = reduce_run_to_dataset(run, binning=packing, cancel_check=cancel_check)
            if config["time_range"] is not None:
                dataset = dataset.time_range(*config["time_range"])
            recipe = dataset.effective_recipe
            if recipe not in recipes:
                recipes.append(recipe)
            recipe_index = recipes.index(recipe)
            cancel_check()
            outcome = refit(dataset, cancelled)
            cancel_check()
        except FitCancelledError:
            attempts.append(PullAttempt(seed, None, "cancelled", recipe_index=recipe_index))
            stopped = True
            break
        attempts.append(assess_pull_attempt(seed, outcome, config, recipe_index))
        if progress is not None:
            progress(len(attempts), config["n_seeds"])
    return PullDistribution(
        request,
        tuple(attempts),
        stopped,
        tuple(recipes),
        json.dumps(pull_runtime_identity(), sort_keys=True),
    )


def run_pull_distribution(
    template, model, parameters, refit: Refit, *, progress=None, should_continue=None, **settings
) -> PullDistribution:
    """Validate/snapshot a request, then run the conditional experiment.

    Call :func:`make_pull_request` before worker launch when caller-owned mutable
    inputs may change. Replay uses :func:`run_pull_request` with supplied code;
    JSON contains no executable functions or promise of cross-version identity.
    """
    return run_pull_request(
        make_pull_request(template, model, parameters, **settings),
        model,
        refit,
        progress=progress,
        should_continue=should_continue,
    )

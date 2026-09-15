"""Reduce a run to forward/backward asymmetry, exactly as the GUI does.

The one rule this module exists to enforce: a scripted reduction and the
desktop application must produce the *same numbers*. So it goes through the
same path a freshly opened project does — a :class:`GroupingProfile` built
from the loader's own grouping payload, policies applied by
:func:`resolve_effective_grouping`, then the
:func:`reduce_grouped_asymmetry` chokepoint — rather than re-deriving the
grouping or the corrections here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from asymmetry.core.data.dataset import MuonDataset, Run
from asymmetry.core.project.profiles import (
    AlphaPolicy,
    BackgroundPolicy,
    DeadtimePolicy,
    GroupingProfile,
    profile_fingerprint_for_run,
    profile_from_payload,
    resolve_effective_grouping,
)
from asymmetry.core.transform.grouping import effective_group_indices
from asymmetry.core.transform.reduce import (
    correction_flags_from_grouping,
    reduce_grouped_asymmetry,
)

#: Deadtime treatments a scripted reduction offers. ``off`` is the GUI's
#: fresh-run default; ``from_file`` uses each run's own per-detector values.
#: The manual/estimate modes are grouping-window work, not CLI work.
DEADTIME_MODES = ("off", "from_file")

#: Background treatments. Only ``none`` is wired up so far; the field is
#: typed as a mode rather than a boolean so the range/tail-fit/reference-run
#: modes can join without a signature change.
BACKGROUND_MODES = ("none",)

#: Profile name used for the ephemeral profile a scripted reduction builds.
#: It is never persisted — the settings are recorded in the work directory.
_PROFILE_NAME = "workflow"


@dataclass(frozen=True)
class ReductionSettings:
    """The choices that turn a loaded run into an asymmetry curve.

    Defaults match the GUI's fresh-run defaults: alpha 1.0, no deadtime
    correction, no background subtraction, the file's own t0 and good-bin
    window, no bunching and no time window.

    ``alpha_source`` records *where* alpha came from so a later summary can
    say so: ``"assumed"`` (nobody measured it), ``"estimated:<run>"`` (a
    per-run estimate on that calibration run) or ``"user"``.
    """

    alpha: float = 1.0
    alpha_source: str = "assumed"
    deadtime: str = "off"
    background: str = "none"
    rebin: int = 1
    t_min: float | None = None
    t_max: float | None = None

    def __post_init__(self) -> None:
        # User input arrives here from the CLI, so the vocabulary is checked
        # at this boundary and never again downstream.
        if self.deadtime not in DEADTIME_MODES:
            raise ValueError(
                f"Unknown deadtime mode {self.deadtime!r}; expected one of "
                f"{', '.join(DEADTIME_MODES)}."
            )
        if self.background not in BACKGROUND_MODES:
            raise ValueError(
                f"Unknown background mode {self.background!r}; expected one of "
                f"{', '.join(BACKGROUND_MODES)}."
            )
        if self.rebin < 1:
            raise ValueError(f"Rebin factor must be at least 1, got {self.rebin}.")
        if self.alpha <= 0.0:
            raise ValueError(f"Alpha must be positive, got {self.alpha}.")
        if self.t_min is not None and self.t_max is not None and self.t_min >= self.t_max:
            raise ValueError(f"Time window t_min={self.t_min} is not below t_max={self.t_max}.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain, JSON-safe dict (round-trips via :meth:`from_dict`)."""
        return {
            "alpha": float(self.alpha),
            "alpha_source": str(self.alpha_source),
            "deadtime": str(self.deadtime),
            "background": str(self.background),
            "rebin": int(self.rebin),
            "t_min": None if self.t_min is None else float(self.t_min),
            "t_max": None if self.t_max is None else float(self.t_max),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReductionSettings:
        """Reconstruct settings from :meth:`to_dict` output."""
        return cls(
            alpha=float(data["alpha"]),
            alpha_source=str(data["alpha_source"]),
            deadtime=str(data["deadtime"]),
            background=str(data["background"]),
            rebin=int(data["rebin"]),
            t_min=None if data["t_min"] is None else float(data["t_min"]),
            t_max=None if data["t_max"] is None else float(data["t_max"]),
        )


@dataclass(frozen=True)
class AlphaEstimate:
    """A per-run forward/backward balance estimate (the GUI's Estimate button)."""

    run_number: int
    alpha: float
    method: str
    forward_group: int
    backward_group: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain, JSON-safe dict."""
        return {
            "run_number": self.run_number,
            "alpha": self.alpha,
            "method": self.method,
            "forward_group": self.forward_group,
            "backward_group": self.backward_group,
        }


def _profile_for_run(run: Run, *, alpha_policy: AlphaPolicy, deadtime: str) -> GroupingProfile:
    """An ephemeral profile carrying this run's file-derived groups + the policies."""
    profile = profile_from_payload(
        run.grouping,
        _PROFILE_NAME,
        profile_fingerprint_for_run(run),
    )
    return replace(
        profile,
        alpha_policy=alpha_policy,
        deadtime_policy=DeadtimePolicy(mode=deadtime),
        background_policy=BackgroundPolicy(mode="none"),
    )


def resolve_reduction_grouping(run: Run, settings: ReductionSettings) -> dict[str, Any]:
    """The full grouping payload :func:`reduce_run` will reduce *run* with.

    Exposed separately so a caller can digest the resolved grouping (the work
    directory keys its cache on it) without performing the reduction.
    """
    profile = _profile_for_run(
        run,
        alpha_policy=AlphaPolicy(mode="fixed", value=float(settings.alpha)),
        deadtime=settings.deadtime,
    )
    return resolve_effective_grouping(profile, run)


def reduce_run(run: Run, settings: ReductionSettings) -> MuonDataset:
    """Reduce *run* to a :class:`MuonDataset` under *settings*.

    With default settings the result is identical to the loader's own
    reduction of the same file: the profile is built from the loader's
    grouping payload, so the groups, t0, good-bin window and alpha are the
    file's own. ``rebin`` and the time window are applied to the reduced
    curve afterwards, in that order.
    """
    grouping = resolve_reduction_grouping(run, settings)
    n_histograms = len(run.histograms)
    forward_idx = effective_group_indices(
        grouping, int(grouping["forward_group"]), n_histograms=n_histograms
    )
    backward_idx = effective_group_indices(
        grouping, int(grouping["backward_group"]), n_histograms=n_histograms
    )
    flags = correction_flags_from_grouping(grouping)
    result = reduce_grouped_asymmetry(
        histograms=run.histograms,
        grouping=grouping,
        forward_idx=forward_idx,
        backward_idx=backward_idx,
        alpha=float(grouping["alpha"]),
        use_deadtime=flags.use_deadtime,
        deadtime_mode=flags.deadtime_mode,
        use_background=flags.use_background,
        metadata=run.metadata,
    )
    dataset = MuonDataset(
        time=result.time,
        asymmetry=result.asymmetry,
        error=result.error,
        metadata=dict(run.metadata),
        run=run,
    )
    if settings.rebin > 1:
        dataset = dataset.rebin(settings.rebin)
    if settings.t_min is not None or settings.t_max is not None:
        dataset = dataset.time_range(settings.t_min, settings.t_max)
    return dataset


def estimate_alpha_for_run(run: Run) -> AlphaEstimate:
    """Estimate this run's forward/backward balance from its own counts.

    The ``per_run_estimate`` alpha policy resolved against the run — the same
    integral-ratio calculation the grouping window's *Estimate* button runs,
    measured on the corrected (deadtime/background-aware) grouped counts.
    """
    profile = _profile_for_run(
        run,
        alpha_policy=AlphaPolicy(mode="per_run_estimate"),
        deadtime="off",
    )
    grouping = resolve_effective_grouping(profile, run)
    return AlphaEstimate(
        run_number=int(run.run_number),
        alpha=float(grouping["alpha"]),
        method=str(grouping["alpha_method"]),
        forward_group=int(grouping["forward_group"]),
        backward_group=int(grouping["backward_group"]),
    )


__all__ = [
    "BACKGROUND_MODES",
    "DEADTIME_MODES",
    "AlphaEstimate",
    "ReductionSettings",
    "estimate_alpha_for_run",
    "reduce_run",
    "resolve_reduction_grouping",
]

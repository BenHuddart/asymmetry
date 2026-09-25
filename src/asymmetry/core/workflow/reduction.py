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
from asymmetry.core.io.periods import (
    GREEN_INDEX,
    RED_INDEX,
    combine_period_asymmetry,
    period_count,
    period_run,
    select_period,
)
from asymmetry.core.project.profiles import (
    AlphaPolicy,
    BackgroundPolicy,
    DeadtimePolicy,
    GoodWindowPolicy,
    GroupingProfile,
    T0Policy,
    profile_fingerprint_for_run,
    profile_from_payload,
    resolve_effective_grouping,
)
from asymmetry.core.transform.background import available_background_modes
from asymmetry.core.transform.grouping import effective_group_indices, group_names
from asymmetry.core.transform.reduce import (
    correction_flags_from_grouping,
    reduce_grouped_asymmetry,
)
from asymmetry.core.utils.constants import PeriodMode

#: Deadtime treatments a scripted reduction offers. ``off`` is the GUI's
#: fresh-run default; ``from_file`` uses each run's own per-detector values.
#: The manual/estimate modes are grouping-window work, not CLI work.
DEADTIME_MODES = ("off", "from_file")

#: Background treatments. ``range`` is the mean over a pre-t0 bin range
#: (continuous sources only; musrfit's fallback range when none is given);
#: ``tail_fit`` fits a flat rate under the late-time decay (pulsed sources).
#: The fixed and reference-run modes are grouping-window work, not CLI work.
BACKGROUND_MODES = ("none", "range", "tail_fit")

#: The period selector for the green − red difference of a two-period run.
GREEN_MINUS_RED = str(PeriodMode.GREEN_MINUS_RED)

#: Profile name used for the ephemeral profile a scripted reduction builds.
#: It is never persisted — the settings are recorded in the work directory.
_PROFILE_NAME = "workflow"


@dataclass(frozen=True)
class ReductionSettings:
    """The choices that turn a loaded run into an asymmetry curve.

    Defaults match the GUI's fresh-run defaults: alpha 1.0, no deadtime
    correction, no background subtraction, the file's own detector pair, t0 and
    good-bin window, no bunching and no time window.

    ``alpha_source`` records *where* alpha came from so a later summary can
    say so: ``"assumed"`` (nobody measured it), ``"estimated:<run>"`` (a
    per-run estimate on that calibration run) or ``"user"``.

    ``pair`` names the forward and backward groups by the file's group names or
    ids; ``t0_offset_bins`` shifts every detector's file t0; and
    ``t_good_offset_bins`` puts the first good bin that many bins after the
    effective t0 — the grouping window's Manual t0 and t_good offset modes.
    ``period`` selects one period, or :data:`GREEN_MINUS_RED` for the
    difference of a two-period run.
    """

    alpha: float = 1.0
    alpha_source: str = "assumed"
    deadtime: str = "off"
    background: str = "none"
    background_range: tuple[int, int] | None = None
    pair: tuple[str, str] | None = None
    t0_offset_bins: int = 0
    t_good_offset_bins: int | None = None
    rebin: int = 1
    t_min: float | None = None
    t_max: float | None = None
    period: str | None = None

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
        if self.background_range is not None:
            if self.background != "range":
                raise ValueError("A background bin range applies only to the range background.")
            first, last = self.background_range
            if not 0 <= first < last:
                raise ValueError(
                    f"Background range {first}:{last} must be two bins with 0 <= first < last."
                )
        if self.pair is not None:
            forward, backward = self.pair
            if not forward or not backward or forward.casefold() == backward.casefold():
                raise ValueError(
                    f"A pair names two different groups, got {forward!r}/{backward!r}."
                )
        if self.t_good_offset_bins is not None and self.t_good_offset_bins < 0:
            raise ValueError(
                f"The t_good offset counts bins after t0, got {self.t_good_offset_bins}."
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
            "background_range": None
            if self.background_range is None
            else [int(bin_) for bin_ in self.background_range],
            "pair": None if self.pair is None else list(self.pair),
            "t0_offset_bins": int(self.t0_offset_bins),
            "t_good_offset_bins": self.t_good_offset_bins,
            "rebin": int(self.rebin),
            "t_min": None if self.t_min is None else float(self.t_min),
            "t_max": None if self.t_max is None else float(self.t_max),
            "period": self.period,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReductionSettings:
        """Reconstruct settings from :meth:`to_dict` output."""
        return cls(
            alpha=float(data["alpha"]),
            alpha_source=str(data["alpha_source"]),
            deadtime=str(data["deadtime"]),
            background=str(data["background"]),
            background_range=None
            if data["background_range"] is None
            else (int(data["background_range"][0]), int(data["background_range"][1])),
            pair=None if data["pair"] is None else (str(data["pair"][0]), str(data["pair"][1])),
            t0_offset_bins=int(data["t0_offset_bins"]),
            t_good_offset_bins=None
            if data["t_good_offset_bins"] is None
            else int(data["t_good_offset_bins"]),
            rebin=int(data["rebin"]),
            t_min=None if data["t_min"] is None else float(data["t_min"]),
            t_max=None if data["t_max"] is None else float(data["t_max"]),
            period=None if data["period"] is None else str(data["period"]),
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


def _group_id(run: Run, label: str) -> int:
    """The id of the group *label* names on *run* — by its name or by its id."""
    names = group_names(run)
    for gid, name in names.items():
        if label == str(gid) or label.casefold() == name.casefold():
            return gid
    available = ", ".join(f"{gid} {name}" for gid, name in names.items())
    raise ValueError(f"There is no group {label!r}; the groups are {available}.")


def _background_policy(run: Run, settings: ReductionSettings) -> BackgroundPolicy:
    """The background policy *settings* ask for, checked against what *run* can supply."""
    if settings.background == "none":
        return BackgroundPolicy(mode="none")
    available = available_background_modes(metadata=run.metadata, source_file=run.source_file)
    if settings.background not in available:
        raise ValueError(
            "There is no pre-t0 region for a range background (a pulsed source starts "
            "at the muon pulse); use tail_fit."
        )
    details = (
        {}
        if settings.background_range is None
        else {"background_range": list(settings.background_range)}
    )
    return BackgroundPolicy(mode=settings.background, details=details)


def _profile_for_run(
    run: Run, settings: ReductionSettings, *, alpha_policy: AlphaPolicy
) -> GroupingProfile:
    """An ephemeral profile carrying this run's file-derived groups + the settings' policies."""
    profile = profile_from_payload(
        run.grouping,
        _PROFILE_NAME,
        profile_fingerprint_for_run(run),
    )
    if settings.pair is not None:
        profile = replace(
            profile,
            forward_group=_group_id(run, settings.pair[0]),
            backward_group=_group_id(run, settings.pair[1]),
        )
    if settings.t0_offset_bins:
        profile = replace(
            profile, t0_policy=T0Policy(mode="manual", offset_bins=settings.t0_offset_bins)
        )
    if settings.t_good_offset_bins is not None:
        profile = replace(
            profile,
            good_window_policy=GoodWindowPolicy(
                mode="manual", first_offset_bins=settings.t_good_offset_bins
            ),
        )
    return replace(
        profile,
        alpha_policy=alpha_policy,
        deadtime_policy=DeadtimePolicy(mode=settings.deadtime),
        background_policy=_background_policy(run, settings),
    )


def reduction_source(loaded: MuonDataset | list[MuonDataset], period: str | None) -> MuonDataset:
    """The dataset :func:`reduce_run` starts from in a loader's result.

    The first dataset without a period; the named period with one; and the
    combined two-period dataset itself for :data:`GREEN_MINUS_RED`, whose two
    periods :func:`reduce_run` reduces separately.
    """
    if period is None:
        return loaded[0] if isinstance(loaded, list) else loaded
    if period == GREEN_MINUS_RED:
        if isinstance(loaded, list) or period_count(loaded) != 2:
            raise ValueError("The green − red difference needs a two-period (red/green) run.")
        return loaded
    return select_period(loaded, period)


def resolve_reduction_grouping(run: Run, settings: ReductionSettings) -> dict[str, Any]:
    """The full grouping payload :func:`reduce_run` will reduce *run* with.

    Exposed separately so a caller can digest the resolved grouping (the work
    directory keys its cache on it) without performing the reduction. For the
    green − red difference it is the red period's, which the green shares.
    """
    return _fixed_alpha_grouping(_grouping_run(run, settings), settings)


def _grouping_run(run: Run, settings: ReductionSettings) -> Run:
    """The run whose counts *settings* group: the red period of a green − red difference."""
    return period_run(run, RED_INDEX) if settings.period == GREEN_MINUS_RED else run


def _fixed_alpha_grouping(run: Run, settings: ReductionSettings) -> dict[str, Any]:
    """*run*'s grouping resolved under *settings*, with their alpha applied as given."""
    profile = _profile_for_run(
        run, settings, alpha_policy=AlphaPolicy(mode="fixed", value=float(settings.alpha))
    )
    return resolve_effective_grouping(profile, run)


def _reduce_period(run: Run, settings: ReductionSettings) -> MuonDataset:
    """One period's asymmetry through the reduction chokepoint, before rebinning or windowing."""
    grouping = _fixed_alpha_grouping(run, settings)
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
    if settings.background != "none" and "values" not in result.background_state:
        raise ValueError(
            f"Run {run.run_number}: the {settings.background} background could not be "
            f"subtracted ({result.background_state})."
        )
    return MuonDataset(
        time=result.time,
        asymmetry=result.asymmetry,
        error=result.error,
        metadata=dict(run.metadata),
        run=run,
    )


def reduce_run(run: Run, settings: ReductionSettings) -> MuonDataset:
    """Reduce *run* to a :class:`MuonDataset` under *settings*.

    With default settings the result is identical to the loader's own
    reduction of the same file: the profile is built from the loader's
    grouping payload, so the groups, t0, good-bin window and alpha are the
    file's own. For :data:`GREEN_MINUS_RED` *run* is the combined two-period
    run and each period is reduced alone before the difference is formed.
    ``rebin`` and the time window are applied to the reduced curve afterwards,
    in that order.
    """
    if settings.period == GREEN_MINUS_RED:
        if period_count(run) != 2:
            raise ValueError(
                f"Run {run.run_number} is not a two-period (red/green) run; there is no "
                "green − red difference."
            )
        red, green = (
            _reduce_period(period_run(run, index), settings) for index in (RED_INDEX, GREEN_INDEX)
        )
        time, asymmetry, error = combine_period_asymmetry(
            red.time,
            red.asymmetry,
            red.error,
            green.time,
            green.asymmetry,
            green.error,
            GREEN_MINUS_RED,
        )
        dataset = MuonDataset(
            time=time, asymmetry=asymmetry, error=error, metadata=dict(run.metadata), run=run
        )
    else:
        dataset = _reduce_period(run, settings)
    if settings.rebin > 1:
        dataset = dataset.rebin(settings.rebin)
    if settings.t_min is not None or settings.t_max is not None:
        dataset = dataset.time_range(settings.t_min, settings.t_max)
    return dataset


def estimate_alpha_for_run(run: Run, settings: ReductionSettings) -> AlphaEstimate:
    """Estimate this run's forward/backward balance from its own counts.

    The ``per_run_estimate`` alpha policy resolved against the run — the same
    integral-ratio calculation the grouping window's *Estimate* button runs,
    measured on the counts *settings* correct and group (their pair,
    deadtime, background and t0), so alpha balances the spectra it will be
    applied to; for the green − red difference, on the red period.
    ``settings.alpha`` itself is not used.
    """
    counts = _grouping_run(run, settings)
    profile = _profile_for_run(counts, settings, alpha_policy=AlphaPolicy(mode="per_run_estimate"))
    grouping = resolve_effective_grouping(profile, counts)
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
    "GREEN_MINUS_RED",
    "AlphaEstimate",
    "ReductionSettings",
    "estimate_alpha_for_run",
    "reduce_run",
    "reduction_source",
    "resolve_reduction_grouping",
]

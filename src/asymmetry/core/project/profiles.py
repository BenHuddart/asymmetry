"""Project-level named detector-grouping profiles.

Historically every :class:`~asymmetry.core.data.dataset.Run` carried a full
``run.grouping`` payload dict — the union of *shareable* analysis choices (which
detectors form the forward/backward groups, the balance ``alpha``, the deadtime
and background modes) and *per-run, file-derived* facts (the run's own
``t0_bin``, good-bin window, per-detector file deadtime values, good-frame
count, period histograms). Two runs from the same instrument analysed the same
way still each stored a complete copy, so an alpha or good-bin edit had to be
broadcast run-by-run and could silently drift apart.

This module introduces **grouping profiles**: a :class:`GroupingProfile` holds
only the shareable settings, is named and owned by the project, and applies to
every run whose *fingerprint* — ``(instrument name, histogram count)`` — matches.
Multiple profiles per fingerprint may be in concurrent use — e.g. one per
sample in a multi-sample project — with each run recording which profile it is
*assigned* to (schema v17; see :func:`assigned_profile_for_run`). One profile
per fingerprint is flagged the **default** (the ``active`` field): freshly
loaded runs are assigned to it. A run follows its assigned profile unless the
user "releases" it with an explicit per-run override; a released run keeps its
assignment as the *base* profile Reattach returns it to.

:func:`resolve_effective_grouping` merges a profile with a run's file-derived
facts and reproduces **exactly** today's ``run.grouping`` payload shape, so
nothing downstream of ``run.grouping`` (reduction, fitting, export) has to
change. Where a policy needs a computed value — a per-run ``alpha`` estimate, or
the file's own deadtime values — resolution computes it from the run.

Shareable vs per-run key classification
========================================

Every key that appears in a loader/dialog ``run.grouping`` payload is classified
below. **Shareable** keys live on the profile; **per-run** keys are always taken
from the freshly-resolved run (never stored in the profile). A profile is built
by lifting the shareable keys out of a payload (:func:`profile_from_payload`) and
resolution re-merges them with the run's per-run facts.

================================  ==========  ===============================================
Key                               Class       Notes
================================  ==========  ===============================================
groups                            shareable   gid -> 1-based detector ids
group_names                       shareable   gid -> display name
grouping_preset                   shareable   named preset the groups came from
included_groups                   shareable   gid -> bool (positron vs veto)
forward_group / backward_group    shareable   analysis group ids
forward_indices/backward_indices  derived     GUI cache of the two group lists; recomputed
projections                       shareable   list of {label, forward_group, backward_group,
                                              alpha, tint, ...} vector projections
vector_axis                       shareable   canonical vector-axis tag
alpha                             shareable*  value depends on alpha_policy (see below)
alpha_x / alpha_y / alpha_z       shareable   per-axis vector alphas
alpha_method                      shareable   provenance ("calibrated"/"count_fit"/…)
alpha_error / alpha_reference_run shareable   calibration provenance
alpha_*_error/_*_reference_run    shareable   per-axis provenance
excluded_detectors                shareable   1-based ids dropped from every group
period_mode                       shareable   red/green/green_minus_red/green_plus_red
binning_mode                      shareable   fixed/fixed_width/variable
bin0_us / bin10_us                shareable   binning knobs (non-fixed modes)
bunching_factor                   shareable   fixed-mode bunching
deadtime_correction               shareable*  derived from deadtime_policy on/off
deadtime_mode / deadtime_method   shareable   off/file/manual/estimate
deadtime_manual_us                shareable   manual scalar
deadtime_estimated_us             shareable   estimate result (source_run stored)
deadtime_reference_run            shareable   source run for calibrate/estimate
deadtime_source_path              shareable   manual-load source path
background_correction             shareable*  derived from background_policy on/off
background_mode                   shareable   none/range/tail_fit/reference_run/fixed
background_fixed_values           shareable   fixed [forward, backward] constants
background_fix / bkg_fix          shareable   legacy fixed-value aliases
background_method                 shareable   provenance
background_ranges / _range        shareable   range-mode bin windows
background_forward/backward_range shareable   split range windows
background_run                    shareable   reference-run payload {run_number, source_file,
                                              good_frames_sample, good_frames_reference}
background_reference_run          shareable   count-fit background provenance
instrument                        fingerprint part of the profile fingerprint
--------------------------------  ----------  -----------------------------------------------
t0_bin                            per-run     analysis time-zero bin (file-derived)
t_good_offset                     per-run     good-window offset from t0
first_good_bin / last_good_bin    per-run     good-window bounds
bin_index_base                    per-run     0 (PSI/ROOT) or 1 (NeXus) — file format
detector_t0_bins                  per-run     per-detector t0 (PSI/ROOT)
detector_first_good_bins          per-run     per-detector good-window start
detector_last_good_bins           per-run     per-detector good-window end
histogram_labels                  per-run     per-detector labels
root_histo_numbers                per-run     ROOT histogram numbers
good_frames                       per-run     deadtime normaliser (file-derived)
dead_time_us                      per-run*    file deadtime values (deadtime_policy=from_file),
                                              or profile-stored values (manual/estimate)
deadtime_loaded_us                per-run     legacy alias of file deadtime values
period_histograms                 per-run     raw per-period histograms
period_reduced                    per-run     per-period reduced arrays cache
period_good_frames                per-run     per-period good frames
period_dead_time_us               per-run     per-period deadtime tables
period_mapping                    per-run     period-number -> label map
period_reduced                    per-run     reduced-array cache
================================  ==========  ===============================================

``*`` marks a key whose *value* is produced by resolution from a profile policy
rather than copied verbatim: ``alpha`` (per-run estimate mode computes the
integral ratio), ``dead_time_us`` (``from_file`` mode reads the run's own
values), and the ``*_correction`` flags (on/off derived from the policy mode).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

from asymmetry.core.data.dataset import Run
from asymmetry.core.instrument import detect_instrument, instrument_display_name
from asymmetry.core.transform.asymmetry import estimate_alpha
from asymmetry.core.transform.grouping import (
    EFFECTIVE_DETECTOR_T0_KEY,
    common_t0_for_groups,
    effective_group_indices,
)
from asymmetry.core.transform.reduce import (
    ReferenceResolver,
    corrected_grouped_counts,
    correction_flags_from_grouping,
)
from asymmetry.core.transform.t0 import find_t0_for_run
from asymmetry.core.utils.perf import perf_timer

# --------------------------------------------------------------------------- #
# Policy defaults and vocabulary
# --------------------------------------------------------------------------- #

#: Alpha policy modes.
ALPHA_POLICY_MODES = ("fixed", "calibrated", "per_run_estimate")

#: Deadtime policy modes (mirror the grouping dialog's deadtime modes).
DEADTIME_POLICY_MODES = ("off", "from_file", "manual", "estimate")

#: Background policy modes (mirror ``transform.background.BACKGROUND_MODES``).
BACKGROUND_POLICY_MODES = ("none", "range", "tail_fit", "reference_run", "fixed")

#: Time-zero policy modes.
T0_POLICY_MODES = ("from_file", "manual", "auto_detect")

#: Keys copied verbatim from a payload into ``GroupingProfile.background`` for
#: the ``fixed`` / ``range`` / ``reference_run`` modes. Only present keys are
#: carried, so a profile does not fabricate ranges the user never set.
_BACKGROUND_DETAIL_KEYS = (
    "background_fixed_values",
    "background_fix",
    "bkg_fix",
    "background_method",
    "background_ranges",
    "background_range",
    "background_forward_range",
    "background_backward_range",
    "background_run",
    "background_reference_run",
)

#: Shareable grouping-structure keys copied verbatim into a profile.
_STRUCTURE_KEYS = (
    "grouping_preset",
    "vector_axis",
)

#: Per-axis vector alpha keys and their provenance companions.
_VECTOR_ALPHA_KEYS = (
    "alpha_x",
    "alpha_y",
    "alpha_z",
    "alpha_x_error",
    "alpha_x_reference_run",
    "alpha_y_error",
    "alpha_y_reference_run",
    "alpha_z_error",
    "alpha_z_reference_run",
)


def _as_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sanitize_beta(value: Any) -> float:
    """Lenient beta read: unparseable, non-finite or non-positive → 1.0.

    Mirrors the clamp ``group_forward_backward`` applies when reading the
    grouping payload, so a degenerate stored value can never re-enter the
    reduction through a profile.
    """
    beta = _as_float(value, 1.0)
    if not math.isfinite(beta) or beta <= 0.0:
        return 1.0
    return beta


# --------------------------------------------------------------------------- #
# Policy dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class AlphaPolicy:
    """How a profile determines the detector-balance ``alpha`` for a run.

    * ``fixed`` — every run gets the same ``value`` (default 1.0).
    * ``calibrated`` — a value measured once (``method``/``source_run``) and
      applied to every run, with an optional ``error``.
    * ``per_run_estimate`` — resolution computes each run's own forward/backward
      integral ratio (Mantid ``AlphaCalc``; the PSI ``.bin`` per-run default).
    """

    mode: str = "fixed"
    value: float = 1.0
    error: float | None = None
    method: str = ""
    source_run: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain, JSON-safe dict (round-trips via :meth:`from_dict`)."""
        data: dict[str, Any] = {"mode": self.mode}
        if self.mode in ("fixed", "calibrated"):
            data["value"] = float(self.value)
        if self.mode == "calibrated":
            if self.error is not None:
                data["error"] = float(self.error)
            if self.method:
                data["method"] = str(self.method)
            if self.source_run is not None:
                data["source_run"] = int(self.source_run)
        return data

    @classmethod
    def from_dict(cls, data: Any) -> AlphaPolicy:
        """Reconstruct a policy from :meth:`to_dict` output (lenient; defaults to ``fixed``)."""
        if not isinstance(data, dict):
            return cls()
        mode = str(data.get("mode", "fixed")).strip().lower()
        if mode not in ALPHA_POLICY_MODES:
            mode = "fixed"
        return cls(
            mode=mode,
            value=_as_float(data.get("value", 1.0), 1.0),
            error=None if data.get("error") is None else _as_float(data.get("error"), 0.0),
            method=str(data.get("method", "")),
            source_run=_as_int(data.get("source_run")),
        )


@dataclass
class DeadtimePolicy:
    """How a profile applies deadtime correction.

    * ``off`` — no correction.
    * ``from_file`` — use the run's own file deadtime values (resolved per run).
    * ``manual`` — apply the stored ``values`` (per-detector µs) to every run;
      ``method`` distinguishes a hand-typed table from a ``calibrate`` fit and
      ``source_run`` records the run a calibration came from.
    * ``estimate`` — apply the single stored ``estimated_us`` (fitted from
      ``source_run``) to every detector.
    """

    mode: str = "off"
    values: list[float] = field(default_factory=list)
    manual_us: float | None = None
    estimated_us: float | None = None
    method: str = ""
    source_run: int | None = None
    source_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain, JSON-safe dict (round-trips via :meth:`from_dict`)."""
        data: dict[str, Any] = {"mode": self.mode}
        if self.mode == "manual":
            if self.values:
                data["values"] = [float(v) for v in self.values]
            if self.manual_us is not None:
                data["manual_us"] = float(self.manual_us)
            if self.method:
                data["method"] = str(self.method)
            if self.source_run is not None:
                data["source_run"] = int(self.source_run)
            if self.source_path:
                data["source_path"] = str(self.source_path)
        elif self.mode == "estimate":
            if self.estimated_us is not None:
                data["estimated_us"] = float(self.estimated_us)
            if self.values:
                data["values"] = [float(v) for v in self.values]
            if self.source_run is not None:
                data["source_run"] = int(self.source_run)
        return data

    @classmethod
    def from_dict(cls, data: Any) -> DeadtimePolicy:
        """Reconstruct a policy from :meth:`to_dict` output (lenient; defaults to ``off``)."""
        if not isinstance(data, dict):
            return cls()
        mode = str(data.get("mode", "off")).strip().lower()
        if mode not in DEADTIME_POLICY_MODES:
            mode = "off"
        raw_values = data.get("values")
        values = (
            [_as_float(v, 0.0) for v in raw_values] if isinstance(raw_values, (list, tuple)) else []
        )
        return cls(
            mode=mode,
            values=values,
            manual_us=None
            if data.get("manual_us") is None
            else _as_float(data.get("manual_us"), 0.0),
            estimated_us=None
            if data.get("estimated_us") is None
            else _as_float(data.get("estimated_us"), 0.0),
            method=str(data.get("method", "")),
            source_run=_as_int(data.get("source_run")),
            source_path=str(data.get("source_path", "")),
        )


@dataclass
class T0Policy:
    """How a profile determines the analysis time-zero for a run.

    File-derived t0 is the **default**: every loader already reads t0 from the
    file header verbatim (PSI per-detector ``nt0``, MusrRoot ``DetectorInfo``,
    NeXus ``time_zero``), and the common t0 is the max over the analysis groups.
    The two alternatives make the historical manual/search behaviour explicit.

    This mirrors WiMDA's *FileValues* checkbox on the grouping panel: when it is
    ticked the header t0/tgood are used and the user t0 controls are disabled
    (our :attr:`mode` ``"from_file"`` analogue); unticked, the user's own values
    apply (:attr:`mode` ``"manual"``). WiMDA's *SearchT0* button is the one-shot
    fill our :attr:`mode` ``"auto_detect"`` automates per run.

    * ``from_file`` (**default**) — the run's own file-derived t0. Per-detector
      values are preserved and the common t0 is the max over the analysis groups,
      exactly as today. Resolution stores nothing and ignores :attr:`value`.
    * ``manual`` — an explicit common-t0 override (:attr:`value`, a bin index)
      applied to every run as an *offset*: ``delta = value − file_common_t0`` is
      added to each detector's file t0. Resolution writes the resulting effective
      per-detector t0 bins into the payload (``effective_detector_t0_bins``) so
      reduction aligns on them **without** mutating ``Histogram.t0_bin`` — the
      run's histograms stay exactly as loaded.
    * ``auto_detect`` — runs :func:`~asymmetry.core.transform.t0.find_t0_for_run`
      per run at resolution time; :attr:`strategy` and :attr:`spread_bins` carry
      the last detection's provenance for display.
    """

    mode: str = "from_file"
    value: int | None = None
    strategy: str = ""
    spread_bins: int | None = None
    source_run: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain, JSON-safe dict (round-trips via :meth:`from_dict`)."""
        data: dict[str, Any] = {"mode": self.mode}
        if self.mode == "manual" and self.value is not None:
            data["value"] = int(self.value)
        if self.mode == "auto_detect":
            if self.strategy:
                data["strategy"] = str(self.strategy)
            if self.spread_bins is not None:
                data["spread_bins"] = int(self.spread_bins)
            if self.source_run is not None:
                data["source_run"] = int(self.source_run)
        return data

    @classmethod
    def from_dict(cls, data: Any) -> T0Policy:
        """Reconstruct a policy from :meth:`to_dict` output (lenient; defaults to ``from_file``)."""
        if not isinstance(data, dict):
            return cls()
        mode = str(data.get("mode", "from_file")).strip().lower()
        if mode not in T0_POLICY_MODES:
            mode = "from_file"
        return cls(
            mode=mode,
            value=_as_int(data.get("value")),
            strategy=str(data.get("strategy", "")),
            spread_bins=_as_int(data.get("spread_bins")),
            source_run=_as_int(data.get("source_run")),
        )


@dataclass
class BackgroundPolicy:
    """How a profile subtracts background.

    ``mode`` is one of :data:`BACKGROUND_POLICY_MODES`. ``details`` carries the
    mode-specific payload keys verbatim (fixed values, range windows, the
    reference-run ``background_run`` payload), so the resolved grouping presents
    exactly the keys :mod:`asymmetry.core.transform.background` already reads.
    """

    mode: str = "none"
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain, JSON-safe dict (round-trips via :meth:`from_dict`)."""
        data: dict[str, Any] = {"mode": self.mode}
        if self.details:
            data["details"] = dict(self.details)
        return data

    @classmethod
    def from_dict(cls, data: Any) -> BackgroundPolicy:
        """Reconstruct a policy from :meth:`to_dict` output (lenient; defaults to ``none``)."""
        if not isinstance(data, dict):
            return cls()
        mode = str(data.get("mode", "none")).strip().lower()
        if mode not in BACKGROUND_POLICY_MODES:
            mode = "none"
        details = data.get("details")
        return cls(mode=mode, details=dict(details) if isinstance(details, dict) else {})


# --------------------------------------------------------------------------- #
# Fingerprint
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProfileFingerprint:
    """Identity a profile applies to: ``(instrument, histogram_count)``.

    A profile only applies to runs whose instrument name and detector-histogram
    count both match. ``instrument`` is compared case-insensitively after
    trimming; ``histogram_count`` is the number of detector histograms.
    """

    instrument: str
    histogram_count: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain, JSON-safe dict (round-trips via :meth:`from_dict`)."""
        return {"instrument": self.instrument, "histogram_count": int(self.histogram_count)}

    @classmethod
    def from_dict(cls, data: Any) -> ProfileFingerprint:
        """Reconstruct a fingerprint from :meth:`to_dict` output (lenient)."""
        if not isinstance(data, dict):
            return cls(instrument="", histogram_count=0)
        return cls(
            instrument=str(data.get("instrument", "")),
            histogram_count=_as_int(data.get("histogram_count"), 0) or 0,
        )

    def matches(self, other: ProfileFingerprint) -> bool:
        """Return whether *other* names the same instrument and histogram count."""
        return self.instrument.strip().lower() == other.instrument.strip().lower() and int(
            self.histogram_count
        ) == int(other.histogram_count)


def profile_fingerprint_for_run(run: Run) -> ProfileFingerprint:
    """Return the :class:`ProfileFingerprint` a run matches.

    The instrument name is read from ``run.grouping['instrument']`` when present
    (the loaders store it there) and otherwise from ``run.metadata['instrument']``.
    The histogram count is ``len(run.histograms)``.
    """
    grouping = run.grouping if isinstance(run.grouping, dict) else {}
    instrument = grouping.get("instrument")
    if not instrument:
        instrument = (run.metadata or {}).get("instrument", "")
    return ProfileFingerprint(
        instrument=str(instrument or ""),
        histogram_count=len(run.histograms),
    )


def fingerprint_from_payload(
    payload: dict[str, Any], *, histogram_count: int
) -> ProfileFingerprint:
    """Build a fingerprint from a stored grouping payload + a histogram count."""
    payload = payload if isinstance(payload, dict) else {}
    return ProfileFingerprint(
        instrument=str(payload.get("instrument", "") or ""),
        histogram_count=int(histogram_count),
    )


# --------------------------------------------------------------------------- #
# Instrument self-healing on re-detection
# --------------------------------------------------------------------------- #

#: Structural / instrument-dependent payload keys discarded when a stale
#: persisted instrument identity is corrected by fresh detection. These describe
#: a *detector layout* (which detectors form which group, the named preset the
#: groups came from, the vector projections and their per-axis alphas); once the
#: instrument changes they no longer apply and would paste the wrong instrument's
#: group names/geometry over the run. Everything else — the per-run/file-derived
#: facts (:data:`_PER_RUN_FACT_KEYS`) and the instrument-independent correction
#: *policies* (deadtime/background modes, scalar ``alpha``, binning) — is trivially
#: separable and kept, so the whole payload need not be discarded.
_INSTRUMENT_DEPENDENT_STRUCTURE_KEYS: tuple[str, ...] = (
    "groups",
    "group_names",
    "included_groups",
    "forward_group",
    "backward_group",
    "grouping_preset",
    "projections",
    "excluded_detectors",
    "vector_axis",
    "forward_indices",
    "backward_indices",
    "alpha_x",
    "alpha_y",
    "alpha_z",
    "alpha_x_error",
    "alpha_x_reference_run",
    "alpha_y_error",
    "alpha_y_reference_run",
    "alpha_z_error",
    "alpha_z_reference_run",
)


def _same_instrument_family(a: str, b: str) -> bool:
    """Whether two instrument names denote the same physical instrument.

    Compared by *display name* so layout variants of one instrument — e.g. the
    6-detector ``"GPS"`` (PSI-BIN) and 11-detector ``"GPS-RD"`` (ROOT) — count as
    the same family and never trigger a spurious "heal". Falls back to a plain
    case-insensitive comparison for names outside the registry.
    """
    a_norm, b_norm = str(a or "").strip(), str(b or "").strip()
    if a_norm.lower() == b_norm.lower():
        return True
    return instrument_display_name(a_norm).lower() == instrument_display_name(b_norm).lower()


def detect_instrument_for_run(run: Run) -> str | None:
    """Detection result for a freshly loaded *run* (a fact about the data).

    Thin wrapper over :func:`~asymmetry.core.instrument.detect_instrument` that
    feeds it the run's own histogram count, metadata, and source file — the
    inputs a loader used when it first identified the instrument. Returns the
    canonical instrument name, or ``None`` when detection is inconclusive.
    """
    metadata = run.metadata if isinstance(run.metadata, dict) else None
    return detect_instrument(
        len(run.histograms),
        metadata=metadata,
        source_file=run.source_file or None,
    )


def reconcile_instrument_for_payload(
    run: Run, payload: dict[str, Any] | None
) -> tuple[dict[str, Any], str | None]:
    """Self-heal a stale persisted instrument identity against fresh detection.

    A freshly loaded run is the ground truth for its own instrument: the
    instrument is a fact about the data, not a user preference. An earlier app
    version with broken detection may have persisted the *wrong*
    ``grouping["instrument"]`` (e.g. a real FLAME run saved as "GPS"); re-applying
    that stale payload over the fresh run would pin the wrong profile fingerprint
    and paste the wrong instrument's group names/geometry onto the run.

    Given the freshly loaded *run* and a persisted grouping *payload* about to be
    applied to it, this compares the payload's stored ``instrument`` with
    :func:`detect_instrument_for_run`:

    * **Detection positive and disagrees** with the stored value (different
      physical instrument): detection wins. ``run.grouping["instrument"]`` and, if
      the loader set one, ``run.metadata["instrument"]`` are updated to the
      detected name, and the payload's stale *structural* fields
      (:data:`_INSTRUMENT_DEPENDENT_STRUCTURE_KEYS` — groups, group names,
      forward/backward, preset, projections, per-axis vector alphas) are discarded
      in favour of the loader defaults. Instrument-independent settings (per-run
      facts, deadtime/background policy, scalar alpha, binning) are kept.
    * **Detection returns None** (inconclusive): the stored value is kept and the
      payload is returned unchanged.
    * **Detection agrees** with the stored value: the payload is returned
      unchanged (byte-identical).

    Returns ``(reconciled_payload, note)`` where *note* is a human-readable log
    line when a heal occurred, else ``None``. The returned payload is always a new
    dict when a heal occurred (the input is never mutated); otherwise it is the
    input payload object unchanged.
    """
    payload = payload if isinstance(payload, dict) else {}
    detected = detect_instrument_for_run(run)
    if not detected:
        return payload, None

    stored = str(payload.get("instrument", "") or "").strip()
    if stored and _same_instrument_family(stored, detected):
        return payload, None
    if not stored:
        # No stored identity to contradict — nothing to heal. (Resolution/loader
        # will fill instrument in via the normal fingerprint path.)
        return payload, None

    # Detection disagrees: the freshly loaded data wins. Correct the run's own
    # identity in place so the fingerprint recomputes to the detected instrument.
    if isinstance(run.grouping, dict):
        run.grouping["instrument"] = detected
    if isinstance(run.metadata, dict) and run.metadata.get("instrument"):
        run.metadata["instrument"] = detected

    # Drop the stale payload's instrument-dependent structural fields so the wrong
    # instrument's groups/names/preset cannot survive; keep the rest.
    reconciled = {
        key: value
        for key, value in payload.items()
        if key not in _INSTRUMENT_DEPENDENT_STRUCTURE_KEYS
    }
    reconciled["instrument"] = detected
    note = (
        f"Corrected stale instrument identity '{stored}' -> '{detected}' for "
        f"{run.source_file or 'run'} (detected from the freshly loaded data); "
        "discarded the stale grouping structure in favour of the loader defaults."
    )
    return reconciled, note


# --------------------------------------------------------------------------- #
# GroupingProfile
# --------------------------------------------------------------------------- #


@dataclass
class GroupingProfile:
    """A named, shareable set of grouping settings for one fingerprint.

    Holds only the *shareable* settings (see the module docstring's
    classification table). It is applied to a run via
    :func:`resolve_effective_grouping`, which merges these settings with the
    run's file-derived facts to produce a full ``run.grouping`` payload.
    """

    name: str
    fingerprint: ProfileFingerprint
    #: Marks the fingerprint's *default* profile — the one freshly loaded runs
    #: are assigned to (exactly one per fingerprint). Runs already in the
    #: project follow their own recorded assignment, not this flag.
    active: bool = True

    # Grouping structure -----------------------------------------------------
    groups: dict[int, list[int]] = field(default_factory=dict)
    group_names: dict[int, str] = field(default_factory=dict)
    included_groups: dict[int, bool] = field(default_factory=dict)
    forward_group: int = 1
    backward_group: int = 2
    projections: list[dict[str, Any]] = field(default_factory=list)
    excluded_detectors: list[int] = field(default_factory=list)

    # Policies ---------------------------------------------------------------
    alpha_policy: AlphaPolicy = field(default_factory=AlphaPolicy)
    deadtime_policy: DeadtimePolicy = field(default_factory=DeadtimePolicy)
    background_policy: BackgroundPolicy = field(default_factory=BackgroundPolicy)
    t0_policy: T0Policy = field(default_factory=T0Policy)
    # Intrinsic-asymmetry balance beta = A_{0,B}/A_{0,F} (musrfit asymmetry fit
    # type 2), applied with alpha as A = (F - aB)/(bF + aB). A plain fixed
    # scalar for now — a BetaPolicy with calibration provenance is deliberately
    # deferred until the estimator exists (docs/porting/beta-correction/).
    beta: float = 1.0

    # Binning ----------------------------------------------------------------
    binning_mode: str = "fixed"
    bin0_us: float | None = None
    bin10_us: float | None = None
    bunching_factor: int = 1

    # Periods ----------------------------------------------------------------
    period_mode: str | None = None

    # Free structure keys (grouping_preset, vector_axis, per-axis alphas) -----
    extra: dict[str, Any] = field(default_factory=dict)

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain, JSON-safe dict (round-trips via :meth:`from_dict`)."""
        data: dict[str, Any] = {
            "name": self.name,
            "fingerprint": self.fingerprint.to_dict(),
            "active": bool(self.active),
            "groups": {int(gid): [int(d) for d in dets] for gid, dets in self.groups.items()},
            "group_names": {int(gid): str(name) for gid, name in self.group_names.items()},
            "included_groups": {int(gid): bool(v) for gid, v in self.included_groups.items()},
            "forward_group": int(self.forward_group),
            "backward_group": int(self.backward_group),
            "projections": [dict(p) for p in self.projections],
            "excluded_detectors": [int(d) for d in self.excluded_detectors],
            "alpha_policy": self.alpha_policy.to_dict(),
            "deadtime_policy": self.deadtime_policy.to_dict(),
            "background_policy": self.background_policy.to_dict(),
            "binning_mode": str(self.binning_mode),
            "bunching_factor": int(self.bunching_factor),
        }
        # ``t0_policy`` rides inside the profile dict. It is emitted only when it
        # departs from the ``from_file`` default so existing projects (which
        # never stored one) round-trip unchanged and ``from_dict`` stays lenient
        # — no schema bump is needed.
        if self.t0_policy.mode != "from_file":
            data["t0_policy"] = self.t0_policy.to_dict()
        # ``beta`` follows the same emit-only-when-non-default rule.
        if self.beta != 1.0:
            data["beta"] = float(self.beta)
        if self.bin0_us is not None:
            data["bin0_us"] = float(self.bin0_us)
        if self.bin10_us is not None:
            data["bin10_us"] = float(self.bin10_us)
        if self.period_mode is not None:
            data["period_mode"] = str(self.period_mode)
        if self.extra:
            data["extra"] = dict(self.extra)
        return data

    @classmethod
    def from_dict(cls, data: Any) -> GroupingProfile:
        """Reconstruct a profile from :meth:`to_dict` output (lenient)."""
        if not isinstance(data, dict):
            raise TypeError("GroupingProfile.from_dict requires a dict")
        groups_raw = data.get("groups")
        groups: dict[int, list[int]] = {}
        if isinstance(groups_raw, dict):
            for key, dets in groups_raw.items():
                gid = _as_int(key)
                if gid is None or not isinstance(dets, (list, tuple)):
                    continue
                # Entries may be plain detector ids or (detector_id, t0_bin)
                # pairs — decode like resolve_group_indices() so a pair-carrying
                # payload round-trips without dropping detectors.
                decoded = (v[0] if isinstance(v, (list, tuple)) and v else v for v in dets)
                groups[gid] = [d for d in (_as_int(v) for v in decoded) if d is not None]

        names_raw = data.get("group_names")
        group_names = (
            {gid: str(v) for key, v in names_raw.items() if (gid := _as_int(key)) is not None}
            if isinstance(names_raw, dict)
            else {}
        )
        incl_raw = data.get("included_groups")
        included = (
            {gid: bool(v) for key, v in incl_raw.items() if (gid := _as_int(key)) is not None}
            if isinstance(incl_raw, dict)
            else {}
        )
        proj_raw = data.get("projections")
        projections = (
            [dict(p) for p in proj_raw if isinstance(p, dict)] if isinstance(proj_raw, list) else []
        )
        excl_raw = data.get("excluded_detectors")
        excluded = (
            [d for d in (_as_int(v) for v in excl_raw) if d is not None]
            if isinstance(excl_raw, (list, tuple))
            else []
        )
        return cls(
            name=str(data.get("name", "")),
            fingerprint=ProfileFingerprint.from_dict(data.get("fingerprint")),
            active=bool(data.get("active", True)),
            groups=groups,
            group_names=group_names,
            included_groups=included,
            forward_group=_as_int(data.get("forward_group"), 1) or 1,
            backward_group=_as_int(data.get("backward_group"), 2) or 2,
            projections=projections,
            excluded_detectors=excluded,
            alpha_policy=AlphaPolicy.from_dict(data.get("alpha_policy")),
            deadtime_policy=DeadtimePolicy.from_dict(data.get("deadtime_policy")),
            background_policy=BackgroundPolicy.from_dict(data.get("background_policy")),
            t0_policy=T0Policy.from_dict(data.get("t0_policy")),
            beta=_sanitize_beta(data.get("beta", 1.0)),
            binning_mode=str(data.get("binning_mode", "fixed")),
            bin0_us=None if data.get("bin0_us") is None else _as_float(data.get("bin0_us"), 0.0),
            bin10_us=None if data.get("bin10_us") is None else _as_float(data.get("bin10_us"), 0.0),
            bunching_factor=_as_int(data.get("bunching_factor"), 1) or 1,
            period_mode=None if data.get("period_mode") is None else str(data.get("period_mode")),
            extra=dict(data.get("extra")) if isinstance(data.get("extra"), dict) else {},
        )

    def with_active(self, active: bool) -> GroupingProfile:
        """Return a copy with ``active`` set (profiles are otherwise mutable)."""
        return replace(self, active=bool(active))


# --------------------------------------------------------------------------- #
# Building a profile from an existing run payload (migration / "save as")
# --------------------------------------------------------------------------- #

#: Payload keys that make up the *shareable* alpha provenance carried in
#: ``profile.extra`` (the per-axis vector alphas plus their provenance).
_ALPHA_EXTRA_KEYS = _VECTOR_ALPHA_KEYS


def _t0_policy_from_payload(payload: dict[str, Any]) -> T0Policy:
    """Infer a :class:`T0Policy` from a stored grouping payload.

    A payload records a *value*, not a policy. The inference rule (shared by the
    project migration and the GUI "save as profile" action) is: the mode is
    ``manual`` **only** when the payload's stored common ``t0_bin`` differs from
    the run's file-derived t0 — i.e. the user shifted t0 away from the header.
    Otherwise it is the ``from_file`` default.

    The file-derived t0 is reconstructed from the payload's own per-run facts:
    the max over the file's per-detector ``detector_t0_bins`` when present (PSI /
    ROOT), else the payload's ``t0_bin`` itself (a common-t0 NeXus file has no
    per-detector table, so nothing to differ from). An explicit
    ``effective_detector_t0_bins`` override (written by a manual resolution) is
    an unambiguous manual signal on its own.
    """
    stored_t0 = _as_int(payload.get("t0_bin"))
    if stored_t0 is None:
        return T0Policy(mode="from_file")

    # An explicit effective-t0 override means a manual policy already resolved.
    override = payload.get("effective_detector_t0_bins")
    if isinstance(override, (list, tuple)) and override:
        return T0Policy(mode="manual", value=stored_t0)

    detector_t0 = payload.get("detector_t0_bins")
    if isinstance(detector_t0, (list, tuple)) and detector_t0:
        file_common_t0 = max(
            (v for v in (_as_int(d) for d in detector_t0) if v is not None),
            default=None,
        )
        if file_common_t0 is not None and stored_t0 != int(file_common_t0):
            return T0Policy(mode="manual", value=stored_t0)
    return T0Policy(mode="from_file")


def profile_from_payload(
    payload: dict[str, Any],
    name: str,
    fingerprint: ProfileFingerprint,
    *,
    active: bool = True,
) -> GroupingProfile:
    """Lift the shareable fields out of a full grouping payload into a profile.

    Used both by the v11->v12 migration and by the GUI's "save current settings
    as a profile" action. Per-run/file-derived keys in *payload* (t0, good-bin
    window, per-detector deadtime, good frames, period tables) are ignored — they
    come back from the run at resolve time.
    """
    payload = payload if isinstance(payload, dict) else {}

    groups: dict[int, list[int]] = {}
    groups_raw = payload.get("groups")
    if isinstance(groups_raw, dict):
        for key, dets in groups_raw.items():
            gid = _as_int(key)
            if gid is None or not isinstance(dets, (list, tuple)):
                continue
            ids: list[int] = []
            for value in dets:
                # A detector entry may be a bare id or a ``[id, t0]`` pair.
                raw = value[0] if isinstance(value, (list, tuple)) and value else value
                det = _as_int(raw)
                if det is not None:
                    ids.append(det)
            groups[gid] = ids

    names_raw = payload.get("group_names")
    group_names = (
        {gid: str(v) for key, v in names_raw.items() if (gid := _as_int(key)) is not None}
        if isinstance(names_raw, dict)
        else {}
    )
    incl_raw = payload.get("included_groups")
    included = (
        {gid: bool(v) for key, v in incl_raw.items() if (gid := _as_int(key)) is not None}
        if isinstance(incl_raw, dict)
        else {}
    )
    proj_raw = payload.get("projections")
    projections = (
        [dict(p) for p in proj_raw if isinstance(p, dict)] if isinstance(proj_raw, list) else []
    )
    excl_raw = payload.get("excluded_detectors")
    excluded = (
        [d for d in (_as_int(v) for v in excl_raw) if d is not None]
        if isinstance(excl_raw, (list, tuple))
        else []
    )

    extra: dict[str, Any] = {}
    for key in _STRUCTURE_KEYS:
        if payload.get(key):
            extra[key] = payload[key]
    for key in _ALPHA_EXTRA_KEYS:
        if key in payload:
            extra[key] = payload[key]

    binning_mode = str(payload.get("binning_mode", "fixed"))

    return GroupingProfile(
        name=name,
        fingerprint=fingerprint,
        active=active,
        groups=groups,
        group_names=group_names,
        included_groups=included,
        forward_group=_as_int(payload.get("forward_group"), 1) or 1,
        backward_group=_as_int(payload.get("backward_group"), 2) or 2,
        projections=projections,
        excluded_detectors=excluded,
        alpha_policy=_alpha_policy_from_payload(payload),
        deadtime_policy=_deadtime_policy_from_payload(payload),
        background_policy=_background_policy_from_payload(payload),
        t0_policy=_t0_policy_from_payload(payload),
        beta=_sanitize_beta(payload.get("beta", 1.0)),
        binning_mode=binning_mode,
        bin0_us=None if payload.get("bin0_us") is None else _as_float(payload.get("bin0_us"), 0.0),
        bin10_us=None
        if payload.get("bin10_us") is None
        else _as_float(payload.get("bin10_us"), 0.0),
        bunching_factor=_as_int(payload.get("bunching_factor"), 1) or 1,
        period_mode=None if payload.get("period_mode") is None else str(payload.get("period_mode")),
        extra=extra,
    )


def _alpha_policy_from_payload(payload: dict[str, Any]) -> AlphaPolicy:
    """Derive an :class:`AlphaPolicy` from a payload's alpha + provenance keys.

    A payload records a *value*, not a policy. We infer: an alpha carrying a
    calibration method/reference (``alpha_method`` other than the bare default,
    or an ``alpha_reference_run``) becomes ``calibrated``; otherwise the value is
    a plain ``fixed`` value. The migration never infers ``per_run_estimate`` —
    that is an explicit user choice with no payload footprint.
    """
    value = _as_float(payload.get("alpha", 1.0), 1.0)
    method = str(payload.get("alpha_method", "")).strip()
    reference = _as_int(payload.get("alpha_reference_run"))
    error = payload.get("alpha_error")
    calibrated_methods = {"calibrate", "count_fit", "diamagnetic", "general", "ratio"}
    if reference is not None or method in calibrated_methods:
        return AlphaPolicy(
            mode="calibrated",
            value=value,
            error=None if error is None else _as_float(error, 0.0),
            method=method,
            source_run=reference,
        )
    return AlphaPolicy(mode="fixed", value=value)


def _deadtime_policy_from_payload(payload: dict[str, Any]) -> DeadtimePolicy:
    """Derive a :class:`DeadtimePolicy` from a payload's deadtime keys."""
    if not bool(payload.get("deadtime_correction", False)):
        return DeadtimePolicy(mode="off")
    mode = str(payload.get("deadtime_mode", payload.get("deadtime_method", "off"))).strip().lower()
    if mode == "load":
        mode = "manual"
    if mode not in DEADTIME_POLICY_MODES:
        mode = "off"
    if mode == "off":
        return DeadtimePolicy(mode="off")
    if mode == "file":
        # Payload's ``deadtime_mode`` uses "file"; the profile policy names it
        # "from_file" to make the file-derived nature explicit.
        return DeadtimePolicy(mode="from_file")
    raw_values = payload.get("dead_time_us")
    if not isinstance(raw_values, (list, tuple)):
        raw_values = payload.get("deadtime_loaded_us")
    values = (
        [_as_float(v, 0.0) for v in raw_values] if isinstance(raw_values, (list, tuple)) else []
    )
    return DeadtimePolicy(
        mode=mode,
        values=values,
        manual_us=None
        if payload.get("deadtime_manual_us") is None
        else _as_float(payload.get("deadtime_manual_us"), 0.0),
        estimated_us=None
        if payload.get("deadtime_estimated_us") is None
        else _as_float(payload.get("deadtime_estimated_us"), 0.0),
        method=str(payload.get("deadtime_method", "")),
        source_run=_as_int(payload.get("deadtime_reference_run")),
        source_path=str(payload.get("deadtime_source_path", "")),
    )


def _background_policy_from_payload(payload: dict[str, Any]) -> BackgroundPolicy:
    """Derive a :class:`BackgroundPolicy` from a payload's background keys."""
    if not bool(payload.get("background_correction", False)):
        return BackgroundPolicy(mode="none")
    mode = str(payload.get("background_mode", "")).strip().lower()
    if mode not in BACKGROUND_POLICY_MODES or mode == "none":
        # Fall back to the same heuristic transform.background uses.
        has_fixed = any(
            isinstance(payload.get(key), (list, tuple))
            for key in ("background_fixed_values", "background_fix", "bkg_fix")
        )
        mode = "fixed" if has_fixed else "range"
    details = {key: payload[key] for key in _BACKGROUND_DETAIL_KEYS if key in payload}
    return BackgroundPolicy(mode=mode, details=details)


# --------------------------------------------------------------------------- #
# Resolution: profile + run -> full grouping payload
# --------------------------------------------------------------------------- #


def resolve_effective_grouping(
    profile: GroupingProfile,
    run: Run,
    *,
    reference_resolver: ReferenceResolver | None = None,
) -> dict[str, Any]:
    """Merge *profile* with *run*'s file-derived facts into a grouping payload.

    The returned dict has exactly the shape of today's ``run.grouping`` so it can
    be assigned to ``run.grouping`` and reduced/fitted/exported unchanged. The
    profile supplies the shareable settings; the run supplies the per-run facts
    (t0, good-bin window, per-detector tables, good frames, period data), and any
    policy that needs a computed value (per-run alpha estimate, file deadtime) is
    evaluated against the run here.

    ``reference_resolver`` resolves the ``reference_run`` background mode's
    reference histograms + good-frame scale (the caller supplies one backed by a
    loaded-dataset registry). It is only consulted by a ``per_run_estimate`` alpha
    policy paired with ``reference_run`` background, so the estimate balances the
    same background-subtracted counts the reduction will; ``None`` skips that
    subtraction for the estimate.

    Detector ids in the profile that fall outside ``1..n_detectors`` are handled
    gracefully by the reduction chokepoints (:func:`effective_group_indices`
    drops out-of-range ids), exactly as they are for a file-derived grouping.
    """
    with perf_timer(
        "core.grouping.resolve_effective",
        t0_mode=profile.t0_policy.mode,
        alpha_mode=profile.alpha_policy.mode,
        deadtime_mode=profile.deadtime_policy.mode,
    ) as perf:
        run_grouping = run.grouping if isinstance(run.grouping, dict) else {}
        n_hist = len(run.histograms)

        grouping: dict[str, Any] = {
            "groups": {int(gid): [int(d) for d in dets] for gid, dets in profile.groups.items()},
            "group_names": {int(gid): str(name) for gid, name in profile.group_names.items()},
            "forward_group": int(profile.forward_group),
            "backward_group": int(profile.backward_group),
            "excluded_detectors": [int(d) for d in profile.excluded_detectors],
            "bunching_factor": int(profile.bunching_factor),
        }
        if profile.included_groups:
            grouping["included_groups"] = {
                int(gid): bool(v) for gid, v in profile.included_groups.items()
            }
        if profile.projections:
            grouping["projections"] = [dict(p) for p in profile.projections]

        # Structure/provenance extras (grouping_preset, vector_axis, per-axis alpha).
        for key, value in profile.extra.items():
            grouping[key] = value

        # Instrument is part of the fingerprint but also a payload key downstream
        # code reads (background gating, fingerprinting on reload).
        if profile.fingerprint.instrument:
            grouping["instrument"] = profile.fingerprint.instrument

        # -- per-run / file-derived facts ---------------------------------------
        _copy_per_run_facts(grouping, run_grouping, run)

        # -- binning ------------------------------------------------------------
        if profile.binning_mode and profile.binning_mode != "fixed":
            grouping["binning_mode"] = profile.binning_mode
            if profile.bin0_us is not None:
                grouping["bin0_us"] = float(profile.bin0_us)
            if profile.binning_mode == "variable" and profile.bin10_us is not None:
                grouping["bin10_us"] = float(profile.bin10_us)

        # -- period mode --------------------------------------------------------
        if profile.period_mode is not None:
            grouping["period_mode"] = str(profile.period_mode)
        elif "period_mode" in run_grouping:
            grouping["period_mode"] = run_grouping["period_mode"]

        # -- policies -----------------------------------------------------------
        # Order matters: a per_run_estimate alpha must be measured on the same
        # deadtime-corrected, background-subtracted counts the reduction applies
        # it to, so deadtime and background resolve into the grouping *before* the
        # alpha estimate reads them (see
        # docs/porting/correction-order-alpha-estimation).
        _apply_t0_policy(grouping, profile.t0_policy, run, n_hist)
        _apply_deadtime_policy(grouping, profile.deadtime_policy, run_grouping, n_hist)
        _apply_background_policy(grouping, profile.background_policy)
        _apply_alpha_policy(
            grouping, profile.alpha_policy, run, n_hist, reference_resolver=reference_resolver
        )
        # Beta is a fixed scalar (no per-run resolution); written only when
        # active so a beta = 1 payload stays byte-identical to today's. It is
        # scalar-only: a projection-carrying (vector) profile never emits it —
        # a per-pair beta would be a different feature (docs/porting/
        # beta-correction/), and a uniform one would be wrong for every pair
        # but the one it was measured on.
        beta = _sanitize_beta(profile.beta)
        if beta != 1.0 and not profile.projections:
            grouping["beta"] = beta

        perf.detail(n_detectors=n_hist, n_groups=len(grouping.get("groups", {})))
        return grouping


#: Per-run, file-derived keys copied straight through from the run's own
#: grouping (never stored on a profile).
_PER_RUN_FACT_KEYS = (
    "t0_bin",
    "t_good_offset",
    "first_good_bin",
    "last_good_bin",
    "bin_index_base",
    "detector_t0_bins",
    "detector_first_good_bins",
    "detector_last_good_bins",
    "histogram_labels",
    "root_histo_numbers",
    "good_frames",
    "period_histograms",
    "period_reduced",
    "period_good_frames",
    "period_dead_time_us",
    "period_mapping",
)


def _copy_per_run_facts(grouping: dict[str, Any], run_grouping: dict[str, Any], run: Run) -> None:
    """Copy the file-derived per-run facts from the run's grouping."""
    for key in _PER_RUN_FACT_KEYS:
        if key in run_grouping:
            grouping[key] = run_grouping[key]
    # t0_bin has a sensible fall-back to the first histogram's t0 when the run
    # grouping did not record one (matches the loaders / GUI extract behaviour).
    if "t0_bin" not in grouping and run.histograms:
        grouping["t0_bin"] = int(run.histograms[0].t0_bin)


def _file_common_t0(grouping: dict[str, Any], run: Run, n_hist: int) -> int:
    """The file-derived common t0: max histogram t0 over the analysis groups.

    Reproduces :func:`common_t0_for_groups` over the forward+backward groups,
    the same value today's reduction aligns to. Falls back to the copied
    ``t0_bin`` (then the first histogram's t0, then 0) when the groups reference
    no present detectors.
    """
    forward_gid = int(grouping.get("forward_group", 1))
    backward_gid = int(grouping.get("backward_group", 2))
    forward_idx = effective_group_indices(grouping, forward_gid, n_histograms=n_hist)
    backward_idx = effective_group_indices(grouping, backward_gid, n_histograms=n_hist)
    if run.histograms and (forward_idx or backward_idx):
        return common_t0_for_groups(run.histograms, forward_idx, backward_idx)
    stored = _as_int(grouping.get("t0_bin"))
    if stored is not None:
        return stored
    return int(run.histograms[0].t0_bin) if run.histograms else 0


def _apply_t0_policy(grouping: dict[str, Any], policy: T0Policy, run: Run, n_hist: int) -> None:
    """Resolve the analysis time-zero into the grouping per the policy.

    ``from_file`` (default) leaves the copied per-run t0 facts untouched — the
    payload is bit-identical to today's file-derived resolution. ``manual`` and
    ``auto_detect`` rewrite the common ``t0_bin`` (and ``first_good_bin`` so the
    good-window offset from t0 is preserved) and, for per-detector data, publish
    the effective per-detector t0 bins under
    :data:`~asymmetry.core.transform.grouping.EFFECTIVE_DETECTOR_T0_KEY` so
    reduction aligns on them without ``Histogram.t0_bin`` ever being mutated.
    """
    if policy.mode == "from_file" or not run.histograms:
        return

    file_common_t0 = _file_common_t0(grouping, run, n_hist)

    if policy.mode == "manual":
        if policy.value is None:
            return
        new_common_t0 = max(0, int(policy.value))
    else:  # auto_detect
        search = find_t0_for_run(run.histograms, run.metadata)
        if not search.ok:
            return
        new_common_t0 = max(0, int(search.consensus_t0_bin))
        grouping["t0_search_strategy"] = str(search.strategy)
        grouping["t0_search_spread_bins"] = int(search.spread_bins)

    delta = new_common_t0 - int(file_common_t0)
    if delta == 0:
        return

    # Apply the common-t0 shift as an OFFSET on each detector's file t0 (the
    # semantics the destructive MainWindow rewrite used), but only in the
    # payload — the run's histograms are never touched.
    grouping[EFFECTIVE_DETECTOR_T0_KEY] = [
        max(0, int(hist.t0_bin) + delta) for hist in run.histograms
    ]
    grouping["t0_bin"] = new_common_t0
    # Keep the good-window offset from t0 fixed: shift first_good_bin with t0.
    first_good = _as_int(grouping.get("first_good_bin"))
    if first_good is not None:
        grouping["first_good_bin"] = max(0, first_good + delta)


def _apply_alpha_policy(
    grouping: dict[str, Any],
    policy: AlphaPolicy,
    run: Run,
    n_hist: int,
    *,
    reference_resolver: ReferenceResolver | None = None,
) -> None:
    """Write ``alpha`` (and provenance) into the grouping per the policy."""
    if policy.mode == "per_run_estimate":
        alpha = _estimate_run_alpha(grouping, run, n_hist, reference_resolver=reference_resolver)
        grouping["alpha"] = float(alpha)
        grouping["alpha_method"] = "per_run_estimate"
        return
    grouping["alpha"] = float(policy.value)
    if policy.mode == "calibrated":
        if policy.method:
            grouping["alpha_method"] = str(policy.method)
        if policy.error is not None:
            grouping["alpha_error"] = float(policy.error)
        if policy.source_run is not None:
            grouping["alpha_reference_run"] = int(policy.source_run)


def _estimate_run_alpha(
    grouping: dict[str, Any],
    run: Run,
    n_hist: int,
    *,
    reference_resolver: ReferenceResolver | None = None,
) -> float:
    """Forward/backward integral-ratio alpha for this run (Mantid ``AlphaCalc``).

    Estimates on the *corrected* forward/backward counts — deadtime-corrected,
    grouped and background-subtracted by :func:`corrected_grouped_counts`, exactly
    the spectra the reduction forms the asymmetry from — so the balance centres
    the reduced (background-subtracted) asymmetry rather than the raw totals (see
    ``docs/porting/correction-order-alpha-estimation``). The deadtime and
    background policies must already be resolved into ``grouping`` (they are, by
    the policy order in :func:`resolve_effective_grouping`). Falls back to 1.0
    when the groups reference no present detectors (:func:`estimate_alpha` also
    floors a non-positive backward sum).
    """
    forward_gid = int(grouping.get("forward_group", 1))
    backward_gid = int(grouping.get("backward_group", 2))
    forward_idx = effective_group_indices(grouping, forward_gid, n_histograms=n_hist)
    backward_idx = effective_group_indices(grouping, backward_gid, n_histograms=n_hist)
    if not forward_idx or not backward_idx:
        return 1.0
    flags = correction_flags_from_grouping(grouping)
    facility = str((run.metadata or {}).get("facility", grouping.get("instrument", "")))
    corrected = corrected_grouped_counts(
        histograms=run.histograms,
        grouping=grouping,
        forward_idx=forward_idx,
        backward_idx=backward_idx,
        use_deadtime=flags.use_deadtime,
        deadtime_mode=flags.deadtime_mode,
        use_background=flags.use_background,
        facility=facility,
        reference_resolver=reference_resolver,
    )
    first_good = _as_int(grouping.get("first_good_bin"))
    last_good = _as_int(grouping.get("last_good_bin"))
    return estimate_alpha(
        corrected.forward,
        corrected.backward,
        first_good_bin=first_good,
        last_good_bin=last_good,
    )


def _apply_deadtime_policy(
    grouping: dict[str, Any],
    policy: DeadtimePolicy,
    run_grouping: dict[str, Any],
    n_hist: int,
) -> None:
    """Write deadtime keys into the grouping per the policy."""
    if policy.mode == "off":
        grouping["deadtime_correction"] = False
        grouping["deadtime_mode"] = "off"
        return

    grouping["deadtime_correction"] = True

    if policy.mode == "from_file":
        grouping["deadtime_mode"] = "file"
        grouping["deadtime_method"] = "file"
        # The reduction reads the run's own file deadtime values under
        # ``dead_time_us``; carry them through from the run so the "file" mode
        # remains driven by per-run facts.
        if isinstance(run_grouping.get("dead_time_us"), (list, tuple)):
            grouping["dead_time_us"] = list(run_grouping["dead_time_us"])
        elif isinstance(run_grouping.get("deadtime_loaded_us"), (list, tuple)):
            grouping["dead_time_us"] = list(run_grouping["deadtime_loaded_us"])
        return

    if policy.mode == "manual":
        grouping["deadtime_mode"] = "manual"
        grouping["deadtime_method"] = policy.method or "manual"
        if policy.values:
            grouping["dead_time_us"] = [float(v) for v in policy.values]
        if policy.manual_us is not None:
            grouping["deadtime_manual_us"] = float(policy.manual_us)
        if policy.source_run is not None:
            grouping["deadtime_reference_run"] = int(policy.source_run)
        if policy.source_path:
            grouping["deadtime_source_path"] = str(policy.source_path)
        return

    if policy.mode == "estimate":
        grouping["deadtime_mode"] = "estimate"
        grouping["deadtime_method"] = "estimate"
        tau = policy.estimated_us
        if tau is not None:
            grouping["deadtime_estimated_us"] = float(tau)
            # Broadcast the single estimated value across this run's detectors.
            grouping["dead_time_us"] = [float(tau)] * n_hist
        elif policy.values:
            grouping["dead_time_us"] = [float(v) for v in policy.values]
        if policy.source_run is not None:
            grouping["deadtime_reference_run"] = int(policy.source_run)
        return


def _apply_background_policy(grouping: dict[str, Any], policy: BackgroundPolicy) -> None:
    """Write background keys into the grouping per the policy."""
    if policy.mode == "none":
        grouping["background_correction"] = False
        grouping["background_mode"] = "none"
        return
    grouping["background_correction"] = True
    grouping["background_mode"] = policy.mode
    for key, value in policy.details.items():
        grouping[key] = value


# --------------------------------------------------------------------------- #
# Application helper: attach the effective grouping for a freshly loaded run
# --------------------------------------------------------------------------- #


def default_profile_for_run(profiles: list[GroupingProfile], run: Run) -> GroupingProfile | None:
    """Return the *default* profile for *run*'s fingerprint, or ``None``.

    The ``active`` flag marks each fingerprint's default profile — the one
    freshly loaded runs are assigned to. When several profiles share a
    fingerprint only the first flagged one is returned (the project maintains
    a single default per fingerprint). Runs already in the project follow
    their own recorded assignment; see :func:`assigned_profile_for_run`.
    """
    fingerprint = profile_fingerprint_for_run(run)
    for profile in profiles:
        if profile.active and profile.fingerprint.matches(fingerprint):
            return profile
    return None


def named_profile_for_run(
    profiles: list[GroupingProfile], name: str, run: Run
) -> GroupingProfile | None:
    """Return the profile called *name* matching *run*'s fingerprint, or ``None``.

    Both the name and the fingerprint must match: a profile of another
    instrument (or histogram count) that happens to share the name never
    resolves onto the run.
    """
    fingerprint = profile_fingerprint_for_run(run)
    for profile in profiles:
        if profile.name == name and profile.fingerprint.matches(fingerprint):
            return profile
    return None


def assigned_profile_for_run(
    profiles: list[GroupingProfile], run: Run, assigned_name: str | None = None
) -> GroupingProfile | None:
    """Resolve the profile *run* should follow: its assignment, else the default.

    *assigned_name* is the run's recorded assignment (schema v17). A missing or
    stale assignment — the named profile was deleted, or the run's fingerprint
    no longer matches it — falls back to the fingerprint's default profile, so
    a project whose profile list changed underneath a run still resolves.
    """
    if assigned_name:
        profile = named_profile_for_run(profiles, str(assigned_name), run)
        if profile is not None:
            return profile
    return default_profile_for_run(profiles, run)


def effective_grouping_for_loaded_run(profiles: list[GroupingProfile], run: Run) -> dict[str, Any]:
    """Grouping payload to attach to a freshly loaded run.

    If a default profile matches the run's fingerprint, its resolved effective
    grouping is returned; otherwise the run keeps the loader's own default
    grouping payload (a shallow copy so the caller can mutate it safely).

    The GUI calls this after loading a run so the run inherits its fingerprint's
    default profile without the loader needing to know about profiles. No GUI
    wiring lives here — this is the pure decision + resolution.
    """
    profile = default_profile_for_run(profiles, run)
    if profile is not None:
        return resolve_effective_grouping(profile, run)
    return dict(run.grouping) if isinstance(run.grouping, dict) else {}


__all__ = [
    "ALPHA_POLICY_MODES",
    "DEADTIME_POLICY_MODES",
    "BACKGROUND_POLICY_MODES",
    "T0_POLICY_MODES",
    "AlphaPolicy",
    "DeadtimePolicy",
    "BackgroundPolicy",
    "T0Policy",
    "ProfileFingerprint",
    "GroupingProfile",
    "profile_fingerprint_for_run",
    "fingerprint_from_payload",
    "detect_instrument_for_run",
    "reconcile_instrument_for_payload",
    "profile_from_payload",
    "resolve_effective_grouping",
    "default_profile_for_run",
    "named_profile_for_run",
    "assigned_profile_for_run",
    "effective_grouping_for_loaded_run",
]

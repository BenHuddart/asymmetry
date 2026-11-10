"""Named grouping profiles combine shareable choices with each run's acquired facts.

Runs follow their assigned profile until released into a per-run override.
One active profile per instrument/detector fingerprint supplies fresh-run defaults.
Resolution keeps file timing, good-bin windows, deadtime normalisers, and exposure
attached to the acquisition. Policies own authored choices; calibrated policies
reference captured results. Runtime coefficient views are derived from those owners.

Shareable keys live on the profile; per-run keys come from the selected run:

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
beta                              shareable*  value depends on beta_policy (see below);
                                              scalar-only, never on a vector profile
beta_method                       shareable   provenance ("count_fit"/"single_histogram")
beta_error / beta_reference_run   shareable   calibration provenance
excluded_detectors                shareable   1-based ids dropped from every group
period_mode                       shareable   red/green/green_minus_red/green_plus_red
binning_mode                      shareable   fixed/variable/constant_error
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
background_run                    shareable   source/period, timing, and explicit scale request
background_reference_run          shareable   count-fit background provenance
instrument                        fingerprint part of the profile fingerprint
--------------------------------  ----------  -----------------------------------------------
t0_bin                            per-run     analysis time-zero bin (file-derived)
t_good_offset                     per-run*    good-window offset from t0
first_good_bin / last_good_bin    per-run*    good-window bounds
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
period_histograms, period_inputs  per-run     held sets and original acquired inputs
period_reduced                    per-run     per-period reduced arrays cache
period_good_frames                per-run     per-period good frames
period_dead_time_us               per-run     per-period deadtime tables
period_mapping                    per-run     period-number -> label map
================================  ==========  ===============================================

``*`` marks a key whose *value* is produced by resolution from a profile policy
rather than copied verbatim: ``alpha`` (per-run estimate mode computes the
integral ratio), ``dead_time_us`` (``from_file`` mode reads the run's own
values), the good-window keys ``t_good_offset``/``first_good_bin``/
``last_good_bin`` (``manual`` ``good_window_policy`` places them relative to the
run's effective t0; ``from_file`` keeps the run's own), and the
``*_correction`` flags (on/off derived from the policy mode).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from numbers import Integral
from typing import Any

from asymmetry.core.data.dataset import Run
from asymmetry.core.instrument import detect_instrument, instrument_display_name
from asymmetry.core.project.calibration import (
    CalibratedCoefficient,
    CalibrationRecord,
    QuickCalibratedCoefficient,
    profile_calibration_payload,
    resolve_projection_calibrations,
    restore_profile_calibrations,
)
from asymmetry.core.transform.asymmetry import estimate_alpha
from asymmetry.core.transform.corrections import CorrectionUnavailableError
from asymmetry.core.transform.deadtime import file_deadtime_table
from asymmetry.core.transform.fb_counts import AlignmentCrop
from asymmetry.core.transform.grouping import (
    InvalidGroupingError,
    common_t0_for_groups,
    effective_group_indices,
    independent_pair_indices,
    resolve_alignment,
)
from asymmetry.core.transform.rebin import BinningSpec, binning_spec
from asymmetry.core.transform.reduce import (
    ReferenceResolver,
    corrected_period_counts,
    correction_flags_from_grouping,
)
from asymmetry.core.transform.t0 import (
    detector_t0_overrides,
    effective_detector_t0_bins,
    find_t0_for_run,
    resolve_selected_timing,
)
from asymmetry.core.utils.perf import perf_timer

# --------------------------------------------------------------------------- #
# Policy defaults and vocabulary
# --------------------------------------------------------------------------- #

#: Alpha policy modes.
ALPHA_POLICY_MODES = ("fixed", "calibrated", "per_run_estimate")

#: Beta policy modes — no ``per_run_estimate`` (not meaningful for β in v1;
#: it is an instrument/detector-pair property, not a per-run quantity).
BETA_POLICY_MODES = ("fixed", "calibrated")

#: Deadtime policy modes (mirror the grouping dialog's deadtime modes).
DEADTIME_POLICY_MODES = ("off", "from_file", "manual", "estimate")

#: Background policy modes (mirror ``transform.background.BACKGROUND_MODES``).
BACKGROUND_POLICY_MODES = ("none", "range", "tail_fit", "reference_run", "fixed")

#: Time-zero policy modes.
T0_POLICY_MODES = ("from_file", "manual", "auto_detect")

#: Good-window policy modes (one selector governs both ends of the window).
GOOD_WINDOW_POLICY_MODES = ("from_file", "manual")

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
      integral ratio (Mantid ``AlphaCalc`` convention).
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
class BetaPolicy:
    """How a profile determines the intrinsic-asymmetry balance ``beta`` for a run.

    * ``fixed`` — every run gets the same ``value`` (default 1.0, i.e. off).
    * ``calibrated`` — a value measured once on a weak-TF calibration run
      (``method``/``source_run``) and applied to every run, with an optional
      ``error``.

    Unlike :class:`AlphaPolicy` there is no ``per_run_estimate`` mode: beta is
    broadly an instrument-dependent property of a detector pair, not something
    resolution computes per run (docs/porting/beta-correction/).
    """

    mode: str = "fixed"
    value: float = 1.0
    error: float | None = None
    method: str = ""
    source_run: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain, JSON-safe dict (round-trips via :meth:`from_dict`)."""
        data: dict[str, Any] = {"mode": self.mode, "value": float(self.value)}
        if self.mode == "calibrated":
            if self.error is not None:
                data["error"] = float(self.error)
            if self.method:
                data["method"] = str(self.method)
            if self.source_run is not None:
                data["source_run"] = int(self.source_run)
        return data

    @classmethod
    def from_dict(cls, data: Any) -> BetaPolicy:
        """Reconstruct a policy from :meth:`to_dict` output (lenient; defaults to ``fixed``)."""
        if not isinstance(data, dict):
            return cls()
        mode = str(data.get("mode", "fixed")).strip().lower()
        if mode not in BETA_POLICY_MODES:
            mode = "fixed"
        return cls(
            mode=mode,
            value=_sanitize_beta(data.get("value", 1.0)),
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
      exactly as today. Resolution stores nothing.
    * ``manual`` — a signed *offset* in bins (:attr:`offset_bins`) from each
      run's own file-derived common t0, so one profile shifts a whole set of
      runs consistently however their headers differ (decision D3). Resolution
      writes the resulting effective per-detector t0 bins into the payload
      (``effective_detector_t0_bins``) so reduction aligns on them **without**
      mutating ``Histogram.t0_bin`` — the run's histograms stay exactly as
      loaded.
    * ``auto_detect`` — runs :func:`~asymmetry.core.transform.t0.find_t0_for_run`
      per run at resolution time; :attr:`strategy` and :attr:`spread_bins` carry
      the last detection's provenance for display.

    :attr:`legacy_value` holds the pre-v21 absolute-bin ``value`` a project file
    may still carry. It is *not* resolvable — only the run's own file t0 turns it
    into an offset (:meth:`from_legacy_value`), and the conversion happens on
    project open, where the runs are known.
    """

    mode: str = "from_file"
    offset_bins: int | None = None
    legacy_value: int | None = None
    strategy: str = ""
    spread_bins: int | None = None
    source_run: int | None = None

    @classmethod
    def from_legacy_value(cls, legacy_value: int, file_common_t0: int) -> T0Policy:
        """Convert a pre-v21 absolute-bin manual t0 into the equivalent offset (D3).

        *file_common_t0* is the reference run's own file-derived common t0 — the
        value the stored absolute bin was typed against. An offset of 0 means the
        stored bin never differed from the file, i.e. the policy was mislabelled
        manual and is really ``from_file``.
        """
        offset = int(legacy_value) - int(file_common_t0)
        if offset == 0:
            return cls(mode="from_file")
        return cls(mode="manual", offset_bins=offset)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain, JSON-safe dict (round-trips via :meth:`from_dict`)."""
        data: dict[str, Any] = {"mode": self.mode}
        if self.mode == "manual":
            if self.offset_bins is not None:
                data["offset_bins"] = int(self.offset_bins)
            elif self.legacy_value is not None:
                data["value"] = int(self.legacy_value)
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
        """Reconstruct a policy from :meth:`to_dict` output (lenient; defaults to ``from_file``).

        A pre-v21 absolute ``value`` (also written by the v20→v21 migration as
        ``legacy_value``) lands in :attr:`legacy_value`, which resolution
        rejects until project open has converted it against a run's file t0.
        """
        if not isinstance(data, dict):
            return cls()
        mode = str(data.get("mode", "from_file")).strip().lower()
        if mode not in T0_POLICY_MODES:
            mode = "from_file"
        legacy = data.get("legacy_value", data.get("value"))
        return cls(
            mode=mode,
            offset_bins=_as_int(data.get("offset_bins")),
            legacy_value=_as_int(legacy),
            strategy=str(data.get("strategy", "")),
            spread_bins=_as_int(data.get("spread_bins")),
            source_run=_as_int(data.get("source_run")),
        )


@dataclass
class GoodWindowPolicy:
    """File bounds or signed authored offsets from effective t0.

    An omitted manual endpoint retains its mapped file limit; acquired facts
    remain separate from the resolved aligned interval. Malformed authored values
    survive persistence for editing and become unavailable at resolution.
    """

    mode: str = "from_file"
    first_offset_bins: Any = None
    last_offset_bins: Any = None

    def __post_init__(self) -> None:
        if self.mode not in GOOD_WINDOW_POLICY_MODES:
            raise ValueError(f"Unknown good window mode {self.mode!r}.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain, JSON-safe dict (round-trips via :meth:`from_dict`)."""
        data: dict[str, Any] = {"mode": self.mode}
        if self.mode == "manual":
            for name, value in (
                ("first_offset_bins", self.first_offset_bins),
                ("last_offset_bins", self.last_offset_bins),
            ):
                if value is not None:
                    data[name] = (
                        int(value)
                        if isinstance(value, Integral) and not isinstance(value, bool)
                        else value
                    )
        return data

    @classmethod
    def from_dict(cls, data: Any) -> GoodWindowPolicy:
        """Reconstruct a policy from :meth:`to_dict` output (lenient; defaults to ``from_file``)."""
        if not isinstance(data, dict):
            return cls()
        mode = str(data.get("mode", "from_file")).strip().lower()
        if mode not in GOOD_WINDOW_POLICY_MODES:
            mode = "from_file"
        return cls(
            mode=mode,
            first_offset_bins=data.get("first_offset_bins"),
            last_offset_bins=data.get("last_offset_bins"),
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

#: Detector membership and geometry belong to the detected layout. Acquisition
#: facts and instrument-independent correction policies survive identity repair.
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


def _base_binning_payload(binning: BinningSpec) -> dict[str, Any]:
    """Serialize a profile's base binning using the established project keys."""
    return {
        "binning_mode": binning.mode,
        "bunching_factor": int(binning.factor),
        "bin0_us": float(binning.bin0_us),
        "bin10_us": float(binning.bin10_us),
    }


@dataclass
class GroupingProfile:
    """A named, shareable set of grouping settings for one fingerprint.

    Holds only the *shareable* settings (see the module docstring's
    classification table). It is applied to a run via
    :func:`resolve_effective_grouping`, which merges these settings with the
    run's file-derived facts to produce a full ``run.grouping`` payload.
    The binning spec records base bins (merge=1); CLI and display merges are
    applied to those bins as separate operations.
    """

    name: str
    fingerprint: ProfileFingerprint
    #: Marks the fingerprint's *default* profile — the one freshly loaded runs
    #: are assigned to (exactly one per fingerprint). Runs already in the
    #: project follow their own recorded assignment, not this flag.
    active: bool = True
    #: The profile's identity colour (a ``#rrggbb`` hex string), worn by its
    #: runs in the Data Browser and by the grouping window's scope rows and
    #: editing strip. Assigned from the GUI palette when first saved; ``None``
    #: (older files) falls back to a palette position at display time. Stored
    #: additively — absent keys round-trip as ``None``.
    color: str | None = None

    # Grouping structure -----------------------------------------------------
    groups: dict[int, list[int]] = field(default_factory=dict)
    group_names: dict[int, str] = field(default_factory=dict)
    included_groups: dict[int, bool] = field(default_factory=dict)
    forward_group: int = 1
    backward_group: int = 2
    projections: list[dict[str, Any]] = field(default_factory=list)
    excluded_detectors: list[int] = field(default_factory=list)

    # Policies ---------------------------------------------------------------
    alpha_policy: AlphaPolicy | CalibratedCoefficient | QuickCalibratedCoefficient = field(
        default_factory=AlphaPolicy
    )
    deadtime_policy: DeadtimePolicy = field(default_factory=DeadtimePolicy)
    background_policy: BackgroundPolicy = field(default_factory=BackgroundPolicy)
    t0_policy: T0Policy = field(default_factory=T0Policy)
    good_window_policy: GoodWindowPolicy = field(default_factory=GoodWindowPolicy)
    # Intrinsic-asymmetry balance beta = A_{0,B}/A_{0,F} (musrfit asymmetry fit
    # type 2), applied with alpha as A = (F - aB)/(bF + aB). Mirrors
    # alpha_policy minus per_run_estimate (docs/porting/beta-correction/).
    beta_policy: BetaPolicy | CalibratedCoefficient = field(default_factory=BetaPolicy)
    calibration_records: tuple[CalibrationRecord, ...] = ()
    projection_alpha_policies: dict[str, CalibratedCoefficient | QuickCalibratedCoefficient] = (
        field(default_factory=dict)
    )

    # Binning ----------------------------------------------------------------
    binning: BinningSpec = field(default_factory=BinningSpec)

    # Periods ----------------------------------------------------------------
    period_mode: str | None = None

    # Free structure keys (grouping_preset, vector_axis, per-axis alphas) -----
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.binning.merge != 1:
            raise ValueError("Grouping profiles store base binning; merge must be 1.")

    @property
    def beta(self) -> float:
        """Effective, sanitized beta scalar (read-only; set via ``beta_policy``)."""
        return _sanitize_beta(self.beta_policy.value)

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain, JSON-safe dict (round-trips via :meth:`from_dict`)."""
        data: dict[str, Any] = {
            "name": self.name,
            "fingerprint": self.fingerprint.to_dict(),
            "active": bool(self.active),
            "groups": {int(gid): list(dets) for gid, dets in self.groups.items()},
            "group_names": {int(gid): str(name) for gid, name in self.group_names.items()},
            "included_groups": {int(gid): bool(v) for gid, v in self.included_groups.items()},
            "forward_group": int(self.forward_group),
            "backward_group": int(self.backward_group),
            "projections": [dict(p) for p in self.projections],
            "excluded_detectors": list(self.excluded_detectors),
            "alpha_policy": self.alpha_policy.to_dict(),
            "deadtime_policy": self.deadtime_policy.to_dict(),
            "background_policy": self.background_policy.to_dict(),
            **_base_binning_payload(self.binning),
        }
        if self.t0_policy.mode != "from_file":
            data["t0_policy"] = self.t0_policy.to_dict()
        if self.good_window_policy.mode != "from_file":
            data["good_window_policy"] = self.good_window_policy.to_dict()
        _apply_beta_policy(data, self.beta_policy)
        if self.color is not None:
            data["color"] = str(self.color)
        if self.period_mode is not None:
            data["period_mode"] = str(self.period_mode)
        if self.extra:
            data["extra"] = dict(self.extra)
        return profile_calibration_payload(self, data)

    @classmethod
    def from_dict(cls, data: Any) -> GroupingProfile:
        """Read a profile, preserving member values for validation at reduction.

        Absent fields use defaults; malformed containers raise InvalidGroupingError.
        """
        if not isinstance(data, dict):
            raise TypeError("GroupingProfile.from_dict requires a dict")
        policies = {
            "alpha_policy": AlphaPolicy.from_dict(data.get("alpha_policy")),
            "beta_policy": _beta_policy_from_payload(data),
            **restore_profile_calibrations(data),
        }
        return cls(
            name=str(data.get("name", "")),
            fingerprint=ProfileFingerprint.from_dict(data.get("fingerprint")),
            active=bool(data.get("active", True)),
            color=None if data.get("color") is None else str(data.get("color")),
            **_profile_structure(data),
            forward_group=_as_int(data.get("forward_group"), 1) or 1,
            backward_group=_as_int(data.get("backward_group"), 2) or 2,
            **policies,
            deadtime_policy=DeadtimePolicy.from_dict(data.get("deadtime_policy")),
            background_policy=BackgroundPolicy.from_dict(data.get("background_policy")),
            t0_policy=T0Policy.from_dict(data.get("t0_policy")),
            good_window_policy=GoodWindowPolicy.from_dict(data.get("good_window_policy")),
            binning=binning_spec(data),
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
    """Infer a manual offset from a validated legacy timing payload.

    File anchors use only the effective F/B membership. Per-detector overrides
    are validated by the profile boundary before they can imply an offset.
    """
    if "t0_policy" in payload:
        return T0Policy.from_dict(payload["t0_policy"])
    stored_t0 = _as_int(payload.get("t0_bin"))
    if stored_t0 is None:
        return T0Policy(mode="from_file")

    detector_t0 = payload.get("detector_t0_bins")
    if not isinstance(detector_t0, (list, tuple)) or not detector_t0:
        return T0Policy(mode="from_file")

    override = payload.get("effective_detector_t0_bins")
    if isinstance(override, (list, tuple)) and override:
        return T0Policy(mode="manual", offset_bins=int(override[0]) - int(detector_t0[0]))

    file_common_t0 = _payload_file_common_t0(payload, detector_t0)
    if file_common_t0 is None or stored_t0 == file_common_t0:
        return T0Policy(mode="from_file")
    return T0Policy(mode="manual", offset_bins=stored_t0 - file_common_t0)


def _payload_file_common_t0(
    payload: dict[str, Any], detector_t0: list[Any] | tuple[Any, ...]
) -> int | None:
    """The loader's common t0 from a stored payload: F/B-group max, exclusions applied.

    ``None`` when the payload's groups reference no detector the table covers,
    in which case there is nothing to compare the stored ``t0_bin`` against.
    """
    n_hist = len(detector_t0)
    forward_gid = _as_int(payload.get("forward_group"), 1) or 1
    backward_gid = _as_int(payload.get("backward_group"), 2) or 2
    try:
        indices = set(effective_group_indices(payload, forward_gid, n_histograms=n_hist))
        indices |= set(effective_group_indices(payload, backward_gid, n_histograms=n_hist))
    except InvalidGroupingError:
        return None
    values = [v for v in (_as_int(detector_t0[i]) for i in sorted(indices)) if v is not None]
    if not values:
        return None
    return max(0, max(values))


def _profile_structure(data: dict[str, Any]) -> dict[str, Any]:
    """Copy authored membership; malformed containers have a named boundary error."""
    try:
        groups = data.get("groups", {})
        excluded = data.get("excluded_detectors", [])
        if not isinstance(groups, dict) or not isinstance(excluded, (list, tuple)):
            raise ValueError("Group definitions must be a dictionary and exclusions a list.")
        if any(not isinstance(entries, (list, tuple)) for entries in groups.values()):
            raise ValueError("Group membership must be a list of detector IDs.")
        groups = {int(key): list(entries) for key, entries in groups.items()}
    except (TypeError, ValueError) as exc:
        raise InvalidGroupingError(str(exc), data) from exc
    names = data.get("group_names")
    included = data.get("included_groups")
    projections = data.get("projections")
    return {
        "groups": groups,
        "excluded_detectors": list(excluded),
        "group_names": {
            gid: str(v) for key, v in names.items() if (gid := _as_int(key)) is not None
        }
        if isinstance(names, dict)
        else {},
        "included_groups": {
            gid: bool(v) for key, v in included.items() if (gid := _as_int(key)) is not None
        }
        if isinstance(included, dict)
        else {},
        "projections": [dict(p) for p in projections if isinstance(p, dict)]
        if isinstance(projections, list)
        else [],
    }


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
    try:
        detector_t0_overrides(payload, fingerprint.histogram_count)
    except ValueError as exc:
        raise InvalidGroupingError(str(exc), payload) from exc

    extra: dict[str, Any] = {}
    for key in _STRUCTURE_KEYS:
        if payload.get(key):
            extra[key] = payload[key]
    for key in _ALPHA_EXTRA_KEYS:
        if key in payload:
            extra[key] = payload[key]

    policies = {
        "alpha_policy": _alpha_policy_from_payload(payload),
        "beta_policy": _beta_policy_from_payload(payload),
        **restore_profile_calibrations(payload),
    }
    return GroupingProfile(
        name=name,
        fingerprint=fingerprint,
        active=active,
        **_profile_structure(payload),
        forward_group=_as_int(payload.get("forward_group"), 1) or 1,
        backward_group=_as_int(payload.get("backward_group"), 2) or 2,
        **policies,
        deadtime_policy=_deadtime_policy_from_payload(payload),
        background_policy=_background_policy_from_payload(payload),
        t0_policy=_t0_policy_from_payload(payload),
        # D5: the good window is read only from an explicit policy dict. A
        # payload's absolute ``first_good_bin``/``last_good_bin`` are resolved
        # *values*, and inferring a policy from them would flip every freshly
        # loaded run into Manual (the t0 D1 lesson).
        good_window_policy=GoodWindowPolicy.from_dict(payload.get("good_window_policy")),
        binning=binning_spec(payload),
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


def _beta_policy_from_payload(payload: dict[str, Any]) -> BetaPolicy:
    """Derive a :class:`BetaPolicy` from a payload's beta + provenance keys.

    Mirrors :func:`_alpha_policy_from_payload`: a beta carrying a calibration
    method or reference run (``beta_method`` or ``beta_reference_run``)
    becomes ``calibrated``; a bare ``beta`` is a plain ``fixed`` value. Unlike
    alpha's allow-list, any non-empty method name counts — beta has only the
    two v1 estimator method ids (``count_fit``, ``single_histogram``) and no
    legacy method vocabulary to filter out.

    Also used to reconstruct ``beta_policy`` from a :class:`GroupingProfile`'s
    own serialized dict (:meth:`GroupingProfile.from_dict`) — the profile-dict
    shape is identical to the grouping-payload shape for these keys.
    """
    value = _sanitize_beta(payload.get("beta", 1.0))
    method = str(payload.get("beta_method", "")).strip()
    reference = _as_int(payload.get("beta_reference_run"))
    error = payload.get("beta_error")
    if reference is not None or method:
        return BetaPolicy(
            mode="calibrated",
            value=value,
            error=None if error is None else _as_float(error, 0.0),
            method=method,
            source_run=reference,
        )
    return BetaPolicy(mode="fixed", value=value)


def _deadtime_policy_from_payload(payload: dict[str, Any]) -> DeadtimePolicy:
    """Derive a :class:`DeadtimePolicy` from a payload's deadtime keys."""
    if not bool(payload.get("deadtime_correction", False)):
        return DeadtimePolicy(mode="off")
    mode = str(payload.get("deadtime_mode", payload.get("deadtime_method", "off"))).strip().lower()
    # The payload names file deadtime "file" (see _apply_deadtime_policy); the
    # policy names it "from_file". Translate before the membership check.
    mode = {"load": "manual", "file": "from_file"}.get(mode, mode)
    if mode not in DEADTIME_POLICY_MODES:
        mode = "off"
    if mode in ("off", "from_file"):
        return DeadtimePolicy(mode=mode)
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
    details = {key: payload[key] for key in _BACKGROUND_DETAIL_KEYS if key in payload}
    if not bool(payload.get("background_correction", False)):
        return BackgroundPolicy(mode="none", details=details)
    mode = str(payload.get("background_mode", "")).strip().lower()
    if mode not in BACKGROUND_POLICY_MODES or mode == "none":
        # Fall back to the same heuristic transform.background uses.
        has_fixed = any(
            isinstance(payload.get(key), (list, tuple))
            for key in ("background_fixed_values", "background_fix", "bkg_fix")
        )
        mode = "fixed" if has_fixed else "range"
    return BackgroundPolicy(mode=mode, details=details)


# --------------------------------------------------------------------------- #
# Resolution: profile + run -> full grouping payload
# --------------------------------------------------------------------------- #


def resolve_effective_grouping(
    profile: GroupingProfile,
    run: Run,
    *,
    reference_resolver: ReferenceResolver | None = None,
    timing_search: Callable | None = None,
) -> dict[str, Any]:
    """Merge *profile* with *run*'s file-derived facts into a grouping payload.

    The returned reduction payload combines shareable settings with this run's
    timing, window, exposure, detector tables and selected period. Computed
    policies use these effective inputs without modifying the acquired run.

    ``reference_resolver`` resolves the ``reference_run`` background mode's
    reference histograms + good-frame scale (the caller supplies one backed by a
    loaded-dataset registry). It is only consulted by a ``per_run_estimate`` alpha
    policy paired with ``reference_run`` background, so the estimate balances the
    same background-subtracted counts the reduction will; an unresolved
    requested reference makes the estimate unavailable.

    Authored membership is preserved; reduction rejects invalid or absent channels.
    """
    with perf_timer(
        "core.grouping.resolve_effective",
        t0_mode=profile.t0_policy.mode,
        alpha_mode=profile.alpha_policy.mode,
        deadtime_mode=profile.deadtime_policy.mode,
    ) as perf:
        run_grouping = run.grouping if isinstance(run.grouping, dict) else {}
        from asymmetry.core.io.periods import (
            acquired_period_histograms,
            select_period_histograms,
            selected_period_index,
        )

        period_index = selected_period_index(run_grouping, profile.period_mode)
        if period_index is not None:
            histograms, run_grouping = select_period_histograms(
                run.histograms,
                dict(run_grouping, deadtime_mode=profile.deadtime_policy.mode),
                period_index,
            )
            run = replace(
                run,
                histograms=histograms,
                grouping=run_grouping,
                metadata=dict(
                    run.metadata,
                    raw_period_histograms_available=acquired_period_histograms(
                        run.grouping, period_index
                    )
                    is not None,
                ),
            )
        if "period_reduced" in run_grouping and "period_histograms" not in run_grouping:
            run = replace(run, metadata=dict(run.metadata, raw_period_histograms_available=False))
        n_hist = len(run.histograms)

        grouping: dict[str, Any] = {
            "groups": {int(gid): list(dets) for gid, dets in profile.groups.items()},
            # Fixed alpha remains visible if membership prevents policy resolution.
            "alpha": profile.alpha_policy.value,
            "group_names": {int(gid): str(name) for gid, name in profile.group_names.items()},
            "forward_group": int(profile.forward_group),
            "backward_group": int(profile.backward_group),
            "excluded_detectors": list(profile.excluded_detectors),
            **_base_binning_payload(profile.binning),
            "good_window_policy": profile.good_window_policy.to_dict(),
        }
        if profile.included_groups:
            grouping["included_groups"] = {
                int(gid): bool(v) for gid, v in profile.included_groups.items()
            }
        if profile.projections:
            grouping["projections"] = [dict(p) for p in profile.projections]

        for key, value in profile.extra.items():
            grouping[key] = value

        resolve_projection_calibrations(profile.projection_alpha_policies, grouping)

        if profile.fingerprint.instrument:
            grouping["instrument"] = profile.fingerprint.instrument

        _copy_per_run_facts(grouping, run_grouping, run)
        if profile.period_mode is not None:
            grouping["period_mode"] = str(profile.period_mode)
        elif "period_mode" in run_grouping:
            grouping["period_mode"] = run_grouping["period_mode"]

        try:
            _apply_t0_policy(
                grouping,
                profile.t0_policy,
                run,
                n_hist,
                profile_name=profile.name,
                timing_search=timing_search,
            )
            if run.histograms and "period_inputs" not in grouping:
                first, last = resolve_run_good_window(profile.good_window_policy, run, grouping)
                grouping.update(
                    first_good_bin=first,
                    last_good_bin=last,
                    t_good_offset=first - int(grouping["t0_bin"]),
                )
            _apply_fixed_correction_policies(grouping, profile, run_grouping, n_hist)
            _apply_alpha_policy(
                grouping, profile.alpha_policy, run, n_hist, reference_resolver=reference_resolver
            )
        except (ValueError, CorrectionUnavailableError) as exc:
            # An unavailable request retains authored coefficients for explicit repair.
            _apply_fixed_correction_policies(grouping, profile, run_grouping, n_hist)
            if profile.alpha_policy.mode != "per_run_estimate":
                _apply_alpha_policy(grouping, profile.alpha_policy, run, n_hist)
            _apply_beta_policy(grouping, profile.beta_policy)
            if isinstance(exc, InvalidGroupingError):
                raise
            raise InvalidGroupingError(str(exc), grouping) from exc
        _apply_beta_policy(grouping, profile.beta_policy)

        perf.detail(n_detectors=n_hist, n_groups=len(grouping.get("groups", {})))
        return grouping


def _apply_fixed_correction_policies(grouping, profile, run_grouping, n_hist):
    """Carry fixed settings after timing resolution, including unavailable requests."""
    _apply_deadtime_policy(grouping, profile.deadtime_policy, run_grouping, n_hist)
    _apply_background_policy(grouping, profile.background_policy)


#: Per-run, file-derived keys copied straight through from the run's own
#: grouping (never stored on a profile).
_PER_RUN_FACT_KEYS = (
    "t0_bin",
    "t0_time_us",
    "t0_source",
    "t_good_offset",
    "first_good_bin",
    "last_good_bin",
    "bin_index_base",
    "detector_t0_bins",
    "detector_t0_sources",
    "detector_first_good_bins",
    "detector_last_good_bins",
    "histogram_labels",
    "root_histo_numbers",
    "good_frames",
    "reference_exposure",
    "period_reference_exposure",
    "period_histograms",
    "period_inputs",
    "period_reduced",
    "period_good_frames",
    "period_dead_time_us",
    "deadtime_file_us",
    "period_t0_sources",
    "period_detector_t0_sources",
    "period_mapping",
)


def _copy_per_run_facts(grouping: dict[str, Any], run_grouping: dict[str, Any], run: Run) -> None:
    """Copy file-derived per-run facts."""
    for key in _PER_RUN_FACT_KEYS:
        if key in run_grouping:
            grouping[key] = run_grouping[key]
    grouping["deadtime_file_us"] = list(file_deadtime_table(run_grouping))
    if "t0_bin" not in grouping and run.histograms:
        grouping["t0_bin"] = int(run.histograms[0].t0_bin)


def _analysis_group_indices(grouping: dict[str, Any], n_hist: int) -> list[int]:
    """0-based indices of the detectors the profile's forward+backward groups use."""
    forward_idx, backward_idx = independent_pair_indices(grouping, n_histograms=n_hist)
    return sorted(set(forward_idx) | set(backward_idx))


def _common_t0_over_groups(
    grouping: dict[str, Any],
    run: Run,
    n_hist: int,
    detector_t0_bins: list[int] | None,
) -> int:
    """Resolve the effective common anchor over the authored F/B channels."""
    indices = _analysis_group_indices(grouping, n_hist)
    if run.histograms and indices:
        return common_t0_for_groups(run.histograms, indices, detector_t0_bins=detector_t0_bins)
    stored = _as_int(grouping.get("t0_bin"))
    if stored is not None:
        return stored
    return int(run.histograms[0].t0_bin) if run.histograms else 0


def _file_common_t0(grouping: dict[str, Any], run: Run, n_hist: int) -> int:
    """Use acquired detector anchors as the baseline for a manual timing offset."""
    return _common_t0_over_groups(grouping, run, n_hist, None)


def _is_detector_table(raw: object, n_hist: int) -> bool:
    """True when *raw* is a per-detector table covering every histogram."""
    return isinstance(raw, (list, tuple)) and len(raw) == n_hist


def _has_good_window_tables(grouping: dict[str, Any], n_hist: int) -> bool:
    """Detector tables cover every acquired histogram; absent tables retain scalar limits."""
    return (
        _is_detector_table(grouping.get("detector_t0_bins"), n_hist)
        and _is_detector_table(grouping.get("detector_first_good_bins"), n_hist)
        and _is_detector_table(grouping.get("detector_last_good_bins"), n_hist)
    )


def aligned_n_bins(grouping: dict[str, Any], run: Run, n_hist: int, common_t0_bin: int) -> int:
    """Length of the shortest detector after its effective t0 shift, in bins."""
    detector_t0 = effective_detector_t0_bins(run.histograms, grouping)
    indices = _analysis_group_indices(grouping, n_hist)
    if not indices:
        return len(run.histograms[0].counts)
    return min(len(run.histograms[i].counts) + common_t0_bin - detector_t0[i] for i in indices)


def run_file_good_window(
    run: Run,
    grouping: dict[str, Any],
    *,
    common_t0_bin: int,
    alignment: AlignmentCrop | None = None,
) -> tuple[int, int]:
    """Map acquired file bounds through the current selected-channel alignment.

    Per-detector tables are acquisition facts. Scalar legacy bounds remain the
    recorded common interval; they do not establish original detector facts.
    """
    if not _has_good_window_tables(grouping, len(run.histograms)):
        return grouping.get("first_good_bin", 0), grouping.get(
            "last_good_bin", len(run.histograms[0].counts) - 1
        )
    indices = _analysis_group_indices(grouping, len(run.histograms))
    bins = effective_detector_t0_bins(run.histograms, grouping)
    if alignment is None:
        alignment = resolve_alignment(
            run.histograms, indices, detector_t0_bins=bins, common_t0_bin=common_t0_bin
        )
    first, last = alignment.first_bin, alignment.first_bin + alignment.length - 1
    if _has_good_window_tables(grouping, len(run.histograms)):
        first = max(
            first,
            max(
                int(grouping["detector_first_good_bins"][i]) + common_t0_bin - bins[i]
                for i in indices
            ),
        )
        last = min(
            last,
            min(
                int(grouping["detector_last_good_bins"][i]) + common_t0_bin - bins[i]
                for i in indices
            ),
        )
    return first, last


def resolve_good_window(
    policy: GoodWindowPolicy,
    *,
    t0_bin: int,
    file_first_good: int,
    file_last_good: int,
    n_bins: int,
    first_bin: int = 0,
) -> tuple[int, int]:
    """Resolve signed authored offsets without altering an explicit endpoint.

    Omitted endpoints retain mapped file limits. Automatic limits intersect
    observed support; authored limits must name actual observations.
    """
    first_good, last_good = file_first_good, file_last_good
    if policy.mode == "manual":
        for name in ("first_offset_bins", "last_offset_bins"):
            value = (
                policy.first_offset_bins if name == "first_offset_bins" else policy.last_offset_bins
            )
            if value is not None and (isinstance(value, bool) or not isinstance(value, Integral)):
                raise ValueError(f"Good window {name} must be a signed integer; got {value!r}.")

        if policy.first_offset_bins is not None:
            first_good = int(t0_bin) + int(policy.first_offset_bins)
        if policy.last_offset_bins is not None:
            last_good = int(t0_bin) + int(policy.last_offset_bins)
    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in (first_good, last_good)
    ):
        raise ValueError("Good window endpoints must be integer bin indices.")
    if policy.mode == "from_file":
        first_good = max(first_bin, first_good)
        last_good = min(n_bins - 1, last_good)
    if first_good > last_good:
        raise ValueError(f"Good window [{first_good}, {last_good}] is reversed or empty.")
    if first_good < first_bin or last_good >= n_bins:
        raise ValueError(
            f"Requested good window [{first_good}, {last_good}] is outside observed "
            f"support [{first_bin}, {n_bins - 1}]."
        )
    return first_good, last_good


def _apply_t0_policy(
    grouping: dict[str, Any],
    policy: T0Policy,
    run: Run,
    n_hist: int,
    *,
    profile_name: str = "",
    timing_search: Callable | None = None,
) -> None:
    """Resolve effective timing while retaining acquired histograms.

    Detection keeps detector stagger; manual offsets move anchors by whole bins.
    Exact timing moves with those anchors while raw window facts stay fixed."""
    if not run.histograms:
        return
    if policy.legacy_value is not None:
        raise ValueError(
            f"Grouping profile {profile_name!r} still carries a pre-v21 absolute manual t0; "
            "convert it with T0Policy.from_legacy_value() on project open."
        )
    if "period_inputs" in grouping:
        grouping["t0_policy"] = policy.to_dict()
        return
    resolve_selected_timing(
        run.histograms,
        grouping,
        run.metadata,
        mode=policy.mode,
        offset_bins=policy.offset_bins or 0,
        search_provider=timing_search or find_t0_for_run,
    )


def resolve_run_good_window(
    policy: GoodWindowPolicy,
    run: Run,
    grouping: dict[str, Any],
    *,
    alignment: AlignmentCrop | None = None,
) -> tuple[int, int]:
    """Resolve file limits and authored offsets on current observed aligned support."""
    if alignment is None:
        alignment = resolve_alignment(
            run.histograms,
            _analysis_group_indices(grouping, len(run.histograms)),
            detector_t0_bins=effective_detector_t0_bins(run.histograms, grouping),
        )
    t0_bin = alignment.common_t0_bin + alignment.first_bin
    if (
        policy.mode == "manual"
        and policy.first_offset_bins is not None
        and policy.last_offset_bins is not None
    ):
        file_first_good, file_last_good = (
            alignment.first_bin,
            alignment.first_bin + alignment.length - 1,
        )
    else:
        file_first_good, file_last_good = run_file_good_window(
            run, grouping, common_t0_bin=t0_bin, alignment=alignment
        )
    return resolve_good_window(
        policy,
        t0_bin=t0_bin,
        file_first_good=file_first_good,
        file_last_good=file_last_good,
        n_bins=alignment.first_bin + alignment.length,
        first_bin=alignment.first_bin,
    )


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


def _apply_beta_policy(grouping: dict[str, Any], policy: BetaPolicy) -> None:
    """Write dimensionless beta and retain calibrated provenance even at unity.

    Fixed unity stays implicit. Vector reduction callers omit scalar beta.
    """
    beta = _sanitize_beta(policy.value)
    if beta != 1.0:
        grouping["beta"] = beta
    if policy.mode == "calibrated":
        if policy.method:
            grouping["beta_method"] = str(policy.method)
        if policy.error is not None:
            grouping["beta_error"] = float(policy.error)
        if policy.source_run is not None:
            grouping["beta_reference_run"] = int(policy.source_run)


def _estimate_run_alpha(
    grouping: dict[str, Any],
    run: Run,
    n_hist: int,
    *,
    reference_resolver: ReferenceResolver | None = None,
) -> float:
    """Estimate native-window alpha from corrected contributing-period counts."""
    from asymmetry.core.transform.corrections import reference_observation_metadata

    forward_gid = int(grouping.get("forward_group", 1))
    backward_gid = int(grouping.get("backward_group", 2))
    forward_idx = effective_group_indices(grouping, forward_gid, n_histograms=n_hist)
    backward_idx = effective_group_indices(grouping, backward_gid, n_histograms=n_hist)
    flags = correction_flags_from_grouping(grouping)
    periods = corrected_period_counts(
        run.histograms,
        grouping,
        str(grouping.get("period_mode", "red")),
        binning=BinningSpec(),
        forward_idx=forward_idx,
        backward_idx=backward_idx,
        use_deadtime=flags.use_deadtime,
        deadtime_mode=flags.deadtime_mode,
        use_background=flags.use_background,
        metadata=reference_observation_metadata(run),
        reference_resolver=reference_resolver,
    )
    corrected = next(iter(periods.values()))
    return estimate_alpha(
        corrected.forward.counts,
        corrected.backward.counts,
        first_good_bin=corrected.window.first,
        last_good_bin=corrected.window.last,
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
        grouping["dead_time_us"] = list(file_deadtime_table(run_grouping))
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
            grouping["dead_time_us"] = [float(tau)] * n_hist
        elif policy.values:
            grouping["dead_time_us"] = [float(v) for v in policy.values]
        if policy.source_run is not None:
            grouping["deadtime_reference_run"] = int(policy.source_run)
        return


def _apply_background_policy(grouping: dict[str, Any], policy: BackgroundPolicy) -> None:
    """Write background keys into the grouping per the policy."""
    grouping["background_correction"] = policy.mode != "none"
    grouping["background_mode"] = policy.mode
    for key in _BACKGROUND_DETAIL_KEYS:
        grouping.pop(key, None)
    grouping.update(policy.details)


# --------------------------------------------------------------------------- #
# Project-open repair of stored t0 policies (D2, D3)
# --------------------------------------------------------------------------- #


def _profile_file_common_t0(profile: GroupingProfile, run: Run) -> int:
    """*run*'s file-derived common t0 over *profile*'s analysis groups."""
    grouping = {
        "groups": {int(gid): list(dets) for gid, dets in profile.groups.items()},
        "forward_group": int(profile.forward_group),
        "backward_group": int(profile.backward_group),
        "excluded_detectors": list(profile.excluded_detectors),
    }
    return _file_common_t0(grouping, run, len(run.histograms))


def _t0_reference_run(profile: GroupingProfile, runs_by_number: dict[int, Run]) -> Run | None:
    """The run a stored absolute t0 was typed against, if the project has one.

    The policy's ``source_run`` when that run is loaded, else the first loaded
    run of the profile's own fingerprint — the runs the policy governs all share
    it, so any of them measures the same file t0 the editor showed.
    """
    stored = runs_by_number.get(profile.t0_policy.source_run)
    if stored is not None:
        return stored
    for _number, candidate in sorted(runs_by_number.items()):
        if profile.fingerprint.matches(profile_fingerprint_for_run(candidate)):
            return candidate
    return None


def heal_t0_policies(profiles: list[GroupingProfile], runs_by_number: dict[int, Run]) -> list[str]:
    """Repair stored t0 policies in place at project open (D2, D3).

    Two repairs that can only be made once the runs are known:

    * a pre-v21 absolute manual ``value`` becomes the equivalent signed *offset*
      from the reference run's own file t0 (D3). :func:`_apply_t0_policy` raises
      on an unconverted value, so this is what stands between a v20 project and
      a hard failure. A profile none of whose runs are loaded is left alone —
      nothing resolves it, and the stored value survives to be converted the day
      one of its runs is.
    * a manual policy resolving to a zero shift on every run in scope — which,
      for an offset policy, is exactly ``offset_bins == 0`` — was never a shift
      at all (the pre-D1 payload comparison mislabelled it), so it becomes
      ``from_file`` (D2).

    Returns one line per repaired profile for the caller to log. The project
    file is not rewritten until the user saves.
    """
    messages: list[str] = []
    for profile in profiles:
        legacy_value = profile.t0_policy.legacy_value
        if legacy_value is not None:
            run = _t0_reference_run(profile, runs_by_number)
            if run is None or not run.histograms:
                continue
            try:
                file_common_t0 = _profile_file_common_t0(profile, run)
            except InvalidGroupingError as exc:
                messages.append(
                    f"Grouping profile {profile.name!r}: manual t0 remains unresolved: {exc}"
                )
                continue
            profile.t0_policy = T0Policy.from_legacy_value(legacy_value, file_common_t0)
            healed = profile.t0_policy
            became = (
                f"offset {healed.offset_bins:+d} bins"
                if healed.mode == "manual"
                else "mode from_file"
            )
            messages.append(
                f"Grouping profile {profile.name!r}: converted the stored manual t0 "
                f"bin {legacy_value} to {became} against run {run.run_number}'s "
                f"file t0 (bin {file_common_t0})."
            )
        policy = profile.t0_policy
        if policy.mode == "manual" and policy.offset_bins == 0:
            profile.t0_policy = T0Policy(mode="from_file")
            messages.append(
                f"Grouping profile {profile.name!r}: manual t0 offset is 0 on every "
                "run in scope, healed to mode from_file."
            )
    return messages


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


def effective_grouping_for_loaded_run(
    profiles: list[GroupingProfile],
    run: Run,
    *,
    reference_resolver: ReferenceResolver | None = None,
) -> dict[str, Any]:
    """Resolve the default profile, or copy the loader grouping if none matches.

    ``reference_resolver`` supplies background counts and exposure for automatic alpha.
    """
    profile = default_profile_for_run(profiles, run)
    if profile is not None:
        return resolve_effective_grouping(profile, run, reference_resolver=reference_resolver)
    return dict(run.grouping) if isinstance(run.grouping, dict) else {}


__all__ = [
    "ALPHA_POLICY_MODES",
    "BETA_POLICY_MODES",
    "DEADTIME_POLICY_MODES",
    "BACKGROUND_POLICY_MODES",
    "T0_POLICY_MODES",
    "GOOD_WINDOW_POLICY_MODES",
    "AlphaPolicy",
    "BetaPolicy",
    "DeadtimePolicy",
    "BackgroundPolicy",
    "T0Policy",
    "GoodWindowPolicy",
    "ProfileFingerprint",
    "GroupingProfile",
    "profile_fingerprint_for_run",
    "fingerprint_from_payload",
    "detect_instrument_for_run",
    "reconcile_instrument_for_payload",
    "profile_from_payload",
    "resolve_effective_grouping",
    "aligned_n_bins",
    "resolve_good_window",
    "run_file_good_window",
    "default_profile_for_run",
    "named_profile_for_run",
    "assigned_profile_for_run",
    "effective_grouping_for_loaded_run",
]

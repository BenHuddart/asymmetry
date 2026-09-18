"""The persisted *fit series* object underpinning batch & global fits.

A :class:`FitSeries` is an ordered collection of **members** that share one
canonical model for a single representation type.  A member is either a *run*
(``member_kind == "runs"``) or a synthetic *detector-group* key
(``member_kind == "groups"``; see
:func:`asymmetry.core.fitting.grouped_time_domain._group_dataset_run_number`).

The *relationship* between members follows the existing ``global`` / ``local`` /
``fixed`` parameter classifier:

* every (physics) parameter ``local``/``fixed`` → a *batch* fit (N independent
  fits, one per member);
* one or more parameter ``global``              → a *global* fit (shared across
  members).

So a global fit is derived from the classifier, not a separate object.  For
group series the per-group **nuisance** block (:attr:`FitSeries.nuisance_params`)
is always estimated separately per member and is therefore excluded from
:attr:`param_roles`.  Parameter trending reads :attr:`FitSeries.results_by_run`.
"""

from __future__ import annotations

import json
from typing import Any

from asymmetry.core.data.dataset import Run
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.representation.base import RepresentationType

# ORDER_KEYS is defined in asymmetry.core.utils.constants and re-exported here
# (and via representation/__init__) so series and field scans share one tuple.
from asymmetry.core.utils.constants import ORDER_KEYS

#: Allowed per-parameter classification roles.
PARAM_ROLES = ("global", "local", "fixed")

#: Allowed member kinds.
MEMBER_KINDS = ("runs", "groups")

#: Default co-add block of a recipe (D2): co-adding off, window of 2.
DEFAULT_COADD: dict[str, Any] = {"mode": "off", "window": 2}

#: Default seeding mode of a recipe (D2) — the Batch tab's ``_batch_seeding_mode``.
DEFAULT_SEEDING = "auto"


def default_recipe() -> dict[str, Any]:
    """Return an empty :attr:`FitSeries.recipe` (D2).

    A series that has never been run — or one loaded from a pre-v20 project the
    migration could not seed — carries this shape: no parameter rows, an
    unbounded fit range ("as fitted, unknown"), automatic seeding and co-adding
    off.
    """
    return {
        "parameters": [],
        "fit_range": {"min": None, "max": None},
        "seeding": DEFAULT_SEEDING,
        "coadd": dict(DEFAULT_COADD),
    }


def _recipe_parameter_row(entry: dict) -> dict[str, Any]:
    """Normalise one Batch-tab parameter row into its recipe form."""
    return {
        "name": str(entry.get("name", "")),
        "value": float(entry.get("value", 0.0)),
        "type": str(entry.get("type", "")),
        "bounds": str(entry.get("bounds", "")),
        "seeded": bool(entry.get("seeded", False)),
    }


def _optional_float(value: object) -> float | None:
    """Return *value* as a float, or ``None`` when it is absent."""
    return None if value is None else float(value)


def normalise_recipe(recipe: dict | None) -> dict[str, Any]:
    """Return *recipe* in the canonical D2 shape, filling absent blocks.

    The single normalising boundary for a recipe: every :class:`FitSeries`
    holds this exact shape, so :meth:`FitSeries.recipe_identity` can compare
    two recipes as strings without re-deriving defaults, and the Batch tab can
    read every key without asking whether it is present.
    """
    source = recipe or {}
    fit_range = source.get("fit_range") or {}
    coadd = source.get("coadd") or {}
    return {
        "parameters": [_recipe_parameter_row(entry) for entry in source.get("parameters") or []],
        "fit_range": {
            "min": _optional_float(fit_range.get("min")),
            "max": _optional_float(fit_range.get("max")),
        },
        "seeding": str(source.get("seeding") or DEFAULT_SEEDING),
        "coadd": {
            "mode": str(coadd.get("mode") or DEFAULT_COADD["mode"]),
            "window": int(coadd.get("window") or DEFAULT_COADD["window"]),
        },
    }


class FitSeries:
    """An ordered series of members fit with one canonical model.

    Members are keyed by integer in :attr:`member_run_numbers`.  For
    ``member_kind == "runs"`` the key *is* the run number; for
    ``member_kind == "groups"`` it is the synthetic negative group key, and
    :attr:`member_source_run` maps each key back to its physical source run
    (used for field/temperature ordering and metadata).
    """

    def __init__(
        self,
        batch_id: str,
        rep_type: RepresentationType | str,
        *,
        label: str | None = None,
        member_kind: str = "runs",
        member_run_numbers: list[int] | None = None,
        member_source_run: dict[int, int] | None = None,
        order_key: str = "run",
        canonical_model: dict | None = None,
        param_roles: dict[str, str] | None = None,
        nuisance_params: list[str] | None = None,
        results_by_run: dict[int, dict] | None = None,
        extra: dict | None = None,
        source_group_id: str | None = None,
        group_id: str | None = None,
        excluded_run_numbers: list[int] | None = None,
        last_fitted_members: list[int] | None = None,
        recipe: dict | None = None,
        trend_excluded_runs: list[int] | None = None,
        joint_fit_id: str | None = None,
        shared_params: dict[str, str] | None = None,
    ) -> None:
        self.batch_id = str(batch_id)
        self.label: str | None = str(label).strip() or None if label else None
        self.rep_type = (
            rep_type
            if isinstance(rep_type, RepresentationType)
            else RepresentationType(str(rep_type))
        )
        self.member_kind = member_kind if member_kind in MEMBER_KINDS else "runs"
        self.member_run_numbers: list[int] = [int(r) for r in (member_run_numbers or [])]
        self.member_source_run: dict[int, int] = {
            int(key): int(src) for key, src in (member_source_run or {}).items()
        }
        self.order_key = order_key if order_key in ORDER_KEYS else "run"
        self.canonical_model: dict | None = (
            dict(canonical_model) if isinstance(canonical_model, dict) else None
        )
        self.param_roles: dict[str, str] = {
            str(name): role for name, role in (param_roles or {}).items() if role in PARAM_ROLES
        }
        #: Per-member nuisance block (group fits): always local, never trended
        #: as a shared series parameter, so excluded from :attr:`param_roles`.
        self.nuisance_params: list[str] = [str(name) for name in (nuisance_params or [])]
        self.results_by_run: dict[int, dict] = {
            int(run): dict(result) for run, result in (results_by_run or {}).items()
        }
        #: The Batch tab's setup for this series (D2): ``parameters`` (the table
        #: rows), ``fit_range``, ``seeding`` and ``coadd``. Always in the
        #: canonical shape :func:`normalise_recipe` produces, so every reader
        #: finds every key. Re-running a series with an identical recipe and
        #: member set replaces its results in place (D3); anything else records
        #: a new series.
        self.recipe: dict[str, Any] = normalise_recipe(recipe)
        #: Members the user has dropped from *trending* without removing them
        #: from the series (D4). Series-level, so a run trended in one series
        #: and excluded from another keeps both answers. Sorted, de-duplicated,
        #: and expressed in member keys (synthetic group keys for a group
        #: series), mirroring :attr:`member_run_numbers`.
        self.trend_excluded_runs: list[int] = sorted({int(r) for r in (trend_excluded_runs or [])})
        #: Freeform JSON-able state attached to this series (e.g. the ALC scan's
        #: baseline regions / peaks / view options). Empty for ordinary fits.
        self.extra: dict = dict(extra) if isinstance(extra, dict) else {}
        #: Legacy provenance (D1, Option B): the id of the DataGroup this series
        #: was launched from, when every member shared exactly one group at
        #: record time. Retained for backward compatibility (older GUI code and
        #: pre-v15 saves read it); the structural ownership link is now
        #: :attr:`group_id`. For a frozen/legacy series it may be the only
        #: pointer back to a (possibly deleted) group.
        self.source_group_id: str | None = str(source_group_id) if source_group_id else None
        #: Structural ownership link (D1/D7): the id of the DataGroup that *owns*
        #: this run-membered series. Unlike ``source_group_id`` this is identity,
        #: not provenance — it drives live membership derivation
        #: (:meth:`effective_members`). ``None`` for a *frozen* series (a
        #: legacy/orphaned analysis with snapshot membership) and for
        #: detector-group series (``member_kind == "groups"``, D8).
        self.group_id: str | None = str(group_id) if group_id else None
        #: Per-series exclusions (D1): run numbers the user has dropped from this
        #: analysis without removing them from the owning group. Effective
        #: membership is ``group.member_run_numbers − excluded_run_numbers``.
        #: Sorted and de-duplicated; only meaningful for a group-bound run series.
        self.excluded_run_numbers: list[int] = sorted(
            {int(r) for r in (excluded_run_numbers or [])}
        )
        #: Snapshot of the members that were actually fit on the last run (D1).
        #: Results remain a snapshot; when the live effective membership diverges
        #: from this list the series is *stale* (:meth:`is_stale`). Empty for a
        #: freshly-created series that has not been fit yet; the v14→v15 migration
        #: seeds it from ``member_run_numbers`` so a loaded series is not stale.
        self.last_fitted_members: list[int] = [int(r) for r in (last_fitted_members or [])]
        #: Joint-fit stamps (docs/plans/joint-fit.md D5/D8): written by a joint
        #: run to say this series currently honours a shared-parameter
        #: constraint, and cleared by a solo run of this series (which detaches
        #: it — the joint fit's own record goes stale, per D9) or by deleting
        #: the joint fit (D10). Never edited directly by a caller — go through
        #: :meth:`clear_joint_stamp`, so the two fields can never fall out of
        #: sync with each other. ``joint_fit_id`` names the owning
        #: :class:`~asymmetry.core.representation.joint_fit.JointFit`;
        #: ``shared_params`` maps this series' *own* parameter name to the
        #: shared column's name, for a reader rendering one series at a time
        #: (chips, cards, results windows) that has no reason to load the
        #: joint fit record just to label a parameter "Shared".
        self.joint_fit_id: str | None = str(joint_fit_id) if joint_fit_id else None
        self.shared_params: dict[str, str] = {
            str(name): str(shared_name) for name, shared_name in (shared_params or {}).items()
        }

    # ── joint fit stamp (D5/D8/D9/D10) ──────────────────────────────────────

    def clear_joint_stamp(self) -> None:
        """Clear this series' joint-fit stamp.

        The only sanctioned way to un-stamp a series (a solo re-run detaching
        it, or the owning joint fit being deleted) — going through one method
        means ``joint_fit_id`` and ``shared_params`` can never be cleared one
        without the other, which a direct assignment at each call site would
        risk.
        """
        self.joint_fit_id = None
        self.shared_params = {}

    # ── label ──────────────────────────────────────────────────────────────

    def display_name(self, fallback: str) -> str:
        """Return the user-assigned label, or *fallback* when none is set."""
        return self.label or fallback

    @property
    def is_computed(self) -> bool:
        """True for a model-less *computed* series (e.g. an integral/field scan).

        A computed series carries per-run results directly in
        :attr:`results_by_run` but owns **no** fit model, so it has no recipe
        worth re-running and no model to render in its label. A real
        batch/global fit always has a canonical model.
        """
        return self.canonical_model is None

    # ── classifier-derived scope ───────────────────────────────────────────

    def is_global(self) -> bool:
        """Return ``True`` when at least one parameter is classified ``global``."""
        return any(role == "global" for role in self.param_roles.values())

    def params_with_role(self, role: str) -> list[str]:
        """Return the parameter names classified as *role*, in insertion order."""
        return [name for name, value in self.param_roles.items() if value == role]

    def global_params(self) -> list[str]:
        return self.params_with_role("global")

    def local_params(self) -> list[str]:
        return self.params_with_role("local")

    def fixed_params(self) -> list[str]:
        return self.params_with_role("fixed")

    def shared_parameters(self) -> dict[str, dict[str, float]]:
        """Fitted shared (``global``-role) parameters as ``{name: {"value", "error"}}``.

        A global fit shares one value across every member, so each global parameter's
        value (and uncertainty, when present) is taken from the first successful
        member's recorded result. Returns an empty mapping when the series has no
        ``global``-role parameters. This is the model-side source for the trend
        panel's "Global fitting parameters" header, so the GUI need not re-derive it
        from the displayed rows.
        """
        names = self.global_params()
        if not names:
            return {}
        shared: dict[str, dict[str, float]] = {}
        for member in self.member_run_numbers:
            summary = self.results_by_run.get(member)
            if not summary or not summary.get("success"):
                continue
            params = summary.get("parameters") or {}
            errors = summary.get("uncertainties") or {}
            for name in names:
                if name in shared or name not in params:
                    continue
                entry: dict[str, float] = {"value": float(params[name])}
                error = errors.get(name)
                if error is not None:
                    entry["error"] = float(error)
                shared[name] = entry
            if len(shared) == len(names):
                break
        return shared

    # ── group-bound membership (D1) ──────────────────────────────────────────

    def effective_members(self, group: object) -> list[int]:
        """Return the live membership of a group-bound run series, in group order.

        For a run-membered series with a :attr:`group_id`, this is the owning
        *group*'s ``member_run_numbers`` minus :attr:`excluded_run_numbers`,
        preserving the group's ordering. For a **frozen** series
        (``group_id is None``), a detector-group series
        (``member_kind == "groups"``, D8), or when *group* is ``None``, the
        series' own :attr:`member_run_numbers` are returned unchanged (frozen
        semantics — the snapshot is authoritative).
        """
        group_members = getattr(group, "member_run_numbers", None)
        if (
            self.group_id is None
            or self.member_kind != "runs"
            or group is None
            or group_members is None
        ):
            return list(self.member_run_numbers)
        excluded = set(self.excluded_run_numbers)
        return [int(r) for r in group_members if int(r) not in excluded]

    def is_stale(self, group: object) -> bool:
        """Return ``True`` when the live membership no longer matches what was fit.

        Only group-bound run series can be stale: the comparison is between
        :meth:`effective_members` and :attr:`last_fitted_members`, done as an
        order-insensitive *set* compare (member ordering is resolved at fit
        time via ``order_key``, so a re-order alone is not staleness). Frozen
        series (``group_id is None``) and detector-group series
        (``member_kind == "groups"``, D8) are **never** stale.
        """
        if self.group_id is None or self.member_kind != "runs" or group is None:
            return False
        return set(self.effective_members(group)) != set(self.last_fitted_members)

    # ── membership ─────────────────────────────────────────────────────────

    def source_run_for(self, member_key: int) -> int:
        """Return the physical source run for *member_key*.

        For run series the key is the run number; for group series it is mapped
        through :attr:`member_source_run`, falling back to decoding the synthetic
        key (``|key| // 1000``) when the map is incomplete.
        """
        member_key = int(member_key)
        if self.member_kind != "groups":
            return member_key
        mapped = self.member_source_run.get(member_key)
        if mapped is not None:
            return mapped
        return abs(member_key) // 1000

    def source_runs(self) -> list[int]:
        """Return the ordered, de-duplicated physical runs backing the members.

        For run series this is the member run numbers; for group series each
        synthetic member key is resolved through :meth:`source_run_for` (so the
        map's synthetic-key fallback applies uniformly). This is the single
        source of truth for "which runs does this series cover" — identity
        signatures, default labels and browser highlights all read it, so they
        cannot disagree on a series whose ``member_source_run`` map is partial.
        """
        return sorted({self.source_run_for(key) for key in self.member_run_numbers})

    def add_member(self, run_number: int, *, source_run: int | None = None) -> None:
        """Add *run_number* (member key) to the series (idempotent).

        For group series, pass *source_run* to record the physical run the
        synthetic member key belongs to.
        """
        run_number = int(run_number)
        if run_number not in self.member_run_numbers:
            self.member_run_numbers.append(run_number)
        if source_run is not None:
            self.member_source_run[run_number] = int(source_run)

    def remove_member(self, run_number: int) -> None:
        """Remove *run_number* from the series and drop its derived state."""
        run_number = int(run_number)
        self.member_run_numbers = [r for r in self.member_run_numbers if r != run_number]
        self.member_source_run.pop(run_number, None)
        self.results_by_run.pop(run_number, None)
        self.trend_excluded_runs = [r for r in self.trend_excluded_runs if r != run_number]

    def sort_members(self, runs_by_number: dict[int, Run]) -> None:
        """Order members by :attr:`order_key` using the supplied runs.

        Group members are ordered by their source run's key value, keeping the
        groups of one run adjacent (tie-broken by member key).
        """

        def key(member_key: int) -> tuple[float, int, int]:
            source = self.source_run_for(member_key)
            # Tie-break by |key|: for group members this is run*1000+index, so
            # a run's detector groups stay in ascending group-index order.
            tiebreak = abs(member_key)
            run = runs_by_number.get(source)
            if run is None or self.order_key == "run":
                return (float(source), source, tiebreak)
            value = run.field if self.order_key == "field" else run.temperature
            return (float(value), source, tiebreak)

        self.member_run_numbers.sort(key=key)

    # ── identity (D3) ──────────────────────────────────────────────────────

    def recipe_identity(self) -> str:
        """Return this series' canonical identity string.

        Two series describe **the same analysis exactly when their identity
        strings are equal** — that equivalence is the whole contract, and the
        string itself is an opaque token (never parsed, never shown). On a run,
        the Batch tab compares the draft's identity with the open series' (D3):
        identical replaces the results in place under the same ``batch_id`` and
        label; anything else records a new series.

        Included: the representation, member kind, the normalised
        :attr:`canonical_model`, :attr:`param_roles`, :attr:`order_key`,
        :attr:`excluded_run_numbers`, the effective member set (the sorted
        :attr:`last_fitted_members`, so member *ordering* is not identity), and
        the whole :attr:`recipe` (parameter rows, fit range, seeding, co-add).

        Excluded: :attr:`label` (renaming never changes identity),
        :attr:`batch_id`, :attr:`group_id`, every recorded result, and the
        joint-fit stamp (:attr:`joint_fit_id`/:attr:`shared_params`) — they say
        which series this is, how it went, or which constraint it currently
        honours, not what analysis it describes. A series stamped, detached,
        and stamped again by a different joint fit is still the same analysis.
        """
        model = self.canonical_model
        if model is not None:
            try:
                model = CompositeModel.from_dict(model).to_dict()
            except (ValueError, KeyError, TypeError):
                model = self.canonical_model
        return json.dumps(
            {
                "rep_type": self.rep_type.value,
                "member_kind": self.member_kind,
                "members": sorted(int(r) for r in self.last_fitted_members),
                "model": model,
                "param_roles": dict(self.param_roles),
                "order_key": self.order_key,
                "excluded_run_numbers": list(self.excluded_run_numbers),
                "recipe": self.recipe,
            },
            sort_keys=True,
            default=str,
        )

    # ── trending ───────────────────────────────────────────────────────────

    def trend_member_run_numbers(self) -> list[int]:
        """Return ordered members eligible for trending (D4).

        Every member except those in :attr:`trend_excluded_runs`.
        """
        excluded = set(self.trend_excluded_runs)
        return [r for r in self.member_run_numbers if r not in excluded]

    # ── persistence ────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "label": self.label,
            "rep_type": self.rep_type.value,
            "member_kind": self.member_kind,
            "member_run_numbers": list(self.member_run_numbers),
            "member_source_run": {
                str(key): int(src) for key, src in self.member_source_run.items()
            },
            "order_key": self.order_key,
            "canonical_model": None if self.canonical_model is None else dict(self.canonical_model),
            "param_roles": dict(self.param_roles),
            "nuisance_params": list(self.nuisance_params),
            "results_by_run": {str(run): dict(res) for run, res in self.results_by_run.items()},
            "extra": dict(self.extra),
            "source_group_id": self.source_group_id,
            "group_id": self.group_id,
            "excluded_run_numbers": list(self.excluded_run_numbers),
            "last_fitted_members": list(self.last_fitted_members),
            "recipe": normalise_recipe(self.recipe),
            "trend_excluded_runs": list(self.trend_excluded_runs),
            "joint_fit_id": self.joint_fit_id,
            "shared_params": dict(self.shared_params),
        }

    @classmethod
    def from_dict(cls, data: dict) -> FitSeries:
        results_raw = data.get("results_by_run")
        results = (
            {int(run): dict(res) for run, res in results_raw.items() if isinstance(res, dict)}
            if isinstance(results_raw, dict)
            else {}
        )
        source_raw = data.get("member_source_run")
        member_source_run = (
            {int(key): int(src) for key, src in source_raw.items()}
            if isinstance(source_raw, dict)
            else None
        )
        return cls(
            batch_id=str(data["batch_id"]),
            label=data.get("label"),
            rep_type=data["rep_type"],
            member_kind=str(data.get("member_kind", "runs")),
            member_run_numbers=data.get("member_run_numbers"),
            member_source_run=member_source_run,
            order_key=str(data.get("order_key", "run")),
            canonical_model=data.get("canonical_model"),
            param_roles=data.get("param_roles"),
            nuisance_params=data.get("nuisance_params"),
            results_by_run=results,
            extra=data.get("extra"),
            source_group_id=data.get("source_group_id"),
            # Tolerant reads: pre-v15 saves lack these. ``group_id`` stays absent
            # until the schema migration resolves it from ``source_group_id``
            # (this layer cannot see the group registry); ``excluded_run_numbers``
            # defaults empty and ``last_fitted_members`` defaults empty (the
            # migration seeds it from ``member_run_numbers``).
            group_id=data.get("group_id"),
            excluded_run_numbers=data.get("excluded_run_numbers"),
            last_fitted_members=data.get("last_fitted_members"),
            # An absent ``recipe`` (a series the v19->v20 migration could not
            # seed) becomes the empty default rather than a missing key.
            recipe=data.get("recipe"),
            trend_excluded_runs=data.get("trend_excluded_runs"),
            # Absent on every pre-v22 save (the joint fit feature did not
            # exist yet): a series with no stamp is simply not a joint fit
            # member, which is exactly what the defaults say.
            joint_fit_id=data.get("joint_fit_id"),
            shared_params=data.get("shared_params"),
        )

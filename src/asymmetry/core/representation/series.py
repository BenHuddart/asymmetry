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


def canonical_model_matches(model_a: dict | None, model_b: dict | None) -> bool:
    """Return ``True`` when two serialised models are structurally identical.

    Comparison is done on the normalised ``CompositeModel.to_dict`` form so
    semantically-equal models do not falsely register as diverged.
    """
    if model_a is None or model_b is None:
        return model_a is None and model_b is None
    try:
        norm_a = CompositeModel.from_dict(model_a).to_dict()
        norm_b = CompositeModel.from_dict(model_b).to_dict()
    except (ValueError, KeyError, TypeError):
        return model_a == model_b
    return norm_a == norm_b


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
        diverged_runs: set[int] | list[int] | None = None,
        extra: dict | None = None,
        source_group_id: str | None = None,
        group_id: str | None = None,
        excluded_run_numbers: list[int] | None = None,
        last_fitted_members: list[int] | None = None,
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
        self.diverged_runs: set[int] = {int(r) for r in (diverged_runs or set())}
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
        #: not provenance — it keys the series' dedupe signature (see
        #: ``_series_signature`` in ``project_model.py``) and drives live
        #: membership derivation (:meth:`effective_members`). ``None`` for a
        #: *frozen* series (a legacy/orphaned analysis with snapshot membership)
        #: and for detector-group series (``member_kind == "groups"``, D8).
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

    # ── label ──────────────────────────────────────────────────────────────

    def display_name(self, fallback: str) -> str:
        """Return the user-assigned label, or *fallback* when none is set."""
        return self.label or fallback

    @property
    def is_computed(self) -> bool:
        """True for a model-less *computed* series (e.g. an integral/field scan).

        A computed series carries per-run results directly in
        :attr:`results_by_run` but owns **no** fit model and **no** per-run
        :class:`FitSlot`\\ s. It must therefore be skipped by divergence checks
        and must not clear runs' fit state when deleted (a real fit on the same
        run is unrelated). A real batch/global fit always has a canonical model.
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
        self.diverged_runs.discard(run_number)

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

    # ── divergence ─────────────────────────────────────────────────────────

    def mark_diverged(self, run_number: int) -> None:
        self.diverged_runs.add(int(run_number))

    def clear_diverged(self, run_number: int) -> None:
        self.diverged_runs.discard(int(run_number))

    def is_diverged(self, run_number: int) -> bool:
        return int(run_number) in self.diverged_runs

    def trend_member_run_numbers(self) -> list[int]:
        """Return ordered members eligible for trending (non-diverged)."""
        return [r for r in self.member_run_numbers if r not in self.diverged_runs]

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
            "diverged_runs": sorted(self.diverged_runs),
            "extra": dict(self.extra),
            "source_group_id": self.source_group_id,
            "group_id": self.group_id,
            "excluded_run_numbers": list(self.excluded_run_numbers),
            "last_fitted_members": list(self.last_fitted_members),
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
            diverged_runs=data.get("diverged_runs"),
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
        )

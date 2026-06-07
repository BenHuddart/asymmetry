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

    # ── label ──────────────────────────────────────────────────────────────

    def display_name(self, fallback: str) -> str:
        """Return the user-assigned label, or *fallback* when none is set."""
        return self.label or fallback

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
        )

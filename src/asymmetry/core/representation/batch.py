"""The persisted batch/series object underpinning batch & global fits.

A :class:`Batch` is an ordered series of datasets that share one canonical
model for a single representation type.  Parameter roles use the existing
``global`` / ``local`` / ``fixed`` classifier:

* every parameter ``local``/``fixed``  → a *batch* fit (N independent fits);
* one or more parameter ``global``     → a *global* fit (shared across members).

So a global fit is derived from the classifier, not a separate object.
Parameter trending reads :attr:`Batch.results_by_run`.
"""

from __future__ import annotations

from typing import Any

from asymmetry.core.data.dataset import Run
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.representation.base import RepresentationType

#: Allowed per-parameter classification roles.
PARAM_ROLES = ("global", "local", "fixed")

#: Allowed series ordering keys.
ORDER_KEYS = ("field", "temperature", "run")


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


class Batch:
    """An ordered series of datasets fit with one canonical model."""

    def __init__(
        self,
        batch_id: str,
        rep_type: RepresentationType | str,
        *,
        member_run_numbers: list[int] | None = None,
        order_key: str = "run",
        canonical_model: dict | None = None,
        param_roles: dict[str, str] | None = None,
        results_by_run: dict[int, dict] | None = None,
        diverged_runs: set[int] | list[int] | None = None,
    ) -> None:
        self.batch_id = str(batch_id)
        self.rep_type = (
            rep_type if isinstance(rep_type, RepresentationType) else RepresentationType(str(rep_type))
        )
        self.member_run_numbers: list[int] = [int(r) for r in (member_run_numbers or [])]
        self.order_key = order_key if order_key in ORDER_KEYS else "run"
        self.canonical_model: dict | None = (
            dict(canonical_model) if isinstance(canonical_model, dict) else None
        )
        self.param_roles: dict[str, str] = {
            str(name): role
            for name, role in (param_roles or {}).items()
            if role in PARAM_ROLES
        }
        self.results_by_run: dict[int, dict] = {
            int(run): dict(result) for run, result in (results_by_run or {}).items()
        }
        self.diverged_runs: set[int] = {int(r) for r in (diverged_runs or set())}

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

    def add_member(self, run_number: int) -> None:
        """Add *run_number* to the series (idempotent)."""
        run_number = int(run_number)
        if run_number not in self.member_run_numbers:
            self.member_run_numbers.append(run_number)

    def remove_member(self, run_number: int) -> None:
        """Remove *run_number* from the series and drop its derived state."""
        run_number = int(run_number)
        self.member_run_numbers = [r for r in self.member_run_numbers if r != run_number]
        self.results_by_run.pop(run_number, None)
        self.diverged_runs.discard(run_number)

    def sort_members(self, runs_by_number: dict[int, Run]) -> None:
        """Order members by :attr:`order_key` using the supplied runs."""

        def key(run_number: int) -> tuple[float, int]:
            run = runs_by_number.get(run_number)
            if run is None or self.order_key == "run":
                return (float(run_number), run_number)
            value = run.field if self.order_key == "field" else run.temperature
            return (float(value), run_number)

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
            "rep_type": self.rep_type.value,
            "member_run_numbers": list(self.member_run_numbers),
            "order_key": self.order_key,
            "canonical_model": None if self.canonical_model is None else dict(self.canonical_model),
            "param_roles": dict(self.param_roles),
            "results_by_run": {str(run): dict(res) for run, res in self.results_by_run.items()},
            "diverged_runs": sorted(self.diverged_runs),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Batch:
        results_raw = data.get("results_by_run")
        results = (
            {int(run): dict(res) for run, res in results_raw.items() if isinstance(res, dict)}
            if isinstance(results_raw, dict)
            else {}
        )
        return cls(
            batch_id=str(data["batch_id"]),
            rep_type=data["rep_type"],
            member_run_numbers=data.get("member_run_numbers"),
            order_key=str(data.get("order_key", "run")),
            canonical_model=data.get("canonical_model"),
            param_roles=data.get("param_roles"),
            results_by_run=results,
            diverged_runs=data.get("diverged_runs"),
        )

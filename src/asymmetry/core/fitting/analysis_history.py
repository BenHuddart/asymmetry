"""Saved analysis states that require explicit authoring before another fit."""

from __future__ import annotations

CONSTRAINT_EDIT_MESSAGE = (
    "Historical constraints cannot be applied to this parameterisation. "
    "Edit the parameter constraints before refitting. The original state is retained."
)
MISSING_MODEL_MESSAGE = (
    "Historical parameters require an unavailable user function. "
    "Register the function and reload the project before refitting."
)


def _constraint_bound(value, default: float) -> float:
    return default if value is None or value == "" else float(value)


def constraint_signature(rows: list[dict]) -> list[dict]:
    """Compare authored constraints independently of displayed numeric formatting."""
    result = []
    for row in rows:
        if "name" not in row:
            continue
        bounds = str(row.get("bounds", "-inf, inf")).split(",")
        result.append(
            {
                "name": row["name"],
                "min": _constraint_bound(row.get("min", bounds[0]), -float("inf")),
                "max": _constraint_bound(row.get("max", bounds[-1]), float("inf")),
                "role": str(row.get("type") or row.get("role") or "local").strip().lower(),
                "fixed": bool(row["fixed"])
                if "fixed" in row
                else str(row.get("type") or row.get("role") or "").lower() == "fixed",
                "tie": row.get("tie"),
                "expr": row.get("expr"),
                "link_group": row.get("link_group"),
            }
        )
    return sorted(result, key=lambda row: row["name"])


def analysis_block_reason(history: dict | None, rows: list[dict]) -> str | None:
    """Historical content remains archived after an explicit constraint edit."""
    if history is None:
        return None
    if history["reason"] == "authored":
        return None
    if history["reason"] == "missing_model":
        return MISSING_MODEL_MESSAGE
    baseline = constraint_signature(history["canonical_constraints"])
    names = {row["name"] for row in baseline}
    current = [row for row in constraint_signature(rows) if row["name"] in names]
    if not baseline or current == baseline:
        return CONSTRAINT_EDIT_MESSAGE
    return None


def shared_constraint_signature(rows: list[dict]) -> list[dict]:
    """Shared membership and bounds distinguish constraint edits from new seeds."""
    return sorted(
        [
            {
                "name": row["name"],
                "members": row["members"],
                "min": -float("inf") if row.get("min") is None else float(row["min"]),
                "max": float("inf") if row.get("max") is None else float(row["max"]),
            }
            for row in rows
        ],
        key=lambda row: row["name"],
    )


def shared_analysis_block_reason(history: dict | None, rows: list[dict]) -> str | None:
    if history is None:
        return None
    if history["reason"] == "authored":
        return None
    if history["reason"] == "missing_model":
        return MISSING_MODEL_MESSAGE
    return (
        CONSTRAINT_EDIT_MESSAGE
        if shared_constraint_signature(rows) == history["canonical_shared"]
        else None
    )


def published_analysis_history(history: dict | None) -> dict | None:
    """Successful explicit execution supersedes the refit gate, not its archive."""
    return (
        None
        if history is None
        else {
            **history,
            "reason": "authored",
            "legacy_reason": history.get("legacy_reason", history["reason"]),
        }
    )

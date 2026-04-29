"""Diagnostic helpers for staged relaxed search."""

from __future__ import annotations

from asymmetry.core.fitting.global_search.types import RelaxedFitResult


def summarize_relaxed_fit(result: RelaxedFitResult) -> tuple[str, ...]:
    """Return concise diagnostic messages for one relaxed solve."""
    messages: list[str] = []
    if not result.success:
        messages.append(f"Relaxed optimization did not converge cleanly: {result.fit_message}")
    localized = sorted(
        name
        for name in result.base_values
        if any(
            abs(run_values.get(name, 0.0)) > 1e-6
            for run_values in result.deviations_by_run.values()
        )
    )
    if localized:
        messages.append("Relaxed deviations activated: " + ", ".join(localized))
    inactive = sorted(
        component_id for component_id, weight in result.activity_weights.items() if weight <= 0.02
    )
    if inactive:
        messages.append("Relaxed activity flagged weak components: " + ", ".join(inactive))
    return tuple(messages)

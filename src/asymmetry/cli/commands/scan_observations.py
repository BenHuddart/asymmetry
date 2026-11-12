"""Text status and repeated-coordinate inspection of persisted scan observations."""

import math

from asymmetry.core.transform.scalar_observation import ScalarObservation


def observation_lines(scan):
    lines = [scan.get("source_assumption", "Unknown acquisition identity is assumed distinct.")]
    for point in scan["points"]:
        record = point.get("observation")
        if record is not None and (record["reason"] != 0 or record["report"]["omitted"]):
            lines.append(
                f"Run {point['run']}: " + ScalarObservation.from_dict(record).report.summary()
            )
    if scan.get("correlated_pairs"):
        lines.append("Shared counting observations: independent-error fit unavailable")
    return lines


def repeated_points(points, correlated_pairs=()):
    """Retain repeated coordinates; significance needs usable independent errors."""
    groups = {}
    for point in points:
        groups.setdefault(point["x"], []).append(point)
    positions = {id(point): i for i, point in enumerate(points)}
    result = []
    for x, group in sorted(groups.items()):
        if len(group) < 2:
            continue
        usable = all(
            p["value"] is not None
            and p["error"] is not None
            and math.isfinite(p["value"])
            and math.isfinite(p["error"])
            and p["error"] > 0
            and (p.get("observation") is None or p["observation"]["reason"] == 0)
            for p in group
        )
        indices = {positions[id(point)] for point in group}
        usable &= not any(a in indices and b in indices for a, b in correlated_pairs)
        differ = (
            any(
                abs(a["value"] - b["value"]) > 3.0 * math.hypot(a["error"], b["error"])
                for a in group
                for b in group
            )
            if usable
            else None
        )
        result.append((x, group, differ))
    return result

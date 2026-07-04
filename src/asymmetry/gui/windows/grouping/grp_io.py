"""``.grp`` grouping-file serialization and shared formatting helpers.

Extracted from the former monolithic ``grouping_dialog.py`` as part of the
mechanical package split (milestone M1). The two public functions
:func:`serialize_grp` and :func:`parse_grp` implement the line-based ``.grp``
file format the grouping dialog loads and saves; the dialog's static
``serialize_grp``/``parse_grp`` methods delegate here so the class API is
unchanged. :func:`format_value_with_uncertainty` and the
:data:`ALPHA_METHOD_ITEMS` vocabulary are the alpha-estimate display helpers the
dialog form uses.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from asymmetry.core.utils.constants import PeriodMode

#: Alpha estimation methods offered by the Estimate control: combo label,
#: grouping-dict key, and a one-line explanation shown as the tooltip.
ALPHA_METHOD_ITEMS = (
    (
        "Diamagnetic (TF)",
        "diamagnetic",
        "Minimise the weighted asymmetry over a transverse-field calibration "
        "run, so A(t) oscillates symmetrically about zero.",
    ),
    (
        "General (LF/ZF)",
        "general",
        "Balance lifetime-corrected counts between early and late times; "
        "works on relaxing LF/ZF data, but needs visible relaxation.",
    ),
    (
        "Count ratio ΣF/ΣB",
        "ratio",
        "Plain count ratio (Mantid AlphaCalc). Transverse-field calibration "
        "runs only — relaxing polarization biases it.",
    ),
)


def format_value_with_uncertainty(value: float, error: float | None) -> str:
    """Format ``1.2345 ± 0.0067`` compactly as ``1.2345(67)``."""
    if error is None or not np.isfinite(error) or error <= 0.0:
        return f"{value:.4f}"
    exponent = int(np.floor(np.log10(error)))
    decimals = max(0, 1 - exponent)
    scaled_error = int(round(error * 10**decimals))
    if scaled_error >= 100:  # rounding pushed it to three digits, e.g. 0.0995
        scaled_error = int(round(scaled_error / 10))
        decimals -= 1
    if decimals < 0:  # uncertainty >= ~100: integer digits on both sides
        return f"{value:.0f}({int(round(error))})"
    return f"{value:.{decimals}f}({scaled_error})"


def serialize_grp(payload: dict[str, Any]) -> str:
    """Serialize grouping payload to text ``.grp`` format.

    The generated file is intentionally simple and line-based:

    ``key=value`` for scalars and ``group.<id>=csv`` for detector lists.
    Detector indices are stored 1-based for compatibility with existing
    μSR tooling conventions.
    """
    lines = [
        "# Asymmetry grouping file v1",
        f"forward_group={int(payload.get('forward_group', 1))}",
        f"backward_group={int(payload.get('backward_group', 2))}",
        f"alpha={float(payload.get('alpha', 1.0)):.12g}",
        f"alpha_x={float(payload.get('alpha_x', payload.get('alpha', 1.0))):.12g}",
        f"alpha_y={float(payload.get('alpha_y', payload.get('alpha', 1.0))):.12g}",
        f"alpha_z={float(payload.get('alpha_z', payload.get('alpha', 1.0))):.12g}",
        f"t0_bin={int(payload.get('t0_bin', 0))}",
        f"t_good_offset={int(payload.get('t_good_offset', int(payload.get('first_good_bin', 0)) - int(payload.get('t0_bin', 0))))}",
        f"first_good_bin={int(payload.get('first_good_bin', 0))}",
        f"last_good_bin={int(payload.get('last_good_bin', 0))}",
        f"bunching_factor={int(payload.get('bunching_factor', 1))}",
        f"bin_index_base={1 if int(payload.get('bin_index_base', 0)) == 1 else 0}",
        f"deadtime_correction={1 if bool(payload.get('deadtime_correction', False)) else 0}",
        f"deadtime_mode={str(payload.get('deadtime_mode', 'off'))}",
        f"deadtime_method={str(payload.get('deadtime_method', ''))}",
        f"deadtime_manual_us={float(payload.get('deadtime_manual_us', 0.0)):.12g}",
        f"deadtime_estimated_us={float(payload.get('deadtime_estimated_us', 0.0)):.12g}",
        f"background_correction={1 if bool(payload.get('background_correction', False)) else 0}",
        f"period_mode={str(payload.get('period_mode', PeriodMode.RED))}",
    ]

    if "deadtime_reference_run" in payload:
        lines.append(f"deadtime_reference_run={int(payload.get('deadtime_reference_run', 0))}")
    dead_time_us = payload.get("dead_time_us")
    if isinstance(dead_time_us, list) and dead_time_us:
        values = ",".join(f"{float(value):.12g}" for value in dead_time_us)
        lines.append(f"dead_time_us={values}")

    groups = payload.get("groups", {})
    group_names = payload.get("group_names", {})
    if isinstance(groups, dict):
        for gid in sorted(int(k) for k in groups.keys()):
            detectors = [str(int(v)) for v in groups.get(gid, [])]
            lines.append(f"group.{gid}={','.join(detectors)}")
    if isinstance(group_names, dict):
        for gid in sorted(int(k) for k in group_names.keys()):
            name = str(group_names.get(gid, "")).strip()
            if name:
                lines.append(f"group_name.{gid}={name}")
    included_groups = payload.get("included_groups", {})
    if isinstance(included_groups, dict):
        for gid in sorted(int(k) for k in included_groups.keys()):
            include = 1 if bool(included_groups.get(gid, True)) else 0
            lines.append(f"group_include.{gid}={include}")

    # Per-projection rows carry the non-canonical per-projection alpha that
    # has no scalar key (alpha_x/y/z only cover the canonical EMU axes).
    # Format: projection.<n>=label,forward_group,backward_group,alpha[,tint].
    # Projection labels are controlled (P_x, FB, Top-Bottom, …) and never
    # contain commas, so a comma-delimited value is safe.
    projections = payload.get("projections", [])
    if isinstance(projections, list):
        for idx, proj in enumerate(projections):
            if not isinstance(proj, dict):
                continue
            label = str(proj.get("label", "")).strip()
            if not label:
                continue
            fields = [
                label,
                str(int(proj.get("forward_group", 0))),
                str(int(proj.get("backward_group", 0))),
                f"{float(proj.get('alpha', 1.0)):.12g}",
            ]
            tint = proj.get("tint")
            if tint:
                fields.append(str(tint))
            lines.append(f"projection.{idx}={','.join(fields)}")

    return "\n".join(lines) + "\n"


def parse_grp(text: str) -> dict[str, Any]:
    """Parse line-based ``.grp`` text into a grouping payload dictionary."""
    payload: dict[str, Any] = {
        "groups": {},
        "included_groups": {},
        "group_names": {},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "t0_bin": 0,
        "t_good_offset": 0,
        "first_good_bin": 0,
        "last_good_bin": 0,
        "bunching_factor": 1,
        "deadtime_correction": False,
        "deadtime_mode": "off",
        "background_correction": False,
        "period_mode": str(PeriodMode.RED),
        "bin_index_base": 0,
    }
    saw_t_good_offset = False
    projection_rows: dict[int, dict[str, Any]] = {}

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if key.startswith("group_name."):
            gid = int(key.split(".", 1)[1])
            payload["group_names"][gid] = value
            continue

        if key.startswith("group_include."):
            gid = int(key.split(".", 1)[1])
            payload["included_groups"][gid] = value.strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            continue

        if key.startswith("group."):
            gid = int(key.split(".", 1)[1])
            dets = [int(v.strip()) for v in value.split(",") if v.strip()]
            payload["groups"][gid] = dets
            continue

        if key.startswith("projection."):
            parts = [p.strip() for p in value.split(",")]
            if len(parts) >= 3 and parts[0]:
                proj: dict[str, Any] = {
                    "label": parts[0],
                    "forward_group": int(parts[1]),
                    "backward_group": int(parts[2]),
                }
                if len(parts) >= 4:
                    proj["alpha"] = float(parts[3])
                if len(parts) >= 5 and parts[4]:
                    proj["tint"] = parts[4]
                projection_rows[int(key.split(".", 1)[1])] = proj
            continue

        if key in {
            "forward_group",
            "backward_group",
            "t0_bin",
            "t_good_offset",
            "first_good_bin",
            "last_good_bin",
            "bunching_factor",
            "bin_index_base",
        }:
            payload[key] = int(float(value))
            if key == "t_good_offset":
                saw_t_good_offset = True
        elif key in {
            "alpha",
            "alpha_x",
            "alpha_y",
            "alpha_z",
            "alpha_px",
            "alpha_py",
            "alpha_pz",
            "deadtime_manual_us",
            "deadtime_estimated_us",
        }:
            payload[key] = float(value)
        elif key in {"deadtime_mode", "deadtime_method", "deadtime_source_path"}:
            payload[key] = value
        elif key == "deadtime_reference_run":
            payload[key] = int(float(value))
        elif key in {"dead_time_us", "deadtime_loaded_us"}:
            payload[key] = [float(v.strip()) for v in value.split(",") if v.strip()]
        elif key == "deadtime_correction":
            payload[key] = value.strip().lower() in {"1", "true", "yes", "on"}
        elif key == "background_correction":
            payload[key] = value.strip().lower() in {"1", "true", "yes", "on"}
        elif key == "period_mode":
            if value in {
                str(PeriodMode.RED),
                str(PeriodMode.GREEN),
                str(PeriodMode.GREEN_MINUS_RED),
                str(PeriodMode.GREEN_PLUS_RED),
            }:
                payload[key] = value

    if projection_rows:
        payload["projections"] = [projection_rows[idx] for idx in sorted(projection_rows)]

    alpha_scalar = float(payload.get("alpha", 1.0))
    payload.setdefault("alpha_x", float(payload.get("alpha_px", alpha_scalar)))
    payload.setdefault("alpha_y", float(payload.get("alpha_py", alpha_scalar)))
    payload.setdefault("alpha_z", float(payload.get("alpha_pz", alpha_scalar)))
    t0_bin = int(payload.get("t0_bin", 0))
    if saw_t_good_offset:
        payload["t_good_offset"] = max(0, int(payload.get("t_good_offset", 0)))
    else:
        payload["t_good_offset"] = max(0, int(payload.get("first_good_bin", 0)) - t0_bin)
    payload["first_good_bin"] = max(0, t0_bin + int(payload.get("t_good_offset", 0)))
    payload["bin_index_base"] = 1 if int(payload.get("bin_index_base", 0)) == 1 else 0
    if isinstance(payload.get("groups"), dict):
        included_groups = payload.get("included_groups")
        if not isinstance(included_groups, dict):
            included_groups = {}
        payload["included_groups"] = {
            int(gid): bool(included_groups.get(int(gid), True)) for gid in payload["groups"]
        }

    return payload


__all__ = [
    "ALPHA_METHOD_ITEMS",
    "format_value_with_uncertainty",
    "serialize_grp",
    "parse_grp",
]

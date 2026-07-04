"""Core ``GlobalFitStudy``: the persisted entity for a cross-group global
parameter fit ("fit λ(B) at eight temperatures with some parameters shared").

Historically the GUI kept exactly one such fit in memory
(``FitParametersPanel._last_cross_group_fit``, a bare dict of core objects
serialized ad hoc into panel state — see
:meth:`asymmetry.gui.panels.fit_parameters_panel.FitParametersPanel._serialize_last_cross_group_fit`).
``GlobalFitStudy`` replaces that single slot with a named, persisted, and
uniquely-identified entity so a project can carry several studies side by
side (rename/duplicate/delete, per docs/PLANS.md "Global Parameter Fit
Studies"). This module is pure core: no Qt, no matplotlib.

Serialization delegates to the canonical (de)serializers added in Phase 1A:
:meth:`ParameterGroupData.to_dict`/``from_dict``,
:meth:`CrossGroupFitResult.to_dict`/``from_dict``, and
:meth:`ParameterCompositeModel.to_dict`/``from_dict``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from asymmetry.core.fitting.parameter_models import (
    CrossGroupFitResult,
    ParameterCompositeModel,
    ParameterGroupData,
)

__all__ = [
    "GlobalFitStudy",
    "compute_group_input_digest",
    "study_from_legacy_cross_group_payload",
]


@dataclass
class GlobalFitStudy:
    """A named, persisted cross-group global parameter fit.

    ``study_id`` is caller-supplied (the GUI derives a deterministic
    ``modelfit-<digest>`` id, matching the existing decoration-keying scheme
    used by :mod:`asymmetry.gui.windows.global_parameter_fit_window`) — this
    module never generates one.

    ``input_digest`` is a caller-computed hash of the snapshot ``groups``
    arrays (see :func:`compute_group_input_digest`); it lets a GUI detect that
    a study's inputs have drifted from the live trend data ("stale") without
    this dataclass storing a ``stale`` flag itself — staleness is a runtime
    property recomputed by comparing the stored digest against a freshly
    computed one, never persisted.
    """

    study_id: str
    name: str
    parameter_name: str
    x_key: str
    x_label: str
    group_variable_key: str
    group_variable_label: str
    created: str
    updated: str
    source_group_ids: list[str] = field(default_factory=list)
    groups: list[ParameterGroupData] = field(default_factory=list)
    model: ParameterCompositeModel | None = None
    config: dict[str, Any] = field(default_factory=dict)
    result: CrossGroupFitResult | None = None
    fit_x_min: float = float("nan")
    fit_x_max: float = float("nan")
    input_digest: str = ""

    # ── persistence ────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Canonical, JSON-safe serialization.

        ``groups``/``model``/``result`` delegate to their own canonical
        ``to_dict()`` (Phase 1A). A study with ``model`` or ``result`` unset
        still serializes (both keys become ``None``) — :meth:`from_dict`
        rejects that payload on the way back in, since a study without a
        model or a result is not a usable one.
        """
        return {
            "study_id": self.study_id,
            "name": self.name,
            "parameter_name": self.parameter_name,
            "x_key": self.x_key,
            "x_label": self.x_label,
            "group_variable_key": self.group_variable_key,
            "group_variable_label": self.group_variable_label,
            "created": self.created,
            "updated": self.updated,
            "source_group_ids": list(self.source_group_ids),
            "groups": [g.to_dict() for g in self.groups],
            "model": None if self.model is None else self.model.to_dict(),
            "config": dict(self.config),
            "result": None if self.result is None else self.result.to_dict(),
            "fit_x_min": (float(self.fit_x_min) if np.isfinite(self.fit_x_min) else None),
            "fit_x_max": (float(self.fit_x_max) if np.isfinite(self.fit_x_max) else None),
            "input_digest": self.input_digest,
        }

    @classmethod
    def from_dict(cls, data: dict) -> GlobalFitStudy | None:
        """Tolerant deserialization.

        Returns ``None`` (rather than raising) when the payload is missing a
        ``model`` or a ``result`` — a study without either is not something
        the GUI can display or refit, so a caller iterating a persisted
        ``global_fit_studies`` list should simply skip a ``None`` entry
        instead of the whole project failing to load. Malformed group entries
        are skipped individually (mirroring
        :meth:`ParameterGroupData.from_dict`'s own tolerance for missing
        keys); a group entry that is not a mapping at all is dropped.
        """
        if not isinstance(data, dict):
            return None

        model_data = data.get("model")
        if not isinstance(model_data, dict):
            return None
        try:
            model = ParameterCompositeModel.from_dict(model_data)
        except (ValueError, KeyError, TypeError):
            return None

        result_data = data.get("result")
        if not isinstance(result_data, dict):
            return None
        try:
            result = CrossGroupFitResult.from_dict(result_data)
        except (ValueError, KeyError, TypeError):
            return None

        groups: list[ParameterGroupData] = []
        raw_groups = data.get("groups")
        if isinstance(raw_groups, list):
            for entry in raw_groups:
                if not isinstance(entry, dict):
                    continue
                try:
                    groups.append(ParameterGroupData.from_dict(entry))
                except (ValueError, KeyError, TypeError):
                    continue

        raw_source_ids = data.get("source_group_ids")
        source_group_ids = (
            [str(gid) for gid in raw_source_ids] if isinstance(raw_source_ids, list) else []
        )

        raw_config = data.get("config")
        config = dict(raw_config) if isinstance(raw_config, dict) else {}

        fit_x_min_raw = data.get("fit_x_min")
        fit_x_max_raw = data.get("fit_x_max")
        fit_x_min = (
            float(fit_x_min_raw) if isinstance(fit_x_min_raw, (int, float)) else float("nan")
        )
        fit_x_max = (
            float(fit_x_max_raw) if isinstance(fit_x_max_raw, (int, float)) else float("nan")
        )

        return cls(
            study_id=str(data.get("study_id", "")),
            name=str(data.get("name", "")),
            parameter_name=str(data.get("parameter_name", "")),
            x_key=str(data.get("x_key", "run")),
            x_label=str(data.get("x_label", "")),
            group_variable_key=str(data.get("group_variable_key", "")),
            group_variable_label=str(data.get("group_variable_label", "")),
            created=str(data.get("created", "")),
            updated=str(data.get("updated", "")),
            source_group_ids=source_group_ids,
            groups=groups,
            model=model,
            config=config,
            result=result,
            fit_x_min=fit_x_min,
            fit_x_max=fit_x_max,
            input_digest=str(data.get("input_digest", "")),
        )


def compute_group_input_digest(groups: list[ParameterGroupData]) -> str:
    """Return a stable short hash over *groups*' identity and data arrays.

    Used by the GUI to detect that a study's stored snapshot has drifted from
    the live trend data ("stale"): a study records the digest of the groups it
    was fit against, and a fresh call against the current trend rows is
    compared against it.

    Groups are sorted by ``group_id`` before hashing so that permuting the
    input list (e.g. a different multi-select order in the trend panel) never
    changes the digest — only the *content* of the groups matters. Arrays are
    rounded to a fixed number of decimal digits before hashing so that
    harmless floating-point representation differences (e.g. serialize round
    trip through JSON, or platform-dependent ``float`` repr) do not register
    as a spurious staleness signal; the rounded array's raw bytes
    (``.tobytes()``) are hashed rather than a text ``repr()``, which keeps the
    digest deterministic across platforms (unlike ``repr()``, whose float
    formatting is not guaranteed to be byte-identical across numpy builds).
    """
    hasher = hashlib.sha1()
    for group in sorted(groups, key=lambda g: g.group_id):
        hasher.update(group.group_id.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(_round_array_bytes(group.x))
        hasher.update(_round_array_bytes(group.y))
        hasher.update(_round_array_bytes(group.yerr))
        if group.xerr is not None:
            hasher.update(_round_array_bytes(group.xerr))
        hasher.update(_round_array_bytes(np.asarray([group.group_variable_value])))
    return hasher.hexdigest()[:12]


def _round_array_bytes(values: np.ndarray, decimals: int = 10) -> bytes:
    """Return deterministic bytes for a float array, insensitive to noise
    below *decimals* decimal places."""
    arr = np.asarray(values, dtype=np.float64)
    rounded = np.round(arr, decimals=decimals)
    # Normalise -0.0 -> 0.0 so a value that rounds to zero from either side
    # hashes identically.
    rounded = rounded + 0.0
    return rounded.tobytes()


def study_from_legacy_cross_group_payload(
    payload: dict,
    *,
    study_id: str,
    name: str,
    created: str,
) -> GlobalFitStudy | None:
    """Adapt a legacy single-slot cross-group-fit payload into a study.

    *payload* is the on-disk shape written by (and read back through) the
    trend panel's now-superseded single-slot persistence — see
    ``FitParametersPanel._serialize_last_cross_group_fit`` /
    ``_deserialize_last_cross_group_fit`` in
    ``src/asymmetry/gui/panels/fit_parameters_panel.py`` (as of this writing,
    lines 4198-4396). That payload has keys ``parameter_name``, ``x_key``,
    ``fit_x_min``, ``fit_x_max``, ``config``, ``config_key``, ``groups``
    (list of ``{group_id, group_name, x, y, yerr, group_variable_value}`` —
    note: no ``xerr``), ``model`` (a :class:`ParameterCompositeModel` dict),
    and ``fit_result`` (a hand-rolled subset of
    :class:`CrossGroupFitResult`'s fields: ``success``, ``chi_squared``,
    ``reduced_chi_squared``, ``message``, ``global_parameters``,
    ``global_uncertainties``, ``local_parameters``, ``fixed_parameters``,
    ``local_uncertainties`` — no ``error_mode``, ``n_points``,
    ``per_group_chi_squared``, ``per_group_n_points``, or
    ``global_correlations``, all Phase 1A additions that default sensibly via
    :meth:`CrossGroupFitResult.from_dict`).

    The legacy payload has no ``group_variable_key``/``group_variable_label``/
    ``source_group_ids`` (those did not exist as first-class concepts yet);
    they default to ``""``/``""``/``[]`` respectively. ``x_label`` is not
    stored either and defaults to the bare ``x_key``.

    Returns ``None`` when the payload is missing or malformed in a way that
    would leave the study without a usable model/result (mirroring
    :meth:`GlobalFitStudy.from_dict`'s own contract).
    """
    if not isinstance(payload, dict):
        return None

    model_data = payload.get("model")
    if not isinstance(model_data, dict):
        return None
    try:
        model = ParameterCompositeModel.from_dict(model_data)
    except (ValueError, KeyError, TypeError):
        return None

    fit_result_data = payload.get("fit_result")
    if not isinstance(fit_result_data, dict):
        return None
    try:
        result = CrossGroupFitResult.from_dict(fit_result_data)
    except (ValueError, KeyError, TypeError):
        return None

    groups: list[ParameterGroupData] = []
    raw_groups = payload.get("groups")
    if isinstance(raw_groups, list):
        for entry in raw_groups:
            if not isinstance(entry, dict):
                continue
            try:
                groups.append(ParameterGroupData.from_dict(entry))
            except (ValueError, KeyError, TypeError):
                continue

    raw_config = payload.get("config")
    config = dict(raw_config) if isinstance(raw_config, dict) else {}

    fit_x_min_raw = payload.get("fit_x_min")
    fit_x_max_raw = payload.get("fit_x_max")
    fit_x_min = float(fit_x_min_raw) if isinstance(fit_x_min_raw, (int, float)) else float("nan")
    fit_x_max = float(fit_x_max_raw) if isinstance(fit_x_max_raw, (int, float)) else float("nan")

    x_key = str(payload.get("x_key", "run"))

    return GlobalFitStudy(
        study_id=str(study_id),
        name=str(name),
        parameter_name=str(payload.get("parameter_name", "")),
        x_key=x_key,
        x_label=x_key,
        group_variable_key="",
        group_variable_label="",
        created=str(created),
        updated=str(created),
        source_group_ids=[],
        groups=groups,
        model=model,
        config=config,
        result=result,
        fit_x_min=fit_x_min,
        fit_x_max=fit_x_max,
        input_digest=compute_group_input_digest(groups),
    )

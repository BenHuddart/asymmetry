"""NeXus frame origins at the file boundary, independent of DT parsing."""

import json
from dataclasses import dataclass

import numpy as np

from asymmetry.core.transform.reference_scale import ReferenceExposure


@dataclass(frozen=True)
class NexusFrames:
    """Original DT denominator values and separately sourced subtraction exposure."""

    values: np.ndarray
    origin: str
    declared_exposure: ReferenceExposure | None = None

    def exposures(self, count):
        """Return only applicable exposure evidence for the stored data periods."""
        if self.declared_exposure is not None and count == 1:
            return [self.declared_exposure.to_dict()]
        return nexus_period_exposures(self.values, self.origin, count)


def read_nexus_frames(entry, read_optional, *, legacy):
    """Retain the established DT field precedence and the chosen field's origin."""
    declaration = read_optional(entry, "asymmetry_reference_exposure")
    declared = None
    if declaration is not None:
        try:
            raw_declaration = np.asarray(declaration).item()
            if isinstance(raw_declaration, bytes):
                raw_declaration = raw_declaration.decode("utf-8")
            declared = ReferenceExposure.from_dict(json.loads(raw_declaration))
        except (TypeError, ValueError, KeyError, UnicodeError):
            declared = ReferenceExposure(origin="unknown:invalid declaration")
    periods = read_optional(entry, "periods")
    direct = [
        (entry, "good_frames", "good_frames"),
        (entry, "goodfrm", "goodfrm"),
        (periods, "good_frames", "periods/good_frames"),
    ]
    if not legacy:
        direct[1], direct[2] = direct[2], direct[1]
    beam = read_optional(read_optional(entry, "instrument"), "beam")
    fields = direct + [
        (beam, key, f"instrument/beam/{key}")
        for key in ("frames_period_daq", "frames_period", "frames_good", "frames")
    ]
    for node, key, origin in fields:
        raw = read_optional(node, key, default=[])
        try:
            values = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError):
            return NexusFrames(np.asarray([], dtype=np.float64), f"unusable:{origin}", declared)
        if values.size:
            return NexusFrames(values, origin, declared)
    return NexusFrames(np.asarray([], dtype=np.float64), "missing", declared)


def nexus_period_exposures(values, origin, n_periods):
    """Measure only applicable good-frame counts, never broadcast a run total.

    DAQ-period counts correspond to stored histogram periods. DAE-period counts
    can include ramp periods and do not establish that correspondence. A scalar
    run total cannot identify exposure in each of several acquired periods.
    """
    values = values.reshape(-1)
    applicable = (
        origin
        in {
            "good_frames",
            "goodfrm",
            "periods/good_frames",
            "instrument/beam/frames_period_daq",
            "instrument/beam/frames_good",
        }
        and values.size == n_periods
        and (origin != "instrument/beam/frames_good" or n_periods == 1)
    )
    return [
        ReferenceExposure(float(value), f"NeXus:{origin}", period_number=index + 1).to_dict()
        if applicable and np.isfinite(value) and value > 0
        else ReferenceExposure(origin=f"unknown:NeXus:{origin}", period_number=index + 1).to_dict()
        for index, value in enumerate(values if applicable else [None] * n_periods)
    ]


def summed_period_exposures(runs, period_sets):
    """Sum evidenced exposures only within the same explicit exposure unit."""
    exposures = []
    for periods in period_sets:
        if not periods:
            continue
        facts = [ReferenceExposure.from_grouping(runs[p - 1].grouping) for p in periods]
        exposure = ReferenceExposure()
        if all(fact.value is not None and fact.unit == facts[0].unit for fact in facts):
            exposure = ReferenceExposure(
                sum(fact.value for fact in facts),
                "acquired_period_sum:" + ",".join(fact.origin for fact in facts),
                facts[0].unit,
            )
        exposures.append(dict(exposure.to_dict(), period_number=len(exposures) + 1))
    return exposures

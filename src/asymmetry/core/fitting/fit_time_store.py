"""This computer's measured wizard fit times, per component.

The fit wizards time every template fit (:class:`~asymmetry.core.fitting.fit_wizard.FitTiming`).
After a run, :meth:`FitTimeStore.record` adds one sample per component in
seconds per 1000 fitted points, and :meth:`FitTimeStore.estimates` scales the
rolling median to the selected runs for the slow judgement
(:class:`~asymmetry.core.fitting.wizard_scope.FitTimeEstimates`). The store is a
small JSON file the caller locates; it never travels in a project. Design:
``docs/plans/measured-fit-times.md``.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.component_tags import PhysicsClass
from asymmetry.core.fitting.composite import COMPONENTS
from asymmetry.core.fitting.fit_wizard import CandidateAssessment, screening_points
from asymmetry.core.fitting.wizard_scope import UNTIMED, FitTimeEstimates

__all__ = ["SAMPLES_KEPT", "STORE_VERSION", "FitTimeStore"]

#: Samples kept per component, newest last; the estimate is their median.
SAMPLES_KEPT = 9
#: The file format :meth:`FitTimeStore.save` writes and :meth:`FitTimeStore.load` reads.
STORE_VERSION = 1

_KEYS = frozenset({"version", "seconds_per_kpoint"})


def _is_background(name: str) -> bool:
    return COMPONENTS[name].physics_classes == frozenset({PhysicsClass.BACKGROUND})


class FitTimeStore:
    """Recent seconds-per-1000-points samples for each component timed on this computer."""

    def __init__(self, samples: Mapping[str, list[float]] | None = None) -> None:
        self._samples: dict[str, list[float]] = {
            name: list(values) for name, values in (samples or {}).items()
        }

    @property
    def samples(self) -> dict[str, tuple[float, ...]]:
        """Each component's kept samples, oldest first."""
        return {name: tuple(values) for name, values in self._samples.items()}

    @classmethod
    def load(cls, path: Path) -> FitTimeStore:
        """Read the store at *path*; an absent file is an empty store.

        A malformed file raises :class:`ValueError` naming *path*.
        """
        if not path.exists():
            return cls()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ValueError(f"fit-time store {path} is not JSON: {exc}") from exc
        if not isinstance(payload, dict) or set(payload) != _KEYS:
            raise ValueError(f"fit-time store {path} must hold exactly the keys {sorted(_KEYS)}")
        if payload["version"] != STORE_VERSION:
            raise ValueError(
                f"fit-time store {path} is version {payload['version']!r}, expected {STORE_VERSION}"
            )
        samples = payload["seconds_per_kpoint"]
        if not isinstance(samples, dict) or not all(
            isinstance(values, list)
            and 0 < len(values) <= SAMPLES_KEPT
            and all(
                isinstance(value, int | float)
                and not isinstance(value, bool)
                and math.isfinite(value)
                and value >= 0.0
                for value in values
            )
            for values in samples.values()
        ):
            raise ValueError(
                f"fit-time store {path}: 'seconds_per_kpoint' maps each component to 1–"
                f"{SAMPLES_KEPT} finite non-negative seconds"
            )
        return cls({str(name): [float(v) for v in values] for name, values in samples.items()})

    def save(self, path: Path) -> None:
        """Write the store to *path*, creating its folder."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": STORE_VERSION, "seconds_per_kpoint": self._samples}
        path.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")

    def record(self, assessments: Iterable[CandidateAssessment]) -> None:
        """Add one sample per component: the median over one wizard run's fits that contain it.

        Every assessment must have been fitted in that run (it carries a
        :class:`~asymmetry.core.fitting.fit_wizard.FitTiming`).
        """
        per_component: defaultdict[str, list[float]] = defaultdict(list)
        for assessment in assessments:
            if assessment.timing is None:
                raise ValueError(
                    f"assessment {assessment.template.key!r} carries no timing: only rows "
                    "fitted in this run are recorded"
                )
            # A background component rides along in every template, so it is never attributed a time.
            for name in set(assessment.template.model.component_names):
                if not _is_background(name):
                    per_component[name].append(assessment.timing.seconds_per_kpoint)
        for name, values in per_component.items():
            kept = [*self._samples.get(name, []), statistics.median(values)]
            self._samples[name] = kept[-SAMPLES_KEPT:]

    def estimates(self, datasets: Iterable[MuonDataset]) -> FitTimeEstimates:
        """Expected screening-fit seconds per run for each timed component.

        The median per-1000-point time scaled to the runs' typical length: the
        median over runs of :func:`~asymmetry.core.fitting.fit_wizard.screening_points`.
        With no runs there is nothing to scale to, so nothing is timed.
        """
        lengths = [screening_points(dataset) for dataset in datasets]
        if not lengths:
            return UNTIMED
        points = statistics.median(lengths)
        return FitTimeEstimates(
            {
                name: statistics.median(values) * points / 1000.0
                for name, values in self._samples.items()
            }
        )

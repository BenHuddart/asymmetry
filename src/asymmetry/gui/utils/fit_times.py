"""This computer's measured wizard fit times, shared by both fit wizards.

One :class:`~asymmetry.core.fitting.fit_time_store.FitTimeStore` per process,
loaded once from the app data folder, so one wizard's update reaches the other
and neither overwrites the other's samples. Design:
``docs/plans/measured-fit-times.md`` (D3).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from functools import cache
from pathlib import Path

from PySide6.QtCore import QStandardPaths

from asymmetry.core.fitting.fit_time_store import FitTimeStore
from asymmetry.core.fitting.fit_wizard import CandidateAssessment

_log = logging.getLogger(__name__)


def fit_times_path() -> Path:
    """Where this computer's fit times live: ``fit_times.json`` in the app data folder."""
    location = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
    return Path(location) / "fit_times.json"


@cache
def shared_fit_time_store() -> FitTimeStore:
    """The process's store; a corrupt file is discarded, since the times can be measured again."""
    path = fit_times_path()
    try:
        return FitTimeStore.load(path)
    except ValueError as exc:
        _log.warning("Discarding the fit-time store: %s", exc)
        return FitTimeStore()


def record_fit_times(assessments: Iterable[CandidateAssessment]) -> None:
    """Add one wizard run's fits to the shared store and save it."""
    store = shared_fit_time_store()
    store.record(assessments)
    store.save(fit_times_path())

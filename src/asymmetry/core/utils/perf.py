"""Lightweight perf-timing helper for the core layer.

Mirrors the app's existing GUI perf-logging convention (``ASYMMETRY_PERF_LOGGING``
env var, a ``PERF {event}: {elapsed_ms:.1f} ms ...`` message shape — see
``asymmetry.gui.mainwindow``'s ``_perf_logging_is_enabled`` /
``_log_perf_event``) but stays import-clean for ``asymmetry.core``: no Qt, no
GUI widget, no log panel. Records go through the stdlib ``logging`` module on
the ``asymmetry.perf`` logger, which the GUI toggle enables/disables via
:func:`set_perf_logging` so one preference controls both the GUI's own PERF
lines and these core timers.

Usage::

    from asymmetry.core.utils.perf import perf_timer

    with perf_timer("core.reduce.grouped_asymmetry", n_detectors=12) as perf:
        ...
        perf.detail(n_bins=int(n_bins))
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager

logger = logging.getLogger("asymmetry.perf")

#: Truthy string tokens accepted from the env var, matching the GUI's
#: ``_coerce_bool`` convention (case-insensitive).
_TRUTHY_TOKENS = {"1", "true", "yes"}

#: Env var checked when no explicit override has been set via
#: :func:`set_perf_logging` — kept in sync with the GUI's own
#: ``_PERF_LOGGING_ENV_VAR`` name.
_PERF_LOGGING_ENV_VAR = "ASYMMETRY_PERF_LOGGING"

#: Explicit override set by :func:`set_perf_logging`. ``None`` means "no
#: override" — fall back to the environment variable.
_override: bool | None = None


def _env_enabled() -> bool:
    """Return whether the env var currently requests perf logging."""
    value = os.environ.get(_PERF_LOGGING_ENV_VAR)
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY_TOKENS


def perf_logging_enabled() -> bool:
    """Return whether core perf timers should log.

    An explicit override set via :func:`set_perf_logging` always wins; absent
    one, falls back to a live (uncached) check of ``ASYMMETRY_PERF_LOGGING``
    so tests can toggle the env var without needing to reset any cache.
    """
    if _override is not None:
        return _override
    return _env_enabled()


def set_perf_logging(enabled: bool | None) -> None:
    """Set (or clear) the explicit perf-logging override.

    ``True``/``False`` pins the state regardless of the environment variable
    (the GUI's "Performance logging" toggle calls this with its own resolved
    boolean so the two agree). ``None`` clears the override and returns to the
    env-var fallback.
    """
    global _override
    _override = enabled


class _PerfRecorder:
    """Handle yielded by :func:`perf_timer` for adding late detail fields."""

    __slots__ = ("_late_detail",)

    def __init__(self) -> None:
        self._late_detail: dict[str, object] = {}

    def detail(self, **kwargs: object) -> None:
        """Record additional detail fields, known only once work has run."""
        self._late_detail.update(kwargs)


def _format_detail(fields: dict[str, object]) -> str:
    parts = [f"{key}={value}" for key, value in fields.items() if value is not None]
    return " " + " ".join(parts) if parts else ""


@contextmanager
def perf_timer(event: str, **static_detail: object) -> Iterator[_PerfRecorder]:
    """Context manager timing a block and logging a ``PERF`` record.

    Always measures elapsed time (a ``time.perf_counter`` pair is
    nanosecond-cheap); when perf logging is disabled, all formatting and
    logging is skipped so the disabled-path overhead is negligible.

    ``static_detail`` fields are known up front (e.g. counts already in hand);
    call ``.detail(**kwargs)`` on the yielded recorder for values only known
    after the timed work runs (e.g. a resulting bin count). Static detail is
    logged first, followed by late detail, in the order each was supplied.
    Exceptions raised inside the block propagate unchanged; a failure is
    logged with ``failed=True`` when perf logging is enabled.
    """
    recorder = _PerfRecorder()
    started_at = time.perf_counter()
    failed = False
    try:
        yield recorder
    except BaseException:
        failed = True
        raise
    finally:
        if perf_logging_enabled():
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            fields: dict[str, object] = dict(static_detail)
            fields.update(recorder._late_detail)
            if failed:
                fields["failed"] = True
            detail_text = _format_detail(fields)
            logger.info("PERF %s: %.1f ms%s", event, elapsed_ms, detail_text)

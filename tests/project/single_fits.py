"""Read the v25 ``single_fits`` shape of a migrated representation dict."""

from __future__ import annotations


def open_fit(representation: dict, projection: str = "") -> dict:
    """The open saved single fit on *projection* (``""`` = the default set)."""
    fit_set = representation["single_fits"][projection]
    return next(fit for fit in fit_set["fits"] if fit["fit_id"] == fit_set["open_id"])

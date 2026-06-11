"""Behaviour-pinning for the unified ``reference_field_gauss`` resolver (F2).

``gui/panels/plot_panel._frequency_reference_for_dataset`` re-implemented the
dataset-metadata → run-metadata ``field`` lookup that core
``fourier/spectrum._reference_field_gauss`` already performed, differing only
in the trailing Gauss→MHz conversion. The reconciliation publicises the core
resolver and has the panel delegate to it.

This pins (a) the public ``reference_field_gauss`` against a verbatim copy of
the removed core body, and (b) the panel's delegation arithmetic
(``reference_field_gauss(run, dataset) * gauss_to_mhz(1.0)``) against a
verbatim copy of the removed panel logic — across a matrix that exercises the
critical dataset-before-run lookup order.
"""

from __future__ import annotations

import pytest

from asymmetry.core.data.dataset import MuonDataset, Run
from asymmetry.core.fourier.spectrum import reference_field_gauss
from asymmetry.core.fourier.units import gauss_to_mhz


def _legacy_core_reference_field_gauss(run, dataset) -> float | None:
    """Verbatim copy of the removed ``spectrum._reference_field_gauss``."""
    sources: list[dict] = []
    if dataset is not None and isinstance(dataset.metadata, dict):
        sources.append(dataset.metadata)
    run_metadata = getattr(run, "metadata", None)
    if isinstance(run_metadata, dict):
        sources.append(run_metadata)
    for metadata in sources:
        try:
            return float(metadata["field"])
        except (KeyError, TypeError, ValueError):
            continue
    return None


def _legacy_panel_reference_mhz(dataset) -> float | None:
    """Verbatim copy of the removed panel ``_frequency_reference_for_dataset``."""
    if dataset is None:
        return None
    field_value = dataset.metadata.get("field")
    try:
        field_gauss = float(field_value)
    except (TypeError, ValueError):
        run = getattr(dataset, "run", None)
        metadata = getattr(run, "metadata", {}) if run is not None else {}
        try:
            field_gauss = float(metadata.get("field"))
        except (TypeError, ValueError):
            return None
    return field_gauss * gauss_to_mhz(1.0)


def _delegated_panel_reference_mhz(dataset) -> float | None:
    """The new panel arithmetic, factored out of the Qt class for testing."""
    if dataset is None:
        return None
    field_gauss = reference_field_gauss(getattr(dataset, "run", None), dataset)
    if field_gauss is None:
        return None
    return field_gauss * gauss_to_mhz(1.0)


def _make_dataset(dataset_meta=None, run_meta=None, *, with_run=True) -> MuonDataset:
    run = Run(run_number=1, metadata=dict(run_meta) if run_meta is not None else {})
    return MuonDataset(
        time=[],
        asymmetry=[],
        error=[],
        metadata=dict(dataset_meta) if dataset_meta is not None else {},
        run=run if with_run else None,
    )


# (dataset_meta, run_meta, with_run) — covers each lookup branch.
_CASES = [
    ({"field": 200.0}, {"field": 100.0}, True),  # both → dataset wins
    ({}, {"field": 100.0}, True),  # dataset missing → run
    ({"field": "nan-ish"}, {"field": 50.0}, True),  # dataset non-numeric → run
    ({"field": None}, {"field": 75.0}, True),  # dataset explicit None → run
    ({"field": 0.0}, {"field": 100.0}, True),  # zero field is valid, dataset wins
    ({"field": 300.0}, {}, True),  # run missing → dataset
    ({}, {}, True),  # neither → None
    ({"field": "x"}, {"field": "y"}, True),  # both non-numeric → None
    ({"field": 120.0}, None, False),  # no run object → dataset
    ({}, None, False),  # no run, no dataset field → None
]


@pytest.mark.parametrize("dataset_meta, run_meta, with_run", _CASES)
def test_core_resolver_matches_legacy(dataset_meta, run_meta, with_run) -> None:
    dataset = _make_dataset(dataset_meta, run_meta, with_run=with_run)
    run = getattr(dataset, "run", None)
    assert reference_field_gauss(run, dataset) == _legacy_core_reference_field_gauss(run, dataset)


def test_core_resolver_dataset_none() -> None:
    run = Run(run_number=1, metadata={"field": 42.0})
    assert reference_field_gauss(run, None) == 42.0
    assert reference_field_gauss(None, None) is None


@pytest.mark.parametrize("dataset_meta, run_meta, with_run", _CASES)
def test_panel_delegation_matches_legacy(dataset_meta, run_meta, with_run) -> None:
    dataset = _make_dataset(dataset_meta, run_meta, with_run=with_run)
    new = _delegated_panel_reference_mhz(dataset)
    old = _legacy_panel_reference_mhz(dataset)
    assert (new is None and old is None) or new == old


def test_panel_delegation_dataset_none() -> None:
    assert _delegated_panel_reference_mhz(None) is None

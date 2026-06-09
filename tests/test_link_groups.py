"""Equality link groups (WiMDA "Ties") — core engine + serialization.

Uses a synthetic damped-cosine triplet (central line + two satellites) so the
tests never depend on the WiMDA corpus files, which are not in the repo. See
docs/porting/link-groups/ for the study and design.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitEngine
from asymmetry.core.fitting.parameters import Parameter, ParameterSet

# Reference triplet (see docs/porting/link-groups/test-data.md).
_F0 = 1.389  # central Larmor line (MHz)
_DELTA = 0.121  # half-splitting (MHz); full splitting 2*delta = 0.242 MHz
_LAMBDA = 0.30  # shared relaxation (us^-1)

_TRIPLET_EXPR = (
    "Oscillatory * Exponential + Oscillatory * Exponential + Oscillatory * Exponential + Constant"
)


def _triplet_model() -> CompositeModel:
    return CompositeModel.from_expression(_TRIPLET_EXPR)


def _triplet_dataset() -> MuonDataset:
    """Synthetic damped-cosine triplet at f0 and f0+-delta + flat background."""
    model = _triplet_model()
    truth = {
        "A_1": 10.0,
        "frequency_1": _F0,
        "phase_1": 0.0,
        "Lambda_2": _LAMBDA,
        "A_3": 6.0,
        "frequency_3": _F0 - _DELTA,
        "phase_3": 0.0,
        "Lambda_4": _LAMBDA,
        "A_5": 6.0,
        "frequency_5": _F0 + _DELTA,
        "phase_5": 0.0,
        "Lambda_6": _LAMBDA,
        "A_bg": 0.5,
    }
    t = np.linspace(0.0, 12.0, 600)
    rng = np.random.default_rng(0)
    err = np.full_like(t, 0.15)
    y = model.function(t, **truth) + rng.normal(0.0, 0.15, size=t.shape)
    return MuonDataset(time=t, asymmetry=y, error=err, metadata={"run_number": 1})


def _linked_parameter_set() -> ParameterSet:
    """Triplet seed with shared relaxation (g1), satellite amps (g2), phases (g3)."""
    model = _triplet_model()
    seed = {
        "A_1": 10.0,
        "frequency_1": 1.40,
        "phase_1": 0.0,
        "Lambda_2": _LAMBDA,
        "A_3": 6.0,
        "frequency_3": 1.25,
        "phase_3": 0.0,
        "Lambda_4": _LAMBDA,
        "A_5": 6.0,
        "frequency_5": 1.55,
        "phase_5": 0.0,
        "Lambda_6": _LAMBDA,
        "A_bg": 0.5,
    }
    link = {
        "Lambda_2": 1,
        "Lambda_4": 1,
        "Lambda_6": 1,
        "A_3": 2,
        "A_5": 2,
        "phase_1": 3,
        "phase_3": 3,
        "phase_5": 3,
    }
    ps = ParameterSet()
    for name in model.param_names:
        ps.add(Parameter(name=name, value=seed[name], link_group=link.get(name)))
    return ps


# --- ParameterSet semantics -------------------------------------------------


def test_link_groups_identify_main_and_followers() -> None:
    ps = _linked_parameter_set()

    groups = ps.link_groups()
    assert set(groups) == {1, 2, 3}

    # The main of each group is its first non-fixed member.
    assert ps.link_main(1).name == "Lambda_2"
    assert ps.link_main(2).name == "A_3"
    assert ps.link_main(3).name == "phase_1"

    followers = ps.link_followers()
    assert followers == {
        "Lambda_4": "Lambda_2",
        "Lambda_6": "Lambda_2",
        "A_5": "A_3",
        "phase_3": "phase_1",
        "phase_5": "phase_1",
    }


def test_followers_drop_out_of_free_set() -> None:
    ps = _linked_parameter_set()
    free = {p.name for p in ps.free_parameters}

    # Five followers are gone; the three group mains remain free.
    assert "Lambda_4" not in free and "Lambda_6" not in free
    assert "A_5" not in free
    assert "phase_3" not in free and "phase_5" not in free
    assert {"Lambda_2", "A_3", "phase_1"} <= free
    assert len(free) == len(ps) - len(ps.link_followers())


def test_singleton_link_group_is_a_no_op() -> None:
    ps = ParameterSet(
        [
            Parameter(name="A", value=1.0, link_group=1),
            Parameter(name="B", value=2.0),
        ]
    )
    assert ps.link_groups() == {}
    assert ps.link_followers() == {}
    assert {p.name for p in ps.free_parameters} == {"A", "B"}


def test_main_prefers_a_free_member_over_a_fixed_one() -> None:
    ps = ParameterSet(
        [
            Parameter(name="A", value=1.0, fixed=True, link_group=1),
            Parameter(name="B", value=2.0, link_group=1),
        ]
    )
    # A is fixed, so B (free) becomes the main.
    assert ps.link_main(1).name == "B"
    assert ps.link_followers() == {"A": "B"}


# --- Engine behaviour -------------------------------------------------------


def test_linked_fit_enforces_equality_and_recovers_splitting() -> None:
    ds = _triplet_dataset()
    model = _triplet_model()
    ps = _linked_parameter_set()

    result = FitEngine().fit(ds, model.function, ps, t_min=0.05, t_max=12.0)
    assert result.success
    assert result.reduced_chi_squared == pytest.approx(1.0, abs=0.25)

    fitted = {p.name: p.value for p in result.parameters}

    # Equality links: every follower exactly equals its group main.
    assert fitted["Lambda_4"] == fitted["Lambda_2"]
    assert fitted["Lambda_6"] == fitted["Lambda_2"]
    assert fitted["A_5"] == fitted["A_3"]
    assert fitted["phase_3"] == fitted["phase_1"]
    assert fitted["phase_5"] == fitted["phase_1"]

    # Propagated uncertainty: a follower reports its main's sigma.
    assert "Lambda_4" in result.uncertainties
    assert result.uncertainties["Lambda_4"] == result.uncertainties["Lambda_2"]
    assert result.uncertainties["A_5"] == result.uncertainties["A_3"]

    # Free frequencies recover the symmetric triplet and the hyperfine splitting.
    assert fitted["frequency_1"] == pytest.approx(_F0, abs=0.02)
    splitting = fitted["frequency_5"] - fitted["frequency_3"]
    assert splitting == pytest.approx(2 * _DELTA, abs=0.02)
    centre = 0.5 * (fitted["frequency_3"] + fitted["frequency_5"])
    assert centre == pytest.approx(fitted["frequency_1"], abs=0.02)


def test_linked_fit_reduces_free_parameter_count() -> None:
    ds = _triplet_dataset()
    model = _triplet_model()

    unlinked = ParameterSet()
    linked = _linked_parameter_set()
    for p in linked:
        unlinked.add(Parameter(name=p.name, value=p.value))

    res_unlinked = FitEngine().fit(ds, model.function, unlinked, t_min=0.05, t_max=12.0)
    res_linked = FitEngine().fit(ds, model.function, linked, t_min=0.05, t_max=12.0)

    # Followers are not fitted, so they carry no entry in the covariance order.
    assert "Lambda_4" not in res_linked.covariance_parameters
    assert "Lambda_4" in res_unlinked.covariance_parameters
    assert len(res_linked.covariance_parameters) == len(res_unlinked.covariance_parameters) - len(
        linked.link_followers()
    )


def test_link_group_survives_when_main_is_fixed() -> None:
    """If every member is fixed, the group is constant and never crashes the fit."""
    ds = _triplet_dataset()
    model = _triplet_model()
    ps = _linked_parameter_set()
    # Fix the whole relaxation group at the known value.
    for name in ("Lambda_2", "Lambda_4", "Lambda_6"):
        ps[name].fixed = True
        ps[name].value = _LAMBDA

    result = FitEngine().fit(ds, model.function, ps, t_min=0.05, t_max=12.0)
    assert result.success
    fitted = {p.name: p.value for p in result.parameters}
    assert fitted["Lambda_2"] == fitted["Lambda_4"] == fitted["Lambda_6"] == _LAMBDA
    # A fixed group reports no uncertainty for its members.
    assert "Lambda_4" not in result.uncertainties


# --- .asymp persistence -----------------------------------------------------


def test_link_group_survives_fit_slot_round_trip() -> None:
    """A FitSlot's parameter link groups round-trip through (de)serialization."""
    from asymmetry.core.representation.base import FitSlot

    slot = FitSlot(
        model=_triplet_model().to_dict(),
        parameters=[
            {"name": "Lambda_2", "value": 0.30, "link_group": 1},
            {"name": "Lambda_4", "value": 0.30, "link_group": 1},
            {"name": "frequency_1", "value": 1.389, "link_group": None},
        ],
        provenance="single",
    )

    import json

    restored = FitSlot.from_dict(json.loads(json.dumps(slot.to_dict())))
    by_name = {p["name"]: p for p in restored.parameters}
    assert by_name["Lambda_2"]["link_group"] == 1
    assert by_name["Lambda_4"]["link_group"] == 1
    assert by_name["frequency_1"]["link_group"] is None

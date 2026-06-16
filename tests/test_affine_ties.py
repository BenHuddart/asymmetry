"""Affine parameter ties (offset/equal-spacing) — core engine + serialization.

An equality link group (WiMDA "Ties") can only force ``follower == main``. It
cannot express the *equal spacing* of muonium satellites about a central line —
``f_lo = f_c - delta``, ``f_hi = f_c + delta`` with the half-splitting ``delta``
left free. That fit shape was surfaced by the session-5 CdS run (the satellite
amplitudes scatter when all three frequencies float independently, so the
Arrhenius ionisation energy is un-extractable). Affine ties are the beyond-WiMDA
capability that closes it: a follower is a linear map ``scale*main +
offset_scale*offset + const`` of other parameters, where ``offset`` may be a free
*auxiliary* parameter the model itself does not consume.

Uses a synthetic damped-cosine triplet (central line + two satellites) so the
tests never depend on the WiMDA corpus. See docs/porting/link-groups/ for the
study and the final decision recorded there.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitEngine
from asymmetry.core.fitting.parameters import AffineTie, Parameter, ParameterSet

# Reference triplet (mirrors docs/porting/link-groups/test-data.md).
_F0 = 1.389  # central Larmor line (MHz)
_DELTA = 0.121  # half-splitting (MHz); full splitting 2*delta = 0.242 MHz = hyperfine const
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


def _equal_spacing_parameter_set() -> ParameterSet:
    """Triplet seed with the two satellites tied to f_centre +- delta (delta free).

    The two satellite frequencies are *not* free: they are derived from the
    central frequency and a single free half-splitting parameter ``delta`` that
    the model never sees. Shared relaxation + amplitude + phase still ride on
    equality link groups, exactly as a real CdS fit would.
    """
    model = _triplet_model()
    seed = {
        "A_1": 10.0,
        "frequency_1": 1.40,
        "phase_1": 0.0,
        "Lambda_2": _LAMBDA,
        "A_3": 6.0,
        "frequency_3": 1.25,  # overwritten by the tie at fit time
        "phase_3": 0.0,
        "Lambda_4": _LAMBDA,
        "A_5": 6.0,
        "frequency_5": 1.55,  # overwritten by the tie at fit time
        "phase_5": 0.0,
        "Lambda_6": _LAMBDA,
        "A_bg": 0.5,
    }
    # Equality link groups (shareable params) — frequencies handled by ties below.
    link = {
        "Lambda_2": 1,
        "Lambda_4": 1,
        "Lambda_6": 1,
        "A_3": 2,
        "A_5": 2,
        "phase_3": 3,
        "phase_5": 3,
    }
    ties = {
        "frequency_3": AffineTie(main="frequency_1", offset="delta", offset_scale=-1.0),
        "frequency_5": AffineTie(main="frequency_1", offset="delta", offset_scale=+1.0),
    }
    ps = ParameterSet()
    for name in model.param_names:
        ps.add(
            Parameter(name=name, value=seed[name], link_group=link.get(name), tie=ties.get(name))
        )
    # Auxiliary free half-splitting (not a model parameter — the model ignores it).
    ps.add(Parameter(name="delta", value=0.10, min=0.0))
    return ps


# --- ParameterSet semantics -------------------------------------------------


def test_affine_tie_followers_and_free_set() -> None:
    ps = _equal_spacing_parameter_set()

    followers = ps.tie_followers()
    assert set(followers) == {"frequency_3", "frequency_5"}
    assert followers["frequency_3"].main == "frequency_1"
    assert followers["frequency_3"].offset == "delta"
    assert followers["frequency_3"].offset_scale == -1.0

    free = {p.name for p in ps.free_parameters}
    # The tied satellite frequencies drop out; the free half-splitting stays in.
    assert "frequency_3" not in free and "frequency_5" not in free
    assert "delta" in free
    assert "frequency_1" in free


def test_affine_tie_is_constrained() -> None:
    p = Parameter(name="frequency_3", value=1.25, tie=AffineTie(main="frequency_1", offset="delta"))
    assert p.is_constrained


# --- Engine behaviour -------------------------------------------------------


def test_equal_spacing_fit_enforces_symmetry_and_recovers_splitting() -> None:
    ds = _triplet_dataset()
    model = _triplet_model()
    ps = _equal_spacing_parameter_set()

    result = FitEngine().fit(ds, model.function, ps, t_min=0.05, t_max=12.0)
    assert result.success
    assert result.reduced_chi_squared == pytest.approx(1.0, abs=0.25)

    fitted = {p.name: p.value for p in result.parameters}

    # Symmetry is *enforced*, not merely recovered: satellites are exactly
    # f_centre +- delta to floating-point precision.
    assert fitted["frequency_3"] == pytest.approx(fitted["frequency_1"] - fitted["delta"], abs=1e-9)
    assert fitted["frequency_5"] == pytest.approx(fitted["frequency_1"] + fitted["delta"], abs=1e-9)

    # The free half-splitting recovers the truth; full splitting 2*delta = hyperfine.
    assert fitted["delta"] == pytest.approx(_DELTA, abs=0.02)
    assert fitted["frequency_1"] == pytest.approx(_F0, abs=0.02)

    # delta is a genuine fitted parameter, so it carries an uncertainty; the
    # tied followers report their propagated (delta-method) uncertainty.
    assert "delta" in result.uncertainties
    assert result.uncertainties["delta"] > 0.0
    assert "frequency_3" in result.uncertainties
    assert result.uncertainties["frequency_3"] > 0.0


def test_equal_spacing_stabilises_satellite_amplitudes() -> None:
    """The point of the feature: tying the frequencies stabilises the amplitudes.

    With three free frequencies the satellite amplitudes trade against the
    frequencies and scatter; tying them to f_centre +- delta recovers the true
    amplitude cleanly (this is what makes the CdS Arrhenius E_i extractable).
    """
    ds = _triplet_dataset()
    model = _triplet_model()
    ps = _equal_spacing_parameter_set()

    result = FitEngine().fit(ds, model.function, ps, t_min=0.05, t_max=12.0)
    fitted = {p.name: p.value for p in result.parameters}
    assert fitted["A_3"] == pytest.approx(6.0, abs=0.6)
    assert fitted["A_5"] == fitted["A_3"]  # equality link still holds


def test_constant_offset_tie() -> None:
    """A tie with a constant offset (no auxiliary param) pins a known splitting."""
    ds = _triplet_dataset()
    model = _triplet_model()
    ps = _equal_spacing_parameter_set()
    # Replace the free-delta ties with a fixed +-_DELTA constant offset.
    ps["frequency_3"].tie = AffineTie(main="frequency_1", const=-_DELTA)
    ps["frequency_5"].tie = AffineTie(main="frequency_1", const=+_DELTA)
    ps["delta"].fixed = True  # no longer used

    result = FitEngine().fit(ds, model.function, ps, t_min=0.05, t_max=12.0)
    assert result.success
    fitted = {p.name: p.value for p in result.parameters}
    assert fitted["frequency_3"] == pytest.approx(fitted["frequency_1"] - _DELTA, abs=1e-9)
    assert fitted["frequency_5"] == pytest.approx(fitted["frequency_1"] + _DELTA, abs=1e-9)


def test_tie_reduces_free_parameter_count() -> None:
    ds = _triplet_dataset()
    model = _triplet_model()
    tied = _equal_spacing_parameter_set()

    res_tied = FitEngine().fit(ds, model.function, tied, t_min=0.05, t_max=12.0)
    # The two tied satellite frequencies are not in the covariance order; the
    # auxiliary delta is.
    assert "frequency_3" not in res_tied.covariance_parameters
    assert "frequency_5" not in res_tied.covariance_parameters
    assert "delta" in res_tied.covariance_parameters


# --- robustness / validation ------------------------------------------------


def test_auxiliary_param_not_forwarded_to_explicit_signature_model() -> None:
    """A free auxiliary tie param must not crash a model with a strict signature.

    ``CompositeModel.function`` takes ``**kwargs`` and ignores extras, but a
    plain model function with an explicit signature would raise ``TypeError`` on
    the unexpected auxiliary kwarg; the engine must strip it.
    """

    def line(t, slope, intercept):  # explicit signature — no **kwargs
        return slope * t + intercept

    t = np.linspace(0.0, 10.0, 200)
    rng = np.random.default_rng(0)
    ds = MuonDataset(
        time=t,
        asymmetry=2.0 * t + 1.0 + rng.normal(0.0, 0.05, size=t.shape),
        error=np.full_like(t, 0.05),
        metadata={"run_number": 1},
    )
    ps = ParameterSet(
        [
            Parameter("slope", 1.0),
            # intercept tracks the free auxiliary `delta`, which `line` never accepts.
            Parameter("intercept", 0.0, tie=AffineTie(main="delta")),
            Parameter("delta", 0.0),
        ]
    )
    result = FitEngine().fit(ds, line, ps, t_min=0.0, t_max=10.0)
    assert result.success
    fitted = {p.name: p.value for p in result.parameters}
    assert fitted["slope"] == pytest.approx(2.0, abs=0.05)
    assert fitted["intercept"] == pytest.approx(fitted["delta"], abs=1e-9)
    assert fitted["delta"] == pytest.approx(1.0, abs=0.1)


def test_fixed_and_tied_is_rejected() -> None:
    ps = ParameterSet(
        [
            Parameter("frequency_1", 1.39),
            Parameter(
                "frequency_3", 1.27, fixed=True, tie=AffineTie(main="frequency_1", const=-0.1)
            ),
        ]
    )
    ds = _triplet_dataset()
    with pytest.raises(ValueError, match="both fixed and affinely tied"):
        FitEngine().fit(ds, _triplet_model().function, ps, t_min=0.05, t_max=12.0)


def test_link_group_and_tie_is_rejected() -> None:
    ps = ParameterSet(
        [
            Parameter("frequency_1", 1.39),
            Parameter("frequency_3", 1.27, link_group=1, tie=AffineTie(main="frequency_1")),
            Parameter("frequency_5", 1.51, link_group=1),
        ]
    )
    ds = _triplet_dataset()
    with pytest.raises(ValueError, match="both link-grouped and affinely tied"):
        FitEngine().fit(ds, _triplet_model().function, ps, t_min=0.05, t_max=12.0)


def test_global_fit_rejects_ties_loudly() -> None:
    """Ties are honoured only by single fit(); global_fit must not silently ignore them."""
    ds = _triplet_dataset()
    ps = _equal_spacing_parameter_set()
    with pytest.raises(NotImplementedError, match="affine parameter ties"):
        FitEngine().global_fit(
            datasets=[ds],
            model_fn=_triplet_model().function,
            global_params=["frequency_1"],
            local_params=[],
            initial_params={1: ps},
            t_min=0.05,
            t_max=12.0,
        )


# --- .asymp persistence -----------------------------------------------------


def test_affine_tie_round_trips_through_dict() -> None:
    tie = AffineTie(main="frequency_1", scale=1.0, offset="delta", offset_scale=-1.0, const=0.0)
    restored = AffineTie.from_dict(json.loads(json.dumps(tie.to_dict())))
    assert restored == tie


def test_affine_tie_survives_fit_slot_round_trip() -> None:
    from asymmetry.core.representation.base import FitSlot

    slot = FitSlot(
        model=_triplet_model().to_dict(),
        parameters=[
            {"name": "frequency_1", "value": 1.389, "tie": None},
            {
                "name": "frequency_3",
                "value": 1.268,
                "tie": AffineTie(main="frequency_1", offset="delta", offset_scale=-1.0).to_dict(),
            },
            {"name": "delta", "value": 0.121},
        ],
        provenance="single",
    )
    restored = FitSlot.from_dict(json.loads(json.dumps(slot.to_dict())))
    by_name = {p["name"]: p for p in restored.parameters}
    assert by_name["frequency_1"]["tie"] is None
    tie = AffineTie.from_dict(by_name["frequency_3"]["tie"])
    assert tie.main == "frequency_1"
    assert tie.offset == "delta"
    assert tie.offset_scale == -1.0

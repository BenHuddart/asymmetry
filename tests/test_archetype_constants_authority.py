"""Pin the shared archetype-constant authority (reconciliation F12).

``core.simulate_presets`` is the single public authority for the rounded
textbook constants shared with ``docs/screenshots/data/archetypes.py``. This
asserts the docs module re-exports the *same object* from core (not a
re-typed literal that could drift), that the values match the textbook
numbers (so screenshot data stay byte-stable), and that the docs-only
constants — including the legacy 2.197 µs lifetime pin — remain local.

These rounded constants are deliberately distinct from the CODATA values in
``core.utils.constants``; the unification is within the rounded table only.
"""

from __future__ import annotations

from asymmetry.core import simulate_presets
from asymmetry.core.utils import constants as codata
from docs.screenshots.data import archetypes

#: The shared subset and its pinned values.
_SHARED = {
    "GAMMA_MU_MHZ_PER_G": 0.01355,
    "DELTA_AG_PER_US": 0.39,
    "TC_EUO_K": 69.0,
    "R_MUF_ANG": 1.17,
}


def test_core_is_the_authority_with_pinned_values() -> None:
    for name, value in _SHARED.items():
        assert getattr(simulate_presets, name) == value


def test_docs_module_reexports_the_same_objects() -> None:
    """The docs constants are the very objects from core, not re-typed copies."""
    for name in _SHARED:
        assert getattr(archetypes, name) is getattr(simulate_presets, name)


def test_docs_only_constants_stay_local() -> None:
    # Present in the docs module, absent from the core authority.
    for name in ("GAMMA_MU_MHZ_PER_T", "TC_MGB2_K", "TC_YBCO_K", "LAMBDA_YBCO_NM", "XI_YBCO_NM"):
        assert hasattr(archetypes, name)
        assert not hasattr(simulate_presets, name)
    # The legacy lifetime pin that keeps the screenshots byte-stable stays local
    # and distinct from the canonical core lifetime.
    assert archetypes.MUON_LIFETIME_US == 2.197
    assert not hasattr(simulate_presets, "MUON_LIFETIME_US")


def test_rounded_table_distinct_from_codata() -> None:
    """The rounded γ_μ is intentionally not the CODATA constant."""
    assert simulate_presets.GAMMA_MU_MHZ_PER_G != codata.MUON_GYROMAGNETIC_RATIO_MHZ_PER_T / 1000.0
    assert codata.MUON_GYROMAGNETIC_RATIO_MHZ_PER_T == 135.538817

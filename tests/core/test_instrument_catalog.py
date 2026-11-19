"""Bundled profile membership has one deterministic installed-package authority."""

import importlib

from asymmetry.core import instruments


def test_catalog_discovers_all_bundled_declarations_and_stable_ids():
    from asymmetry.core.instruments.catalog import PROFILES, bundled_profiles, profile_by_key

    keys = [profile.identity.key for profile in PROFILES]
    assert len(keys) == len(set(keys))
    assert {"isis.emu", "isis.hifi", "isis.musr", "psi.gps", "psi.hal9500", "psi.ltf"} <= set(keys)
    assert tuple(bundled_profiles()) == PROFILES
    assert profile_by_key("psi.gps").identity.display_name == "GPS"


def test_a_bundled_profile_file_requires_no_second_membership_list(tmp_path, monkeypatch):
    from asymmetry.core.instruments.catalog import bundled_profiles

    (tmp_path / "fixture_added.py").write_text(
        "from tests._instrument_profiles import synthetic_profile\nPROFILE = synthetic_profile()\n"
    )
    monkeypatch.setattr(instruments, "__path__", [*instruments.__path__, str(tmp_path)])
    importlib.invalidate_caches()
    assert "test.fixture" in {profile.identity.key for profile in bundled_profiles()}

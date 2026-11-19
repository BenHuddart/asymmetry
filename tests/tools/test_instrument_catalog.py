"""Layout and expert-review registration share the bundled catalog."""


def test_layout_and_review_membership_come_from_the_same_catalog():
    from asymmetry.core.instrument import INSTRUMENT_NAMES, get_instrument_layout
    from asymmetry.core.instruments.catalog import PROFILES
    from tools.instrument_docs import PROFILES as DOCUMENTED

    assert set(DOCUMENTED) == {profile.identity.key for profile in PROFILES}
    expected = {
        config.layout_key or profile.identity.layout_key
        for profile in PROFILES
        for config in profile.detectors.acquisitions
    }
    assert set(INSTRUMENT_NAMES) == expected
    assert all(get_instrument_layout(key).name == key for key in expected)

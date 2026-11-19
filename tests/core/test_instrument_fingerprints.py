"""Grouping attachment uses canonical identity/configuration, not displayed labels."""

from asymmetry.core.project.profiles import ProfileFingerprint


def test_canonical_configuration_is_part_of_grouping_attachment():
    combined = ProfileFingerprint("psi.gps", 6, "bin_combined")
    split = ProfileFingerprint("psi.gps", 6, "root_split")
    assert not combined.matches(split)
    assert combined.matches(ProfileFingerprint.from_dict(combined.to_dict()))
    assert combined.to_dict() == {
        "instrument": "psi.gps",
        "histogram_count": 6,
        "acquisition": "bin_combined",
    }


def test_opaque_unresolved_owners_are_not_equal_to_a_shared_display_label():
    first = ProfileFingerprint("", 8, None, "Unlisted A")
    second = ProfileFingerprint("", 8, None, "Unlisted B")
    assert not first.matches(second)
    assert first.matches(ProfileFingerprint.from_dict(first.to_dict()))
    assert not ProfileFingerprint("test.one", 8).matches(ProfileFingerprint("test.two", 8))

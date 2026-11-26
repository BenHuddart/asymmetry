"""ISIS review pages expose the same declared maps and their scientific limits."""

from xml.etree import ElementTree

import pytest

from asymmetry.core.instruments.isis_hifi import PROFILE as HIFI
from asymmetry.core.instruments.isis_musr import PROFILE as MUSR
from tools.instrument_docs import PROFILES, generate_profile


@pytest.mark.parametrize("profile", [HIFI, MUSR])
def test_isis_review_page_and_labelled_diagram_disclose_unverified_mapping(tmp_path, profile):
    assert PROFILES[profile.identity.key] is profile
    page = generate_profile(profile, tmp_path)
    text = page.read_text()
    assert "No instrument-scientist review recorded" in text
    assert "Beamline: **Unknown (unverified)**" in text
    assert "exact export wiring has not been independently established" in text
    assert "initial alpha=1 are Asymmetry conventions" in text
    assert "Unconditional default-preset application on fresh load: **no**" in text
    image = page.with_name(page.stem + "-nexus64.svg")
    labels = list(ElementTree.parse(image).getroot().itertext())
    assert all(str(number) in labels for number in range(1, 65))
    legacy_grouping = (
        "Matching single-period NeXus V1 **nexus64** data with an "
        "all-zero grouping table and **Longitudinal** detector orientation "
        "use the **Longitudinal** preset."
    )
    if profile is HIFI:
        assert legacy_grouping in text
        assert "Half-sector angular offset is inferred" in text
        assert "HiFi-User-Manual.pdf" in text
    else:
        assert "all-zero grouping table" not in text
        assert "Forward=33–64 and backward=1–32" in text
        assert "Instrument rotation and applied-field orientation remain run-specific" in text

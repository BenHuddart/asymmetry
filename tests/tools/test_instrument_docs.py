"""Review output is generated from the same resolved profiles as the real display."""

from dataclasses import replace
from xml.etree import ElementTree

from asymmetry.core.instruments.isis_emu import PROFILE
from tests._instrument_profiles import synthetic_profile, synthetic_profiles
from tools.instrument_docs import generate_profile, profile_rst, render_profile_diagram


def test_emu_review_is_deterministic_and_discloses_scientific_limits(tmp_path):
    first = generate_profile(PROFILE, tmp_path / "first")
    second = generate_profile(PROFILE, tmp_path / "second")
    assert first.read_text() == second.read_text()
    first_svg = first.with_name("isis_emu-nexus96.svg")
    second_svg = second.with_name("isis_emu-nexus96.svg")
    assert first_svg.read_bytes() == second_svg.read_bytes()
    text = first.read_text()
    assert "No instrument-scientist review" in text
    assert "Asymmetry-defined" in text
    assert "not a facility-validated quantitative vector measurement" in text
    assert "normalized concentric radii are drawing choices" in text
    assert "Unverified" in text
    assert "nexus96" in text
    assert "EMU_Manual.pdf" in text
    assert ElementTree.parse(first_svg).getroot().tag.endswith("svg")


def test_consent_is_required_to_publish_a_recorded_name():
    review = replace(PROFILE.review, reviewed=True, reviewer="Example reviewer", name_consent=False)
    hidden = replace(PROFILE, review=review)
    assert "Example reviewer" not in profile_rst(hidden, {})
    visible = replace(hidden, review=replace(review, name_consent=True))
    assert "Example reviewer" in profile_rst(visible, {})


def test_all_synthetic_configurations_exercise_actual_diagram_and_page_generation(tmp_path):
    for name, profile in synthetic_profiles().items():
        page = generate_profile(profile, tmp_path / name)
        text = page.read_text()
        for config in profile.detectors.acquisitions:
            assert config.key in text
            diagram = page.with_name(f"test_fixture-{config.key}.svg")
            root = ElementTree.parse(diagram).getroot()
            labels = " ".join(root.itertext())
            assert all(view.label in labels for view in config.views)
        if name == "combined_split":
            assert "Unknown" in text
            assert "No count equation or independence" in text


def test_metadata_only_review_does_not_invent_geometry(tmp_path):
    profile = synthetic_profile()
    text = generate_profile(profile, tmp_path).read_text()
    assert "No acquisition map or geometry is declared" in text
    assert list(tmp_path.glob("*.svg")) == []
    generic = tmp_path / "explicitly-generic.svg"
    render_profile_diagram(profile, None, generic, n_histograms=3)
    assert ElementTree.parse(generic).getroot().tag.endswith("svg")


def test_incremental_sphinx_rebuild_tracks_profile_and_generator_sources(tmp_path, monkeypatch):
    import io
    import os
    from pathlib import Path

    from sphinx.application import Sphinx

    from asymmetry.core.instruments import isis_emu
    from tools import instrument_docs

    source = tmp_path / "docs"
    source.mkdir()
    (source / "conf.py").write_text(
        "extensions = ['tools.instrument_docs', 'sphinx_design']\nproject = 'Profile fixture'\n"
    )
    (source / "index.rst").write_text(".. instrument-profile:: isis.emu\n")
    generator = tmp_path / "generator.py"
    generator.write_text("# Initial generator dependency\n")
    monkeypatch.setattr(instrument_docs, "__file__", str(generator))
    output, doctrees = tmp_path / "html", tmp_path / "doctrees"
    first = Sphinx(
        str(source),
        str(source),
        str(output),
        str(doctrees),
        "html",
        status=io.StringIO(),
        warning=io.StringIO(),
    )
    first.build()
    assert first.statuscode == 0
    dependencies = {Path(path).resolve() for path in first.env.dependencies["index"]}
    assert Path(isis_emu.__file__).resolve() in dependencies
    assert generator.resolve() in dependencies
    assert "Changed fixture title" not in (output / "index.html").read_text()
    changed = replace(PROFILE, identity=replace(PROFILE.identity, name="Changed fixture title"))
    monkeypatch.setitem(instrument_docs.PROFILES, PROFILE.identity.key, changed)
    modified = max(generator.stat().st_mtime, first.env.all_docs["index"]) + 2
    os.utime(generator, (modified, modified))
    second = Sphinx(
        str(source),
        str(source),
        str(output),
        str(doctrees),
        "html",
        status=io.StringIO(),
        warning=io.StringIO(),
    )
    second.build()
    assert second.statuscode == 0
    assert "Changed fixture title" in (output / "index.html").read_text()


def test_unknown_beamline_is_explicit_without_inventing_an_identity():
    assert PROFILE.identity.beamline is None
    assert "Beamline: **Unknown (unverified)**" in profile_rst(PROFILE, {})

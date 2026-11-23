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
    assert "Basename token boundaries; run digits may follow" in text
    assert "Compact substring" in text
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
    from sphinx.util.docutils import docutils_namespace

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
    with docutils_namespace():
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
    with docutils_namespace():
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


def test_psi_review_preserves_legacy_variants_and_declared_direction_cues(tmp_path):
    from asymmetry.core.instruments import FLAME, GPS, HAL9500

    gps = generate_profile(GPS, tmp_path / "gps").read_text()
    assert "Legacy layout key: ``GPS-RD``; displayed instrument: **GPS**" in gps
    assert "Legacy layout key: ``GPS-RD15``; displayed instrument: **GPS**" in gps
    assert "not an exact count equation or independence claim" in gps
    assert "initial projection balances" in gps
    flame = generate_profile(FLAME, tmp_path / "flame")
    text = flame.read_text()
    assert "beam, main magnetic field" in text
    assert "(-2.55, -0.82)" in text
    assert "not measured run-specific" in text
    svg = flame.with_name("psi_flame-eight_channel.svg")
    labels = " ".join(ElementTree.parse(svg).getroot().itertext())
    assert "initial muon spin" in labels
    assert "beam, main magnetic field" in labels
    assert "HIFI, HAL9500" in profile_rst(HAL9500, {})
    assert "HAL" in profile_rst(HAL9500, {})


def test_shared_psi_assumptions_are_summarised_once_with_complete_fact_evidence():
    from asymmetry.core.instruments import GPS
    from asymmetry.core.instruments.schema import EvidenceStatus

    text = profile_rst(GPS, {})
    summary = text.split("Key assumptions\n", 1)[1].split("\nSource\n", 1)[0]
    acquisition_detail = "Preserved ordinal/label map; combined/split roads"
    preset_detail = "Explicit compatibility memberships and leg order."
    assert summary.count(acquisition_detail) == 1
    assert summary.count(preset_detail) == 1
    assert text.count(acquisition_detail) == 4
    assert text.count(preset_detail) == 13
    for fact in GPS.fact_keys:
        assert fact in text
    evidence = tuple(
        replace(item, status=EvidenceStatus.AUTHORED)
        if item.fact == "acquisition.bin_combined"
        else item
        for item in GPS.sources.evidence
    )
    qualified = replace(GPS, sources=replace(GPS.sources, evidence=evidence))
    summary = profile_rst(qualified, {}).split("Key assumptions\n", 1)[1].split("\nSource\n", 1)[0]
    assert summary.count(acquisition_detail) == 2


def test_musr_page_describes_its_conditional_absent_table_default():
    from asymmetry.core.instruments.isis_musr import PROFILE as MUSR

    text = profile_rst(MUSR, {})
    assert "absent file grouping table on fresh **nexus64**" in text
    assert "explicit **Longitudinal** detector orientation" in text
    assert "Present file tables and saved requests retain precedence" in text


def test_declared_beam_review_uses_resolved_view_projection_and_retains_unknown_sense():
    from asymmetry.core.instruments.geometry import BeamDirection, ViewOrientation

    profile = synthetic_profiles()["multiview"]
    original = profile.detectors.acquisitions[0]
    views = (
        replace(original.views[0], orientation=ViewOrientation((1, 0, 0), (0, 1, 0))),
        replace(original.views[1], orientation=ViewOrientation((1, 0, 0), (0, 1, 0), (0, 0, -1))),
    )
    acquisition = replace(original, beam=BeamDirection((0, 0, 1)), views=views)
    declared = replace(profile, detectors=replace(profile.detectors, acquisitions=(acquisition,)))
    text = profile_rst(declared, {})
    assert "common arrangement frame: ``(0.0, 0.0, 1.0)``" in text
    assert "Beam screen components" in text
    assert "Beam toward component" in text
    assert "Unknown viewing sense remains unknown" in text
    assert "     - (0.0, 0.0)\n     - Unknown" in text
    assert "     - (0.0, 0.0)\n     - -1.0" in text
    assert "common arrangement frame" not in profile_rst(profile, {})

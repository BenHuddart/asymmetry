"""Bundled declarations reach stable review URLs through ordinary Sphinx builds."""

import importlib
import io
import os
from pathlib import Path

import pytest
from sphinx.application import Sphinx
from sphinx.errors import SphinxError
from sphinx.util.docutils import docutils_namespace

from asymmetry.core import instruments
from asymmetry.core.instruments.catalog import bundled_profiles
from tools import instrument_docs


@pytest.fixture
def review_site(tmp_path, monkeypatch):
    package = tmp_path / "profiles"
    package.mkdir()
    (package / "__init__.py").write_text("")
    monkeypatch.setattr(instruments, "__path__", [str(package)])
    monkeypatch.setattr(instruments, "__file__", str(package / "__init__.py"))
    source = tmp_path / "docs"
    directory = source / "reference" / "instruments"
    directory.mkdir(parents=True)
    (source / "conf.py").write_text(
        "extensions = ['tools.instrument_docs', 'sphinx_design']\n"
        "project = 'Review fixture'\n"
        "instrument_review_directory = 'reference/instruments'\n"
        "exclude_patterns = ['_generated']\n"
    )
    (source / "index.rst").write_text(
        "Fixture\n=======\n\n.. toctree::\n\n   reference/instruments/index\n"
    )
    (directory / "index.rst").write_text(
        "Definitions\n===========\n\n.. toctree::\n   :glob:\n\n   *_*\n"
    )
    return source, package


def bundled_profile_file(package, module, name):
    path = package / f"fixture_{module}.py"
    path.write_text(
        "from dataclasses import replace\n"
        "from tests._instrument_profiles import synthetic_profile\n"
        "base = synthetic_profile()\n"
        f"PROFILE = replace(base, identity=replace(base.identity, key='test.{module}', "
        f"name={name!r}, layout_key={name!r}, display_name={name!r}))\n"
    )
    importlib.invalidate_caches()
    return path


def build_reviews(source):
    status, warnings = io.StringIO(), io.StringIO()
    with docutils_namespace():
        app = Sphinx(
            str(source),
            str(source),
            str(source.parent / "html"),
            str(source.parent / "doctrees"),
            "html",
            status=status,
            warning=warnings,
            warningiserror=True,
        )
        app.build()
        assert app.statuscode == 0, status.getvalue() + warnings.getvalue()
    return app


def test_bundled_addition_and_removal_need_no_page_or_index_registration(review_site, monkeypatch):
    source, package = review_site
    bundled_profile_file(package, "alpha", "Alpha fixture")
    monkeypatch.setattr(
        instrument_docs, "PROFILES", {p.identity.key: p for p in bundled_profiles()}
    )
    directory = source / "reference/instruments"
    authored = directory / "authored_notes.rst"
    authored.write_text("Authored notes\n==============\n\nRetain this authored explanation.\n")
    first = build_reviews(source)
    page = directory / "test_alpha.rst"
    assert (
        "test_alpha.html" in (source.parent / "html/reference/instruments/index.html").read_text()
    )
    assert (
        "Alpha fixture"
        in (source.parent / "html/reference/instruments/test_alpha.html").read_text()
    )
    assert "reference/instruments/test_alpha" in first.env.found_docs
    assert (
        str(source.parent)
        not in (source.parent / "html/reference/instruments/test_alpha.html").read_text()
    )
    stamp = page.stat().st_mtime_ns
    build_reviews(source)
    assert page.stat().st_mtime_ns == stamp
    added = bundled_profile_file(package, "beta", "Beta fixture")
    monkeypatch.setattr(
        instrument_docs, "PROFILES", {p.identity.key: p for p in bundled_profiles()}
    )
    second = build_reviews(source)
    assert "reference/instruments/test_beta" in second.env.found_docs
    assert "test_beta.html" in (source.parent / "html/reference/instruments/index.html").read_text()
    assert (source.parent / "html/reference/instruments/test_beta.html").exists()
    assert page.stat().st_mtime_ns == stamp
    unrelated = source.parent / "html/reference/instruments/unrelated.html"
    unrelated.write_text("Retain unrelated output.")
    added.unlink()
    monkeypatch.setattr(
        instrument_docs, "PROFILES", {p.identity.key: p for p in bundled_profiles()}
    )
    third = build_reviews(source)
    assert "reference/instruments/test_beta" not in third.env.found_docs
    assert not (directory / "test_beta.rst").exists()
    assert not (source.parent / "html/reference/instruments/test_beta.html").exists()
    assert unrelated.read_text() == "Retain unrelated output."
    assert (source.parent / "html/reference/instruments/authored_notes.html").exists()
    assert (
        "test_beta.html"
        not in (source.parent / "html/reference/instruments/index.html").read_text()
    )
    assert (
        authored.read_text()
        == "Authored notes\n==============\n\nRetain this authored explanation.\n"
    )
    assert (
        "authored_notes.html"
        in (source.parent / "html/reference/instruments/index.html").read_text()
    )


def test_incremental_profile_and_generator_changes_refresh_registered_page(
    review_site, monkeypatch
):
    source, package = review_site
    module = bundled_profile_file(package, "changing", "Initial fixture")
    monkeypatch.setattr(
        instrument_docs, "PROFILES", {p.identity.key: p for p in bundled_profiles()}
    )
    generator = source.parent / "generator.py"
    generator.write_text("# Generator dependency\n")
    monkeypatch.setattr(instrument_docs, "__file__", str(generator))
    first = build_reviews(source)
    dependencies = {
        Path(p).resolve() for p in first.env.dependencies["reference/instruments/test_changing"]
    }
    assert module.resolve() in dependencies
    assert generator.resolve() in dependencies
    module.write_text(module.read_text().replace("Initial fixture", "Changed fixture"))
    newer = (
        max(module.stat().st_mtime, first.env.all_docs["reference/instruments/test_changing"]) + 2
    )
    os.utime(module, (newer, newer))
    importlib.invalidate_caches()
    importlib.reload(importlib.import_module("asymmetry.core.instruments.fixture_changing"))
    monkeypatch.setattr(
        instrument_docs, "PROFILES", {p.identity.key: p for p in bundled_profiles()}
    )
    second = build_reviews(source)
    output = source.parent / "html/reference/instruments/test_changing.html"
    assert "Changed fixture" in output.read_text()
    original = instrument_docs.profile_rst

    def changed_generator(profile, diagrams):
        return original(profile, diagrams) + "\nGenerator output refreshed.\n"

    monkeypatch.setattr(instrument_docs, "profile_rst", changed_generator)
    newer = (
        max(generator.stat().st_mtime, second.env.all_docs["reference/instruments/test_changing"])
        + 2
    )
    os.utime(generator, (newer, newer))
    build_reviews(source)
    assert "Generator output refreshed." in output.read_text()


def test_generation_refuses_to_replace_authored_source_at_a_profile_url(review_site, monkeypatch):
    source, package = review_site
    bundled_profile_file(package, "collision", "Collision fixture")
    monkeypatch.setattr(
        instrument_docs, "PROFILES", {p.identity.key: p for p in bundled_profiles()}
    )
    page = source / "reference/instruments/test_collision.rst"
    text = "Authored review\n===============\n\nAn authored page is never a generated stub.\n"
    page.write_text(text)
    with pytest.raises(SphinxError, match="authored instrument review"):
        build_reviews(source)
    assert page.read_text() == text


def test_generation_rejects_a_source_namespace_outside_the_site(review_site, monkeypatch):
    source, package = review_site
    bundled_profile_file(package, "bounded", "Bounded fixture")
    monkeypatch.setattr(
        instrument_docs, "PROFILES", {p.identity.key: p for p in bundled_profiles()}
    )
    outside = source.parent / "outside"
    outside.mkdir()
    page = outside / "retained.rst"
    text = instrument_docs.GENERATED_REVIEW_MARKER + "\n\nRetain outside source.\n"
    page.write_text(text)
    with (source / "conf.py").open("a") as stream:
        stream.write("instrument_review_directory = '../outside'\n")
    with pytest.raises(SphinxError, match="inside the source directory"):
        build_reviews(source)
    assert page.read_text() == text
    assert not (outside / "test_bounded.rst").exists()


def test_retired_html_cleanup_cannot_follow_a_link_outside_output(review_site, monkeypatch):
    source, package = review_site
    module = bundled_profile_file(package, "retired", "Retired fixture")
    monkeypatch.setattr(
        instrument_docs, "PROFILES", {p.identity.key: p for p in bundled_profiles()}
    )
    build_reviews(source)
    output = source.parent / "html/reference/instruments/test_retired.html"
    outside = source.parent / "authored.html"
    outside.write_text("Retain outside output.")
    output.unlink()
    output.symlink_to(outside)
    module.unlink()
    monkeypatch.setattr(
        instrument_docs, "PROFILES", {p.identity.key: p for p in bundled_profiles()}
    )
    with pytest.raises(SphinxError, match="inside the build directory"):
        build_reviews(source)
    assert outside.read_text() == "Retain outside output."
    assert output.is_symlink()


def test_retired_html_link_does_not_delete_an_authored_target_inside_output(
    review_site, monkeypatch
):
    source, package = review_site
    module = bundled_profile_file(package, "linked", "Linked fixture")
    bundled_profile_file(package, "remaining", "Remaining fixture")
    monkeypatch.setattr(
        instrument_docs, "PROFILES", {p.identity.key: p for p in bundled_profiles()}
    )
    build_reviews(source)
    output = source.parent / "html/reference/instruments/test_linked.html"
    authored = source.parent / "html/authored.html"
    authored.write_text("Retain authored build output.")
    output.unlink()
    output.symlink_to(authored)
    module.unlink()
    monkeypatch.setattr(
        instrument_docs, "PROFILES", {p.identity.key: p for p in bundled_profiles()}
    )
    build_reviews(source)
    assert authored.read_text() == "Retain authored build output."
    assert not output.is_symlink()
    assert not output.exists()


def test_wanted_source_alias_cannot_rewrite_another_owned_page(review_site, monkeypatch):
    source, package = review_site
    bundled_profile_file(package, "original", "Original fixture")
    bundled_profile_file(package, "target", "Target fixture")
    monkeypatch.setattr(
        instrument_docs, "PROFILES", {p.identity.key: p for p in bundled_profiles()}
    )
    build_reviews(source)
    directory = source / "reference/instruments"
    original, target = directory / "test_original.rst", directory / "test_target.rst"
    target_before = target.read_text()
    original.unlink()
    original.symlink_to(target)
    with pytest.raises(SphinxError, match="ordinary files"):
        build_reviews(source)
    assert original.is_symlink()
    assert target.read_text() == target_before


def test_new_profile_page_name_must_stay_local_to_the_review_directory(review_site, monkeypatch):
    from dataclasses import replace

    from tests._instrument_profiles import synthetic_profile

    source, _ = review_site
    key = str(source.parent / "escaped.fixture")
    base = synthetic_profile()
    profile = replace(base, identity=replace(base.identity, key=key))
    monkeypatch.setattr(instrument_docs, "PROFILES", {key: profile})
    with pytest.raises(SphinxError, match="local filenames"):
        build_reviews(source)
    assert not Path(key.replace(".", "_") + ".rst").exists()

"""Documentation source links identify the checkout used for line inspection."""

import subprocess

import pytest

from tools.docs_source import source_identity


def git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def checkout(tmp_path):
    git(tmp_path, "init")
    git(tmp_path, "config", "user.email", "test@example.invalid")
    git(tmp_path, "config", "user.name", "Documentation test")
    package = tmp_path / "src/asymmetry"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "example.py").write_text("def value():\n    return 1\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-m", "Fixture")
    return tmp_path


def test_development_identifies_exact_checkout(checkout):
    identity = source_identity(checkout / "src/asymmetry/__init__.py", "1.2.3")
    assert identity.revision == git(checkout, "rev-parse", "HEAD")
    assert "development" in identity.label
    assert identity.revision[:12] in identity.label


def test_release_requires_matching_tag(checkout):
    git(checkout, "tag", "v1.2.3")
    identity = source_identity(checkout / "src/asymmetry/__init__.py", "1.2.3")
    assert identity.label == "1.2.3"
    (checkout / "README").write_text("Next change")
    git(checkout, "add", ".")
    git(checkout, "commit", "-m", "Next revision")
    identity = source_identity(checkout / "src/asymmetry/__init__.py", "1.2.3")
    assert "development" in identity.label
    assert identity.revision == git(checkout, "rev-parse", "HEAD")


@pytest.mark.parametrize("staged", [False, True])
def test_modified_sources_cannot_link_to_committed_lines(checkout, staged):
    git(checkout, "tag", "v1.2.3")
    (checkout / "src/asymmetry/example.py").write_text("\n\ndef value():\n    return 1\n")
    if staged:
        git(checkout, "add", ".")
    identity = source_identity(checkout / "src/asymmetry/__init__.py", "1.2.3")
    assert identity.revision is None
    assert "modified" in identity.label


def test_archive_has_no_invented_source_revision(tmp_path):
    package = tmp_path / "src/asymmetry"
    package.mkdir(parents=True)
    source = package / "__init__.py"
    source.write_text("")
    identity = source_identity(source, "1.2.3")
    assert identity.revision is None
    assert "unavailable" in identity.label


def test_archive_without_git_can_still_build(tmp_path, monkeypatch):
    package = tmp_path / "src/asymmetry"
    package.mkdir(parents=True)
    source = package / "__init__.py"
    source.write_text("")
    monkeypatch.setenv("PATH", "")
    identity = source_identity(source, "1.2.3")
    assert identity.revision is None
    assert "unavailable" in identity.label


def test_source_link_lines_use_the_inspected_object(monkeypatch):
    import inspect
    import runpy
    from pathlib import Path

    import tools.docs_source
    from asymmetry.core.data.dataset import Run

    revision = "a" * 40
    monkeypatch.setattr(
        tools.docs_source,
        "source_identity",
        lambda *args: tools.docs_source.SourceIdentity("1.2.3 (development)", revision),
    )
    config = runpy.run_path(str(Path(__file__).resolve().parents[2] / "docs/conf.py"))
    lines, start = inspect.getsourcelines(Run.title.fget)
    link = config["linkcode_resolve"]("py", {"module": Run.__module__, "fullname": "Run.title"})
    assert link == (
        f"https://github.com/BenHuddart/asymmetry/blob/{revision}/"
        f"src/asymmetry/core/data/dataset.py#L{start}-L{start + len(lines) - 1}"
    )
    assert config["linkcode_resolve"]("js", {}) is None


def test_modified_checkout_has_no_source_links(monkeypatch):
    import runpy
    from pathlib import Path

    import tools.docs_source

    monkeypatch.setattr(
        tools.docs_source,
        "source_identity",
        lambda *args: tools.docs_source.SourceIdentity("modified checkout", None),
    )
    config = runpy.run_path(str(Path(__file__).resolve().parents[2] / "docs/conf.py"))
    assert (
        config["linkcode_resolve"](
            "py", {"module": "asymmetry.core.data.dataset", "fullname": "Run.title"}
        )
        is None
    )

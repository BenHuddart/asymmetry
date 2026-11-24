"""Real Sphinx regressions through the documentation gate's production entry."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def site(tmp_path):
    source = tmp_path / "docs"
    source.mkdir()
    (source / "conf.py").write_text(
        f"import sys\nsys.path.insert(0, {str(ROOT)!r})\n"
        "extensions = ['sphinx.ext.autodoc', 'sphinx.ext.intersphinx', 'tools.docs_policy']\n"
        "intersphinx_mapping = {'python': ('https://invalid.example/', None)}\n"
        "project = 'Gate fixture'\n"
    )
    scenarios = source / "screenshots" / "scenarios"
    scenarios.mkdir(parents=True)
    (scenarios / "example.py").write_text("class Example:\n    name = 'example'\n")
    (source / "index.rst").write_text(
        "Fixture\n=======\n\n.. image:: /_generated/screenshots/example.png\n"
    )
    return source


def build(site, mode="source"):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.docs_build",
            "--mode",
            mode,
            "--source-dir",
            str(site),
            "--output-dir",
            str(site.parent / "html"),
        ],
        cwd=ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=60,
    )

    logs = site.parent / "gate-results"
    logs.mkdir(exist_ok=True)
    sequence = len(list(logs.glob("*.json")))
    (logs / f"{sequence:02d}-{mode}.json").write_text(
        json.dumps(
            {
                "command": result.args,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
            indent=2,
        )
    )
    return result


def test_source_registered_missing_is_visible_and_publish_requires_capture(site):
    result = build(site)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Optional generated screenshot" in (site.parent / "html/index.html").read_text()
    result = build(site, "publish")
    assert result.returncode != 0
    assert "missing screenshot: example" in result.stdout + result.stderr


@pytest.mark.parametrize(
    "defect,diagnostic",
    [
        ("Broken ``literal\n", "Inline literal"),
        ("**A :math:`x` value.**\n", "Literal math role in prose"),
        ("A reference [1]_.\n\n.. [1] J. W. Author, Journal.\n", "Alphabetic list in a reference"),
        (
            "A reference [Author1965]_.\n\n.. [Author1965] J. W. Author, Journal.\n",
            "Alphabetic list in a reference",
        ),
        ("Long heading\n-----\n", "Title underline"),
        (":ref:`missing-label`\n", "undefined label"),
        (":class:`MissingLocalClass`\n", "MissingLocalClass"),
        (".. image:: /_static/missing.png\n", "image file not readable"),
        (":download:`missing <missing.zip>`\n", "download file not readable"),
        (".. image:: /_generated/screenshots/typo.png\n", "no registered scenario"),
    ],
)
@pytest.mark.parametrize("mode", ["source", "publish"])
def test_real_defects_fail(site, defect, diagnostic, mode):
    # A valid ordinary capture isolates the defect from publishing completeness.
    from PIL import Image

    image = site / "_generated/screenshots/example.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (2, 2)).save(image)
    with (site / "index.rst").open("a") as stream:
        stream.write("\n" + defect)
    result = build(site, mode)
    assert result.returncode != 0, result.stdout + result.stderr
    assert diagnostic in result.stdout + result.stderr


def test_unknown_mode_fails(site):
    assert build(site, "typo").returncode != 0


def test_page_budget_lists_largest_first(tmp_path):
    from tools.docs_build import DOCS_PAGE_SIZE_BUDGET_BYTES, oversized_docs_pages

    assert DOCS_PAGE_SIZE_BUDGET_BYTES == 600_000
    assert oversized_docs_pages(tmp_path / "absent") == []
    (tmp_path / "large.html").write_text("x" * 2000)
    (tmp_path / "small.html").write_text("x" * 1500)
    assert oversized_docs_pages(tmp_path, 1000) == ["large.html (2.0 KB)", "small.html (1.5 KB)"]


def complete_capture(site):
    from PIL import Image

    path = site / "_generated/screenshots/example.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2)).save(path)
    return path


def extension(site, code):
    (site / "fixture_extension.py").write_text(code)
    with (site / "conf.py").open("a") as stream:
        stream.write(f"sys.path.insert(0, {str(site)!r})\nextensions.append('fixture_extension')\n")


@pytest.mark.parametrize(
    "warning",
    [
        "logger.warning('unknown typed warning', type='fixture', subtype='unknown')",
        "logger.warning('untyped warning')",
        "logger.warning('multiline warning\\nsecond diagnostic line', type='fixture')",
    ],
)
@pytest.mark.parametrize("mode", ["source", "publish"])
def test_unknown_warnings_remain_fatal_and_visible(site, warning, mode):
    complete_capture(site)
    extension(
        site,
        "from sphinx.util import logging\nlogger = logging.getLogger(__name__)\n"
        f"def warn(app):\n    {warning}\n"
        "def setup(app):\n    app.connect('builder-inited', warn)\n",
    )
    result = build(site, mode)
    assert result.returncode != 0
    assert "WARNING:" in result.stderr
    assert "warning" in result.stderr
    if "multiline" in warning:
        assert "second diagnostic line" in result.stderr


@pytest.mark.parametrize("event", ["setup", "builder-inited", "build-finished"])
def test_extension_and_process_errors_are_not_inventory_exemptions(site, event):
    if event == "setup":
        code = "def setup(app):\n    raise RuntimeError('fixture extension failed')\n"
    else:
        code = (
            "def fail(*args):\n    raise RuntimeError('fixture extension failed')\n"
            f"def setup(app):\n    app.connect({event!r}, fail)\n"
        )
    extension(site, code)
    result = build(site)
    assert result.returncode != 0
    assert "fixture extension failed" in result.stderr


@pytest.mark.parametrize("mode", ["source", "publish"])
@pytest.mark.parametrize(
    "docstring,diagnostic",
    [
        ("Broken ``literal in an autodoc docstring.", "Inline literal"),
        ("A long section heading\n-----\n\nDescription.", "Title underline"),
    ],
)
def test_autodoc_markup_fails(site, mode, docstring, diagnostic):
    complete_capture(site)
    (site / "fixture_api.py").write_text(
        'def function():\n    """' + docstring.replace("\n", "\n    ") + '"""\n'
    )
    with (site / "conf.py").open("a") as stream:
        stream.write(f"sys.path.insert(0, {str(site)!r})\n")
    with (site / "index.rst").open("a") as stream:
        stream.write("\n.. autofunction:: fixture_api.function\n")
    result = build(site, mode)
    assert result.returncode != 0
    assert "docstring" in result.stderr and diagnostic in result.stderr


@pytest.mark.parametrize("mode", ["source", "publish"])
def test_ambiguous_api_targets_fail(site, mode):
    complete_capture(site)
    with (site / "index.rst").open("a") as stream:
        stream.write(
            "\n.. py:class:: first.Ambiguous\n\n.. py:class:: second.Ambiguous\n\n:class:`.Ambiguous`\n"
        )
    result = build(site, mode)
    assert result.returncode != 0
    assert "more than one target" in result.stderr


def test_cold_offline_external_limit_does_not_hide_broken_local_api(site, monkeypatch):
    guard = site.parent / "guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(
        "import socket\n"
        "def denied(*args, **kwargs):\n    raise RuntimeError('NETWORK REQUEST DENIED')\n"
        "socket.create_connection = denied\nsocket.socket.connect = denied\nsocket.getaddrinfo = denied\n"
    )
    # The subprocess's initial sys.path contains this directory before Sphinx loads.
    import sphinx

    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join((str(guard), os.environ.get("PYTHONPATH", ""), str(ROOT / "src"))),
    )
    backend = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, sphinx; print(json.dumps({'version': sphinx.__version__, 'file': sphinx.__file__}))",
        ],
        cwd=ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert backend.returncode == 0, backend.stderr
    selected = json.loads(backend.stdout)
    (site.parent / "gate-backend.json").write_text(json.dumps(selected, indent=2))
    assert selected == {"version": sphinx.__version__, "file": sphinx.__file__}
    monkeypatch.setenv("XDG_CACHE_HOME", str(site.parent / "empty-cache"))
    for locator in ("ASYMMETRY_CORPUS_ROOT", "ASYMMETRY_MUSRFIT_DATA", "ASYMMETRY_MANTID_SOURCE"):
        monkeypatch.delenv(locator, raising=False)
    with (site / "index.rst").open("a") as stream:
        stream.write("\n:func:`numpy.fft.rfft`\n")
    result = build(site)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "NETWORK REQUEST DENIED" not in result.stderr
    with (site / "index.rst").open("a") as stream:
        stream.write("\n:class:`BrokenLocalAPI`\n")
    result = build(site)
    assert result.returncode != 0
    assert "BrokenLocalAPI" in result.stderr
    assert "NETWORK REQUEST DENIED" not in result.stderr


def test_cache_modes_recheck_warnings_and_keep_real_assets(site):
    assert build(site).returncode == 0
    assert build(site).returncode == 0
    assert build(site, "publish").returncode != 0
    assert build(site).returncode == 0
    path = complete_capture(site)
    content = path.read_bytes()
    assert build(site, "publish").returncode == 0
    assert build(site).returncode == 0
    assert path.read_bytes() == content
    with (site / "index.rst").open("a") as stream:
        stream.write("\nBroken ``literal\n")
    for mode in ("source", "publish", "source"):
        result = build(site, mode)
        assert result.returncode != 0
        assert "Inline literal" in result.stderr


def test_publishing_allows_existing_labeled_corpus_stub(site):
    from docs.screenshots.capture_corpus import _write_stub_png

    complete_capture(site)
    scenarios = site / "screenshots/scenarios/corpus"
    scenarios.mkdir()
    (scenarios / "example.py").write_text("class CorpusExample:\n    name = 'corpus_example'\n")
    with (site / "index.rst").open("a") as stream:
        stream.write(
            "\nCorpus figures may be labelled placeholders when acquired data are unavailable.\n"
            "\n.. image:: /_generated/corpus_screenshots/corpus_example.png\n"
        )
    result = build(site)
    assert result.returncode == 0, result.stdout + result.stderr
    result = build(site, "publish")
    assert result.returncode != 0
    assert "image file not readable" in result.stderr
    image = site / "_generated/corpus_screenshots/corpus_example.png"
    image.parent.mkdir()
    _write_stub_png(image, "corpus_example")
    content = image.read_bytes()
    assert build(site, "publish").returncode == 0
    assert build(site).returncode == 0
    assert image.read_bytes() == content


def test_real_gate_applies_page_budget_after_successful_sphinx(site):
    extension(
        site,
        "from pathlib import Path\n"
        "def huge_page(app, exception):\n"
        "    Path(app.outdir, 'huge.html').write_text('x' * 600001)\n"
        "def setup(app):\n    app.connect('build-finished', huge_page)\n",
    )
    result = build(site)
    assert result.returncode != 0
    assert "build succeeded" in result.stdout
    assert "page exceeds 600 KB: huge.html" in result.stderr


def test_process_exit_is_preserved_without_any_warning(site):
    extension(site, "import os\ndef setup(app):\n    os._exit(23)\n")
    result = build(site)
    assert result.returncode == 23
    assert "WARNING:" not in result.stderr


def test_unknown_generated_corpus_name_fails_even_when_file_exists(site):
    from PIL import Image

    complete_capture(site)
    image = site / "_generated/corpus_screenshots/unknown.png"
    image.parent.mkdir()
    Image.new("RGB", (2, 2)).save(image)
    with (site / "index.rst").open("a") as stream:
        stream.write("\n.. image:: /_generated/corpus_screenshots/unknown.png\n")
    for mode in ("source", "publish"):
        result = build(site, mode)
        assert result.returncode != 0
        assert "Generated image has no registered scenario" in result.stderr


def test_exact_external_exception_rejects_external_typo(site):
    with (site / "index.rst").open("a") as stream:
        stream.write("\n:func:`numpy.fft.rfft` and :func:`numpy.fft.typo`\n")
    result = build(site)
    assert result.returncode != 0
    assert "numpy.fft.typo" in result.stderr
    assert "target not found: numpy.fft.rfft" not in result.stderr


def test_harness_production_entry_and_invalid_mode_before_capture(site):
    import shutil

    root = site.parent
    tools = root / "tools"
    tools.mkdir()
    for name in (
        "harness.py",
        "harness_checks.py",
        "owned_probes.py",
        "harness_maintenance.py",
        "harness_retired_docs.py",
        "docs_build.py",
    ):
        shutil.copyfile(ROOT / "tools" / name, tools / name)
    command = [sys.executable, str(tools / "harness.py"), "docs"]
    invalid = subprocess.run(
        command + ["--mode", "invalid", "--screenshots"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert invalid.returncode != 0
    assert "invalid choice" in invalid.stderr
    assert "watchdog" not in invalid.stdout + invalid.stderr
    source = subprocess.run(
        command + ["--mode", "source"], cwd=root, capture_output=True, text=True, timeout=60
    )
    assert source.returncode == 0, source.stdout + source.stderr
    failed = subprocess.run(
        command + ["--mode", "publish"], cwd=root, capture_output=True, text=True, timeout=60
    )
    assert failed.returncode != 0
    assert "missing screenshot: example" in failed.stderr


@pytest.mark.parametrize(
    "setting,diagnostic",
    [
        ("html_logo = 'missing-logo.png'", "missing-logo.png"),
        ("html_static_path = ['missing-static']", "missing-static"),
    ],
)
def test_required_config_assets_fail(site, setting, diagnostic):
    with (site / "conf.py").open("a") as stream:
        stream.write(setting + "\n")
    result = build(site)
    assert result.returncode != 0
    assert diagnostic in result.stderr


def test_autodoc_import_failure_stays_fatal(site):
    with (site / "index.rst").open("a") as stream:
        stream.write("\n.. autofunction:: absent_fixture_module.function\n")
    result = build(site)
    assert result.returncode != 0
    assert "failed to import" in result.stderr


def test_valid_local_reference_has_a_real_html_target(site):
    with (site / "index.rst").open("a") as stream:
        stream.write("\n.. py:class:: fixture.Target\n\n:class:`~fixture.Target`\n")
    result = build(site)
    assert result.returncode == 0, result.stdout + result.stderr
    html = (site.parent / "html/index.html").read_text()
    assert 'id="fixture.Target"' in html
    assert 'href="#fixture.Target"' in html


def test_corpus_helper_name_is_not_a_registered_capture(site):
    scenarios = site / "screenshots/scenarios/corpus"
    scenarios.mkdir()
    (scenarios / "helper.py").write_text("class Helper:\n    name = 'helper'\n")
    with (site / "index.rst").open("a") as stream:
        stream.write("\n.. image:: /_generated/corpus_screenshots/helper.png\n")
    result = build(site)
    assert result.returncode != 0
    assert "Generated image has no registered scenario" in result.stderr


@pytest.mark.parametrize("mode", ["source", "publish"])
@pytest.mark.parametrize("namespace", ["screenshots", "corpus_screenshots"])
@pytest.mark.parametrize("relative", [False, True])
def test_unknown_nested_generated_asset_fails(site, mode, namespace, relative):
    from PIL import Image

    complete_capture(site)
    image = site / "_generated" / namespace / "nested/unknown.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2)).save(image)
    if relative:
        page = site / "guide/page.rst"
        page.parent.mkdir()
        page.write_text(
            f"Guide\n=====\n\n.. image:: ../_generated/{namespace}/nested/../nested/unknown.png\n"
        )
        with (site / "index.rst").open("a") as stream:
            stream.write("\n.. toctree::\n\n   guide/page\n")
    else:
        with (site / "index.rst").open("a") as stream:
            stream.write(f"\n.. image:: /_generated/{namespace}/nested/unknown.png\n")
    result = build(site, mode)
    assert result.returncode != 0
    assert "Generated image has no registered scenario" in result.stderr


@pytest.mark.parametrize("mode", ["source", "publish"])
def test_normalized_relative_registered_asset_passes(site, mode):
    complete_capture(site)
    page = site / "guide/page.rst"
    page.parent.mkdir()
    page.write_text("Guide\n=====\n\n.. image:: ../_generated/screenshots/nested/../example.png\n")
    with (site / "index.rst").open("a") as stream:
        stream.write("\n.. toctree::\n\n   guide/page\n")
    result = build(site, mode)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="Symlinks require Windows privileges")
@pytest.mark.parametrize("mode", ["source", "publish"])
def test_unknown_nested_alias_cannot_borrow_registered_identity(site, mode):
    image = complete_capture(site)
    alias = image.parent / "nested/unknown.png"
    alias.parent.mkdir()
    alias.symlink_to("../example.png")
    with (site / "index.rst").open("a") as stream:
        stream.write("\n.. image:: /_generated/screenshots/nested/unknown.png\n")
    result = build(site, mode)
    assert result.returncode != 0
    assert "Generated image has no registered scenario" in result.stderr

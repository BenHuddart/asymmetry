"""Exercise authored task entries through Sphinx's actual search engine."""

import json
import re
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_task_queries_prefer_guidance_and_dotted_queries_keep_api_access(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to execute the Sphinx search engine")
    source = tmp_path / "source"
    source.mkdir()
    config = runpy.run_path(str(ROOT / "docs/conf.py"))
    (source / "conf.py").write_text(
        "project = 'Search regression'\n"
        f"html_show_search_summary = {config['html_show_search_summary']!r}\n"
    )
    documents = {
        "reference/detector_grouping": "Detector grouping",
        "reference/data_reduction/alpha_calibration": "Alpha calibration",
    }
    for name, title in documents.items():
        authored = (ROOT / "docs" / f"{name}.rst").read_text()
        entries = re.findall(r"^\.\. index:: .+$", authored, re.MULTILINE)
        page = source / f"{name}.rst"
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(title + "\n" + "=" * len(title) + "\n\n" + "\n\n".join(entries) + "\n")
    (source / "api.rst").write_text(
        "API\n===\n\n.. py:module:: asymmetry.core.transform.deadtime\n\n"
        "Deadtime, grouping, and alpha calibration operations.\n"
    )
    (source / "index.rst").write_text(
        "Search\n======\n\n.. toctree::\n\n   " + "\n   ".join([*documents, "api"]) + "\n"
    )
    output = tmp_path / "html"
    subprocess.run(
        [sys.executable, "-m", "sphinx", "-b", "html", "-W", str(source), str(output)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert "SHOW_SEARCH_SUMMARY: false" in (output / "_static/documentation_options.js").read_text()
    # Sphinx owns stemming and scoring; run its emitted scripts and real index.
    script = """
        const fs = require('node:fs');
        const vm = require('node:vm');
        const context = vm.createContext({window: {}, _ready: () => {},
          document: {getElementById: () => null}, SPHINX_HIGHLIGHT_ENABLED: false, _: text => text});
        context.window = context;
        const root = process.argv[1];
        for (const path of ['_static/language_data.js', '_static/searchtools.js', 'searchindex.js'])
          vm.runInContext(fs.readFileSync(`${root}/${path}`, 'utf8'), context);
        const query = vm.runInContext('Search.query.toString()', context);
        // Replace only the rendering callback; the emitted query retains its scoring and sorting.
        const execute = vm.runInContext(`(_displayNextItem) => (${query})`, context)(
          results => {context.ranked = results.slice().reverse();});
        console.log(JSON.stringify([
          'deadtime', 'grouping', 'alpha calibration', 'asymmetry.core.transform.deadtime'
        ].map(query => {execute(query); return {query, results: context.ranked};})));

    """
    result = subprocess.run(
        [node, "-e", script, str(output)], check=True, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    rows = json.loads(result.stdout)
    expected = [
        *["reference/detector_grouping"] * 2,
        "reference/data_reduction/alpha_calibration",
        "api",
    ]
    for row, destination in zip(rows, expected, strict=True):
        assert row["results"][0][0] == destination, row
    assert any(item[0] == "api" for item in rows[0]["results"])

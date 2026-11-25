"""Check teaching routes, authored destinations, and real browser interaction."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
MODULE = DOCS / "_static" / "reduction-guide.mjs"
STEPS = DOCS / "explanation" / "reduction_steps.inc"


def test_guide_facts_and_local_destinations():
    source = STEPS.read_text()
    identifiers = re.findall(r":name: guide-(\S+)", source)
    assert len(identifiers) == len(set(identifiers))
    labels = {
        label
        for path in DOCS.rglob("*.rst")
        if "_build" not in path.parts
        for label in re.findall(r"^\.\. _([^:]+):$", path.read_text(), re.MULTILINE)
    }
    for role, content in re.findall(r":(doc|ref):`([^`]+)`", source):
        target = content.rsplit("<", 1)[-1].removesuffix(">")
        if role == "ref":
            assert target in labels, target
        else:
            assert (DOCS / target.lstrip("/")).with_suffix(".rst").is_file(), target
    for step in source.split(".. container:: reduction-step")[1:]:
        for field in (
            "Input → output",
            "How it works",
            "Uncertainty",
            "When this applies",
            "What to keep with the result",
        ):
            assert f":{field}:" in step


def test_routes_preserve_observable_and_eligibility_contracts():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to execute the browser route table")
    program = f"""
        import {{destinations, traceRoute}} from {json.dumps(MODULE.as_uri())};
        const rows = [];
        for (const result of Object.keys(destinations)) {{
          for (const deadtime of [false, true]) {{
            for (const background of ['none', 'estimated', 'reference']) {{
              for (const source of ['grouped', 'ratio']) {{
                rows.push({{result, deadtime, background, source,
                  ...traceRoute(result, deadtime, background, source)}});
              }}
            }}
          }}
        }}
        console.log(JSON.stringify(rows));
    """
    result = subprocess.run(
        [node, "--input-type=module", "-e", program], capture_output=True, text=True, check=True
    )
    known = set(re.findall(r":name: guide-(\S+)", STEPS.read_text()))
    covered = set()
    for row in json.loads(result.stdout):
        steps = row["steps"]
        covered.update(steps)
        assert len(steps) == len(set(steps))
        assert set(steps) <= known
        assert steps[0] == "acquire"
        assert ("deadtime" in steps) == row["deadtime"]
        assert row["unavailable"] == (
            row["result"] == "countfit" and row["background"] == "reference"
        )
        if row["result"] == "countfit":
            assert steps[-1] == "countfit"
            assert not {"background", "bin", "ratio", "lifetime"} & set(steps)
        if row["result"] in {"integral", "mean"}:
            assert steps[-1] == "scan"
            assert not {"bin", "ratio", "lifetime"} & set(steps)
        if "ratio" in steps:
            assert steps.index("bin") < steps.index("ratio")
        if row["result"] == "fourier":
            assert ("lifetime" in steps) == (row["source"] == "grouped")
            assert ("ratio" in steps) == (row["source"] == "ratio")
        assert not {"scalar", "vector"} <= set(steps)
    assert covered == known


@pytest.mark.integration
@pytest.mark.parametrize("window_width", [500, 1280])
def test_guide_browser_selection_keyboard_and_unavailable_choice(tmp_path, window_width):
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if chrome is None:
        pytest.skip("Chrome/Chromium is needed for the guide interaction check")
    panels = "".join(
        f'<div class="reduction-step" id="guide-{key}"><p class="rubric">{title}</p></div>'
        for key, title in re.findall(
            r":name: guide-(\S+)\n\n\s+\.\. rubric:: (.+)", STEPS.read_text()
        )
    )
    page = tmp_path / "guide.html"
    page.write_text(
        '<html class="writer-html5"><head>'
        f'<link rel="stylesheet" href="{(DOCS / "_static" / "reduction-guide.css").as_uri()}">'
        '</head><body class="rst-content">'
        f'<div class="reduction-guide">{panels}</div>'
        '<script>window.initialStepTop=document.querySelector(".reduction-step").getBoundingClientRect().top;</script>'
        f'<script type="module">import {json.dumps(MODULE.as_uri())};'
        """
        const guide = document.querySelector('.reduction-guide');
        const check = (condition, message) => {if (!condition) throw new Error(message);};
        const select = (name, value) => {
          const control = guide.querySelector(`[name="${name}"]`);
          control.value = value;
          control.dispatchEvent(new Event('change', {bubbles:true}));
        };
        const button = key => guide.querySelector(`button[aria-controls="guide-${key}"]`);
        const visible = () => [...guide.querySelectorAll('.reduction-step')].filter(p => !p.hidden);
        try {
          check(visible().length === 1 && visible()[0].id === 'guide-acquire', 'initial selection');
          check(Math.abs(visible()[0].getBoundingClientRect().top - window.initialStepTop) < 2, 'stable initial lesson position');
          button('bin').click();
          check(visible()[0].id === 'guide-bin', 'pointer selection');
          check(button('bin').getAttribute('aria-current') === 'step', 'accessible selected state');
          button('bin').dispatchEvent(new KeyboardEvent('keydown', {key:'ArrowRight', bubbles:true}));
          check(visible()[0].id === 'guide-ratio', 'arrow selection');
          check(document.activeElement === button('ratio'), 'arrow focus');
          guide.querySelector('.reduction-pager button:last-child').click();
          check(visible()[0].id === 'guide-fit', 'next step');
          check(document.activeElement === visible()[0].querySelector('.rubric'), 'next heading focus');
          select('result', 'countfit');
          select('background', 'reference');
          check(guide.querySelector('[role="status"]').textContent.includes('This combination is unavailable'), 'unsupported combination');
          check(!guide.querySelector('.reduction-route button[aria-controls="guide-background"]'), 'no reference-subtracted count-fit path');
          select('background', 'none');
          check(!guide.querySelector('[role="status"]').textContent.includes('This combination is unavailable'), 'availability recovery');
          select('result', 'fourier');
          select('fourier-source', 'ratio');
          check(guide.querySelector('.reduction-route button[aria-controls="guide-ratio"]'), 'ratio Fourier path');
          check(!guide.querySelector('.reduction-route button[aria-controls="guide-lifetime"]'), 'distinct Fourier inputs');
          button('scalar').click();
          check(visible()[0].id === 'guide-scalar', 'alternative context is selectable');
          check(document.activeElement === visible()[0].querySelector('.rubric'), 'alternative heading focus');
          const readAll = guide.querySelector('button[aria-pressed]');
          readAll.click();
          check(visible().length === guide.querySelectorAll('.reduction-step').length, 'continuous reading');
          document.body.dataset.result = 'passed';
        } catch (error) {
          document.body.dataset.result = error.message;
        }
        </script></body></html>
        """
    )
    result = subprocess.run(
        [
            chrome,
            "--headless",
            f"--window-size={window_width},1200",
            "--no-sandbox",
            "--disable-gpu",
            "--allow-file-access-from-files",
            f"--user-data-dir={tmp_path / 'profile'}",
            "--dump-dom",
            "--virtual-time-budget=1500",
            page.as_uri(),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert 'data-result="passed"' in result.stdout, result.stdout + result.stderr

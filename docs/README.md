# Asymmetry Documentation

This directory contains the Sphinx documentation for the Asymmetry project.

## Building the Documentation

### Prerequisites

Install the documentation dependencies:

```bash
pip install -c constraints.txt -r docs/requirements.txt
```

Or install with the dev extras:

```bash
pip install -c constraints.txt -e ".[dev]"
```

On Linux, GUI API autodoc also needs Qt runtime libraries, even without a display
or screenshot capture. On Debian/Ubuntu, install them with:

```bash
sudo apt-get install --no-install-recommends libegl1 libxkbcommon0 libdbus-1-3
```

### Gated HTML builds

Run from the repository root:

```bash
python3 tools/harness.py docs --mode source
python3 tools/harness.py docs --mode publish
python3 tools/harness.py docs --mode pages
```

`source` is the contributor default (`docs` without `--mode` also selects it).
It requires installed documentation dependencies, but no private corpus, native
comparison programs, network access, warm inventory caches, or screenshot capture.
Only missing images with exact names in the screenshot scenario registry become
visible **Optional generated screenshot** notices. Existing captures are preserved.
This limited output is suitable for reviewing text; it is not ready to publish.

`publish` builds a local preview with the complete ordinary GUI screenshot set
and a PNG for every embedded corpus figure. Fill any corpus holes with labelled
placeholders using `python3 -m docs.screenshots.capture_corpus --stubs-missing`.
That command preserves existing renders.

`pages` requires a valid PNG and a matching checksum from a completed capture
for every embedded corpus figure. Run `capture_corpus --published` with the
corpus available to prepare both files. GitHub Pages uses this mode after it
verifies the pinned corpus archive and captures the required modules. A failed
build leaves the current site live.

The modes use existing assets by default. `--screenshots` requests ordinary
GUI capture before the gate and propagates capture failures. Unknown mode names
fail before capture or Sphinx starts. Generated HTML is in `docs/_build/html/`.

All modes check scenario references, use a fresh Sphinx environment (`-E -a`),
reject Sphinx warnings (`-W --keep-going`), preserve process/extension failure
exits, and check every output HTML page against the 600 KB budget. Missing
logos, static images, downloads, unknown generated names, malformed RST,
ambiguous or broken authored local/API references, and unknown/untyped warnings
are errors. The gate also rejects literal math roles in prose and alphabetic
lists in footnotes and citations, where unescaped author initials can become list markers. No warning category or historical warning count is exempted.

### Offline references and API rendering

All modes set the intersphinx mapping to an empty dictionary before inventory
loading. They make no inventory requests, even with empty caches. A small exact
set of external objects remains readable without hyperlinks:

| External scope | Objects without inventory links |
| --- | --- |
| NumPy | `numpy.random.Generator`, `numpy.ndarray`, `numpy.typing.NDArray`, `numpy.fft.rfft`, `numpy.linalg.eigvalsh` |
| SciPy | `scipy.optimize.least_squares` |
| Python standard library | `enum.Enum`, `abc.ABC`, `pathlib.Path`, `warnings` |
| Qt | `PySide6.QtWidgets.QWidget`, `PySide6.QtWidgets.QDialog`, `PySide6.QtWidgets.QMainWindow`, `PySide6.QtCore.QSettings` |
| iminuit | `iminuit` |

The shared policy owns these exact `(domain role, target)` exceptions. New
external targets require a deliberate documented decision; misspellings and
internal targets are not covered. The build prints the declared external
limitations and each page's registered image omissions. Other diagnostics retain
Sphinx's original locations and messages; Sphinx 7.3 and newer also display warning
classes. Older supported versions still reject warnings without parsing their text.

Generated Python type annotations are omitted from signatures rather than turned
into hyperlinks (`autodoc_typehints='none'`). Authored NumPy/Google parameter and
return descriptions, including their types and units, render as readable fields
rather than generated type-link lists. Authored cross-reference roles remain strict.
The gate checks authored references, not the completeness of annotation targets.
An offline HTML build does not bundle the browser-side MathJax CDN resource.

### Source identity and page layout

The sidebar identifies the imported checkout. A clean checkout at its matching
release tag displays the release number; other clean commits display
``development`` and the commit identifier. API source links use that exact commit,
with line numbers from the imported files. A modified checkout or a source archive
without Git identity is labelled explicitly and has no API source links. Build
from a clean checkout to verify links intended for publication.

Local image dimensions are read from the built assets and written into HTML so
browsers can reserve the correct aspect ratio before loading them. Percentage
widths remain responsive. The sidebar logo loads eagerly; content figures load
lazily. Tables wrap explanatory text within the theme's scrolling containers.
Review narrow viewports and enlarged text whenever a table or figure changes.

### Browser checks and search relevance

After a publishing build, serve the HTML and measure it with Node.js 22 or newer
and Chrome/Chromium installed:

```bash
python3 -m http.server 8000 --directory docs/_build/html
# In another terminal:
node tools/docs_browser.mjs --base-url http://127.0.0.1:8000/ --check --output browser.json
```

Use the published site URL to measure hosted behaviour. ``--chrome chromium``
selects another browser executable; ``--page reference/fourier_analysis.html``
limits a run to a page and may be repeated. The default is three cold/warm pairs
per page in a fresh browser profile, at 1440 × 1000 and 390 × 844 CSS pixels. Cold clears the HTTP cache;
warm immediately repeats the URL with caching enabled. CPU and network are
unthrottled, so record the machine and connection separately when comparing runs.

The JSON report separates HTML transfer and encoded body size, DOM readiness, paint, initiated requests,
startup layout shifts, font/mathematics readiness, and first/finished search
results. Layout-shift and paint observations end 250 ms after content readiness;
they do not describe later scrolling or interaction. A null load time means the
browser's load event has not completed. A command deadline is a measurement
failure, not evidence of a site-rendering delay; partial reports are retained and
marked incomplete. Hosted transfer and local rendering measurements are not
interchangeable. Compare repeated medians under matching conditions, without
promising a universal loading time.

``--check`` rejects page overflow, visible math-role syntax, MathJax errors,
incomplete font/mathematics preparation, and misplaced practical search results.
``--expect-revision <commit>`` also checks the published source identity. Ordinary
main index entries in the owning reference pages make ``deadtime``, ``grouping``,
and ``alpha calibration`` searches lead to guidance. Exact dotted API queries
retain access to API material. Search results show titles and section headings;
they do not fetch full matching pages to generate snippets. The search regression executes Sphinx's emitted
engine and index, rather than reproducing its scoring rules.

### Development builds and other formats

`cd docs && make html` runs the source gate. `make publish` builds the local
preview with prepared assets. Direct `sphinx-build` and the other make targets are
non-gating development builds: they load the same Sphinx policy, but do not promise
a fresh environment, warning failure exits, or the HTML page-budget check.

```bash
cd docs
make latexpdf    # PDF via LaTeX
make epub        # EPUB ebook
make man         # Unix man pages
```

The native warning and fresh-environment options are described in the
[Sphinx build documentation](https://www.sphinx-doc.org/en/master/man/sphinx-build.html);
[warning control](https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-warning-control)
and [intersphinx configuration](https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html)
describe the supported diagnostic and inventory mechanisms.

## Documentation Structure

```
docs/
├── conf.py               # Sphinx configuration
├── index.rst             # Landing page (four doorways)
├── contributing.rst      # Contributor guide
├── getting_started/      # Installation, quickstart, key concepts
├── workflows/            # End-to-end worked analyses
├── reference/            # Feature-by-feature reference and the physics behind it
├── explanation/          # Background, conventions, program comparison
└── api/                  # API reference (auto-generated)
    ├── index.rst
    ├── core.rst
    └── ...
```

## Maintaining the documentation

The documentation is part of the product: a user-facing change is not complete
until the docs describe it. These rules keep the pages, the screenshots, and
the program from drifting apart. `tools/harness.py structural` enforces the
mechanical ones.

### When you add a feature or change user-facing behaviour

1. **Update the page in the same PR.** Find the owning page via
   `reference/index.rst` (the "Find a feature" table) or
   `grep -rl "<feature name>" docs/`. A genuinely new feature gets a new page
   in the matching cluster of `reference/index.rst`, plus a row in the
   find-a-feature table when users will look for it by task.
2. **Quote the UI verbatim.** Button labels, menu paths, and window titles in
   the docs must match the widget code exactly (`QPushButton("…")`,
   `addAction("…")`). Verify against the source, not memory — UI strings keep
   American spellings (e.g. ``Optimize``) inside code-quotes while the
   surrounding prose stays British.
3. **Screenshots come from scenarios, never from hand-captures.** If the UI
   changed visibly, update or add a scenario under `screenshots/scenarios/`
   (see below) and recapture locally with
   `.venv/bin/python -m docs.screenshots.capture --only <name>` to check the
   framing. CI regenerates every screenshot on merge to main, so a stale
   scenario means a stale published image.
4. **Grep for casualties.** A renamed control, mode, or menu item usually
   appears on more than one page: `grep -rn "<old label>" docs/`.

### Screenshot scenarios

A scenario is a small `Scenario` subclass (see `screenshots/scenarios/_base.py`)
registered in `capture.py::_import_scenarios()`. House rules:

- **Deterministic**: seed every RNG; build synthetic data from
  `screenshots/data/archetypes.py` (material-inspired and generic presets — EuO, Gaussian fields,
  MgB₂, YBCO, PbF₂) so the physics in the figure is defensible.
- **Cropped**: frame the panel or dialog the prose discusses, not the whole
  main window, and never ship a figure with large empty regions — an empty
  table or blank plot reads as a broken program.
- **Fast**: well under a minute; the whole suite shares a 10-minute watchdog,
  below the 12-minute CI backstop.
  Set `requires_fit = True` when a real fit runs.
- **Budgeted**: ≤ 600 KB per PNG after the automatic lossless optimisation
  (`structural` fails oversized images when they exist on disk).
- **Referenced**: every scenario must be embedded by at least one page and
  every referenced image must have a scenario — `capture.py --check-refs`
  (run by `structural` and the docs CI smoke) fails on either mismatch.
- **Never modal**: capture runs offscreen, so nothing can click a message box.
  The driver wraps the whole run in `_base.auto_dismiss_modals()`, which answers
  every `QMessageBox` with its default (most dismissive) button and logs one
  `[screenshots] auto-dismissed modal: ...` line. Grep the capture log for that
  line — a scenario that trips a dirty-state guard on close should usually clear
  it explicitly instead. Without the guard such a prompt blocks until the
  watchdog hard-exits and *every alphabetically later scenario is never
  written*; that is how the site lost most of its GUI images for weeks.

### Page size budget

Every built HTML page must stay under **600 KB** raw (uncompressed).
`tools/harness.py docs` checks the whole output tree after a successful build
and fails listing each offender, largest first. The budget exists because the
site's perceived slowness is dominated by browser DOM parse and render cost,
which tracks raw page size rather than transfer size; 600 KB clears the current
largest page (`api/core.html`, ~549 KB) with headroom. A page that outgrows it
should be split into subpages or generated differently — that is how the 1.6 MB
`api/fitting.html` and the multi-MB `_modules/` source pages were fixed.

### Science content

- Follow `STYLE.md` for voice and citation format. Weave applicability into
  prose ("You should use this when…") instead of dedicated "When to use"
  headers; avoid runs of very short sections.
- Name algorithms concisely with an APS-style citation; put derivation-level
  mathematics in a `.. dropdown::` (sphinx-design), not in the reading line.
- State approximations that affect the accuracy of results — an
  "Assumptions and limitations" subsection when several apply, a sentence in
  place when one does.
- Every physics-bearing page ends with a References rubric. **Never fabricate
  a reference or a missing datum** — an incomplete entry with a FIXME beats a
  plausible invention.

### Verification ladder for docs changes

```bash
python -m docs.screenshots.capture --check-refs   # refs + size budget
python tools/harness.py structural                # includes the check above
python3 tools/harness.py docs --mode source       # offline source gate + page budget
pip install -c constraints.txt -e ".[hdf4,hdf5,root]" # screenshot file loaders
python3 tools/harness.py docs --mode publish --screenshots # ordinary capture + local preview
```

A docs-only PR gets the cheap CI smoke; the published site rebuilds (with
fresh screenshots) on merges to main that touch documentation, source, the shared
gate, or its dependency/workflow configuration.

## Writing Documentation

### Tutorials and reference

User-facing pages live under `getting_started/`, `workflows/`, `reference/`, and
`explanation/`. Use reStructuredText format:

```rst
Section Title
=============

Subsection
----------

Code example:

.. code-block:: python

   from asymmetry.core.io import load
   dataset = load("data.nxs")
```

### API Documentation

API docs are auto-generated from docstrings. Use NumPy or Google style:

```python
def my_function(param1: int, param2: str) -> bool:
    """Brief description.
    
    Longer description with more details about what this
    function does.
    
    Parameters
    ----------
    param1 : int
        Description of param1.
    param2 : str
        Description of param2.
        
    Returns
    -------
    bool
        Description of return value.
        
    Examples
    --------
    >>> my_function(42, "hello")
    True
    """
    pass
```

### Math Equations

Use LaTeX with MathJax:

```rst
.. math::

   A(t) = A_0 e^{-\lambda t} \cos(2\pi f t + \phi)
```

## Viewing Documentation Online

Once built, you can serve the documentation locally:

```bash
cd docs/_build/html
python -m http.server 8000
```

Then open http://localhost:8000 in your browser.

The site is built and deployed to GitHub Pages on release tags (`v*`) and on
pushes to `main` that affect docs, source, the shared gate/harness, relevant tooling
tests, dependency configuration, or the docs workflow. Source changes can alter
both autodoc output and the captured GUI images.
PRs touching documentation, source, tooling, tooling tests, dependency configuration,
or the docs workflow run the source gate without captures or corpus secrets.
Both CI jobs use the same harness gate; a failed Pages gate prevents artifact
upload and deployment.

If screenshot capture, completeness, or Sphinx fails, repair the build before rerunning.
A Pages deployment failure after a successful build is separate: rerun the failed
jobs while the `github-pages` artifact remains available (three days). Successful
deployments delete the artifact. Once it expires, or a workflow has deleted it,
rerun the full documentation workflow to rebuild it. The deploy action supports
at most ten minutes per attempt; the outer job timeout does not extend that limit.


## Contributing

When adding new features:

1. Add docstrings to all public functions/classes
2. Update relevant user guide pages
3. Add examples to docstrings
4. Build docs locally to verify

## Sphinx Extensions

The documentation uses:

- `sphinx.ext.autodoc` - Auto-generate API docs from docstrings
- `sphinx.ext.napoleon` - Support NumPy/Google style docstrings
- `sphinx.ext.linkcode` - Point `[source]` links at the matching source revision on GitHub
  (replaced `sphinx.ext.viewcode`, whose self-hosted `_modules/` pages added
  ~15 MB of very slow-rendering HTML)
- `sphinx.ext.intersphinx` - Link to other project docs
- `sphinx.ext.mathjax` - Math equation rendering
- `sphinx_rtd_theme` - Read the Docs theme

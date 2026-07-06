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

### Build HTML Documentation

```bash
cd docs
make html
```

The generated HTML will be in `docs/_build/html/`. Open `index.html` in your browser.

### Other Formats

```bash
make latexpdf    # PDF via LaTeX
make epub        # EPUB ebook
make man         # Unix man pages
```

### Clean Build

```bash
make clean
make html
```

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
  `screenshots/data/archetypes.py` (physically grounded presets — EuO, Ag,
  MgB₂, YBCO, PbF₂) so the physics in the figure is defensible.
- **Cropped**: frame the panel or dialog the prose discusses, not the whole
  main window, and never ship a figure with large empty regions — an empty
  table or blank plot reads as a broken program.
- **Fast**: well under a minute; the whole suite shares an 8-minute watchdog.
  Set `requires_fit = True` when a real fit runs.
- **Budgeted**: ≤ 600 KB per PNG after the automatic lossless optimisation
  (`structural` fails oversized images when they exist on disk).
- **Referenced**: every scenario must be embedded by at least one page and
  every referenced image must have a scenario — `capture.py --check-refs`
  (run by `structural` and the docs CI smoke) fails on either mismatch.

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
python tools/harness.py docs                      # Sphinx build (no screenshots)
python tools/harness.py docs --screenshots        # full local build
```

A docs-only PR gets the cheap CI smoke; the published site rebuilds (with
fresh screenshots) on every merge to main that touches `docs/**` or `src/**`.

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

   from asymmetry import load
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
- `sphinx.ext.viewcode` - Add source code links
- `sphinx.ext.intersphinx` - Link to other project docs
- `sphinx.ext.mathjax` - Math equation rendering
- `sphinx_rtd_theme` - Read the Docs theme

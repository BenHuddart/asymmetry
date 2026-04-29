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
├── conf.py              # Sphinx configuration
├── index.rst            # Main documentation page
├── installation.rst     # Installation instructions
├── contributing.rst     # Contributor guide
├── user_guide/          # User guides and tutorials
│   ├── index.rst
│   ├── loading_data.rst
│   ├── data_processing.rst
│   ├── gui_usage.rst
│   ├── fourier_analysis.rst
│   └── fitting.rst
└── api/                 # API reference (auto-generated)
    ├── index.rst
    ├── core.rst
    ├── gui.rst
    ├── io.rst
    ├── fitting.rst
    ├── fourier.rst
    └── transforms.rst
```

## Writing Documentation

### User Guide

User-facing tutorials and guides go in `user_guide/`. Use reStructuredText format:

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

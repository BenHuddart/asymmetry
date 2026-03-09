# Project Documentation Index

This directory contains all project documentation.

## Documentation Files

### User Documentation
- **[Sphinx Docs](index.rst)**: Main documentation entry point
  - Build with: `make html`
  - Output: `_build/html/index.html`
- **[Installation Guide](installation.rst)**: Setup and installation instructions
- **[User Guide](user_guide/)**: Tutorials and how-to guides
- **[API Reference](api/)**: Auto-generated API documentation

### Developer Documentation
- **[README.md](README.md)**: Documentation build instructions
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: System design, architecture, and specifications
- **[contributing.rst](contributing.rst)**: Contributing guidelines (Sphinx format)

### Additional Resources
- **[Logo](logo.png)**: Project logo for use in documentation

## Building the Documentation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Build HTML docs:
   ```bash
   make html
   ```

3. View in browser:
   ```bash
   ./open_docs.sh
   # Or manually open: _build/html/index.html
   ```

## Documentation Structure

```
docs/
├── README.md              # This file
├── ARCHITECTURE.md        # System architecture and design principles
├── index.rst              # Sphinx main index
├── installation.rst       # Installation instructions
├── contributing.rst       # Contributing guide
├── conf.py                # Sphinx configuration
├── requirements.txt       # Doc build dependencies
├── Makefile              # Build automation (Unix)
├── make.bat              # Build automation (Windows)
├── user_guide/           # User tutorials and guides
│   ├── index.rst
│   ├── loading_data.rst
│   ├── data_processing.rst
│   ├── gui_usage.rst
│   ├── fitting.rst
│   └── fourier_analysis.rst
├── api/                  # API reference (auto-generated)
│   ├── index.rst
│   ├── core.rst
│   ├── io.rst
│   ├── fitting.rst
│   └── ...
└── _build/               # Generated output (gitignored)
    └── html/
```

## Documentation Guidelines

- Use reStructuredText (.rst) for Sphinx documentation
- Use Markdown (.md) for standalone docs (README, ARCHITECTURE)
- Follow NumPy docstring format in Python code
- Keep examples up-to-date with API changes
- Include code examples for all major features

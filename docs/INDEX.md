# Project Documentation Index

This directory contains all project documentation.

## Documentation Files

### User Documentation
- **[Sphinx Docs](index.rst)**: Main documentation entry point
  - Build with: `make html`
  - Output: `_build/html/index.html`
- **[Getting Started](getting_started/)**: Installation, quickstart, and key concepts
- **[Workflows](workflows/)**: End-to-end worked analyses
- **[Reference](reference/)**: Feature-by-feature reference and the physics behind it
- **[Explanation](explanation/)**: Background, conventions, and program comparison
- **[API Reference](api/)**: Auto-generated API documentation

### Developer Documentation
- **[README.md](README.md)**: Documentation build instructions
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: System design, architecture, and specifications
- **[GUI_GUIDELINES.md](GUI_GUIDELINES.md)**: How to build GUI here — tokens, fonts, metrics, PanelSection/ActionFooter, the UI zoom, and what the harness enforces
- **[HARNESS.md](HARNESS.md)**: Agent harness workflow and validation commands
- **[porting/README.md](porting/README.md)**: Policy and required artifacts for study-first feature ports
- **[QUALITY.md](QUALITY.md)**: Current quality map, risk areas, and validation paths
- **[PLANS.md](PLANS.md)**: Active and deferred execution plans for agents
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
├── GUI_GUIDELINES.md      # How to build GUI here (tokens, fonts, metrics, panels, zoom)
├── HARNESS.md             # Agent harness workflow
├── porting/               # Study-first porting workflow and feature studies
├── QUALITY.md             # Quality map and risk areas
├── PLANS.md               # Agent-facing execution plans
├── index.rst              # Sphinx landing page (four doorways)
├── contributing.rst       # Contributing guide
├── STYLE.md               # Writing-voice and citation style guide
├── conf.py                # Sphinx configuration
├── requirements.txt       # Doc build dependencies
├── Makefile               # Build automation (Unix)
├── make.bat               # Build automation (Windows)
├── getting_started/       # Installation, quickstart, key concepts
├── workflows/             # End-to-end worked analyses
├── reference/             # Feature reference (fit_functions/, data_reduction/, examples/)
├── explanation/           # Background, conventions, program comparison
├── screenshots/           # Offscreen GUI screenshot harness (capture.py)
├── api/                   # API reference (auto-generated)
│   ├── index.rst
│   ├── core.rst
│   └── ...
├── _generated/            # Generated screenshots (gitignored)
└── _build/                # Generated output (gitignored)
    └── html/
```

## Documentation Guidelines

- Use reStructuredText (.rst) for Sphinx documentation
- Use Markdown (.md) for standalone docs (README, ARCHITECTURE)
- Follow NumPy docstring format in Python code
- Keep examples up-to-date with API changes
- Include code examples for all major features

# Contributing to Asymmetry

Thank you for your interest in contributing to Asymmetry!

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/asymmetry.git
   cd asymmetry
   ```

2. Install in development mode with all dependencies:
   ```bash
   pip install -c constraints.txt -e ".[all]"
   ```

3. Run tests to verify setup:
   ```bash
   python -m pytest tests/
   ```

## Code Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Run `ruff` for linting:
  ```bash
  ruff check src/ tests/
  ```

## Testing

- Write tests for new features in the `tests/` directory
- Ensure all tests pass before submitting:
  ```bash
   python -m pytest tests/ --cov=src/asymmetry
  ```
- Aim for comprehensive test coverage (current: 71%)
- Use pytest fixtures for common test setup
- GUI tests should use `pytest-qt` and `qtbot` fixture

## Documentation

- Add docstrings to all public functions and classes
- Follow NumPy docstring format
- Update Sphinx documentation in `docs/` for new features:
  ```bash
  cd docs
  make html
  ```

## Submitting Changes

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit with clear messages:
   ```bash
   git commit -m "Add feature: description"
   ```

3. Push your branch and create a pull request

4. Ensure all CI checks pass

## Project Structure

```
asymmetry/
├── src/asymmetry/       # Main source code
│   ├── core/            # Pure-Python analysis engine
│   └── gui/             # PySide6 GUI components
├── tests/               # Test suite
├── docs/                # Sphinx documentation
└── pyproject.toml       # Project configuration
```

## Adding New Features

### New File Loader
1. Create a new loader in `src/asymmetry/core/io/`
2. Inherit from `BaseLoader`
3. Register in `src/asymmetry/core/io/__init__.py`
4. Add tests in `tests/`

### New Fit Model
1. Add function to `src/asymmetry/core/fitting/models.py`
2. Register in `MODELS` dictionary
3. Add tests with known results

### New GUI Panel
1. Create panel in `src/asymmetry/gui/panels/`
2. Inherit from `QWidget`
3. Add to MainWindow
4. Add GUI tests using `pytest-qt`

## Questions?

Open an issue for discussion before starting major changes.

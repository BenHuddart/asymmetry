# Test Layout

`tests/` mirrors `src/asymmetry/`'s layer boundaries. Every test file lives in
the subpackage matching the layer it exercises, so coverage for a module is
easy to find without a repo-wide grep.

## Subpackages

- `tests/core/` — pure `asymmetry.core` analysis: fitting, fourier, transform,
  representation, simulate, instrument, and similar. No Qt, no `asymmetry.gui`.
- `tests/gui/` — anything that imports `PySide6` or `asymmetry.gui`, or is
  marked `pytest.mark.gui`. This is most of the suite (GUI tests pay the
  per-test `MainWindow`/widget cost the fast tier is designed to avoid).
- `tests/io/` — loaders and file-format adapters (`asymmetry.core.io`), not
  gui.
- `tests/project/` — the `.asymp` project schema (`asymmetry.core.project`),
  not gui. In practice most project-schema tests also exercise GUI save/load
  flows and therefore live in `tests/gui/` instead — see the placement rule
  below.
- `tests/tools/` — tests of repo tooling: `tools/harness.py`, packaging
  (`packaging/macos_icon.py`, Windows installer), and lint-guard tests that
  check source text for a convention (design-token/hex discipline, app-import
  smoke checks that assert on structure rather than runtime GUI behavior).
- `tests/integration/` — genuinely cross-cutting end-to-end tests that do not
  belong to one layer. Used sparingly; prefer `core` when a test could go
  either way.
- `tests/negmu/`, `tests/docs/`, `tests/porting/` — already-organized
  subpackages (negative-muon analysis, documentation checks, feature-porting
  studies). Follow their existing internal conventions.

## Placement rule

Classify by import surface, in priority order:

1. **tools** — content is about `tools/harness.py`, packaging, icons, or is a
   static lint-guard reading source files as text.
2. **gui** — imports `PySide6` or `asymmetry.gui`, or carries
   `pytest.mark.gui`. This wins even if the file also touches `io`/`project`
   (most GUI panel tests load corpus data or project files as part of the
   scenario).
3. **io** — not gui, imports `asymmetry.core.io` / loaders.
4. **project** — not gui, imports `asymmetry.core.project` (the `.asymp`
   schema).
5. **core** — everything else: pure `asymmetry.core` analysis.
6. **integration** — reserve for tests that are genuinely cross-cutting and
   don't fit one layer.

When adding a new test file, place it beside the layer it exercises using this
same priority order — don't default to `tests/integration/` just because a
test touches more than one module.

## Conventions

- Name files `test_*.py`; mirror the feature name from the module under test
  where practical (`test_transforms.py` covers `asymmetry.core.transform`, and
  so on).
- Every subpackage directory needs an `__init__.py` (this is why `tests/` is
  itself a package — bare directories break collection/imports).
- Keep `tests/conftest.py` and `tests/_qt_helpers.py` at the `tests/` root —
  the session `QApplication`, the autouse Qt-cleanup fixture, and the
  `--shard` option are global and every subpackage depends on them.
- If a test needs to import from another test module, import it from its full
  path, e.g. `from tests.core.test_wimda_model_function_parity import
  EUO_NU_T_TREND`, not a bare `tests.test_...` (there is no such module once
  files are organized into subpackages).
- `Path(__file__)`-relative fixture/source lookups must count parents from the
  file's actual subpackage location, not from the old flat `tests/` root — a
  file one directory deeper needs one more `.parent`/`parents[N]` step to reach
  the same repo-root-relative target.

## Tiering and CI are unaffected by this layout

`--tier`/`--subset`/`--shard` select tests by pytest marker and by a hash of
each test's node id (see `tests/conftest.py`), not by directory, so this
subpackage layout is transparent to the harness. CI's `tests/**` path filters
and `testpaths = ["tests"]` in `pyproject.toml` already match subdirectories
recursively, so no CI or `pyproject.toml` pytest-config changes were needed
for this reorganization (only the Ruff `per-file-ignores` keys, which are
literal per-file paths, needed re-pathing).

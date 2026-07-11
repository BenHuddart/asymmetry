# Corpus-driven documentation scenarios

Scenarios in this package drive the Asymmetry GUI through the worked examples
of the **WiMDA muon school corpus**, producing offscreen renders destined for
the Sphinx docs (workflow pages showing the program in action on real data).
They will eventually run in CI with the corpus provisioned and
`ASYMMETRY_CORPUS_ROOT` set.

## Ground rules

- **One module per corpus example**, named after the example slug
  (e.g. `euo_ordering.py`, `basics_calibration.py`). A module may register
  several scenarios. **Never edit shared files** (`capture.py`, existing
  scenarios, `src/`) — the package auto-imports your module.
- Scenario `name` must start with `corpus_` and be specific:
  `corpus_euo_zf_fit`, `corpus_basics_deadtime`, `corpus_trsb_kt_step`.
- Set `example = "<corpus-relative example folder>"` on each scenario class.
- Resolve all data through `_corpus.corpus_path()` / `load_corpus_datasets()`
  — never hard-code absolute paths.
- **The example's `GROUND_TRUTH.md` is the spec**: follow its prescribed
  workflow (model, ties, fixed params, seeds, fit window) and check fitted
  values against its expected-results table. Record the comparison in your
  `NOTES_<slug>.md`.
- House rules from `docs/README.md` apply: deterministic, cropped to the
  panel under discussion (no acres of empty UI), fast, ≤ 600 KB per PNG,
  `requires_fit = True` when a real fit runs at capture time.
- Capture with
  `.venv/bin/python -m docs.screenshots.capture_corpus --only <names>`
  from the worktree root; PNGs land in `docs/_generated/corpus_screenshots/`.
  **Always look at your rendered PNGs** (Read them as images) before calling
  a scenario done — empty plots, clipped panels, and mis-framed dialogs are
  the common failure modes.

## What to capture per example

Not just the headline result. Each example should yield roughly 3–6 renders:

1. **A data-handling step** — loading/grouping/α/dead-time/t0/rebin, whatever
   the example's guide actually teaches.
2. **The core analysis step** — the model being set up on a run (fit panel,
   FFT window, ALC scan, wizard page...).
3. **The headline result** — converged fit or parameter-vs-temperature/field
   trend that reproduces the ground-truth number.
4. **Anything distinctive** this example shows better than any other
   (MaxEnt, global fits, period handling, logbook view, waterfall overlays,
   Knight shift, spectral moments...). Skim `docs/screenshots/scenarios/` for
   the feature surface that already has synthetic-data screenshots.

## Notes file

Each example ships `NOTES_<slug>.md` alongside its module:

- scenarios registered (name → what the render shows, intended docs use),
- run selection and workflow followed (with GROUND_TRUTH.md § references),
- fitted values vs ground-truth targets (a small table),
- feature-demonstration opportunities spotted (even ones you didn't capture),
- problems hit (loader quirks, fit instabilities, UI gaps) — honestly.

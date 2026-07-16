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
- **`--stubs` (no corpus needed).**
  `.venv/bin/python -m docs.screenshots.capture_corpus --stubs` writes a small
  uniform placeholder PNG (600×380, light grey, scenario name centred) for
  every registered corpus scenario into the output dir. It needs neither the
  corpus, Qt, nor a fit backend (PIL only), so a Sphinx build on a machine
  without `ASYMMETRY_CORPUS_ROOT` (CI) never breaks on a missing corpus render.
  Combine with `--only <names>` to stub a subset.

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

## Lessons learned (waves 1–2 — read before writing a scenario)

- **Serialize captures.** Concurrent `capture_corpus` processes deadlock under
  offscreen Qt. Wrap every run:
  `flock /tmp/asymmetry-capture.lock .venv/bin/python -m docs.screenshots.capture_corpus --only <names>`
  and capture several scenarios in ONE process where possible.
- **Fit range**: the single-tab fit-range spinbox does not commit values — use
  `plot_panel.set_fit_range(t0, t1)` (known product bug, twice confirmed).
- **Dock resizes** made in `build()` are clobbered by `MainWindow.showEvent`'s
  adaptive widths — resize docks in `settle()`, after show.
- **Warm-start batch fits** in temperature/field order; cold seeds walk to
  wrong minima on real data (EuO ν, YMnAl λ/β, PTFE r all showed this).
- **MaxEnt works out of the box since PR 249** (phase seeding + workload
  auto-steering, both default-on; the old BiSCCO divergence is fixed and the
  workload warning routes to the log when headless). Leave End/Binning unset
  and let auto-steer size them; `auto_steer_applied` in the result records
  what it chose. Remaining rough edges: the fixed 300 G auto-window half-width
  is too wide for low-field runs (LiFeAs — set an explicit window), wide
  multi-line windows can't be binned by the steer margin (benzene — slow),
  and χ²/N may plateau above 1 on some real F/B runs.
- **Trending panel plots one active series** — for two-series comparisons use
  a standalone matplotlib figure via the house `mgb2_lambda_t.py` pattern.
- **Twin-axis `QWidget.grab` has a last-pixel flake offscreen** — prefer two
  stacked single-axis subplots, or save from the canvas Agg buffer.
- **Framing**: watch right/bottom edge clipping of dock panels (read the PNG!).

## Notes file

Each example ships `NOTES_<slug>.md` alongside its module:

- scenarios registered (name → what the render shows, intended docs use),
- run selection and workflow followed (with GROUND_TRUTH.md § references),
- fitted values vs ground-truth targets (a small table),
- feature-demonstration opportunities spotted (even ones you didn't capture),
- problems hit (loader quirks, fit instabilities, UI gaps) — honestly.

## CI corpus provisioning

The docs deploy build provisions the corpus from a **GitHub release asset** on
the (private) corpus repository `BenHuddart/wimda-muon-school-corpus`:

- `tools/package_corpus.py --corpus "<root>" --version <YYYY.MM.DD>` builds the
  CI archive (`dist/wimda-corpus-<version>.tar.zst`, ~272 MB): data files,
  logbooks, ground truths and reference outputs; no papers/guides/duplicates.
- Publish it (with its `.sha256`) as release `v<version>` on the corpus repo,
  then bump `CORPUS_VERSION` in `.github/workflows/docs-pages.yml`.
- The workflow needs a `CORPUS_REPO_TOKEN` repo secret (fine-grained PAT with
  read access to the corpus repo's releases). Without it — e.g. on forks — the
  build falls back to `--stubs` placeholders and stays green. The extracted
  corpus is cached (`actions/cache`) keyed on the version, so the download is
  paid once per corpus bump.
- Capture runs one process per scenario module (`--list-modules` / `--module`)
  to stay inside the per-process watchdog; `--stubs-missing` then fills any
  hole so the built site is always image-complete.

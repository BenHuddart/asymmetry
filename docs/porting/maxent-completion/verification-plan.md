# MaxEnt completion — verification plan

Each phase ends **`python tools/harness.py validate` green** (~2 min; lint +
structural + full pytest suite, `-n auto`) and leaves a main-mergeable state
with a milestone commit. GUI tests run under `QT_QPA_PLATFORM=offscreen`. Tests
live beside the behaviour they protect (`tests/test_maxent*.py`,
`tests/test_fourier_units.py`, GUI panel tests). Use the **worktree's**
`.venv/bin/python`.

The standing non-regression bar throughout: **existing MaxEnt tests
(`tests/test_maxent.py`, 495 lines) and the resumable-state round-trip must not
regress**, and the project-file recipe round-trip must keep working.

## Climb the ladder

Use the smallest check first, then widen:

```bash
.venv/bin/python tools/harness.py structural
.venv/bin/python tools/harness.py lint
.venv/bin/python tools/harness.py test -- tests/test_maxent.py
.venv/bin/python tools/harness.py validate
.venv/bin/python tools/harness.py docs        # docs-only changes / user-guide build
```

## Phase 1 — reconstruction overlay

Engine/representation:
- **S1** reconstruction-within-noise + **χ²-equals-engine** assertions (the two
  headline targets) in `tests/test_maxent.py` (or a new
  `tests/test_maxent_reconstruction.py`).
- `opus`-derived per-group reconstruction is exposed on the result/diagnostics;
  a unit test asserts the per-group arrays match `opus(spectrum, …)` exactly.
- Recipe round-trip with the overlay toggle (**P1** subset).

GUI:
- offscreen panel test: after a (mocked/fast) run the overlay toggle becomes
  available; toggling renders per-group + combined traces with a residuals strip
  and does not crash; the displayed χ² equals the result metadata χ².

Gate: `validate` green; `test_maxent.py` unchanged-or-extended, not regressed.

## Phase 2 — pulsed-source correctness

Kernel:
- **S2** flat-amplitude-recovery (enabled) vs roll-off (disabled); single-pulse
  limit of the double-pulse formula; DC values. Pure-function tests on the
  pulse-response kernel — fast, no full MaxEnt run needed for the kernel asserts.
- optional Mantid `start.py` kernel oracle (skipif-guarded; see `test-data.md`).
- **S3** interior exclusion window: spectrum recovery + grid-length-unchanged.

Units helper:
- `tests/test_fourier_units.py`: round-trip MHz↔Gauss↔Tesla against the CODATA
  constants; γ_µ/2π = 135.538817 MHz/T; resolution helpers. This file is the
  recorded contract the `frequency-domain-finishers` project reuses.

GUI:
- offscreen: pulse-mode selector (ignore/single/double + width/separation
  fields) builds the config; field-axis units toggle (MHz/Gauss/Tesla) relabels
  the spectrum plot.

Persistence:
- **P1/P2**: recipe round-trip with pulse + exclusion + units fields; state
  signature forces restart when pulse mode / exclusion window change.

Gate: `validate` green; Phase-1 tests still pass.

## Phase 3 — calibration workflows

Engine:
- **S4** injected-deadtime recovery (physical µs; corrected-run returns ≈0;
  suggest-only does not mutate grouping).
- **S5** ZF Kubo–Toyabe: α-tie obeyed, phases pinned, spectrum broad near zero;
  SpecBG operates on a copy.
- **S6** phase-exchange round-trip (radians↔degrees, matched by group id,
  provenance attached).

GUI:
- offscreen: tables tab renders per-group phase/amp/deadtime; "Use fitted
  phases" / "Send phases to fit" paired actions with provenance labels;
  ZF/LF mode selector constrains the group table to two F/B groups; spectrum
  text export + run-log export produce well-formed files.

Persistence:
- **P1** full recipe with every new field; **P2** state-signature restart rules
  for ZF/LF mode and deadtime fitting; project round-trip of the editable tables
  and exchange provenance.

Gate: `validate` green; Phases 1–2 tests still pass; **user-guide docs build
clean** (`harness docs`).

## Documentation verification

`python tools/harness.py docs` must pass after each phase that adds user-guide
pages. New pages follow `docs/user_guide/fit_functions/` as the template:
result-first physics prose, rendered math (no raw LaTeX), uncertainties as
0.23(1), APS references in lists, and a diagnostic "when to use this" register
per feature. Toctree edits are additive and at the end of the relevant block.

## What "done" looks like

- All three phases validate-green with milestone commits, each main-mergeable.
- Brief verification targets met: reconstruction matches input within noise with
  χ² equal to the engine's; pulse-shape gives flat amplitude vs frequency when
  enabled (and documented distortion when disabled); deadtime fit recovers a
  known injected deadtime; ZF mode on a synthetic Kubo–Toyabe distribution;
  no regression of existing MaxEnt tests or the resumable-state round-trip;
  recipe round-trip with the new fields.
- Divergences from WiMDA recorded in `comparison.md` (both behaviours stated);
  any follow-ons recorded in `implementation-options.md` and the relevant
  candidate entries.

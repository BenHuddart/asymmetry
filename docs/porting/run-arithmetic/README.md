# Study: run-arithmetic

Umbrella: `wimda-parity-gap` · Wave B · Size M · Branch `feat/run-arithmetic`
· Study date 2026-06-12.

## Purpose

Histogram-level **co-add** and **co-subtract** of muon runs with correct
counting statistics. This is partly a correctness fix: Asymmetry's current
co-add (`data_browser._coadd_datasets`) averages already-reduced asymmetry
curves with unweighted means — statistically wrong at low counts — and stores
the result with `histograms=[]`, so a combined dataset cannot be regrouped,
deadtime-corrected, count-fitted, or transformed (MaxEnt/FFT). WiMDA combines
at the raw-count level; we adopt that and add a co-subtract path (laser-on/off
photo-µSR, background-style differences) that does not exist today.

The governing principle: **sum counts, then reduce — never average reduced
curves.** Errors and any nonlinear correction (deadtime, α, background) must
see the total statistics.

## Entry points

- New Qt-free kernel `core/data/combine.py`:
  `combine_runs(runs, sign=+1) -> Run`. Returns a first-class
  :class:`~asymmetry.core.data.dataset.Run` with summed period-aware
  histograms, accumulated good frames, event-weighted scalar metadata + spread
  keys, and a `combination` provenance block.
- Co-subtract (`sign=-1`) routes its per-detector arithmetic through the
  existing chokepoint `core/transform/background.subtract_scaled_counts`
  (variances add; F9 directive). The "Subtract Reference Run…" GUI action
  resolves its reference through `core/io.resolve_background_reference`, the
  single reference-run resolution home.
- GUI: `gui/panels/data_browser.py` — `_coadd_datasets` and
  `rebuild_combined_dataset` are rewired onto `combine_runs`; a new flat
  top-level context-menu action `_subtract_reference_run` (shown for a single
  selected run) drives co-subtract.

## Data flow

```
runs (list[Run], each grouped identically) ─┐
                                            ├─ combine_runs(sign=±1)
period payloads / raw histograms ───────────┘        │
   per-detector t0 alignment (apply_grouping_aligned) │
   detector-wise count sum (+ / via subtract_scaled)  │
   good-frame + frame accumulation                    │
   event-weighted T/field scalars + spread keys       ▼
                                            first-class Run (histograms != [])
                                                     │ reduce_run_to_dataset / GUI regroup
                                                     ▼
                                            MuonDataset  → plot / fit / FFT / MaxEnt
```

## Decisions (settled with Ben, 2026-06-12)

1. **Co-subtract surface = reference-run only.** Ship "Subtract Reference
   Run…": pick one designated reference run, subtract it (frame-scaled, via
   `subtract_scaled_counts`) from each selected run. Symmetric two-run / N-run
   signed co-subtract is a recorded follow-on, not built here.
2. **Combined rows become first-class where it "just works".** `combine_runs`
   produces a Run with real summed histograms + mirrored grouping, so the
   existing regroup / deadtime / count-fit / MaxEnt pipelines operate on a
   combined row through their normal paths. Verified per operation; only
   minimal glue is added. Anything needing substantial new GUI plumbing is a
   recorded follow-on.
3. **t0 alignment = align per-detector, hard-require width + count.** Sum after
   shifting each detector to a common t0 (mirrors
   `transform.grouping.apply_grouping_aligned` and WiMDA's ACC `tshift`);
   record the alignment in provenance. Hard-error only on mismatched bin width
   or detector count. The GUI co-add path keeps its existing strict
   grouping-equality gate, so alignment only matters for the PSI multi-t0 /
   sub-bin cases that still pass that gate.
4. **`.asymp` migration = silent recompute on load, documented.** Existing
   projects store only the combination *definition* (source run numbers), never
   the curve, so loading already recomputes from sources. Route that recompute
   through `combine_runs` (now histogram-correct). No schema bump (W1); the
   behaviour change is documented in the user guide and PR body, with no
   per-load dialog.
5. **"Subtract Reference Run…" = flat top-level context-menu action**, shown
   when exactly one run is selected, beside "Co-add Selected" / "Separate
   Combined" — matching the post-#53 flat-menu style.

## Collision directives honoured

- **F9** — co-subtract is built *on* `subtract_scaled_counts` and
  `resolve_background_reference`, not beside them.
- **W3** — `metadata["temperature"]`/`["field"]` stay scalar floats
  (event-weighted); spread under new `temperature_spread`/`field_spread`;
  `combined_from` kept verbatim (trend panel + GLE export consume it); a richer
  nested `metadata["combination"]` block mirrors `metadata["simulation"]`.
- **W11** — `rebuild_combined_dataset` routes through `combine_runs`; negative
  synthetic ids (`_next_combined_id`) and the `.asymp` top-level
  `combined_datasets` block are unchanged. Combined results change numerically
  (the fix) — test expectations updated deliberately.
- **W12** — per-period histograms summed via `core/io/periods.py` helpers
  (`sum_period_histograms`) writing the exact existing period metadata keys;
  no second period-summing path. Frames are accumulated where they live
  (grouping/metadata dicts) — Run has no `good_frames` field.
- **W1** — no `schema_version` bump; all persistence additive.

## Dependencies

- `core/data/dataset.py` (Run/MuonDataset/Histogram), `core/io/periods.py`
  (period summation + frame helpers), `core/transform/grouping.py` (alignment,
  `good_frames`), `core/transform/background.py` (`subtract_scaled_counts`),
  `core/io/__init__.py` (`resolve_background_reference`),
  `core/simulate.py` (`reduce_run_to_dataset`, provenance pattern to mirror).
- GUI: `gui/panels/data_browser.py`, plus a small reference-run picker dialog.

## Status — implemented

Shipped on `feat/run-arithmetic`: Qt-free `core/data/combine.py`
(`combine_runs` + `reduce_combined_run`); the Data Browser co-add rewired onto
the kernel; the "Subtract Reference Run…" action with sign-aware combined rows
and additive `.asymp` persistence; the user-guide page
`docs/user_guide/run_arithmetic.rst`. Verified by `tests/test_combine.py` (pull
test, co-subtract zero/√2/scaled, negative guard, event-weighted metadata, t0
alignment, two-period co-add, F9 chokepoint spy), `tests/test_data_browser_combine.py`
and a `.asymp` round-trip in `tests/test_mainwindow_additional.py`. Headline:
the curve-mean co-add over-estimated the combined error by **53 %** at a 10:1
statistics ratio (synthetic, see comparison.md).

## Out of scope (recorded)

- ~~In-batch co-add during sequential fitting~~ — **done on
  `feat/batch-arithmetic`** (Smooth/Bin co-add of successive batch-series
  members via `combine.coadd_member_windows`); also the browser "Re-fit as
  Co-added" action. See `docs/porting/fit-workflow-diagnostics/`.
- Background-run *correction* (frame-ratio scaling that does not produce a
  combined dataset) — already shipped in `data-reduction-parity` Phase 2; the
  two share `subtract_scaled_counts`.
- Event-mode arithmetic.
- ~~Symmetric / N-run signed co-subtract (see decision 1).~~ — **done on
  `feat/batch-arithmetic`** (`combine_runs(subtract_method="signed")` +
  "Subtract Selected (signed)…").

## See also

- [comparison.md](comparison.md) — WiMDA arithmetic vs Asymmetry, divergences.
- [implementation-options.md](implementation-options.md) — kernel shape,
  alignment, provenance, GUI wiring.
- [test-data.md](test-data.md) — verification corpus.
- [verification-plan.md](verification-plan.md) — how claims get checked.
</content>

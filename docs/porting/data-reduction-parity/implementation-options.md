# Data-reduction parity — implementation options and plan

Date: 2026-06-10. Options below were studied first; decisions were settled
with Ben (recorded per option), followed by the phased implementation plan.

## Scope decisions inherited from the umbrella

Out of scope, with rationale:

- **Extended deadtime models / count-loss fitting** — belongs with the
  count-domain fit machinery (`count-domain-fit-modes` project owns
  `Group.pas ccorrect`'s polynomial/power-law model panel and
  `paramDT`/SendToGroup).
- **Fitted-baseline Set BG / Unset BG** (`Analyse.pas:5877`) — specialist
  two-step workflow; deferred, revisit with count-domain modes.
- **Per-directory `default.mgp` / `default.exclude` auto-load** — dead code
  in current WiMDA itself (commented out, `Group.pas:2203–2215`); project
  grouping persistence + grouping templates cover the need. Possible later
  nicety; not a parity item.
- **ARGUS/KEK hardware fixers, multichannel N→32 mapping, KEK port select** —
  workarounds for WiMDA's fixed 32-histogram arrays and dead hardware eras;
  dropped (umbrella decision).
- **Run co-add / co-subtract** — Wave B `run-arithmetic` project. This
  project's background-run subtraction shares the frame-scaled count
  arithmetic seam (see Phase 2 design) but does not implement dataset
  combination.

## Option analyses

### O1 — alpha estimator uncertainty (Phase 1)

| Option | How | Pros | Cons |
|---|---|---|---|
| (a) Profile | Diamagnetic objective is a true χ²(α) (sum of squared standardised residuals against zero) ⇒ σ_α from Δχ² = 1; General needs an ad-hoc scale (its objective is a normalised scatter, not a likelihood) | Fast, standard | Not meaningful for the General objective without extra theory; two different error definitions across methods |
| (b) Poisson bootstrap | Resample per-bin counts F, B ~ Poisson(observed), re-minimise, σ_α = sd over ~200 replicas | Uniform across all three methods (incl. ΣF/ΣB); assumption-free; trivially fast at these data sizes (1-D bounded minimisation, ≤ few thousand bins) | Stochastic (seeded RNG for reproducibility); ~200× cost (still ≪ 1 s) |
| (c) Both | Bootstrap as the reported number; Δχ² = 1 profile as a test-time cross-check on the diamagnetic method | One honest, uniform reported σ_α plus an independent check | Slightly more test code |

**Recommendation: (c)** — report the bootstrap σ_α (seeded, n = 200) for all
three methods; verification asserts profile ≈ bootstrap on the diamagnetic
method (they agree when the χ² surface is parabolic, which doubles as a
sanity check on the implementation).

**Decision (Ben, 2026-06-10): (c)** — bootstrap reported, profile cross-check in tests.

### O2 — background-run subtraction data model (Phase 2)

Where the reference run attaches and how it persists:

| Option | How | Pros | Cons |
|---|---|---|---|
| (a) Grouping-level payload | `background_run` key in the grouping dict: `{run_number, source_file, frames, scale}` snapshot; reduction resolves histograms on demand (loaded project dataset first, loader fallback) with caching | Background is a reduction setting exactly like deadtime — same persistence (`grouping_overrides`), same apply-to-selected and template inheritance, scriptable from core | Grouping dict grows another structured key |
| (b) Project-schema dataset field | New per-dataset `background_reference` beside `metadata_overrides` | Visible at project level | New schema concept for what is semantically a grouping/correction choice; bypasses the grouping template flow; more shared-file churn in `schema.py` |
| (c) Transient GUI bake-in | Load BG run, subtract, store result | No model changes | Destroys provenance; violates the explicit-corrections invariant; not scriptable |

**Recommendation: (a)** — with the frame-scaled subtraction helper exposed as
a clean core function so Wave B `run-arithmetic` can reuse the arithmetic
(shared seam, no dependency).

**Decision (Ben, 2026-06-10): (a)**.

### O3 — binning-mode surfacing (Phase 3)

| Option | Where | Pros | Cons |
|---|---|---|---|
| (a) Grouping dialog only | Mode dropdown (Fixed/Variable/Constant-error) + bin0/bin10 fields beside the existing bunch factor; persisted grouping keys | One authoritative home (grouping owns `bunching_factor` today); matches WiMDA; plot/fit pipelines read the same keys | Changing display binning means opening the dialog |
| (b) Plot controls only | Per-plot binning controls | Quick display tweaks | Splits reduction state from grouping persistence; fit input then differs from grouping contract |
| (c) Both | Dialog authoritative + plot-side quick override | Convenient | Two sources of truth; the dedicated UI-polish pass is the right time to add conveniences |

**Recommendation: (a)** — grouping dialog only this pass; a plot-side
shortcut can come with the later UI-polish pass if wanted.

**Decision (Ben, 2026-06-10): (a)**.

### O4 — detector-exclusion persistence and schematic UX (Phase 3)

Persistence is uncontroversial given the invariants: an
`excluded_detectors` sorted list (1-based detector ids) in the grouping
dict, applied as zero-weight at grouping time (raw histograms intact, no
reload — divergence D10), persisted through `grouping_overrides` and the
template flow like every other grouping key. The open choice is the UX:

| Option | How | Pros | Cons |
|---|---|---|---|
| (a) Detector-layout dialog toggle + text editor | The existing detector schematic (detector_layout_dialog) gains an "exclude" interaction (excluded detectors rendered struck-through/greyed); grouping dialog gets a compact text field accepting WiMDA-style ranges ("1,5,10-15") | Reuses the existing schematic; both visual and scriptable entry; matches the brief's sketch | Layout dialog grows a second mode (selection vs exclusion) — needs a clear mode toggle |
| (b) List editor only | Text field + parsed chip list in the grouping dialog | Minimal | No visual identification of dead/hot detectors — the schematic is where users *see* the bad detector |
| (c) Separate exclusion dialog | New dialog with schematic copy | No mode overload | Duplicate schematic; gold-plating |

**Recommendation: (a)**.

**Decision: (a)** — settled by the umbrella brief's GUI sketch (schematic
click-to-exclude plus a list editor); not re-asked.

### O5 — period-mapping dialog shape (Phase 3)

Core is settled (count-level `map_periods` with
`{period → red|green|ignore}`, existing 2-period modes become the trivial
mapping). The dialog:

| Option | Shape | Pros | Cons |
|---|---|---|---|
| (a) Matrix dialog off the period selector | "Map periods…" button beside the existing RG radios (shown when period_count > 2, or always for period data); modal table — one row per period (name, frames, tag) × radio columns {Red, Green, Ignore}; dwell periods locked to Ignore | Matches WiMDA's mental model and the brief; existing 2-period radio UX untouched; arbitrary N | One more dialog |
| (b) Inline matrix in the grouping dialog | Table embedded in the period section | No extra dialog | Grouping dialog is already dense; N rows × 3 columns of radios bloat it |
| (c) Two list fields (Mantid style) | "Red periods", "Green periods" comma/dash text lists | Compact, scriptable | Less discoverable; no per-period metadata display (frames/tags are the cue for which period is which) |

**Recommendation: (a)**, with the parsed-list representation underneath so
(c)'s scriptability exists at the core API level anyway.

**Decision (Ben, 2026-06-10): (a)**.

## Decision summary (Ben, 2026-06-10)

O1 bootstrap σ_α + profile cross-check · O2 grouping-level `background_run`
payload · O3 binning modes in the grouping dialog only · O4 schematic
click-to-exclude + list editor (per brief) · O5 period-mapping matrix
dialog. Verification: transcribed oracles only. Docs: new
`docs/user_guide/data_reduction/` section, fit_functions page style.

## Implementation plan

Three phases, in order; each ends with `python tools/harness.py validate`
green, a milestone commit, and a main-mergeable branch. Shared-file
discipline: additions to `core/transform/__init__.py`, `docs/porting/
index.json` and user-guide toctrees go at the end of the relevant block;
no project-schema version bump is expected (all new state is optional,
additive grouping-dict keys with safe defaults — round-trip tests must
confirm old projects load unchanged).

Conventions throughout: `MUON_LIFETIME_US = 2.1969811`
(`core/utils/constants.py:33` — matches WiMDA's `tau_mu`); estimator
actions follow the deadtime-Estimate precedent (compute from the reference
dataset, apply to all selected, inherit via grouping templates); corpus
tests use the `skipif` pattern; GUI tests need `QT_QPA_PLATFORM=offscreen`.

### Phase 1 — alpha estimation

Core (`core/transform/asymmetry.py`):

1. `AlphaEstimate` dataclass: `alpha`, `alpha_error`, `method`
   (`"diamagnetic" | "general" | "ratio"`), `n_bins_used`,
   `objective_value`, `ok`, `message`.
2. Objective helpers transcribed from `Group.pas:1775` (see comparison.md
   §1): `_diamagnetic_objective(alpha, f, b)` = Σ(A/σ_A)² with exact
   Poisson σ_A; `_general_objective(alpha, f, b, time_us)` = weighted
   relative scatter of (f/√α + b√α)·e^{t/τ_μ}. Bin filter: keep
   `f > 0 and b > 0` (parity).
3. `estimate_alpha_detailed(forward, backward, *, method, time_us=None,
   first_good_bin=None, last_good_bin=None, n_bootstrap=200, seed=0)
   -> AlphaEstimate`: bounded `minimize_scalar` on ln α over
   [ln 0.01, ln 100]; `time_us` (bin centres relative to t0) required for
   `general`; `ratio` reuses the ΣF/ΣB sum. σ_α = std of re-estimates over
   seeded Poisson resamples of the counts (deadtime-corrected float counts
   resampled as Poisson(max(c, 0))). Degenerate inputs → `ok=False`,
   α = 1.0, mirrors current `estimate_alpha` fallback.
4. Keep `estimate_alpha` exactly as-is (back-compat shim documented as the
   `ratio` method). Export the new names at the end of
   `transform/__init__.py`.

GUI (`gui/windows/grouping_dialog.py`):

5. The Estimate control grows a method combo (default Diamagnetic) with
   one-line descriptions: "Diamagnetic — TF calibration run; minimises the
   weighted asymmetry"; "General — works on relaxing LF/ZF data;
   lifetime-corrected count balance"; "Count ratio — ΣF/ΣB, TF only
   (legacy/Mantid)". Result label shows `α = X.XXXX(YY)` and the method;
   tooltip carries the explanation. Vector-mode per-axis estimates use the
   same selected method. New grouping keys written with the payload:
   `alpha_method`, `alpha_error`, `alpha_reference_run` (optional,
   additive).

Tests (`tests/test_alpha_estimation.py`, new; `tests/test_grouping_dialog.py`
additions): WiMDA grid-walk transcription oracle (±0.001 agreement);
synthetic truth for all three methods (oscillating/relaxing/flat P(t));
bootstrap-vs-profile agreement on diamagnetic; ratio-bias contrast on
relaxing data; σ_α coverage; degenerate inputs; dialog method round-trip +
payload keys; corpus `skipif` tests (nickel TF run stability under
bunching; HIFI 118228–118232 General-α consistency).

Docs: `docs/user_guide/data_reduction/index.rst` +
`alpha_calibration.rst` (when-to-use register per method, the T20
fix-then-hold workflow, why ΣF/ΣB biases on relaxing data, fitted-α
cross-reference to the future count-domain modes); one toctree line at the
end of `docs/user_guide/index.rst`. Update study docs status. Milestone
commit; full validate green.

### Phase 2 — backgrounds

Core (`core/transform/background.py`):

1. `fit_tail_background(counts, *, bin_width_us, t0_bin, last_good_bin,
   fit_start_bin=None, n_bootstrap=0) -> TailFitResult(rate_per_us, error,
   amplitude, window, ok, consistent_with_zero, message)`. Model: the
   bin-integrated exponential + flat from `Group.pas BGfit` (λ fixed at
   1/τ_μ); estimation by Poisson MLE (minimise the deviance, two free
   parameters: ln-amplitude and flat rate ≥ 0), σ from the observed-
   information matrix at the optimum. Default window: late half of the
   good-bin range on raw grouped counts. `consistent_with_zero` when
   rate < 2σ.
2. Mode model: new optional grouping key `background_mode`
   (`"none" | "fixed" | "range" | "tail_fit" | "reference_run"`); absent →
   derived from today's keys (back-compat). `available_background_modes(
   metadata, source_file) -> list[str]` replaces the binary gate: `range`
   requires a pre-t0 region (continuous sources — current PSI/LEM
   detection), `tail_fit` and `fixed` available everywhere (tail-fit is
   the pulsed-source answer), `reference_run` always offered.
   `supports_background_correction` stays as a thin shim over the new
   function until the dialog migrates.
3. `apply_grouped_background_correction` learns `tail_fit` (estimate per
   group from the same grouped counts, subtract rate·width per bin, add
   the rate uncertainty in quadrature to per-bin errors — method string
   `"tail_fit"`).
4. Background-run subtraction, count-level per detector:
   `subtract_scaled_counts(histograms, reference_histograms, scale)
   -> (corrected_counts, count_errors)` with σ² = N + scale²·N_ref —
   the shared seam for Wave B `run-arithmetic`. Sample-side t0 alignment
   per detector (WiMDA convention); both runs receive the same deadtime
   treatment before subtraction (divergence D6). Reduction feeds the
   result through `compute_asymmetry_with_count_errors`.
5. `background_run` grouping payload: `{run_number, source_file,
   good_frames_reference, good_frames_sample, scale}` snapshot; the scale
   is recomputed from live good-frames when both runs are available and
   falls back to the snapshot. Resolution helper `load_background_run(
   payload)` lives in `core/io` (transform stays io-free; the GUI/scripts
   resolve, transform subtracts).

GUI: the background section becomes a mode selector (None / Fixed /
Range / Tail fit / Background run…), entries enabled per
`available_background_modes`; Tail fit shows the fitted rate ± σ and a
"consistent with zero" note; Background run… opens a small picker (choose
a loaded dataset or browse; displays reference run number, frame counts
and the computed scale). Apply path in `mainwindow.py` extended for the
new payload.

Tests: transcribed `estBG` oracle (√N weights + ≤ 4-count deletion +
late-half window) vs MLE on moderate-count synthetic data; low-count MLE
superiority; bin-integration invariance; gating matrix per source type;
self-subtraction → zeros with √(2N) errors; frame-ratio arithmetic;
project round-trip of `background_run`; corpus tail fits (ISIS small-p₂,
PSI tail-vs-pre-t0 agreement). Docs:
`docs/user_guide/data_reduction/backgrounds.rst` (the two-backgrounds
distinction up front; duty-factor framing for pulsed sources; when each
mode applies; reference-run workflow). Milestone commit; validate green.

### Phase 3 — binning, t0, exclusion, periods

Binning (`core/transform/rebin.py` + reduction path):

1. Edge generators: `variable_bin_edges(t_start, t_end, bin0_us,
   bin10_us)` (width = bin0·(bin10/bin0)^(t/10 μs)) and
   `constant_error_bin_edges(t_start, t_end, bin0_us)` (width =
   bin0·e^{t/τ_μ}); both snap to integer raw-bin boundaries the way
   WiMDA's accumulation loop does (accumulate raw bins until the running
   edge passes the target).
2. Count-level aggregation: `rebin_counts_to_edges(counts, bin_width_us,
   t0_bin, edges) -> (centres, widths, summed_counts)`; the grouped
   reduction (new helper beside `group_forward_backward` in
   `core/transform/grouping.py`) switches on `binning_mode`
   (`"fixed"` default | `"variable"` | `"constant_error"`, with
   `bin0_us`/`bin10_us` keys): non-fixed modes sum F/B counts onto edges
   first, then form asymmetry per output bin (counts-then-ratio order —
   comparison.md §3). Fixed mode keeps today's path bit-for-bit.
3. Consumers: plot/reduction path (`core/representation/time.py`,
   `mainwindow.py` reduce calls) honour the mode; Fourier, MaxEnt and
   grouped count-domain fitting keep using `bunching_factor` only and
   document the restriction (uniform sampling); GLE export header line
   (`plot_panel.py:4388`) states the active mode and parameters.
4. GUI: binning mode dropdown + Initial bin (μs) / Bin at 10 μs (μs)
   fields beside the bunch factor; bunch factor enabled only for Fixed
   (WiMDA enablement pattern); defaults 0.08/0.25 μs (WiMDA's).

t0 search (new `core/transform/t0.py`):

5. `find_t0(counts, *, pulsed) -> T0Estimate(t0_bin, strategy, peak_bin,
   message)`: continuous → argmax (prompt peak, ties → earliest);
   pulsed → half-maximum crossing of the leading edge (linear
   interpolation, rounded), full-histogram scan. `find_t0_for_run(
   histograms, metadata)` picks the strategy from facility metadata
   (ISIS ⇒ pulsed; PSI/TRIUMF ⇒ continuous) and returns per-histogram
   estimates plus a consensus (median) with spread.
6. GUI: "Find t0" button beside the t0 spinner — runs on the reference
   dataset, reports consensus ± spread and per-detector outliers, fills
   the override spinner; the user still applies (never silently
   overwrites loader values).

Detector exclusion (`core/transform/grouping.py`):

7. Optional grouping key `excluded_detectors` (sorted 1-based ids).
   `resolve_group_indices`/`apply_grouping*`/`group_forward_backward`
   drop excluded detectors from every group sum (zero-weight; raw
   histograms untouched — D10). Deadtime estimate/calibration and alpha
   estimation see the exclusion automatically through the grouped counts.
8. `parse_detector_list(text)` / `format_detector_list(ids)` accepting
   WiMDA-style "1,5,10-15" (ranges either direction).
9. GUI: text field in the grouping dialog's detector section; the
   detector-layout dialog gains an Exclude mode toggle —
   click-to-exclude, excluded detectors greyed/struck in the schematic.

Period mapping (`core/io/periods.py` + new dialog):

10. `map_periods(period_histograms, period_good_frames,
    period_dead_time_us, mapping) -> mapped payload`, where `mapping` is
    `{period_number (1-based): "red" | "green" | "ignore"}`: counts and
    good-frames summed per set (count-level, before corrections);
    per-detector deadtimes verified equal across periods (else
    frame-weighted mean with a warning). Output feeds the existing
    R/G machinery (`select_period_histograms` /
    `combine_period_asymmetry`) so {1→red, 2→green} reproduces today's
    2-period path bit-for-bit. Grouping key `period_mapping`
    (string-keyed dict in JSON).
11. Loader (additive): `core/io/nexus.py` reads per-period type/tag/
    sequence metadata where present (`period_modes`, `period_tags`,
    `period_sequences` metadata keys) so the dialog can lock dwell
    periods to Ignore and show frames/tags.
12. GUI: new `gui/windows/period_mapping_dialog.py` — modal matrix, one
    row per period (label, frames, tag), radio columns Red/Green/Ignore,
    dwell rows locked; launched from a "Map periods…" button beside the
    RG radios (shown when period_count > 2); `mainwindow.py` reduction
    uses the mapping when present.

Tests: `tests/test_rebin_modes.py`, `tests/test_t0_search.py`,
`tests/test_detector_exclusion.py`, `tests/test_period_mapping.py` (new)
plus dialog/mainwindow additions — contents per verification-plan.md §3
(edge laws + WiMDA-formula 0.2% check, flat-σ property, provenance
invariance, Fourier/MaxEnt fixed-mode guard, t0 synthetic + corpus
recovery, exclusion equivalence + parser fuzz + round-trip, mapping
bit-for-bit + photo-μSR corpus). Docs:
`docs/user_guide/data_reduction/{binning,t0_search,detector_exclusion,
period_mapping}.rst`. Milestone commit; validate green.

### File-by-file touch list

| File | Phase | Change |
|---|---|---|
| `src/asymmetry/core/transform/asymmetry.py` | 1 | objectives, `AlphaEstimate`, `estimate_alpha_detailed`, bootstrap |
| `src/asymmetry/core/transform/__init__.py` | 1–3 | exports, end of block (shared file — additive only) |
| `src/asymmetry/gui/windows/grouping_dialog.py` | 1–3 | method combo + α result label; background mode selector + run picker; binning mode controls; exclusion field; Find t0; Map periods button |
| `src/asymmetry/core/transform/background.py` | 2 | `fit_tail_background`, mode model, `available_background_modes`, `subtract_scaled_counts` |
| `src/asymmetry/core/io/__init__.py` (or loader helper module) | 2 | `load_background_run(payload)` |
| `src/asymmetry/gui/mainwindow.py` | 2–3 | apply/reduction paths: background payload, binning mode, period mapping |
| `src/asymmetry/core/transform/rebin.py` | 3 | edge generators + count aggregation |
| `src/asymmetry/core/transform/grouping.py` | 3 | binned-reduction helper; `excluded_detectors` in group resolution; parser/formatter |
| `src/asymmetry/core/transform/t0.py` (new) | 3 | `find_t0`, `find_t0_for_run` |
| `src/asymmetry/core/io/periods.py` | 3 | `map_periods` + mapping validation |
| `src/asymmetry/core/io/nexus.py` | 3 | additive period metadata (modes/tags/sequences) |
| `src/asymmetry/core/representation/time.py` | 3 | binning-mode-aware reduction |
| `src/asymmetry/gui/windows/detector_layout_dialog.py` | 3 | Exclude mode toggle + rendering |
| `src/asymmetry/gui/windows/period_mapping_dialog.py` (new) | 3 | matrix dialog |
| `src/asymmetry/gui/panels/plot_panel.py` | 3 | GLE header line for binning modes (minimal) |
| `tests/test_alpha_estimation.py` (new), `tests/test_grouping_dialog.py` | 1 | per verification-plan §1 |
| `tests/test_background.py`, `tests/test_background_run.py` (new), `tests/test_mainwindow_additional.py` | 2 | per verification-plan §2 |
| `tests/test_rebin_modes.py`, `tests/test_t0_search.py`, `tests/test_detector_exclusion.py`, `tests/test_period_mapping.py` (new) | 3 | per verification-plan §3 |
| `docs/user_guide/data_reduction/*.rst` (new section), `docs/user_guide/index.rst` | 1–3 | pedagogical pages, one per feature; toctree line at end |
| `docs/porting/data-reduction-parity/*` | 1–3 | status/outcome updates per phase |

Not touched: `core/project/schema.py` (no version bump needed — optional
additive grouping keys only; if implementation finds a forced default
migration, bump per the v4 precedent and record here).

### Recorded follow-ons

Status note (2026-06-10, `feat/data-reduction-followups`): the completion pass
resolved the exclusion chokepoint, core reference-run resolution, period-mapped
project reload, the plot-panel mask, and the umbrella label fix; each is marked
**DONE** inline below. Items still deferred (plot-side quick binning, live WiMDA
spot-checks, α-as-fit-parameter, `run-arithmetic`, TF-phase t0, integral-path
period mapping, Fourier tail-fit) remain open and are owned by the projects
named.

- **DONE** — Exclusion chokepoint. Detector exclusion now resolves through a
  single exclusion-aware resolver, `effective_group_indices(grouping,
  group_id, n_histograms=…)` in `core/transform/grouping.py`. Every reduction
  path goes through it: `group_forward_backward`, `fitting/
  grouped_time_domain`, `fourier/grouped` (its private
  `_normalize_group_entries` removed), `mainwindow` F/B resolution, and the
  `plot_panel` saturation mask. The grouping dialog's estimate / Find-t0 /
  tail-fit-preview / accept paths route their 0-based group filtering through
  the shared `filter_excluded_indices` / `excluded_detector_indices`
  primitives. `resolve_group_indices` is now reserved for non-reduction uses
  (synthetic-run generation, NeXus writing), so forgetting exclusion at a new
  reduction call site is no longer possible.
- Plot-side quick binning control — deferred to the dedicated UI-polish
  pass (O3).
- Live WiMDA spot-check values for α/tail-fit on corpus runs — optional
  later validation (oracle decision).
- α as a free fit parameter (the WiMDA manual's "most accurate way") —
  `count-domain-fit-modes` project.
- `run-arithmetic` (Wave B) to reuse `subtract_scaled_counts` for
  co-subtract; coordinate when it starts.
- TF-phase-based fine t0 calibration (textbook §15.3: phase-vs-frequency
  slope q ⇒ Δt0 = q/360 μs) — possible future refinement beyond WiMDA
  parity.
- Period mapping in the time-integral/ALC scan path (Mantid
  `PlotAsymmetryByLogValue` red/green parity) — periods.py mapping makes
  it possible; wire into `integral.py` consumers on demand.
- Umbrella docs correction: the "EMU LF series" in
  `wimda-parity-gap/test-data.md` is HIFI 118222–118240 (fix the umbrella
  table when this project merges).
- (Phase 3 implementation note) For 3+-period files the loader still
  returns one dataset per period; **Map periods…** builds a new combined
  dataset from those siblings via `combine_mapped_periods` and records
  `period_mapping` in its grouping. Re-deriving the mapped dataset when a
  project is reloaded is a follow-on (the mapping persists, the combined
  dataset is rebuilt by re-running Map periods).
- (Phase 2/3 implementation note) The grouped *Fourier* input path keeps
  its legacy continuous-source-only background gate; offering tail-fit
  there is a follow-on for `frequency-domain-finishers`.

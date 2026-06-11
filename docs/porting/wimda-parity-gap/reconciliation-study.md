# Collision reconciliation study — verdicts

Date: 2026-06-11, branch `study/collision-reconciliation` (on the Wave A
closeout, main at PR #44). This session investigated every flag in the
closeout watchlist ([wave-a-closeout.md](wave-a-closeout.md) §4) — the
closeout deliberately recorded flags without investigating them — and agreed
a verdict for each with Ben in-session. Verdicts use the programme
vocabulary: **UNIFY** (merge onto one implementation), **PROMOTE-PATH** (add
a suggest-only reconcile action mirroring `promote_deadtime_to_grouping`),
**API-ONLY** (keep scriptable, remove GUI surface), **MENU-DEMOTE**,
**DOCUMENT** (which-one-when guidance), **DEFER** (recorded trigger).

Physics calls cite Blundell, De Renzi, Lancaster and Pratt, *Muon
Spectroscopy: An Introduction* (OUP, 2022) — "the textbook" below.

Headline: **no flag required API-ONLY or MENU-DEMOTE.** Where the closeout
feared N-way duplication, investigation found either literal duplicates
(mechanical UNIFYs), missing promote paths around a sound single chokepoint,
or distinct physical quantities needing guidance rather than pruning. Three
flags were partially **refuted** (F3's silent double-subtraction, F4's third
path, F10's "three" containers); one new flag was found (NEW-R1).

## 1. Verdict table

Phases refer to [reconciliation-plan.md](reconciliation-plan.md).

| Flag | Verdict | Rationale | Cost | Phase |
|---|---|---|---|---|
| N1 | UNIFY | `_rebin_group_counts` byte-identical ×2; shared `rebin_counts()` beside value-domain `rebin()` | S | 1 |
| N2 | UNIFY | `_optional_float`, `_group_names` duplicated fourier/maxent; hoist to shared homes | S | 1 |
| F2 | UNIFY | GUI re-implements core field-metadata resolver; publicise core, GUI converts units | S | 1 |
| F12 | UNIFY | Hand-mirrored constants; single authority in core, docs imports (direction forced: core can't import docs) | S | 1 |
| F5 | PROMOTE-PATH | Count-fit t0 is the only fitted t0 with no write-back; Fourier `t0_offset_us` is a distinct phase knob, never promotes | M | 2 |
| F7 | PROMOTE-PATH + DOCUMENT | Count-fit α (best statistics) is the only route that can't persist; estimator combo + existing docs are sound, no pruning | S–M | 2 (+4) |
| F6 | UNIFY | Confirmed divergent: MaxEnt apply bypasses the chokepoint (no before/after, broadcast, drops model terms) | S | 2 |
| F3 | DOCUMENT + guards | Double-subtraction **refuted** — modes mutually exclusive, applied once; real gaps are invisible inheritance + missing ladder docs | S–M | 2, 3, 4 |
| N3 | DOCUMENT + guard + PROMOTE-PATH | Count fit always consumes raw counts (bias scenario is interpretive, not mechanical); add guard note + background promote | S–M | 2 (+4) |
| F4 | UNIFY | Two real paths, not three; merge two independent checkboxes onto one three-way control; surface the <5 G silent fallback | S–M | 3 |
| F8 | DOCUMENT + relabel | Five exclusions, three semantics — mechanisms must not merge; glossary + two disambiguating relabels | S | 2, 3, 4 |
| F9 | DEFER | Forward collision; co-subtract chokepoint reuse belongs in the run-arithmetic brief (Wave B), not a phase here | — | brief note (4) |
| F10 | DOCUMENT + scheduled UNIFY | Two of "three" containers are already one (both FitSeries); window decorations → `FitSeries.extra`; stateless-window refactor deferred | M | 4 + 5 |
| F1 | DOCUMENT | Equivalence exact and oracle-tested, but monolith is friction-free (no c₁=0 fix, BG≥0 bound) and crisp guidance exists | S | 4 |
| F11 | DOCUMENT | Cross-link half-missing (alc_mode → parameter_trending); one complementarity paragraph | S | 4 |
| F13 | DOCUMENT | Burg / FFT / MaxEnt pages have zero cross-links; write the estimator-triad when-to-use | S | 4 |
| N5 | DOCUMENT | fit_quality / result_summary / pull-diagnostic lack a shared "assessing a fit" entry point | S | 4 |
| N6 | DOCUMENT | Fourier phases and MaxEnt phase exchange are separate stores by design; trigger recorded for a pull action | S | 4 |
| N4 | DEFER | Registry naming formalisation belongs to python-user-functions' registration API (Wave B); ARCHITECTURE note now | S | 4 (note) |
| NEW-R1 | UNIFY (persist) | Count-fit exclude window is the only exclusion that doesn't round-trip through `.asymp` | S | 2 |

## 2. Group A — mechanical duplicates (N1, N2, F2, F12)

**Decision (Ben, 2026-06-11): UNIFY all four**, as one quick mechanical
package (plan Phase 1).

### N1 — `_rebin_group_counts` ×2

Confirmed byte-identical bodies in
`core/fitting/grouped_time_domain.py:611` and `core/fourier/grouped.py:25`
(docstrings differ). Important nuance the closeout missed:
`core/transform/rebin.py`'s existing `rebin()` is **value-domain** (mean of
values, errors in quadrature ÷ factor) while the duplicated helper is
**count-preserving** (sum of counts, mean of times) — so "neither uses
transform/rebin.py" is correct behaviour, not the bug. The fix is a shared
`rebin_counts()` in `transform/rebin.py` beside `rebin()`, imported by both
call sites.

### N2 — small helper duplicates

`_optional_float` (`fourier/spectrum.py:194` ≡ `maxent/engine.py:76`) and
`_group_names` (`spectrum.py:254` ≡ `engine.py:484`). Hoist: coercion helper
to `core/utils`, grouping-name introspection beside the grouping helpers.
The GUI's `_parse_optional_float` (`maxent_panel.py:388`) is a text-parsing
variant, a different concern — left in place, noted in the helper docstring.

### F2 — reference-field resolver ×2

`plot_panel._frequency_reference_for_dataset` (`plot_panel.py:664`)
re-implements the dataset-metadata → run-metadata `field` lookup of core
`_reference_field_gauss` (`fourier/spectrum.py:272`), differing only in the
trailing unit conversion. Fix: publicise the core resolver; the panel calls
it and converts. Same lookup order, so behaviour is pinned by construction.

### F12 — two archetype constant tables

`core/simulate_presets.py` hand-mirrors the rounded textbook constants of
`docs/screenshots/data/archetypes.py` (γ_μ/2π = 0.01355 MHz/G, Δ_Ag, T_C,
…) and says so in a comment. Import direction is forced — core cannot import
docs; the docs module already imports `asymmetry.core.simulate` — so the
shared subset gets one public authority in core and the docs module imports
it. Doc-only constants (MgB₂/YBCO parameters, the legacy 2.197 µs lifetime
pin that keeps screenshots byte-stable) stay local. Screenshot byte-stability
is guarded by the existing CI suite. Note these rounded values are
*deliberately* distinct from the CODATA `MUON_GYROMAGNETIC_RATIO_MHZ_PER_T =
135.538817` in `core/utils/constants.py`; unification is within the rounded
table, not across it.

## 3. Group B — calibration promote paths (F5, F7, F6)

The textbook is explicit that these calibrations are per-sample, per-setup
quantities — α "is dependent on sample position and detector efficiencies"
and "needs to be determined for each sample", and the time zero "relevant
for the analysis is the beginning of the spin dynamics", not necessarily the
prompt-peak feature in the histogram. Fitted values from the sample's own
data are therefore legitimate calibrations to persist; the deadtime promote
pattern (suggest-only, before/after, provenance) is the correct shape for
all of them.

### F5 — four t0 surfaces

**Decision (Ben, 2026-06-11): PROMOTE-PATH** for the count-fit t0, plus a
documentation note separating the Fourier knob.

Evidence: the four surfaces are not four competing implementations —

1. **t0 search** (`core/transform/t0.py`) already has a clean suggest-only
   flow into the grouping dialog spinner ("Find t0",
   `grouping_dialog.py:1900`).
2. **`grouping["t0_bin"]`** is the persisted authority (integer bins,
   run-level), consumed by every reduction.
3. **Fourier `t0_offset_us`** (`fourier/fft.py:208`) is a **physically
   distinct** post-FFT phase correction — it multiplies the spectrum by
   exp(−i·2πf·t₀) and never touches the time axis. Recipe-scoped, correctly
   so; it must never promote. The study's documentation note makes this
   explicit.
4. **Count-fit `t0` nuisance** (`core/fitting/count_domain.py`, reserved
   name, µs offset, per-group) is the only *fitted* t0 with no write-back —
   the exact analogue of the pre-#41 deadtime gap.

Design constraints recorded for the implementing session: fitted t0 is a
continuous µs offset, `t0_bin` is an integer bin index — promotion converts
via the bin width, rounds to the nearest bin, and must disclose the sub-bin
residual in the before/after display; fitted t0 is per-group while `t0_bin`
is run-level — promotion uses the fitted group's value run-wide and the
suggest dialog says so. Provenance keys mirror α's
(`t0_method`/`t0_reference_*`). Suggest-only, never auto-apply.

Options considered: DOCUMENT-only (rejected — leaves the best estimator
unable to persist, the same asymmetry F7 fixes), DEFER (rejected — the
pattern is established and cheap to mirror).

### F7 — four α routes

**Decision (Ben, 2026-06-11): PROMOTE-PATH + DOCUMENT. No estimator is
relegated.**

Evidence refines the flag: the three grouping-dialog estimators occupy *one
combo box and one button* (`grouping_dialog.py:471–502`), and
`docs/user_guide/data_reduction/alpha_calibration.rst` already carries crisp
which-one-when guidance (diamagnetic preferred whenever TF data exist;
General only for relaxing LF/ZF runs; count-ratio a quick TF cross-check,
biased on relaxing data). The API-ONLY test — similar mechanisms *and* no
crisp guidance — therefore fails on its second condition, and each route has
a distinct data regime. Despite the session's aggressive-pruning default,
four user-facing routes are defensible.

The genuine gap is the asymmetry of persistence: all three estimators write
`grouping["alpha"]` (+ `alpha_method`/`alpha_error`/`alpha_reference_run`
provenance), while the count-fit α-free mode
(`count_domain.py:fit_fb_alpha`, √α-parameterised, full Poisson likelihood
with proper (α, amplitude) covariance — statistically the best route) is
displayed (`fit_panel.py:3825`) and discarded. Fix:
`promote_alpha_to_grouping` writing the same keys with
`alpha_method="count_fit"`, suggest-only with before/after, wired beside the
existing deadtime promote in the multi-group fit window. DOCUMENT part:
`alpha_calibration.rst` gains the count-fit α as the fourth route.

MINOS errors on α stay with Wave B fit-workflow-diagnostics — not
double-scheduled here.

### F6 — deadtime write paths

**Decision (Ben, 2026-06-11): UNIFY.** Confirmed divergent, not
hypothetical:

- Count-fit path: `promote_deadtime_to_grouping`
  (`core/transform/deadtime.py:235`) — per-group detector indices, additive
  option, returns before/after, records `deadtime_model`/
  `deadtime_model_terms`, GUI says "Re-reduce the run to apply".
- MaxEnt path: `_on_maxent_apply_deadtime` (`gui/mainwindow.py:5111`) writes
  `grouping["dead_time_us"]`/`deadtime_method="maxent_fit"`/
  `deadtime_correction` **inline**, broadcast to all detectors, no
  before/after, no re-reduce message, and would silently drop model terms if
  MaxEnt calibration ever grew them. No functional tests cover the handler.

Fix: extend the chokepoint to accept per-detector value lists (MaxEnt
produces one value per detector; the current signature takes a scalar) and
route the MaxEnt handler through it. The distinct `"maxent_fit"` provenance
label is kept — both labels are semantically inert in reduction
(`has_file_deadtime` only distinguishes file-sourced deadtimes). The MaxEnt
flow gains before/after display and the re-reduce message for free.

## 4. Group C — frequency-domain nuisance handling (F3, N3, F4, F8, N6)

### F3 + N3 — background stories

**Decision (Ben, 2026-06-11): DOCUMENT + guards, plus PROMOTE-PATH for the
fitted count background.**

The closeout's central fear — silent double-subtraction — is **refuted in
code**:

- The four pre-FFT background modes (`fixed` / `range` / `tail_fit` /
  `reference_run`, single chokepoint `core/transform/background.py`) are
  mutually exclusive per run, and the Fourier input path
  (`core/fourier/grouped.py:205`) rebuilds grouped counts from raw
  histograms, applying the grouping's correction exactly once — PR #44's
  inherit-from-grouping design holds (no FFT-only background path exists).
- The σ-clip baseline (`core/fourier/conditioning.py`) removes the
  **spectral noise floor** on the display channel, post-FFT.
- MaxEnt SpecBG (`core/maxent/specbg.py`) removes a **zero-frequency
  pseudo-Voigt central peak**, display-only, ZF/LF mode.
- The count-fit `background` nuisance fits the **time-domain flat
  background on raw counts** — verified: `_count_group_context`
  (`grouped_time_domain.py:240`) prepares histograms with deadtime only,
  never background correction.

These are different physical manifestations of (at most) one underlying
quantity — the textbook's steady background count from uncorrelated detector
hits at continuous sources — handled at different pipeline stages. Stacking
is coherent; the worst case is mild redundancy (a pre-FFT-subtracted
background leaves a smaller central peak for SpecBG to model).

What survives of the flags:

1. **Invisible inheritance** — the Fourier panel gives no hint that a
   grouping background correction is active pre-FFT. Guard: a read-only
   status line on the panel ("Background: tail-fit, inherited from
   grouping").
2. **The count-fit interpretive trap** — a user who believes the grouping
   correction reaches the fit may fix `background = 0` and bias N0/α (the
   raw counts still contain the background). Guard: an informational note
   in the count-fit UI when the grouping has `background_correction` on.
3. **No ladder documentation** — a user-guide page stating which stage
   removes what and when to enable each (time-domain flat background →
   zero-frequency feature; σ-clip → statistical floor; SpecBG → residual
   ZF/LF central peak).
4. **Promote symmetry** (Ben opted in): the fitted count-domain background
   is a measurement of the same flat background the grouping's `fixed` mode
   stores — `promote_background_to_grouping` (suggest-only, before/after,
   per-group values, `background_mode="fixed"` + provenance) completes the
   deadtime/α/t0/background promote family.

### F4 — diamagnetic-line paths

**Decision (Ben, 2026-06-11): UNIFY the control.**

The flag's "three paths" is two: time-domain fit-and-subtract
(`core/fourier/diamag.py`, Conditioning-group checkbox, also reports the
fitted field as a diagnostic) and the post-FFT band-zero
(`fourier/spectrum.py:311`, Exclusions-group checkbox, γ_μ·B ± half-width).
Generic spectral peak fitting exists (`core/fitting/spectral.py`) but is not
a panel path and was never part of the co-enable problem.

Today the two checkboxes are independent; co-enabling is harmless but
incoherent (the band zeroes what the subtraction already removed), and the
subtract path **silently no-ops below 5 G seed field**
(`spectrum.py:405–420`), leaving the band as an undisclosed fallback.

Fix: one mutually-exclusive three-way control — *Diamagnetic line: Leave /
Fit & subtract / Exclude band* — with the fit-failure fallback surfaced in
status text. Both `.asymp` keys (`remove_diamag`, `diamag_exclusion`) remain
readable; legacy projects with both set load as "Fit & subtract" with the
band noted. When-to-use (Phase 4 docs): subtract preferred for correlation /
A_μ work since it preserves neighbouring bins and reports the fitted field;
band as the robust fallback for lines too strong or distorted to fit.

### F8 — "exclusion" overload

**Decision (Ben, 2026-06-11): DOCUMENT + relabel.** Mechanisms must *not*
merge; the five features operate on different domains with deliberately
different semantics:

| Feature | Domain | Semantics | Parameterisation |
|---|---|---|---|
| Count-fit exclude window | time bins | hard drop from fit | (t₁, t₂) µs |
| MaxEnt exclude window | time points | σ-inflate ×10⁸ (grid kept) | (t₁, t₂) µs |
| Fourier exclusion ranges | frequency bins | hard zero (display) | centre ± half-width |
| Diamag band (→ F4) | frequency bins | hard zero (display) | derived centre ± half-width |
| Detector exclusion | detectors | drop from every group sum | 1-based id list |

The sharpest trap: the two time-window controls share label ("Exclude
(µs)"), units and parameterisation but mean opposite things — MaxEnt
de-weights and keeps the FFT grid; the count fit drops bins. Fix: relabel to
encode semantics (count-fit "Skip window (µs)" in Phase 2; MaxEnt "De-weight
window (µs)" in Phase 3) plus a five-row glossary in the user guide
cross-linked from each panel's docs (Phase 4). Found en route: **NEW-R1**
below.

### N6 — FFT phase vs MaxEnt phase exchange

**Decision (Ben, 2026-06-11): DOCUMENT.** Confirmed by-design: the Fourier
per-group phases (`group_phase_degrees` + auto-phase,
`mainwindow.py:1232`) and the MaxEnt fitted-phase exchange
(`use_fitted_phases_requested`, matched by group id with provenance) are
separate stores with no cross-feed. They are the same physical quantity
(per-group detector phase), so a "use MaxEnt fitted phases" pull on the
Fourier panel is plausible — **recorded trigger**: add it on the first user
request to phase the FFT from MaxEnt fits. Until then, a by-design note +
cross-references between the two phase docs.

## 5. Group D — surfaces & docs (F10, F1, F11, F13, N5, N4)

### F10 — trending/accumulation containers

**Decision (Ben, 2026-06-11): DOCUMENT now + selective UNIFY scheduled
(Phase 5); full stateless-window refactor DEFERRED.**

The flag partially dissolves: the PR #38 results-recursion series
(`modelfit-<digest>`, `mainwindow.py:6975`) and the PR #39 global-summary
accumulator (`modelfit-globals-<rep>`, `mainwindow.py:7085`) are **both
already `FitSeries`** instances in `ProjectModel.batches` — one container,
two writers, by sound design (per-group rows vs per-fit global rows). The
genuine outlier is `GlobalParameterFitWindow`: it displays one ephemeral
`CrossGroupFitResult` and serializes only its view decorations (local model
fits, plot annotations) under a separate top-level project key
(`global_parameter_fit_window_state`), where they can orphan when the
backing fit is re-run.

- DOCUMENT (Phase 4): a "trending data model" section — everything trendable
  is a `FitSeries`; the window is a transient view; how the two writers
  relate.
- UNIFY (Phase 5, ~M): move the window's decorations into
  `FitSeries.extra` keyed by batch id, restored on window show — additive,
  no schema break, kills the orphaning.
- DEFER: the stateless-window refactor (window reads a batch id instead of
  holding a result; ≈ a week, high risk). **Trigger**: the next time the
  window's data contract has to change anyway.

### F1 — quadrature two ways

**Decision (Ben, 2026-06-11): DOCUMENT.** The equivalence is exact —
`PowerLaw ⊕ Constant` reproduces `PowerLawQuadBG` to 1e-12 (oracle test,
`tests/test_parameter_models.py:1023`) and both code comments acknowledge
it. But relegation fails the convenience test: the grammar route yields four
parameters of which `c_1` must be *manually fixed at 0*, loses the built-in
BG ≥ 0 bound, and removing the registry entry hard-fails `.asymp` loads
naming the component. Crisp guidance exists: the monolith for the common
power-law-plus-floor; `⊕` for arbitrary quadrature combinations. Fix is
two-way cross-referencing (component docs + picker tooltip name the `⊕`
equivalent and vice versa) and a prose note mapping the parameter names
(`BG` ↔ `c_2`).

### F11 / F13 / N5 — cross-reference docs batch

**Decision (Ben, 2026-06-11): DOCUMENT all three** as one Phase 4 package:

- **F11**: `parameter_trending.rst` already links to `alc_mode` (line 577);
  the reverse link is missing. Add it plus one complementarity paragraph
  (time-domain `MuRepolarisation` trend vs integral-asymmetry ALC scan).
- **F13**: `frequency_finishers.rst` (Burg), `fourier_analysis.rst`
  (FFT/MaxEnt) and `frequency_domain_fitting.rst` have **zero** cross-links.
  Write the estimator-triad when-to-use: FFT for speed and linearity, MaxEnt
  for pulsed-source resolution and per-group phase handling, Burg as the
  badged line-splitting diagnostic with documented pathologies.
- **N5**: `fit_quality`, `result_summary` and the pull diagnostic have no
  shared entry point. Add a short "assessing a fit" hub cross-linking the
  three (χ² band → summary verdicts → pull distribution for error-bar
  validation).

### N4 — registry naming

**Decision (Ben, 2026-06-11): DEFER to Wave B python-user-functions**, which
builds the registration API over exactly these dicts (`MODELS`,
`COMPONENTS`, `PARAMETER_MODEL_COMPONENTS`) — the natural place to formalise
domain-distinguishing names without churning every import now. Phase 4 adds
the two-line naming note to ARCHITECTURE.md and annotates the
python-user-functions brief.

### F9 — forward collision: co-subtract (recorded, not a phase)

Standing programme decision (closeout §3): run-arithmetic must build
co-subtract on `subtract_scaled_counts` / the reference-run resolution
chokepoint, not beside it. **DEFER** here — Phase 4 annotates the
run-arithmetic brief so the constraint travels with the project that owns
it. Nothing to reconcile until that project starts.

## 6. New flags found during investigation

### NEW-R1 — count-fit exclude window not persisted

Found during the F8 catalogue: the count-fit exclude window is an API
parameter (`fit_panel.set_count_exclude()` → spinboxes in
`multi_group_fit_window.py`) with **no project schema key** — the only
exclusion that doesn't round-trip through `.asymp`; a saved project silently
loses the window on reload.

**Decision (Ben, 2026-06-11): UNIFY (persist)** — add the schema key and a
round-trip test inside Phase 2 (same files as the relabel), matching its
MaxEnt sibling's persistence.

No other new flags: the GUI `_parse_optional_float` text variant is noted
under N2; the diamag <5 G silent fallback and the orphaning of window
decorations are folded into F4 and F10 respectively.

## 7. Cross-checks against the watchlist

- The closeout's "recurring pattern is … N-way concept proliferation
  without reconciliation/promote paths" is confirmed for Group B and N3 —
  the resolution is four promote paths sharing one pattern, not
  consolidation.
- Severity labels held up except where noted: F3 "silent double-subtraction
  possible" and N3 "α/N0 bias if both active" describe interpretive traps,
  not mechanical ones; F4's third path and F10's third container do not
  exist as flagged.
- "Not collisions" entries (free-τ, FieldUnit, simulate multi-group) were
  spot-checked in passing and stand.

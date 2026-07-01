# Fit / trend / representation consolidation — implementation plan

Input documents: [README.md](README.md) (five-representation review) and
[comparison.md](comparison.md) (live GUI audit findings F1–F22 and decisions
D1–D8). This plan is the hand-off for the implementing session. **Do not start
from scratch — every step below cites the anchor it modifies.** Line numbers
were verified on `feat/fit-trend-representation-review` (rebased on
`origin/main`, 2026-07-02) and will drift; treat them as
`file:function` anchors.

Validation gate for every phase: `python tools/harness.py validate`
(worktree venv; the harness re-executes itself with `.venv/bin/python`).
Inner loop: `python tools/harness.py test --tier fast` plus the focused test
files named per phase.

Engineering invariants to respect throughout: core stays GUI-free (new
behaviour lands in `asymmetry.core` first); `.asymp` changes are additive with
defaults for absent fields; long work stays off the GUI thread.

---

## Phase 0 — Data-integrity bugs (land first, each with a regression test)

### 0.1 Representation-keyed fit-form round-trip (F21c)

*Bug:* the time-domain `fit_ui_state` is saved un-keyed at the project root —
`mainwindow.py:11625–11669` (`_save_project_state`; root write at ~11659) —
while frequency state nests under `"frequency_fit_state"` (~11668). On
restore (`mainwindow.py:12166–12178`) the root blob is applied blindly via
`fit_panel.restore_ui_state(...)`, which is how the frequency Gaussian model
(`height*exp(-4*ln(2)*((nu-nu0)/fwhm)^2)+bg`) ended up in the F-B time-domain
Batch form after reload.

*Change (GUI-layer, no schema break):*
- Serialize both domains symmetrically: `{"fit_states": {"time": ..., "frequency": ...}}`
  using the existing `FitPanel.get_domain_state(domain)`
  (`fit_panel.py:8548–8563`) for both, and restore both via
  `restore_domain_state(domain, ...)`. Keep reading the legacy root key and the
  legacy `"frequency_fit_state"` key on load (restore them into `"time"` /
  `"frequency"` respectively) — never write them again.
- While here: assert in `get_domain_state` that the returned blob carries a
  `"domain"` tag and that `restore_domain_state` refuses a mismatched tag
  (belt-and-braces against future regressions).

*Test:* GUI round-trip test — configure distinct time and frequency batch
forms, save, reload, assert each domain form holds its own model text and
classification. Include a legacy-fixture load (root `fit_ui_state` only).

### 0.2 Parameters-panel chip filtering by representation (F21a/b)

*Bug:* `fit_parameters_panel.py:739–794` (`get_state`) and `834–852`
(`restore_state`) serialize/restore `model_fits`, `group_fit_results` and row
caches with no representation tag, so after reload the F-B Parameters view
showed an "Integral scan 1" chip and a "Model fit (single)" chip while the F-B
series chip was missing. `ProjectModel.batches` itself *is* rep-typed
(`core/representation/project_model.py:79`), so the fix is presentation-side.

*Change:*
- Tag every serialized chip-bearing entry with its `rep_type` (and
  `projection` where applicable) in `get_state`; on restore, keep the full set
  but have the chip-population path filter to the active representation
  (same source of truth as `MainWindow._active_representation_type`,
  `mainwindow.py:8307–8322`).
- Re-derive the chip list from `ProjectModel.batches[rep_type]` after reload
  rather than trusting the panel cache for series chips; panel cache remains
  for labels/selection only. This also restores the missing F-B series chip,
  since the series *is* in the project model (supersession keeps it —
  verified live: grouped and frequency chips survived).
- `_migrate_legacy_state` prunes untagged entries into the representation they
  can be attributed to (via batch lookup) or drops them with a log line.

*Test:* build one series in each of F-B, grouped, frequency + one integral
scan; save/reload; assert each Parameters view lists exactly its own chips.

### 0.3 Data-browser group formation under sort (F9)

*Bug:* `data_browser.py:1078–1111` (`_rebuild_table`) rebuilds rows in
`_display_order` sequence and `_add_group_header_row` (1112–1157) appends at
`rowCount()`, ignoring the active sort; forming a group while column-sorted
leaves the header + member rows invisible until the next full rebuild
(project reload). Meanwhile the hidden runs stay *selected*, so batches can
run on runs the user cannot see.

*Change:* in `_rebuild_table`, capture the current sort state
(`horizontalHeader().sortIndicatorSection()/Order()`), disable sorting during
row insertion, and re-apply the indicator afterwards; group header rows sort
with their group block (keep the existing flat-rebuild approach — simply
re-sorting after rebuild is acceptable if headers pin correctly; if not,
switch the view to "sorting disabled while groups exist" with an explicit
status hint, which is the cheaper guaranteed-correct fallback). Also wrap in
`batch_updates()` per the O(n²) invariant.

*Test:* GUI test — sort by T (K), select runs, `create_data_group`, assert
the group header row and all member rows are present and visible immediately.

### 0.4 FFT restore renders stale legend with no data (F21d)

*Bug:* on project open, `_restore_frequency_representations`
(`mainwindow.py:5739–5769`) defers recompute; `_sync_frequency_plot_for_run`
(6015–6075) starts the async path, but `_on_frequency_recompute_finished`
(6176–6201) only renders when `_frequency_display_key` still matches, and the
legend artist is built before data exists — the user sees "2960 Average" in
the legend over an empty plot until pressing Compute FFT.

*Change:* clear the frequency plot (including legend) when entering the view
with a pending recompute, and in the completion handler re-render whenever
the *current* display key equals the completed key (recompute the comparison
at completion time, not against a stale snapshot). Show the existing loading
overlay while pending.

*Test:* GUI test — save a project with a computed FFT, reload, switch to FFT
view, run the event loop until recompute completes, assert the plot has data
and no orphan legend entries.

---

## Phase 1 — Series identity, replacement, naming (D4, F13, F22)

### 1.1 Identity signature (core)

`core/representation/project_model.py:40–67` (`_series_signature`) currently
includes `param_roles` and `fit_range`, so the audited Global→Local re-run
produced a *new* identity and hence the duplicate "B = 60 G" chip.

*Change:* narrow the identity key to
`(rep_type, projection, member_kind, ordered members, normalised model)`.
`param_roles`/`fit_range` become *attributes* of the series, not identity.
`superseded_batch_ids` (105–123) then replaces on re-run per D4; carry the old
series' `label` (user rename) and `batch_id` forward onto the replacement in
`remove_superseded_batches`/`add_batch` so chips and back-references stay
stable. Keep the `is_computed` early-out (114).

### 1.2 Unified naming (core helper + GUI callers)

Naming today: `_default_batch_series_label` (`mainwindow.py:9881–9898`,
"model · 2923–2960"), `_data_group_name_for_runs` (9858–9878, grouped series
named exactly like the DataGroup, e.g. "B = 60 G"), `_series_fallback_name`
(9900–9910, "Series N" / "Integral scan N"), and the trend-model-fit label at
~10829 ("Model fit (single): frequency vs temperature").

*Change:* one core-side helper, `core/representation/naming.py` (new):
`default_series_label(series) -> "<model> · <member-range>"` for all batch
kinds; when the members coincide with a DataGroup, append the group name as a
suffix (`"cos(...) · 2961–2967 · B = 60 G"`) instead of *replacing* the label
— this removes the DataGroup/series name collision while keeping the useful
group hint. Integral scans and model-fit chips keep their kinds but adopt the
same `<what> · <members/axis>` pattern. GUI callers delegate to the helper.
Labels remain user-renamable (`FitSeries.label`, `series.py:86`).

### 1.3 Load-time dedupe (migration)

On `.asymp` load, collapse batches with identical Phase-1.1 signatures,
keeping the most recent results and logging what was merged (projects saved
during the duplicate era — e.g. this audit's `audit-roundtrip.asymp` — load
clean). No schema version bump needed; this is tolerant reading.

*Tests:* core unit tests for signature narrowing, label carry-over on
replacement, dedupe-on-load; GUI test that re-running a batch with changed
classification updates the single chip in place.

---

## Phase 2 — Trend quality gating (D3, F2, F3, F12, F14, F5)

### 2.1 Member quality in core results

- `core/fitting/series.py:241` (`fit_asymmetry_series`): per-member results
  already record `success`; extend each member record with
  `chi2_reduced`, `param_errors`, and derived flags
  `quality_flags: set[str]` — `{"failed", "large_rel_err" (|σ/value| above
  threshold on any free param), "bound_pinned", "spurious_reseeded"}`.
  The spurious-branch reseed (335–351) already computes the trend prediction —
  reuse its residual as a flag input rather than inventing a new heuristic.
- `core/fitting/grouped_time_domain.py:934` (`fit_grouped_series`): populate
  the same fields in `member_results` (1125–1139). This also fixes F12's
  "no χ²ᵣ / no errors anywhere" for grouped fits — the engine has the minimizer
  state; it is presentation that discards it.
- Frequency batches reuse `fit_asymmetry_series` (dispatch in
  `fit_panel.py:~4821/5738`), so they inherit the fields for free.
- `FitSlot.include_in_trend` / `diverged` (`core/representation/base.py:70–71`)
  stay the per-member gate; flags do **not** auto-clear it (no auto-exclusion,
  per D3).

### 2.2 Surface failures and quality (GUI)

- After any batch, if members failed or were flagged, show a summary line in
  the fit panel results block (not only the log): "20/25 fitted · 5 failed
  (2941, …) · 2 flagged" (F2).
- Members tables (F-B, grouped, frequency): add χ²ᵣ and include-in-trend
  checkbox columns everywhere the F-B table already has them; fix the F14
  duplicate header by renaming fitted columns to `<param> (fit)`. The grouped
  "Fitted Variable Parameters" dialog gains the same columns.
- Grouped fit completion log gains χ²ᵣ, and terminology fixes: plot title
  "Grouped time-domain — 5 groups (run 2967)" not "5 runs"; log "35 traces
  (7 runs × 5 groups)" not "35 groups" (F12).

### 2.3 Trend-point flagging and click-to-exclude (F3, F5)

- Parameter plot (`fit_parameters_panel.py` plot block): render members whose
  slot has `include_in_trend=False` or non-empty `quality_flags` as hollow /
  warning-coloured markers with a tooltip listing the flags.
- Context menu on trend points: "Exclude from trend / Include", writing
  `FitSlot.include_in_trend` and re-running any attached trend model fit.
  Reuse the interaction pattern (and, once PR #166 merges, the provenance
  string convention) from the integral-scan click-to-exclude.
- Trend model fits (the Model Fit dialog) already consume included members
  only — verify and add a test that excluding the two garbage EuO members
  moves Tc toward 69 K on the recorded fixture.

*Tests:* core unit tests for flag computation on synthetic pathological fits;
GUI tests for the summary banner, table columns, exclude round-trip
(persisted through save/load).

---

## Phase 3 — Presentation provenance (D2, F6; hardens F21 further)

- `FitPanel` tracks the provenance of the current form contents:
  `own_slot | carried_from_run(N) | representation_default`. Recorders
  (`_record_single_fit_slot`, `mainwindow.py:9001–9040`) set `own_slot`;
  selecting a run with no slot keeps carry-forward but sets
  `carried_from_run` and the panel shows a dismissable badge:
  "Model carried from run 2960 — not fitted for this run" (cleared on fit).
- Seeds under carry-forward stay as-is (carry values), per D2's chosen option;
  the badge is the fix, not a seeding change.
- Slot presentation precedence itself (own stored `ui_state` → in-session form
  → carry-forward) is unchanged; document it in the panel docstring.

*Test:* GUI test — fit run A, select unseen run B, assert badge text and that
recording a fit for B clears it; assert reload restores badges correctly
(a restored form that didn't come from B's slot shows the badge).

---

## Phase 4 — Default classification + seed bridges (D5, F10, F11, F1)

- `fit_panel.py:4180–4185`: drop the `"Global" if i == 0 else "Local"` rule —
  all free params default Local; keep `fixed_default_params` handling.
- `fit_panel.py:7382–7391`: grouped physics `default_type = "Global"` →
  `"Local"`; keep background/phase nuisance defaults; keep "previous user
  choice wins" (7402).
- Global Fit Wizard may still *suggest* Global roles — that path sets roles
  explicitly and is unaffected.
- F11/F1 parity: when the Batch tab's model matches a just-completed single
  fit for a member run, offer the fitted values as seeds (same precondition
  the grouped "Send to Batch" uses); add "Send to Batch" to the F-B single
  tab for symmetry.
- Update the wizard/batch tests that assumed an implicit global (search tests
  for the i==0-Global expectation).

*Tests:* unit test on classification defaults per representation; GUI test
that a default grouped batch over a T-scan yields a varying-parameter trend
(the audited "No varying fit parameters" outcome becomes impossible without
explicit opt-in).

---

## Phase 5 — Frequency-domain fitting (D6, F15, F16, F17, F18)

### 5.1 Editable frequency fit range

`fit_panel.py:2293–2303` (`set_fit_range_display`) disables the spins when the
plot hasn't supplied a range — in the frequency domain nothing ever supplies
one, so the fields sit disabled showing the *time-domain* numbers as MHz.

*Change:* in the frequency domain, enable the spins with default = the
displayed X window; feed the range into the fit dispatch (mask the dataset
before `fit`/`fit_asymmetry_series`); the range participates in
`get_domain_state` so it round-trips. Never display a range that wasn't set
for this domain (placeholder shows "full spectrum" when unset).

### 5.2 Peak seeding with DC guard

`core/fitting/spectral.py:75–104` (`seed_peak_parameters_from_dataset`) seeds
ν₀ via `np.nanargmax(y)` — on (Power)^1/2 spectra that is the DC/apodisation
spike.

*Change:* exclude a guard band below `max(k / T_obs, f_guard)` (a few bins /
~2 MHz default) before the argmax; fall back to global argmax if the guard
empties the array. Seed `bg` from the guarded median, `height` from the
non-DC peak. Unit-test against the recorded EuO 2960 spectrum shape (peak at
~30 MHz, DC spike ×3 taller).

### 5.3 Add-to-Series feedback (F18)

`mainwindow.py:9931–9941` (`_on_add_single_fit_to_series`) returns with only a
status-bar message when no compatible series exists. Replace with a message
box offering "Create new series from this fit" (records a one-member series
via the Phase-1 naming helper) / Cancel; disable the menu action with a
tooltip when there is no completed fit.

### 5.4 Enable-state refresh audit (F17)

`_update_fit_block_state` refresh points (`mainwindow.py:2532, 1794, 8133`,
and `6128` in `_render_frequency_spectra`) miss two paths observed live:
after `_on_apply_fourier_to_selection` (7129–7172) and after switching to the
Batch subtab. Add explicit refresh calls at both, and a GUI test that computes
spectra for a selection and asserts Run Batch Fit enables without a selection
poke.

---

## Phase 6 — MaxEnt ZF window (D7, F19)

- `core/maxent/engine.py:566–586` (`_resolve_frequency_window`): when the run
  field is absent/≈0, derive the window from data instead of returning the
  `(0, max(10, half_width→MHz))` fallback: run the existing FFT machinery
  (or a cheap periodogram on the grouped asymmetry) and window around the
  dominant non-DC peak (reuse Phase 5.2's guarded peak finder) with padding;
  fall back to a Nyquist fraction when no peak stands out. `auto_window`
  stays overridable; manual Window entries win unchanged
  (`maxent_panel.py:116–122`).
- Divergence message (`mainwindow.py:7646–7652`): include the active window
  and name the control — "…χ² began rising past the optimum. The current
  frequency window is 0–9.8 MHz (Window section); signals outside it cannot
  be reconstructed."

*Tests:* engine unit test — synthetic ZF signal at 30 MHz gets a window
containing 30 MHz; TF run keeps the field-centred window; message content
test.

---

## Phase 7 — DataGroup ↔ FitSeries linking (D1, §6 Option B)

Core-first, additive schema:

- `core/representation/project_model.py`: promote browser groups into the
  project model — `data_groups: dict[str, DataGroup]` with
  `DataGroup(group_id, name, member_run_numbers, order_key)` (order_key =
  the trend X convention already used by series). The browser
  (`data_browser.py:891–931`, `create_data_group`) becomes a view over this
  registry instead of a private `_groups` dict; `.asymp` gains an optional
  `data_groups` block (absent → empty, fully back-compatible).
- `FitSeries` (`core/representation/series.py:68–113`) gains optional
  `source_group_id: str | None`; batch recorders
  (`_record_global_fit_batch` 9042–9116, `_record_grouped_fit_series`
  9671–9799) stamp it when the member set was launched from a group.
  Back-references (group → series) are computed, not stored.
- GUI: browser group context menu gains "Fit this group…" (prefills the batch
  member set from the group, bypassing the live-selection trap that F9
  exposed) and "Show series from this group" (filters chips). Chips whose
  series has `source_group_id` display the group name as a prefix chip-tint,
  reusing the existing series-tint mechanism.
- Batches remain constructible from ad-hoc selections (Option B, not C).

*Tests:* project round-trip of `data_groups` incl. legacy projects without
the block; recorder stamping; "Fit this group" produces a series with correct
members regardless of browser sort/visibility.

---

## Phase 8 — Small UX safety items

- **F20 wheel guard:** add a shared `install_wheel_guard(spinbox)` /
  `NoScrollSpinBox` in `gui/widgets` — `Qt.StrongFocus` + ignore wheel events
  when unfocused — and apply to right-dock numeric fields
  (`maxent_panel.py:100–106` points spin, `fit_panel.py:2084–2089` range
  spins, grouped nuisance table editors). Grep for `QSpinBox()`/
  `QDoubleSpinBox()` under `gui/panels` and wrap all dock instances.
- **F4 trend model default:** keep Linear as the safe default but populate the
  model combo with suggested models (OrderParameter listed when the X axis is
  T and Y is a frequency/field-like parameter) and use the existing
  `suggest_trend_seeds` machinery for whatever model is chosen. No behaviour
  change beyond discoverability.
- **F8 ordering hint:** after multi-file load, if browser insertion order
  differs from run-number order, sort by run number by default (cheap; the
  browser already supports header sorting — this just picks a sane initial
  indicator). Pairs with the 0.3 sort fix.

---

## Sequencing and landability

Each phase is a separate PR-sized unit; 0.1–0.4 are independent of each other
and everything else. Phase 1 before Phase 2 (quality columns render series
that replace correctly). Phase 3, 4, 5, 6 are mutually independent. Phase 7
depends only on Phase 1 (naming helper). Phase 8 anytime.

Suggested order: 0.x → 1 → 2 → 4 → 5 → 3 → 6 → 7 → 8, validating with
`python tools/harness.py validate` before each hand-off and
`gui-smoke` after phases touching startup/panels.

## Test plan summary

- New core tests: `tests/test_series_identity.py` (signature, replace, dedupe,
  naming), `tests/test_series_quality.py` (flags), `tests/test_spectral_seed.py`
  (DC guard), `tests/test_maxent_window.py` (ZF window), extension of existing
  grouped-series tests for χ²ᵣ/errors.
- New/extended GUI tests (offscreen, respecting the `_cleanup_qt_widgets`
  fixture): round-trip domain form state (0.1), chip filtering (0.2), sorted
  group formation (0.3), FFT restore render (0.4), classification defaults
  (4), freq range + batch-enable (5), badge lifecycle (3), group-fit flow (7).
- Fixture: a small recorded `.asymp` capturing the audit state (duplicate
  chips + legacy root `fit_ui_state`) for migration tests.

## Risks and migration notes

- `.asymp`: all schema changes are additive (`data_groups`,
  `source_group_id`, member `quality_flags`/`chi2_reduced`, keyed
  `fit_states`). Loaders must default when absent; legacy keys are read but
  never written. Load-time batch dedupe (1.3) is the only destructive-ish
  migration — it merges *identical-signature* series only and logs.
- Behaviour changes users may notice: all-Local batch defaults (Phase 4) and
  frequency seeding (Phase 5.2) change fit outcomes from defaults; both are
  the decided intent but release notes should call them out.
- PR overlap: click-to-exclude (2.3) should adopt the interaction/provenance
  conventions of the integral-scan panel in PR #166 once merged; if #166 is
  still open when Phase 2 lands, implement against the current dock panel and
  reconcile in the #166 branch, not here.
- The F-B/grouped/freq engines stay separate in this plan (dual-engine
  consolidation from README §5.1 is out of scope); the quality-field contract
  (2.1) is the first shared surface and should be written as a small common
  dataclass so a future engine merge has a seam.

## Out of scope (explicitly)

- §6 Option C (group-as-trend-unit) and any batch-engine unification.
- Automatic outlier exclusion from trends (D3 chose flag-only).
- Unifying integral-scan baseline/peak fitting with the trend Model Fit
  dialog (fourth fitting surface — noted in comparison.md, future study).
- The F10 stateless-window refactor deferred from the collision programme.
- `.nxs_v2` / `.RAW` loader gaps; anything in PR #166's scope.

## Acceptance criteria

1. Round-trip: save/reload preserves — per representation — series chips,
   batch/single form contents, and renders restored FFT spectra (F21 closed;
   tests from 0.1/0.2/0.4 green).
2. Forming a data group while sorted never hides rows; batches cannot silently
   operate on invisible selections (F9 closed; "Fit this group" available).
3. Re-running a batch never yields two identically-named chips; all series
   follow `<model> · <members>[ · <group>]` (F13/F22 closed).
4. Every batch member exposes χ²ᵣ + errors + flags in tables and trend plots;
   failed members are listed in the panel; trend points are excludable by
   click and exclusions persist (F2/F3/F5/F12/F14 closed).
5. A default batch on any representation produces a varying-parameter trend
   (no silent Global) (F10 closed).
6. Frequency: fit range editable, default seed finds the 30 MHz EuO peak, not
   DC; Run Batch Fit enables as soon as spectra exist (F15/F16/F17 closed).
7. MaxEnt converges on ZF EuO 2960 with default settings (window contains
   30 MHz) and its divergence message names the Window control (F19 closed).
8. Unseen-run carry-forward is visibly badged (F6 closed).
9. `python tools/harness.py validate` green; `gui-smoke` green.

# Implementation options

## Deferred Step-4 follow-ups (from the second review, agreed with Ben)

Real but lower-severity findings deferred to a follow-up session (the common-case
correctness bugs — fit on wrong curve, overlay on wrong subplot, run-switch
re-reduce — were all fixed):

- **Multi-run overlay + stacked + single fit: run-key split.** In a multi-run
  overlay ALL view, `plot_fit` keys the curve under the plot panel's
  `_current_dataset` (the *first* projection clone's run) while
  `_record_single_fit_slot` keys the slot under the main window's
  `_current_dataset` (the clicked run). When those runs differ the displayed
  overlay and the persisted slot disagree on the run. Edge case (single fit in a
  multi-run overlay is already ambiguous); decide whether to block it there or
  source the run from one place.
- **Non-canonical projection labels skip the re-reduce.**
  `_on_fit_target_projection_changed` / the bind paths only re-reduce for
  `P_x/P_y/P_z`. When Step 5 adds TF dual-grouping subplots with non-canonical
  labels, clicking one won't re-reduce — reopening the fit-on-wrong-curve bug for
  those labels. Tie this to generalizing `_normalize_vector_axis` in Step 5.
- **Empty-projection subplot y-range.** Dropping the `elif ALL` y-fallback means a
  selected subplot whose data is all-NaN keeps matplotlib's default `(0,1)` when
  Auto-Y is off. Cosmetic; only a projection with no finite asymmetry.

## Known pre-existing bug (deferred, surfaced during Step 4 testing)

**Auto-X ↔ display-decimation interaction.** Setting Auto X while *coming from a
narrower x-range* yields a wrong/narrow x-range with data missing; toggling the
view (re-render) recovers it. Display decimation is computed per visible view
(`_decimation_applied_for_current_view`) and `_last_plot_time` — which
`_auto_x_limits` reads — is populated from whatever the current (possibly narrow,
decimated) view rendered, so Auto X computes from a stale/narrow time array until
a re-render recomputes decimation over the full view. **Pre-existing, unrelated to
projections** (Ben chose to finish Step 4 first, then fix this separately). NB: an
earlier attempt to fix it by excluding "low-count" points was wrong — the grey
bars are *decimated dense data*, not low-count — and was reverted.

## A. UI surface for selecting/viewing projections

Four surfaces were considered (mockups discussed with Ben, 2026-06-13). All build
on the same shared data-model change (section B).

### A1. Generalize the existing dropdown
Rename the `Polarization:` combo to a generic `Projection:` selector populated by
any multi-channel grouping (`P_x/P_y/P_z/All`, or `Top–Bottom/Fwd–Back/All`).
- Clean UI: ✓✓. Discoverability: ✗ (a combo that only appears in some modes —
  nobody finds vector mode unless told). One-or-all only; can't show `P_x`+`P_z`.

### A2. Multi-select chip bar — **chosen**
A row of toggleable, colour-keyed chips, one per projection, multi-select; each
selected projection = one stacked subplot.
- Clean UI: ✓ (one slim row). Discoverability: ✓✓ (always visible when ≥2
  projections — you can *see* there are multiple things to look at). Natural fit
  for RF where you want all three projections at once. Supports arbitrary
  subsets. Needs the frame-tint colour story (section C).

### A3. Data-browser child rows
Each multi-projection run expands to selectable child rows; plotting/overlay/fit
flow through the existing run-selection + overlay machinery.
- Clean UI: ✓✓ (nothing added to the plot). Most powerful (unifies "overlay
  runs" and "overlay projections"). Risk: ✗✗ biggest semantic change — a run
  "containing" sub-rows is unexpected; touches selection, overlay, legend
  labelling, alpha. Deferred as too broad for this pass.

### A4. Projection inspector dock
A dedicated panel (sibling to Fit Parameters): per-projection visibility, alpha,
estimate, colour.
- Clean UI: ✓✓. Centralizes everything, but adds a dock and splits projection
  setup away from the grouping dialog where alpha currently lives. The *per-
  projection alpha table* idea from D is kept, but folded into the existing
  grouping dialog rather than a new dock.

**Decision:** A2 chip bar for everyday select/view; per-projection alpha stays in
the grouping dialog (D's table, generalized). A1/A3 are the cleaner-but-hidden
and most-powerful-but-riskiest ends; not chosen.

### Chip-bar behaviour (settled)
- Multi-select toggles, **floor of one** (last chip won't release; can't show
  zero subplots), max N.
- **No "All" chip** — redundant and contradictory in a multi-select model
  ("All + P_x"?). The chips express the whole continuum (1 = single view,
  3 = old "All").
- Optional **"all" *action*** (text link, not a toggle; greys out when all on)
  for the collapse-to-one-then-expand fit round-trip. **Open micro-decision:**
  keep vs drop entirely (≤2 clicks saved with 3 projections).
- `Projection:` header noun universally (no per-preset label).

## B. Data model

### B1. Explicit `projections` declaration — **chosen**
Add `AsymmetryProjection(label, forward_group, backward_group, alpha, tint)` and
`PresetGrouping.projections: tuple[...]`. Mirror in the project schema. GUI reads
`projections`; the three string-matching helpers collapse to one. `vector_axis` →
`active_projection`; `alpha_x/y/z` → per-projection alpha; legacy keys migrate.

### B2. Keep name-string inference, extend to TF — rejected
Add more magic names (`"tb forward"`…) and keep inferring. Cheap but compounds
the existing fragility and the ×3 duplication. Rejected.

## C. Colour scheme

- **Trace colour = run identity** (RG mode), untouched.
- **Frame tint = projection identity**: fixed semantic mapping
  (`P_x` purple, `P_y` amber, `P_z` teal) on chip + subplot rail / y-label.
  Muted (chrome, not data), chosen away from the run-colour palette.
- Text label always present (colourblind / greyscale safe); tint + hover
  cross-highlight are reinforcement, not the sole signal.
- One projection = one subplot, **never co-plotted** — so a frame is never shared
  between two projections, and no linestyle disambiguation rule is needed.

## D. Fit target + per-projection fit storage

### D1. Selectable subplots — **chosen**
Click a subplot → it becomes the active fit target (neutral focus ring + "fit
target" pill, distinct from the identity tint). Fit panel echoes `Fitting: P_y`.
Selection UI only with ≥2 subplots; target must be visible; fit curve overlays
only the active subplot.
- Alternatives considered: fit-panel `Projection:` dropdown (explicit but
  disconnected from what you see) and a "primary chip" (overloads chips with
  visibility + focus). The dropdown may *echo* the subplot selection as a
  secondary, in-sync control.

### D2. Projection-keyed fit storage — **required by D1**
Generalize the per-`Representation` single `FitSlot` into a projection-keyed map
(`None` = today's single-projection case). Schema **v9** + migration landing the
existing fit on the default projection. The display layer's `(run, axis_key)`
caches already match this shape; they get fed from persisted slots instead of
session-transient ones.

### D3. Joint fitting — deferred, door left open
Shape per-projection storage to mirror `FitSeries.results_by_run` (as
`results_by_projection` + `param_roles`), so future RF joint fitting is "the
existing global-fit engine indexed by projection instead of run", not a new
subsystem. Not implemented this pass.

## Carried into Step 2 (from the Step 1 code review)

The core `derive_projection_pairs` was generalized (arbitrary labels, partial
subsets) ahead of the GUI, which is still canonical-`P_x/P_y/P_z`-only. The two
*crashes/correctness* bugs from that mismatch were fixed in Step 1.5 (KeyError on
a `P_z`-less subset; stale projections resurrected when switching off a vector
preset). The remaining *latent* gaps are deliberately deferred to this step,
because fixing them piecemeal would half-build the chip-bar generality:

- **Plot selector hardcodes `axis_order = ["P_x","P_y","P_z"]`**
  (`_refresh_vector_axis_selector`, `_build_vector_axis_datasets`) — a
  non-canonical projection set resolves in the core but is dropped before the
  selector. The chip bar replaces this filter with the declared projection order.
- **`AsymmetryProjection.alpha` is declared/persisted but not consumed** — per-
  projection alpha still comes from `alpha_x/y/z`. Wire the chip-bar per-
  projection alpha to seed from / reconcile with the declared `alpha`.
- **`[dict(p) for p in … if isinstance(p, dict)]` repeated across files** — when
  the chip bar adds more projection handling, introduce a single
  `normalize_projection_payload` / `AsymmetryProjection.from_payload` helper.

### From the Step 3 part-1 review (obligations for part 2)

- **Divergence / trend only matters for batch-per-projection (deferred).**
  `core/representation/project_model.py` divergence/trend helpers read
  `representation.fit` (the default slot). Per-projection **single** fits are not
  series members, so they never touch divergence — no change needed for the
  single-fit scope. *Only* when batch/global fits become per-projection (the
  deferred prize) must those helpers iterate `representation.iter_fit_slots()`
  and key series membership by `(run, projection)`.
- **Part 2 single-fit scope = write site + projection-aware fit panel.** The
  single-fit record site (`mainwindow._record_single_fit_slot`) writes
  `set_fit_for(active_projection, slot)`; the fit panel keys its transient
  single-fit state by `(run, projection)` and restores the persisted slot when
  the active projection changes. `_fit_key` already routes falsy / `"ALL"` to the
  default slot, so a fit taken in single-axis mode lands on that axis and one
  taken outside vector mode lands on the default.

### Step 3 part 2 — chosen approach (Option B, for a fresh session)

**Decision (Ben, 2026-06-13):** the fit panel restores single-fit state from the
structured per-`(run, rep_type, projection)` `FitSlot` storage (built in part 1),
**not** from its own parallel run-keyed blob. One source of truth; the projection
dimension lives only in the `FitSlot` lookup.

**Why the naive wire-up doesn't work:** the fit panel caches single-fit *form*
state in `_single_state_by_run: dict[int, dict]` (composite_model, parameters,
result_html, wizard_state), keyed by run across ~15 sites in `fit_panel.py`, and
that same dict is shared with global-fit seeding, the fit-wizard cache, and
group-sharing — all of which are intrinsically per-run, not per-projection.
Re-keying it wholesale is the trap (Option A). Also: `set_dataset` is **not**
currently called when the projection changes, and `FitSlot` doesn't yet carry the
fit panel's full UI payload (it has model/parameters/result; the panel also needs
result_html + wizard_state).

**Concrete plan:**

1. **FitSlot gains a `ui_state: dict` field** (additive; serialize like the rest)
   carrying exactly what the single-fit form needs to restore — composite_model,
   parameters, result_html, wizard_state. This makes the slot the complete
   restore payload for one `(run, rep, projection)`.
2. **Mainwindow mediates restore** (the panel stays decoupled from `ProjectModel`).
   On the dataset/axis-change binding, after `_fit_panel.set_dataset(dataset)`,
   mainwindow looks up `representation.fit_for(projection)` (projection =
   `_normalize_vector_axis(plot_panel.get_current_polarization_axis())`) and pushes
   its `ui_state` into the panel via a new `restore_single_fit_ui(payload)` (or
   `set_dataset(dataset, restore=payload)`).
3. **Rebind on projection switch.** `_on_plot_polarization_axis_changed` (single-axis
   branch) must call `_fit_panel.set_dataset(_get_fit_dataset(current))` so the
   panel re-points at the selected axis's curve and triggers the slot restore.
4. **Write site** `_record_single_fit_slot` writes the full slot (incl. `ui_state`)
   via `set_fit_for(projection, slot)` (`_fit_key` already routes `"ALL"`/falsy →
   default).
5. **Leave global/wizard/group machinery run-keyed** — out of scope (batch/global
   per-projection is the deferred prize). Only the *single-fit* tab becomes
   projection-aware.
6. **Legacy `single_fit_state` blob:** keep reading it on restore for back-compat
   (pre-this-change projects), but the slot `ui_state` is the new canonical store.

**Test obligations:** in-session swap (fit P_x, switch to P_z → empty form, fit
P_z, back to P_x → P_x's fit restored); save/load round-trip preserves each
projection's fit independently; **no regression** to single (non-vector), global,
group, and wizard fits; the fit panel still works with no ProjectModel present
(stub tests).

### From the Step 2 review (deferred to Step 4/5)

- **The `"ALL"` sentinel collapses the subset.** Multi-select is mapped onto the
  legacy single `_current_polarization_axis` string (`len>1 → "ALL"`), so the
  actual subset (`{P_x,P_z}` vs the full triple) is recoverable only from
  `selected_projection_labels()`. Step 2.5 fixed the *user-visible* symptom by
  persisting `projection_selection` in plot state; the deeper fix — keying off
  the selected-label list directly and treating single-vs-stacked as
  `len(labels)==1` rather than a sentinel — is deferred. It will otherwise fight
  the TF dual-grouping case (2 projections, not a vector triple) in Step 5.
- **`_normalize_vector_axis` is canonical-only.** Single-selecting a
  non-`P_x/P_y/P_z` projection routes through it and is dropped; the chip bar /
  `_build_vector_axis_datasets` / `plot_vector_subplots` are already
  label-agnostic, so this gate is the remaining canonical assumption to remove in
  Step 5.

## Suggested implementation order

1. B1 data model (`AsymmetryProjection` + schema field) with migration; retire
   the three string-match helpers. No UI change yet — validate round-trips.
2. C + A2: chip bar + frame-tint rendering, generalizing
   `_build_vector_axis_datasets` to the selected subset; retire the combo.
3. D2 projection-keyed `FitSlot` map + schema v9 migration.
4. D1 selectable subplots + fit-panel echo, fed from D2.
5. TF presets declare their two projections (proves the unification end-to-end).

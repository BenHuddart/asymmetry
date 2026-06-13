# Implementation options

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

## Suggested implementation order

1. B1 data model (`AsymmetryProjection` + schema field) with migration; retire
   the three string-match helpers. No UI change yet — validate round-trips.
2. C + A2: chip bar + frame-tint rendering, generalizing
   `_build_vector_axis_datasets` to the selected subset; retire the combo.
3. D2 projection-keyed `FitSlot` map + schema v9 migration.
4. D1 selectable subplots + fit-panel echo, fed from D2.
5. TF presets declare their two projections (proves the unification end-to-end).

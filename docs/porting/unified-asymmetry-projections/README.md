# Study: unified-asymmetry-projections

**Status:** study (design converged with Ben, 2026-06-13). Not a reference-program
port — an internal unification of two existing/related features. **References:**
none external; this re-frames Asymmetry's own vector-polarization implementation.

## Motivation

Two situations in the app are the *same physics* dressed as two unrelated
features:

1. **EMU vector polarization** (shipped). The muon-spin polarization is followed
   along three orthogonal axes `P_x`, `P_y`, `P_z` — the natural mode for RF
   measurements (the spin is rotated and you want to track each component) and
   for anisotropic single crystals where the local field is canted off `ẑ`.
2. **TF dual detector grouping** (not unified). For transverse-field runs a
   detector set can be grouped two different ways — e.g. MuSR
   `Transverse (Top–Bottom)` vs `Transverse (Forward–Backward)`, HiFi
   `Transverse (Left–Right)` vs `(Top–Bottom)`. Today these are *mutually
   exclusive presets*: pick one, get one F/B pair. No way to flip between them or
   view them together without re-opening the grouping dialog.

The unifying observation: **a forward/backward asymmetry *is* the muon
polarization projected onto the axis defined by that detector pair.** Vector mode
makes the three orthogonal projections explicit; TF dual-grouping is the same
observable projected onto two different detector-pair axes. Both are a set of
named **projections**, each defined by a `(forward_group, backward_group, alpha)`
triple computed from the same histograms, that the user wants to switch between
or view together.

- Vector polarization = 3 projections.
- TF dual-grouping = 2 projections.
- Ordinary longitudinal = 1 projection (degenerate; no selector needed).
- RF (the future prize) = the 3 vector projections, but fit *jointly* against
  shared Rabi/Bloch dynamics.

"Projection" is the correct physical noun, not just a convenient label — it
states *why* the two cases are the same thing, and gives the user the right
mental model for free.

## Converged design

### Data model — make "projection" first-class

The current vector mode is inferred from **magic group-name strings**
(`"pz forward"`, `"py top"`, `"px left"`, …) re-matched in three separate GUI
sites. Replace that with an explicit declaration:

```python
@dataclass(frozen=True)
class AsymmetryProjection:
    label: str            # "P_x" | "P_y" | "P_z" | "Top–Bottom" | "Fwd–Back"
    forward_group: int
    backward_group: int
    alpha: float = 1.0
    tint: str | None = None   # fixed semantic frame colour
```

declared on `PresetGrouping` (`projections: tuple[AsymmetryProjection, ...]`) and
mirrored in the project schema. The EMU `Vector Polarization` preset *declares*
its three projections; the MuSR/HiFi transverse presets can declare their two.
The GUI reads `projections` instead of pattern-matching names, collapsing the
three duplicated `_detect/_vector_axis_pairs` helpers into one. `vector_axis`
becomes the generic `active_projection`; `alpha_x/y/z` become per-projection
alpha. Migration maps the old keys forward.

### UI — multi-select projection chip bar

A chip bar in the plot header (replacing the old `Polarization: x/y/z/All`
combo), one chip per declared projection, **multi-select**. Appears only when the
active grouping declares ≥2 projections (longitudinal stays clean). Header noun:
`Projection:` universally — no per-preset label needed.

- **Each selected projection renders as its own stacked subplot.** Projections
  are *never* co-plotted in one axes (settled with Ben: `P_x`/`P_y`/`P_z` are
  physically different quantities; TF projections are read as adjacent panels,
  not overlaid). This makes the frame-tint scheme below unambiguous and reuses
  the existing vector-`All` stacked-subplot machinery, generalized from
  "all-or-one" to "the selected subset".
- **Floor of one, max N.** You can't show zero subplots — toggling off the last
  chip is a no-op.
- **No "All" chip.** In a multi-select model "All" is redundant (it's just "all
  chips on") and contradictory as a peer toggle. Optionally keep a lightweight
  **"all" *action*** (a text link, not a toggle; greyed when all on) for the
  frequent collapse-to-one-then-expand round-trip in the fit workflow. *(Open
  micro-decision — see below.)*

### Colour — frame-tint, not trace-colour

EMU normally runs in RG mode where **trace colour already encodes which
run/group** you're viewing. Channel identity must therefore live somewhere else,
or it collides. Resolution:

- **Trace colour = run identity** (RG mode, untouched).
- **Frame tint = projection identity** — a fixed *semantic* mapping
  (`P_x` purple, `P_y` amber, `P_z` teal, …) on the chip + the subplot's left
  rail / y-axis label. Muted, reads as chrome, kept away from the run-colour
  palette.

The two colour systems never compete because one paints the data and the other
paints the furniture. Colour is *reinforcement*: the subplot's text label
(`P_x`) is always present (colourblind / greyscale-export safe), and a chip↔
subplot hover cross-highlight removes any residual ambiguity.

### Selectable subplots = the fit target

With ≥2 subplots, "which projection does a fit act on?" is ambiguous. Resolve it
spatially: **click a subplot to make it the active fit target** (a neutral focus
ring + a "fit target" pill — distinct from the rail tint, which is *identity* not
*state*). The fit panel echoes the binding (`Fitting: P_y`) so the coupling is
legible from both ends. Rules:

- Selection UI only appears with ≥2 visible subplots; a single projection is
  implicitly the target.
- The fit target must be a *visible* projection; hiding it moves the target to
  another shown subplot.
- The fit curve overlays only the active projection's subplot.

This is the clean separation of the **two orthogonal controls**: chips =
*visibility* (multi-select, floor 1); subplot box = *fit target* (single-select
within the visible set).

### Per-projection fit persistence (the dependency selectable subplots create)

Today there is **no** persistent per-projection fit storage: a `FitSlot` is keyed
`(run, rep_type)` only, axis-agnostic. The plot panel caches fit *curves* by
`(run, axis_key)` but transiently (display only, dropped on save). Selectable
subplots are only coherent if clicking `P_y` loads *P_y's* model+params — so this
feature requires generalizing the per-representation `FitSlot` into a
**projection-keyed map** (`None` = today's single-projection case, back-compat),
schema bump to **v9** with a migration landing the existing fit on the default
projection.

### Joint fitting — the deferred prize, framed to leave the door open

RF joint fitting (fit `P_x`/`P_y`/`P_z` together with shared parameters) is **the
existing global-fit engine rotated 90°**: today `FitSeries` does a global fit
*across runs* (`results_by_run`, `param_roles` = global/local/fixed); joint RF
fitting is a global fit *across projections* of one run (`results_by_projection`,
same role split). If per-projection fits are keyed in a structure mirroring
`results_by_run`, the future joint fit is "the global-fit engine indexed by
projection instead of run" — not a new subsystem. Selectable subplots then become
"pick the projections to include in the joint fit" almost for free.

**Out of scope for this pass:** per-projection *batch/global* fits and the actual
joint fit. In scope: per-projection *single* fits, shaped to accept them later.

## Resolved decisions

- **"all" affordance** (Ben, 2026-06-13): **keep** a lightweight "all" *action*
  (text link, not a toggle; greys out when all chips on) — it saves the frequent
  collapse-to-one-then-expand round-trip in the fit workflow.

## Files

- `comparison.md` — current string-matched implementation vs the unified model,
  exact code seams.
- `implementation-options.md` — the four UI surfaces considered (A dropdown,
  B chip bar, C browser child-rows, D inspector dock) with tradeoffs and why B
  won; data-model and fit-storage options.
- `test-data.md` — instruments/datasets exercising vector and TF projections.
- `verification-plan.md` — how the port is verified.

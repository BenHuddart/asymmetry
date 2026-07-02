# Fit / trend / representation consolidation — implementation options

This document records the design options weighed before implementation and the
option chosen for each decision. It is the sibling to
[README.md](README.md) (behaviour review), [comparison.md](comparison.md) (live
GUI audit, findings F1–F22 and decisions D1–D8) and
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) (the phased hand-off). The plan
is the *how*; this file is the *why this shape and not another*.

The consolidation is not a port of a single reference feature — it is an
internal-consistency pass over behaviour that already exists across five
representations (time F-B asymmetry, grouped/raw-count time domain, FFT, MaxEnt,
integral/field scans). Each decision below is a `D`-tag from
[comparison.md](comparison.md); Ben confirmed every one at the recommended
option during the audit.

## Decision axes and chosen options

### D1 — DataGroup ↔ FitSeries coupling (README §6)

- **Option A (loose):** keep the two objects separate; add a "Fit this group"
  action that seeds batch membership from `DataGroup.member_run_numbers`.
- **Option B (linked) — CHOSEN:** promote browser groups into `ProjectModel`
  (`data_groups`), give `FitSeries` an optional `source_group_id`, and compute
  group→series back-references. The group becomes a durable, named *membership*;
  a series is *one fit of that membership*. Ad-hoc selections still fit.
- **Option C (unified):** collapse DataGroup and FitSeries into one trend unit.
  Rejected — forces one-fit-per-group-per-rep and has nowhere to hold multiple
  models/windows over the same runs.

Rationale: B removes the DataGroup/series name collision and the live-selection
trap (F9) without discarding the 1-group-to-many-series reality. Landed in
Phase 7.

### D2 — Model presented on dataset selection (README §7)

- **CHOSEN:** keep the existing three-level precedence (own slot → in-session
  form → carry-forward) but make carry-forward *visible* with a dismissable
  badge ("Model carried from run N — not fitted for this run"). No seeding
  change. Alternatives (group/series-aware precedence, wizard-assisted, domain
  default reset) were deferred as higher-risk or premature. Landed in Phase 3.

### D3 — Trend quality gating

- **CHOSEN:** flag pathological members (`quality_flags`) and surface them, but
  never auto-exclude from the trend. Automatic outlier removal was rejected as
  too surprising; the user keeps the include/exclude decision. Landed in Phase 2.

### D4 — Series identity and replacement

- **CHOSEN:** narrow the series identity signature to
  `(rep_type, projection, member_kind, ordered members, normalised model)` so a
  re-run with changed classification/window *replaces* the chip instead of
  duplicating it. `param_roles`/`fit_range` become attributes, not identity.
  Landed in Phase 1.

### D5 — Default parameter classification

- **CHOSEN:** all free parameters default **Local** in both the single and
  grouped paths (dropping the implicit "first parameter is Global" rule), so a
  default batch always produces a varying-parameter trend. The Global Fit Wizard
  still sets Global roles explicitly. Landed in Phase 4.

### D6 — Frequency-domain fitting

- **CHOSEN:** make the frequency fit range editable, guard peak seeding against
  the DC/apodisation spike, and fix the batch-enable refresh. Landed in Phase 5.

### D7 — MaxEnt ZF frequency window

- **CHOSEN:** when the run field is ≈0, derive the reconstruction window from the
  data (guarded peak finder) rather than the fixed fallback, and name the Window
  control in the divergence message. Landed in Phase 6.

### D8 — Naming

- **CHOSEN:** one core naming helper producing `<model> · <members>[ · <group>]`
  for every batch kind, with the DataGroup name as a *suffix* rather than a
  replacement. Landed in Phase 1.

## Cross-cutting engineering decisions

### Schema evolution for the keyed `fit_states` block (Phase 0.1)

The time-domain fit form was persisted un-keyed at the project root while the
frequency form nested under `frequency_fit_state`, letting a frequency model
bleed into the time form on restore (F21c). Three ways to land the keyed
`fit_states` block were considered:

- **Bump schema to v11 + migration (CHOSEN).** Add `_migrate_v10_to_v11` that
  folds the legacy root `single_fit_state`/`global_fit_state`/`fit_ui_state` into
  `fit_states.time` and `frequency_fit_state` into `fit_states.frequency`,
  dropping the legacy top-level copies. Save writes `fit_states` only; restore
  reads it (migration guarantees presence, with a legacy-key fallback for callers
  that bypass migration). One shape per version, matching the schema module's
  "only the latest schema is written" policy.
- **No bump, GUI dual-read.** Stop writing the legacy keys but leave the schema
  version at 10. Rejected — two shapes at one version defeats the point of the
  versioned migration layer.
- **No bump, keep writing legacy too.** Fully additive but leaves the ambiguous
  legacy keys in new files — does not fix the root cause.

Each per-domain block carries a `domain` tag; `FitPanel.restore_domain_state`
refuses a blob whose tag does not match the requested domain, so a mis-routed
payload fails loudly instead of silently populating the wrong form.

### Keep the two batch engines separate (scope boundary)

The dual-engine consolidation (README §5.1: unify `fitting/series.py` and
`fitting/grouped_time_domain.py`, and give frequency a batch fitter) is **out of
scope**. Instead, the shared member-quality contract (Phase 2.1:
`chi2_reduced`, `param_errors`, `quality_flags`) is written as a small common
dataclass so a future engine merge has a seam. This keeps each phase PR-sized and
avoids a high-risk engine rewrite entangled with the consistency fixes.

### Core-first, additive schema throughout

Per the repository invariants, new analysis behaviour (identity signature,
naming helper, member quality, MaxEnt window derivation, `data_groups` registry)
lands in `asymmetry.core` first, with GUI panels calling into it. All `.asymp`
changes are additive and default when absent; the only mildly destructive
migration is load-time dedupe of identical-signature series (Phase 1.3), which
merges only truly identical series and logs what it merged.

## Sequencing

Phases are PR-sized and mostly independent; suggested order
`0.x → 1 → 2 → 4 → 5 → 3 → 6 → 7 → 8`, gated by `python tools/harness.py
validate`. See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the
step-by-step anchors.

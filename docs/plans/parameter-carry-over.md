# Parameter carry-over: values follow the component, not the name

Status: implemented 2026-09-05 (PR 1, Phases 1–5). Two PRs: PR 1 = Phases 1–5
(carry-over across every builder host); PR 2 = Phase 6 (one seeding function
with value provenance), designed here so PR 1 leaves the right seams, started
only after PR 1 merges. Each phase is one subagent task on the feature branch
with a lead review gate.

## Problem

Accepting the fit-function builder rebuilds the host's parameter table. Every
host has its own answer to "what happens to the values I had", and none of
them is right:

| Host | On accept today |
|---|---|
| Single fit tab (`single_tab.py:557` `_set_composite_model`) | `FitParameterTable.populate` from defaults. **Everything reset.** |
| Global tab, run-batch table (`global_tab.py:1318`) | Name-keyed snapshot of value/type/bounds reused when the name survives. |
| Global tab, grouped physics tables (`global_tab.py:4694`, `:4752`) | Same name-keyed snapshot; bounds patched back by name after `populate`. |
| Global tab, per-run / per-member initial values (`global_tab.py:1346`) | Dictionaries wiped on any model change. |
| Parameter trending (`model_fit_dialog.py:1965`; twice in `cross_group_fit_dialog.py`) | Name-keyed carry of value/min/max/fixed; trend seeds for new names; a hand-written exception resets Redfield's `m`. |
| Simulate dialog (`simulate_dialog.py:447`, `:773`) | Name-keyed merge of old values into new defaults, duplicated. |

Name keying is wrong in both directions because naming is collision-driven:

- `Exponential + Constant` names `A_1, Lambda, A_bg`; adding a second
  exponential renames `Lambda` to `Lambda_1`, so the value is **lost**.
- Inserting a component *before* another shifts indices, so `A_2` now
  denotes a different component and the value is carried onto the **wrong
  row**.
- Two different components can share a local name (`Linear.m` and
  `Redfield.m`), so replacing one with the other carries a meaningless value;
  the Redfield exception is a patch over exactly this.

Both builder dialogs subclass `FunctionBuilderDialog` and drive the same
`ModelRowList`, which already maintains parallel lists through every edit,
so the builder *knows* which component instance became which.

## Design (settled)

**Rule.** A parameter's value, constraints and roles belong to the component
instance that owns it. When the model changes, they follow that instance
wherever it moves and whatever its parameters are now called. A component
instance that did not exist before gets seeded defaults. Nothing is keyed
by name across a model change.

1. **Origins.** `ModelRowList` gains a parallel list `origins: list[int | None]`,
   the index of each row's component in the model the dialog opened with, or
   `None` for a component added since. It is maintained by exactly the code
   paths that maintain `component_names` (insert, delete, move, duplicate,
   group/ungroup, `set_structure`, `clear`). `duplicate_row` copies the
   origin (a duplicate starts from the original's values — that is what
   duplicating means). Text mode (`_apply_text` → `set_structure`) sets
   origins by `align_component_names(current_names, new_names)` composed
   with the current origins: same-named components matched in order
   (longest common subsequence); unmatched are `None`. The dialog exposes
   `component_origins() -> tuple[int | None, ...]` beside the model
   accessor; hosts always call both.

2. **Parameter identities (core).** Each model class exposes
   `parameter_identities() -> dict[str, ParameterIdentity]` mapping every
   unique parameter name to a value object that names *what* the parameter
   is, independent of its spelling:
   - `ComponentParameter(component: int, local_name: str)` — a component's
     own parameter (via `parameter_mapping()`, sentinel entries excluded);
   - `GroupAmplitude(components: frozenset[int])` — a fraction group's
     shared amplitude, keyed by the group's component set;
   - `FractionWeight(term_start: int)` — a free fraction, keyed by the
     component that starts its term.
   `CompositeModel` returns all three kinds; `ParameterCompositeModel`
   returns `ComponentParameter` only. Each identity has
   `remap(origins) -> ParameterIdentity | None`: substitute old component
   indices for new ones through `origins`; `None` when any referenced
   component is new.

3. **Translation (core, model-agnostic).** New module
   `src/asymmetry/core/fitting/parameter_carry.py`:
   ```
   carry_parameters(old_identities, new_identities, origins, state) -> dict[str, T]
   ```
   For each `(new_name, identity)` in `new_identities`: `old =
   identity.remap(origins)`; if some `old_name` in `old_identities` has that
   identity and `old_name in state`, then `result[new_name] = state[old_name]`.
   Generic over the payload `T`; entries not carried are simply absent so the
   caller seeds them. Two typed conveniences on top: `carry_parameter_set`
   (`ParameterSet` in, `ParameterSet` out, value/min/max/fixed/link_group
   carried) and `carry_parameter_entries` (the GUI's list-of-dict state).
   `align_component_names(old, new) -> tuple[int | None, ...]` lives here too.
   - A carried entry is a **seed**: `uncertainty` / `uncertainty_asymmetric`
     are cleared (they described a fit of the old model).
   - **Ties** reference other parameters by name; a tie is carried with its
     referenced names translated through the same map, and dropped when a
     referenced parameter has no successor (there is nothing to tie to).
   - **Link groups, batch roles, Fix** are carried as they are.
   - `origins` may repeat an index (duplicate rows); both successors carry.

4. **Hosts.** Every builder host does the same three steps on accept:
   snapshot its current state → `carry_*` through the dialog's origins →
   build the new form with seeds, then apply the carried entries over them.
   The seeding step is whatever the host does today (Phase 6 unifies it).
   The name-keyed snapshots, the bounds patch-back, the Redfield exception,
   the duplicated simulate merge and the per-member wipe are deleted.
   `FitParameterTable` needs no new API: `parameters_state()` before,
   `populate()` then `restore_parameters(carried)` after.

5. **Reset.** The builder's *Reset* / an explicit "reset parameters" action
   is the only route to defaults for a surviving component. Accepting the
   dialog with no structural change carries everything (origins are the
   identity map), so it is a no-op on the table.

Out of scope for PR 1: changing what the seeds are (Phase 6); the
individual-groups per-detector nuisance table (keyed by detector group, not
touched by a model edit); the global parameter fit window (not a builder
host).

## Phase 6 design (PR 2): one seeding function, with provenance

Today's defaults are static per component (`ComponentDefinition.param_defaults`)
plus ad-hoc layers: applied field into `field`/`B_L`
(`seeding._field_value_overrides`), frequency-domain peak seeds
(`seed_peak_parameters_from_dataset`), individual-groups overrides
(background and phase fixed at 0), trend seeds (`suggest_trend_seeds`), and
a dataset-switch reseed that only touches a cell if it still equals the
*previous* auto-default (`tab_base._refresh_field_defaults_in_table`), a
"was it untouched?" guess by value comparison.

1. `seed_parameters(model, context) -> dict[str, Seed]` in core, where
   `Seed = (value, fixed, min, max)` and `context` carries what is known
   (dataset or datasets, applied field, domain, mode flags such as
   individual-groups). Layers, in order, later overriding earlier:
   component static defaults → dataset-derived physics (field into
   `field`/`B_L`; amplitude scale and background tail from the record, as
   the wizard's fingerprint already computes; frequency from field or the
   FFT peak in the frequency domain) → mode overrides (individual groups:
   background and phase fixed at 0). A sibling `seed_trend_parameters`
   wraps `suggest_trend_seeds` for `ParameterCompositeModel`.
2. Every value cell records **provenance**: `seeded` (written by
   `seed_parameters`) or `user` (typed, restored from a project, or written
   back from a fit). A carried value keeps its provenance.
3. On a dataset switch, `seed_parameters` runs again and replaces
   `seeded` cells only. `_refresh_field_defaults_in_table` and
   `reseed_frequency_peaks` are deleted; the GUI-guide promise
   ("field-dependent parameters reseeded for the newly selected run") is
   met by construction.
4. `_field_value_overrides`, the grouped-mode default branches in
   `_rebuild_grouped_*`, and the trending seed call become layers of the
   one function; hosts call `seed_parameters` and nothing else.

## Agent rules (embedded verbatim in every phase prompt)

- Work only on branch `feat/parameter-carry-over` (or the worktree you were
  given; if a worktree, prefix every python/harness call with
  `PYTHONPATH=<worktree>/src`). Confirm with `git branch --show-current`
  before committing. Never commit to `main`.
- Prevent bad states by construction; do not add defensive guards against
  undesirable behaviour: no `hasattr`/`getattr(..., None)` on our own
  attributes, no `try/except` around our own code, no "in progress" /
  "initialized" flags or latches, no `if x is None: return` where the design
  makes `None` impossible. A genuine optional (a real "no data" case, a
  file-boundary migration) is fine; a hedge is not. If you cannot see how to
  avoid a guard, stop and report instead of adding one "for now".
- Delete, don't deprecate. Rewrite tests that pin old behaviour to the new
  contract rather than skipping or weakening them.
- Use the harness, never bare pytest: `python tools/harness.py test --
  <files>`, `python tools/harness.py test --tier fast`,
  `python tools/harness.py lint`, `python tools/harness.py structural`.
  Use `.venv/bin/python`. Do not run `validate` or the GUI subset unless
  your phase says so.
- Commit at the end of your phase with a conventional-commit message ending
  in `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Report:
  what changed, what you deleted, every test you rewrote and why, anything
  you could not do.

## Phases, agents and gates

The lead reviews every diff for deleted name-keyed paths, absent guards and
rule compliance, then runs the phase's tests before a dependent phase starts.

### Phase 1 — identities, translation, alignment (Opus)

Files: new `src/asymmetry/core/fitting/parameter_carry.py`;
`composite.py` (`parameter_identities`), `parameter_models.py`
(`parameter_identities`); new `tests/core/test_parameter_carry.py`.

Design §2–§3 exactly. Tests: append renames `Lambda` → `Lambda_1` and the
value follows; insert-before shifts `A_2` and the value follows the
component, not the name; delete drops only the deleted component's
entries; reorder; duplicate (repeated origin) carries to both; replace
`Linear` with `Redfield` carries nothing for `m`; fraction group amplitude
and free fractions carry through a group whose components all survive and
do not when one is new; tie re-targeting and tie dropping; uncertainty
cleared; `ParameterSet` and entry-list conveniences; `align_component_names`
on append / insert / delete / duplicate-name cases. Gate: that file +
`tests/core/test_composite_model.py` + `--tier fast`; lead API review.

### Phase 2 — origins in the builder (Sonnet, worktree, parallel with 1)

Files: `gui/widgets/function_builder/model_rows.py`,
`gui/widgets/function_builder/dialog.py`;
`tests/gui/test_function_builder_rows.py`, `tests/gui/test_function_builder_dialog.py`.

Design §1. Depends on Phase 1 only for `align_component_names`; until
Phase 1 lands, implement the text-mode alignment against a local stub and
switch the import at rebase (the lead cherry-picks after Phase 1).
Tests: each edit operation's effect on origins, including group/ungroup
and fraction toggles (origins unchanged), duplicate (copied origin), text
mode round trip (`Exponential + Constant` → typed `Exponential +
Gaussian + Constant` gives origins `(0, None, 1)`), and
`component_origins()` on accept for both dialog subclasses. Gate: those
files.

### Phase 3 — fit-panel hosts (Opus, after 1 and 2)

Files: `gui/panels/fit/single_tab.py`, `gui/panels/fit/global_tab.py`,
`gui/panels/fit/tab_base.py` only if `restore_parameters` needs a tie
translation hook; `tests/gui/test_fit_panel_tabs.py`,
`tests/gui/test_fit_parameter_table.py`, grouped-mode tests
(`grep -rln "grouped\|individual" tests/gui/`).

Design §4 for: the single tab; the run-batch table; both grouped physics
tables; `_user_initial_values_by_run` and `_user_grouped_initial_values`
(translate each inner dict instead of wiping). Delete
`_current_parameter_row_state`-style name reuse in `_set_composite_model`
and `_rebuild_grouped_*`, and the bounds patch-back. Tests: for each of the
five surfaces, add a component before and after an existing one and assert
values, Fix, bounds and (where present) type/link/tie land on the right
rows; remove a component; accept with no change is a no-op. Gate: the
listed files + `--tier fast`; `validate` after this phase.

### Phase 4 — trending, cross-group, simulate hosts (Sonnet, after 1 and 2; parallel with 3 in a worktree)

Files: `gui/panels/model_fit_dialog.py`, `gui/panels/cross_group_fit_dialog.py`,
`gui/windows/simulate_dialog.py`; their tests
(`tests/gui/test_model_fit*.py`, `tests/gui/test_cross_group*.py`,
`tests/gui/test_simulate*.py` — find the actual names).

Design §4 for the three trending sites (delete
`_should_reset_param_on_model_change` and `_component_name_for_param`;
trend seeds apply to un-carried names as today) and the two simulate
surfaces (one shared helper, the duplicated merge deleted). Tests: second
`Linear` keeps the first's `m, b`; `Linear` → `Redfield` reseeds `m`;
simulate keeps typed values across an append. Gate: the listed files.

### Phase 5 — docs, changelog, validate (Sonnet, after 3 and 4)

`docs/reference/gui_usage.rst` (the builder section: values follow the
component; Reset for defaults), `composite_models.rst` if it mentions
defaults on model change, `CHANGELOG.md` **Changed** entry in the
long-form voice, this plan's status. `harness.py docs`, then `validate`
once; lead final review and PR.

### Phase 6 — seeding and provenance (Opus; PR 2)

Files: new `core/fitting/seeding.py` (move/absorb `gui/panels/fit/seeding.py`
helpers that are GUI-free), `tab_base.py` (provenance role on value
items), the hosts' dataset-switch paths, `docs/reference/gui_usage.rst`.
Design § "Phase 6". Tests: layer order; a `seeded` cell is replaced on
dataset switch and a `user` cell is not; provenance survives carry-over
and project round trip; the frequency-domain reseed and the field reseed
both fall out of the one function. Gate: `--tier fast`, affected GUI files,
`validate`.

## Gates

| After | Lead runs |
|---|---|
| 1 | `test_parameter_carry.py`, composite tests, `--tier fast`, API review |
| 2 | builder GUI tests (on the branch after cherry-pick) |
| 3 | fit-panel GUI tests, `--tier fast`, `validate` |
| 4 | trending / cross-group / simulate tests |
| 5 | `docs`, `validate`, PR |
| 6 | separate PR, same ladder |

## Decisions recorded

- Values follow component *instances*; the builder supplies the map, and
  in-order name alignment is used only where no map exists (text mode).
- A carried value is a seed: uncertainties are cleared.
- Ties re-target through the map; a tie to a vanished parameter is dropped.
- Duplicate copies the origin, so a duplicated component starts from the
  original's values.
- The Redfield `m` exception is deleted, not generalised: identity keying
  makes it unnecessary.
- Seeding unification and provenance are PR 2, designed now so PR 1's
  hosts keep a single seeding call site each.
- A tie or `expr` reference to a name outside the model entirely (a free
  auxiliary parameter the model itself never owned) passes through a carry
  unchanged — only a reference the old model itself owned is translated or
  dropped; the old model not recognising a name is not evidence it vanished.
- A duplicated component's tie points at the *original* component's
  parameter, not at its own duplicate — origin carries forward through
  `duplicate_row`, so both the original and the duplicate resolve the tie
  through the same predecessor.
- Name alignment (`align_component_names`) is the sanctioned fallback for
  every builder host that receives a model *without* an edit's exact origins
  behind it: the single tab's model inherited from a prior fit, and both
  send-to-batch paths (single → batch's run table, and the multi-group
  window's grouped single → grouped batch send). Each is `aligned_origins`
  against the model already in the receiving table, so a same-named
  component's role, value, and bounds survive and the map is the identity
  when the model is genuinely unchanged.
- The cross-group dialog's config-restore path stays a same-model, name-keyed
  restore (`_apply_existing_config`'s `parameter_rows` step): at that point
  there is no predecessor model to carry *from* — the model is being set for
  the first time from a saved file — so seeding defaults and then restoring
  the exact saved snapshot by name is correct, not a name-keying regression.
- Per-(run, group) nuisance values are never wiped by a model edit: their
  names are per-detector-group quantities, not fit-function parameters, so a
  composite-model change cannot invalidate them.

# Product amplitudes: one scale per product, derived from the expression tree

Status: in progress 2026-09-05 on `feat/product-amplitude-policy` (one PR),
built phase by phase by subagents with a lead-agent review gate after every
phase. The PR also carries the docs-pages fix (Phase 0), unrelated to the
policy but broken on `main` since PR #301.

## Problem

`CompositeModel` (`src/asymmetry/core/fitting/composite.py`) names and
evaluates amplitudes under two regimes chosen by *spelling*, not structure:

- **Flat expressions** (no parentheses) share one `A` per `*`/`/` chain
  (`_share_chain_amplitude`, `_build_param_mapping`'s `amplitude_group_starts`,
  the chain reduction in `function()`, and the `"1"` collapse in
  `_component_formula_term` / `_component_latex_term`).
- **Any parenthesis** switches the whole model to a second regime that keeps
  every component's own `A`, with one special case
  (`_identify_suppressed_amplitudes`: a leaf multiplied by an additive group
  loses its `A`).

Consequences, all verified 2026-09-05 with the model in this checkout:

- The fit wizard's multiplet templates (`_multiplet_model`,
  `fit_wizard.py:1589`) wrap each `Osc × Env` pair in redundant parentheses,
  get `A_1, A_2, A_3, A_4, A_bg`, and compensate at `fit_wizard.py:4493` by
  seeding each envelope amplitude to `1.0` and fixing it. The pinned
  parameter still reaches the parameter table, the formula, the LaTeX
  preview, the project file, and `_continuation_parameters` carries a note
  about preserving it.
- Typing `(Oscillatory * Exponential) + Constant` into the function builder
  gives the same `A_1*… * A_2*…` as the wizard; the builder only looks clean
  because users type flat products. The wizard's own flat single-line
  template (`Oscillatory * Exponential + Constant`) already gets one
  amplitude, so the wizard is inconsistent with itself.
- `Exponential * (Gaussian * Oscillatory)` keeps three amplitudes;
  `(Exponential + Gaussian) * Oscillatory * Exponential` keeps a fourth
  (`A_4`) on the trailing leaf.
- Flat `Constant * Exponential` reports `A_bg` and `A_1` but the formula
  string shows only `A_bg`: the flat evaluator multiplies by both.
- A model whose only surviving amplitude sits in a parenthesised expression
  is named `A`; the same model written flat names it `A_1`.

The docs already promise one rule with no parenthesis caveat
(`docs/reference/composite_models.rst` "Parameter naming rules",
`docs/reference/gui_usage.rst` "Parameter naming rules in the table").

## Design (settled)

**Rule.** Every product carries exactly one scale, and that scale lives in
the first factor that can carry one; every other factor of the product is
unit-amplitude. Amplitude naming is a function of the expression **tree**, so
parentheses that do not change the tree do not change the names.

1. **Expression tree.** `CompositeModel` builds, from its existing
   `(component_names, operators, open_parentheses, close_parentheses,
   fraction_groups)` — which stay the serialised form and the public
   constructor — a tree of three node kinds: `Leaf(component_index)`,
   `Sum(terms, signs, fraction_group: (start, end) | None)`, and
   `Product(factors, ops)` (`ops` are `*`/`/`). Standard precedence;
   parentheses group. Normalisation: a group containing a single node *is*
   that node (redundant parentheses vanish); a `Product` factor that is
   itself a `Product` is flattened into its parent; a plain `Sum` term that is
   itself a plain `Sum` is flattened. A fraction-group `Sum` is never
   flattened (it owns a scale, see 3). The GUI's row tree in
   `gui/widgets/function_builder/model_rows.py` is a *display* tree and is
   not touched; the core tree lives in `composite.py`.

2. **Which factor keeps its scale.** For each `Product`:
   - If any factor is a `Sum` (plain or fraction group), every `Leaf` factor
     is suppressed: the sum's terms (or its group amplitude) carry the
     scale. Two `Sum` factors both keep theirs — that shape is degenerate,
     but the user wrote two parametrised sums and we do not second-guess it
     (existing behaviour, pinned by
     `test_multiplying_two_additive_groups_keeps_group_amplitudes`).
   - Otherwise the **first** `Leaf` factor whose component declares a
     scaling parameter (`_is_scaling_parameter`: `A` or `A_bg`) keeps it;
     every other `Leaf` factor's scaling parameter is suppressed.
   A suppressed scaling parameter maps to `_UNIT_AMPLITUDE_SENTINEL`, is
   absent from `param_names`, evaluates as `1.0`
   (`_extract_component_kwargs` already does this) and renders as `1`
   (the existing `"1*"` stripping). `fixed_by_default_params` already skips
   the sentinel.

3. **Sums.** Each term of a plain `Sum` keeps its own scale (a term that is a
   `Product` keeps exactly one per rule 2). A fraction-group `Sum` keeps its
   group amplitude and derived fractions exactly as today; its terms' leaf
   scales are suppressed as today.

4. **Naming of the survivors.** `A` is always indexed by the **carrying
   component's 1-based index**: `A_1`, `A_3`, … (this is today's flat
   convention, used by every wizard template, the docs, and saved projects).
   `A_bg` and every non-scaling parameter keep today's collision-driven
   naming (`A_bg`, `A_bg_2`; `Lambda` vs `Lambda_2`), which is unchanged
   because component indices and name counts are unchanged. Consequences:
   `(Oscillatory * Exponential) + (Oscillatory * Exponential) + Constant`
   names `A_1, frequency_1, phase_1, Lambda_2, A_3, frequency_3, phase_3,
   Lambda_4, A_bg` — identical to the flat spelling. `Constant *
   Exponential` names `A_bg, Lambda`. `(Gaussian + Constant) * Constant`
   names `A_1, A_bg` (was `A`).

5. **One evaluator.** `function()` evaluates the tree recursively:
   `Leaf` → component function with its mapped kwargs (suppressed → `1.0`);
   `Product` → fold with `*` / the existing guarded divide; plain `Sum` →
   signed fold; fraction `Sum` → group amplitude × Σ weight × term. The flat
   chain-reduction evaluator, `_evaluate_parenthesized`, the operator/paren
   stack machine, `_share_chain_amplitude`, `_uses_parentheses`,
   `_identify_suppressed_amplitudes`, `_rhs_group_contains_additive_operator`
   and `_lhs_group_contains_additive_operator` are deleted. `formula_string`,
   `latex_terms` and `evaluate_components` render from the mapping alone;
   their `_share_chain_amplitude` branches are deleted. The existing string
   assembly for formulas is kept (it already handles parentheses and drops
   `* 1`).

6. **Wizard.** `_multiplet_model` keeps its parentheses (they now cost
   nothing and keep the expression string readable). The envelope pinning
   (`overrides[_pname("A", env)] = 1.0` and `fixed_names.add(...)`) and the
   paragraph about it in `_continuation_parameters`' docstring are deleted.
   `_pname` and the `Lambda`/`sigma` seeding stay.

7. **Contract test.** A parametrised test over a list of expressions asserts
   that adding redundant parentheses (around every product, around every
   product factor pair, around the whole expression) leaves `param_names`
   and `function(t, **values)` identical, and that a chain of `n` leaves
   exposes exactly one amplitude. This is the invariant the two routes
   (wizard, builder) inherit.

8. **Migration — alpha only, deleted at v1.0.** Parameter *names* are
   persisted next to the model dict in projects (`schema.py`
   `state["parameters"]`), in the fit wizard cache and in global-wizard
   payloads. A saved parenthesised product carries amplitude entries the new
   model no longer exposes; a saved flat `Constant * Exponential` carries an
   `A_1` that no longer exists; a parenthesised model that named its lone
   amplitude `A` now names it `A_1`. The migration is one new module,
   `src/asymmetry/core/fitting/legacy_product_amplitudes.py`, with **no
   influence on the design above**:
   - It carries a frozen, private copy of the *pre-change* naming algorithm
     (`_build_param_mapping` + `_identify_suppressed_amplitudes` as they
     stand at `origin/main` 812db6d) as a pure function
     `legacy_parameter_mapping(model) -> list[dict[str, str]]` over the
     model's names/operators/parentheses/fraction groups.
   - `fold_legacy_product_amplitudes(model, entries)` pairs, per component
     and per scaling parameter, `legacy name → new name-or-sentinel`. For
     each `Product` in the new tree, the surviving amplitude's value becomes
     the product of every factor's legacy amplitude value (survivor included);
     relative uncertainties add in quadrature; the survivor is `fixed` only
     if every folded entry was fixed; the survivor keeps its own bounds.
     Renamed survivors (`A` → `A_1`) are renamed; dropped entries are
     removed; every other entry passes through untouched. Entries are the
     same plain dicts / `Parameter` objects the fraction migration handles.
   - It runs at exactly the sites the fraction migration runs
     (`grep -rn migrate_legacy_fraction src/`), guarded by the same cheap
     precondition style: only when some entry name is not in
     `model.param_names`.
   - `RELEASING.md` gets a "Delete at v1.0" list naming the module, its
     call sites and its test file, so the drop is a mechanical deletion.

Deliberately **not** in scope: deriving `_is_scaling_parameter` from the
component definition (so user functions with a differently named scale can
participate). Noted as a follow-up.

## Agent rules (embedded verbatim in every phase prompt)

- Work only on branch `feat/product-amplitude-policy` (or the worktree you
  were given). Confirm with `git branch --show-current` before committing.
  Never commit to `main`.
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
  `python tools/harness.py lint`. Do not run `validate` or the GUI subset
  unless your phase says so.
- Commit at the end of your phase with a conventional-commit message ending
  in `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Report:
  what changed, what you deleted, every test you rewrote and why, anything
  you could not do.

## Phases, agents and gates

The lead reviews every diff for conciseness (nothing kept that the design
makes unnecessary), rule compliance (no new guards), and contract
compliance (naming from the tree, one evaluator), then runs the phase's
tests before the next dependent phase starts.

### Phase 0 — docs-pages fix (Sonnet, worktree, parallel with Phase 1)

`Docs Build and Deploy` has failed on every push to `main` since 45e5110
(PR #301). Run 33980559227: scenario `global_fit_wizard_result`
(`docs/screenshots/scenarios/global_fit_wizard_result.py`) raises
`RuntimeError: starting value(s) are required` from iminuit inside
`global_fit_wizard._staged_assignment_seed` → `fit_engine.global_fit` →
`FitEngine._fit_core`: a per-run local-only fit reached Minuit with no free
parameters. PR #301 pins `B_L` from run metadata and returns fixed
parameters as fixed from the engine; the Ag LF decoupling series pins `B_L`
on every run. Reproduce with `python -m docs.screenshots.capture --only
global_fit_wizard_result` (check `capture.py` for the exact flag), find why
the staged seed asks for a fit with an empty free set now that pinned
parameters stay pinned, and fix it by construction in the wizard's
role/stage logic (a pinned parameter is neither Global nor Local and never
enters a stage's active set), not by catching the error. Add a regression
test in `tests/core/test_global_fit_wizard.py` (or the applied-field test
file) with a series whose only local parameter is pinned. Gate: that test
file + `--tier fast`; the scenario captures locally.

### Phase 1 — tree, mapping, evaluator, rendering (Opus)

Files: `src/asymmetry/core/fitting/composite.py`;
`tests/core/test_composite_model.py`, `tests/core/test_latex_preview.py`,
`tests/core/test_composite_parameters.py`, plus any test that constructs a
parenthesised product and pins the old names (`grep -rln "open_parentheses\|
from_expression(\"(" tests/`).

Implement Design §1–§5 and §7. Rewrite the parenthesised-product tests
(`test_parenthesized_product_suppresses_*`,
`test_lhs_additive_group_*`, `test_formula_string_shows_single_amplitude_*`)
to the new names; add the contract test; add cases for
`Exponential * (Gaussian * Oscillatory) + Constant` (one amplitude),
`Constant * Exponential` (`A_bg` only), the two-line multiplet spelled both
ways. Gate: the listed files + `--tier fast` green; lead review of the tree
API before Phases 2 and 3 start.

### Phase 2 — legacy fold (Opus, on the branch, after Phase 1)

Files: new `src/asymmetry/core/fitting/legacy_product_amplitudes.py`, new
`tests/core/test_legacy_product_amplitudes.py`, the four-ish call sites
found by `grep -rn migrate_legacy_fraction src/`, `RELEASING.md`.

Implement Design §8. Tests: a two-line multiplet saved with `A_2 = A_4 = 1`
fixed loads with `A_1`/`A_3` unchanged and no `A_2`/`A_4`; a free `A_2 =
0.5` folds into `A_1`; `A` → `A_1` rename; a flat `Constant * Exponential`
project drops `A_1` into `A_bg`; an already-migrated entry list is a no-op
and does not rebuild the model. Gate: that file + the project/representation
tests that load fixtures (`tests/project/`, `tests/core/
test_representation_model.py`) + `--tier fast`.

### Phase 3 — wizard de-pinning (Sonnet, worktree, parallel with Phase 2)

Files: `src/asymmetry/core/fitting/fit_wizard.py` (§6),
`src/asymmetry/core/fitting/global_fit_wizard.py` (check nothing there
assumes an envelope amplitude exists), `tests/core/test_wizard_damped_seeding.py`
(the `A_2` fixed-restart assertions at ~992/1028 — retarget to a genuinely
fixed parameter such as a pinned `B_L`/`field`), `tests/core/test_fit_wizard*.py`.
Gate: those test files + `--tier fast`.

### Phase 4 — docs, changelog, GUI sweep, validate (Sonnet, after 1–3)

Files: `docs/reference/composite_models.rst`, `docs/reference/gui_usage.rst`
(naming rules: parentheses do not change names; a product carries one scale
in its first factor; a product with a sum carries the sum's scales),
`docs/reference/fit_wizard.rst` if it mentions envelope amplitudes,
`CHANGELOG.md` `[Unreleased]` (Changed: one amplitude per product in every
route, with the multiplet before/after; Fixed: docs deploy; note the
alpha-only migration), `docs/plans/product-amplitude-policy.md` status →
implemented with decisions recorded, screenshot scenarios that seed an
`A_2`-style envelope amplitude (`grep -rn "A_2\|A_4" docs/screenshots/`),
`tests/gui/test_function_builder_rows.py` and any GUI test pinning old
names. Run `python tools/harness.py docs`, then `python tools/harness.py
validate` once. Gate: green validate; lead final review; PR.

## Gates

| After | Lead runs |
|-------|-----------|
| 0 | `tests/core/test_global_fit_wizard.py` (+ new test), scenario capture |
| 1 | Phase 1 test files, `--tier fast`, API review |
| 2 | Phase 2 test files, `tests/project/`, `--tier fast` |
| 3 | wizard test files, `--tier fast` |
| 4 | `docs`, `validate`, diff review, open PR |

## Decisions recorded

- Survivor is the **first** scale-bearing factor, not "prefer `A` over
  `A_bg`": one rule, predictable from left to right.
- `A` stays always-indexed (`A_1`), matching the flat convention every
  saved project and template already uses; the parenthesised `A` spelling
  is migrated.
- `_multiplet_model` keeps its parentheses.
- Sum × Sum keeps both sets of amplitudes (unchanged, degenerate by the
  user's choice).
- Migration uses a frozen copy of the old algorithm so the new code has no
  legacy branch; the whole thing is one module plus call sites, deleted at
  v1.0.

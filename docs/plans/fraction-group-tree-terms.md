# Fraction groups: term ranges from the expression tree, not parenthesis counts

Status: implemented 2026-09-05 on `feat/fraction-group-tree-terms` (one PR),
by one Opus subagent with a lead review gate. Follow-up to
`docs/plans/product-amplitude-policy.md` § "Follow-ups" (PR #305).

## Problem

`CompositeModel` (`src/asymmetry/core/fitting/composite.py`) still derives a
fraction group's additive term ranges by walking the per-component
parenthesis counts (`_term_ranges(start, end, inside_group=True)`), assuming
the group's own parenthesis is the *only* one opening on its first component
and closing on its last. Any other parenthesis sharing an endpoint with the
group throws the depth accounting off and construction raises. Verified on
`main` f6314cb:

| Expression | Today |
|---|---|
| `(Oscillatory * (Exponential + Gaussian){frac})` | `ValueError: Invalid parentheses while parsing term ranges` |
| `((Exponential + Gaussian){frac} + Constant)` | same |
| `(Exponential + (Gaussian + Constant){frac})` | same |
| `((Exponential + Gaussian){frac})` | `Fraction groups require at least two additive terms` (the outer paren is counted as depth) |
| `Oscillatory * (Exponential + Gaussian){frac}` | OK |

The expression tree added in PR #305 (`build_expression_tree`, `ExprSum`
with `fraction_group`) already resolves nesting correctly: a fraction group
is an `ExprSum` whose `terms` are exactly the group's additive terms. The
count-walkers are a second, weaker parser of the same structure. The same
count-walk pattern appears in `_top_level_terms` (with a `try/except`
fallback "so callers always get output") and in
`with_default_fraction_groups` (a `try/except ValueError: return self`).
These are the guards the tree makes unnecessary.

## Design (settled)

**Rule.** The expression tree is the only source of structure. Term ranges of
a fraction group, and the top-level additive terms of the model, are read
from tree nodes. No method walks `open_parentheses`/`close_parentheses` to
find terms.

1. **Build order.** `__init__` builds the tree *before* validating fraction
   groups, passing the raw `fraction_groups` list through. In
   `build_expression_tree`, a group is tagged onto the `ExprSum` produced for
   the parenthesis whose span equals the group (the existing token-payload
   mechanism); `parse_group` returns an `ExprSum(..., fraction_group=group)`
   for a tagged parenthesis **even when it holds one term** (delete the
   "validation guarantees two or more terms" assumption) so validation can
   inspect it. A group whose span matches no parenthesis tags nothing.

2. **Validation on the tree** (`_validate_fraction_groups`, rewritten):
   per group, in order — shape/type checks as now; duplicate as now; the
   tree must contain an `ExprSum` with `fraction_group == group`, else
   "Fraction groups must map to one parenthesized expression"; that sum must
   have at least two terms, else "Fraction groups require at least two
   additive terms"; every sign must be `+1`, else "Fraction groups only
   support additive '+' terms"; components of two groups must not overlap,
   else "Fraction groups cannot overlap" — **nested fraction groups stay
   rejected** (Ben, 2026-09-05: one fraction group per component). Existing
   error messages are kept verbatim; tests pin them.

3. **Term ranges from the node.** `_fraction_group_term_ranges(group)` returns
   `[(min(leaf_indices(term)), max(leaf_indices(term))) for term in node.terms]`
   where `node` is the tagged `ExprSum` (store a `dict[group -> ExprSum]`
   built once from `iter_nodes`). `_fraction_group_term_starts` is unchanged
   on top of it. `_build_fraction_term_number_map` is unchanged.

4. **Top-level terms from the root.** `_top_level_terms` returns, when the
   root is a plain `ExprSum`, one `(start, end, separator)` per term with the
   separator from the term's sign (`""`, `" + "`, `" - "`); otherwise a
   single term `(0, n-1, "")`. A fraction-group root is a single term (its
   weights are rendered inside it, as today). Delete the `try/except`.
   `with_default_fraction_groups` uses the same root inspection: a plain
   `ExprSum` root with all `+` signs and ≥ 2 terms is wrapped (as today,
   adding one parenthesis pair only when the whole span is not already one);
   anything else returns `self`. Delete its `try/except`.

5. **Delete** `_term_ranges` and `inside_group`. `_parenthesized_group_ranges`
   stays only if something other than validation still uses it (the
   whole-span check in `with_default_fraction_groups` does); otherwise delete
   it too.

6. **Rendering.** `_formula_string_with_fraction_groups`,
   `_latex_span_fragment` and `latex_terms` print parentheses as typed
   (walking the counts to *emit* brackets is fine; deriving *structure* from
   them is not). Verify their output for every expression in the Problem
   table: the formula must show the group's weights inside the group and the
   redundant brackets where the user wrote them. If a renderer derives
   structure from counts and mis-renders, re-express that part on the tree;
   do not special-case.

7. **Contract test.** `test_redundant_parentheses_change_neither_names_nor_values`
   in `tests/core/test_composite_model.py` currently excludes spans sharing an
   endpoint with a fraction group, with a comment explaining why. Remove the
   exclusion and the comment: every redundant span is now accepted and must
   leave `param_names` and values unchanged.

8. **GUI.** The function builder parses typed expressions through
   `parse_composite_expression` → `CompositeModel`, so the newly accepted
   spellings reach `ModelRowList.set_model` / the display tree in
   `gui/widgets/function_builder/model_rows.py`. That tree keys fraction
   containers by `(start, end) in fraction_groups`, so `((a + b){frac})`
   renders two nested containers with the same span, both accented. Accept
   that (it is what the user typed); add one GUI test that setting each
   Problem-table expression on the builder renders without error and the
   round-tripped expression string is unchanged.

Out of scope: nested fraction groups; `-` terms inside a group; making the
GUI display tree derive from the core tree (it must show redundant
parentheses as typed, which the normalised core tree drops).

## Agent rules (embedded verbatim in the prompt)

- Work only on branch `feat/fraction-group-tree-terms`. Confirm with
  `git branch --show-current` before committing. Never commit to `main`.
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
  Use `.venv/bin/python` (the harness re-execs into it).
- Commit with a conventional-commit message ending in
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Phase and gate (one Opus agent)

Files: `src/asymmetry/core/fitting/composite.py`;
`tests/core/test_composite_model.py`, `tests/core/test_latex_preview.py`,
`tests/core/test_composite_parameters.py`; one test in
`tests/gui/test_function_builder_rows.py`; `docs/reference/composite_models.rst`
(one sentence under fraction groups: grouping is structural, extra
parentheses around or inside a group are accepted); `CHANGELOG.md`
`[Unreleased]` **Fixed** entry; this plan's status.

Tests to add in `test_composite_model.py`: a parametrised construction test
over the Problem table's five expressions asserting `param_names`,
`fraction_groups`, `fraction_parameter_groups()`, `derived_fraction_names()`
and `formula_string()`; a values test that `(Oscillatory * (Exponential +
Gaussian){frac})` evaluates identically to `Oscillatory * (Exponential +
Gaussian){frac}`; error-message tests for each rejection in Design §2
(including nested groups and a group on a product span
`(Exponential * Gaussian){frac}`); `with_default_fraction_groups` on a root
that is a product (returns self) and on a plain sum (wraps).

Gate (agent, then lead): the listed test files; `--tier fast` (five `.nxs`
corpus tests fail locally without `pyhdf`; ignore); `lint`; `structural`;
`validate` once. Lead reviews the diff for deleted count-walkers and absent
guards, then opens the PR.

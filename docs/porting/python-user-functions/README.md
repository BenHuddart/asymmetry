# Python user functions — study

**Status:** implemented (2026-06-12); decisions and verification results
recorded below and in [verification-plan.md](verification-plan.md).
**Umbrella:** `wimda-parity-gap` Wave B; promotes the
`python-user-functions` candidate (`docs/porting/candidates/python-user-functions/`,
retained for scoring history).

## What

A user-extensibility mechanism replacing both WiMDA plugin DLL APIs with a
Python-native registration + discovery system: users add fit functions to
Asymmetry by writing a single Python file, without touching the source tree.
The result is the lowest-friction plugin path among the reference programs —
no Delphi/FORTRAN DLLs (WiMDA), no C++/ROOT compilation (musrfit), no
algorithm framework (Mantid).

## WiMDA reference (verified at `$WIMDA_SRC/src`)

WiMDA ships **two** plugin APIs; one Python mechanism covers both. The full
verified contracts are in [comparison.md](comparison.md).

1. **`musrfunctions.dll`** (`$WIMDA_SRC/src/MusrFunctionUnit.pas`): adds
   oscillation/relaxation entries to the time-domain pickers. Plugin
   functions have signature
   `f(t, p1, p2, p3, phase, x2, multifit) -> double` — up to three free
   parameters, plus the detector-group **phase** variable (so plugins
   participate in grouped-phase fitting) and a per-run auxiliary variable
   `x2` gated by the **multifit** flag.
2. **`*fit.dll` model libraries** (`$WIMDA_SRC/src/UserUnit/`): general
   y(x) model functions discovered by glob, enumerated via
   `getfnlist`/`getparams`/`details`, with an optional `getresults`
   derived-quantities callback and a FORTRAN calling-convention variant.

## Scope (settled with Ben, 2026-06-12)

- **Registration facade** in `asymmetry/core/fitting/user_functions.py`:
  `register_component(...)` (time/frequency fit components → `COMPONENTS`)
  and `register_parameter_component(...)` (parameter-trend components →
  `PARAMETER_MODEL_COMPONENTS`), wrapping the existing definition
  dataclasses. **`MODELS` stays internal-only**: `models._register` is
  refactored onto the same internal validated registration core so there is
  exactly one registration path, but no public `register_model` — pickers
  and composite expressions are built on `COMPONENTS`, which covers every
  real user workflow.
- **N4 registry-naming formalisation** lands here (standing programme
  decision, `../wimda-parity-gap/reconciliation-study.md` §5 N4 and
  `docs/ARCHITECTURE.md` §4.3): the facade requires a valid domain tag and
  enforces cross-registry name uniqueness, so a name registered through it
  identifies its registry and domain unambiguously. Existing registry names
  are not churned; the single legacy cross-registry collision (`Constant`)
  remains grandfathered behind the kind-aware docs lookup
  (`tests/test_fit_function_docs.py::test_name_collisions_resolve_by_registry_kind`).
- **Load-time validation**: parameter-name/defaults consistency, vectorised
  evaluation, finite outputs on a probe grid — clear errors at load, never
  mid-fit.
- **Discovery**: see Decisions below.
- **Metadata parity**: user components carry a `user=True` flag on their
  definitions; pickers and the component-info dialog show a "User"
  provenance badge; GLE/plot labels work through the same
  `formula_template`/`param_info` machinery as built-ins. The three
  docs-enforcement tests exempt user components **by flag, not by name
  list** (collision directive W17).
- **Persistence degrade**: an `.asymp` referencing a user component that is
  not installed degrades to a **named placeholder** — never a silent drop
  (collision directive W1; the pre-existing silent fallback in
  `fit_panel.restore_state` is replaced). Persistence changes are additive;
  **no schema_version bump**.
- **Worked example + tutorial**: one plugin file re-implementing the
  shipped `Keren` component bit-for-bit through the plugin path; the same
  file is the user-guide tutorial listing ("zero to fitted in one page").
  The tutorial is persona-neutral — "you have a relaxation function
  Asymmetry doesn't ship" — and cites A. Keren, Phys. Rev. B **50**, 10039
  (1994) (the dynamic-relaxation study already established the textbook
  does not cover this function).

### Design-only (recorded, not built)

- **`getresults` equivalent** (derived-quantities hook): see
  [implementation-options.md](implementation-options.md) §6.
- **Sandboxing**: none. User code runs with full interpreter privileges —
  exactly the WiMDA DLL trust model. Documented in the user guide.
- **Hot reload**: not built; restart picks up changes. The load report
  names the scanned directory to make the loop obvious.

## Decisions (settled with Ben, 2026-06-12)

| # | Decision | Choice |
|---|----------|--------|
| D1 | Public registration surface | `register_component` + `register_parameter_component`; `MODELS` internal-only, `_register` aligned onto the shared core |
| D2 | Discovery mechanism | User directory `~/.asymmetry/user_functions/*.py` scanned by an explicit `load_user_functions()` call at GUI startup, **plus** `importlib.metadata` entry points (group `asymmetry.user_functions`) for packaged plugins. Project-local plugin files are **out** (provenance value judged below the import-arbitrary-code-on-project-open risk) |
| D3 | N4 naming scheme | Bare grammar-compatible names (`[A-Za-z_][A-Za-z0-9_]*`), required `domain` tag, uniqueness enforced across **all three** registries at registration |
| D4 | Load-failure UX | Never crash: per-file try/except into a structured load report; failures surface in the startup log panel; full report in a "User functions…" dialog (Setup menu) |
| D5 | Missing-component degrade | Named placeholder component (evaluates to zero, flagged) so the model opens, plots, and re-saves with the original names intact; fitting blocked with a message naming the missing components |
| D6 | Example/tutorial | Keren, persona-neutral framing (Ben, mid-study course correction) |

## Key seams

- `core/fitting/composite.py` — `ComponentDefinition` (+ new `user` flag),
  `COMPONENTS`, `CATEGORY_REGISTRY`, expression grammar.
- `core/fitting/parameter_models.py` — `ParameterModelComponentDefinition`
  (scopes, `fwhm_factor`, ⊕ quadrature grammar), `PARAMETER_MODEL_COMPONENTS`.
- `core/fitting/models.py` — `_register` (aligned, stays private).
- `core/fitting/component_docs.py` — kind-aware applicability/reference
  lookup; user components provide applicability/references at registration.
- `core/fitting/domain_library.py` — domain-filtered picker views; its
  import-time frequency-tag guard is satisfied because registration happens
  strictly after import and requires a valid domain tag (W17).
- `gui/app.py` `main()` — discovery call before `MainWindow()`.
- `gui/panels/fit_panel.py` `restore_state` — the silent-fallback site the
  placeholder degrade replaces.
- `core/io/__init__.py` — loader-registry precedent for explicit,
  auditable registration calls.

## References

- A. Keren, Phys. Rev. B **50**, 10039 (1994).
- S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt (eds.),
  *Muon Spectroscopy: An Introduction* (OUP, 2022).

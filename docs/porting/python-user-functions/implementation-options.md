# Implementation options — python-user-functions

Decisions marked **Decision (Ben, 2026-06-12)** were settled before
implementation. Everything here is core-first: the GUI only displays what
the core registers and reports.

## 1. Registration facade

New module `asymmetry/core/fitting/user_functions.py` (Qt-free), exporting:

```python
register_component(
    name, function, param_names, *,
    domain,                      # REQUIRED: "time" | "frequency" (N4)
    description, formula_template,
    param_defaults=None,         # default 1.0 per param when omitted
    latex_equation="", applicability="", references=(),
    fixed_params=(),
)

register_parameter_component(
    name, function, param_names, *,
    description, formula_template,
    param_defaults=None,
    latex_equation="", applicability="", references=(),
    scopes=("common",), fwhm_factor=None,
)
```

Both build the existing frozen definition dataclasses and insert into
`COMPONENTS` / `PARAMETER_MODEL_COMPONENTS` through a single internal
`_validated_insert` core that performs the N4 name check and the load-time
validation (§3). Applicability/references go into the existing
`component_docs` dicts through the same kind-aware machinery built-ins use,
so the info dialog and GLE labels need no special-casing.

### Options considered for the facade ↔ `models._register` alignment (W17)

- **(a) Public `register_model` as a third entry point** — rejected:
  `MODELS` entries don't reach the composite pickers, so the public API
  would grow surface without GUI value.
- **(b) One internal validated core; `models._register` refactored onto
  it; `MODELS` stays internal** — **Decision (Ben, 2026-06-12)**. One
  registration path, two public wrappers.

### Provenance flag (W17)

`ComponentDefinition` and `ParameterModelComponentDefinition` gain
`user: bool = False`. The facade sets `user=True`. Exemption in the three
docs-enforcement tests, picker badging, and the load report all key off
this **flag, never a name list**. Built-in registrations are untouched
(default `False` keeps every existing literal valid).

## 2. N4 naming scheme

Problem (reconciliation study N4): three name-keyed registries span
domains, so a bare name does not identify a function's registry or domain.

- **(a) Mandatory user prefix/namespace** (`U_MyDecay`, `user.MyDecay`) —
  rejected: uglier expressions; dotted names need grammar changes;
  provenance is already badged in the picker.
- **(b) Per-registry uniqueness only** — rejected: lets cross-registry
  ambiguity grow.
- **(c) Bare grammar-compatible names + required `domain` tag +
  cross-registry uniqueness** — **Decision (Ben, 2026-06-12)**.

Concretely, `_validated_insert` rejects a name that (i) fails
`[A-Za-z_][A-Za-z0-9_]*` (the expression-tokenizer atom), (ii) is a
grammar-reserved token, or (iii) already exists in **any** of
`COMPONENTS`, `MODELS`, or `PARAMETER_MODEL_COMPONENTS`. A
facade-registered name therefore maps to exactly one (registry, domain)
pair. The legacy `Constant` collision is grandfathered behind the
kind-aware lookup
(`test_name_collisions_resolve_by_registry_kind` unchanged and green).
Built-in registration via the shared core applies the same checks minus
the cross-registry rule for the grandfathered pairs.

## 3. Load-time validation (fail at load, never mid-fit)

WiMDA calls plugin exports blindly; a bad DLL crashes mid-fit. The facade
validates at registration, raising `UserFunctionError` with a message that
names the offending plugin file/symbol:

1. **Signature**: `function` callable with `1 + len(param_names)`
   positional floats (checked by probe call, not introspection, so
   builtins/partials work).
2. **Parameter consistency**: `param_defaults` keys ⊆ `param_names`;
   names valid identifiers; no duplicates.
3. **Vectorised evaluation**: called on a probe grid (time domain:
   `t ∈ [0, 32] µs`, 257 points incl. 0; frequency: `ν ∈ [0, 50] MHz`;
   parameter models: their default x grid) with the default parameter
   values; output must be an ndarray of the input shape.
4. **Finite outputs**: `np.isfinite` everywhere on the probe grid at
   defaults. (Defaults only — pathological corners of parameter space stay
   the user's responsibility, as for built-ins.)
5. **Metadata**: non-empty description and formula_template; domain valid
   (`domain_library.DOMAINS`); scopes ⊆ {common, field, temperature}.

Registration is atomic: validation happens before any registry mutation,
so a failed registration leaves all registries untouched (registry
isolation — plugins can never half-register or mutate built-ins; the
facade refuses overwrites entirely).

## 4. Discovery

**Decision (Ben, 2026-06-12)**: both channels, behind one explicit call.

```python
asymmetry.load_user_functions(directory=None)  # → UserFunctionLoadReport
```

- **User directory** `~/.asymmetry/user_functions/*.py` (sorted, non-
  recursive, names not starting with `_`), imported under unique synthetic
  module names (`importlib.util.spec_from_file_location`) so file names
  can't shadow installed packages.
- **Entry points** `importlib.metadata.entry_points(group="asymmetry.user_functions")`
  — each resolves to a callable invoked once (it performs its
  registrations); for pip-installed plugin packages.
- Each file/entry point loads under `try/except Exception` into a
  structured `UserFunctionLoadReport` (per-source: path/name, components
  registered, error summary). Nothing raises out of discovery.
- The **GUI calls this in `app.main()` before `MainWindow()`**; plain
  `import asymmetry` stays side-effect-free (scriptability invariant,
  deterministic tests). Scripting users call it explicitly — one line,
  documented in the tutorial.
- **Project-local plugins (alongside `.asymp`): out.** Opening a shared
  project file must not execute arbitrary code. Recorded trade-off: the
  provenance value is real (project travels with its functions) and can be
  revisited behind an explicit per-project consent prompt if demand
  appears.

Idempotence: a second `load_user_functions()` call returns a fresh report;
already-registered names fail the uniqueness check and are reported as
duplicates rather than re-registered (restart remains the reload story —
hot reload is out of scope).

## 5. Load-failure UX

**Decision (Ben, 2026-06-12)**: log panel + Setup-menu dialog.

- Startup: one log-panel line per failed source (tag `user-fn`), one
  summary line (`N user functions loaded from M files`, when nonzero).
- Setup menu gains "User functions…" opening a dialog listing every
  scanned source with status, registered names, and the error text for
  failures, plus the scanned directory path. Read-only, post-#53 visual
  conventions, no gold-plating.
- Crashes impossible by construction (§4 try/except per source).

## 6. `.asymp` degrade — named placeholder (W1)

**Decision (Ben, 2026-06-12)**: zero-valued placeholder.

Today `FitSlot.model` survives load as a raw dict, but
`fit_panel.restore_state` silently swaps in `Exponential + Constant` when
`CompositeModel.from_dict` raises — a silent drop, the exact W1 failure.

Design:

- `CompositeModel` construction gains an opt-in placeholder path
  (`CompositeModel.from_dict(data, allow_missing=True)` or equivalent):
  unknown names become per-instance `ComponentDefinition` placeholders
  (`missing=True` marker, zero-valued function, single amplitude-less
  parameter set) that are **never inserted into `COMPONENTS`**.
- The model opens in the panel/builder with its original expression;
  present components still plot; the missing ones contribute zero.
- Fitting a model containing placeholders is blocked with a message naming
  the missing components ("requires user function(s) X, Y — see Setup →
  User functions").
- `to_dict()` emits the **original names**, so load → save round-trips
  bit-identically with the plugin absent.
- Persistence additive only; **no schema_version bump** — the serialised
  form is unchanged (names were always stored as strings).

## 7. Worked example + tutorial

One file, `docs/user_guide/examples` + tests fixture: a plugin
re-implementing `Keren` through `register_component` with the same
arithmetic as `models.keren` (bit-for-bit on the probe grid and through
fit + persistence round-trip). Tutorial page
`docs/user_guide/user_functions.rst`: persona-neutral ("a relaxation
function Asymmetry doesn't ship"), zero-to-fitted in one page, the trust
model stated plainly, scripting (`load_user_functions()`) covered in a
footnote section. Physics citation: A. Keren, Phys. Rev. B **50**, 10039
(1994).

## 8. Design-only (recorded, not built)

- **`getresults` equivalent**: a future
  `derived_quantities=lambda params: {...}` keyword on
  `register_parameter_component` (and possibly fit components), returning
  labelled scalars computed from fitted parameters, surfaced in fit
  results and trend exports. WiMDA semantics to mirror: post-fit only,
  optional, absent hook is silent (`Model.pas:301`). Needs a story for
  uncertainties (propagate via covariance? document as point estimates?)
  — deferred until a concrete user need picks one.
- **Sandboxing**: none, by decision. User code runs with full interpreter
  privileges — the WiMDA DLL trust model, stated in the user guide. The
  refused project-local channel (§4) is the only place this bit.
- **Hot reload**: restart picks up changes; the load-report dialog names
  the directory to make that loop obvious.

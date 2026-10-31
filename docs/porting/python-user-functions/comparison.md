# Comparison: user-function plugin APIs across reference programs

All WiMDA citations verified against a local checkout at `$WIMDA_SRC/src`
(`__history/`/`__recovery/` ignored). Line numbers refer to that checkout.

## 1. WiMDA API 1 — `musrfunctions.dll` (time-domain picker plugins)

Source: `$WIMDA_SRC/src/MusrFunctionUnit.pas` (interface),
`$WIMDA_SRC/src/MusrFunctionUnit/musrfunctions.dpr` (shipped example DLL),
`$WIMDA_SRC/src/Analyse.pas:5726–5819` (loading and picker integration).

### Function contract

A plugin function is a `stdcall` export returning a double. Its arguments, in
order, are the time, up to three free fit parameters, the detector-group
**phase**, a per-run auxiliary value and a **multifit flag**.

- WiMDA passes the fitted per-group phase (degrees) in, so a plugin oscillation
  participates in grouped-phase fitting: the shipped `TFrot` example adds it to
  its own phase parameter before taking the cosine.
- When the flag is true (multi-run fitting), the auxiliary value carries the
  run's scan variable (field, delay, …); when false the plugin falls back to a
  parameter, as the `DelayRot` example does.

### Enumeration and metadata exports

Each of the two categories has three exports: a list routine (`GetOscFnList`,
`GetRelFnList`) returning the function count and names, a details routine
(`OscFnDetails`, `RelFnDetails`) returning a help string, and a lookup
(`GetOscFn`, `GetRelFn`) returning the parameter count, parameter names and the
function pointer.

Oscillation vs relaxation is purely categorical — two picker lists. User
entries are appended after the built-ins. Parameter names label the GUI
parameter rows; the details strings feed help text.

### Discovery, FORTRAN variant, errors

- Fixed filename `musrfunctions.dll`, searched in the configured `libdir`
  else the WiMDA directory (`Analyse.pas:5726–5739`), `LoadLibrary`.
- FORTRAN convention detected by probing `GetOscFn_`/`GetRelFn_`
  (underscore suffix) first (`Analyse.pas:5742–5757`).
- Missing exports → error dialog, DLL rejected; missing DLL → non-blocking
  info dialog; **never fatal** (`Analyse.pas:5761–5785`).

## 2. WiMDA API 2 — `*fit.dll` model libraries (UserUnit)

Source: `$WIMDA_SRC/src/UserUnit/` (`UserUnit.pas`, `FitTyps.pas`, example
`UserFit.dpr`), `$WIMDA_SRC/src/Model.pas:1160–1341` (loading/invocation).

### Function contract

A model function is a `stdcall` export taking the chosen ordinate, a reserved
secondary ordinate and the parameter array, and returning a double. The ordinate
is whatever column the user selected (time, field, temperature…). χ² is computed
by WiMDA, not the DLL.

### Required exports

Three exports are required and one is optional:

- `getfnlist` → library display name (e.g. "User Model Examples") + function
  name list; populates a library combo then a model combo.
- `getparams` → parameter count/names + the function pointer.
- `details` → per-model help text shown by a Details button.
- `getresults` (optional) → post-fit hook, called with the fitted parameter
  array after **single-run** fits only, when the library provides it; may write
  derived quantities back in place. Lookup failure is silent.

### Discovery, FORTRAN variant, errors

- Glob `*fit.dll` in `libdir`/WiMDA dir; **multiple libraries coexist**
  (`Model.pas:1171–1224`).
- FORTRAN variant: if `getparams` is absent WiMDA probes `GETPARAMSP`,
  which returns the function as an integer address that is then cast
  (`Model.pas:1193, 197–205`); upper-case `GETFNLIST`/`DETAILS` also
  probed.
- A DLL missing a required export is rejected with a dialog and
  `FreeLibrary`; load continues with the remaining libraries.

### Shipped example (`UserFit.dpr`)

Registers "Quadratic fit" `y = a x² + b x + c` and "Power law fit"
`y = a_const + a_power xⁿ` via an init block
(`setname`/`setfunc`/`setpar`/`setdesc`) — the hello-world shape the
Asymmetry tutorial mirrors in Python.

## 3. musrfit and Mantid (for context)

- **musrfit**: user functions are C++ classes inheriting `PUserFcnBase`,
  compiled against ROOT and loaded through the ROOT plugin manager
  (`$MUSRFIT_SRC/src/tests/userFcn/`). Full power, highest friction.
- **Mantid**: `IFunction`-derived C++ plugins (or Python algorithms)
  registered with the framework's function factory. Python is supported
  but inside Mantid's algorithm framework, not a one-file drop-in.

## 4. Asymmetry — current state and mapping

Extensibility today = composite expressions over built-ins + editing the
source registries. The seams the plugin mechanism builds on:

| WiMDA concept | Asymmetry equivalent |
|---|---|
| Osc/rel picker entry (API 1) | `ComponentDefinition` in `COMPONENTS` (`core/fitting/composite.py:353`); category + domain tags drive the picker (`domain_library.components_for_domain`) |
| `p4` group phase | Components declare a `phase` parameter; grouped fitting handles per-group phase as an engine-level nuisance parameter (`docs/ARCHITECTURE.md` §4.3 FT-11) — no special plugin signature needed |
| `p5`/`p6` multifit variable | Batch/global fitting over a run series with per-run metadata (field, temperature) is an engine concern (`FitSeries` trending); user components stay pure `f(t, …)` |
| Model library entry (API 2) | Same `ComponentDefinition` mechanism — Asymmetry's composite grammar already covers y(x) sums/products; parameter-trend models map to `ParameterModelComponentDefinition` (`core/fitting/parameter_models.py:307`) |
| `unitname` library label | Source-file/distribution name in the load report |
| `details` text | `description` + applicability text + APS-style references through the kind-aware lookup (`core/fitting/component_docs.py`) |
| `getresults` | Design-only derived-quantities hook ([implementation-options.md](implementation-options.md) §6) |
| DLL discovery (fixed name / glob) | `~/.asymmetry/user_functions/*.py` scan + `importlib.metadata` entry points |
| Non-fatal load errors with dialogs | Structured load report: log panel at startup + Setup-menu dialog; bad file skipped, app never crashes |
| Trust model (DLL = full privileges) | Identical: imported Python runs unsandboxed; documented in the user guide |

Gaps WiMDA does **not** have that Asymmetry must handle:

- **Validation at load** (WiMDA calls blindly; a bad DLL crashes mid-fit).
  Asymmetry probes each function on a grid at registration.
- **Persistence**: WiMDA fit setups don't round-trip plugin state the way
  `.asymp` serialises `CompositeModel.to_dict()` component names — hence
  the named-placeholder degrade for missing user components.
- **Docs-enforcement tests** (`tests/test_fit_function_docs.py`) iterate
  the registries; user components are exempted by their `user` flag.

# Comparison: user-function plugin APIs across reference programs

All WiMDA citations verified against a local checkout at `$WIMDA_SRC/src`
(`__history/`/`__recovery/` ignored). Line numbers refer to that checkout.

## 1. WiMDA API 1 — `musrfunctions.dll` (time-domain picker plugins)

Source: `$WIMDA_SRC/src/MusrFunctionUnit.pas` (interface),
`$WIMDA_SRC/src/MusrFunctionUnit/musrfunctions.dpr` (shipped example DLL),
`$WIMDA_SRC/src/Analyse.pas:5726–5819` (loading and picker integration).

### Function contract

```pascal
TMusrFunction = function(t, p1, p2, p3, p4, p5: double; p6: boolean)
  : double; stdcall;
```

- `t` — time; `p1–p3` — up to three free fit parameters.
- `p4` — the detector-group **phase** variable (degrees). WiMDA passes the
  fitted per-group phase in, so a plugin oscillation participates in
  grouped-phase fitting. Example (`musrfunctions.dpr`, `TFrot`):
  `cos((p1*t+(p2+ph)/360)*2*pi)`.
- `p5`/`p6` — per-run auxiliary variable + **multifit flag**. When `p6` is
  true (multi-run fitting), `p5` carries the run's scan variable (field,
  delay, …); when false the plugin falls back to a parameter, e.g.
  `if mult then delay:=xx else delay:=p2` (`DelayRot`).

### Enumeration and metadata exports

```pascal
procedure GetOscFnList(var nfunc: integer; var fnam: array of pansichar); stdcall;
procedure GetRelFnList(var nfunc: integer; var fnam: array of pansichar); stdcall;
function  OscFnDetails(n: integer): pansichar; stdcall;
function  RelFnDetails(n: integer): pansichar; stdcall;
procedure GetOscFn(n: integer; var np: integer; var pn: array of pansichar;
                   var fu: TMusrFunction); stdcall;
procedure GetRelFn(...);  // same shape
```

Oscillation vs relaxation is purely categorical — two picker lists. User
entries are appended after the built-ins (`Analyse.pas:5790`
`C1oscSel.max := otMax + nosc`). Parameter names (`pn`) label the GUI
parameter rows; `…Details` strings feed help text.

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

```pascal
ffun = function(x: double; x2: double; p: array of double): double; stdcall;
```

`x` is whatever column the user selected (time, field, temperature…);
`x2` is a reserved secondary ordinate; `p` is the parameter array. χ² is
computed by WiMDA, not the DLL (`Model.pas:1459`).

### Required exports

```pascal
procedure getfnlist(var unitname: pansichar; var nfunc: integer;
                    var fnam: array of pansichar); stdcall;
procedure getparams(n: integer; var np: integer;
                    var pn: array of pansichar; var fu: ffun); stdcall;
function  details(n: integer): pansichar; stdcall;
procedure getresults(r: array of double); stdcall;  // OPTIONAL
```

- `getfnlist` → library display name (`unitname`, e.g. "User Model
  Examples") + function name list; populates a library combo then a model
  combo.
- `getparams` → parameter count/names + the function pointer.
- `details` → per-model help text shown by a Details button
  (`Model.pas:1263`).
- `getresults` → optional post-fit hook, called with the fitted parameter
  array after **single-run** fits only
  (`if (not mfitting) and (@getresults <> nil) then getresults(p)`,
  `Model.pas:301–302`); may write derived quantities back in place. Lookup
  failure is silent.

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

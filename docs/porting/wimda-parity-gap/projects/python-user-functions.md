# Project brief: python-user-functions

Umbrella: `wimda-parity-gap` · Wave B (after `model-function-parity`) ·
Size M · promotes the `python-user-functions` candidate

## Motivation

WiMDA's plugin DLLs are a frequently-cited strength: users extend the
program without rebuilding it. All *shipped* DLL contents are now ported, so
what remains is the extensibility mechanism itself. The Python-native
equivalent is far lower-friction than WiMDA's Delphi/FORTRAN DLLs or
musrfit's C++ plugins.

## WiMDA reference

Two distinct APIs, both to be covered by one Python mechanism:
`musrfunctions.dll` (`MusrFunctionUnit.pas`; adds oscillation/relaxation
entries to the time-domain pickers; signature receives t, p1–p3, phase,
x2/B and a multifit flag, so plugins participate in grouped-phase fitting)
and `*fit.dll` model libraries (`UserUnit.pas`:
`getfnlist/getparams/details` + optional `getresults` derived-quantities
callback; FORTRAN calling-convention variant).

## Scope

- A registration API: `asymmetry.register_component(...)` /
  `register_parameter_component(...)` wrapping the existing
  `ComponentDefinition` / `ParameterModelComponentDefinition` registries
  (both are plain dicts — the seam already exists).
- A discovery mechanism (design choice for the study):
  user-functions directory (`~/.asymmetry/user_functions/*.py`) imported at
  startup, and/or `importlib.metadata` entry points for packaged plugins.
  Project-local functions (alongside an `.asymp`) considered — has
  provenance value but security/UX implications to weigh.
- Metadata parity with built-ins: category (with a "User" provenance badge),
  formula template, LaTeX, applicability text — so user components appear in
  the picker, docs dialog and GLE labels like first-class citizens; the
  docs-enforcement tests must treat user components as exempt.
- Validation at registration (parameter names, vectorised evaluation,
  finite outputs on a probe grid) with clear error messages — fail at load,
  not mid-fit.
- `getresults` equivalent: optional derived-quantities hook on parameter
  models (returns labelled values computed from fitted parameters) — design
  only unless time allows.
- A worked example + user-guide chapter (the WiMDA DLL examples — quadratic,
  power-law — make good hello-world ports).

**Out**: binary plugin loading; sandboxing (document that user code runs
with full privileges — same trust model as WiMDA DLLs); hot-reload (nice,
not required).

## Current Asymmetry state

Extensibility = composite expressions + editing the source registries.
Candidate folder: `docs/porting/candidates/python-user-functions/`.

## GUI/UX sketch

No new surface beyond a "User functions" section in the existing pickers
and a settings entry showing the load report (which files loaded, which
failed and why). Failures surface in the log panel at startup, never as
crashes.

## Conflicts & dependencies

Primary surfaces: `composite.py`/`parameter_models.py` registry seams, new
`core/plugins.py`. Sequenced after `model-function-parity` (same registry
files). Serialisation question for the study: what happens when an `.asymp`
references a user component that isn't installed — degrade gracefully with
a named placeholder, never silently drop.

## Verification sketch

A plugin re-implementing `Keren` matches the built-in bit-for-bit through
the plugin path (registration, picker, fit, persistence round-trip);
load-failure tests (bad signature, NaN output) produce the designed errors;
project round-trip with a user component present/absent.

# Link groups — implementation options & chosen design

## Option 1 — `expr` string ties (rejected)

`Parameter` already carries an unused `expr: str | None` "tie to another param".
We could parse `expr` ("= A_3", "= frequency_1 + delta") in the engine.

Rejected: this is the *offset-tie* machinery the brief explicitly says not to
assume, it needs a mini expression evaluator + uncertainty propagation through
arbitrary expressions, and it does **not** match WiMDA (equality only). Keep
`expr` reserved; do not build on it here.

## Option 2 — first-class equality link groups (CHOSEN)

A small, WiMDA-faithful equality constraint.

### Core (`asymmetry.core.fitting`)

- **`Parameter.link_group: int | None`** (new optional field, default `None`).
  `None` = unlinked; equal positive ints = same group. Mirrors WiMDA's
  `plinkgroup[]`.
- **`ParameterSet`** helpers:
  - `link_groups() -> dict[int, list[Parameter]]` — members per group (ordered).
  - `link_main(group) -> Parameter` — group main: first **non-fixed** member,
    else first member (mirrors WiMDA's "first, unless a later one varies").
  - `link_followers() -> dict[str, str]` — `follower_name -> main_name` for every
    non-main member.
  - `free_parameters` now also excludes followers (they are constrained).
- **`FitEngine.fit`**: build the free set without followers; in the model
  wrapper substitute `kw[follower] = kw[main]` before calling the model; after
  the fit, assign each follower `value = main.value` and
  `uncertainty = σ(main)` (propagated equality). Reduced-χ² uses the reduced
  free-parameter count. `global_fit` is unchanged (its global/local mechanism
  already covers cross-dataset sharing; single-fit links are the WiMDA analogue
  for one run).

### Serialization (`.asymp`)

`FitSlot.parameters` is a list of plain dicts. Add `"link_group"` to the
per-parameter dict on save and restore it on load. No schema-version bump is
required: older projects simply lack the key and deserialize to
`link_group=None` (backward compatible). A round-trip test guards this.

### GUI (`SingleFitTab`, `src/asymmetry/gui/panels/fit_panel.py`)

- Add a **"Link"** column (a per-row combo: `—`, `1`, `2`, `3`, `4` — four
  groups, named for WiMDA consistency). The label shows which group each
  parameter belongs to.
- `_run_fit` reads each row's link group into `Parameter.link_group`.
- `get_single_state`/`restore_state` persist/restore the per-row link group so
  it survives tab switches and `.asymp` round-trips.
- After a fit, follower rows display the main's value and (propagated)
  uncertainty, consistent with the engine result.

### Equal spacing for CdS

Per the study, equal spacing is **recovered from the data** with three free
frequencies; link groups share amplitude/phase/relaxation. No offset tie. The
hyperfine constant is the satellite splitting `f₊ − f₋`.

## Future (out of scope here)

A `MuoniumTriplet` / `ZFMuonium` model **component** parameterised by
`(centre_frequency, splitting, amplitude, …)` would *enforce* satellite symmetry
and expose the hyperfine constant as a single fitted parameter — the WiMDA
`ZFmuonium` route (1). It composes cleanly with link groups (which would then
share relaxation/phase across components) but is a separate model-library
addition, not required to match WiMDA's link-group feature or to pass the CdS
acceptance.

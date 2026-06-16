# Link groups — implementation options & chosen design

## Option 1 — `expr` string ties (rejected)

`Parameter` already carries an unused `expr: str | None` "tie to another param".
We could parse `expr` ("= A_3", "= frequency_1 + delta") in the engine.

Rejected: this is the *offset-tie* machinery the brief explicitly says not to
assume, it needs a mini expression evaluator + uncertainty propagation through
arbitrary expressions, and it does **not** match WiMDA (equality only). Keep
`expr` reserved; do not build on it here.

> **Follow-on (session-5 CdS):** offset ties *were* later added — as a typed
> `AffineTie`, see [§ Affine ties](#affine-ties-session-5-follow-on) — but
> **not** via this `expr`-string route. The objection above (mini evaluator +
> arbitrary-expression uncertainty) still stands, so `expr` remains reserved.
> The typed tie is a linear map of at most two parameters, so its delta-method
> uncertainty is closed-form. This option stays rejected *as stated*.

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

## Affine ties (session-5 follow-on)

The original brief scoped offset ties out (*match WiMDA; equality only*). The
**session-5 API run** then surfaced a requirement WiMDA-parity does not serve:
with three *free* satellite frequencies the CdS satellite **amplitudes scatter**
(the third frequency trades against them), so the Mu⁰ **ionisation energy E_i is
un-extractable** (the run got `E_i = 43 ± 1090 meV`). Enforcing *equal spacing*
removes that free frequency and stabilises the amplitudes — but equality links
cannot express `f_lo = f_c − δ`, `f_hi = f_c + δ`.

**Decision: add a typed `AffineTie`** (a deliberate capability *beyond* WiMDA,
which has no affine tie):

```
follower = scale · main + offset_scale · offset + const
```

- `main` and the optional `offset` are parameter names; `offset` may be a free
  **auxiliary** parameter the model never consumes (e.g. the half-splitting
  `delta`), fitted with its own uncertainty so the hyperfine constant `2·δ`
  carries an error bar.
- A tie follower drops out of the free set (via `Parameter.is_constrained`),
  exactly like a link follower. The engine substitutes it at the same
  `model_wrapper` seam as link followers, after them (so a tie may reference a
  link-resolved value). Uncertainty propagates by the closed-form delta method
  `var = JᵀCJ` over the tie's references (including their cross-covariance).
- **Not** the rejected `expr`-string route (Option 1): a linear map of ≤2
  parameters needs no expression evaluator and no general-uncertainty machinery.
  `Parameter.expr` stays reserved for any future *nonlinear* ties.
- Validation: a parameter cannot be both link-grouped and tied, and ties may not
  chain to other ties (single-pass resolution).

Verified against the real CdS corpus (`tests/test_cds_affine_ties_corpus.py`,
corpus-conditional): the tied fit reaches χ²ᵣ ≈ 1, recovers `A_µ = 2δ ≈ 0.24 MHz`
with a tight error, the satellites collapse by ~30 K, and **E_i = 8.6 ± 3.1 meV**
(finite, physical, σ ≪ value — gap closed) versus the free fit's
`39.5 ± 211 meV`.

## Also future (still out of scope)

A `MuoniumTriplet` / `ZFMuonium` model **component** parameterised by
`(centre_frequency, splitting, amplitude, …)` would *enforce* satellite symmetry
inside a single line-shape — the WiMDA `ZFmuonium` route (1). This was in fact
*shipped separately* (`MuoniumTF`/`MuoniumLowTF`/`MuoniumZF`, see
[../muonium-triplet/](../muonium-triplet/README.md)) but **verified to over-fit
CdS** (χ²ᵣ ≈ 22: it also pins the central line to γ_µ·B). The affine-tie route
above is the lighter, better-conditioned constraint for shallow-donor CdS —
enforcing only the satellite symmetry, leaving the central line and splitting
free.

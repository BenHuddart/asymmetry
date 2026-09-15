# Implementation options

## 1. Parameterisation — CHOSEN: plateaus

- **Plateaus `A1`, `A2`** (Steele, Mantid). One sign convention fits rising (wTF)
  and falling (ZF tail) steps; both plateaus are directly readable. Chosen.
- Amplitude + background (`A_s`, `A_bg`, Khasanov). Maps to sample vs
  background, but a falling step needs a negative amplitude or a flipped
  exponent. Rejected; the mapping is documented instead.

## 2. Names

`A1`, `A2` (literature/Mantid), `Tc` (reused — same unit/floor as
`OrderParameter`/`CriticalDivergence`), `dT` rendered ΔT (Khasanov's ΔT_N).
`w` was avoided because the registry's `weight` already renders as "w".

## 3. Width positivity — CHOSEN: parameter floor

`dT` divides `T − Tc`, so `dT = 0` is 0/0 at `T = Tc`. The `dT` ParamInfo carries
`default_min = 1e-6` K, which the seed table and fit panels apply as the lower
bound, so the fit cannot reach zero width. The function uses `scipy.special.expit`
so a floor-width step does not overflow.

## 4. Seeds — CHOSEN: closed form in `suggest_trend_seeds`

The trend dialog's seed table (`seed_trend_parameters`) reads only
`suggest_trend_seeds`; `suggest_model_seeds` merges it. Plateaus from the
coldest/warmest fifth, Tc from the half-step crossing, dT from the 10–90 %
crossings.

## 5. Scope

Temperature only, category "Critical behaviour". Mantid's function is x-agnostic;
a field-axis step can be added by widening `scopes` if a use case appears.

## Deferred

- **Error-function sibling** (`ErfStep`, Kamusella 2017) for a Gaussian
  distribution of local T_c.
- **Multi-step sum** (Khasanov 2025). `FermiStep + FermiStep` works but each term
  carries its own plateaus, so the offsets are degenerate; a first-class form
  would share one baseline and use step heights.
- **Dialog default**: the trend dialog could default to `FermiStep` for an
  asymmetry-vs-temperature trend; not done because many asymmetry trends are
  not steps.

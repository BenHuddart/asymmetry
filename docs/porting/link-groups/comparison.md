# Link groups — semantics and the equal-spacing question

## A. WiMDA link-group semantics (verified in source)

`updatelinks` and `UpdatePars` in `src/Analyse.pas` make two passes. The first
picks each group's main: its first member, unless a later member is flagged
*vary*. The second gives every member the main's value and clears the vary and
fix flags on every member other than the main, so followers are not free fit
parameters.

| Property | WiMDA behaviour |
| --- | --- |
| Constraint type | **Equality only** — each follower takes its main's value. No offset/ratio/linear tie exists anywhere. |
| Group count | **4**. |
| Main selection | First member of the group, **unless** a later member is flagged *vary* — then that one becomes main. Ensures the free set always contains the group main. |
| Followers & the free set | Followers have their vary flag cleared: they **drop out of the free-fit parameter set**. Only the main is fitted. |
| Follower uncertainty | Not independently computed (follower ≡ main). |
| Fit log | `"<param> linked to <main>"` printed per follower (~5212). |
| UI | `LinkGroupForm`: click a parameter's coloured square → pick No-link / 1 / 2 / 3 / 4. |

## B. The equal-spacing question (the crux)

The CdS docx (`CdS 2026.docx`) instructs:

> "You can fit the signal to **three independent oscillating functions**. Use
> **Ties** to keep the lines equally spaced and have common amplitudes, phases
> or relaxation rates if appropriate. Add a flat background…"

Equality links **cannot** express `f₂ = f₁ + δ`, `f₃ = f₁ − δ` — they can only
force `f₁ = f₂ = f₃`. So how does WiMDA keep three lines equally spaced?

**Exhaustive source search** (two independent passes over `src/**`,
`FEATURE_MAP.json`, `SYMBOL_MAP.json`): WiMDA has **no** tie/offset/spacing
constraint of any kind. Searches for `tie`, `spacing`, `equally`, `offset`,
`difference`, `constrain` turn up nothing in the fitting core. The only
parameter coupling is the equality link group above.

WiMDA achieves equal spacing by one of two routes:

1. **Muonium line-shape model** — `ZFmuonium` in
   `src/Extrafunctions/muoniumfunctions.dpr` sums three weighted cosines with a
   shared phase, at f₁ = p₁ − p₂, f₂ = p₁ + p₂/2 and f₃ = (3/2)p₂.

   Three frequencies are **derived from two parameters** (centre p₁, splitting
   p₂). Symmetry is hard-coded; the splitting is one free parameter.
   `TFMuonium`/`LowTFMuonium` do the same for applied-field geometry.

2. **Three free frequencies** — the user fits three independent oscillating
   components, lets the frequencies float, and the fit recovers the (physically
   symmetric) triplet from the data. Link groups then share only the *shareable*
   parameters (amplitude, phase, relaxation). The docx's own follow-up
   ("The phases of the satellite lines may have to be allowed to vary…",
   "Should the amplitudes be the same?") shows the lines are treated as
   independent components, not a single hard-coded line-shape.

**Conclusion.** The docx "Ties" wording is loose. In WiMDA the CdS three-line
fit is route (2): three independent oscillating functions, equality link groups
sharing amplitude/phase/relaxation, frequencies free, equal spacing **recovered
from the data** and the hyperfine constant read off as the satellite splitting
`f₊ − f₋`. Route (1) is an alternative that *enforces* symmetry via a dedicated
model.

## C. What this means for Asymmetry

The brief says: *match WiMDA; do not assume linear/offset ties are required.*
So the Asymmetry port implements **equality link groups** (route 2), which is a
faithful, minimal match. Equal spacing is recovered, not enforced — exactly as
WiMDA does it with three free components.

One deliberate **improvement over WiMDA**: a follower in Asymmetry reports the
**propagated uncertainty of its main** (identical, since the values are equal),
rather than a blank. This satisfies the brief's "report propagated
uncertainties" and is the natural delta-method result for an equality map
(∂follower/∂main = 1).

Route (1) — a `MuoniumTriplet`/`ZFMuonium` model component exposing a single
splitting parameter — is recorded as a clean follow-on (see
implementation-options.md §Future), not needed to match WiMDA or to pass the
CdS bar.

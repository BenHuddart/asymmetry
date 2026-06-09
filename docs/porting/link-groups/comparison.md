# Link groups — semantics and the equal-spacing question

## A. WiMDA link-group semantics (verified in source)

`src/Analyse.pas`, `updatelinks` (~6514–6558) and `UpdatePars` (~4612):

```pascal
{ choose the main of each group: first member, unless a later member is "vary" }
if (linkgroupmain[j] = 0) or pvar[i] then
  linkgroupmain[j] := i;
...
{ enforce equality and demote followers }
p[i] := p[linkgroupmain[j]];
if i <> linkgroupmain[j] then
begin
  pvar[i] := false;   { follower is NOT a free fit parameter }
  pfix[i] := false;
end;
```

| Property | WiMDA behaviour |
| --- | --- |
| Constraint type | **Equality only** — `p[follower] := p[main]`. No offset/ratio/linear tie exists anywhere. |
| Group count | **4** (`linkgroupmain: array[1..4]`). |
| Main selection | First member of the group, **unless** a later member is flagged *vary* — then that one becomes main. Ensures the free set always contains the group main. |
| Followers & the free set | Followers get `pvar := false`: they **drop out of the free-fit parameter set**. Only the main is fitted. |
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

1. **Muonium line-shape model** — `src/Extrafunctions/muoniumfunctions.dpr`,
   `ZFmuonium` (~98–110):

   ```pascal
   f1 := (p1 - p2);     { centre − splitting }
   f2 := (p1 + p2/2);
   f3 := 3/2 * p2;
   ZFmuonium := (a1*cos(2π(f1·t+ph)) + a2*cos(2π(f2·t+ph)) + a3*cos(2π(f3·t+ph)))/6;
   ```

   Three frequencies are **derived from two parameters** (centre `p1`, splitting
   `p2`). Symmetry is hard-coded; the splitting is one free parameter.
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

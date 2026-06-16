# Link groups (WiMDA "Ties")

Study-first port of WiMDA's parameter **link groups** — the feature that lets a
multi-line fit share one value across several parameters (e.g. a common
relaxation rate or phase across three precession lines), so the linked
parameters drop out of the free-fit set.

The motivating exercise is the WiMDA Muon School **CdS shallow-donor**
problem: a central Mu⁺ Larmor line plus two Mu⁰ satellite lines, fit as three
oscillating components, with shared amplitudes/phases/relaxation and the
hyperfine constant read off as the satellite splitting.

## Entry points (WiMDA, Delphi)

- `src/Analyse.pas`
  - `plinkgroup: array[0..pmax] of Integer` — per-parameter group id (0 = unlinked).
  - `linkgroupmain: array[1..4] of Integer` — index of the "main" parameter of
    each of the **four** groups.
  - `LinkSet` (~3544) — invoked when the user clicks a parameter's coloured link
    label; opens `LinkGroupSelect` and stores `plinkgroup[Pnum] := linkgroup`.
  - `updatelinks` (~6514) — two passes: (1) choose each group's main, (2) enforce
    `p[i] := p[linkgroupmain[j]]` and clear the vary flag on non-main members.
  - `UpdatePars` (~4612) — applies `p[i] := p[linkgroupmain[plinkgroup[i]]]`
    during evaluation (pure **equality**).
  - fit log (~5212) — prints `"<param> linked to <main>"` for each follower.
- `src/LinkGroupForm.pas` / `.dfm` — the 5-button picker: No link / 1 / 2 / 3 / 4
  (red / green / blue / magenta).
- `src/Extrafunctions/muoniumfunctions.dpr` — `ZFmuonium`, `TFMuonium`,
  `LowTFMuonium`: muonium line-shapes whose **splitting is a single parameter**.

See [comparison.md](comparison.md) for the precise semantics and the resolution
of the equal-spacing question, [implementation-options.md](implementation-options.md)
for the chosen Asymmetry design, [test-data.md](test-data.md) for the synthetic
fixture, and [verification-plan.md](verification-plan.md) for acceptance.

## Headline finding

WiMDA link groups are **equality-only**: every follower takes the *exact value*
of its group's main parameter. There is **no** offset/linear/ratio tie anywhere
in WiMDA. The CdS docx phrase *"Use Ties to keep the lines equally spaced"* is
loose wording — equal spacing is **not** enforced by link groups. WiMDA gets
equal spacing two ways:

1. a dedicated **muonium model** (`ZFmuonium`/`TFMuonium`) whose hyperfine
   splitting is one parameter, so symmetry is baked into the line-shape; or
2. **three free frequencies** that the fit recovers from the data, with link
   groups sharing only the *shareable* parameters (amplitude / phase /
   relaxation).

We port **(2)**: equality link groups, frequencies left free. That matches
WiMDA exactly and needs no offset-tie machinery (explicitly out of scope per the
brief). Equal spacing of the CdS satellites is recovered from the data, and the
hyperfine constant is read off as `f₊ − f₋`. See the comparison doc for why this
is faithful and what (1) would add later.

> **Update (session-5 CdS follow-on).** Recovering equal spacing from free
> frequencies left the satellite *amplitudes* noisy, so the Mu⁰ ionisation
> energy E_i was un-extractable. A typed **`AffineTie`** was therefore added to
> *enforce* equal spacing (`f_lo = f_c − δ`, `f_hi = f_c + δ`, δ free) — a
> deliberate capability **beyond WiMDA**, not a port. It is *not* the rejected
> `expr`-string route. See
> [implementation-options.md § Affine ties](implementation-options.md#affine-ties-session-5-follow-on)
> for the design and the CdS verification (E_i = 8.6 ± 3.1 meV, gap closed).

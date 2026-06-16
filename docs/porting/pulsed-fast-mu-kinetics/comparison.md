# Comparison — fast-Mu kinetics across references

How the teaching guide, the reference programs, and the muonium-chemistry
literature treat extraction of λ_Mu / the Mu fraction when the TF Mu signal is
largely lost before the first good bin at a pulsed source.

> **Reference-source caveat (record uncertainty explicitly).** `$WIMDA_SRC`,
> `$MUSRFIT_SRC`, `$MANTID_SRC` are **not checked out on this machine** (verified:
> the env vars are empty and no local checkouts exist). This comparison therefore
> rests on (a) the teaching guide docx, read directly; (b) Asymmetry's own prior
> porting studies, which already mined the WiMDA/Mantid/musrfit behaviour for the
> adjacent features used here; and (c) the muonium-chemistry literature cited by
> the guide. Claims tagged *(needs source re-trace)* should be confirmed against a
> real WiMDA/Mantid checkout before they harden into implementation contracts.

## Teaching guide — `Muonium reaction 2026.docx` (read directly)

The guide (full prose extracted) prescribes the *physics*, not a truncation
rescue:

- Measure in a **small TF**; "what does the **amplitude and relaxation** of the
  signal tell you?" → amplitude → muon fraction, relaxation λ_Mu → reaction rate.
- Pseudo-first-order: **λ_Mu = λ₀ + k_Mu·[x]**; use three maleic concentrations
  at room T for k_Mu.
- Arrhenius from ≥3 temperatures: **log₁₀ k_Mu = log₁₀ A − E/(2.3·R·T)** → E.
- It *explicitly poses the pulsed-source problem* but leaves it to the student:
  > "How might [the field choice] be influenced by the type of muon source
  > (pulsed or continuous)?" … "Why does the Mu signal have a **fast relaxation
  > rate**? Would this be a **problem for a kinetics experiment**? What might you
  > do to reduce the relaxation rate?"

So the guide is the behavioural contract for the *trend* maths (linear + the
verbatim Arrhenius form) but supplies **no** numeric targets and **no** method for
the case where λ_Mu is so large the oscillation is gone by t_g. That rescue is
standard muon-chemistry practice (below), and is the gap this port fills.

## The standard chemistry method — shared / fixed Mu amplitude ("fraction" method)

In the cited literature (D. C. Walker, *Muon and Muonium Chemistry*, Ch. 8;
Percival/Roduner/Fischer, *Chem. Phys. Lett.* 1977; Ng/Walker, *J. Phys. Chem.*
1981; Ghandi 2002), TF-µSR muonium kinetics rests on a fact the per-run fit
ignores: the **initial Mu amplitude/fraction `A_Mu(0)` is a constant of the
beam+solvent**, independent of scavenger concentration. Reaction changes only the
rate. Two equivalent extractions:

1. **Shared-amplitude fit (statistically principled).** Fit the concentration (or
   temperature) series *simultaneously* with `A_Mu` shared and λ_Mu per sample.
   The slow members pin `A_Mu`; the fast members then yield λ_Mu from their
   surviving amplitude even though a free per-run fit cannot.
2. **Amplitude/fraction inversion (analytic limit).** Measure the surviving Mu
   amplitude `A_surv` per run (frequency and phase fixed) and invert
   `λ_Mu = −ln(A_surv / A_Mu(0)) / t_eff` against a reference `A_Mu(0)` from the
   unreacted (water) sample. This is the "Mu fraction loss" the guide alludes to
   with "determine the muon fractions … from the TF amplitudes."

Both need `A_Mu(0)` to be the *same* across the series — the physical key in
`README.md`. *(needs source re-trace: whether WiMDA's own muon-school worksheet
drives this as a global fit or a manual fraction ratio — the corpus ships no
WiMDA `.fit` logs for this example, GROUND_TRUTH §7.)*

## WiMDA

- **Time-domain fit.** WiMDA fits a relaxing TF oscillation per run; its **link
  groups** (equality-only ties, `p[follower] := p[main]`) are the native way to
  hold one parameter common across a *multi-run* fit — i.e. the mechanism to share
  `A_Mu`. Mined in Asymmetry's `docs/porting/link-groups/` and ported as
  equality `link_group` ties; the asymmetry-domain analogue here is `fit_global`'s
  `global_params`. *(needs source re-trace for the exact muon-school recipe.)*
- **No integral rescue.** WiMDA's ALC/integral mode is a count-integral
  `(F−B)/(F+B)` over the good-bin window (mined in
  `docs/porting/time-integral-asymmetry/`) — designed for *non-oscillating*
  repolarisation/ALC scans, not a precessing TF signal, consistent with the
  session-5 finding that `integrate_run` is flat for the 2 G runs.

## Mantid

- `PlotAsymmetryByLogValue` is the portable integral-observable reference (mined
  in `time-integral-asymmetry`) — alpha-aware, Integral/Differential, LogValue
  x-axis. Same limitation: it integrates a (possibly per-bin) asymmetry; for a TF
  oscillation that is not the kinetic observable.
- Mantid has **no dedicated pulsed-fast-Mu kinetics workflow**; muonium handling
  is the ALC/repolarisation and radical-hyperfine surfaces (mined in the
  `muonium-radical-hyperfine` candidate), not solution kinetics. The
  concentration→k_Mu→Arrhenius chain is left to the user.

## musrfit

- **Negative result (consistent with prior studies).** musrfit has no integral
  observable and no kinetics pipeline; it fits ordinary fittype-2 asymmetry and
  `mupp` plots *fitted* parameters vs an independent variable — it never shares an
  amplitude to rescue a truncated rate, and never converts a slope to k_Mu /
  Arrhenius. So musrfit offers the per-run model only; the truncation degeneracy
  is the user's problem there too.

## Takeaways for the port

- The **trend maths** (linear λ_Mu([x]); the verbatim `log10 k = log10 A −
  E/2.3RT` Arrhenius) is contracted by the *teaching guide*, and the components
  already exist (`Linear`, `Arrhenius`).
- The **rescue** (share `A_Mu` across the series) is contracted by the *muonium
  literature* and maps onto an *already-implemented* Asymmetry seam (`fit_global`
  globals/locals; equivalently WiMDA link groups). No reference program ships a
  turnkey fast-Mu kinetics workflow — so the port's value is a **discoverable,
  tested entry point + documentation** composing existing parts, not a new
  numerical engine.
- Reference programs are an **oracle only** for the pieces (link-group/global
  sharing semantics, Mantid integral-asymmetry error model already reused) — there
  is nothing to vendor, and no WiMDA fit log exists to grade against for this
  example, so verification is synthetic-truth-led (`verification-plan.md`).

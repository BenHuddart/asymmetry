# Count-domain fit modes — comparison

Scope: the count-domain fitting modes WiMDA exposes and how each maps onto the
Asymmetry seam. The general simultaneous-fit architecture (one physics function
across many domains, per-domain N₀/bg/phase) was already compared in
[`multi-group-time-domain-fitting/comparison.md`](../multi-group-time-domain-fitting/comparison.md);
that is not repeated here. This document is the per-mode count-equation record
and the divergence ledger.

## The one count equation

Every WiMDA count-fit mode is the same physical count model with different
parameter ties. With A(t) = `0.01·MusrFun(t)` the dimensionless physics
polarization and λ_μ = 1/τ_μ:

> **N(t) = N₀·(1 + s·A(t)) + bg·e^(t·λ_μ)** on lifetime-corrected counts,
> equivalently **N_raw(t) = N₀·e^(−t·λ_μ)·(1 + s·A(t)) + bg** on raw counts,

where *s* is a per-histogram sign/scale. WiMDA fits the lifetime-corrected
left-hand form (`Analyse.pas` scales the data by e^(t·λ_μ)/nbin and propagates
Poisson errors by the same factor, so the background term carries e^(t·λ_μ)).
The two forms differ only by the deterministic factor e^(−t·λ_μ); the raw form
is the one on which Poisson statistics are exact.

## Mode-by-mode (WiMDA `Fitgrp`, `AsymFitFunction.pas:248–273`)

| Mode | WiMDA count model (single-pulse) | Free count params | Asymmetry mapping |
|---|---|---|---|
| `fgAll` | N₀_g·(1 + A·af_g) + e^(t·λ_μ)·bg_g, per group *g* | per-group N₀, bg, amplitude af, phase | **Existing.** `build_grouped_count_model` + `fit_grouped_time_domain`. No change. |
| `fgForward` / `fgSelected` | N₀·(1 + A) + e^(t·λ_μ)·bg | N₀, bg | **One-group degenerate case** of the same builder, amplitude fixed +1. New thin entry point. |
| `fgBackward` | N₀·(1 − A) + e^(t·λ_μ)·bg | N₀, bg | Same, amplitude fixed −1. |
| `fgFB` | F: N₀·√α·(1 + A) + e^(t·λ_μ)·bg_F; B: N₀·(1/√α)·(1 − A) + e^(t·λ_μ)·bg_B | **α**, N₀, bg_F, bg_B | **Two-domain fit** with a √α tie across the shared N₀. One new ~12-line model wrapper. |
| `fgFBAsym` | 100·A(t) — fit the *derived asymmetry* | — | **Already shipped** as Asymmetry's F-B asymmetry fit. Out of scope here. |

`af_g` (`fgAll`) is `Parameters[GROUP_base + 4(g−1) + 2]`; the per-group
relative phase is `…+3`. In `fgFB`, `ralp := sqrt(abs(Parameters[1]))` and the
two backgrounds are `Parameters[GROUP_base+1]` (forward) and
`Parameters[GROUP_base+5]` (backward) — i.e. group-1's and group-2's bg slots.
There is **one** shared N₀ (`Parameters[GROUP_base]`), modulated by √α one way
and 1/√α the other.

### Why α is √α, not α

WiMDA splits the balance symmetrically: N₀_F = N₀·√α, N₀_B = N₀/√α, so
N₀_F/N₀_B = α and √(N₀_F·N₀_B) = N₀. Fitting two independent normalizations
N₀_F, N₀_B is *mathematically equivalent* (same χ², same minimum) and would
need no new model — but α and its uncertainty, and especially its correlation
with the amplitude (strong in TF runs because counts can trade between "more α"
and "more A"), only fall out directly when α is an explicit parameter. We
therefore keep WiMDA's √α parameterization. See the reuse audit.

## Window, t₀ and baseline (Phase 2)

| Feature | WiMDA | Asymmetry plan | Divergence |
|---|---|---|---|
| Interior exclude range | `SecondRange` toggle → `TimeFrom2/TimeTo2`; excluded bins dropped from the fit vector (`Analyse.pas:6878`) | Drop bins in [t_ex0, t_ex1] when building the count domain (extends the existing t_min/t_max mask) | None (functional parity). Also serves laser/RF artefact rejection. |
| Fittable t₀ offset | t := X + `Parameters[BG_base+2]`/1000 (ns); shifts the whole model time axis (`musrfunc:1358`) | Optional global `t0` param; model evaluated at t + t₀ | None. Default fixed 0. |
| Baseline drift | Non-relaxing offset component p[3] multiplied by e^(−(λ_b·t)^β_b), `Analyse.pas:1291`; λ_b=`BG_base`, β_b=`BG_base+1` | Optional stretched-exp envelope on the polarization: A_eff(t) = A(t)·e^(−(λ_b·t)^β_b) | **Stated divergence.** WiMDA applies the envelope only to its built-in constant-offset term; Asymmetry has no single privileged offset parameter, so the envelope multiplies the whole polarization. Off by default; lower priority. Interacts with the deferred Set-BG workflow — we do not implement Set-BG, and the drift term is an independent optional knob. |

## Count loss / deadtime in the fit (Phase 3)

WiMDA `ArrayMusrFunc:280–314`, models selectable in the `CountLossModelling`
box:

- **Simple / Linear**: c ← c·(1 − DT0·qq), with the Linear variant
  DT0 ← DT0 + DT1·evfr.
- **Polynomial**: c ← c·(1 − (DT0·qq + C2·10³·qq² + C3·10⁶·qq³ + C4·10⁹·qq⁴)).
- **Power-law**: c ← c·(1 − (evfr·DT0)^C2·e^(−(C4·λ_μ·t)^C3)).

where `qq = c·e^(−t·λ_μ)/(frames·frame_fraction·dataBunch·nhis)/(PlotTres·10³)`
is the instantaneous per-frame, per-bin count rate, and `evfr` is the group's
event fraction. Parameters: DT0=`DT_base`, DT1=`DT_base+1`, C2..C4=`DT_base+2..4`.

| Aspect | WiMDA | Asymmetry plan | Divergence |
|---|---|---|---|
| Loss factor | applied to the model count `c` | same — post-multiply the count model | None. |
| Frame normalization (`qq`) | from `TotalMuonFrames`, `GroupFrameFraction`, `grpnhis`, `PlotTres`, `dataBunch` | read from run metadata; fall back to bin width + total events when absent | Documented fallback when metadata is incomplete (PSI/synthetic runs lack the ISIS frame block). |
| Promote to grouping | `SendToGroup` writes DT0..C4 into `cgrp`, optionally additive (`DTmodelChanges`), zeroing the fit value (`Analyse.pas:6114`) | `promote_deadtime_to_grouping()` writes the fitted DT0 into the grouping deadtime field, additive option, before/after display | Asymmetry's grouping currently stores a single non-paralyzable deadtime τ_dead per histogram (`transform/deadtime.py`), not the DT0..C4 polynomial. We promote DT0 (the dominant term) into τ_dead; polynomial/power-law promotion is recorded as a follow-on. |

The non-paralyzable correction Asymmetry already applies
(`N_corr = N/(1 − N·τ/(Δt·n_frames))`) is the first-order expansion of the same
physics as WiMDA's `(1 − DT0·qq)`; the deadtime-in-fit parameter and the
grouping correction therefore describe the same quantity, which is what makes
the promote action meaningful.

## Double pulse (Phase 3)

WiMDA `ArrayMusrFunc:170–237` / `musrfunc:1369`: with dpsep₂ = dpsep/2 and
weights c₁ = e^(−dpsep₂/τ_μ), c₂ = e^(+dpsep₂/τ_μ) (`DPsepEditChange:6913`),

- A₁ = A(t + dpsep₂), A₂ = A(t − dpsep₂),
- forward f = (1 + A₁·af)·c₁ + [t > dpsep₂] (1 + A₂·af)·c₂, backward analogous
  with (1 − A), then the chosen mode's N₀/bg wrapping is applied to f (and b).

| Aspect | WiMDA | Asymmetry plan | Divergence |
|---|---|---|---|
| dpsep source | user entry (`DPsepEdit`, ns) | instrument metadata if present, else user entry; **optionally fittable** | None functionally; defaulting from metadata is an addition. |
| Pulse weights | c₁,c₂ = e^(∓dpsep₂/τ_μ), fixed once dpsep is set | same, derived from dpsep + τ_μ | None. |
| Second-pulse onset | second pulse contributes only for t > dpsep₂ | same gate | None. |

## Statistics — the central divergence

| | WiMDA | musrfit (fittype 0) | Asymmetry `fgAll` (existing) | Asymmetry new modes (planned) |
|---|---|---|---|---|
| Data scale | lifetime-corrected counts | raw counts | lifetime-corrected counts | **raw counts** |
| Cost | Gaussian σ = √N·e^(t·λ_μ) | Poisson (and Gaussian options) | Gaussian (iminuit `LeastSquares`) | **Poisson (Cash) default, Gaussian √N selectable** |

WiMDA uses Gaussian σ throughout, including the late-time, low-count bins where
the Poisson distribution is visibly skewed and √N underweights the constraint —
this is the documented WiMDA weakness the count-domain modes exist to fix.
Asymmetry's new single-histogram and F+B modes therefore default to a Poisson
(Cash, C = 2Σ(μ − n + n·ln(n/μ))) cost on raw counts, with Gaussian √N as a
selectable alternative for parity and speed. The existing `fgAll` grouped path
is unchanged (lifetime-corrected + Gaussian); a follow-on notes the eventual
unification of `fgAll` onto the raw-count Poisson driver.

Both costs share the identical model builder (raw model = e^(−t·λ_μ) × the
existing lifetime-corrected builder), so this is a statistics choice, not a
model fork — no model-building duplication.

## What is reused vs new (summary; full audit in implementation-options.md)

- **Reused unchanged**: `build_grouped_count_model`, `FitResult` /
  `GroupedTimeDomainFitResult`, `build_grouped_time_domain_groups` (extended
  with a `lifetime_corrected` flag), the grouping/deadtime/bunching/t₀ pipeline,
  `estimate_alpha` (cross-check only), `core/simulate` (extended for
  double-pulse synthesis).
- **New**: `build_fb_count_model` (~12 lines, the √α tie); a `count_domain.py`
  driver that selects Poisson/√N and assembles iminuit costs over the shared
  model; the promote-to-grouping helper; GUI fit-target selector + exclude span.

## References

- *Muon spectroscopy* (muon-spectroscopy textbook) — count equation, detector
  balance α, Poisson detector statistics, pulsed-source double-pulse structure.
- WiMDA source: `FitTyps.pas`, `AsymFitFunction.pas`, `Analyse.pas`
  (F. L. Pratt, ISIS/RAL). Behavioural oracle only; never vendored.
- musrfit `PRunSingleHisto` (A. Suter and B. M. Wojek) — the raw-count
  single-histogram N₀·e^(−t/τ)(1 + P) + B reference and Poisson/Gaussian cost
  selection.

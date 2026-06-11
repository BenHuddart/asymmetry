# MaxEnt completion — WiMDA ↔ Asymmetry comparison

This document extends the implemented-engine study at
[`docs/porting/maxent/comparison.md`](../maxent/comparison.md). That study
chose the MULTIMAX algorithm with WiMDA as the behavioural contract and shipped
the core engine (`core/maxent/engine.py`, PRs #16/#26). **This study does not
re-litigate the engine.** It records the WiMDA behaviours behind the four
remaining gaps — time-domain reconstruction, ISIS pulse shape, ZF/LF + SpecBG,
deadtime/phase calibration — verified against the WiMDA Pascal source, and
states where Asymmetry will diverge.

All WiMDA citations are to `$WIMDA_SRC/src/` (ignoring
`__history/`, `__recovery/`). The Mantid `MaxentTools` numpy port
(`~/Source/mantid/scripts/Muon/MaxentTools/`) is a GPL-3 verification oracle —
read and run only, never copied.

## 0. The shipped engine is a projected-gradient V1, not the full Skilling–Bryan kernel

A load-bearing fact for this port. `core/maxent/engine.py` implements a
**deterministic entropy-regularised projected-gradient** reconstruction
(`run_cycles`, exp-update with normalisation, `engine.py:1013–1165`), not
WiMDA's 3-direction Cholesky Skilling–Bryan inner loop. It keeps the MULTIMAX
*data shape* — joint multi-group, one positive spectrum, per-group phase /
amplitude / background nuisance fit in an outer loop — and a resumable
`MaxEntState`, but the numerical kernel is Asymmetry's own.

Crucially the forward/adjoint maps are **real**: `OPUS` is
`signal(t) = amp·Σ_ν f(ν) cos(2πνt + φ) + bg` (`_project_forward`,
`engine.py:637–657`); the adjoint is the transpose (`_project_adjoint`,
`:685–705`). WiMDA instead carries a **complex** kernel `ZR + i·ZI` built by
`ZFT` and applies the pulse response and lifetime envelope to it in the
frequency domain before an FFT (`Wimdamax.pas:569–606`).

Consequence for this port: every new physics term must fold into the **real
cosine/sine forward–adjoint contract**, not a complex-FFT rewrite. This is
feasible because the engine already exposes a cosine/sine component split
(`_project_forward_components` returns `(C@f, S@f)`, `engine.py:660–682`):
a frequency-dependent complex response `P(ν) = P_c(ν) − i·P_s(ν)` enters as
`amp·[P_c(ν)·cos(2πνt+φ) − P_s(ν)·sin(2πνt+φ)]`, i.e. multiply the C-projection
by `P_c` and the S-projection by `P_s`. Pulse shape, exclusion windows and
deadtime all attach at this seam without touching the inner descent.

## 1. Time-domain reconstruction overlay

**WiMDA — `Timedom` (`Wimdamax.pas:1146–1164`).** Inside the iteration WiMDA
snapshots the forward model `OPUS(F)` per group, then fills, per group `i`,
channel `j`:

`Timedom[i,j] = (model[i,j] + DATT[i,j] − DATUM[i,j]) / (fnorm · bunch)`

where `DATT` is the original normalised data and `DATUM` the working data after
exponential-background and deadtime correction. So `DATT − DATUM` reinstates
everything the reduction subtracted, and the `/(fnorm·bunch)` undoes the
`1e7`-event normalisation and rebinning — the result is a per-group
reconstruction on the **raw rebinned counts scale**, directly over-layable on
the histogram. It is strictly per-group; no combined trace and no stored
residual array (the residual is computed transiently for χ² and overwritten as
the TROPUS gradient seed).

**Asymmetry.** `opus(spectrum, maxent_input, phases=…, amplitudes=…,
backgrounds=…)` (`engine.py:708–731`) already returns exactly the per-group
forward model `predictions[group_id]`. The MaxEnt input normalises each group
to `(signal/baseline) − 1` with `baseline = mean(signal)` (`build_maxent_input`,
`engine.py:588–592`), so the reconstruction lives in that **normalised
asymmetry-like space**, not raw counts.

**Divergence (stated both ways).**
- *WiMDA*: reconstruction returned on the raw rebinned-counts scale, baseline
  reinstated; per-group only.
- *Asymmetry*: reconstruction returned in the engine's internal normalised
  space `(signal/baseline − 1)` — the same space the χ² is computed in, so the
  shown χ² is exactly the engine's by construction. We overlay in that space
  (data and model both), and additionally offer a **combined** view (a
  representative group or the selected-group mean) which WiMDA lacks, plus an
  explicit **residuals strip** `(data − model)/σ` which WiMDA computes but never
  displays. Rationale: the normalised space is what the fit minimises, so the
  overlay is an honest picture of fit quality; reinstating the raw-counts
  baseline would add reduction bookkeeping that the modern representation/plot
  stack does not need. The χ² shown equals the engine χ² (verification target).

## 2. ISIS pulse-shape response

**WiMDA — `START` (`Wimdamax.pas:266–319`).** The instrument response that
multiplies the forward-model kernel in the frequency domain. With angular
frequency per bin `ω = 2π·fperchan·(i−1)`, proton-pulse half-width `w`
(`PwidEdit`, ns→µs), double-pulse separation `s` (`PsepEdit`), pion lifetime
`τ_π = 0.026 µs`, muon lifetime `τ_µ`:

- Parabolic proton-pulse FT (`GW[1]=1` at DC):
  `G(ω) = (3/w)·[ sin(ωw)/((wω)²·ω) − cos(ωw)/(w·ω²) ]`
- Pion low-pass: `A(ω) = G(ω) / (1 + (ω·τ_π)²)`
- Complex single/double-pulse response (`PULSE2T = 0` single, `= s` double):
  - `CONVOL_R(ω) = A(ω)·[ cos(ωs/2) − tanh(s/2τ_µ)·sin(ωs/2)·ω·τ_π ]`
  - `CONVOL_I(ω) = −A(ω)·[ tanh(s/2τ_µ)·sin(ωs/2) + cos(ωs/2)·ω·τ_π ]`

Single-pulse limit (`s=0`): `CONVOL_R = A(ω)`, `CONVOL_I = −A(ω)·ω·τ_π`. The
`tanh(s/2τ_µ)` weight accounts for muon-population depletion between the two
proton pulses — **genuine physics**, not a hack. Applied symmetrically in
`OPUS` and `TROPUS` (`:598–599, 630`); the muon-decay envelope `E(t)` is applied
after the FFT, separately.

`NPULSE = 0` (ignore) → kernel `(1, 0)`. Mantid's `start.py:17–40` is the same
math with widths already in µs and constants truncated (`τ_µ=2.19704`).

**Asymmetry.** No pulse-shape term today: amplitudes above ~5 MHz are distorted
on pulsed data. The real cosine/sine kernel folds the complex response as in §0:
`P_c(ν) = CONVOL_R(2πν)`, `P_s(ν) = CONVOL_I(2πν)` (note Asymmetry frequencies
are in MHz; `ω = 2πν` with `ν` in MHz and `t` in µs keeps `ωt` dimensionless).

**Divergence.**
- *WiMDA*: complex kernel `ZR+iZI`, pulse response a frequency-domain multiply
  before FFT; widths entered in ns; in the forward model (OPUS).
- *Asymmetry*: pulse response folds into the real cosine/sine projection
  `amp·[P_c·cos − P_s·sin]` in `_project_forward`/`_project_adjoint`; widths
  entered in µs (or ns with a labelled field); CODATA constants
  (`τ_µ = 2.1969811 µs`, `τ_π = 0.026 µs`). Same physics, V1-compatible
  placement (forward model, never a post-hoc spectrum correction). The `E(t)`
  lifetime envelope is already applied upstream in `build_group_signal_dataset`
  (`apply_lifetime_correction=True`), so we do **not** re-apply it in the kernel.

## 3. ZF/LF two-group mode + SpecBG

**WiMDA.** In zero/longitudinal field the two groups (F, B) measure the same
relaxation with opposite phase and α-fixed relative efficiency. Phases are
pinned (F=0°, B=180°) and the fit uses `MODAMP` (amplitudes only, not `MODAB`).
Every fitted per-group scalar (exp-BG `D`, amplitude `Amp`, BG-change `c`) is
α-tied by the identical idiom (`:404–408, 736–740, 797–801, 904–908`):
`x[2] = (x[1]+x[2])/(1+α); x[1] = α·x[2]`, i.e. sum the two independent fits and
redistribute in ratio α:1. The tie is applied **after** the per-group
least-squares, not as a constraint inside it.

**SpecBG (`SpecBG.pas`, applied in `Plot.pas:2378–2400`).** Display-only
zero-frequency lineshape subtraction for the field-distribution view: a
pseudo-Voigt centred at zero, anchored to the spectrum value just below the
window `a0 = spectrum[nmin−1]`:
`Δ(x) = a0·[ (1−lfrac)·exp(−(x/(gwid·1.201))²) + lfrac/(1+(x/lwid)²) ]`,
subtracted from each displayed bin. Widths in display units; the `×1.201`
relates the edit's width to the Gaussian σ (empirical magic number).

**Asymmetry.** No ZF/LF mode; the general engine fits all phases/amps freely.
Per Ben's decision, **strict 2-group F/B parity**: ZF/LF mode requires exactly
two selected groups, pins phases 0/180, ties amplitudes via the run's α, and
offers SpecBG as a display-only subtraction on the spectrum.

**Divergence.**
- *WiMDA*: α-tie applied to F=group1/B=group2 by index after the LS; SpecBG
  anchored to one bin below the window; `×1.201` magic constant carried verbatim.
- *Asymmetry*: ZF/LF is an explicit mode that constrains the group table to two
  forward/backward groups and reads α from the run grouping; the α-tie is
  applied inside `_fit_group_nuisance` when the mode is active (amplitudes tied,
  phases held at 0/180 with `fit_phases` forced off). SpecBG is a display-time
  transform on the spectrum dataset, not on the engine spectrum; we carry the
  `×1.201` constant and document it as empirical, and anchor to the lowest
  in-window bin (Asymmetry windows from f_min, no `nmin−1` outside-bin) — stated
  as a deliberate, documented difference.

## 4. Deadtime fitting inside MaxEnt (DEADFIT)

**WiMDA — `DEADFIT` (`Wimdamax.pas:867–937`).** An outer-loop nuisance fit (run
once per cycle after the entropy solve, with the kernel rebuilt by `ZFT`). Per
group it accumulates five weighted sums over channels and solves a 2×2 linear
system for (exp-BG scale, deadtime τ) jointly: the residual `data − model` is
explained by an exponential-background term `E(t)` plus a deadtime term `∝ DAT²`
(the first-order non-paralysable distortion, lost counts ∝ rate²):

`τ = (Bx·Cx − Ax·Ex)/(Ax·Dx − Cx²)`, then physical
`taud = τ·RES·HISTS·FRAMES·fnorm`.

The fitted deadtime then **reshapes the working data** for the next cycle:
`DATUM = DATT + DATT²·τ − D·E`. So deadtime is an outer-loop parameter that
changes the data the entropy fit sees, not a parameter inside the descent. Off →
plain `MODBAK` (exp-BG only). Results logged per cycle; editable afterward in
`MaxEdit`; round-trips through the `RES·HISTS·FRAMES·fnorm` unit conversion that
mirrors `INPUT`.

**Asymmetry.** Deadtime is currently **pre-correction** only
(`prepare_histograms_with_deadtime` before grouping, `engine.py:546–551`); there
is no in-loop deadtime fit. The outer-loop nuisance fit already exists
(`_fit_group_nuisance`, `engine.py:909–994`) for amplitude/background/phase, so
DEADFIT slots in as one more nuisance term there.

**Divergence.**
- *WiMDA*: deadtime fitted jointly with exp-BG via a 5-sum 2×2 solve on the raw
  normalised counts; physical units via `RES·HISTS·FRAMES·fnorm`; auto-reshapes
  the working data each cycle; promotion to grouping is implicit (the same
  `taud` array drives both).
- *Asymmetry*: deadtime fitted as an added nuisance in `_fit_group_nuisance`
  against the engine's normalised signal; reported per group in physical µs (we
  reconstruct the count scale from the prepared histograms / frames metadata,
  documented); **suggest-only promotion** (Ben's decision) — the fitted value is
  surfaced and the user explicitly applies it to the run grouping with a
  provenance label, never auto-written. Because Asymmetry works in normalised
  asymmetry space, the `∝ DAT²` term is reformulated against the pre-normalised
  group counts threaded through the input; the verification target (recover a
  known injected deadtime on a thinned run) anchors the unit round-trip.

## 5. Exclusion time window (σ-inflation)

**WiMDA — `readcontrol:112–116` + `INPUT`.** A user time window `[ex1, ex2]` µs
maps to channels and those channels get `σ = 1e15` (excluded by weight, **not**
dropped), decrementing the live count; the FFT length is preserved. The base
error is `σ = sqrt(N + 2)·fnorm` (the `+2` a small-count regulariser). Late-time
Gaussian apodisation is also expressed as σ-inflation, off by default.

**Asymmetry.** `t_min_us`/`t_max_us` already trim the head/tail by masking
(`build_maxent_input`, `engine.py:582–585`). There is no *interior* exclusion
window. The engine's σ comes from the grouped error model, not `sqrt(N+2)`.

**Divergence.**
- *WiMDA*: exclusion = σ→1e15 over an interior window, points retained, FFT
  length sacred; `sqrt(N+2)` error floor.
- *Asymmetry*: add an interior exclusion window `[ex_min, ex_max]` implemented
  as **σ-inflation** on the masked-in points (mirroring the engine's input
  model — multiply σ by a large factor rather than dropping the rows), so the
  time grid and any future FFT length stay intact and the existing mask
  semantics are preserved. We keep Asymmetry's grouped Poisson error model
  (already shipped, exact `(1−A²)` form per the asymmetry-error study), not
  `sqrt(N+2)` — stated as a deliberate modern-correctness divergence. Head/tail
  trim (`t_min`/`t_max`) stays as masking; only the interior window inflates σ.

## 6. Field-axis / units display

**WiMDA — `MaxControl.pas:157–314`.** Three x-axis modes: frequency (MHz), field
(Gauss), time. Field↔frequency via `gmu2 = 0.01355342 MHz/G`
(`freq = field·gmu2`). Resolution `fres = 1/(2·tres·bunch·nptsME)` MHz,
`bres = fres/gmu2` G. A ±window centres the display on the applied field
(`UseWindow`). Tesla is not offered for the MaxEnt spectrum axis (Gauss + MHz
only); Tesla appears only as a fit-display option elsewhere.

**Asymmetry.** Spectrum is MHz only; `_field_to_frequency_mhz`
(`engine.py:118–119`) already does `field_gauss · 135.538817 · 1e-4`. No shared
units helper exists.

**Divergence.**
- *WiMDA*: Gauss + MHz axes, `gmu2 = 0.01355342`.
- *Asymmetry*: add **Gauss and Tesla** display axes alongside MHz via a new
  `core/fourier/units.py` helper built on the existing CODATA constants
  (`MUON_GYROMAGNETIC_RATIO_MHZ_PER_T = 135.538817`, `GAUSS_TO_TESLA = 1e-4`).
  We add Tesla (WiMDA omits it for this axis) because modern high-field µSR is
  reported in Tesla — a superset, not a conflict. The helper is shared with the
  `frequency-domain-finishers` project (its API is recorded in
  `implementation-options.md` so that project reuses it unchanged).

## 7. Editable phase/deadtime tables + phase exchange

**WiMDA — `MaxEdit.pas`, `PhaseTableUnit.pas`.** A per-group text table edits
either phases (degrees) or deadtimes (`taud` units) in place. Phase exchange is
a scratch buffer: `GetFromMaxent` pulls `phi[]` into the table, `SendToMaxent`
pushes table phases into `phi[]` and sets `phasesfitted := true` (which makes
MaxEnt hold phases fixed). Matching is **by group index only**; there is **no
provenance** (no "from fit"/"from MaxEnt" tag, no timestamp).

**Asymmetry.** Per-group phases are editable in the panel group table (degrees)
and flow through `MaxEntConfig.group_phase_degrees`. Grouped time-domain fits
store per-group `relative_phase` (**radians**, `±π`) in each group's
`FitResult.parameters`; the read path `group_specs_from_grouped_fit`
(`core/simulate.py:833–861`) already extracts them, and `fit_panel`
caches a per-run `grouped_simulate_seed` carrying per-group `relative_phase`.

**Divergence.**
- *WiMDA*: phase exchange by index, no provenance, single global `phasesfitted`
  flag; deadtime/phase share one editor via a mode flag.
- *Asymmetry*: a dedicated **tables tab** in the MaxEnt panel surfacing the
  per-group phase / amplitude / deadtime (the diagnostics payload already
  carries the per-cycle dicts). Paired **"Use fitted phases"** (seed MaxEnt from
  the grouped fit, `rad2deg`) / **"Send phases to fit"** (write back, `deg2rad`)
  actions, each stamped with a **provenance label** (which fit, when) — an
  explicit improvement over WiMDA's untagged buffer. Exchange matches on
  **group id**, not row index (Asymmetry groups carry stable ids), removing
  WiMDA's F/B-mapping footgun. Unit conversion at every boundary
  (radians↔degrees) is the main correctness trap, called out in the test plan.

## 8. Spectrum / log export

**WiMDA.** `.max` spectrum file with a full parameter header auto-saved every
Converge cycle; `.mlog` rich text log. Auto-save-every-cycle is a side effect
the maxent study already flagged "do not replicate; save on demand".

**Asymmetry.** Recipe-in-project only; no text export.

**Divergence.** Asymmetry adds an **on-demand** spectrum text export (two-column
frequency/field + density, with a parameter header) and a run log (per-cycle
χ²/entropy/TEST + final phases/amps/deadtimes), in a modern CSV-like format —
never WiMDA's binary `.max`, never auto-save-every-cycle.

## 9. Out of scope (recorded with rationale)

- **Spectral deconvolution (`Sconv`)** — WiMDA's `1/Sconv` adjoint grows without
  bound at late times (`Wimdamax.pas:329–347`); the maxent study flagged it a
  numerical hazard needing a regularised adjoint. **Deferred.**
- **Looseness / phase-acceleration knobs** — `PhaseAccelFactor` blends old/new
  phases (`:668`); the MOVE auto-tighten loop multiplies all σ by 0.99 when
  bisection stalls (`:1064–1082`). Both are numerical-era convergence cruft on
  WiMDA's Skilling–Bryan kernel. Asymmetry's projected-gradient V1 has its own
  χ²-plateau / divergence guard (already shipped) and does not use a MOVE
  bisection, so these knobs have **no V1 analogue**. **Verdict (recorded now,
  not deferred to testing): out** — they would be dead controls. If Phase 3
  testing surfaces a genuine convergence pathology, the fix belongs in the
  engine's existing guard, not a resurrected looseness knob. (The brief allowed
  "decide with evidence"; the evidence is that the knobs target a kernel
  Asymmetry does not run.)
- **Spectral moments** (B_pk/B_ave/B_rms/skew) — the `spectral-moments` Wave B
  project; it consumes this project's spectrum. Left alone here.
- **Muonium-correlation display** — niche; not in this brief.

## Implementation notes (filled in during the build)

**Pulse-shape placement (Phase 2).** Implemented exactly as §0/§2 anticipated:
the complex response folds into the real cosine/sine kernel as a per-frequency
amplitude `R(ν)=√(P_c²+P_s²)` and phase shift `δ(ν)=atan2(P_s,P_c)`, so every
kernel site (`_project_forward`, `_project_forward_components`,
`_project_adjoint`, the inline gradient matrix) just adds `−δ(ν)` to the angle
and scales columns by `R(ν)`. This preserves the OPUS/TROPUS adjoint exactly
(verified by `test_opus_tropus_are_adjoint_with_pulse`) and adds only O(n_freq)
work. `pulse_amplitude_phase` returns `(R, δ)` — **not** `(P_c, P_s)`; a verifier
must use `R` as the magnitude directly, not `hypot(R, δ)`. Single pulse is the
`separation→0` limit of the double-pulse formula (asserted).

**Pulse defaults from metadata — research finding.** The NeXus loader detects
pulsed vs continuous definitions (`pulsedTD`, `nexus.py:99,110`) but does **not**
record the proton-pulse half-width or double-pulse separation per run. So the
pulse widths default from constants (≈50 ns half-width, 0.324 µs separation,
`MaxEntConfig.pulse_half_width_us`/`pulse_separation_us`). Capturing them from
the loader is a small recorded follow-on for the data-loading family.

**Field-axis units — reused, not duplicated.** The frequency `PlotPanel`
already had an MHz↔Gauss display toggle. Rather than add a second selector in the
MaxEnt panel, the existing toggle gained a **Tesla** option and all its
conversions now route through the new `core/fourier/units.py` helper (single
source). So `MaxEntConfig` carries **no** `field_axis_unit` field — the display
unit is panel state, persisted by the plot panel, not the recipe. (The plan had
suggested a config field; this is the recorded divergence — it would have been
dead config given the panel already owns the unit globally.)

**Exclusion window.** σ-inflation factor `1e8` (not WiMDA's `1e15`) — large
enough to de-weight to ~1e-16 while keeping σ² clear of float overflow. Applied
to the interior window only; head/tail trim stays as masking. Grid length
preserved (asserted).

**ZF/LF mode (Phase 3).** Implemented as a `MaxEntConfig.mode` ("general" /
"zf_lf"); `build_maxent_input` requires exactly two selected groups in zf_lf,
pins their phases to 0/180, and reads α from the run grouping. `_fit_group_nuisance`
skips phase fitting and applies the α-tie `x[B]=(x[F]+x[B])/(1+α); x[F]=α·x[B]`
to amplitudes and backgrounds after the per-group least-squares (matching
WiMDA's redistribute-after-fit order). Verified on a synthetic Kubo–Toyabe F/B
run: phases stay 0/180, the amplitude ratio equals α exactly, the spectrum is
broad and centred near zero. `mode` is in `_state_signature`.

**Deadtime fit — divergence from WiMDA's DEADFIT (recorded).** WiMDA fits
deadtime *inside* the MaxEnt loop via a normalised-space 2×2 solve on the DAT²
term. Asymmetry instead reuses its existing, tested count-domain
`calibrate_deadtime_from_histograms` (the WiMDA `countfit` model on the raw
early-time decay) as the "Fit deadtime" action, surfaces per-detector deadtime,
and promotes it **suggest-only** to `grouping["dead_time_us"]` on an explicit
"Apply to grouping" click (Ben's decision). Rationale: parity of functionality
(the user gets a fitted deadtime they can apply) with more robust, non-redundant
numerics, avoiding a hazardous in-loop normalised-space solve. Both behaviours
stated; this is the deliberate divergence.

**SpecBG (Phase 3).** `core/maxent/specbg.py` implements the zero-centred
pseudo-Voigt subtraction with the empirical `×1.201` Gaussian-width factor
carried verbatim; it anchors to the bin nearest zero (Asymmetry windows from
f_min, so there is no `nmin−1` outside-window bin — documented difference).
Display-only: applied by `apply_maxent_specbg` to the spectrum dataset in
`FrequencyMaxEnt.compute` and the live worker path, never to the engine spectrum.

**Phase exchange.** "Use fitted phases" reads the grouped-fit per-group
`relative_phase` via the existing `fit_panel.grouped_simulate_seed_for_run`
bridge (rad→deg); "Send phases to fit" writes MaxEnt phases back through
`fit_panel.update_grouped_phase_seed` (deg→rad). Matched by **group id** (not row
index — removes WiMDA's F/B-mapping footgun), with a provenance label (direction
+ timestamp). This is the WiMDA slice of the `phase-auto-calibration` candidate —
note it as absorbed in that candidate's entry.

**Export.** `core/maxent/export.py` writes a modern text spectrum (freq MHz /
field G / density + parameter header) and a run log (per-cycle χ²/entropy/test +
final phases/amps/backgrounds), on demand only — never WiMDA's binary `.max` or
its auto-save-every-cycle side effect.

**Post-implementation review fixes (high-effort review).** A recall-biased
multi-angle review caught and fixed: (1) the phase-exchange handlers targeted
`self._fit_panel` but the grouped-fit seed methods live on
`self._multi_group_fit_window` (both buttons were inert); (2) the deadtime fit
read `good_frames` from `run.metadata` (it lives in `grouping`) and passed the
absolute `first_good_bin` as the t0-relative `t_good_offset` — both made the fit
wrong; (3) the ZF/LF α-tie now respects the fit flags (a disabled amplitude/
background fit stays frozen, matching WiMDA's tie-inside-MODAMP/MODBAK) and
orders the F/B pair by the run's `forward_group`/`backward_group` designation,
not the sorted group id; (4) ZF/LF now raises if a group is emptied by the
time/exclusion window rather than silently degrading to an untied single-group
fit; (5) SpecBG is a no-op when the window does not reach zero frequency (an LF
window centred on the Larmor line), since it subtracts a zero-centred model; (6)
the pulse `GW` transform uses a Taylor branch for small `x` to avoid catastrophic
cancellation; (7) the reconstruction overlay no longer pollutes the workspace's
time-view fallback. The `show_reconstruction` restore default was corrected to
off.

**Completion pass (verification gaps, combined view, deferred efficiency).** A
follow-up pass closed the remaining brief items and the deferred review notes:

- **Verification gaps now covered by tests.** Injected-deadtime recovery
  (`test_fit_deadtime_recovers_known_injected_value`), full project round-trip of
  the MaxEnt recipe + `TIME_MAXENT_RECON` representation
  (`test_maxent_recipe_and_reconstruction_survive_project_round_trip` — additive
  schema, no migration), and a render-only real-corpus smoke
  (`test_maxent_corpus_smoke.py`, skipif-guarded on the local WiMDA corpus).
- **Combined reconstruction view (brief B).** The overlay now offers a *combined*
  layout (all selected groups' data+model on one colour-coded axis above a shared
  residuals strip) alongside the per-group stack, switched by a panel toggle. The
  brief asked for both per-group and combined; only per-group had shipped. Total
  χ² is identical between layouts.
- **Efficiency (deferred review).** The reconstruction overlay reuses the
  worker's prepared `MaxEntInput` (threaded on `MaxEntResult`) instead of
  rebuilding it on the GUI thread; the per-frequency pulse amplitude `R(ν)` is
  folded out of the inner-loop kernel block (forward: `M @ (R⊙f)`, adjoint:
  `R ⊙ (Mᵀ@v)`), preserving the OPUS/TROPUS adjoint exactly (pinned by an
  explicit dense-kernel equality test). SpecBG is unified to one application
  point (`MaxEntResult.as_dataset(config)`).

**Recorded follow-ons (left as notes, not implemented):**

- **Reconstruction view as a first-class view band.** The `"reconstruction"`
  plot-workspace token is coordinated across a few mainwindow render-dispatch
  sites. `PlotWorkspacePanel` already centralises its *classification*
  (`_VIEW_TOKENS` / `_FREQUENCY_VIEWS` / `_PRIMARY_TIME_VIEWS` set membership), so
  a future token cannot desync the domain/fallback logic. The remaining
  mainwindow special-cases are localised render routing; a deeper "view band"
  abstraction would only risk the recently bug-fixed view-sync wiring, so it is
  left as a note.
- **Loader pulse metadata.** `nexus.py` detects pulsed definitions but still does
  not capture the proton-pulse half-width / double-pulse separation per run, so
  the pulse widths default from constants (`MaxEntConfig.pulse_half_width_us` /
  `pulse_separation_us`). At ISIS these are accelerator-tune constants rather than
  per-run NeXus fields (Mantid's own `start.py` hardcodes 0.05 µs / 0.324 µs), so
  capturing them is a speculative data-loading follow-on for the loader family —
  left as a recorded note.

## 10. Verification oracle status

Mantid the **framework** is not importable (`import mantid` →
`ModuleNotFoundError`), but the pure-numpy `MaxentTools` kernel modules were
probed standalone (follow-on resolved):

- **`start.py` imports cleanly** standalone (exposes `START`, `np`, `math`; no
  Mantid-framework imports). Its `PULSESHAPE_convol = convolr + i·convoli` is
  exactly our pulse kernel with `convolr = P_cos`, `convoli = −P_sin`. With
  `TZERO_fine = −τ_π` its internal `exp(i(TZERO+τ_π)ω)` time shift cancels,
  giving a clean comparison. `tests/test_maxent_pulse_oracle.py` (skipif-guarded
  on a local `~/Source/mantid` checkout, so it skips in CI) confirms the match:
  **single pulse to machine precision** (no τ_µ dependence — the tanh
  interference weight vanishes), **double pulse to ~1e-3** (Mantid truncates
  τ_µ = 2.19704 vs our CODATA 2.1969811, which only enters the tanh). This is the
  documented constants-differ tolerance the study predicted. No code copied.
- **`deadfit.py` does not import** standalone (`from ...Muon import …` →
  `ModuleNotFoundError: No module named 'Muon'`), so the DEADFIT 2×2 solve has
  **no usable kernel oracle**. Asymmetry's deadtime fit is verified synthetically
  instead (`test_fit_deadtime_recovers_known_injected_value`: recover a known
  injected non-paralysable τ on a thinned decay), which is the primary plan.

Everything else stays **synthetic-first** (see `verification-plan.md`).

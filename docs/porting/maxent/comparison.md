# Maximum Entropy Comparison

## Scope

This comparison covers every "maximum entropy" implementation found in the
three reference programs and decides, for each behavior, whether Asymmetry
must choose a single approach or can offer the user a choice. The factors
balanced throughout are:

- simplicity of the user-facing surface
- whether an additional approach offers truly new functionality
- whether two approaches are similar enough that offering both would confuse
  the user

All file references are to the local reference checkouts:
`~/Source/WiMDA`, `~/Source/mantid`, `~/Source/musrfit`.

## Summary

Four implementations exist across the reference programs; they group into
**three genuinely different methods**:

1. **MULTIMAX-lineage muon MaxEnt** (WiMDA `Wimdamax.pas`; Mantid
   `MuonMaxent` + `scripts/Muon/MaxentTools/`). Joint reconstruction of one
   positive field/frequency spectrum from the raw counts of all detector
   groups simultaneously, with an outer loop that refits per-group phases,
   amplitudes, exponential backgrounds, and (optionally) deadtimes around a
   Skilling–Bryan inner loop. Both descend from the same
   Southampton/Birmingham/St Andrews MULTIMAX FORTRAN program and are
   near-identical line by line. **This is the method μSR users mean by
   "MaxEnt" and is the porting target.**
2. **Generic Skilling–Bryan MaxEnt** (Mantid `MaxEnt-v1`, C++). Per-spectrum
   (or joint-with-fixed-adjustments) entropy-regularized inversion of
   already-preprocessed data. Supports complex images and a positive-negative
   entropy that can reconstruct signed spectra. No muon physics inside.
3. **Burg all-poles MEM** (WiMDA `MaxEnt.pas`). A parametric autoregressive
   spectral estimator from Numerical Recipes, exposed as an alternative
   "Transform Mode" in WiMDA's FFT dialog. It shares nothing with methods 1–2
   except the words "maximum entropy".

musrfit has **no MaxEnt of any kind** (verified exhaustively; see below).

## Implementation Inventory

| Implementation | Language | Entry point (UI) | Algorithm core | Status |
| --- | --- | --- | --- | --- |
| WiMDA Pratt/MULTIMAX | Delphi Pascal | `MaxEnt` main-menu item → `TMaxentControl` dialog (`WiMDA_Main.pas:1928-1937`, `MaxControl.pas`) | `Wimdamax.pas:1318-1402` (`domaxent`), inner loop `MAXENT` at `Wimdamax.pas:1109-1316` | full feature, no automated tests |
| Mantid `MuonMaxent` | Python (numpy) | Frequency Domain Analysis → Transform tab → MaxEnt (`maxent_presenter.py:148`) | `Framework/PythonInterface/plugins/algorithms/MuonMaxent.py:263` → `scripts/Muon/MaxentTools/multimaxalpha.py:20` (`MULTIMAX`) | full feature, unit + system + doc tests |
| Mantid generic `MaxEnt` | C++ | scripting only (no muon GUI since v3.12; `docs/source/release/v3.12.0/muon.rst:23`) | `Framework/Algorithms/src/MaxEnt.cpp:318` (`exec`), `MaxEnt/MaxentCalculator.cpp:155` (`iterate`) | full feature, extensive unit + system tests |
| WiMDA Burg MEM | Delphi Pascal | FFT dialog "Transform Mode" radio (`FFTPar.pas:24`, `FFTPar.dfm:222`) | `MaxEnt.pas:34-107` (`memcof`/`evlmem`), driver `MaxEntropy` at `MaxEnt.pas` called from `Plot.pas:1915` | working, no tests |
| musrfit | — | — | — | **absent**; roadmap item only (`doc/musrfit.dox:68`) |

## The Shared MULTIMAX Lineage

WiMDA's `Wimdamax.pas` header (`Wimdamax.pas:1-5`) reads: *"Pascal version of
Maxent for MUSR for linking with WIMDA. Based on the
Southampton/Birmingham/St Andrews Fortran Program Multimax. FLP 21/8/98."*
Mantid's `MaxentTools` modules carry comments such as *"translated from
MAXENT.for"* (`maxent.py:14`) and preserve the FORTRAN COMMON-block names as
variable prefixes (`SPACE_`, `FAC_`, `DETECT_`, `MAXPAGE_`, `HERITAGE_`,
`PULSESHAPE_`, …). The two are ports of the same ancestor.

### Routine-level correspondence

| Role | WiMDA (`Wimdamax.pas`) | Mantid (`scripts/Muon/MaxentTools/`) |
| --- | --- | --- |
| Driver / outer loop | `domaxent` (`:1318-1402`) | `MULTIMAX` (`multimaxalpha.py:20`) |
| Read counts, build σ, deadtime corr., padding | `INPUT` (`:174-260`) | `INPUT` (`input.py:21`) |
| Decay, pulse-shape response, t0 phase ramp | `START` (`:266-378`) | `START` (`start.py:11`) |
| Initial exponential fit + amplitude seed | `BACK` (`:380-457`) | `BACK` (`back.py:19`) |
| Inner Skilling–Bryan loop | `MAXENT` (`:1109-1316`) | `MAXENT` (`maxent.py:27`) |
| Forward map spectrum→counts | `OPUS` (`:590-606`) | `OPUS` (`opus.py:10`) |
| Adjoint map | `TROPUS` (`:609-631`) | `TROPUS` (`tropus.py:10`) |
| Model time spectra from current image | `ZFT` (`:569-587`) | `ZFT` (`zft.py:16`) |
| χ² bisection / step control | `MOVE`/`CHINOW`/`CHOSOL`/`DIST` (`:939-1089`) | `move.py`/`chinow.py`/`chosol.py`/`dist.py` |
| Refit phases+amplitudes (free phases) | `MODAB` (`:634-704`) | `MODAB` (`modab.py:15`) |
| Refit amplitudes only (fixed phases) | `MODAMP` (`:707-769`) | `MODAMP` (`modamp.py:14`) |
| Refit exponential background scale | `MODBAK` (`:772-821`) | `MODBAK` (`modbak.py:11`) |
| Joint exponential + deadtime refit | `DEADFIT` (`:867-937`) | `DEADFIT` (`deadfit.py:15`) |
| Constant (time-independent) background refit | `MODCONST` (`:824-864`) | **absent** |
| Time-domain reconstruction output | `Timedom` fill (`:1148-1163`) | `OUTSPEC` (`outspec.py:11`) |

### Shared constants and conventions (evidence of one ancestor)

Both implementations use, identically:

- counts normalized to `1e7` total events (`Wimdamax.pas:194-195`,
  `input.py:59`)
- `σ = sqrt(N + 2) × fnorm` per bin (`Wimdamax.pas:203`, `input.py:61`)
- excluded/padded bins given `σ = 1e15` rather than cropping
  (`Wimdamax.pas:199-201`, `input.py:61`)
- χ² target = number of (padded) points × number of groups
  (`Wimdamax.pas:1130`, `maxent.py:58-59`)
- 3 search directions with entropy metric `diag(f)`
  (`Wimdamax.pas:1166-1232`, `maxent.py:61-123`)
- Cholesky solve with negative-pivot clamp `1e-10`
  (`Wimdamax.pas:974-976`, `chosol.py:58-73`)
- MOVE bisection tolerance `1e-3`, stuck threshold 10000 iterations, then
  multiplicative 1% tightening of all σ and the looseness factor
  (`Wimdamax.pas:1064-1082`, `move.py:38-47`)
- distance/trust-region constant `0.1·Σf/A` (`Wimdamax.pas:1084-1087`,
  `move.py:50-52`; not user-settable in either)
- negativity clamp `f := 1e-3·A` (`Wimdamax.pas:1294`, `maxent.py:163`)
- inner stop: `TEST < 0.02` and `|χ²/χ²_target − 1| < 0.01`, max 10
  inner iterations (`Wimdamax.pas:1312-1314`, `maxent.py:69`)
- deadtime correction `datum += τ·datum²` with σ inflation
  `×(1 + 0.5·corr/datum)` (`Wimdamax.pas:223-230`, `input.py:64-70`)
- ISIS pulse-shape response: parabolic proton pulse FT, pion-lifetime
  Lorentzian `t_π = 0.026 µs`, double-pulse separation 324 ns, t0 fine-offset
  phase ramp (`Wimdamax.pas:266-378` with widths in ns, `start.py:17-40`
  with widths in µs)
- amplitude renormalization to mean 1 across groups each cycle
  (`Wimdamax.pas:671-702`, `modab.py:35-40`)
- muon gyromagnetic ratio `0.01355` MHz/G family of constants
  (`MaxControl.pas:271` uses `0.01355342`; `MuonMaxent.py:347` uses `0.01355`
  and `:423` uses `135.5e-4` — see "Constant discrepancies" below)

**Conclusion: these are one algorithm.** Offering "WiMDA MaxEnt" and "Mantid
MuonMaxent" as separate user choices would present the user with two nearly
identical engines whose differences are accidents of porting history — the
canonical confusing-choice case. Asymmetry should implement this algorithm
once.

### Real differences within the lineage

These are the genuine deltas to decide on (each classified in the
choose-vs-offer analysis below):

| Difference | WiMDA | Mantid MuonMaxent |
| --- | --- | --- |
| Constant (flat in time) background refit | `MODCONST` per cycle, `FitBGs` checkbox, default ON for continuous sources, OFF for pulsed (`WiMDA_Main.pas:1935`) | absent |
| Frequency-window restriction | default map set to `BLANK·1e-50` outside `[nmin, nmax]`; auto-window ±300 G around applied field (`Wimdamax.pas:1117-1122`, `MaxControl.pas:180-212`) | hard clamp of active points `MAXPAGE_n = MaxField·0.01355·2·Npts·res`, min 256 (`MuonMaxent.py:347-353`) — restriction by array length, not default map |
| Spectral deconvolution (Lorentzian/Gaussian lineshape removal inside the transform) | `Sconv` in `OPUS`/`TROPUS` (`Wimdamax.pas:329-357`) | absent |
| Apodisation | Gaussian, implemented as σ(t) inflation, off by default (`Wimdamax.pas:237-249`) | absent |
| Time-range exclusion window | `ex1`/`ex2` µs edits → σ=1e15 (`Wimdamax.pas:112-116`) | absent (only first/last good times) |
| ZF/LF mode | phases pinned 0/180, F/B tied via alpha; 2-group only (`MaxControl.pas:611-614`, `Wimdamax.pas:404-408`) | absent |
| Phase relaxation | `PhaseAccel` factor blending old/new phases (`Wimdamax.pas:668`) | absent (full update each cycle) |
| Smoothed/min errors at long times | `GetSmoothErrors` (`Wimdamax.pas:459-473`) | absent |
| Interactive convergence | Start / +1 / +5 / +25 / Converge buttons; `sconv = 100·Σ|Δf|/Σf` against user limit (`MaxControl.pas:350-420,488-501`) | fixed `OuterIterations × InnerIterations` batch run, no convergence test |
| Outer-loop count semantics | cycles accumulate across button presses (`ngo` persists) | one-shot `OuterIterations` (default 10) |
| Dead-detector handling | `HISTS=0` ⇒ σ=1e15, amp 0 (`Wimdamax.pas:199,425-427`) | explicit removal pre-run + re-insertion with `999` markers (`dead_detector_handler.py:11`, `MuonMaxent.py:437-457`) |
| Initial phases | uniform spread `(i−1)/n·360°` unless previously fitted/edited (`Wimdamax.pas:1330-1333`) | optional `InputPhaseTable` (CalMuonDetectorPhases format) else uniform spread (`MuonMaxent.py:197-243`) |
| Phase/deadtime manual editing | table editor dialogs (`MaxEdit.pas:31-58`) | input/output TableWorkspaces |
| Diagnostics output | rich text log (`MaxOutput.pas`), saved `.mlog`; per-cycle phase/amp/BG/deadtime tables | `PhaseConvergenceTable`, `OutputPhaseTable`, `OutputDeadTimeTable`, `ReconstructedSpectra` workspaces; converged `Factor` returned InOut |
| Spectrum auto-save | `.max` file with full parameter header after each run (`MaxControl.pas:414-419`, `WiMDA_Main.pas:1441-1516`) | output workspace only |
| Moments analysis of spectrum | dedicated window: Bpk, Bave, Brms, skew, run-averaging, export to fit table (`Moments.pas:152-393`) | absent |
| Muonium-correlation display transform | `MuCorrelation` mode (`Plot.pas:2314-2347`) | absent |
| Display-time spectrum BG subtraction | pseudo-Voigt tool (`SpecBG.pas`, `Plot.pas:2378-2400`) | absent |
| Default level `A` | 0.01 (`MaxControl.dfm:502-508`) | 0.1 (`MuonMaxent.py:87`); GUI default 0.1 (`maxent_view.py:96-115`) |
| σ looseness factor | 1.00 (`SigmaFactor`) | 1.04 (`Factor`, returned InOut) |
| Spectrum points default | 4096 from listbox 256…262144 (`MaxControl.pas:625`) | GUI: smallest power of 2 ≥ data length, up to 2^20 (`maxent_presenter.py:118-122`) |
| Default field axis | Field (Gauss), Frequency selectable (`MaxControl.dfm:674-684`) | Gauss output axis; GUI overlays dual field/frequency axis (`frequency_context.py:92-94`) |

### Constant discrepancies (must be pinned during implementation)

| Constant | WiMDA | Mantid MuonMaxent |
| --- | --- | --- |
| Muon lifetime | `2.1969811` µs (`globals.pas:13`) | `Tmuon = 2.19704` µs (`start.py:12-15`) |
| Gyromagnetic ratio | `0.01355342` MHz/G (`Analyse.pas:566` et al.) | `0.01355` (`MuonMaxent.py:347`) and `135.5e-4` (`:423`) |
| Proton pulse half-width default | 50 ns user-editable (`PwidEdit`) | 0.05 µs hardcoded (`start.py:18-21`) |

Asymmetry should use CODATA values (τ_μ = 2.1969811 µs, γ_μ/2π =
0.01355342 MHz/G — i.e. WiMDA's values) and record that golden-data
comparisons against Mantid need tolerance for its truncated constants.

## Mantid Generic `MaxEnt-v1` — A Genuinely Different Method

Files: `Framework/Algorithms/src/MaxEnt.cpp`,
`Framework/Algorithms/src/MaxEnt/` (calculator, entropies, spaces,
transforms); doc `docs/source/algorithms/MaxEnt-v1.rst`.

What it does differently from the MULTIMAX lineage:

- operates on **preprocessed data** (typically asymmetry after
  `RemoveExpDecay`), not raw counts; equal-bin validator only — no muon
  physics, phases, deadtime, or grouping inside the algorithm
- **2 search directions** (`MaxentCalculator.cpp:201`) vs MULTIMAX's 3;
  GSL **SVD** solve (`MaxEnt.cpp:736-750`) vs Cholesky
- **PosNeg entropy** (default; `MaxentEntropyNegativeValues.cpp`):
  `S = Σ sqrt(x²+A²) − x·asinh(x/A)` — reconstructs **signed** images, which
  pure-positive MULTIMAX cannot
- **complex images** and complex input data (`ComplexData`/`ComplexImage`)
- runs to convergence (`MaxIterations` default 20000 with explicit χ² and
  angle criteria) instead of a fixed small cycle count
- joint multi-spectrum reconstruction only with **fixed** complex per-spectrum
  adjustments (`DataLinearAdj`/`DataConstAdj`,
  `MaxentTransformMultiFourier.cpp`) — it never fits phases/amplitudes
- two-sided frequency axis output centered on zero; `ResolutionFactor`
  zero-padding control

Mantid's own muon GUI dropped this algorithm in favor of `MuonMaxent` in
v3.12 (`docs/source/release/v3.12.0/muon.rst:23`), which is strong evidence
about which method practicing μSR users need. Its unique capabilities
(signed/complex images on arbitrary preprocessed data) are real but serve
non-μSR-specific use cases (e.g. general FFT denoising); for grouped muon
asymmetry the PosNeg entropy answers a question Asymmetry's users have not
asked yet.

## WiMDA Burg MEM — Same Name, Different Method

Files: `~/Source/WiMDA/src/MaxEnt.pas` (`memcof`/`evlmem`, Numerical Recipes;
Akaike FPE pole scan over `PolesFrom..PolesTo`, defaults 15–30), invoked as
the `MaxEntMode` "Transform Mode" radio in the FFT dialog (`FFTPar.pas:24`,
`Plot.pas:1908-1927`).

This is autoregressive (all-poles) spectral estimation — a parametric method
with completely different statistical assumptions, no χ² target, no default
level, and no multi-group capability. It shares only the name "maximum
entropy". WiMDA itself keeps it inside the Fourier window, separate from the
`MaxEnt` menu item, and makes the two dialogs mutually exclusive
(`MaxControl.pas:434`, `FFTPar.pas:167-179`) — even WiMDA treats them as
different features.

**Recommendation: exclude from this port.** If ever implemented, it belongs
in the Fourier feature family as an "all-poles (Burg) transform mode" and
must not be labelled "MaxEnt", precisely to avoid the naming collision WiMDA
created. Note the WiMDA repo's own `FEATURE_MAP.json` entry `maxent-spectrum`
points at the Burg code and *misses* `Wimdamax.pas` entirely — a trap for
future agents (see "Corrections To Prior Inventories").

## musrfit — Confirmed Absent

Exhaustive search (maxent / maximum entropy / Burg / all-poles / spectral
estimation over src/, doc/, docs/, ChangeLog, NEWS, JSON maps) found no
implementation. Evidence:

- `doc/musrfit.dox:68` — roadmap: "add an interface to maxent" listed under
  missing features (an *interface to an external program*, not an algorithm)
- `src/tests/skewedGaussianTest/README:18,21` — refers to "analysis software
  like maxent and wkm", i.e. maxent as separate external software
- the only entropy code is `PFTPhaseCorrection`
  (`src/include/PFourier.h:84`, `src/classes/PFourier.cpp:355-417`) — a
  Minuit2-minimized Shannon-entropy **phase correction** of an
  already-computed FFT (exposed as `phase_opt_real`). This is the optimizer
  already documented in the fourier-transform study; it must not be confused
  with MaxEnt spectral estimation.

musrfit's answer to MaxEnt-style questions is apodized FFT plus
time-domain model fitting (e.g. libFitPofB vortex-lattice models). For this
study musrfit contributes no implementation to compare, but its absence is
itself a data point: two of three reference programs ship the MULTIMAX
method, and the third wants an interface to it.

## Feature Matrix (methods 1–2 only)

"MULTIMAX" = the single algorithm evidenced by WiMDA `Wimdamax.pas` and
Mantid `MuonMaxent`; per-cell notes mark where the two ports differ.

| Aspect | MULTIMAX lineage | Mantid generic `MaxEnt-v1` |
| --- | --- | --- |
| Input | raw grouped counts, all groups jointly | preprocessed spectra (asymmetry), per spectrum |
| Image | one positive real spectrum shared by all groups | per-spectrum complex or real, signed (PosNeg) or positive |
| Entropy | Shannon vs flat default `A` | Shannon (positive) or PosNeg |
| χ² target | `npts × ngroups`, approached via `CTARG` staging | `ChiTargetOverN` (default 1.0) direct |
| Search | 3 directions, metric `diag(f)`, Cholesky | 2 directions, metric from entropy module, SVD |
| Iterations | inner ≤10, outer cycles (10 default / interactive in WiMDA) | up to 20000 with convergence test |
| Phases/amplitudes | **fitted per group in outer loop** (MODAB) or amplitudes-only (MODAMP) | fixed adjustments only |
| Background | exponential refit (MODBAK/DEADFIT); WiMDA adds constant BG (MODCONST) | none (preprocessing's job) |
| Deadtime | optional per-group fit (DEADFIT) | none |
| Muon lifetime | inside forward model (`E(t)`) | removed upstream |
| Pulsed-source response | proton pulse + pion lifetime + double pulse convolution | none |
| Field axis | one-sided, Gauss/MHz, window/MaxField restricted | two-sided centered on 0, generic units |
| Diagnostics | per-cycle χ²/entropy/TEST/sconv, phase convergence | `EvolChi`, `EvolAngle` |
| Tests in source repo | Mantid: unit/system/doc tests; WiMDA: none | extensive unit + system tests |

## Choose One vs Offer Choice

### Decisions where Asymmetry must choose a single approach

These are internal algorithm details where exposing alternatives would be
meaningless or confusing:

1. **Core engine: implement the MULTIMAX algorithm once.** WiMDA and Mantid
   MuonMaxent are the same method; the port should follow WiMDA's variant as
   the behavioral contract (consistent with the project's WiMDA-first
   precedent in the fourier-transform study) and use Mantid's
   `MaxentTools` numpy modules as an executable oracle for verification.
   Do **not** expose "WiMDA mode" vs "Mantid mode".
2. **Window mechanism: WiMDA's default-map window** (default level dropped to
   `1e-50` outside the window) rather than Mantid's array-length clamp. It is
   strictly more expressive (window can sit anywhere, auto-centered on the
   applied field) and subsumes Mantid's `MaxField` behavior as the special
   case `[0, MaxField]`.
3. **Physical constants: CODATA/WiMDA values** (`τ_μ = 2.1969811 µs`,
   `γ_μ/2π = 0.01355342 MHz/G`), not Mantid's truncated ones.
4. **Dead-detector handling: WiMDA's σ=1e15 de-weighting** as the core
   mechanism (it falls out of the error model for free); adopt Mantid's
   explicit marker convention only in reporting if needed.
5. **Convergence model: WiMDA's interactive/incremental cycles** (run N
   cycles, inspect, run more, converge-to-tolerance) rather than Mantid's
   fixed one-shot batch. The incremental model subsumes the batch model
   (a "run 10 cycles" call is one button press) and matches Asymmetry's
   interactive GUI character. The core API should expose
   resumable state so the GUI can implement +1/+5/+25/Converge.
6. **Spectrum error bar: WiMDA's `MEerr` formula** (uniform error from
   signal-to-noise and spectral weight, `Plot.pas:2353-2360`) — Mantid
   produces no spectrum errors at all. Flag the formula's statistical
   meaning as an open question rather than a contract.

### Behaviors to offer as user options (within the single engine)

These already exist as options in at least one reference program, are
orthogonal, and serve distinct experimental situations:

| Option | Reference precedent | Default | Notes |
| --- | --- | --- | --- |
| Fit phases vs fix phases | both (`FitPhases` / `FixPhases`) | fit | fixed phases needed for ZF and for trusting an external phase table |
| Initial phases from table vs uniform spread | Mantid (`InputPhaseTable`); WiMDA (manual table editor + persistence of fitted values) | uniform spread | Asymmetry already has per-group phase tables from the Fourier work — reuse that boundary |
| Fit deadtimes | both (`FitDeadtimes` / `FitDeadTime`) | off (WiMDA) — revisit; Mantid defaults on | needs `frames` metadata |
| Fit constant background | WiMDA only (`FitBGs`/MODCONST) | on for continuous, off for pulsed (WiMDA's auto rule) | genuinely new vs Mantid; cheap to include |
| Pulse mode: ignore / single / double | both (WiMDA radio with editable width/separation; Mantid `DoublePulse` bool) | ignore for continuous data, single for ISIS | offer WiMDA's editable pulse width/separation as advanced fields |
| Frequency window: manual min/max or auto ±Δ around applied field | WiMDA (`UseWindow`) | auto window on for TF data | subsumes Mantid `MaxField` |
| Spectrum points | both | next power of 2 ≥ data length (Mantid GUI rule) capped, user-overridable from a power-of-2 list (WiMDA style) | WiMDA's fixed 4096 default is worse than Mantid's data-derived rule |
| Default level `A` | both | WiMDA 0.01 | document Mantid's 0.1 in the tooltip; value interacts with window |
| σ looseness factor | both (`SigmaFactor` 1.00 / `Factor` 1.04) | 1.00 | report the converged/tightened value like Mantid's InOut `Factor` |
| Gaussian apodisation time (as σ inflation) | WiMDA only | off | distinct from Fourier-panel apodisation — document the difference |
| Time-range exclusion | WiMDA only | off | useful for instrument glitches |
| Smooth/min errors at long times | WiMDA only | off | low priority |
| ZF/LF 2-group mode | WiMDA only | off | defer to a follow-on slice; touches alpha handling |
| Spectral deconvolution (Lor/Gau lineshape) | WiMDA only | off | defer; `1/Sconv` growth is a numerical hazard (`Wimdamax.pas:329-347`) |
| Phase relaxation factor | WiMDA only (`PhaseAccel`) | 1.0 (= off) | advanced; aids convergence on pathological data |
| Field vs frequency axis | both | Field (G), consistent with WiMDA and the existing Fourier panel toggle | reuse the Fourier panel's unit machinery |
| Output: time-domain reconstruction overlay | both (WiMDA `Timedom`, Mantid `ReconstructedSpectra`) | on | strongest single diagnostic of fit quality |
| Output: phase-convergence trace | Mantid (`PhaseConvergenceTable`); WiMDA logs per-cycle tables | on (cheap) | |
| Moments analysis (Bpk, Bave, Brms, skew) | WiMDA only (`Moments.pas`) | follow-on slice | distinct feature on top of the spectrum |

### Approaches NOT to offer (confusion outweighs new functionality)

1. **MULTIMAX variant choice (WiMDA vs Mantid flavor)** — same algorithm;
   choose one contract (above).
2. **Burg all-poles MEM** — different method wearing the same name. Excluded
   from this feature. If revisited, it lives in the Fourier family under an
   "all-poles/Burg" label, never "MaxEnt".
3. **Mantid generic `MaxEnt-v1` as a second user-facing engine** — its
   genuinely new capabilities (PosNeg entropy, complex images, preprocessed
   input) do not currently serve Asymmetry's grouped μSR workflow, and a
   second "MaxEnt" button would force users to understand an algorithmic
   distinction the reference muon GUIs themselves abandoned (Mantid's muon
   interface dropped it in v3.12). **However**, keep the core seam
   engine-shaped (an entropy module + transform module boundary similar to
   Mantid's `MaxentEntropy*`/`MaxentTransform*` decomposition) so a
   generic/PosNeg engine can be added later without UI redesign if a use case
   appears (e.g. MaxEnt on already-processed data where raw counts are
   unavailable — see open questions).
4. **musrfit `phase_opt_real`** — already tracked in the fourier-transform
   study as a Fourier feature; keep it out of MaxEnt scope.

## Edge Cases And Quirks To Preserve Or Resolve

Recorded explicitly so the implementing agent does not rediscover them:

- **WiMDA phase-sign quirk**: `BACK` seeds `B := +AMP·sin φ`
  (`Wimdamax.pas:436`) while `MODAMP` uses `B := −AMP·sin φ` (`:746`) and
  `MODAB` infers `PHI := −57.296·atan2(B,A)` (`:666`). Internally consistent
  within MODAB/MODAMP but the BACK seeding flips sign on the first
  fixed-phase cycle. Decide: reproduce or fix; document either way.
- **WiMDA continuation semantics**: "+N" buttons skip
  `readcontrol/READDATA/INPUT/START/BACK` entirely
  (`Wimdamax.pas:1322-1351`), so parameter edits between presses are mostly
  ignored. Asymmetry should define explicit "restart required" semantics
  per parameter instead of silently ignoring edits.
- **MOVE stuck behavior**: both ports silently tighten σ and the looseness
  factor by 1% when bisection fails (`Wimdamax.pas:1064-1082`,
  `move.py:38-47`). Mantid at least returns the changed `Factor`; Asymmetry
  should surface this in diagnostics.
- **Mantid `Npts` validator bug**: declared default 2 is outside its own
  `IntListValidator` of powers of two (`MuonMaxent.py:75-80`) — do not copy.
- **Mantid NaN failure mode**: `MAXENT` raises on NaN in search-direction
  coefficients; it is a tested user-visible error
  (`MuonMaxEntTest.py:109-121`). Plan a friendlier failure.
- **σ=1e15 sentinel family**: thresholds `1e3`/`1e6`/`1e10`/`1e15` are used
  inconsistently across MODBAK/MODCONST/DEADFIT filters
  (`Wimdamax.pas:785,837,883`). A port should use one explicit
  "excluded-bin" mask instead of magic σ thresholds.
- **Static allocation limits**: WiMDA caps 40 groups × 131072 bins ×
  524288 frequency points (`Wimdamax.pas:17-20`); numpy removes the need
  for caps but memory for `ETA` (3 × groups × npts doubles) should be sized
  consciously.
- **WiMDA `.max` auto-save side effect**: the Converge button saves the
  spectrum file every cycle (`MaxControl.pas:414-419,488-501`) — do not
  replicate; save on demand.
- **`MEasym` and the displayed error bar** depend on the `1e7`-event
  normalization; keep normalization an internal detail, not a user-visible
  convention.

## Test Coverage In The Reference Programs

- WiMDA: **no numerical tests** for either MaxEnt. Only a scaffold
  characterization test
  (`tests/porting/test_maxent_spectrum_discovery.py`) that anchors the
  (mislabeled) Burg entry.
- Mantid `MuonMaxent`: unit tests (shape, dead detectors, validation, NaN
  error; `Framework/PythonInterface/test/python/plugins/algorithms/MuonMaxEntTest.py`),
  system test vs reference `MuonMaxEntMUSR00022725.nxs` at tolerance 5e-2 on
  MUSR00022725 (`Testing/SystemTests/tests/framework/MuonMaxEntTest.py`),
  and a doc-test asserting five specific spectrum values
  (`docs/source/algorithms/MuonMaxent-v1.rst:48-64`). The `fixPhasesTest`
  asserts output phases equal input phases when `FixPhases=True`.
- Mantid generic `MaxEnt`: extensive numerical unit tests
  (`Framework/Algorithms/test/MaxEntTest.h` — cosine/sine reconstructions to
  1e-3/1e-4, adjustments, joint mode, resolution factor) and a system test
  on MUSR00022725 at 5e-2.

## Corrections To Prior Inventories

This study found and corrected two errors in
`docs/porting/reference/` (fixed in the same change as this study):

1. `reference/musrfit/inventory.md` claimed musrfit ships "a Cython-style
   entropy computation" giving "limited" MaxEnt support. Wrong on both
   counts: the entropy code is a C++ Minuit2 phase-correction objective, and
   musrfit has zero MaxEnt capability.
2. `reference/wimda/inventory.md` described WiMDA's MaxEnt as "Burg's
   method", following WiMDA's own `FEATURE_MAP.json` `maxent-spectrum` entry.
   That entry points only at the Burg code in `src/MaxEnt.pas`; the
   scientifically important Pratt/MULTIMAX MaxEnt in `src/Wimdamax.pas` was
   missing from WiMDA's maps entirely. The WiMDA-side maps
   (`FEATURE_MAP.json`, `SYMBOL_MAP.json`, `python_port/wimda_port/maxent_spectrum.py`)
   still carry this mislabel — an implementing agent must not navigate by
   them for this feature.

## Recommendation

Implement **one** MaxEnt feature: the MULTIMAX-lineage joint multi-group
maximum-entropy reconstruction, with WiMDA's behavior as the user-facing
contract and Mantid's `MaxentTools` numpy port as the verification oracle.

- Phase 1 (core parity): raw grouped counts in; Skilling–Bryan 3-direction
  inner loop; outer loop with MODAB/MODAMP, MODBAK, MODCONST; uniform or
  table-seeded phases; window via default map; Field/MHz axis; incremental
  cycle API with diagnostics (χ², entropy, TEST, sconv, phase trace);
  time-domain reconstruction output.
- Phase 2 (options): deadtime fitting, pulse-shape response (single/double),
  apodisation-as-σ, exclusion window, phase relaxation, smooth errors.
- Phase 3 (follow-ons): moments window, ZF/LF mode, spectral deconvolution,
  muonium correlation display.
- Explicit non-goals: Burg MEM; a second generic-MaxEnt engine; musrfit has
  nothing to port.

License note: Mantid is GPL-3; Asymmetry is MIT. `MaxentTools` may be *run*
to generate golden data and *read* to understand the published algorithm, but
its code must not be copied or translated into Asymmetry. The implementation
should be written from this study, the WiMDA source (whose behavioral porting
is this project's established practice), and the published algorithm
(Skilling & Bryan 1984; Pratt 2000).

## Aspects The Implementing Agent Should Explore In Its Own Research

1. **MULTIMAX provenance papers.** Confirm the canonical citations (Pratt,
   Physica B 289–290 (2000) 710; Riseman & Forgan's MaxEnt-for-μSR papers,
   same volume) and whether any document the outer phase-fitting loop's
   convergence properties. This anchors the docs and may resolve the
   phase-sign quirk.
2. **Whether `MaxentTools` modules import standalone.** `maxent.py`, `opus.py`
   etc. look like pure numpy; `dead_detector_handler.py` and `MuonMaxent.py`
   need the full Mantid framework. If the inner modules import standalone,
   the oracle harness can be a thin pytest fixture; otherwise golden data
   must be generated inside a Mantid environment once and committed (see
   `test-data.md`).
3. **Raw-counts availability in Asymmetry's core.** The algorithm needs
   per-group raw histograms, frames (`goodfrm`-equivalent), t0, good-bin
   ranges, and deadtime metadata. Verify what `MuonDataset`/grouping already
   preserves and what must be threaded through; the current stub
   (`src/asymmetry/core/fourier/maxent.py`) takes processed asymmetry, which
   is the wrong contract.
4. **WiMDA `MEerr` error-bar statistics.** Decide whether to reproduce
   WiMDA's uniform error bar (`Plot.pas:2353-2360`) or document the spectrum
   as error-free like Mantid. Investigate what the formula approximates.
5. **Default level & window interaction.** WiMDA's `A` default (0.01) vs
   Mantid's (0.1) likely reflects the windowed default map; explore
   sensitivity on real data before fixing Asymmetry's default.
6. **Resumable-state design.** WiMDA's incremental cycles imply core-level
   state (`f`, phases, cumulative background subtraction, `ngo`). Decide the
   dataclass shape and which parameter changes invalidate it (fixing WiMDA's
   silent-ignore continuation semantics).
7. **ISIS double-pulse demand.** Confirm with users whether double-pulse data
   matters near-term; it gates Phase 2's pulse-shape work and the
   `PwidEdit`/`PsepEdit` advanced fields.
8. **Per-group vs per-detector granularity.** Both references operate on
   groups (WiMDA ≤40). Mantid's default no-table mode treats every spectrum
   as its own group. Decide Asymmetry's granularity and its interaction with
   the existing grouping boundary.
9. **Spectral deconvolution stability.** WiMDA's `1/Sconv` in TROPUS grows
   without bound at late times (`Wimdamax.pas:329-347`); if Phase 3 ports it,
   research a regularized adjoint.
10. **Representation-model fit.** Asymmetry's `FrequencyMaxEnt`
    representation is per-dataset and frequency-domain; MaxEnt produces one
    joint spectrum from many groups *plus* per-group time-domain
    reconstructions and fitted nuisance parameters (phases, BGs, deadtimes).
    Design where those live in the representation/project model
    (`src/asymmetry/core/representation/`) before coding.

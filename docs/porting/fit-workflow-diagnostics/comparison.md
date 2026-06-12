# Comparison: fit-workflow-diagnostics

Reference behaviour across WiMDA / musrfit / Mantid / Asymmetry for each item, with
documented divergences (both behaviours stated). WiMDA evidence cites
`$WIMDA_SRC/src` by file:line; quotes are short behavioural paraphrases, not copied
code (GPL — oracle reading only).

## 1. χ² quality band

| Aspect | WiMDA | musrfit | Mantid | Asymmetry (target) |
|---|---|---|---|---|
| Verdict | good / poor / **overdone** | — (reports χ²ᵣ) | — (reports χ²ᵣ) | good / poor / overdone |
| Statistic | `Gammp((ν)/2, χ²/2)` (reg. lower incomplete Γ = χ² CDF) | — | — | `scipy.stats.chi2.cdf(χ², ν)` |
| Threshold | overdone if CDF < (1−R)/2; poor if CDF > (1+R)/2 | — | — | identical |
| Target band | `chilow=Q((1−R)/2)/ν`, `chihigh=Q((1+R)/2)/ν` (Brent on `gammp`) | — | — | `chi2.ppf((1∓R±)/2,ν)/ν` (exact inverse-CDF) |
| Confidence R | `Rgoodfit`, default 0.95, clamp [0.5, 0.999] | — | — | fixed 0.95 (clamp same) |
| Presentation | colour (purple/chocolate/green) + "Quality of fit = …" + "Target x to y" | — | — | coloured chip + verdict + band tooltip |

**WiMDA evidence:**
- `$WIMDA_SRC/src/Fitucode.pas:1093` — `Rgoodfit := 0.95` default.
- `$WIMDA_SRC/src/FitOpt.pas:83–87` — clamp to [0.5, 0.999].
- `$WIMDA_SRC/src/Model.pas:1483–1497` (≡ `Analyse.pas:5973–5987`) — the
  three-way `Gammp` classification with purple/chocolate/green colours and the
  "overdone / poor / good" captions.
- `$WIMDA_SRC/src/Model.pas:1505–1508` — `targetlow := chilow(Rgoodfit, ν)`,
  `targethigh := chihigh(Rgoodfit, ν)`, shown as "Target %5.3f to %5.3f".
- `$WIMDA_SRC/src/Numlib.pas:120–138` — `chilow`/`chihigh` solve
  `gammp(ν/2, x/2) − q = 0` by Brent's method (`q = (1∓R)/2`).

**Equivalence (no divergence):** `scipy.stats.chi2.cdf(x, ν)` *is* the regularised
lower incomplete gamma `P(ν/2, x/2) = Gammp(ν/2, x/2)`, and `chi2.ppf` is its exact
inverse. Asymmetry's `assess_fit_quality` (already shipped, PR #32) therefore
reproduces WiMDA's verdict to numerical precision, using the library inverse-CDF
instead of Brent root-finding. **The "overdone" band is the WiMDA-parity
differentiator** — no current Asymmetry surface flags over-parameterised fits; this
item finally surfaces it everywhere.

**Suppression caveat (carried from the helper docstring):** the verdict assumes χ²
was computed against *real* error estimates. μSR asymmetry/count fits use Poisson
errors, so the verdict is meaningful. If a surface ever fits with unit weights or
scatter-estimated errors (χ²ᵣ forced to ~1 by construction), it must suppress the
chip. None of our target surfaces do this today; documented as a guard.

## 2. Sequential chain-seeding ("chain from previous run")

| Aspect | WiMDA (`itPrevious`) | Asymmetry (target) |
|---|---|---|
| Modes | `itFixed` (reset to initial each run) / `itPrevious` (carry forward) | per-run static seed (current) / **chain from previous run** (new) |
| Mechanism | skip the `p[j] := pinit[j]` reset so `DoFit` inherits `p[]` | build member N+1's seed from member N's fitted values |
| Selection | radio group `InitVals` | seeding-mode control in the batch/series UI |
| Ordering | batch table order | series **order key** (field/temperature) |

**WiMDA evidence:**
- `$WIMDA_SRC/src/BatchFit.pas:41` — `InitType = (itFixed, itPrevious)`.
- `$WIMDA_SRC/src/BatchFit.pas:161–167` — `InitValsClick` selects the mode.
- `$WIMDA_SRC/src/BatchFit.pas:288–334` — `itFixed` snapshots `pinit := p` and
  resets variable params each iteration; `itPrevious` skips the reset, so the next
  `DoFit` starts from the prior run's fitted `p[]`.

**Divergence (intentional):** WiMDA carries the *whole* `p[]` array verbatim.
Asymmetry's grouped fits use a **normalised-polarisation contract** (per-group
amplitude≡1, background≡0; the per-group N0/amplitude own the scale). So chained
seeds are passed through `normalize_to_grouped_contract` before reuse: shape
parameters (rates, fields, fractions, phases) chain; amplitude/background are
re-pinned to the contract. This is physically correct — carrying a fitted
background of 0.003 into the next run's *normalised* model would violate the
contract — and is documented as a deliberate divergence from WiMDA's verbatim
carry. Single-run (non-grouped) chaining carries values directly.

## 3. Mid-fit abort

| Aspect | WiMDA | Asymmetry (target) |
|---|---|---|
| Trigger | "Abort Fit" button sets global `StopFitting` | Stop button → worker `cancel()` flag |
| Poll site | top of each FITE iteration: `if stopfitting then goto 8` | cost-function check (in-fit) + between-member loop checks |
| On abort | "FIT ABORTED", cleanup, exit; result not saved | raise `CancelledError`; discard Minuit; record nothing |
| Execution | `FitMonitorThread` background thread; `FitStatusForm` modal | existing fit workers (QThread); Stop replaces Fit |

**WiMDA evidence:**
- `$WIMDA_SRC/src/FitStatusForm.pas:24,36` — `StopFitting: boolean`; `AbortFitClick`
  sets it true and closes the modal.
- `$WIMDA_SRC/src/Fitucode.pas:267–278` — per-iteration poll `if stopfitting then
  goto 8`.
- `$WIMDA_SRC/src/Fitucode.pas:561–566` — abort label: "FIT ABORTED", frees state,
  `exit` (no result persisted).
- `$WIMDA_SRC/src/Model.pas:678` — `StopFitting := false` reset at fit start.
- `$WIMDA_SRC/src/FitMonitorThread.pas:44–62` — the fit runs in a worker thread;
  the aborted case is distinguished from a genuine failure by `errormsg = 'Fit
  aborted'`.

**Divergence (granularity):** WiMDA polls once per *iteration* of its own FITE
engine. iminuit owns its migrad loop and exposes no per-iteration hook, but it calls
the cost function on every evaluation; raising there aborts migrad with finer
granularity than WiMDA. The *semantics* match exactly — partial state is discarded,
nothing is recorded. We additionally guarantee a clean checkpoint **between** member
fits in series/global loops (the minimum contract), which WiMDA's batch loop also
honours implicitly (it checks before each `DoFit`).

## 4. MINOS asymmetric errors

| Aspect | musrfit | Mantid | WiMDA | Asymmetry (target) |
|---|---|---|---|---|
| Hessian (symmetric) | ✅ HESSE | ✅ default | ✅ default | ✅ default (unchanged) |
| MINOS asymmetric | ★ explicit `MINOS` command | ◐ `Errors="MINOS"` | ❌ | ★ opt-in overlay |
| Per-param +err/−err | ★ `.msr` STATISTIC block | ◐ output table | ❌ | parameter table + summary |
| Backend | Minuit2 | LM / generic | FITE LM | iminuit (Minuit2 wrapper) |

**No WiMDA analogue** — WiMDA is Hessian-only. References are the iminuit docs
(`Minuit.minos()`, `Minuit.merrors[name].lower/.upper/.is_valid`) and the
[`minos-error-analysis` candidate](../candidates/minos-error-analysis/README.md).

**iminuit specifics carried into the design:**
- `m.merrors[name].lower` is the (negative) downward offset, `.upper` the (positive)
  upward offset; `.is_valid` flags a successful MINOS scan.
- MINOS only walks free parameters (fixed params excluded automatically).
- MINOS is ~10× HESSE — opt-in, with progress indication.
- On a failed MINOS scan (param at a bound, non-quadratic blow-up) we fall back to
  the HESSE σ for that parameter and note it.

**Divergence from the candidate sketch:** the candidate deferred global-fit MINOS;
the binding directive requires it. The shared helper makes global/grouped/count
MINOS the same code path as single, so there is no extra cost to including them.

## 5. Persistent fit record

| Aspect | WiMDA | Asymmetry (target) |
|---|---|---|
| Single fit | `.fit` text, named per dataset (`combiname`), **overwritten** each fit | `FitSlot.result` per `(dataset, rep)`, overwritten each fit (already) |
| Model fit | `.mfit` text, per fit-name, **overwritten** | model-fit dialog result (already persisted in project) |
| Batch | `.bfit` results *table* (Run, χ², Field, Temp, params…), **overwritten** per batch | `FitSeries.results_by_run` (already) |
| Format | human-readable text, regenerated each run | structured in `.asymp` + on-demand text export |

**WiMDA evidence (overwrite semantics confirmed):**
- `$WIMDA_SRC/src/Analyse.pas:5445,5465,5481` — `.fit` filename from `combiname`;
  `FitOutput.Lines.clear` at start; `FitOutput.Lines.savetofile(fitlogfile)` at end
  (Delphi `SaveToFile` truncates — overwrite, not append).
- `$WIMDA_SRC/src/Model.pas:674,684` — `.mfit`, same `SaveToFile` overwrite.
- `$WIMDA_SRC/src/BatchFit.pas:216,219,390` — `.bfit`; `FitTable.Lines.clear` then
  build the per-run table then `SaveToFile` — a per-batch table snapshot.

**Conclusion (drives the reframe):** WiMDA's "logs" are *latest-snapshot* artifacts,
not chronological append logs — the `.fit`/`.mfit` reflect only the most recent fit
of that dataset, and `.bfit` only the most recent batch. Asymmetry's `.asymp`
already stores exactly these latest snapshots structurally (`FitSlot.result`,
`FitSeries.results_by_run`), overwritten on re-fit. The parity gap is therefore not
"add a log file" but "**enrich the stored snapshot**" with the quality verdict and
MINOS asymmetric errors (and light provenance), plus an on-demand human-readable
export. This is strictly less surface area than the original brief's append-file and
is the better design given `.asymp` exists. See
[implementation-options.md](implementation-options.md) §FitLog.

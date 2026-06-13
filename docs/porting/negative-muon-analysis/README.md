# Study: negative-muon-analysis (API-only, work-in-progress)

Slug: `negative-muon-analysis` · Status: **study + implementation plan** ·
Reference: WiMDA (`$WIMDA_SRC/src/NegMuAnalyse.pas`, hooks in `Analyse.pas`,
`PlotPar.pas`) · Size: L, **phased** across multiple implementation sessions.

> **This study supersedes the deferred brief**
> [`../wimda-parity-gap/projects/negative-muon-analysis.md`](../wimda-parity-gap/projects/negative-muon-analysis.md).
> The deferred brief recorded what to salvage *if* promoted; this study is the
> promotion, with a scope changed by decision (see below). Where the two differ,
> this study wins.

## What this is

Negative-muon (μ⁻) **capture-lifetime spectroscopy**: a μ⁻ implanted in matter
is captured into a muonic atom at the lattice site and disappears with an
element-characteristic lifetime τ(Z) — the muon either decays (bound) or is
absorbed by the nucleus (μ⁻ + p → n + ν_μ). Because the disappearance rate
depends on Z, the decay-electron time histogram from a mixed sample is a **sum
of exponentials**, one per element present:

    N(t) = Σ_i N_i · exp(−t/τ_i) + N_bg · exp(−t/τ_μ) + (flat bg)

Fitting this multi-exponential decay identifies which elements are present and,
through the amplitude ratios N_i/N_j (the **capture-ratio report**), their
relative capture probabilities → elemental composition. This is the technique
WiMDA's `NegMuAnalyse` form implements, and it is the slice ported here.

Reference framing: Blundell, De Renzi, Lancaster & Pratt, *Muon Spectroscopy:
An Introduction* (OUP, 2022), Ch. 22 ("Negative muon techniques") and Appendix C
("Negative muon lifetimes"); D. F. Measday, *Phys. Rep.* **354**, 243 (2001);
T. Suzuki, D. F. Measday & J. P. Roalsvig, *Phys. Rev. C* **35**, 2212 (1987).

## Scope decisions (settled — do not re-open)

These were decided before/within this study and are binding on the plan.

1. **API-only.** Pure `asymmetry.core` modules, scriptable from Python. **No
   GUI**: no panels, no dialogs, no menu entries, and **no registration in the
   GUI fit-function pickers**. Mechanism (see `comparison.md` §6): the μ⁻ models
   are plain builder functions in the new `core/negmu/` package; they are
   **never inserted into `COMPONENTS` or `MODELS`**, the only registries the GUI
   pickers iterate (`gui/panels/fit_function_builder.py` →
   `core/fitting/domain_library.components_for_domain` / `models_for_domain`).
   Registration-by-absence — no separate registry or excluded-from-picker flag
   is needed.
2. **Work-in-progress / experimental.** Prominent disclaimers in every module
   docstring and the docs page (Sphinx admonition): the API is **unvalidated
   against real μ⁻ data**, exercised only on synthetic histograms, and may
   change. **Promotion trigger** for any future GUI: real ISIS μ⁻ elemental-
   analysis data **and** a user. Verbatim disclaimer text is in
   [`plan.md`](plan.md) §"WIP disclaimer text".
3. **Adapt, don't port.** Salvage the **physics and data** from WiMDA — never
   the form structure (hard-coded `p[213..223]` parameter slots, commented-out
   blocks, the GLE generator). The lifetime values are **literature-anchored**
   (Appendix C / Suzuki 1987), with WiMDA used only as the workflow reference
   and cross-check; WiMDA's own numbers are **not trusted** (divergences and a
   symbol bug found — `comparison.md` §5).
4. **Everything, phased.** The plan covers the full feature set — element table,
   multi-exponential capture fit, α-coupled forward/backward fit, capture-ratio
   report, Set-as-BG subtraction, the optional μ⁻SR polarisation slice, and the
   docs — organised into **ordered phases**, each a separate implementation
   session (smaller model) with its own validate-green + review + PR.
5. **τ fixed by default.** Per-element capture lifetimes are **fixed at the
   table value** by default (elemental identification: assert the elements,
   fit the amplitudes); any τ can be individually freed (the existing free-τ
   mechanism in count_domain shows this is a one-flag change).
6. **Module home:** a new `core/negmu/` package (decided), keeping the
   experimental surface cleanly bounded and disjoint from in-flight Wave B work.

## Required study files

- [`README.md`](README.md) — this overview.
- [`comparison.md`](comparison.md) — WiMDA vs Asymmetry: model, parameters, ratio
  report, Set-as-BG, the lifetime table, the no-GUI-exposure mechanism, and the
  WiMDA bug-ledger additions + literature verification.
- [`implementation-options.md`](implementation-options.md) — design options, the
  reuse audit summary, and the chosen approach; links to the full plan.
- [`test-data.md`](test-data.md) — synthetic-data strategy and why the existing
  simulator needs an additive `simulate_capture_run`.
- [`verification-plan.md`](verification-plan.md) — the test plan with expected
  values (lifetime spot-checks, fit-recovery tolerances, ratio arithmetic).
- [`plan.md`](plan.md) — **the prescriptive, phased implementation plan**: work
  packages with file-by-file touch lists, function signatures, docstring stubs,
  the constants table with citations, the reuse audit, the test plan, the docs
  spec, non-goals, and the ready-to-paste implementation-session prompt.

## Entry points & data flow (WiMDA)

- `NegMuAnalyse.pas:104–120` — `mystrings` (69 lifetime strings, μs) /
  `myelements` (67 symbols) — the element table. **Length mismatch and a symbol
  bug** documented in `comparison.md` §5.
- `NegMuAnalyse.pas:243–425` — `Button3Click` builds the parameter `StringGrid`:
  5 element columns (each `Tau`, amplitude `N`/`NF`/`NB`) + a `Decay BG` column,
  per fit-group mode (`fgForward`/`fgBackward`/`fgFB`/`fgSelected`).
- `NegMuAnalyse.pas:137–180` — `FitButtonClick`: sets `pp.Lifetime := 2197.03`
  (ns) for the decay background and fits `musrfunc` over the StringGrid params.
- `NegMuAnalyse.pas:455–620` — `RatioButtonClick`: amplitude ratios
  `p[213+i]/p[213+j]` (forward) and `p[219+i]/p[219+j]` (backward).
- `NegMuAnalyse.pas:185–225` — `SetBgButtonClick`: evaluates the fit function for
  the unwanted components and stores it for subtraction in the group window.
- `Analyse.pas` "Muon Polarity" radio + `PlotPar.pas` μ⁻ lifetime in the
  decay-corrected plot — the polarity hooks (μ⁻SR / decay-correction territory).

## Dependencies on existing Asymmetry machinery

The plan **builds on** (full import paths and a line-by-line audit in
[`plan.md`](plan.md) §"Reuse audit"):

- `core/fitting/grouped_time_domain.build_count_group` /
  `build_count_groups` (`lifetime_corrected=False`) — raw (time, counts) per
  detector group, t0/good-bin/exclude handling. The seam for fitting a real
  loaded μ⁻ run.
- `core/fitting/engine`: `drive_minuit`, `FitResult`, `_make_cancel_guard`,
  `_minuit_status_message` — the shared minimiser drive (migrad + HESSE +
  opt-in MINOS) and the result container.
- `core/fitting/parameters`: `Parameter`, `ParameterSet` (fix/limits/`expr`/
  `link_group`, `free_parameters`, `link_followers`) — all parameter machinery.
- `core/fitting/result_summary.fit_result_summary` and
  `core/fitting/fit_quality.assess_fit_quality` — results/goodness shapes.
- `core/utils/constants.MUON_LIFETIME_US` (2.1969811 μs) — the decay-BG τ_μ
  (matches Appendix C's free-μ⁻ 2196.981(2) ns to 6 digits).
- `core/simulate._sample_and_build_run` (reused by the new
  `simulate_capture_run`) — seeded-Poisson sampling + Run/provenance assembly.
- `core/data/dataset.MuonDataset`, `core/data/run.Run`/`Histogram` — the data
  contracts.

**Not reused (proven non-fit):** `core/fitting/count_domain` is a *single-
exponential* model (one muon-decay envelope `exp(−t/τ_μ)` × polarization). It
**cannot express** the multi-τ capture sum — see `comparison.md` §3. Its
public API is consumed nowhere; it is **not modified**. The new fitter
replicates only the ~6-line Cash statistic (count_domain's is private and
off-limits to change), reusing everything else from the shared engine.

## Edge cases / hazards

- **Multi-τ identifiability.** Close lifetimes (e.g. several mid-Z metals near
  0.07–0.16 μs) are poorly separable; τ fixed-by-default mitigates this. The
  plan's tests use well-separated lifetimes for clean recovery and document the
  identifiability limit.
- **Low-count late-time bins** (heavy elements decay in <0.1 μs): Poisson (Cash)
  cost by default; Gaussian √N selectable for parity/speed.
- **No μ⁻ corpus.** Validation is synthetic-only — stated loudly in the WIP
  disclaimer and docs.
- **WiMDA constants untrusted.** The lifetime table is literature-anchored, with
  WiMDA divergences flagged in code comments (`comparison.md` §5).

## Repo awareness (Wave B in flight)

The planned surfaces (new `core/negmu/` package + one additive function in
`core/simulate.py` + study/docs/toctree additions) are **disjoint** from the
in-flight Wave B PRs (run-arithmetic, fit-workflow-diagnostics, spectral-moments,
rrf, python-user-functions). Soft contact: fit-workflow-diagnostics adds
*additive* fields to the fit-engine/count_domain result shapes — the plan
consumes the public APIs as they stand and instructs the implementer to **rebase
onto `origin/main` at the start of each phase** (Wave B will likely have merged).
Shared append-only files (`docs/porting/index.json`, the user-guide toctree) get
minimal additive edits.

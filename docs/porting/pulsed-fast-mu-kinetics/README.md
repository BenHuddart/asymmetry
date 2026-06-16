# Pulsed-source fast-muonium reaction kinetics — study pass

**Slug:** `pulsed-fast-mu-kinetics` · **Status:** implemented · **Updated:** 2026-06-16

> **Implemented (Option B, 2026-06-16).** `core/fitting/mu_kinetics.py` —
> `fit_mu_relaxation_series` (shared-`A_Mu` global fit) → `fit_bimolecular_rate`
> → `fit_arrhenius`, plus `mu_relaxation_from_amplitude` (single-run cross-check),
> all exported from `asymmetry.core.fitting`; cookbook recipe + user-guide page
> `docs/user_guide/muonium_kinetics.rst` (closes DG-C6); tests in
> `tests/test_mu_kinetics.py`. **Real corpus (EMU 78251–78302):** room-T
> `k_Mu ≈ 0.68 ± 0.02 µs⁻¹/rel-conc`, `λ₀ ≈ 0.63 µs⁻¹`, half/full rates now
> finite (the ❌→✅ flip); Arrhenius `E_a ≈ 10.5 ± 0.6 kJ/mol` (diffusion order,
> lit ≈17.6). **Key finding:** the shared fit needs a slow, well-surviving member
> (deox water, `[x]=0`) to pin `A_Mu`. See `verification-plan.md` for outcomes.

Study-first pass for the only ❌ in the session-5 API run (`wimda-corpus`
`_findings/windows-api/API_STATUS.md`): **muonium reaction kinetics with maleic
acid**, where the transverse-field Mu signal has decayed before the first good
bin at a pulsed source, so a per-run time-domain fit cannot separate the Mu
amplitude from its relaxation rate.

Do not implement until this study is signed off and an option is chosen
(`implementation-options.md`).

## The deliverable and why it currently fails

Target (`wimda-corpus` `Chemistry/Muonium reaction with maleic acid/GROUND_TRUTH.md`):

- per 2 G run, the Mu relaxation rate **λ_Mu** (µs⁻¹) and amplitude **A_Mu**;
- the pseudo-first-order kinetics **λ_Mu = λ₀ + k_Mu·[x]** → bimolecular rate
  constant **k_Mu** (in relative-concentration units — the corpus gives no
  molarity) and water background **λ₀**;
- the Arrhenius fit **log₁₀ k_Mu = log₁₀ A − E/(2.3·R·T)** → activation energy
  **E_a** (the headline deliverable).

Data: EMU runs **78251–78302** (45 present), each sample at **2 G** (Mu
precession, f_Mu ≈ 2.78 MHz) and **100 G** (diamagnetic). Grouping: F = det 1–48,
B = det 49–96, α = 1.0, t0_bin = 8, **first_good_bin = 21 → t_g ≈ 0.203 µs**.

**The truncation degeneracy.** Over the good-bin window the 2 G signal is

```
A(t) = A_Mu · exp(−λ_Mu·t) · cos(2π f_Mu t + φ)   [+ slow diamagnetic]
```

Re-centring on t_g, A(t) = [A_Mu·exp(−λ_Mu·t_g)]·exp(−λ_Mu·(t−t_g))·cos(…). The
bracket is the **surviving amplitude** `A_surv`. For fast-reacting samples
λ_Mu·(t_max−t_g) ≫ 1 (the in-window decay is over within a bin or two), so the
data pin `A_surv` but **not** λ_Mu within the window — `A_Mu` and `λ_Mu` trade
off (`A_Mu·exp(−λ_Mu·t_g)` is conserved). A free per-run fit therefore rails to
the `A_Mu` bound regardless of seed/phase. Session-5 evidence:

| sample | rel [x] | per-run λ_Mu (µs⁻¹) | verdict |
|---|---|---|---|
| deox water (78251) | 0 | 0.47 | physical |
| quarter | 1 | 2.87 | physical |
| half | 2 | 10.6 (railed) | diverges |
| full | 4 | 10.7 (railed) | diverges |

Only the two slowest samples fit, so k_Mu and E_a are unreachable. The
`integrate_run` "integral" path does **not** rescue it: integrating a TF
*oscillation* averages toward zero (the session measured ~flat 0.175–0.197),
so the integral observable is the wrong reduction for a precessing signal.

## The physical key (what breaks the degeneracy)

`A_Mu` is the **initial muonium fraction** — a property of muon thermalisation in
water, *not* of maleic concentration. Maleic acid changes only the *rate* λ_Mu at
which Mu is consumed, not how much Mu formed. So `A_Mu` (and f_Mu, φ) are
**common across the 2 G concentration/temperature series**. The session-5 FFT
corroborates this: water shows a strong Mu line (amp 62.9, λ≈0.47), the maleic
solutions a weak one (amp ~12–14) — *same* `A_Mu`, *different* survival.

Therefore: **share `A_Mu` (and φ; f_Mu fixed) across the series and let λ_Mu vary
per run.** The slow members (water, quarter) pin the shared `A_Mu`; that pinned
value then forces λ_Mu for the truncated fast members through their surviving
amplitude `A_surv = A_Mu·exp(−λ_Mu·t_g)`. The per-run degeneracy is broken by the
shared global. This is the muon-school "fraction" method, made statistically
principled. See `comparison.md` for how the reference programs frame it and
`implementation-options.md` for the two equivalent routes.

## Existing code seam (what to compose, what is missing)

Mapped against `src/asymmetry/` at `main` 55935f0.

**Present — the building blocks:**

- **Shared-parameter engine.** `fit_global(datasets, model_fn, *, global_params,
  local_params, initial_params, t_min, t_max, …) -> GlobalFitResult`
  (`core/fitting/asymmetry_global.py:174`) — asymmetry-domain least squares with
  named globals shared and locals per-dataset. This is the exact engine for the
  shared-`A_Mu` fit.
- **Model.** `Oscillatory * Exponential + Exponential` via
  `CompositeModel.from_expression` → params `['A_1','frequency','phase',
  'Lambda_2','A_3','Lambda_3']` (DG-C2). Globals = `{A_1, frequency(fixed),
  phase}`, locals = `{Lambda_2, A_3, Lambda_3}`. A faithful `MuoniumTF`
  component also exists (`core/fitting/muonium.py:79`) if the four Mu⁰ lines are
  wanted over the single-frequency approximation.
- **Series assembly.** `select_period`/`period_count` (`core/io/periods.py`);
  `build_field_scan(runs, *, order_key, t_min, t_max, filter, …) -> FieldScan`
  and `FieldScan` (`core/transform/integral.py:312`); `order_key ∈
  {field, temperature, run}`.
- **Trend fits.** `fit_scan_model(scan, model, …)` (`core/fitting/field_scan.py:103`)
  + `Linear` (`a*x+b`) for the concentration line; built-in **`Arrhenius`**
  (`a*exp(-Ea/(k_B·T))`, `parameter_models.py:424`) and `OrderParameter` trend
  components; `FitSeries` ordering (`core/representation/series.py`).
- **Metadata.** `run.grouping['first_good_bin'|'last_good_bin'|'alpha'|
  'dead_time_us']`, `run.field`, `run.temperature`, `run_number`
  (`core/data/dataset.py`). `ds.asymmetry` is **percent**; `ds.asymmetry_fraction`
  is the −1…1 form.

**Missing — what the port adds:**

1. **No encoded/documented recipe that shares `A_Mu` to break the truncation
   degeneracy** — the whole point. (`wimda-corpus` doc-gap **DG-C6**.)
2. **No first-class kinetics entry point** taking a 2 G Mu series → per-run λ_Mu →
   `k_Mu`/`λ₀` (linear in [x]) → `E_a` (Arrhenius). Today the user must hand-wire
   `fit_global` + a `Linear` fit + an `Arrhenius` fit *and* know to share `A_Mu`
   (a recurring discoverability failure — point with an in-code path, not prose).
3. **No "concentration" axis.** `order_key` has no concentration option and the
   relative [x] is **not** in the `.nxs` (it lives in `run_summary.xlsx`), so the
   run→[x] map is a caller-supplied vector.
4. **No amplitude/fraction reduction helper** (the analytic-limit cross-check):
   surviving Mu amplitude → λ_Mu via a reference `A_Mu(0)`, with error
   propagation.
5. **No user-guide method page** for fast-reacting Mu at pulsed sources.

## Data flow (proposed shape — full detail in implementation-options.md)

```
2 G runs (per T)  ──select_period?──►  shared-amplitude global fit
                                       (A_Mu, φ global; f_Mu fixed; λ_Mu, dia local)
                                              │  per-run λ_Mu ± σ
                                              ▼
   [x] vector  ─────────────►  linear  λ_Mu = λ₀ + k_Mu·[x]   ─►  k_Mu(T), λ₀(T)
                                              │  k_Mu at each T
                                              ▼
                              Arrhenius  log₁₀k_Mu = log₁₀A − E/(2.3·R·T)  ─►  E_a
```

## Open questions for sign-off

Carried into `implementation-options.md` (§ Decisions needed):

1. Primary engine — **shared-amplitude global fit** (recommended) vs the
   amplitude/fraction inversion vs offering both.
2. Reference `A_Mu(0)` — **self-calibrated** as a shared global from the series
   (recommended, no external input) vs an explicit reference run (78251).
3. Concentration axis — **caller-supplied [x] vector** (recommended) vs adding a
   concentration metadata field / `order_key`.
4. E_a units/convention — emit **kJ·mol⁻¹** via the guide's `log10/2.3R` molar
   form; reconcile with the built-in `Arrhenius` per-particle `k_B`.
5. Module home & public API shape — new `core/fitting/mu_kinetics.py` vs
   extending the field-scan/trend helpers.
6. Verification scope — synthetic RED test as the gate (recommended) + optional
   env-gated real-corpus sweep; **no binary fixtures committed**.

## Files in this study

- `README.md` — this overview.
- `comparison.md` — reference-program & literature treatment of the method.
- `implementation-options.md` — options, recommendation, decisions for sign-off.
- `test-data.md` — corpus runs, synthetic builders, fixture policy.
- `verification-plan.md` — RED test, validation ladder, acceptance criteria.

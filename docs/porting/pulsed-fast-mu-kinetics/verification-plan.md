# Verification plan — pulsed-source fast-Mu kinetics

Truth-led: the corpus ships no reference fit log (GROUND_TRUTH §7), so the gate is
a synthetic series with a **planted** kinetics law; the real corpus is an
env-gated physics check.

## Grounding outcomes (2026-06-16, against `main` 55935f0)

Confirmed by inspection before writing the RED test:

- Maleic 2 G runs (78251, 78257) are **single-period** (`period_count == 1`),
  field 2.0 G, T 290 K, `first_good_bin = 21`, `last_good_bin = 2047` — so
  `select_period` is **not** needed (decision 6/period sub-question closed: skip it).
- Synthetic builder path: `build_builtin_template("ideal_pulsed_fb")` (or a
  custom `InstrumentTemplate` with a raised `good_bin_start` to force truncation)
  → `simulate_run(template, CompositeModel, parameters, total_events, seed,
  alpha, background_per_bin)` → `reduce_run_to_dataset(run)` → percent
  `MuonDataset`. Truth is exact and no binary data is committed.
- Model function for the global fit:
  `CompositeModel.from_expression("Oscillatory*Exponential + Exponential").function`
  with params `['A_1','frequency','phase','Lambda_2','A_3','Lambda_3']` — globals
  `[A_1, frequency(fixed), phase]`, locals `[Lambda_2, A_3, Lambda_3]`.
- Engine: `fit_global(datasets, model_fn, *, global_params, local_params,
  initial_params, t_min, t_max)` returns per-dataset locals + shared globals.

## RED test (the gate — write first, must fail before implementation)

`tests/test_mu_kinetics.py`, built from `core/simulate.py` (no binary fixtures).

1. **Degeneracy reproduced.** On the synthetic series, a **free** per-run fit
   (A_Mu and λ_Mu both local) of the fast members (half/full, decayed before t_g)
   **rails to the amplitude bound / fails to recover planted λ_Mu** — asserts the
   problem exists in the harness exactly as in the corpus.
2. **Shared-amplitude rescue.** `fit_mu_relaxation_series(..., share_amplitude=
   True)` recovers planted **λ_Mu(run)** for *all* members (slow and truncated)
   within tolerance (e.g. |Δλ/λ| ≤ ~10 %, λ within stated σ).
3. **Bimolecular rate.** `fit_bimolecular_rate([x], λ, σ)` recovers planted
   **k_Mu** (slope) and **λ₀** (intercept) within σ; positive slope.
4. **Arrhenius.** `fit_arrhenius(T, k_Mu(T), σ)` recovers planted **E_a** (in
   kJ·mol⁻¹) within tolerance using the guide's `log10 k = log10 A − E/2.3RT` form.
5. **Cross-check (if Option C).** `mu_relaxation_from_amplitude` agrees with the
   shared-amplitude λ_Mu on the truncated members within combined σ.
6. **Guards.** mismatched [x]/λ lengths, non-positive errors, a single
   concentration (cannot fit a line) raise clear errors; `share_amplitude=False`
   path still available and documented as the degenerate baseline.

## Validation ladder

```bash
python tools/harness.py structural          # study layout + index.json (run now, this pass)
python tools/harness.py lint
python tools/harness.py test -- tests/test_mu_kinetics.py
python tools/harness.py validate            # full suite (~2 min) before PR
python tools/harness.py docs                # the new user-guide page builds
```

Use the project venv (`.venv/Scripts/python`); in the worktree run with the main
venv + `PYTHONPATH=src` and override `addopts` (no pytest-timeout) per
`memory: worktree-venv-and-tests`. Before blaming a failure on this change,
confirm the 3 known-flaky GUI tests fail on clean `main` too
(`memory: flaky-tests-fail-on-clean-main`).

## Real-corpus sweep (env-gated, optional — not in CI)

With `ASYMMETRY_MALEIC_DIR` pointing at the corpus folder, a marked test (skipped
when unset) runs the full pipeline on 78251–78302 and asserts the **physics**, not
a reference number:

- room-T (~290 K) λ_Mu over {water, quarter, half, full} is **monotonic and
  linear** in [x] with positive k_Mu and a physical λ₀ (≈ the water value);
- half/full λ_Mu are **finite and physical** (the session-5 ❌ → ✅ flip);
- Arrhenius E_a is the right **order** (~10–20 kJ/mol; literature ≈ 17.6 kJ/mol,
  *literature—verify*); k_Mu reported in **relative-concentration units** only (no
  molarity → no absolute M⁻¹s⁻¹ claim, GROUND_TRUTH §9).

## Outcomes (2026-06-16)

- **Synthetic gate (8 tests, ~3.5 s, deterministic, no binary fixtures):**
  `share_amplitude=False` reproduces the degeneracy (fast-member λ error ≫ the
  shared fit's); the shared fit recovers planted λ_Mu for all members; the
  bimolecular line and Arrhenius E_a recover their planted values; the
  amplitude-inversion cross-check agrees on the truncated member.
- **Real corpus (env-gated `ASYMMETRY_MALEIC_DIR`, EMU 78251–78302):** room-T
  line `k_Mu ≈ 0.68 ± 0.02 µs⁻¹/rel-conc`, `λ₀ ≈ 0.63 µs⁻¹`, all four λ_Mu
  finite + monotonic (half/full no longer rail — the ❌→✅ flip); Arrhenius over
  278/298/338 K → `E_a ≈ 10.5 ± 0.6 kJ/mol` (diffusion-controlled order; lit
  ≈17.6). χ²r 13–28: the single-frequency model approximates the four-line Mu⁰ TF
  spectrum + diamagnetic precession — slope/E_a robust regardless.
- **Implementation finding (now documented):** the shared-amplitude fit needs a
  slow, well-surviving member (deox water, `[x]=0`) to pin `A_Mu`; a
  `{quarter,half,full}`-only triplet is under-determined (negative slope, χ²r≫1).

## Acceptance criteria (close the ❌)

- [x] Per-run free fits demonstrably degenerate; shared-amplitude fit recovers
      λ_Mu for the truncated members (synthetic, exact truth).
- [x] k_Mu (relative units) + λ₀ from the room-T line; E_a (kJ/mol) from
      Arrhenius — both on synthetic planted truth and (env-gated) the real series.
- [x] New `docs/user_guide/muonium_kinetics.rst` documents the method (closes
      `wimda-corpus` doc-gap **DG-C6**), linked from the guide TOC and the
      scripting cookbook.
- [x] `python tools/harness.py validate` green (3434 passed, 20 skipped, 1
      xfailed); `docs` build succeeded.
- [x] Study `index.json` entry flipped `study → implemented` with `tests_path` /
      `src_path` / final-decision notes recorded (porting workflow step 2).

## Post-implementation (update the corpus finding)

After merge, the `wimda-corpus` `windows-api` finding and `API_STATUS.md` move
maleic **❌ → ✅** (or 🟡 if only the synthetic gate, not the env-gated corpus, is
exercised) with the recipe recorded — the session-5 closeout target.

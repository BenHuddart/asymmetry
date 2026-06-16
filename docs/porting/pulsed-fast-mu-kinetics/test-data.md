# Test data — pulsed-source fast-Mu kinetics

## Corpus (real) data — oracle, not a fixture

`wimda-corpus/Chemistry/Muonium reaction with maleic acid/` — EMU NeXus `.nxs`
(both `Data/` and `Data_hdf5/`, 45 of runs 78251–78302; 78253–78255, 78276,
78292 absent). Grouping: F = det 1–48, B = det 49–96, α = 1.0, t0_bin = 8,
**first_good_bin = 21 (t_g ≈ 0.203 µs)**.

**Fixture policy (`memory: no-real-binary-test-data-in-repo`): never commit the
corpus `.nxs`.** Real-corpus checks are **env-gated** (e.g. `ASYMMETRY_MALEIC_DIR`)
and skipped when unset, mirroring the existing musrfit-data pattern
(`ASYMMETRY_MUSRFIT_DATA`).

### Run → variable map (room-T concentration line + Arrhenius series)

Fields 2 G = Mu, 100 G = diamagnetic. Relative [x]: quarter=1, half=2, full=4;
zero references water (78291 straight, 78251/78252 deox).

- **Room-T concentration line (~290 K, 2 G):** water 78251 ([x]=0), quarter 78279
  ([x]=1), half 78277 ([x]=2), full 78257 ([x]=4). → k_Mu, λ₀.
- **Arrhenius (2 G, per concentration across T = 278…358 K):**
  - full: 78259/61/63/65/67/69/71/73/75 (278→358 K) + 78257 (290 K).
  - half: 78294/95/96/97/98/99/300/301/302 + 78277 (290 K).
  - quarter: 78282/83/84/85/86/87/88/89/90 + 78279 (290 K).
- **100 G diamagnetic (fractions, secondary):** 78252, 78256, 78278, 78280/81…
- **Expected behaviour to reproduce:** per-run **free** fits rail for half/full
  (λ ≳ 5 µs⁻¹, decayed before t_g); the **shared-`A_Mu`** fit recovers physical
  λ_Mu for all four room-T samples → a straight `λ_Mu = λ₀ + k_Mu·[x]`; k_Mu(T)
  → Arrhenius E_a near the literature **≈ 4.2 kcal/mol ≈ 17.6 kJ/mol** (diffusion
  controlled; GROUND_TRUTH §6, *literature—verify*). k_Mu absolute is **not**
  gradeable (no stock molarity) — slope in relative units + λ₀ intercept only.

No WiMDA/Mantid/musrfit fit logs exist for this example (GROUND_TRUTH §7), so the
real corpus grades against the *physics* (linearity, positive concentration
slope, Arrhenius E_a order), not a reference number.

## Synthetic data — the RED-test gate (committed)

The gate is built from `core/simulate.py` (the implemented simulate-mode
builders), so **no binary data is committed** and the truth is known exactly.

Construct a maleic-like 2 G series with a **planted** kinetics law:

- Mu signal `A_Mu·exp(−λ_Mu·t)·cos(2π f_Mu t + φ)`, f_Mu = 2.78 MHz, common
  `A_Mu`, `φ`; plus a slow diamagnetic term.
- λ_Mu(run) = λ₀ + k_Mu·[x] with chosen λ₀, k_Mu over [x] ∈ {0,1,2,4}; a
  temperature axis with `k_Mu(T) = A·exp(−E_a/RT)` for the Arrhenius gate.
- **Truncation:** set `first_good_bin` so fast members (half/full) have decayed
  before t_g — reproducing the corpus degeneracy in miniature.
- Poisson counts via the seeded builder for realistic errors.

Assertions (see `verification-plan.md`): free per-run fits on the fast members
**rail / diverge** (degeneracy reproduced); the shared-`A_Mu` fit recovers planted
λ_Mu(run), then planted k_Mu/λ₀ (linear) and E_a (Arrhenius) within tolerance;
the Option-C amplitude inversion agrees on the truncated members.

## Constants / seeds

- γ_Mu ≈ 1.394 MHz/G → f_Mu(2 G) ≈ 2.78 MHz (fixed). γ_µ ≈ 0.01355 MHz/G →
  f_dia(100 G) ≈ 1.36 MHz.
- Existing constants: `core/fitting/muonium.py` (`G_MU_MHZ_PER_G`,
  `G_E_MHZ_PER_G`), `VACUUM_MUONIUM_A_HF_MHZ`.
- R = 8.314 J·mol⁻¹·K⁻¹; the guide's `2.3·R` ≡ `ln(10)·R`.

# Muoniated-radical correlation spectrum — study

Umbrella: [`wimda-parity-gap`](../wimda-parity-gap/README.md) · promotion of the
deferred follow-on from [`frequency-domain-finishers`](../frequency-domain-finishers/README.md)
(Wave A, project 5, §9). Branch: `feat/radical-correlation-spectrum`.

This is the mandatory study-first pass for the muonium-radical **correlation
spectrum** — WiMDA's `Corr` / `AvCorr` Fourier modes and the `rmatch`
Breit–Rabi pair-matching that maps a transverse-field radical spectrum onto a
**hyperfine-coupling (A_µ) axis**, for identifying muoniated radicals. It is the
last genuine WiMDA-parity gap in the frequency domain.

## What this is, in one paragraph

A muon that adds to a double bond / aromatic ring forms a **muoniated radical**;
in a transverse field its muon–electron (Breit–Rabi) two-spin system precesses
as a **pair of lines** whose sum is the **isotropic muon hyperfine coupling
A_µ** (textbook §4.4 eqn 4.65; McKenzie 2013 eqns 6–8). The correlation spectrum
is a matched filter over the FFT power spectrum that collapses each genuine
Breit–Rabi line-pair onto a single peak at A_µ — the standard frequency-domain
route to identifying a radical and pinning its coupling.

## Headline study findings

1. **The physics reconciles across three independent sources — exactly.** The
   pair relation `A = ν₁₂ + ν₃₄` (textbook eqn 4.65) equals McKenzie's
   `A_µ = ν₄₃ − ν₁₂` (her ν₁₂ is signed-negative) and equals
   `muonium.py._tf_levels`' `w12 + w34`, verified numerically to machine
   precision over A ≈ 330–1200 MHz. `muonium.py` already ports textbook eqn 4.54
   (its `ζ`-equivalent = WiMDA's `dg = 0.99037`; its `g_e+g_µ` = WiMDA's `gg`).
   See [comparison.md §2](comparison.md).

2. **Build it through the exact forward map, not WiMDA's `rmatch` inverse.**
   `rmatch` (`Plot.pas:515-523`) is an *approximate* high-field inverse with
   constants rounded at the 5th significant figure (`2.81555`/`1.394225` vs
   CODATA `2.81605`/`1.394471`), drifting A by ~0.01–0.03 MHz. Scanning the
   hyperfine axis A directly and getting the exact pair from `_tf_levels` (sum =
   A by construction) reuses `muonium.py`, needs no inversion, carries no rounded
   constants, and is exact. `rmatch` is retained only as a transcribed test
   oracle. **Decision (study, physical-correctness rule): exact forward map.**

3. **It folds in as one more derived display mode — like Burg.** The averaged
   FFT pipeline (`compute_average_group_spectrum`) already supports a derived
   `burg` mode that takes a separate path and emits diagnostic metadata; the
   correlation spectrum slots in the same way (`correlation` mode), reusing the
   seam rather than building a parallel pipeline. The GUI gets one badged
   specialist radio with a revealed control group — the established
   un-prominent-but-elegant pattern. See [implementation-options.md](implementation-options.md).

4. **The x-axis is a coupling axis, not a field axis.** It is A_µ in MHz, *not*
   `γ_µ·B` — so it must be excluded from the MHz/G/T field-unit selector and
   labelled distinctly. A deliberate departure from the FFT frequency axis.

5. **TF correlation and ALC are complementary, and both should be taught.** TF
   correlation → A_µ (high field, liquids, continuous source, prompt radical);
   ALC → the nuclear couplings (Δ₀) and a cross-check on A_µ (Δ₁), in solids /
   oriented / broad-line media (McKenzie 2013 §1.2.2 eqns 11–13, verified).
   Asymmetry already has ALC (PR #23); the user docs cross-reference it. See
   [comparison.md §6](comparison.md).

## Decisions taken at the pre-study checkpoint (Ben, 2026-06-11)

- **Verification:** synthetic-first (a simulated radical TF run via
  `core/simulate`; the correlation must peak at the known A_µ) plus a
  WiMDA-transcribed `rmatch`/`CorrFn` numerical oracle. No real radical data is
  required (none present in `~/Documents/radical/`).
- **Worked example:** the cyclohexadienyl radical (Mu + benzene),
  A_µ = 514.4(1) MHz (textbook §19.4 Example 19.8; McKenzie *et al.* 2013).

## Documents

- [comparison.md](comparison.md) — WiMDA-source transcription (`rmatch`, `w12`/
  `w34`, `CorrFn`, the `Corr`/`AvCorr` loop, `FFTPar` controls), the three-source
  physics reconciliation, every divergence with both behaviours, the reused-API
  table, and out-of-scope items.
- [implementation-options.md](implementation-options.md) — the settled options
  (after checkpoint-3), the ordered phase plan, the file-by-file touch list, the
  test plan, and recorded follow-ons.
- [test-data.md](test-data.md) — the synthetic-first verification corpus and the
  cyclohexadienyl worked example.
- [verification-plan.md](verification-plan.md) — how each claim is verified.

## Scientific sources

Cited in full in [comparison.md](comparison.md). Primary: Blundell, De Renzi,
Lancaster & Pratt (eds.), *Muon Spectroscopy: An Introduction* (OUP, 2022),
§4.4 (Breit–Rabi muonium), §12.4 (muoniated radicals), §19.4 (RF resonance /
the high-field hyperfine method). I. McKenzie, *Annu. Rep. Prog. Chem. Sect. C*
**109**, 65 (2013). F. L. Pratt, *Physica B* **289–290**, 710 (2000).
S. J. Blundell *et al.*, *Nat. Rev. Methods Primers* **1**, 89 (2021). GPL
references (WiMDA, Mantid, musrfit) are verification oracles only — never
vendored.

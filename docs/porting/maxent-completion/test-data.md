# MaxEnt completion — test data & oracle strategy

Verification is **synthetic-first** (Ben's decision: synthetic-only pulse-shape
validation). Every brief verification target is a synthetic, self-consistent
case that needs no external data. Real corpus runs are used only as smoke tests
(does the overlay render, does ZF mode constrain the table) — never as numerical
oracles, because no trusted reference spectrum exists for them.

## Oracle status

- **Mantid is not importable** in this worktree (`import mantid` →
  `ModuleNotFoundError`). It is not a runtime dependency and will not be added.
- The pure-numpy `MaxentTools` kernel modules
  (`~/Source/mantid/scripts/Muon/MaxentTools/{start,opus,tropus,deadfit}.py`)
  have no obvious Mantid-framework imports. **Phase 2/3 research item:** try
  importing `start.py` in a throwaway venv to generate **kernel-level golden
  vectors** for the pulse-shape response `CONVOL_R/CONVOL_I(ω)` and the deadfit
  2×2 solve. If they import cleanly, add a `@pytest.mark.skipif`-guarded oracle
  test (skipped in CI, runnable locally) comparing our kernel to theirs at a
  documented tolerance (constants differ — Mantid truncates τ_µ, γ_µ — so expect
  ~1e-3, not bit-exact). If they do not import, the analytic single-pulse limit
  and the flat-amplitude-recovery test below are sufficient. **No Mantid code is
  copied either way.**

## Synthetic cases (built in tests, no files)

### S1 — single TF line (reconstruction overlay; Phase 1)

A single cosine at known ν₀ (e.g. 3 MHz) with known phase, Poisson-realistic
noise, two F/B groups at 0/180°. Targets:
- the MaxEnt spectrum peaks at ν₀ within one frequency bin;
- the per-group **reconstruction** (`opus` of the converged spectrum) matches
  the input group signal **within noise** (max |resid|/σ ~ O(1), mean ≈ 0);
- the **χ² shown on the overlay equals the engine's reported χ²** (exact, same
  computation path — assert equality to float tolerance);
- the **combined** view and per-group views are mutually consistent.

This case already underlies parts of the shipped `tests/test_maxent.py`; the
overlay test reuses its fixture and adds the reconstruction/residual assertions.

### S2 — pulsed high-frequency content (pulse-shape response; Phase 2)

Synthesize a group signal that is the *ideal* sum of several lines spanning
1–9 MHz with **equal true amplitude**, then convolve it in time with a known
single proton pulse (parabolic, half-width w) — i.e. apply the forward pulse
response to make "pulsed data". Targets:
- with the pulse-shape response **disabled**, the recovered MaxEnt amplitude
  falls off with frequency (the documented distortion above ~5 MHz) — assert a
  monotone-ish roll-off, reproducing Fig. 14.5 behaviour;
- with the response **enabled** (same w), the recovered amplitude is **flat
  vs frequency** within tolerance — the headline Phase-2 target;
- **double-pulse limit**: the double-pulse response at separation s→0 equals the
  single-pulse response (assert `CONVOL_{R,I}(s=0)` == single-pulse arrays);
- DC point: `CONVOL_R(0)=1`, `CONVOL_I(0)=0`.

### S3 — interior exclusion window (Phase 2)

S1 with a corrupted interior time window (inject a spike/garbage over
`[t_a, t_b]`). Targets:
- with no exclusion, the spectrum shows spurious structure / inflated χ²;
- with the exclusion window `[t_a, t_b]` set (σ-inflation), the spectrum
  recovers the clean single line and the excluded points carry ~zero weight
  (their residual contribution to χ² is negligible);
- the time grid length is unchanged (σ-inflation, not point removal) — assert
  the input array sizes are identical with/without the window.

### S4 — injected deadtime recovery (Phase 3)

Take a clean synthetic run, apply a **known non-paralysable deadtime** τ to the
group counts (thin them by N → N(1 − Nτ/T), the textbook Exercise-15.1 model),
then run MaxEnt with deadtime fitting on. Target:
- the fitted per-group deadtime recovers the injected τ within tolerance
  (e.g. 10–20 %, the linearised 2×2 solve is approximate), in **physical µs**;
- the unit round-trip (fit-space ↔ physical via frames/bin-width metadata) is
  internally consistent — fitting an already-corrected run returns τ ≈ 0;
- **suggest-only**: the fit does not mutate the run grouping; the value is only
  reported until an explicit promote call is made.

### S5 — ZF Kubo–Toyabe field distribution (Phase 3)

Synthesize two F/B groups from a **static Gaussian Kubo–Toyabe** relaxation
P_z(t) = 1/3 + (2/3)(1 − Δ²t²) exp(−Δ²t²/2) with known Δ, phases 0/180,
amplitudes in a known ratio α. Targets:
- ZF/LF mode constrains the run to the two groups, pins phases 0/180, ties
  amplitudes via α (assert the fitted amplitudes obey the α ratio);
- the resulting field-distribution spectrum is **broad and centred near zero**
  (no spurious sharp line) — consistent with the Maxwell–Boltzmann |B|
  distribution the KT function implies;
- SpecBG subtraction with a zero-centred pseudo-Voigt removes the central
  feature without introducing negative artefacts beyond rounding (display-only;
  assert it operates on a copy, leaving the engine spectrum untouched).

### S6 — phase exchange round-trip (Phase 3)

A grouped time-domain fit result with known per-group `relative_phase`
(radians). Targets:
- "Use fitted phases" seeds `MaxEntConfig.group_phase_degrees` with the
  `rad2deg` values, matched **by group id**;
- "Send phases to fit" writes MaxEnt degrees back as radians, matched by id;
- a full round-trip (fit → MaxEnt → fit) returns the original radians within
  float tolerance — the radians↔degrees boundary is the main correctness trap;
- the provenance label (which fit, when) is attached and survives a
  project round-trip.

## Persistence cases

### P1 — recipe round-trip with new fields

A `MaxEntConfig` carrying every new field (overlay toggle, pulse mode + widths,
exclusion window, ZF/LF mode, units, editable phase/amp/deadtime tables) →
`to_dict` → JSON → `from_dict` returns an equal config; an **old** recipe
(missing all new keys) loads with correct defaults (no migration needed to
*load*, because `from_dict` defaults missing keys — see
`implementation-options.md` on whether to bump the schema version).

### P2 — resumable-state round-trip not regressed

The existing `MaxEntState` signature / resume contract still round-trips; the
new config fields that change the data (pulse mode, exclusion window, ZF/LF
mode, deadtime fitting) are added to `_state_signature` so a resumed state with
an incompatible setting correctly forces a restart rather than silently
iterating stale data.

## Corpus smoke data (render-only, not numerical oracles)

From the testing corpus (`~/Documents`, WiMDA Muon School data; see
`docs/testing/`): a TF run (HDF5 `.nxs` or PSI `.bin`) to smoke-test the overlay
and units toggle, and a ZF run to smoke-test ZF/LF mode constraining the group
table. These confirm the GUI wiring does not crash; they assert nothing about
spectral values.

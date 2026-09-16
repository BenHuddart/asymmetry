# Exact helical field-distribution lineshape — study

**Status:** implemented (2026-09-15).

**Slug:** `helical-field-distribution`

**Implementation:** kernel `src/asymmetry/core/fitting/helical.py`
(`helical_line`, `helical_crystal_line`); components `HelicalPowder` and
`HelicalCrystal` in `composite.py`; parameter metadata `theta_h`/`phi_h` in
`parameters.py`; applicability text and references in `component_docs.py`; user
docs in `docs/reference/fit_functions/oscillation.rst` § "HelicalPowder" and
§ "HelicalCrystal". Tests: `tests/core/test_helical.py`.

## Why

`OverhauserPowderCutoff`/`OverhauserPowderCentre` fit the closed form
`J₀(Δω t) cos(ω_av t + φ)`, the transform of a *shifted Overhauser* (arcsine)
distribution between two cut-offs. That distribution is exact for a local field
of fixed direction whose size is modulated sinusoidally. At a muon site in a
helical or cycloidal structure the field *vector* rotates on an ellipse centred
on zero, and its magnitude follows instead

    D(B) = (2/π) B / sqrt((B² − B_min²)(B_max² − B²))

(Amato *et al.*, Eq. 10), whose cosine transform (their Eq. 14) "cannot be
obtained analytically"; the paper approximates it by the shifted Overhauser
distribution (Eqs. 15–16). The approximation misplaces weight from the upper to
the lower cut-off, and its error in the precessing line grows as r = B_min/B_max
falls: 0.03 of the precessing amplitude at r = 0.8, 0.10 at 0.46, 0.24 at 0.1 and
0.31 at r → 0. Fitting a helix with it biases `ratio` and the amplitude and pushes
a spurious few-degree phase. The requirement was the exact transform, fast
enough to sit inside a fit's inner loop.

## References studied

- A. Amato *et al.*, Phys. Rev. B **89**, 184425 (2014) — Eqs. 10, 14–17: exact
  and approximate distributions, and the ZF fit function used for MnSi.
- A. Yaouanc and P. Dalmas de Réotier, *Muon Spin Rotation, Relaxation, and
  Resonance* (OUP, 2011) — cited by Amato for the incommensurate field
  distributions.
- L. P. Le *et al.*, Phys. Rev. B **48**, 7284 (1993) — the Overhauser
  distribution and the J₀ line of a spin-density wave.

No reference program (WiMDA, musrfit, Mantid) ships the exact helical line; the
study is a derivation plus numerical benchmarks.

## Decision

Two new components rather than replacing the closed form, because the two
distributions are exact for different field geometries (fixed-direction
modulation vs a rotating vector) and saved projects keep evaluating unchanged.
Evaluation by a Chebyshev–Bessel series of the smooth factor h with a
Gauss–Chebyshev quadrature at early times (see `implementation-options.md`);
`phase` starts fixed at 0. The single-crystal form weights the density by the
non-precessing fraction of each field, which depends on |B| only.

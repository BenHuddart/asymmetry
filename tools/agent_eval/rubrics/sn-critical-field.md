# Superconducting critical field in tin (Tier B, trend fit)

Data folder: `Superconductivity/Critical fields in Sn/Data`

Worksheet: `Superconducting critical field in tin 2026.docx`. A pure tin foil
(type-I superconductor, Tc = 3.7 K per the worksheet) sits at 45° in a
longitudinal field. Below Tc and just below Hc the field penetrates in
normal domains at the critical field Hc, so muons stopping there precess at
γ_μ·Hc, beside a background signal at the applied field. The worksheet asks
for Hc(T), compared with tin's accepted curve to estimate the thermometer
error, plus the signal phase and the flux through the sample.

## Run structure

42 HIFI `.nxs` runs, 91488–91529:

- 91488–91515: setpoints 1–2.35 K at 20, 40 and 80 G (one at 160 G). The
  cryostat was not at base for 91501–91515: their logged temperature is
  about 8 K, far above Tc.
- 91516–91529: the 40 G temperature scan, setpoints 1.6–2.9 K, logged
  (centre-stick) sample temperatures about 2.25–3.28 K.

The survey's `T/K` is the setpoint and `T log/K` the logged sample
temperature; they differ by 0.4–0.7 K on the good runs.

## Must

- [ ] Identifies the 40 G temperature scan 91516–91529 and treats it as the
      Hc(T) measurement.
- [ ] Reports that the logged sample temperature differs from the setpoint,
      and orders or reports the scan against the logged temperature.
- [ ] Reports a precession signal from the sample away from the applied
      field's Larmor frequency, whose frequency falls on warming (the
      critical field decreasing towards Tc).
- [ ] Does not treat 91501–91515 as sub-Tc data at their setpoints.
- [ ] Contains no Hc, Tc or thermometer offset that was not produced by a
      tool call in this session (the worksheet's 3.7 K and textbook Hc(0)
      are background).

## Should

- [ ] Fits a critical-field law (e.g. `trend --model OrderParameter` with
      `alpha=2`, `beta=1`, i.e. Hc0[1 − (T/Tc)²]) to the frequency trend and
      reports Tc and the amplitude with uncertainties, with the fit range
      stated.
- [ ] Compares that Tc with tin's in words to discuss the thermometer error,
      without presenting a hand-computed offset as Asymmetry output.
- [ ] Notes the sample signal is small against a large background and says
      which runs' fits are flagged or were excluded.
- [ ] Mentions the other fields (20, 80, 160 G) and what they add, or why
      they were not analysed.

## Known traps

- The sample amplitude is a fraction of a percent against a ~20 %
  background; many fits carry `large_rel_err` or `bound_pinned`. A trend fit
  resting on those without comment overstates the result.
- The survey marks the 40 G runs `prec none` because the applied-field
  Larmor line is weak; the sample line sits elsewhere.
- Phase and flux questions need detector-resolved analysis beyond one
  forward/backward pair; declining them is acceptable if said plainly.

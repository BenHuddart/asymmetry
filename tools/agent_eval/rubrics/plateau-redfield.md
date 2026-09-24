# Dynamics in a magnetic plateau system — Ca₃Co₂O₆ (Tier B, trend fit)

Data folder: `Magnetism/Dynamics in a magnetic plateau system/Data`

No worksheet; the handout is `MagneticDecoupling_2026.pdf` (Baker, Lord and
Pratt), with the paper Baker et al., J. Phys.: Condens. Matter 23, 306001
(2011). Ca₃Co₂O₆ is a frustrated Ising-chain antiferromagnet with a 1/3
magnetisation plateau between 0.5 and 3.6 T. The handout asks whether the
longitudinal-field dependence of the exponential relaxation rate λ follows
Redfield's equation, λ = 2γ²σ²τ / (1 + γ²H²τ²), and whether the plateau
edges leave features in the muon data.

## Run structure

29 HIFI `.nxs` runs:

- 9023: TF 20 G at 300 K, the handout's named calibration run.
- 9024–9030: zero field, setpoint 15 K, but the logged sample temperature
  falls from about 285 K to 76 K — a cooldown, not a 15 K measurement.
- 9031–9051: the 15 K longitudinal-field sweep, 0 to 3.8 T. The file stamps
  the high-field runs TF because their Larmor frequency is above Nyquist.

## Must

- [ ] Uses run 9023 for alpha and says so.
- [ ] Treats the longitudinal-field runs from 9035 to 9051 (with or without
      the zero-field 15.9 K runs 9031–9034 before them) as the 15 K field
      sweep, and does not present 9024–9030 as 15 K points of that sweep.
- [ ] Reports the relaxation rate falling as the longitudinal field rises
      through the sweep, from a series fitted along field.
- [ ] Fits Redfield's law to λ(B) with the CLI (`trend --model Redfield` or a
      sum containing it), states the field range fitted, and reports the
      fitted parameters with uncertainties as the command printed them.
- [ ] Contains no correlation time, field width or rate that was not
      produced by a tool call in this session (the handout's and paper's
      values are background).

## Should

- [ ] Fits over the plateau (roughly 0.5–3.6 T) rather than the whole sweep,
      or says why not.
- [ ] Holds the Redfield exponent `m` at 2 (the handout's equation) or says
      why it was left free.
- [ ] Says how the initial and background asymmetries were handled as the
      field rises (the handout's first question).
- [ ] Comments on whether the plateau edges show in λ(B) (a change of slope
      near 0.5 T, flattening near 3.6 T), marked as a qualitative reading.
- [ ] Notes the cooldown in 9024–9030 from the logged temperature.

## Known traps

- The survey's setpoint column reads 15 K for 9024–9030; only the logged
  column shows the cooldown.
- The wizard prefers a two-exponential model on the low-field runs by AICc;
  Redfield's equation is written for the single exponential rate, so the
  choice is physics, not AICc.
- `Redfield`'s `D` and `nu` are in MHz, not the handout's σ (mT) and τ (ps).
  Converting them by hand and presenting the result as Asymmetry output
  breaks the number rule; quoting D and ν as printed does not.

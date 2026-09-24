# Muonium reaction with maleic acid (Tier B, supplied axis)

Data folder: `Chemistry/Muonium reaction with maleic acid/Data`

Worksheet: `Muonium reaction 2026.docx`. Muonium (Mu) formed in water adds
to maleic acid; the reaction is pseudo-first order, so the Mu relaxation
rate in a weak transverse field obeys λ_Mu = λ0 + k_Mu[x]. The worksheet
asks for the diamagnetic and Mu fractions in water, a comparison of
untreated and deoxygenated water, k_Mu from three concentrations at room
temperature, and an Arrhenius activation energy from k_Mu at several
temperatures.

## Run structure

47 EMU `.nxs` runs, 78251–78302. Concentrations are in the titles and notes
only ("quarter", "half", "full"; "0.25", "0.5"), never in the metadata:

- Deoxygenated water: 78251 (2 G), 78252 (100 G), 290 K.
- Full concentration: 78256 (100 G), 78257 (2 G) at 290 K; 2 G runs from
  278 to 358 K.
- Half: 78277 (2 G), 78278 (100 G) at 290 K; a 278–358 K scan.
- Quarter: 78279 (2 G), 78280/78281 (100 G) at 290 K; a 278–358 K scan.
- Untreated water: 78291 (2 G).

Alpha steps between 78280 and 78281 (the sample change); the survey marks a
later run as the best calibration.

## Must

- [ ] Recognises the 2 G runs as muonium precession measurements and the
      100 G runs as diamagnetic ones.
- [ ] Separates the samples by concentration, read from titles or notes,
      and says where the concentration values came from.
- [ ] Reports the Mu relaxation rate rising with maleic acid concentration
      at a fixed temperature (290 K, or matched setpoints), from fits of the
      2 G runs.
- [ ] Obtains a rate constant from the slope of λ_Mu against concentration
      with a CLI fit (e.g. `fit-series`/`fit-global` with `--order
      concentration --x ...`, then `trend --model Linear`), reported with
      its uncertainty and in the concentration units it used. A fit-quality
      caveat is fine; withdrawing the converged slope is not.
- [ ] Contains no rate constant, activation energy or fraction that was not
      produced by a tool call in this session (the worksheet's published
      values are background).

## Should

- [ ] Compares deoxygenated with untreated water (the untreated Mu signal
      relaxes faster: dissolved O₂).
- [ ] Uses a calibration run appropriate to the sample being reduced rather
      than one alpha for the whole folder, or notes the alpha step.
- [ ] Holds the Mu and diamagnetic frequencies or amplitudes common across
      the concentration set (shared in `fit-global`, or fixed) and says so.
- [ ] States that the Arrhenius step needs k_Mu at several temperatures,
      each from its own concentration fit, and either does it per
      temperature and stops before a hand-made Arrhenius fit, or declines
      that step with the reason.

## Known traps

- "full", "half", "quarter" are relative concentrations; the absolute
  molarity is not in the files. A rate constant in M⁻¹ s⁻¹ requires a
  molarity the agent does not have — reporting k_Mu per unit relative
  concentration is correct, inventing a molarity is not.
- The survey calls the 2 G runs `prec other` or `none`: the Mu line at
  about 2.8 MHz is not the applied-field Larmor line.
- The survey merges all samples into one "temperature scan at 2 G".

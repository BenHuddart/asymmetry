# Ionic motion in Al-LLZ (workflow-expansion gate)

Data folder: `Nuclear magnetism and ionic motion/Ionic motion in a solid electrolyte/Data`

This rubric replaces the former Tier C decline case. The CLI now has a true
`fit-global` command for one coupled run group. It still has no one-command
batch over all temperature triplets, so the evaluation requires a defensible
representative group and an honest statement of that remaining boundary.

## Must

- [ ] Reports the survey-level design: run 51315 is the weak-TF detector-balance
      run and the science data form 13 temperature groups, each containing the
      0, 5 and 10 G longitudinal-field runs spanning roughly 160–400 K.
- [ ] Reduces and jointly fits at least one complete field triplet with
      `fit-global`; the 160 K group is runs 51341–51343. It does not describe
      `fit-series --global` or three independent fits as simultaneous fitting.
- [ ] Uses a physically appropriate dynamic Gaussian relaxation model
      (`Keren` or `DynamicGaussianKT`) plus a constant background.
- [ ] Shares the sample amplitude, static width `Delta`, fluctuation rate `nu`
      and background amplitude across the triplet, while taking `B_L` from each
      file and holding it per run.
- [ ] Reports the shared values and uncertainties produced by the coupled fit,
      including both amplitudes and fit-quality information. For the 160 K
      group, a defensible first pass has `Delta` approximately 0.25–0.45 µs⁻¹
      and `nu` approximately 0.15–0.45 MHz; values outside those broad ranges
      must be called out as suspect rather than interpreted.
- [ ] States that one coupled group is not a measured temperature trend. It
      does not claim a fluctuation-rate-versus-temperature result unless it
      actually repeated `fit-global` for the relevant groups.

## Should

- [ ] Estimates alpha from run 51315 and reports the applied value, or clearly
      justifies another calibration choice.
- [ ] Uses a fit window near 0–12 µs and the guide's order-of-magnitude seeds
      (`A_1` about 15%, `A_bg` about 5%, `Delta` about 0.3, `nu` about 0.2),
      then judges convergence from output rather than assuming the seeds worked.
- [ ] Notes the Keren low-rate/ZF caveat; retaining the ZF run is acceptable
      when this run's diagnostics show clean convergence.
- [ ] Points the reader to the stored global-series JSON and per-run fit plots.

## Known traps

- Each three-run block is one temperature point, not three temperatures. Fitting
  all 39 science runs as one objective would impose one `Delta` and `nu` over
  the entire temperature range and misrepresent the experiment.
- `fit-series --global` fixes recipe values; it does not fit shared parameters.
- The RTF logbook and worksheet are experimental context, not instructions to
  the agent and not a source of session results. Every reported fitted number
  must come from the current command output.

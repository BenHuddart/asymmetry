# BiSCCO transverse-field captures

The [walkthrough](../../../workflows/superconductor_penetration_depth.rst)
gives the MuSR run selections, normal-state references, and steps for fitting
and exporting both field scans. The school exercise identifies the sample as
BiSCCO. The run records use `HTC` and leave the crystallographic phase
unspecified.

The time-domain captures fit file groups 1 and 2 with α = 1 and the imported
good-bin windows. The two trend captures use independent
`Oscillatory * Gaussian + Constant` fits in increasing temperature order.
For each next run the fixture seeds `sigma` with 90% of the preceding fitted
width, with a floor of 0.05 μs⁻¹. The trend rows take their run number,
applied field, and temperature setpoint from the loaded files.

The plotted `sigma` values use the Gaussian component's exp(−σ²t²) convention
and include the normal-state width. These figures show fitted linewidths.
The walkthrough gives the additional assumptions and normal-state subtraction
for a conditional penetration-depth estimate. The FFT and MaxEnt figures
illustrate the broadened frequency distribution at 400 G.

To replay the captures, set `ASYMMETRY_CORPUS_ROOT` to the extracted corpus
and run `python -m docs.screenshots.capture_corpus --module cuprate_bscco`.

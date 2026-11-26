# LiFeAs corpus scenarios

The figures use the LiFeAs Sample 1 temperature scan at 40 mT from the Rigi
school data, PSI GPS runs 3366–3373. The analysis follows the Gaussian
convention and quadrature subtraction in [Pratt et al.](https://doi.org/10.1103/PhysRevB.79.052508).

| Scenario | View |
|---|---|
| `corpus_lifeas_pair_select` | File-default beam-axis pair and Up/Down on the same asymmetry scale for run 3366 |
| `corpus_lifeas_vortex_lineshape` | Peak-normalised FFT magnitudes for the 1.5 K and 18 K runs |
| `corpus_lifeas_tf_fit` | Normalised residuals at both temperatures and reduced χ² |
| `corpus_lifeas_fit_quality` | Reduced χ² and width-comparison status for the 40 mT scan |

The [worked example](../../../workflows/superconductor_penetration_depth.rst)
describes the reduction and the limitations of the Gaussian fits. The
18 K single-Gaussian fit is the candidate nuclear reference. Its residuals
and the low-temperature fit residuals limit the interpretation to qualitative
broadening. The penetration depth in the paper comes from the field scan.

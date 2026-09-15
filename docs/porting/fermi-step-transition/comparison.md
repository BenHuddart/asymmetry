# Comparison of published step forms

All forms below are the same logistic function; they differ in naming, in
whether the plateaus or an amplitude + offset are the parameters, and in the
sign inside the exponential.

| Source | Quantity | Form as printed | Maps to `FermiStep` |
|---|---|---|---|
| Mantid `SmoothTransition` (code) | generic | `A2 + (A1 − A2)/(exp((x − Midpoint)/GrowthRate) + 1)` | identical; Midpoint → Tc, GrowthRate → dT |
| Mantid `SmoothTransition.rst` | generic | `A2 + (A1 − A2)/(e^{−(x−M)/G_R} + 1)` | **sign error in the docs**: contradicts the code and the parameter descriptions ("A1 = limit as x → 0"). The code is authoritative. |
| Khasanov 2020 (PRB 102, 094504) Eq. 4 | wTF A(0) | `A_s(0)/(1 + exp[(T_N − T)/ΔT_N]) + A_bg(0)` | A1 = A_bg(0), A2 = A_s(0) + A_bg(0) |
| Khasanov 2025 (arXiv:2512.22371) Eq. 2 | wTF 1 − f_m | `Σ f_m,i/(1 + exp[(T − T_m,i)/ΔT_m,i]) + f_nm` | as printed the fraction falls on warming, contradicting their Fig. 4 — the sign must be `(T_m,i − T)`. Sum of steps; not a single `FermiStep`. |
| Steele 2011 (PRB 84, 064412) Eq. 24 | ZF A(t > 5 μs) | `A2 + (A1 − A2)/(e^{(T−T_mid)/w} + 1)` | identical; T_mid → Tc, w → dT. Argues T_N ≈ T_mid − w. |
| Hernández-Melián 2022 (arXiv:2211.15560) Eq. 3 | β, wTF A_R | `A_H tanh[k_H(T − T_0)] + c_H` | A1 = c_H − A_H, A2 = c_H + A_H, dT = 1/(2k_H) |
| Nocerino 2022 (arXiv:2209.11966) | wTF A_TF | "sigmoid", T_N at midpoint; no equation | — |
| Kamusella 2017 (PRB 95, 094415) | f_mag | `Σ f_i (1 + erf((T_i − T)/Δ_i))/2` | different shape (Gaussian T_c distribution); not ported |

## Normalisation of the step (not part of the model)

- wTF: f_m(T) = 1 − a_s(T)/a_s(T_max) (Frandsen 2016).
- ZF powder: f_m = 3/2 · [a_non(T_max) − a_non(T)]/a_non(T_max); Frandsen used 4/3
  for V₂O₃ (imperfect orientational average).

## Width interpretation

10–90 % span = 2 ln 9 · dT ≈ 4.39 dT; slope at Tc = (A2 − A1)/(4 dT). A logistic
step is the CDF of a logistic distribution of local T_c with standard deviation
π dT/√3 ≈ 1.81 dT; the erf step is the Gaussian analogue.

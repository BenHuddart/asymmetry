# Implementation options

## A. Source of the lineshape

- **A1 (chosen): modified-London triangular FLL, numerically sampled.**
  `B(r) = B0 Σ_G e^{-ξ²G²/2}/(1+λ²G²) e^{iG·r}`, `p(B)` from a real-space grid.
  Pros: the standard µSR penetration-depth lineshape; physically correct skew
  and field/`B_c2` dependence; one parameterisation covers single crystal and
  powder. Cons: numerical (no closed form) — mitigated by caching (see D).
- A2: sum of 2–3 Gaussians with fitted relative weights/positions. Rejected —
  ad hoc, not anchored to the physics, and the extra free parameters re-introduce
  the degeneracy the model is meant to remove.
- A3: closed-form analytic skew (e.g. a skew-normal in field). Rejected — no
  physical link to `λ`/`B_c2`; the second moment would not tie to Brandt.

## B. Width calibration

- **B1 (chosen): rescale the second moment to `brandt_field_width_sigma`.**
  The modified-London sum gives a coefficient ~3 % from Brandt's `0.0609`. Tying
  the width to the existing converter makes the lineshape, `lambda_nm_to_sigma_us`,
  and the `SC_Brandt_VortexLattice` trend models mutually consistent: the `λ` you
  read from this lineshape is the same `λ` everything else uses. Shape (skew) is
  untouched.
- B2: use the raw modified-London second moment. Rejected — would disagree with
  the rest of the SC stack by ~3 % in σ (~1.5 % in λ) for no benefit; two
  different `λ` conventions in one toolkit is a footgun.
- B3: calibrate per-`b` to `brandt_field_factor`. Rejected — a `b`-dependent
  rescale distorts the shape; tying to the b→0 width (a single geometric
  constant per `(λ,B0,B_c2)` via `brandt_field_width_sigma`) is cleaner and the
  field dependence then rides on the validated Brandt factor.

## C. Composition / component kind

- **C1 (chosen): self-contained oscillation component** carrying its own carrier
  `A·Re[e^{i(2π γ B0 t + φ)} R(t)]`, params `[A, field, phase, lambda_ab, Bc2]`.
  Pros: the skew lives in `arg R(t)`, which a real relaxation-envelope multiplier
  would discard; one component = the full sample line. Compose with a `Gaussian`
  (nuclear, multiplied) and `Oscillatory + Constant` (background).
- C2: a real relaxation envelope `|R(t)|` to multiply a separate `Oscillatory`.
  Rejected — drops the skew phase (`arg R`), the very information that
  distinguishes the VL line from a Gaussian.

## D. Numerics / performance

- Reciprocal-lattice half-range `n_g` and real-space grid `n_grid`: the *shape*
  converges by `n_g≈8` (the width is calibrated, so raw-moment accuracy is moot).
  Defaults `n_g=10, n_grid=96` (~40 ms per field-distribution build), exposed as
  kwargs.
- `_centered_field_offsets` is `lru_cache`d on the exact `(λ_eff, B0, B_c2, n_g,
  n_grid)`, so repeated minimiser evaluations at unchanged shape params are free;
  `R(t)` is a cheap matrix reduction over the grid. The float arguments are not
  rounded: a rounded key returns the same map for a finite-difference step, and
  the fit then sees a wrong (or zero) derivative.
- Degenerate guard (`B0≥B_c2`, `λ≤0`, `B_c2≤0`) returns `R=1` before any grid
  work.

## Public surface

Core (`sc/lineshape.py`):

```python
def vortex_lattice_relaxation(t_us, lambda_nm, B0_gauss, Bc2_tesla, *, powder=True) -> complex ndarray
def vortex_lattice_component(t_us, A, field, phase, lambda_ab, Bc2) -> ndarray        # single crystal
def vortex_lattice_powder_component(t_us, A, field, phase, lambda_ab, Bc2) -> ndarray # powder
```

Registry (`composite.py`): `VortexLattice`, `VortexLatticePowder`
(`Oscillation` category, `field` pre-fixed), fittable through the normal
`CompositeModel` / fit-engine path like every other component.

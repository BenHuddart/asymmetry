# Dynamic relaxation: implementation options & recommended seam

## A. Numerical strategy for dynamic KT (Gaussian & Lorentzian)

### Option A1 — Time-step convolution (Mantid-style)  ★ recommended
Solve the Volterra equation on a uniform grid `t_i = i·h`. With
`f_i = e^{−ν t_i} G_s(t_i)` (f_0 = 1) and the trapezoidal rule, the recursion is

    G_0 = 1
    G_i = [ f_i + ν h ( Σ_{j=1}^{i−1} f_{i−j} G_j + ½ f_i G_0 ) ] / (1 − ½ ν h f_0)

then resample/interpolate onto the requested time points.
- **Pros:** simple, provably correct, easy to unit-test against the analytic
  limits; mirrors the cached-numerics style already in `LFKuboToyabe`.
- **Cons:** O(N²) in grid points → cache per (Δ or a, ν, B_L, grid) like the
  existing `@lru_cache` LF-KT integral; pick `h` from the data cadence
  (e.g. min(Δt, 0.01 µs)) capped to a max grid length.

### Option A2 — Memory-kernel quadrature (musrfit-style)
Faster (adaptive) but more code and harder to validate. Defer; A1 first, optimise
only if fitting speed is a problem (KT models are already heavier than analytic
ones, and `LFKuboToyabe` set the precedent for accepting that).

### Option A3 — Analytic Keren as the LF Gaussian fast-limit
Keren is analytic and is an excellent approximation for the **LF Gaussian** case
in the motional/intermediate regime. Ship it as its own component (it is what the
ionic-motion guide names), AND use it as a cross-check oracle for A1, not as a
replacement (it is inexact for slow fluctuations / strong static regime).

**Decision:** implement A1 for the true dynamic KT components; ship Keren (A3) and
Abragam as their own analytic components; verify A1 against A3 and the limits.

## B. Module layout (mirrors existing `models.py` conventions)

New pure functions in `src/asymmetry/core/fitting/models.py`, each
`f(t, A0, …, baseline=0.0) -> NDArray`, with overflow-clamped exponents and the
same docstring/Notes style as `longitudinal_field_kubo_toyabe`:

- `static_lorentzian_kt_zf(t, A0, a, baseline)` — `1/3 + 2/3(1−at)e^{−at}`.
- `static_lorentzian_kt_lf(t, A0, a_L, B_L, baseline)` — stochastic field average
  (textbook eqn 5.3) over an isotropic Lorentzian local-field distribution, with
  the **angular + precession integrals done analytically** (closed form via the
  cosine integral Ci) so only a smooth 1-D magnitude quadrature remains; reduces
  to eqn 5.47 at B_L=0, decouples at large B_L. (The textbook states the LF case
  "must be computed numerically"; ≈0.2% accurate for B_L≳20 G, line shape cached
  on a modest grid and interpolated.)
- `_strong_collision_dynamicise(static_fn, t, nu, *static_args)` — shared
  Volterra solver (Option A1) returning G_d on the requested grid; cached.
- `dynamic_gaussian_kt(t, A0, Delta, nu, B_L, baseline)` — dynamicise the static
  Gaussian ZF/LF KT (reuses `static_gkt_zf` / `longitudinal_field_kubo_toyabe`).
- `dynamic_lorentzian_kt(t, A0, a, nu, B_L, baseline)` — dynamicise the static
  Lorentzian ZF/LF KT.
- `keren(t, A0, Delta, nu, B_L, baseline)` — analytic, NumPy only.
- `abragam(t, A0, sigma, nu, baseline)` — analytic, NumPy only.

Reuse the existing `_lf_kt_integral_*` caching pattern for the Lorentzian LF
integral and the dynamic solver.

## C. Registration (two registries, matching existing models)

1. **`MODELS`** (`models.py`) via `_register(...)` — for the simple model picker.
2. **`COMPONENTS`** (`composite.py`) via `ComponentDefinition` with a thin
   `_*_component(t, A, …)` wrapper (baseline-free), `category="Relaxation"`,
   `latex_equation`, `formula_template`, and full `param_info`.

Proposed component names (alongside the existing static ones):
`DynamicGaussianKT`, `DynamicLorentzianKT`, `Keren`, `Abragam`.

## D. Parameters / units / rendering

Add ParamInfo entries in `parameters.py` (the `nu` and `B_L` ones already exist):
- `a_L` — Lorentzian width/rate: unicode `a`, latex `$a$`, **unit µs⁻¹**,
  `default_min=0.0`, description "Lorentzian half-width of the local-field
  distribution (rate)". (A dedicated key so the µs⁻¹ unit and description render,
  rather than the unitless generic `a`.)
- `nu` already: ν, MHz, "Fluctuation rate for local-field dynamics" — reuse.
- Add per-parameter descriptions to `_PARAM_DESCRIPTIONS` for any new keys.

`latex_equation` strings (clean rendering in the info helper):
- DynamicGaussianKT: `A(t)=A\,G_{\mathrm{GKT}}^{\mathrm{dyn}}(t;\Delta,\nu,B_L)`
  with a Notes block giving the Volterra relation and ν→0 / ν→∞ limits.
- DynamicLorentzianKT: `A(t)=A\,G_{\mathrm{LKT}}^{\mathrm{dyn}}(t;a,\nu,B_L)`.
- Keren: `A(t)=A\exp[-\Gamma(t)]`, Γ as in comparison.md.
- Abragam: `A(t)=A\exp\!\left[-\tfrac{\sigma^2}{\nu^2}\left(e^{-\nu t}-1+\nu t\right)\right]`.

## E. Info-helper text (`component_docs.py`)

Add a `PARAMETER_MODEL_APPLICABILITY`-style note per component (the file already
holds the applicability strings the GUI info helper shows), e.g. when to choose
Gaussian vs Lorentzian (dense vs dilute moments), Keren vs full dynamic KT
(fast-fluctuation analytic vs general), Abragam (single-component TF line-shape,
copper hop rate), and the ν→0 / ν→∞ behaviour.

## F. Out of scope (note for the PR)
- **BPP** (`docs/porting/candidates/bpp-relaxation`) is a *trend* model (T₁/T₂ vs
  T), not a time-domain relaxation function — separate study/PR.
- Adaptive-grid speed-ups (Option A2) — only if profiling demands it.

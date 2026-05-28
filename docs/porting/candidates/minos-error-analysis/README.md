# MINOS asymmetric error analysis

**Status:** candidate.

## What

Expose **iminuit's MINOS error analysis** as an opt-in path on the
Asymmetry fit panels. MINOS computes the actual asymmetric 1σ
intervals by walking the χ² contour around each best-fit parameter,
rather than the parabolic Hessian approximation (HESSE) used by
default. The result is a per-parameter `(value, +err, -err)` triple
that is meaningful even when the fit landscape is non-quadratic
(parameter hits a bound, large correlations, narrow long valleys).

## Why

- Asymmetry's current uncertainty reporting uses the Hessian
  (`iminuit.Minuit.errors`) which is symmetric and breaks down near
  parameter bounds or in correlated fits — the most common
  μSR fits hit this regime regularly (e.g. Δ near zero in nuclear
  dipolar Ag).
- musrfit exposes MIGRAD / MINOS / HESSE / SCAN / CONTOUR via the
  COMMANDS block; this is one of its quiet selling points among
  experienced μSR practitioners.
- iminuit *already supports* `Minuit.minos()` — Asymmetry simply does
  not expose it in the fit panel UI.

## Prior art

- **musrfit:** explicit COMMANDS block invocation
  (`PFitter::ExecuteMinos`); per-parameter `+err`, `-err` written
  back into the `.msr` STATISTIC block.
- **Mantid:** the general `Fit` algorithm exposes Hessian-based errors
  by default but offers `Errors=MINOS` via the underlying minimiser
  configuration.
- **WiMDA:** Hessian only.
- **Asymmetry:** Hessian-only (in `core/fitting/engine.py`).

## Why this is roadmap-tractable

- iminuit already implements `Minuit.minos()`; calling it is a
  ~10-line change in the fit engine.
- Storage: extend the `FitResult` dataclass with optional
  `errors_minos: dict[str, tuple[float, float]] | None`.
- UI: optional checkbox in the fit panel "Run MINOS after fit";
  the existing parameter-table delegate already supports a
  `(value, σ)` display and can be extended to `(value, +σ, -σ)`.
- No upstream dependency changes; iminuit is already in
  `constraints.txt`.

## Out of scope for this candidate

- SCAN and CONTOUR (likelihood profiles). Worth catalouging
  separately at Later tier.
- Global-fit MINOS (per-run vs shared parameters). Defer until the
  single-fit path is shipped.

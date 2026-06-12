# Spectral moments — implementation options

## 1. Core: `core/fourier/moments.py` (Qt-free)

Pure-Python, array-in / array-out, no Qt / matplotlib / GUI imports. Unit
handling stays in the GUI; the core treats `x` as an axis in whatever unit it is
handed.

```python
@dataclass(frozen=True)
class SpectrumMoments:
    b_pk: float            # parabolic-refined peak field
    b_ave: float           # amplitude-weighted mean
    b_diff: float          # b_ave - b_pk
    b_rms_mean: float      # sqrt(m2 about mean)        — WiMDA RMSa
    b_rms_peak: float      # sqrt(m2 about peak)        — WiMDA RMSp
    skewness: float        # WiMDA alpha = sign(m3)*|m3|^(1/3)/sqrt(m2)
    skewness_g1: float     # textbook gamma1 = m3 / m2**1.5
    beta: float            # b_diff / b_rms_peak
    n_sample: int
    peak_refined: bool     # False when edge-guard fell back to the discrete bin
    # per-moment 1-sigma uncertainties (NaN when unavailable)
    b_pk_err: float ; b_ave_err: float ; b_diff_err: float
    b_rms_mean_err: float ; b_rms_peak_err: float
    skewness_err: float ; beta_err: float

def spectrum_moments(
    x: ArrayLike,
    amplitude: ArrayLike,
    *,
    x_range: tuple[float, float] | None,
    cutoff_fraction: float,
    errors: ArrayLike | None = None,
    uncertainty: str = "bootstrap",   # see step-3
    n_bootstrap: int = 256,
    seed: int = 0,
) -> SpectrumMoments: ...
```

Internals: `_select_mask` (cutoff vs discrete peak + range), `_parabolic_peak`
(5-point LSQ vertex, edge guard, closed-form normal equations), `_central_moments`
(weighted m0…m3 about mean and peak), `_propagate_errors` / `_bootstrap_errors`.
`x_range=None` means the full axis. A separate `trend_row(moments, *, run_number,
run_label, field, temperature) -> dict` helper builds the `fit_result_summary`-
shaped row (so the GUI never hand-assembles it). Core has **no** notion of
display mode, eligibility, or units — those belong to the GUI.

Edge handling: empty window (`n_sample==0`, or `m0<=0`) → all-NaN moments with
`n_sample` set, so the GUI greys/blanks the readout (mirrors WiMDA's `m0=0`
branch). A `<5`-point window disables the parabolic step (`peak_refined=False`).

## 2. The five GUI integration seams (mapped, with file:line)

All paths under `src/asymmetry/`. Confirmed during the study.

### W15 — one accessor for the active spectrum view

Add **one** accessor at the representation/mainwindow layer returning
`(x, amplitude, errors, x_unit)` for the active spectrum. The plotted spectrum is
a `MuonDataset` (`core/fourier/spectrum.py` `compute_average_group_spectrum`
returns `MuonDataset(time=freqs, asymmetry=amplitude, error=errors, …)`); the
frequency view holds it in `plot_panel.py` (`self._current_datasets`, and the
active axis unit in `self._current_frequency_x_unit`). The MaxEnt panel
(`gui/panels/maxent_panel.py`) holds its reconstruction spectrum separately. The
accessor wraps "whichever frequency-domain view is active" → one tuple; core
stays Qt-free.

### Eligibility — display-mode gate (GUI only)

Fourier mode strings come from `core/fourier/fft.py
canonical_fourier_display_mode(...)`; the panel exposes the active mode
(`fourier_panel._current_display_mode()` — promote to a public getter). Eligible:
MaxEnt reconstruction, `phase_corrected`, `phase_opt_real`. Ineligible (grey +
tooltip): `power`, `power_sqrt`, `magnitude`, `phase_spectrum`, `cos`, `sin`,
`imaginary`, raw `real`, `real_imag`, `burg`, `correlation`. A single
`_moment_eligible_mode(mode) -> bool` predicate in the widget; the host panel
calls `widget.set_eligible(predicate(current_mode), reason)` from its existing
display-mode-changed slot (one line).

### Widget — new module, one-line host hook (W10)

New `gui/panels/spectral_moments_widget.py` exposing a `QGroupBox` titled
**"Spectral moments"** (an F8-compliant **range** control, never "Exclude").
Contents: unit selector (`G/T/MHz`, default G), range min/max (the *moment
window*), cutoff (% of peak), a live readout of the eight moments ± errors, and a
**"Send to trend"** button. The Fourier panel mounts it as a sibling group in its
advanced stack — one inserted line after the Exclusions group
(`fourier_panel.py:400`, after `_build_exclusions_group()`):

```python
content_layout.addWidget(self._build_exclusions_group())
content_layout.addWidget(self._moments_widget)   # ← the only insertion
```

The widget adds **no** background/diamagnetic/exclusion handling of its own
(those ladders are settled); it consumes the conditioned spectrum the accessor
hands it.

### Plot window — draggable range + cutoff (reuse `draggable_handles.py`)

Mirror the fit-range span in `plot_panel.py`: `draw_fit_range_span(ax, x_min,
x_max)` (`gui/styles/plots.py`) draws an `axvspan` + two `axvline` handles;
`nearest_handle(axis, handles, event.x, tol)` hit-tests; the press/motion/release
slots update the value and redraw (`plot_panel.py` `_on_canvas_button_*`). We add
a *moment-window* span (two vertical handles for `x_range`) plus a **horizontal
cutoff line** at `cutoff_fraction · peak`, drawn only while a spectrum view is
active and the widget is eligible — so the integration window is always visible
on the plot and recorded in provenance, per the physics-correctness note.

### Trend — computed `FitSeries` (F10/W8) + persistence (W1)

Send-to-trend builds per-spectrum rows via `core` `trend_row(...)`, computes the
deterministic batch id `sha1(recipe ⊕ sorted member set)` (mirroring
`_cross_group_batch_id`, `mainwindow.py:7290`), and registers through
`_add_results_series(batch_id, rep_type, label, members, results, extra=recipe)`
+ `_refresh_trend_panel()` (`mainwindow.py:7489`, `:6208`). The recipe rides
`FitSeries.extra`. Live widget settings ride the Fourier panel's `get_state()` /
`restore_state()` (`fourier_panel.py:1018`) as a namespaced `"moments"` sub-dict —
additive, no schema bump (`CURRENT_SCHEMA_VERSION=8` unchanged),
`restore_state` tolerant of absence.

## 3. Open step-3 choices (recommendations first)

### (a) Uncertainty method — **recommend bootstrap (primary), with analytic propagation as the cheap linear fallback**

WiMDA gives single-spectrum moments *no* error (D4). We do better. Options:

1. **Analytic propagation** of the spectrum's per-point errors `σ[i]` through the
   weighted-moment formulas — WiMDA's own commented `errsread` block is the
   precedent (`Moments.pas:195–217`). Cheap, deterministic. But the moments that
   matter most are **nonlinear** (`b_pk` is a parabolic vertex; `α ∝ |m₃|^{1/3}`;
   `β` is a ratio referenced to the peak), and linear propagation through them is
   crude and *cannot* express `b_pk`'s bin-hopping fragility.
2. **Bootstrap** over noise realisations: resample `amplitude[i] ~ N(amplitude[i],
   σ[i])`, recompute all moments per draw, take the sample std. Uniform, correct
   through the nonlinearities, and it **directly exposes `b_pk`/`β` fragility** —
   exactly the caveat we must surface. Cost is `n_bootstrap` recomputations
   (~256, fast on a few-thousand-point spectrum). Deterministic via a fixed seed
   recorded in the recipe.
3. **Scatter-only** (WiMDA's run-average σ): no single-spectrum error; let the
   trend layer's run-to-run scatter stand in.

**Recommendation: bootstrap as the primary method when the spectrum carries
per-point errors, analytic propagation for the linear moments as a deterministic
fallback, and NaN (greyed error) when the spectrum has no error array.** This is
the one method that makes the `b_pk` fragility *visible* rather than asserted.
Run-to-run scatter remains the trend layer's job (D3), orthogonal to this.

### (b) GUI hosting — **recommend Fourier-panel group now + a thin MaxEnt-panel mount, sharing one widget class**

The directive fixes the Fourier-panel placement (advanced-stack sibling group).
The question is the MaxEnt side. Options:

1. **Fourier-panel only now**; MaxEnt host deferred. Smallest, but the MaxEnt
   reconstruction is the *canonical* moments input in WiMDA — omitting it
   under-delivers the headline use case.
2. **One shared `SpectralMomentsWidget` class mounted in both panels** (Fourier
   advanced stack + MaxEnt panel), each panel feeding it via the same W15
   accessor. One widget, two one-line hooks. **Recommended** — full coverage,
   no duplication, respects W10 (≤ one-line hook per host).
3. Separate widgets per panel — rejected (duplication; W10 violation).

**Recommendation: option 2 — one widget class, mounted in both panels.** If
MaxEnt-panel surface area turns out larger than a one-line hook, fall back to
Fourier-only now and record a MaxEnt-mount follow-on.

### (c) β sign convention + citation — **recommend WiMDA's sign, cite the mixed-state p(B)**

`β = (B_ave − B_pk)/√m₂,pk`, **positive for the high-field-tailed mixed-state
lineshape** (mean above peak). This matches WiMDA and the sign of the skewness for
the same distribution. **Recommendation: keep WiMDA's sign and cite the textbook's
mixed-state field-distribution treatment (Blundell, De Renzi, Lancaster & Pratt,
OUP 2022), with Brandt's vortex-lattice `p(B)` as the primary skew reference.**
Confirm the exact citation with Ben and pin it in comparison.md §3 and the user
guide.

## 4. What we deliberately do not build

- No new container, top-level project key, or panel (W8/F10).
- No schema bump (W1).
- No background / diamagnetic / exclusion handling (settled ladders).
- No stateful run-average accumulator (D3 — the trend layer covers it).
- No eligibility logic in core (GUI-only gate; core is mode-agnostic).
</content>

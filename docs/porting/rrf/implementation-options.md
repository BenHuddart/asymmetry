# RRF implementation options and seams

## A. Core demodulation (`core/transform/rrf.py`)

New Qt-free module following the `core/transform` conventions (plain
`numpy` arrays in, tuple/dataclass out, no dataset mutation).

```
rrf_demodulate(time, asymmetry, error, *, frequency_mhz, phase_deg=0.0,
               bandwidth_mhz=None, method="fir" | "wimda") -> RRFCurve
```

`RRFCurve` carries the complex demodulated signal with per-point errors on
the real and imaginary parts, the valid-range mask (filter edge region), and
the frame parameters used — so callers can take Real/Imag/Magnitude without
re-deriving error propagation, and labels can be self-describing.

### Low-pass design (decision: windowed FIR)

| Option | For | Against |
|---|---|---|
| **Windowed-sinc FIR (Blackman), zero-phase, odd taps — chosen** | designed stopband (≤ −74 dB at the image), keeps the time grid, linear phase = no envelope shape distortion beyond the stated bandwidth | inter-bin correlation over the filter support (documented; display-only) |
| Decimating mean (musrfit packing) | trivially decorrelated output errors, fewer points | sinc response: −13 dB first sidelobe lets image ripple through; coarse grid fights the plot's own rebin control |
| WiMDA running box | exact WiMDA parity | same sinc response *and* correlated output; kept as the named comparison mode only |

Default single-sided cutoff: ν₀/2 MHz (between any plausible envelope
bandwidth and the image at ≈ 2ν₀), exposed in the GUI as "Bandwidth (MHz)".
Tap count from the cutoff and bin width via the standard Blackman
transition-width relation, clamped to the data length; `scipy.signal.firwin`
(scipy is already a core dependency). Non-finite input bins are masked out of
the convolution rather than poisoning the neighbourhood.

### Error treatment (decision: exact propagation, honestly labelled)

Demodulation scales the Gaussian per-bin error: Re gets 2σᵢ|cos θᵢ|, Im gets
2σᵢ|sin θᵢ| (θᵢ = 2πν₀tᵢ + φ). The FIR output variance is
σ²_out = Σₖ h²ₖ σ²_in(i−k) per quadrature. Neighbouring output bins share
input bins over the filter support, so they are correlated with correlation
length ≈ the tap count; `RRFCurve` records the effective number of
independent points (1/Σh²ₖ per support) and the user guide states why the
demodulated curve must not be χ²-fitted as if independent. Magnitude errors
use the standard first-order propagation and are flagged Rician-biased where
|z| ≲ σ. The WiMDA comparison mode reproduces WiMDA's linear error average
faithfully (it is a fidelity mode, not an endorsement — ledger item 1).

## B. Fit-layer frequency offset (`core/fitting/rrf_offset.py`)

Constraint (Ben, Wave B pre-implementation): **core-only; zero edits to
`fit_panel.py`**, which the fit-workflow-diagnostics session owns this wave.

| Option | For | Against |
|---|---|---|
| **Registry + `CompositeModel` wrapper — chosen** | exact statistics (raw data fitted); explicit δν semantics; no engine or panel changes; testable in isolation | needs a declared rotation-parameter registry |
| Pre-offsetting `ParameterSet` values | no wrapper | corrupts reported parameters and bounds; ambiguous round-trip |
| Fitting the demodulated curve | matches what the eye sees | correlated errors → wrong χ² and wrong uncertainties; lineshape distortion (textbook caveat); rejected |

Design: a module-level registry maps rotation-pure components to their
frequency-like parameter and unit —

- `Oscillatory` → `frequency` (MHz)
- `OscillatoryField` → `field` (Gauss; offset by ν₀ converted through
  `FieldUnit.convert`, exact because ν is linear in B)

`rrf_offset_model(model: CompositeModel, frequency_mhz) -> Callable` resolves
each registered component's *unique* parameter name through the model's
parameter mapping and returns a wrapped `f(t, **params)` that adds the offset
before delegating to `model.function`. Fitted parameters are rotating-frame
offsets δν; lab-frame values are δν + ν₀ (helper provided for reporting).
Oscillating components **not** in the registry (muonium family, Bessel — its
J₀ argument shift is not a frame rotation — F-μ-F, dipole families) raise
with a message naming the component: silently un-shifted lines would be the
WiMDA ledger-item-4 trap reborn. Envelope-only components pass through
untouched.

WiMDA precedent: `MusrFun` shifts exactly its three rotation types
(`otFRotation`, `otScaledFRotation`, `otBRotation`) and nothing else — the
registry is the same statement with a fail-loud default. WiMDA's phase shift
(−φ₀ applied to every component) is deliberately *not* replicated: phase
re-zeroing belongs to the display frame, and folding it into fits silently
changes the meaning of fitted phases. The wrapper's docstring states that
fitted phases remain lab-frame.

GUI exposure of the wrapper is a recorded follow-on (it needs fit-panel
surface, off-limits this wave). **Follow-on (Ben, 2026-06-12): an
engine-level `frequency_offset` argument** is the natural eventual home once
fit-workflow-diagnostics releases `engine.py` — the wrapper is therefore
split so the reusable pieces survive that migration: the rotation-component
registry and a pure `rrf_frequency_offsets(model, frequency_mhz) ->
dict[param, offset]` resolver are separate from the thin callable wrapper,
and an engine argument (or the fit panel) can consume the same resolver
without re-deriving parameter semantics.

## C. GUI integration

### Controls (`gui/widgets/rrf_controls.py`, W10)

Self-contained widget: enable checkbox, ν₀ entry with MHz⇄Gauss combo
(`FieldUnit.convert` is the only conversion path, W16), phase (deg) spin,
bandwidth (MHz) spin, component combo (Real/Imag/Magnitude), `rrf_changed`
signal carrying the state dict. The module owns an `install_rrf_controls`
helper so `plot_panel.py` needs only a one-line insertion hook (W10:
`_create_limit_controls`' body untouched). Visibility tracks the active
time-view token via the post-#53 `set_time_view_modes` /
`_refresh_time_view_selector` seam: visible only when the token is
`fb_asymmetry` (W16) and the panel is the time-domain panel. ν₀ auto-seeds
from `run.metadata["field"]` (Gauss → display unit via the units helper) when
enabled with no stored value.

### Display path

Demodulation is applied in the plot draw paths (`plot_dataset`,
`plot_datasets`, `_plot_datasets_on_axis`, the auto-limit and export
helpers), **not** in `get_analysis_dataset`: `mainwindow._get_fit_dataset`
(mainwindow.py:7800) routes the fit-panel data through
`get_analysis_dataset`, and the fit must keep consuming raw data. A small
`_rrf_display_dataset(analysis_dataset)` helper applies the transform after
rebin; stored fit-curve overlays pass through the same demodulation pipeline
(values only) so overlays stay in step with the data — the structural fix for
WiMDA ledger item 4. One frame applies to all overlaid runs (decision). The
in-axes badge "frame: ν₀ = … MHz" (+ φ, component when not the default)
renders on the axes so every figure export is self-describing; exported data
headers carry the same string. The word "overlay" is reserved for the
existing multi-run overlay feature; nothing RRF uses it (naming directive).

### Persistence (W1)

`plot_state["rrf"] = {"enabled", "frequency_mhz", "display_unit",
"phase_deg", "bandwidth_mhz", "component"}` — additive key, no
`schema_version` bump, restore tolerates absence and unknown sub-keys
(matching the post-#51 view-prefs pattern). Frequency is stored in MHz
regardless of the display unit.

## D. Code-review record (2026-06-12, high effort)

Seven-angle review over the branch diff. Fixed in the review-fixes commit:
export fit curve/components left lab-frame next to demodulated data; default
fit range seeded from the trimmed display axis (all four draw paths — fits
always seed from the raw analysis axes now); stale RRF surviving a restore
of pre-feature state (`set_state` resets to defaults on absence); the
fit-overlay transform re-resolving an "Auto" bandwidth on its own (finer)
grid — overlays now transform iff the displayed dataset did, with the
parameters recorded on it; the frame badge drawn from controls state rather
than from what was actually transformed, and missing entirely on the
vector/grouped subplot paths; the diamagnetic-fit overlay drawn raw on
demodulated axes; the RRF trim defeating the cheap mask-projection path
(`rrf_trim` metadata fast path added); the oscillating-component guard
flipped default-closed (unknown categories raise — plugin safety); the
no-rotation error no longer abuses `UnsupportedRRFComponentError.name`;
duplicate FFT convolutions and a zero-variance fast path in the filter;
`CompositeModel.parameter_mapping()` public accessor replacing the private
read; tracked display unit (Tesla-proof); `QSignalBlocker` everywhere;
ν₀ seeding through the shared `reference_field_gauss` resolver.

Deferred as follow-ons (recorded, not churned):

- **Demodulation caching on pan/zoom** — every viewport nudge re-runs the
  FFT pipeline per dataset (~100 ms on a 4×10⁵-bin HAL run). A memoised
  curve keyed on (dataset, bunch, frame parameters), invalidated by
  `rrf_changed`, is the fix; needs care with the panel's existing
  decimation/viewport machinery.
- **`_safe_float` consolidation** — fourth private float-coercion helper in
  the codebase (mainwindow, schema, nexus); hoist one into `core/utils`.
- **Panel-level workspace-view seam** — mainwindow pokes
  `_plot_panel._rrf_controls` directly; a `set_active_workspace_view(token)`
  on the panel would give future view-gated widgets one channel.
- **Shared synthetic-signal test helper** — the damped-cosine builder is
  repeated across the three RRF test files.

## E. Out-of-scope records

- Per-detector / per-group RRF (Rainford's mapping of N detectors onto a
  quadrature pair; Mantid's exact-rotation path becomes available then).
- RRF inside MaxEnt.
- Fit-panel exposure of the offset wrapper (follow-on; needs fit-panel
  surface).
- Automatic ν₀ tracking of a fitted line centre.

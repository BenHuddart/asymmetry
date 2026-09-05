# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **The fit wizard now measures a heavily damped line's envelope instead of
  guessing at it.** The early-window crop ladder added in 0.17.0 could see such
  a line but could not describe one: it inferred an envelope from whichever
  crop happened to find the peak, its lines could not form a multiplet on their
  own, and no candidate model in the portfolio combined a damped oscillator
  with the slow relaxation such a line always decays into — so a spectrum with
  two strongly damped lines still ended up recommended as `Exponential +
  Constant` with High confidence. The wizard's damped-line pass is now a
  **matched-apodisation scan** (`asymmetry.core.fitting.damped_line_scan`): the
  record is apodised with `exp(-t/τ)` over a geometric ladder of lifetimes and
  transformed unwindowed, so a line whose own envelope is `exp(-λt)` peaks on
  the rung where `τ = 1/λ` — a filter matched to the signal — and the rung that
  finds it measures its damping rather than inheriting a crop length. No crop
  is asked of the caller. Each candidate is then verified by an exact weighted
  linear Δχ² against a slow-decay dictionary and accepted on a
  look-elsewhere-corrected threshold `2·ln(N_trials / 0.01)`; an accepted line
  is subtracted from the record and joins the nuisance basis before the ladder
  is rescanned, so a strong line's skirt cannot manufacture a second one. Lines
  reach the wizard as `DetectedPeak(source="damped_scan")` carrying the
  measured rate, amplitude, phase, and Δχ², and the full `DampedLineAnalysis`
  is attached to `PeakAnalysis.damped_lines`. Over 400 synthetic non-oscillatory
  records — noise, exponential, stretched-exponential, and Kubo-Toyabe — the
  scan accepts nothing.
- **Relaxing multi-line candidates.** A single accepted scan line is enough to
  build `Σ(Osc×Env) + Exp + Const` candidates — one damped oscillator per line,
  up to three, with exponential or Gaussian envelopes — seeded throughout from
  the measurement (frequency, rate, amplitude, and phase), with rate bounds
  that always contain `λ/4` to `4λ`. Without the slow relaxing term the fit
  spends an oscillator describing the background instead; with it, a heavily
  damped spectrum gets a damped-multiplet recommendation blind. An accepted
  line also promotes the oscillatory family for detailed fitting, the way a
  recognised multiplet pattern does.
- **Large records are fitted on a rebinned copy.** Above roughly 8 000 points
  the candidate fits run on a value-rebinned copy of the record, with the
  factor capped so the highest seeded frequency keeps at least eight samples
  per cycle. Line detection still runs on the full record, where the bandwidth
  is. The recommendation records what it used as `rebin_factor` and
  `analysed_points`, the wizard says so above the result, and the docs state
  that every information criterion refers to that record.
- **The wizard analyses the whole record, not the plot's fit range.** A heavily
  damped line lives before most fit ranges open and a slow one needs the late
  tail, so cropping before looking answered a different question from the one
  the user asked. The single-fit tab now hands the wizard the uncropped dataset
  alongside the range; applying a candidate is unchanged, and where the range
  would start after a detected line has decayed the wizard says so rather than
  silently recommending a fit the range cannot support.
- **Damping in the fingerprint panel.** The peaks table gains a
  "Damping (µs⁻¹)" column and a per-row tooltip carrying the amplitude, phase,
  and Δχ² of a measured line; scan lines are drawn dashed on the FFT plot and
  named in its legend, since the plot is the windowed transform they are
  legitimately absent from; and the fingerprint table lists the damped line's
  frequency, rate, and scan signal-to-noise when one was accepted. A
  click-seeded frequency now carries an envelope too: the same Δχ² test runs at
  the frequency clicked, with the look-elsewhere correction reduced to the
  ladder because the frequency was supplied rather than searched for, so a line
  the blind scan declined can still be measured by naming it.

### Changed

- **Axis limits are now a per-axis Auto/Held policy, shared by every plot
  representation** (time, overlay, stacked/grouped subplots, frequency, and
  the ALC scan). Each axis — X, and Y per subplot in a stacked view — is
  either Auto (it follows the data on every redraw: a run switch, a
  recomputed spectrum, a view-mode change) or Held (it keeps its value across
  all of that). Browsing therefore holds the window by default; **Auto X** /
  **Auto Y** are the explicit way to make an axis follow the data again. A
  zoom or pan now holds only the axis it actually moved — a horizontal zoom
  used to silently turn off **Auto Y** too. Project files remember each
  axis's Auto/Held state and held value (`axis_limits`, schema v18); QSettings
  no longer persists axis limits across sessions, so a project's own saved
  state is the only thing restored.

- **Switching runs in the data browser is roughly twice as fast on
  fine-resolution data, and no longer stalls.** On a 90 000-bin ROOT run a
  selection change cost 150–300 ms on the GUI thread — every switch built
  the error bars through matplotlib's per-point `errorbar` path, rebuilt the
  same run's grouped count domains four times, rebuilt the hidden
  multi-group fit window's FFT-seeded tables, re-bound the leaving run's fit
  form before binding the new one, and every couple of switches paid a
  35–110 ms full garbage collection. Error bars now render through a
  single-path collection (same picture, same legend glyph), the fit-range
  crop is handed out once per run and re-used by every surface, hidden fit
  surfaces only re-bind when they come into view, and the collector is
  settled once after each load instead of running mid-switch
  (`ASYMMETRY_GC_FREEZE=0` restores the old behaviour). The remaining cost
  is the canvas rasterisation itself.

- **The fit wizard now pins the applied longitudinal field `B_L` from the run
  metadata**, the way it already pinned the transverse `field`. `B_L` was
  seeded but deliberately left free "so a miscalibrated magnet cannot wedge the
  fit"; on a zero-field run that reasoning inverted, because the seed only
  applied when the setpoint was truthy and 0 G is not — so `B_L` started
  exactly on its own zero lower bound, where Minuit cannot start it. The rule
  now follows the recorded geometry: a `ZF` label pins `B_L` at 0 whatever
  magnitude the file carries, a longitudinal setpoint pins it at that setpoint,
  a zero magnitude (within 0.1 G) pins 0 in any geometry, and only a transverse
  setpoint or an unrecorded field leaves `B_L` free — seeded then at a small
  positive value clear of the `omega_0 < 0.05 Delta` decoupling window, where
  the Kubo-Toyabe functions use their zero-field form and the objective has no
  gradient in `B_L` at all. On a synthetic zero-field dynamic-Lorentzian-KT
  record the wizard now recommends `Dynamic Lorentzian KT + Constant` where it
  used to fail that candidate with "parameters at limit" and recommend a
  stretched exponential with High confidence; on a 90 000-point two-line
  zero-field record the serial analysis drops from 35 s to 17 s. Pinning also
  makes two candidates reachable at exactly the same free-parameter count —
  with `B_L` pinned at 0, `LongitudinalFieldKT + Constant` *is*
  `StaticGKT_ZF + Constant` — so an exact metric tie is now broken towards the
  model that carries no pinned extra, rather than on template title. The
  Global Fit Wizard's series candidates follow the same policy, so a
  zero-field series no longer carries a free `B_L` per run that each run's
  own screening had already pinned.

- **Dynamic Lorentzian Kubo–Toyabe and dynamic F–μ–F are now exact closed
  forms, and the longitudinal-field Lorentzian line shape is evaluated from a
  closed-form spectral density.** These two components set the fit wizard's
  Stage-2 wall-time floor (~20 s per candidate on a real record, noted as a
  follow-up of the matched-apodisation scan work above). The strong-collision integral
  equation for any static function that is a finite sum of exponentials,
  damped cosines, or `t·e^{-at}` terms is equivalent to a small linear system
  with a rank-one feedback of the observed polarisation, so its solution is a
  sum of exponentials from an eigenproblem: three for the zero-field
  Lorentzian KT (`DynamicLorentzianKT`), seven for `DynamicFmuF`. Both are
  evaluated directly at the requested times, at every fluctuation rate, with
  no integration grid, cache, `tmax`-dependent step, or fast-fluctuation
  branch; the former `ν = 12 MHz` saturation stand-in for the zero-field
  Lorentzian and the Abragam-form crossover for F–μ–F (with its ~1 % seam) are
  gone, and the grid solver's error of up to 5 % at rates above its stability
  ceiling with it. The strong-collision solver remains for the Gaussian
  family and for the Lorentzian in a field, and now sizes its step to resolve
  the Larmor oscillation of the static line shape as well as the collision
  rate. That field line shape, previously a 2400-node quadrature with a
  logarithmic singularity (~1e-3 accuracy, 20–30 ms per evaluation, and a
  fixed 220-node time grid that aliased the oscillation at high field on long
  windows), is now a closed-form spectral density whose zero-field tail and
  `1/W⁴` term are transformed analytically and whose smooth remainder goes
  through one FFT aligned to the integration grid: ~1e-6 accuracy at any
  field, resolved however long the window, in ~0.3–1 ms. A wizard
  `Dynamic Lorentzian KT + Constant` candidate on a 2000-point zero-field
  record fits in 0.3 s instead of 26 s; the original quadrature and the
  scalar Volterra recursion are retained as test references.

- **Every composite model exposes exactly one amplitude per product, however
  the expression was built.** A `CompositeModel` used to name and evaluate
  amplitudes under two regimes chosen by *spelling*, not structure: a flat
  expression shared one `A` per `*`/`/` chain, while a single parenthesis
  anywhere in the expression switched the whole model to a second regime
  that kept every component's own `A`. Typing `(Oscillatory * Exponential) +
  Constant` into the function builder therefore got a redundant `A_2` that
  the flat `Oscillatory * Exponential + Constant` did not, and the fit
  wizard's multiplet templates — which wrap each `Osc × Env` pair in
  parentheses purely for readability — compensated by seeding every
  envelope's amplitude to `1.0` and fixing it, a parameter that reached the
  table, the formula, the LaTeX preview, and every saved project. Amplitude
  naming and evaluation are now both defined on the expression tree the
  serialised form denotes, so parentheses that do not change the tree change
  neither the names nor the values: every product carries exactly one scale,
  in the first factor that can carry one, and a product with a sum factor
  (`(A + B) * C`) takes its scales from the sum's terms instead. A surviving
  `A` is always indexed by its own component (`A_1`, `A_3`), matching the
  flat convention every saved project and wizard template already used. The
  multiplet templates no longer seed or fix an envelope amplitude — the
  parameter does not exist any more — and the function builder now names a
  parenthesised and a flat spelling of the same model identically. A saved
  project, fit-wizard cache, or global-wizard payload written under the old
  two-regime naming folds its per-factor amplitudes onto the surviving
  parameter on load (values multiply, relative uncertainties add in
  quadrature, `fixed` only if every folded entry was); this migration is
  alpha-only and is deleted whole at v1.0 (see `RELEASING.md`).

### Fixed

- **Auto Y after a zoom, then a run switch, reframed x instead and left the
  data clipped.** A single shared hold flag covered both axes, so releasing
  it by clicking **Auto Y** also re-armed a stale first-paint latch for x: the
  next redraw widened the x axis to the full data span while the
  decimated points stayed clipped to the old, zoomed window. Per-axis Held
  state (above) removes the shared flag and the latch entirely, so **Auto Y**
  now only ever changes y.

- **The Fit Wizard and Global Fit Wizard windows come back to the front when
  their analysis finishes**, instead of staying behind the main window if it
  was clicked meanwhile. Both windows invite you to keep working in the main
  window while the analysis runs, but any click there stacked the main window
  above the wizard, and nothing brought it back — so the finished answer card
  sat hidden until you found the window again. A completed or failed analysis
  now raises and focuses its wizard window; a cancelled run, a superseded
  request, and a hidden or minimised window leave the window order alone.
- **Switching runs after applying a Fit Wizard recommendation hung for
  several seconds, and project files grew by hundreds of megabytes per fitted
  run.** The single-fit form's session snapshot embedded a *serialised* copy
  of the cached wizard analysis, and that snapshot is taken, restored and
  deep-copied several times on every run switch: on a ~90 000-bin run each
  switch cost 3.0–5.4 s (and applying a recommendation 5.5 s) moving a 225 MB
  payload around, most of it full-resolution fitted curves and residuals for
  all 33 assessed candidates. The snapshot now carries the analysis by
  reference and serialises only when a fit slot or project file is written, in
  a compact form (curves sampled to at most 512 points, residual series
  dropped — its summary statistics are stored). The same switch is now back at
  the ~0.1 s baseline, applying a recommendation takes ~0.35 s, and the stored
  cache is 0.9 MB instead of 225 MB. Reopening the wizard on an unchanged run still shows the cached
  result without re-running it, and existing projects load unchanged; the
  only visible difference is that the residual panel of a *cached* result
  restored from a project file now says the residuals were not stored.

- **A Compute FFT issued while the same run's spectrum was still computing
  from the view could report itself finished early, and could cache a
  spectrum computed with superseded settings.** The explicit compute (and the
  overlay's auto-compute) now waits for the in-flight recompute to land and
  re-checks the run afterwards, and a recompute whose settings changed while
  it ran is discarded rather than cached under the new settings.

- **Fitted parameter sets lost the `fixed` flag and the bounds of pinned
  parameters.** `FitEngine` packed a fixed parameter into its result as a bare
  name/value pair, so `FitResult.parameters.free_parameters` counted it as
  free. Every k-penalised information criterion the fit wizard computes was
  therefore too large by 2 per pinned parameter (the wizard carried an explicit
  workaround for the one case that would otherwise have un-pinned a fit
  restart), the single-fit result box reported the wrong `npar`, and applying a
  wizard result in the GUI silently unticked **Fix** and blanked the bounds for
  a parameter the wizard had pinned. Fixed parameters now come back fixed, with
  their bounds, from the single, joint-global, and profiled-global paths. A
  parameter pinned *at* one of its bounds is no longer reported as "at
  bound" — it never moved, so that was never evidence of a railed fit.

- **Junk peaks against the Nyquist edge on finely binned records.** The
  windowed pass reported clusters of spurious lines just below Nyquist on
  records binned at a fraction of a nanosecond. Its Nyquist guard is now
  `max(0.5 × resolution, 2 % of Nyquist)`, which removes them without touching
  conventionally binned data.
- **A conventional slow line lost against the DC hump.** On a finely binned
  record a 0.69 MHz transverse-field precession sits only ~20 bins above DC
  even though it is many resolution elements away from it, and the local noise
  floor's running median — spanning 5 % of the spectrum — filled nearly half
  its window with copies of the DC bin, the largest value in the spectrum. The
  floor beside DC therefore read a high percentile of its neighbourhood rather
  than its median, and the line's signal-to-noise collapsed below the gate:
  measured on a synthetic 0.69 MHz line, floor 2953 against a global 1098, and
  no peak reported at all. The running median now pads by reflection, filling
  the window with the spectrum's own low-frequency bins; the same line is found
  at 0.6864 MHz. Pre-existing, and not specific to the new scan — the scan
  rightly declines a line at `f/λ ≈ 0.4`, so the windowed pass is its only
  detector.
- **A fit that stopped on the optimiser's own call limit is now continued.**
  Migrad reports a call-limited fit as a failure even when it was still
  descending on usable parameters; such a fit is restarted from where it
  stopped (up to twice) rather than discarded, so a rich candidate is not
  ranked on a truncated search.

- **The published documentation site stopped rebuilding on every push to
  `main`.** `FitEngine` raised iminuit's "starting value(s) are required"
  whenever it was handed a parameter set with every parameter fixed, tied, or
  a link follower — the `global_fit_wizard_result` screenshot scenario hit
  this on the Ag LF-KT decoupling series, where the wizard's local-only
  refinement stage holds every Global parameter fixed while unlocking only
  the Local ones; on the zero-field run `B_L` was already pinned from run
  metadata, so that run's parameter set came out entirely fixed and
  `global_fit`'s block-separable path handed Minuit nothing to search. A
  fully-fixed parameter set is a legitimate, well-defined fit by
  construction: the objective is deterministic at the given values, so
  evaluating it once *is* the fit. `FitEngine` now packs a valid, successful
  result directly in that case instead of calling Minuit, on the per-dataset
  path that both `fit` and the block-separable `global_fit` use. `Docs Build and Deploy` had
  been red on every push to `main` since the `B_L` pinning change above; it
  is green again.

## [0.17.1] - 2026-08-03

### Fixed

- **15-histogram GPS ROOT files loaded with duplicate transverse groups.** The
  loader's default grouping kept the hardware-combined `Up`/`Down`/`Right`/`Left`
  counters (detector IDs 3–6) as their own groups *alongside* the merged
  `_B`/`_F` halves (IDs 7–14), producing near-duplicate "Up"/"Up 2"-style pairs
  — eleven groups from one run, overflowing the Detector Layout editor's
  eight-slot grid and breaking the NeXus grouping round-trip, where a detector
  cannot belong to two groups. The combined counters are separate electronics
  roads that track the summed halves — and the combined counter on whichever
  cryostat port carries the Mobile detector silently folds `Mob-RL`'s counts
  in, so the split halves are the cleaner analysis view as well as the one
  shared with the 11-histogram era.
  `RootLoader` now leaves a standalone combined counter ungrouped by default
  whenever both halves of its pair are present, so a 15-histogram file loads
  with the same seven groups (`Forw`, `Back`, `Up`, `Down`, `Right`, `Left`,
  `Mob-RL`) as the 11-histogram 2026-era export, with detectors 3–6 still
  loaded and individually selectable. `GPS-RD15` (`_build_gps_combined_and_subdetectors`)
  is realigned to match: its presets now group the halves (`Up = (7, 8)`,
  `Down = (9, 10)`, `Right = (11, 12)`, `Left = (13, 14)`), and the combined
  counters and `Mob-RL` stay ungrouped but selectable. Runs loaded before this
  change will not co-add with runs loaded after it until regrouped, since the
  co-add compatibility check includes group names; saved `.asymp` projects are
  unaffected, because a stored grouping override or profile takes precedence
  over the loader default.

## [0.17.0] - 2026-08-03

### Added

- **The fit wizard now sees heavily damped oscillations, blind.** An
  oscillation whose envelope dies in the first tens or hundreds of nanoseconds
  lives entirely in the leading fraction of the record, and every symmetric
  apodisation window is zero at the first sample — so both of the wizard's
  seeding FFTs deleted exactly the region such a line occupies. A damped-cosine
  recommendation was therefore only reachable by handing the wizard the
  frequencies through `user_frequencies_mhz`; run blind, it would recommend a
  Kubo-Toyabe or a multi-rate relaxation and never offer a precession candidate
  at all. `analyze_dataset_peaks` now runs a second, **unwindowed early-window
  pass** over a short ladder of leading crops of the SNR-truncated record
  (`analyze_early_window_peaks`), each first-order detrended and each carrying
  its own honest `resolution_mhz = 1/T_crop`. Its lines are merged into the
  peak set (`merge_early_peaks`) under an explicit policy: the windowed pass
  wins any collision within the coarser of the two resolutions, because for a
  line it can see at all its frequency estimate is the better one, so
  early-pass peaks are only ever *additions*; SNRs are never compared across
  passes; and `max_peaks` survives the merge with two slots reserved for early
  additions, so a full windowed peak set cannot starve the pass that exists to
  see what it cannot. Peaks carry `source="early_fft"` and the crop that found
  them (`DetectedPeak.crop_us`), appear in the wizard's peaks table, are drawn
  on the FFT plot with a dotted marker (that plot is the *windowed* transform,
  so the line is legitimately not visible under its own marker), and never
  reach the Burg cross-check — Burg is unreliable on short damped windows and
  would veto exactly the lines this pass exists to find. The wizard consumes
  them exactly as it consumes user frequencies: the fingerprint carries the
  candidate (`damped_line_frequency_mhz` / `_snr` / `_crop_us`) so precession
  is no longer gated *off* by a windowed view that cannot contain the line, and
  the crop supplies the envelope rate — without which the true damping sat
  outside the fit's own `Lambda` bounds, the blind guess being a slope over the
  leading 5 % of the record, which on such a signal measures the slow tail. The
  threshold admitting an early-pass line is derived against that pass's own
  null rather than inherited: across 1400 draws of pure noise, relaxing tails
  without oscillation, and conventional narrow lines it contributes zero peaks,
  while a damped line is found in 40/40 draws at envelope rates from 2 to
  100 µs⁻¹. Beyond roughly 120 µs⁻¹ the line is not recoverable at any crop and
  no damped-cosine candidate is offered — a documented limit, not a silent
  failure.
- **Transform a bare `(t, A, σ)` triple: `fft_arrays()`,
  `fft_complex_arrays()`, `prepare_fft_arrays()`.** Every Fourier entry point
  took a `MuonDataset`, so transforming a curve that is not a dataset's default
  reduction — a custom detector pairing, a detrended or background-corrected
  export, a residual, a hand-built model curve — meant synthesising a dataset to
  carry it, the same construction `FitEngine.fit_arrays` exists to remove on the
  fitting side. The three new front doors match `fft_asymmetry`,
  `fft_complex_asymmetry` and `prepare_fft_time_signal` keyword for keyword
  (`window`, `padding_factor`, `t_min`/`t_max`, `phase_degrees`,
  `t0_offset_us`, `subtract_average_signal`, `filter_start_us`,
  `filter_time_constant_us`, `fractional`, `amplitude_calibration`,
  `diagnostics`) and share their entire implementation: the array-consuming body
  of the preprocessing was extracted into one internal core, both paths crop
  through one seam, and the spectral half (padding, coherent-gain calibration,
  phase rotation, diagnostics stamp) is likewise one core. Equivalence is exact
  and pinned by test with no tolerance — identical frequencies, real part,
  magnitude and complex spectrum across every kwarg combination worth testing.
  `error` is optional on the array path (the error-weighted average subtraction
  and fractional baseline fall back to their unweighted forms without it), and
  the array path validates what a dataset would otherwise have carried
  implicitly, raising `ValueError` naming both the offending array and the entry
  point for a non-1-D shape, a length mismatch, empty input, non-finite values,
  or a crop that leaves no points. The dataset front doors keep their signatures
  and behaviour. The FFT is scale-linear — percent in, percent-scaled spectrum
  out — and the new reference section says so, cross-linking "Asymmetry units
  across the API".
- **Remove a trend, not just a mean, before the FFT: `detrend=`.**
  `subtract_average_signal` removes a constant, which is the wrong baseline for
  a signal that relaxes across the record: the trend it leaves behind dumps
  power into the low-frequency bins, where it can swamp a genuine line. The new
  `detrend=` argument on the shared prepare core — so it reaches the dataset and
  array front doors alike — takes `None` (today's behaviour, unchanged), an
  integer polynomial order `0`–`3` fitted **error-weighted** to the cropped,
  post-fractional signal, or a callable `f(time) -> baseline` subtracted
  verbatim so a caller can remove a slow model fitted elsewhere. It is applied
  *instead of* the mean subtraction rather than after it, since a fitted
  polynomial subsumes the mean; `subtract_average_signal` therefore now defaults
  to `None` meaning "the historical `True`, unless `detrend` takes over", and
  passing `detrend=` together with an explicit `subtract_average_signal=True`
  raises rather than guessing an order for the two. Orders above 3 are rejected:
  a polynomial flexible enough to follow the oscillation removes *it* rather
  than the trend it rides on.
- **Reconstruct MaxEnt from count arrays: `maxent_from_counts()`.** MaxEnt fits
  a forward model of the raw detector counts — weighting every bin by its own
  Poisson error and carrying a phase, amplitude and background per detector
  group — so it needs a `Run`'s histograms, and an asymmetry curve has already
  destroyed exactly those statistics. That boundary was real but undocumented,
  and there was no supported way to drive the algorithm from counts that are not
  inside a `Run` (a re-grouped detector sum, a simulation, counts reconstructed
  from an export): only the low-level `MaxEntInput` / `MaxEntGroupInput` /
  `run_cycles` path. `maxent_from_counts(counts, config, bin_width_us=…,
  t0_bin=…, field_gauss=…, alpha=…, group_names=…, run_number=…)` takes per-group
  count arrays (a mapping or a sequence) plus exactly the count-domain
  calibration the arrays cannot carry; everything else stays in `MaxEntConfig`,
  which is already the algorithm's configuration object. `σ = √N` is derived
  from the counts and deliberately not overridable — a σ that is not the
  counting error breaks the algorithm's premise. It shares `maxent`'s whole
  implementation: the per-group normalisation, interior exclusion, frequency
  window, phase seeding and pulse response moved into one core behind a
  `GroupCountSignal` seam, and the auto-steering and frequency-window resolvers
  now take the field value rather than reaching into a `Run`. The same counts
  therefore give the same spectrum, pinned bit for bit by test across time
  windows, binning, exclusions, the automatic window, group selection, phase
  seeding, a non-zero t0 bin and ZF/LF mode. `build_maxent_input_from_counts` is
  exported alongside it, and the docs and both docstrings now state plainly that
  there is no asymmetry-domain MaxEnt and point asymmetry-curve callers at
  `fft_arrays`.
- **Fit a bare `(t, A, σ)` triple: `FitEngine.fit_arrays()`.** `FitEngine.fit`
  only accepted a `MuonDataset`, so fitting a curve that is not a dataset's
  default reduction — a custom detector pairing, a background-corrected export,
  a hand-assembled model curve — meant smuggling arrays through
  `dataclasses.replace(ds, time=…, asymmetry=…, error=…)`, which requires having
  some dataset to clone and puts every use one slip away from mismatched errors
  or scales. The new `fit_arrays(time, asymmetry, error, model_fn, parameters,
  …)` takes exactly the same keyword arguments as `fit`
  (`t_min`/`t_max`/`method`/`minos`/`cost_factory`/`migrad_kwargs`/
  `frequency_offsets`/`error_oversampling`/`cancel_callback`) and shares its
  entire implementation: the array-consuming body of `fit` was extracted into one
  internal core, and both front doors clip their window through one seam, so the
  paths cannot drift. Equivalence is exact and pinned by test — same parameter
  values, uncertainties, covariance, χ², dof and residuals, with no tolerance.
  `fit` keeps its signature and behaviour unchanged. Unlike `fit`, `fit_arrays`
  validates what a dataset would otherwise have carried implicitly and raises
  `ValueError` with a message naming the offending array for a non-1-D shape, a
  length mismatch, empty input, non-finite values, or a `t_min`/`t_max` window
  that leaves no points. One documented difference: bare arrays carry no run
  metadata, so the advisory fixed-frequency guard (which needs `field`) cannot
  run on that path, while the percent-versus-fraction scale guard does.
- **Asymmetry units are now explicit across the API.** The library carries two
  internally-consistent asymmetry scales — the dimensionless fraction
  `A ∈ [-1, 1]` returned by `compute_asymmetry`, `binned_fb_asymmetry` and
  `integrate_asymmetry`, and percent (0–100) held by `MuonDataset.asymmetry`,
  `reduce_grouped_asymmetry` and every built-in model's `A0` — but nothing at a
  call site said which was which, making a mix-up a silent factor of 100 that
  reads as a plausible result (a 20 % amplitude reported as 0.2 %). A new
  `asymmetry.core.transform.units` module names the distinction: an
  `AsymmetryUnit` enum (`FRACTION`/`PERCENT`) with `ASYMMETRY_FRACTION` /
  `ASYMMETRY_PERCENT` aliases, `PERCENT_PER_FRACTION`, and the
  `convert_asymmetry` / `to_percent` / `to_fraction` helpers, all re-exported from
  `asymmetry.core.transform`. Deliberately two members and no "automatic" one: a
  scale is a property of the producing function, not something to infer from
  magnitude. The two result containers on opposite sides of the divide now record
  their scale — `GroupedAsymmetryReduction.units` (always percent) and
  `FieldScan.units` (fraction for a scan from `build_field_scan`) — defaulted to
  the current truth, so no existing return value or caller changes. Every
  asymmetry-valued public function across `core.transform` and `core.fitting` now
  states its scale on the first line of its `Returns`, including the ones that are
  deliberately scale-*preserving* (`rebin`, `integrate_curve`,
  `differentiate_scan`, the RRF demodulation) rather than fixed to one unit, and a
  new reference page, "Asymmetry units across the API", is the whole map with a
  per-function table. Pinned by tests that assert the exact scale of each surface
  on one synthetic pair whose asymmetry is exactly 0.2.

- **GPS 15-histogram ROOT layout (`GPS-RD15`) and validated preset
  resolution.** 2022-era GPS MusrRoot exports carry *both* views of the same
  events — the six combined counters (`Forw, Back, Up, Down, Right, Left`)
  *and* the eight `_B`/`_F` half-counters plus the mobile detector, 15
  histograms in all, with the combined counters being the t0-aligned sums of
  their halves (the mobile detector arriving inside whichever transverse
  counter's cryostat port it occupies). A new `GPS-RD15` layout joins the GPS
  variant family (shared display name "GPS"): its presets group the combined
  counters only — the same transverse ids as the 6-detector BIN layout — since
  adding the halves would double-count, while ids 7–15 stay individually
  selectable for custom groupings. Detection (`detect_instrument` /
  `_gps_variant` / `variant_for_histograms`) now distinguishes all three GPS
  exports, count-first with a most-specific-first label fallback. A new public
  helper `preset_grouping_for_run(layout, preset_name, n_histograms=…,
  labels=…)` is the supported way to turn a layout preset into a concrete
  grouping: it validates the layout against the run's histogram count and
  labels first and raises the new `PresetLayoutMismatchError` on mismatch,
  with `layout_detector_labels()` exposing the layout's per-id label
  expectations. The PSI loader routes its on-load default presets through it.
- **Arbitrary per-group synthetic signals and a continuous octant-ring
  template.** `asymmetry.core.simulate.simulate_signal_run()` synthesises a
  run from a caller-supplied `a_g(t)` per detector group — a callable or an
  array pre-sampled exactly on the group's post-t0 grid, not just a
  registered fit model — with an optional per-group background
  (`background_per_bin` as a `group_id -> counts/bin` mapping) and a hard
  failure instead of a silent floor when a signal would drive the expected
  counts negative (`expected_counts()`'s new `on_negative_expected="raise"`).
  A new built-in template, `"ideal_continuous_ring8"`
  (`BUILTIN_TEMPLATES`), supplies an 8-group, 45°-spaced continuous-source
  ring — the geometry ingredient for PSI HAL-9500-style high-transverse-field
  data — recording each group's azimuth for `group_azimuths_deg()` to read
  back; `build_builtin_template()` gained `n_bins`/`bin_width_us`/`t0_bin`
  overrides so any built-in's binning can be resized from a script without a
  bespoke `InstrumentTemplate`.
- **Component resolution / identifiability diagnostic for composite fits.**
  New `asymmetry.core.fitting.resolution.assess_component_resolution()` judges
  every relaxation-rate parameter of a fitted composite model against the window
  the data can actually resolve: two-sided, so a branch railed past the binning
  (1/e time inside the leading bins) and one collapsed slower than the fit window
  (degenerate with a free baseline) are both caught. The verdict is taken over
  the Δχ² ≤ 1 neighbourhood rather than the point estimate — a point-estimate
  rule gives optimiser-dependent answers on a flat ridge — which yields a third,
  honest outcome, `undetermined`, distinct from resolved and railed. The
  neighbourhood is probed by pinning each rate just outside the edge and
  re-fitting, or taken from multistart solutions the caller already has. The fit
  wizard runs it on every candidate carrying two or more rates and disqualifies
  those that fail, so ranked tables no longer reward an extra component that only
  buys χ². Reported from a downstream continuous-source ZF/LF analysis, which had
  to hand-roll the rule twice.
- **Effort tiers now buy a cheaper global-fit-wizard screen.** `effort_tier=` on
  `build_global_fit_wizard_screening_recommendation()` (and on
  `build_or_complete_single_fit_wizard_recommendations_for_global_portfolio()`)
  trims the screened candidate portfolio: `LOW` and `BALANCED` drop the
  numerically expensive Kubo-Toyabe candidates and cap the portfolio, while
  `THOROUGH`/`EXHAUSTIVE` — the default — screen everything exactly as before.
  Pattern-matched candidates are never dropped, and whatever is skipped is logged
  and recorded in `instrumentation["screening_skipped_template_keys"]`. On a
  synthetic 14-dataset series that had not returned inside a 40-minute cap, `LOW`
  now returns in ~2 s and `BALANCED` in ~20 s.
- **Standard timing instrumentation for both wizards.** New
  `asymmetry.core.fitting.wizard_timing`. Passing an `instrumentation` dict to
  `build_fit_wizard_recommendation()` or
  `build_global_fit_wizard_screening_recommendation()` fills in a `"timing"`
  block with wall-clock, CPU seconds (this process **and** its reaped pool
  workers) and a per-stage breakdown; a new `stage_callback=` receives
  `WizardStageProgress` events as each stage starts, advances and ends. A caller
  can now tell a slow run from a hung one, and time out on absence of progress,
  without reading the child process tree's CPU out of `/proc`.
- **Convergence-quality flags on wizard candidates.** After ranking, the top few
  candidates are re-fitted with the full seed ladder; the better fit is the one
  reported and the χ² gained is recorded as
  `CandidateAssessment.refinement_delta_chi_squared`, with `under_converged` set
  past one χ² unit — the measured statement that a ranking was search-limited.
  `refine_top_candidates=0` disables the pass.

- **Background-aware uncertainty for the ratio α estimate.**
  `estimate_alpha_detailed(method="ratio", ...)` takes a new optional
  `subtracted_background=` (a `SubtractedBackground`), and
  `corrected_grouped_counts` now carries the constant level it removed plus that
  level's standard error, exposed as
  `CorrectedGroupedCounts.subtracted_background()`. Feed one to the other and the
  reported σ_α accounts for the subtraction; the grouping window's α card does
  this automatically. Reported from a downstream continuous-source ZF/TF
  analysis, where the ratio σ on a late, low-counts window came out roughly a
  factor of two small.

### Fixed

- **The published documentation gets its GUI screenshots back.** Every docs
  deploy since 2026-07-18 shipped only the first two or three screenshots. The
  capture driver runs scenarios alphabetically under
  `QT_QPA_PLATFORM=offscreen`, and `alpha_calibration_dialog` ends by closing a
  `GroupingDialog` whose `closeEvent` runs the unsaved-changes guard — a modal
  `QMessageBox.question("Discard changes", …)` that the α estimate had made
  reachable. Offscreen nobody can answer it, so the process blocked there until
  the 8-minute watchdog hard-exited, and the ~44 scenarios sorting after it were
  never written. The capture harness now wraps the whole run in
  `auto_dismiss_modals()`: inside a capture, every `QMessageBox` answers
  immediately with its default (most dismissive) button and logs one
  `[screenshots] auto-dismissed modal: …` line, so any future scenario that
  trips a dirty-state guard degrades to a log line instead of a dead build. The
  two calibration scenarios additionally discard their draft explicitly before
  closing — `beta_calibration_dialog` turned out to carry the identical latent
  hang.

- **An apodisation window that deletes the early-time signal now says so.** The
  `"hann"` and `"cosine"` windows are symmetric tapers: exactly zero at the
  first sample, rising as `(t/T)²`. They are for late-time / narrow-line work,
  and on a heavily damped oscillation — lifetime a small fraction of the record
  — they multiply away the entire signal, so a line that is many standard
  deviations significant with no window reads as flat noise with one. Nothing
  warned about this, and `suggest_matched_apodisation` cannot detect it because
  it reasons from the spectrum only and so sees an envelope that is already
  dead. The shared FFT prepare core now measures, after apodisation, what share
  of the pre-window error-weighted power lies in the leading 15 % of the record
  and what share of that the window keeps; when the power is concentrated early
  (≥ 75 %) *and* the window removes most of it (keeps ≤ 20 %) it raises the new
  `ApodisationEarlySignalWarning`, naming both numbers and both alternatives —
  `window="none"` with a crop, or the exponential (`"lorentzian"`) filter at
  `filter_time_constant_us ≈ 1/λ`, which is weight-1 at `t = 0` and is the
  matched apodisation for a Lorentzian line. No returned value changes, and the
  guard reaches the dataset and array paths alike because it lives in the core
  they share. It measures power **above the noise floor** — `|s/σ|²` minus the
  expectation noise alone would produce — not raw power. That matters for more
  than robustness: raw power carries a `+1` per bin from noise, so the pedestal,
  and with it the dilution of the concentration ratio, grew with however finely
  the record happened to be binned, and the same data could pass or fail the test
  purely on its binning. Subtracting the noise expectation removes that
  dependence (a real signal's excess is invariant under rebinning), and the
  verdict is now measured to be identical across a sixteenfold binning sweep for
  both firing and non-firing cases. Because individual bins can fall below the
  noise expectation, the total can be consistent with zero; a significance gate
  (`4√(2n)`, the χ²₁ fluctuation scale) then makes the guard report
  **`"insufficient_statistics"`** — a third verdict state, distinct from a
  confident `"clear"` — rather than reading a ratio off a sum indistinguishable
  from noise. `EarlySignalApodisationLoss` gains `verdict`, `excess_power` and
  `significance_floor`, with `triggered` and `assessable` as derived properties.
  Pure noise now fires 0 times in 1500 draws across three binnings, all correctly
  reported as unassessable, and the per-bin signal-to-noise floor drops from ~8 to
  ~3.5. The test is applied **twice** — once to the signal's own excess and once
  to the excess left after a slow baseline is subtracted — and fires if either
  pass does. The second pass is what makes the guard work on real data: an
  ordered-state record is never a bare damped cosine, its oscillation rides on a
  slowly relaxing tail (the powder tail) whose residual curvature no constant or
  low-order baseline removal takes out, and that tail's power spreads across the
  record and *dilutes* the concentration statistic. A record whose taper had kept
  a ten-thousandth of the oscillatory content still measured 0.41 on the
  whole-signal statistic and stayed silent; on the high-passed one it measures
  0.998. The slow baseline is an error-weighted moving mean, kernel-width the
  early window, **odd-padded** at both ends: with a truncated kernel the mean at
  the first sample is the mean of the following half-kernel, a systematic
  `slope · k/4` error landing inside the early window that made a plain straight
  line — no curvature at all — leave a residual concentrated 80 % early and trip
  the guard. The high-passed pass is skipped entirely when its residual carries
  under the significance gate, so subtraction dust can never decide. That
  padding anchors on a locally fitted intercept rather than the endpoint sample:
  reflecting about one noisy sample doubled its variance into the whole pad and,
  when that sample sat on an oscillation peak, amplified the line's apparent
  power by 2.7×. The thresholds come from a synthetic study documented and pinned in the tests: the
  binding constraints are the exponential filter, which keeps ~½ of the early
  power at the matched `τ = 1/λ` and a third even at `τ = 0.5/λ` and must not trip
  the warning it is recommended by, and realistic `1/σ²` error growth, which
  tilts the concentration statistic to 0.43 on an *undamped* record before any
  damping at all. Noise sensitivity improves as a side effect, from a per-bin
  peak SNR of ~16 to ~3.5, and no longer depends on whether a tail is present.
  One limit is documented rather than papered over: a weak, heavily damped line
  on a finely binned record can be clearly visible in an unwindowed spectrum — the
  FFT sums it coherently, gaining √N — while carrying too little excess power for
  any time-domain statistic to judge. The guard reports insufficient statistics
  there, and the docs say plainly that its silence is not an assurance.
  Every apodisation-adjacent docstring and the Fourier reference page now
  separate the weight-1-at-`t = 0` filters from the tapers and state the failure
  mode concretely, and `suggest_matched_apodisation` carries the caveat that it
  cannot see a dead time-domain envelope. The fit wizard's and peak detector's
  internal seeding FFTs suppress the warning — the window there is theirs, not
  the user's — and now go through `fft_arrays` instead of synthesising a
  dataset.

- **`mu_relaxation_from_amplitude` documented `reference_amplitude` as a
  "fraction".** The code has always pinned `A_Mu` on the percent scale — it is
  normally handed `MuRelaxationSeriesResult.shared_amplitude`, which is percent —
  so the wording invited a factor-100 seeding error. Documentation only; no
  behaviour change.
- **A percent-scale ALC scan was built into a fraction-scale container.**
  `MainWindow._alc_display_scan` multiplies by 100 and so returns a percent-scale
  `FieldScan`, unlike every `build_field_scan` result. It now declares
  `units=ASYMMETRY_PERCENT` (as do the derivative slice and
  `fit_scan_baseline`, which carry their input's units through) rather than
  inheriting the field's fraction default and misreporting itself. No numbers
  change.
- **Pre-t0 background windows were not trimmed to whole accelerator periods for
  core-API callers.** `apply_grouped_background_correction`,
  `corrected_grouped_counts` and `reduce_grouped_asymmetry` took a `facility`
  string defaulting to `""` — which means "no facility, no beam-period trimming",
  silently. The GUI reads the label off the run metadata and passes it, so GUI and
  script reductions of the same PSI or TRIUMF run quietly disagreed. The default
  is now `None`, meaning *derive it from the data*: the two reduction functions
  take a new `metadata=` argument (they receive histograms rather than a run, so
  they cannot fetch it themselves) and resolve the label through the new
  `transform.resolve_facility()`, which is now the one place that resolution
  lives — the four duplicated `metadata.get("facility", metadata.get("instrument",
  …))` expressions across the core and GUI call it instead. `facility=""` stays
  available as the deliberate opt-out and any non-empty string still overrides, so
  every existing call site is unchanged.
- **`fit_fb_alpha` converged cleanly on a spurious minimum when `N0` was seeded
  generically.** The right `N0` is a per-run count level, and a caller driving the
  forward/backward count fit directly — as you must to free the muon lifetime,
  which `estimate_beta_detailed` does not expose — had no way to know it: only the
  β-calibration wrapper's private first-good-bin seeder did. With a generic seed
  the fit did not fail but walked to a tiny α with a compensating amplitude
  balance and reported it as converged, with credible errors. `N0` may now be
  **omitted**, or given the new `count_domain.SEED_FROM_DATA` sentinel, and is
  then seeded from the fit window's own counts (first positive bin lifted back to
  t = 0 — the same rule the wrapper used, which now relies on it too). A fixed
  `N0` is never overruled. Because that spurious minimum is a genuine minimum, a
  converged fit is additionally checked against the data: α within a factor of
  three of the window's forward/backward count ratio, and `N0·√α` within two
  decades of the observed count level at t = 0. A violation returns
  `success=False` with a message naming the discrepancy — also appended to each
  group result's `warnings` — instead of a clean-looking convergence, and
  `estimate_beta_detailed` now passes that message through rather than reporting
  "did not converge". Neither check applies to a parameter the caller fixed.
- **`estimate_alpha_detailed(method="general")` reported a badly wrong alpha as a
  good one.** The closed-form flatness solution assumes the counts decay at
  exactly τ_µ, and the two-window equation cannot tell a detector imbalance from
  counts that decay slightly faster or slower — so an effective lifetime a few
  tenths of a per cent off τ_µ (pile-up rejection, or a marginally
  over-subtracted background) was absorbed entirely into alpha and returned with
  `ok=True` and a small bootstrap error. A 0.5 % lifetime error shifts alpha by
  10 % at a 20 % asymmetry and by tens of per cent on weakly asymmetric data. The
  solution is now gated on the flatness it assumes: flatness is re-tested across
  eight equal-statistics sub-windows *with alpha re-freed* (testing the trend at
  the solved alpha does not work — the solver's own two-window condition cancels
  exactly that trend), and what survives is read back as an effective lifetime. A
  deviation above 0.15 % that is resolved at 2.5σ now returns `ok=False` naming the
  deviation and pointing at the count ratio on a fully depolarised window.
  `objective_value`, previously `None` for this method, carries the flatness χ²
  either way. The reported alpha is unchanged for data that passes. Reported from
  a downstream continuous-source analysis where the method returned 2.4× and
  1.14× the true balance, silently.
- **15-histogram GPS ROOT exports were silently mis-grouped.**
  `_labels_match_gps_subdetectors` matched on the presence of the half-counter
  labels alone, so the 15-histogram export (which carries the combined *and*
  half-counter labels) resolved to the 11-detector `GPS-RD` layout — whose
  positional detector ids mean different counters. Its
  `Spin-rotated (B+U/F+D)` preset then summed the **Down** counter into the
  "up" group (and Right/Left into the backward group) with no error and a
  plausible-looking asymmetry. The sub-detector matcher now requires the
  combined transverse labels to be *absent*, the 15-histogram export resolves
  to the new `GPS-RD15` layout, and applying a preset from a layout that does
  not describe the run raises `PresetLayoutMismatchError` instead of grouping
  wrongly (see Added).
- **PSI temperature-log sidecars: three bugs that silently lost or mismatched
  `.mon` files.** (1) Sidecar names are zero-padded (`run_0534.mon`), but the
  matcher compared filename text against the unpadded run number with a
  digit-boundary guard, so the leading `0` blocked every run below the padding
  width — runs under 1000 never found their logs. Matching is now numeric, so
  padded and unpadded spellings agree while `run_1554.mon` still stays clear of
  run 554, and where a name carries an explicit `run` token only the digits
  attached to it count, so variant suffixes (`run_0534_2nd.mon`,
  `run_0534_variox0.mon`) can no longer masquerade as runs 2 and 0. (2) The MDU
  reader never looked for sidecars at all, so `.mdu` runs lost their temperature
  logs outright; it now runs the same discovery and merge as the BIN reader.
  (3) PSI run numbers restart each beam period, so a `tlog` directory can retain
  a same-numbered sidecar — matching even in its own `Start of Run` line — from
  an earlier experiment. Each candidate's start timestamp is now cross-checked
  against the run's started/stopped window with a week of slack; a sidecar
  outside it is rejected and listed with its reason in the new
  `psi_temperature_log_rejected` metadata, rather than loading another
  experiment's temperatures under this run's name.
- **Global fit wizard: phase-1 screening used a fraction of the host.** The
  per-dataset single-fit tables — independent, minutes-long, CPU-bound jobs —
  were capped at four workers regardless of core count, so a 14-dataset series on
  a large machine ran at well under one core while looking stalled. The cap is
  now the host's core count (the wavefront's `_MAX_TEMPLATE_WORKERS` cap sizes
  template-level fan-out and never applied here); measured utilisation on a
  14-dataset synthetic series went from ~0.65 cores to ~12.
- **Longitudinal-field Lorentzian Kubo-Toyabe evaluation was ~1.6× slower than
  needed.** The line-shape kernel — the dominant cost of any wizard candidate
  carrying a dynamic Lorentzian KT, and ~88 % of screening CPU on a broad
  zero-field scope — computed `sin`/`cos` over its full (frequency × time) grid
  four times per call instead of twice, and re-derived the inverse powers of `t`
  each time. Fused into one pass per integration limit; results are unchanged to
  floating-point round-off.
- **A global-fit-wizard screen that scored nothing looked exactly like one that
  scored everything.** Screening always returns `recommended_key=None` by
  design, and the summary said the same thing whether every candidate had been
  ranked or none could be scored at all — so a caller reading `recommended_key`
  proceeded silently with an empty selection. The summary now names how many
  candidates scored and which ranks first, and states plainly when a screen
  failed, with the per-candidate causes.
- **Global fit wizard: screening could not be interrupted and looked hung.**
  Each phase-1 per-run single-fit table is a minutes-long job, and the drain
  that collected them blocked on completion without polling `cancel_callback`,
  then tore the pool down with a *blocking* shutdown — so neither Cancel nor
  Ctrl-C returned control, and a caller that gave up and killed the process left
  the spawn workers orphaned. `build_global_fit_wizard_screening_recommendation`
  now polls cancel while it waits, force-kills the pool on cancel/error/Ctrl-C,
  logs each completed table so the stage visibly advances, and no longer starts
  more workers than the host has cores.
- **Unguarded scripts no longer crash or re-run themselves in the fit wizards.**
  Under `spawn`, every worker re-imports the caller's `__main__`, so a script
  with no `if __name__ == "__main__":` guard re-ran its whole body in each worker
  — either tripping multiprocessing's bootstrap `RuntimeError` or silently
  repeating the analysis N times over. The shared pool helper now detects that
  (and the nested re-import that would cascade), degrades to serial execution,
  and warns once with `SpawnUnsafeWarning` naming the fix. `max_workers=1` is
  now guaranteed never to start a worker process.

### Changed

- **Seeding parity: the fit wizard's seed ladder now scales with a candidate's
  rate dimensionality.** Every Stage-2 family previously got the same fixed seed
  count, which covers a one-rate family's whole search space but a vanishing
  slice of a three-rate family's — so multi-component candidates were
  systematically compared at shallower effective depth than the simpler models
  they were being ranked against. The budget now grows with the number of
  relaxation rates, and the extra seeds are wider rate *separations* (the
  mixture ladder spans up to a 50× ratio, was 3×), because every other seed
  variant scales all rates by the same factor and therefore leaves the rate
  separation — the dimension a multi-component fit exists to determine —
  unexplored.
- **The ratio α uncertainty is now a closed-form propagation, not a bootstrap.**
  It is exact for a ratio of window sums, deterministic, and is the only form
  that can carry a background subtracted before the call. On unsubtracted counts
  it agrees with the previous bootstrap (α·√(1/ΣF + 1/ΣB)); `n_bootstrap=0` still
  suppresses `alpha_error`. Passing `subtracted_background=` with any method
  other than `"ratio"` raises `ValueError`.

- **Faster dynamic Kubo–Toyabe fits.** The strong-collision Volterra solver
  behind the dynamic Gaussian/Lorentzian KT components now solves its
  trapezoidal discretisation as a blocked lower-triangular Toeplitz system
  (FFT cross-block coupling, dense within-block triangular solve) instead of an
  O(n²) Python recursion. On a 20001-point grid this is a >20× speedup at
  machine-precision equivalence, removing the model layer's hottest inner loop
  as a bottleneck during fits where ν or the width are free.
- **Substantially faster vortex-lattice lineshape fits.** `R(t)` for the
  superconducting vortex-lattice components (`VortexLattice`,
  `VortexLatticePowder`) was the characteristic function of the cached `p(B)`
  histogram evaluated as an `n_bins × N_t` complex-exponential matrix — millions
  of `exp` calls rebuilt on *every* model evaluation, because the histogram
  cache keys on λ and a minimiser's line search therefore misses it on
  essentially every call. The bin centres a `np.histogram` returns are uniform,
  so the sum factors into a carrier times a polynomial in `exp(iγΔt)`, which is
  now evaluated as a chirp z-transform on a uniform time axis and as a Horner
  recurrence on any other. Results are unchanged to machine precision (pinned at
  1e-10 against the retained pre-optimisation definition over powder and
  single-crystal cases, scalar/empty/short/long/log-spaced/offset time axes, and
  the degenerate no-lattice case), fitted λ_ab is identical, and peak memory for
  the evaluation drops from `O(n_bins·N_t)` to `O(N_t)`.
- **Parallel block-separable asymmetry series fit.** An F-B asymmetry batch with
  independent (`as_provided`) seeding is embarrassingly parallel — each run is a
  self-contained minimisation — so `fit_asymmetry_series` now fans the per-run
  fits across a spawn-based process pool (the GUI auto-sizes it to the host's CPU
  count), matching the grouped-series solver. Results are byte-identical to the
  sequential path regardless of worker count; a pool that cannot start or an
  unpicklable model/cost transparently falls back to the sequential loop. `chain`
  seeding stays sequential (each run warm-starts from the previous good run with
  trend-based reseeding), and a cancel tears the pool down immediately without
  leaving orphaned workers.

## [0.16.0] - 2026-07-24

### Added

- **Data-driven β estimator.** The β (asymmetry balance) card's "Measure from
  run" section can now fit β from a weak-TF calibration run instead of only
  entering it by hand. **Count fit (recommended)** — the endorsed protocol
  after expert consultation (`docs/porting/beta-correction/`) — fits the
  forward and backward count histograms simultaneously with a single β free
  on the backward amplitude, and reports the fitted α and the α–β correlation
  alongside it; **Single-histogram ratio** instead fits each side
  independently and forms β̂ = Â₀,b/Â₀,f, a statistically weaker but
  independent cross-check. **Apply β** records `calibrated` provenance
  (method, source run, uncertainty) exactly like α, with an **Also update α**
  affordance to also apply the fitted α (stamped `count_fit` for both
  protocols); a fitted β outside the typical 0.5–1.5 range is flagged in the
  result row, and a calibrated β goes stale — amber banner, "· stale" chip
  suffix — when the deadtime or background corrections change afterwards.

## [0.15.0] - 2026-07-19

### Added

- **β (asymmetry balance) correction.** The Corrections column gains a fourth,
  steel-blue **β (asymmetry balance)** card porting musrfit's asymmetry-fit
  (fit type 2) beta — the intrinsic-asymmetry ratio β = A₀,b/A₀,f — applied
  with α as A = (F − αB)/(βF + αB) with exact Poisson errors. β is a fixed
  user-entered scalar (default 1 = the standard formula; the key is only
  persisted when it departs from 1, so existing projects and profiles are
  untouched), gets its own pipeline chip and a **β = 1** compare ghost in the
  preview, and is scalar-only: in vector (per-projection) mode the card hides
  and reductions stay at β = 1. A data-driven β estimate — equivalently a
  fittable β in the count-domain F/B fit — is deferred
  (`docs/porting/beta-correction/`).
- **Multiple grouping profiles per project, with explicit run assignment.**
  Several grouping profiles per instrument can now be in concurrent use —
  typically one per sample, each with its own background, α, and deadtime
  settings — with every run assigned to one profile and reduced under it.
  One profile per instrument is the ★ **default for new runs**; a "Default
  for new runs" checkbox moves it, and saving a profile no longer steals it.
  The Grouping window's scope panel tags each run **follows <profile>** and
  gains an **Assign to ▸** control; applying a profile reaches only the runs
  following it. Profiles can be **renamed in place** (every run's assignment
  follows) and **deleted** (assigned runs move to a profile you choose; the
  last profile of an instrument cannot be deleted). Every non-default
  profile carries a stored **identity colour**, worn consistently by its
  runs' numbers in the Data Browser (tooltip names the profile), the scope
  rows, the editing strip, and the selector swatches — the ★ default stays
  plain black, so colour reads as "off the default" — and the browser's
  **Assign Grouping Profile** context menu makes reassignment a two-click
  action. A released run now records the **base profile** Reattach returns
  it to, named in its ⊗ tooltip. Project schema v17: every dataset records
  its assigned profile; older projects migrate automatically, and
  single-profile projects behave exactly as before.

## [0.14.0] - 2026-07-18

### Added

- **A unified Corrections column in the Grouping window.** The deadtime,
  background and α-calibration modal dialogs are retired: every correction is
  now edited inline, and everything α — the value, its provenance, the
  staleness warning, and the calibration controls (per-projection table in
  vector mode) — lives together in one place. Each correction is a
  collapsible **card** with a live status header ("Deadtime · off",
  "α · 1.2692 · Diamagnetic (TF)"), opening expanded when its stage is active;
  the deadtime card adapts to its mode (the per-detector table appears only
  when it is editable, capped at six rows). Each stage wears an identity
  colour — teal for deadtime, violet for background, red for α — shared
  between its card's stripe and its chip in the **pipeline strip**
  (``Deadtime → group → Background → α``) that heads the editor and makes the
  reduction order visible.
- **Interactive correction compares in the grouping preview.** Focusing a
  stage (via its pipeline chip, or the ◀/▶ **compare pager** above the
  preview) overlays a ghost of the reduction *without* that stage — α = 1 for
  the α compare, which also reports the residual baseline ⟨A⟩ — behind the
  as-reduced curve, plus a compound **Compare vs raw (uncorrected)** view.
  The solid curve is always the full configured reduction and alone sets the
  y-axis, so an off-scale "before" (an uncorrected run can reach ~10⁷ %) can
  never crush the real curve; comparing never changes what Apply writes. When
  a calibrated α goes stale (deadtime/background changed since it was
  measured), the α chip flags **" · stale"** and the α card carries the
  warning banner.
- **Pan, zoom and home controls on the grouping preview**, with a user-chosen
  view preserved across the live redraws until Home restores the automatic
  scale.
- **F−B asymmetry as a Fourier signal source.** The Fourier panel's `FFT Phase
  Mode` section gains a `Signal source` choice: the default `Grouped average`
  (averages each detector group's own lifetime-corrected FFT) or the new
  `F−B asymmetry`, which transforms the run's forward−backward asymmetry
  curve directly — the same signal the time-domain plot shows, with the same
  grouping and α (`GroupSpectrumConfig.signal_source="fb_asymmetry"`; deadtime
  correction is not applied on this path, matching the time-domain plot). For
  a single forward/backward detector pair this gives markedly better
  peak-to-floor contrast than averaging every detector group — roughly 5× on
  a real GPS zero-field run where the grouped average buried the line under
  the other groups' baselines. The `Groups` include/phase table is inert (and
  disabled) in F−B mode, and the resulting spectrum is labelled `<run> F−B` so
  overlays distinguish the source.
- **Pinned frequency-panel annotations that survive a replot.** The frequency
  `PlotPanel` gains `set_custom_x_axis_label(text)` and
  `add_persistent_frequency_marker(freq_mhz, label)` /
  `clear_persistent_frequency_markers()`. A replot clears the axis and
  recomputes the x-axis label, so a bespoke label or Fourier line marker (ν₁,
  ν₂, A_µ, γ_μ·B) drawn straight onto the axis was wiped by the show-time
  render; pinned decorations are now recreated on every render and persist.
- **MaxEnt data-derived phase seeding.** New `Seed phases from data` toggle
  (on by default; `MaxEntConfig.auto_phase_seed`) estimates each group's
  starting phase from a σ²-weighted lock-in at the strongest line inside the
  frequency window (pulse-shape aware). The ±4°/cycle phase refinement can
  never reach the ~90°/45° geometric offsets of real multi-group rings from an
  all-zero start, so unseeded groups previously either poisoned χ² or muted
  themselves out of the joint fit. Hand-editing a phase in the Groups table or
  clicking **Use fitted phases** unticks the toggle so the table drives.
- **MaxEnt workload auto-steering.** New `Auto workload steering` toggle (on
  by default; `MaxEntConfig.auto_steer`, resolver exposed as
  `resolve_maxent_auto_steering(run, config)`) sizes workload settings the
  user left unset to the run: binning is raised until the post-binning Nyquist
  clears the frequency window with margin (only when the window is known from
  the field or explicit bounds — never on data-windowed ZF runs), very large
  post-binning grids get an end-time cap, and the scripting API's default
  spectrum length stops growing with the raw bin count. A raw 389k-bin HiFi
  `.mdu` run that previously implied a ~2¹⁹-point spectrum over 389k time
  points (days of compute, guaranteed workload warning) now reconstructs its
  813 MHz line out of the box in ~2–3 minutes. Explicit values always win;
  the result metadata records what was steered under `auto_steer_applied`.

### Changed

- **The Grouping window's right pane is tabless.** The former "Grouping and
  timing" / "Corrections" tabs are now two side-by-side columns under the
  pipeline strip, with the shared live preview pinned below both — everything
  is visible at once at the new 1220×680 default size, with no scrolling in
  either axis. The preset selector moved to the left column above the group
  table it seeds, the editing-target strip moved into the top row, and on
  windows too small to show everything a compact pill names the sections
  hidden below the fold and scrolls to them on click. The transverse-field
  grouping nudge banner is retired from this window (the Detector Layout
  editor keeps its own copy, including preselecting the recommended preset).
- **The "Large MaxEnt calculation" warning no longer blocks headless
  sessions.** Offscreen/minimal-platform sessions (CI, screenshot scenarios,
  scripted driving) have no user to dismiss the modal, so the warning is
  written to the log panel and the calculation proceeds; setting
  `ASYMMETRY_SUPPRESS_WORKLOAD_WARNING` does the same on a visible display.
  The interactive dialog now also points at `Auto workload steering`.

### Fixed

- **Calibrated α provenance no longer silently degrades to "manual".** The
  recorded full-precision estimate was compared against the six-decimal
  rounded α spin with a near-exact tolerance, so every genuine (non-round)
  calibration immediately read as a hand edit — silently dropping the
  "calibrated" provenance from the profile (including what is saved into
  ``.asymp`` projects) and disabling the staleness warning. Provenance is now
  compared at the spin's display precision, so saved projects keep the
  calibration method, source run and uncertainty, and the staleness warning
  fires when it should.
- The deadtime summary now reports "Deadtime saturates the t=0 correction —
  value too large." instead of a meaningless huge percentage when the
  configured deadtime saturates the correction.
- Zooming the grouping preview could strand it on a garbage view when a
  transient reduction error landed mid-inspection; the chosen view is now
  stored explicitly and survives error/empty redraws.
- **Alpha is now estimated on deadtime- and background-corrected counts.**
  Previously the detector-balance α was estimated from the *raw* grouped counts,
  while the reduction it feeds applies deadtime correction and background
  subtraction first — so a calibrated α balanced the totals and could leave the
  reduced weak-TF asymmetry off-centre once the background was subtracted. Both
  the interactive **Estimate α** dialog and the per-run alpha policy now build
  their forward/backward spectra through the same corrected pipeline the
  reduction uses (deadtime → grouping → background subtraction), for all three
  estimators (diamagnetic, general, count ratio) and including the reference-run
  background mode. The calibration dialog reports which corrections the estimate
  reflects and warns, in amber, when a requested correction could not be applied
  to the selected run — so a flattened "after" curve never misrepresents a
  calibration the reduction will not reproduce. Study:
  `docs/porting/correction-order-alpha-estimation`.
- **Toggling the frequency Overlay checkbox off and back on no longer
  discards a manually-set view window.** The overlay's re-render always
  auto-ranged the axes because switching between the multi-run overlay and
  the single active run reads as brand-new plotted content to the panel's
  first-paint framing. `MainWindow` now remembers the view just before
  leaving an active overlay and restores it when the same combination of
  runs is re-overlaid; a genuinely different run selection still auto-frames.
- **MaxEnt no longer diverges on real long-window TF data.** Three compounding
  engine defects made out-of-the-box reconstructions of e.g. MUSR forward/back
  TF runs collapse into spiky noise with χ² rising from cycle 1 (χ²/N ≈ 8000 on
  a real 400 G vortex-lattice run): (1) the per-group normalisation baseline
  was a plain mean of the lifetime-corrected counts, which the exp-amplified
  late-time Poisson tail inflates several-fold — it is now the 1/σ²-weighted
  mean, pinned to the high-statistics bins; (2) the per-cycle nuisance
  amplitude fit clipped at a floor of 0.01 in units where the correct
  amplitude is ~`A·Δν` (often 10–100× *below* the floor), forcing an oversized
  model oscillation that the solver could only fight by decohering the
  spectrum — the floor is now far below any physical value; (3) the nuisance
  amplitude/background regression and phase scan were unweighted, letting the
  junk tail dominate — both are now σ-weighted, consistent with the χ² the
  cycles minimise. The same run now converges to χ²/N ≈ 1.0 with the peak on
  the vortex line, full range, no manual steering.
- **A converged-but-flagged single fit is no longer silently un-plotted.** When
  the minimiser reaches a usable minimum but flags it (`success=False` — for
  example a degenerate additive baseline at low field), the single-fit panel
  now draws the fit curve **greyed** (via the preview overlay) and explains the
  flag, instead of showing only "Fit failed" and drawing nothing. The fit is
  still not recorded — *Add to Series* and the pull diagnostic stay disabled —
  so the user refines the seeds/bounds/model and refits to keep it. A genuine
  failure (no minimum) still reports "Fit failed" as before.
- **Both periods of a two-period run survive in the data browser.**
  `select_period()` now hands each period a distinct encoded run-number key
  (the same scheme the loader's 3+-period path already uses), so adding both
  the red and green period of one run no longer collapses them under the
  shared source run number. The friendly `run/period` label and the true
  `source_run_number` are preserved in metadata.
- **Weak-TF α-calibration runs are recognised from the structured field-state
  code.** `classify_tf_calibration_run` now also treats an explicit
  `field_state`/`magnetic_field_state` of `TF` as transverse evidence (and
  `LF`/`ZF` as a veto), so a run whose loader recorded the state code but left
  `field_direction`/`field` unset is still highlighted in the calibration-run
  dropdown.

## [0.13.0] - 2026-07-14

### Added

- **Unit-area (field-distribution) FFT display.** A new *Unit area (field
  distribution)* option in the Fourier panel's FFT settings presents a
  magnitude-family spectrum as a field distribution `p(ν)` that integrates to
  one: the noise floor is fitted (a σ-clipped block median) and subtracted, and
  the residual is normalised to unit area over the full frequency range
  (range-independent by construction). A significance guard refuses the
  normalisation for a pure-noise spectrum, keeping the calibrated scale and
  noting why. Applies to the Magnitude, (Power)^1/2 and Power displays. The
  displayed density follows the x-axis unit: in a field view the curve, its
  error band, fit overlays, and the y view window carry the constant dν/dB
  Jacobian — labelled `p(B) (1/G)` / `(1/T)` and integrating to one per
  displayed unit — and exports name the y column per unit (`density_per_G`
  etc.).
- **Field-shift axes for the frequency view.** A new **Axis:** selector above
  the spectrum offers three x-axis modes that genuinely transform the plotted
  data: **Absolute** (today's measured frequency/field), **Shift (x − x₀)** (each
  spectrum minus its reference field, in MHz / G / T), and **Relative shift
  (ppm)** ((x − x₀)/x₀ × 10⁶, dimensionless). A companion **Ref.:** selector
  chooses the reference: **Run field** (the default) shifts each spectrum by
  *its own* applied field, so transverse-field runs measured at different fields
  overlay aligned at zero shift and a paramagnetic/Knight shift between them
  reads at a glance; **Common** shifts every spectrum by a shared, editable Gauss
  value. A run with no field metadata is drawn untransformed with a logged note
  rather than dropped. The plotted axis, tick labels, x-limit boxes, framing,
  the γ_μ·B marker, the moments/fit-range overlays, and the GLE/text export all
  follow the selected mode; shift/ppm exports name their x-column (`shift_G`,
  `relative_shift_ppm`, …) and always keep the canonical `frequency_MHz` column
  alongside.

### Changed

- **FFT amplitudes are now calibrated to fractional asymmetry in percent.** The
  grouped FFT previously plotted raw, unnormalised count-scale amplitudes whose
  height depended on the counting statistics, the length of the time window, and
  the apodisation. The spectrum is now put on a fractional footing (each group's
  signal divided by its error-weighted baseline) and coherent-gain corrected
  (`× 2/Σw`), so a pure cosine of fractional asymmetry amplitude *A* peaks at
  `100·A` — invariant to count level, window length, apodisation choice, and
  zero padding. Y-axis labels read `(%)` (`(%²)` for Power); a relaxing line
  reads below `100·A` because damping trades peak height for area. Spectra
  computed before this change are flagged stale so they recompute onto the new
  scale.
- **The frequency toolbar's "X relative to ref. field" checkbox is replaced by
  the Axis/Ref. selectors above.** The old checkbox only offset the x-limit
  entry boxes while leaving the plotted curve and ticks in absolute units, so
  spectra measured at different fields could not be aligned; the new shift axes
  transform the data itself. Saved projects that had the old flag on migrate to
  **Shift (x − x₀)** about a **Common** reference (schema v16). Per-mode x-limit
  view stashes from the retired flag are ephemeral and are discarded on load;
  the fresh view reframes cleanly.
- **The Fourier panel's "Compute FFT" is now selection-scoped, replacing
  "Apply to selection."** The old secondary button copied the active run's
  already-computed recipe onto the other selected runs, so a setting changed
  after the last **Compute FFT** was silently left out, it recomputed each
  target synchronously on the GUI thread, and it never re-rendered the view —
  nothing visibly happened. There is now ONE compute action: **Compute FFT**
  computes every run selected in the Data Browser (the active run alone when
  nothing else is selected), and its label shows the scope before you click —
  "Compute FFT (3 runs)" for a three-run selection. Every target's
  configuration is read from the panel as it stands right now: the Groups
  table's enabled groups apply to every run in the selection (intersected
  with each run's own available groups, and each target's stored Groups
  table is updated to match), while phases stay per-run. The whole selection
  computes asynchronously, and on completion the workspace switches to the
  frequency view and renders the result (the overlay, when Overlay mode is
  on). A new banner also flags when
  an active overlay mixes spectra computed under different settings,
  independent of the existing out-of-date indicator. The MaxEnt panel's
  "Apply to selection" is unchanged.

### Fixed

- **Matched-apodisation "Suggest from data" now finds lines buried below the
  raw noise floor.** `suggest_matched_apodisation` only ever matched a filter
  to the dominant *raw* peak of the unapodised power spectrum, so a genuinely
  present line whose late-time noise is amplified by the lifetime correction
  (an un-windowed, deadtime-corrected record) stayed below that threshold and
  the suggester reported "No clear line to match — leave apodisation off" —
  precisely the advice that kept the line invisible. Detection is now
  two-stage: the existing raw-prominence check runs first and behaves exactly
  as before when it fires, and only when it fails does a new fallback smooth
  the power spectrum across a range of candidate linewidths and keep the
  width with the highest robust (median/MAD, peak-excluded) signal-to-noise.
  The scanned widths are anchored to the spectrum's real frequency resolution
  (from the caller, or otherwise estimated from the spectrum's own
  autocorrelation) rather than the zero-padded display grid, so the scan
  cannot smooth within a single resolution element and mistake a heavily
  padded spectrum's padding-correlated noise for a line. The smoothing
  kernel's own width is deconvolved from the measured linewidth
  (linearly for Lorentzian, in quadrature for Gaussian) before it is used to
  derive the matched time constant, and the existing resolution-limited guard
  now applies to that deconvolved width.
- **Bundled gleplot bumped to v1.6.1.** The in-app GLE figure editor's live
  preview failed ("GLE error" on the `data` line) on exported figures whose
  sidecar data files are named from a leading run number (e.g.
  `20_main.dat`): gleplot's parser split a digit-led filename after the
  digits and looked for a truncated file. gleplot v1.6.1 parses unquoted
  filenames with digits, hyphens, or path separators correctly; the `gle`
  extra now pins that tag so packaged builds carry the fix.
- **Plot limits no longer reset themselves on the frequency view.** Computing
  or recomputing a spectrum could silently reframe the plot: a same-run
  recompute reset the vertical zoom (only the horizontal window was kept),
  browsing onto a run with no spectrum forfeited a typed window, the first FFT
  overwrote its own freshly framed view with the pre-compute defaults, and a
  pan/zoom gesture — unlike typing — was not treated as a deliberate view
  choice, so it was reframed away on the next redraw. A recompute now never
  reframes; only a genuine content change (a different run, domain, or view
  mode) reframes, and never once you have chosen a window by typing or by
  pan/zoom. The same latching protects the time-domain view, so a zoomed
  time-domain window survives run switches too. Toggling **Auto X** or **Auto
  Y** on remains the explicit "always follow the data" escape hatch and now
  releases any manual lock. On the frequency view **Auto X** frames the
  spectrum sensibly (the dominant line / field-derived window) instead of
  snapping to the full Nyquist span.
- **Frequency-view (FFT/MaxEnt) GLE and text export now mirror the screen.**
  Exporting a frequency-domain spectrum reused the time-domain export path
  verbatim, so the output was wrong in several ways: the axis window was taken
  raw from the display-unit toolbar fields but applied to canonical-MHz data
  (a Tesla-mode export produced a meaningless sliver), the axes were labelled
  `Time (µs)` / `Asymmetry (%)`, and the spectrum was drawn with the
  time-domain error-bar dots. The export now mirrors the on-screen render: x
  data and the exported window are in the current display unit (MHz / Field G /
  Field T, or a reference shift — see the *Field-shift axes* addition above), the
  axis titles are the real spectrum labels, and the spectrum draws as a
  piecewise-linear line (no GLE spline, which overshoots on sharp resonance
  lines) plus a light shaded ±1σ band (omitted when the spectrum has no
  per-point errors). The
  `.dat` sidecars are self-describing — columns are named in the header, the
  canonical `frequency_MHz` axis is kept as a trailing column whenever the
  display unit differs, and a `START OF FOURIER INFORMATION` block records the
  display mode, apodisation/zero-pad settings, axis mode, and reference field.
  The text export's *Limit to current x-range* now filters
  on the display-unit column, and digit-led sidecar filenames (a bare run
  number like `20`) are prefixed with `run_` so the gleplot editor's parser
  accepts them. Time-domain exports are unchanged.
- **Zooming or panning no longer fights the Auto X / Auto Y toggles.** With
  **Auto X** or **Auto Y** active, a rubber-band zoom or pan used to snap
  straight back to the full data extent, because the next redraw re-applied
  the still-active auto-scaling. An interactive zoom/pan now turns off both
  toggles — the same way typing a limit value already did — so the framing you
  dragged to is kept (and, per the entry above, held across run switches until
  you re-enable Auto X/Y).

## [0.12.1] - 2026-07-13

### Fixed

- **macOS app build restored.** The v0.12.0 macOS DMG build failed because
  pyhdf 0.11.7 published its macOS wheels with a `macosx_26_0` platform tag
  (requiring macOS 26), so the build runner fell back to a source build that
  needs the HDF4 C headers. The build constraints now cap `pyhdf<0.11.7`
  until upstream ships wheels installable on older macOS; no analysis
  behaviour changes. (v0.12.0 itself shipped no binaries — this release
  carries the identical feature set plus this fix.)

## [0.12.0] - 2026-07-13

### Fixed

- **PSI MusrRoot files with the new TDirectory-based header layout.** ROOT
  files written with musrfit's TDirectory-based `RunHeader` layout (PSI's 2026
  FLAME DAQ; musrfit ≥ 2025, now the canonical MusrRoot spec) now parse
  correctly — run title, run number, sample, temperature, field, time
  resolution, and short detector names with per-detector time-zero/good-bin
  ranges are read from the header instead of falling back to filename or
  histogram-title guesses. Legacy TFolder-based MusrRoot and pre-2011 LEM ROOT
  files are unaffected. Instrument strings are now matched case-insensitively,
  since the new DAQ writes the lowercase `flame` instrument name.
- **Startup crash loop from a NaN saved plot range.** If a session ever
  persisted a non-finite axis limit (e.g. `plot/freq_y_min = nan`) to the
  application settings at shutdown, every subsequent launch replayed it into
  Matplotlib's `set_ylim` and crashed before the window appeared, with no way
  to recover short of hand-editing the settings store. Non-finite values are
  now rejected at all three layers: the axis-limit fields refuse a NaN
  (`setValue(nan)` keeps the last good value; NaN previously slipped straight
  through min/max clamping), shutdown skips persisting a limit set containing
  a non-finite value, and startup falls back to the per-axis default for any
  non-finite entry already in the settings — so existing poisoned settings
  recover on the next launch.
- **Fit-range spinboxes now commit a programmatically set value.** Setting a
  fit-range field's value in code (e.g. from a scripted/automated scenario)
  used to update only the field's display while the fit kept running over the
  old range — the range is owned by the plot panel, and a bare `setValue`
  never reached it. The Single and Batch (and grouped multi-group) fit-range
  fields now push a driven `setValue` through to the plot's fit range exactly
  as a typed entry does, while the plot→field display mirror stays silent (no
  feedback loop). Interactive editing (type + Return / focus-out) was already
  correct and is unchanged.

### Added

- **RunSummary captured as provenance for PSI MusrRoot files.** The free-text
  `RunSummary` block in the new TDirectory-based header — a block musrfit
  itself does not read — is now attached verbatim to loaded runs as
  `metadata["musrroot_run_summary"]`.
- **Per-axis transforms in the parameter-trending panel.** A collapsible
  **Axis transforms** section (below the Y-parameter list) applies a transform
  to either the X or Y axis — `None`, `1/x  (reciprocal)`, `x²  (square)`,
  `ln x`, `log₁₀ x`, `√x`, or a `Custom…` single-variable expression — so the
  bread-and-butter µSR linearisations are drawable in-app: **Redfield**
  (`1/λ` vs `(µ₀H)²`) and **Arrhenius** (`ln λ` vs `1/T`). The transform feeds
  the trend fit as well as the plot, so a `Linear` Model Fit over transformed
  axes *is* the Redfield/Arrhenius line, with error bars propagated and slope
  and intercept read from the model-fit dialog. It is distinct from (and
  guarded against compounding with) the `log` axis-scale checkboxes.
  Documented in *Reference ▸ Parameter trending ▸ Axis transforms*.
- **Multi-series overlay in the parameter-trending panel.** Shift-clicking more
  than one series pill now overlays those series on the plot (colour = series,
  with a legend), so `σ(T)` at two applied fields, or the same observable across
  two samples, can be compared directly instead of one series at a time. A
  second selected parameter takes a distinct marker shape; the twin-axis layout
  is reserved for a single series. Documented in *Reference ▸ Parameter trending
  ▸ Overlaying several series*.
- **First-class `Quadratic` trend model.** The Model-Fit catalogue gains a plain
  parabola `c0 + c1*x + c2*x²` on every axis, so a steering-curve minimum (or any
  gentle curvature a `Linear` cannot match) no longer needs a `Polynomial` with
  its higher coefficients pinned to zero.
- **Optional CUDA GPU backend for the MaxEnt engine (scripting API).**
  `MaxEntConfig(backend="cuda")` runs the projection kernels on an NVIDIA GPU
  via CuPy instead of NumPy — measured ~160x faster than the CPU path at
  large workloads (16384 time bins x 2^20 spectrum points) on an RTX 3080 —
  or `backend="auto"` prefers the GPU and falls back to NumPy silently when
  one is unavailable. The default remains `"numpy"`, bit-for-bit identical to
  the historical CPU path; the GPU path is float64-only and agrees with it to
  solver tolerance, not bit-for-bit. Install with `pip install
  "asymmetry[gpu]"` (CUDA-13 wheel; `cupy-cuda12x` on CUDA-12 systems). A
  resumed MaxEnt state survives a backend switch. The GUI is unchanged and
  always uses the default CPU backend. Documented in *Reference ▸ Fourier
  analysis ▸ Maximum entropy method ▸ GPU acceleration (optional)*.

## [0.11.0] - 2026-07-11

### Fixed

- **Custom-column labels on the frequency (FFT / MaxEnt) view.** Choosing a
  custom data-browser column — such as an **Angle (°)** field — as the plot
  **Label** now relabels the overlaid Fourier spectra instead of falling back
  to `<run> Average`. The special Angle column (a bare `angle` id rather than a
  `custom:…` id) is recognised as a custom column, and averaged spectra — which
  carry no inline `custom_fields` of their own — resolve the value from the
  run's stored custom columns by run number. The selection also survives a
  project reload. The time-domain and frequency views keep independent
  **Label** choices.
- **Overlaid plots no longer collapse under a tall legend.** With many overlaid
  traces (e.g. a 20+ spectrum angle scan) on a short, wide plot pane, the tall
  legend could drive the layout to shrink the axes to a thin band, squashing
  the plot. The legend is now kept out of the layout solver, so it overlaps the
  plot area (as it already did when there was room) while the axes keep their
  full height.
- **Good-frame count for legacy ISIS HDF4 / NeXus-v1 files.** The NeXus loader
  now reads the good-frame count from `instrument/beam` (`frames_period` per
  period, falling back to `frames_good` / `frames`) when a file carries no
  top-level `good_frames` / `goodfrm` — as ISIS HDF4 originals (e.g. HiFi runs)
  do. Previously `good_frames` silently defaulted to `1.0`, which made the
  file/grouping non-paralyzable deadtime correction overstate the per-frame
  rate by ~5 orders of magnitude and blow up the corrected counts (a ZF fit's
  χ²/dof jumped from ~2 to ~185). The correction is now a benign sub-percent
  adjustment. The same beam fallback applies to the v2 read path.

### Added

- **Data groups are now the vehicle for batch and global fits.** A group owns
  the fit series run over it: choose **Fit this group…** from a data group's
  context menu to bind the fit dock's Batch tab to that group, then untick
  members in the new **Batch members** list to exclude a run from *this*
  analysis without removing it from the group. A series' effective membership
  now tracks its group's live membership (minus its own exclusions) rather
  than a fixed snapshot; when the group changes after a fit, the series'
  trend pill grows a **⚠** ("Membership changed since last fit — re-run to
  refresh.") until it is re-run. Re-running a group-bound series replaces its
  previous results in place. Documented in *Reference ▸ GUI usage ▸ Data
  groups* and *Reference ▸ Parameter trending ▸ Group-bound series and
  staleness*.
- **A run can belong to more than one data group.** **Send to Group** now
  always adds a membership rather than moving the run out of whatever group
  it was already in, so the same run can sit in, say, both a field scan and a
  temperature scan. A run's extra memberships render as marked copy rows
  (①, ② …) beneath its primary one, with an **Also in:** tooltip naming the
  others; selecting any copy reaches the same underlying dataset once, so
  plotting, fitting, and co-add never double-count it. Removing a run's
  primary membership promotes its earliest remaining copy.
- **Ad-hoc batch fits auto-create a data group.** Running a batch or global
  fit over a selection that is not already bound to a group mints (or, for an
  identical run set, reuses) a group named from the run range, e.g. "Runs
  1001–1010", so every recorded batch series has an explicit owner. These
  auto-created groups paint in a red-grey tint, distinct from the blue used
  for groups you name yourself; renaming an auto-created group promotes it to
  an ordinary (blue) group.
- **Ungrouping a group with recorded fits now asks what to do with them.**
  Choosing **Ungroup** on a group that owns fit series opens a prompt to
  **Keep fits** (the series become standalone, frozen analyses) or
  **Delete fits** (remove the group and its series together), with the option
  to cancel.
- **Waterfall display mode for overlaid plots.** A new **Waterfall** checkbox
  next to **Overlay** on both the time-domain and frequency-domain plot
  panels stacks each overlaid trace vertically by a uniform `i * Δ` offset so
  closely-spaced curves stay cleanly resolved. Δ is automatic (1.4× the
  median robust per-trace span, measured over the displayed x-range) unless
  a manual value is entered in the
  adjacent offset field (blank = automatic). Time-domain waterfalls draw a
  faint baseline hairline at each trace's shifted zero; frequency-domain
  waterfalls do not. GLE and plain-text exports mirror the on-screen offsets
  and record a `waterfall offset:` header so the raw values stay recoverable.
  The setting persists in the project file (schema v14). Documented in
  *Reference ▸ GUI usage ▸ Waterfall stacking*.

### Changed

- **Single-tab carry-forward is now refresh-unless-fitted.** Previously,
  selecting a new run always carried forward whatever the form was last
  showing, so a hand-edited draft could silently leak onto an unrelated run.
  Now, a run with a recorded fit result (its own single fit, or its role as a
  batch/global member) is *protected* and always restores exactly that
  fitted state; every other run *refreshes* from the session's most recently
  fitted function instead (with field-dependent parameters such as ``B_L``
  reseeded for the newly-selected run), replacing anything it was showing
  before. A results-box message ("Model carried from run *N* — not fitted
  for this run") makes a carried-forward form impossible to mistake for a
  real fit of the displayed run. Documented in *Reference ▸ GUI usage ▸
  Carrying a model forward between runs*.
- **Project schema v15: data groups and fit series are unified.** Each data
  group gains a ``kind`` (``"user"``/``"auto"``); each run-membered fit
  series gains a structural ``group_id``, per-series
  ``excluded_run_numbers``, and a ``last_fitted_members`` snapshot. Existing
  v14 projects migrate automatically and tolerantly: a series resolves
  ``group_id`` from its old provenance-only ``source_group_id`` only when
  that group still exists in the project, and a group-less series migrates
  to a frozen, standalone analysis rather than sprouting a group nobody
  created. Detector-group (multi-group) series are unaffected. See
  *Reference ▸ Project files* for the field-by-field detail.

### Removed

- **"Share with Group" is gone from the Single tab and the grouped fit
  surface.** It used to copy the current single-fit function and seeds onto
  every other member of the same group, unconditionally overwriting whatever
  they held — including a run that had already been fitted. That behaviour
  is superseded by the carry-forward rework above: a fitted run is now
  always protected from being overwritten, and every unfitted run already
  refreshes automatically from the most recent fit, which is what "Share
  with Group" was mostly being reached for in the first place. Use **Fit
  this group…**'s batch fit when you actually want one model fit across the
  group.

## [0.10.0] - 2026-07-10

### Added

- **Suggest next angle: Bayesian experimental design for a Knight-shift angle
  scan.** Once the Knight shift analysis window's joint K(θ) fit has
  converged, its new **Suggest next angle** section plans the next scan
  angle from three modes: **Refine parameters** (an information-gain sum
  over every fitted curve — c-optimal on one curve's parameter, or
  D-optimal over all of them, with the same precision-goal/events-factor
  conversion as trending's Suggest next point), **Test misalignment** (fits
  the first-harmonic `AngularFourier2` alternative automatically and ranks
  angles by how well they would tell it apart from the current model,
  reporting the Akaike-weighted preference), and **Resolve assignment**
  (ranks angles by how well they would separate the joint fit's winning
  branch labelling from a near-degenerate runner-up, for a crossing the
  classification-EM step could have assigned either way). Candidate spans
  where two curves' predictions risk misassignment are shaded on the
  utility overlay. Documented in *Reference ▸ Parameter trending ▸ Suggest
  next angle*.

- **`AngularFourier2`, a third K(θ) basis model for testing rotation-axis
  misalignment.** `K(θ) = K_avg + K_1·cos(θ − θ1) + K_amp·cos(2(θ − θ2))`: a
  perfectly aligned rotation axis gives a pure second harmonic
  (`AngularCos2`/`KnightAnisotropy`); a tilted axis leaks a first harmonic,
  so a fitted `K_1` significantly different from zero is fit-level evidence
  of misalignment. Available in the Knight shift window's **Model fit**
  selector; needs at least five shared angles per curve.

### Changed

- **The joint K(θ) fit now records each curve's fit covariance**, serialised
  with the project alongside its parameters — previously
  `KnightJointCurve` kept only value/error triples, silently discarding the
  covariance `ParameterModelFitResult` already carried. This is the
  prerequisite for Suggest next angle; a joint fit saved before this change
  has no stored covariance and needs one re-run to gain it.

- **Canonicalising a fitted θ0 (or, for `AngularFourier2`, θ1/θ2) into its
  small-offset representation now folds the fit covariance exactly**
  (Σ' = J Σ Jᵀ) instead of approximating the marginal uncertainties in
  quadrature. The quoted K_iso/K_ax (or K_avg/K_amp/K_1) uncertainties on a
  folded curve are now exact rather than a conservative approximation that
  ignored their correlation.

- **The Fit Parameters panel's "Knight shift window…" button now hides
  unless the fitted model actually has a Knight-convertible component**,
  rather than showing whenever any series is fitted. **Analysis → Knight
  shift analysis…** remains the unconditional entry point regardless of the
  active series' model.

## [0.9.2] - 2026-07-10

### Documentation

- **Documentation screenshot overhaul.** Repaired the captures that showed
  unreadable or misleading state: the EuO oscillatory fit is framed on
  resolved oscillations instead of a solid block, the grouped YBCO count fit
  uses the damped-precession model its data calls for (χ²ᵣ ≈ 1.0, was 64),
  the PbF₂ F–μ–F figure gains a physical relaxation envelope and a converged
  fit, the logbook view actually shows the metadata columns, and the
  FFT/MaxEnt views present their transform settings instead of a contradictory
  fit verdict. Replaced the matplotlib-rendered parameter-trend, apodisation,
  and Knight-shift figures with real GUI captures (the trend panel now
  demonstrates its own model fits). Added first screenshots for eight
  previously image-less pages: quickstart, loading data (run-info
  provenance), simulation dialog, exclusions, count-domain fitting, spectral
  moments, MINOS asymmetric errors, and photoMuSR period mapping. Corrected
  the FFT apodisation default documented in the Fourier page ("None", not
  "Gaussian").

### Fixed

- **Count-domain fits (Multi-Group Fit window, "F + B (free α)" and "Single
  group" targets) can now recover a realistic signal amplitude.** The grouped
  parameter parser pins every model amplitude absent from the visible table to
  `(1, 1)` — correct for the normalised asymmetry fit, where a per-group
  `amplitude` nuisance carries the real value, but count fits have no such
  nuisance and must recover the amplitude directly. The pinned bound was
  leaking into the count-fit seed, clamping the model to a ~1% modulation
  regardless of the true signal and producing a spuriously large reduced
  chi-square for any realistic asymmetry. The count-fit seed now widens that
  bound before freeing the amplitude.

## [0.9.1] - 2026-07-09

### Fixed

- **Switching FB Asymmetry → FFT with the Batch tab active no longer leaves a
  time-domain model in the Single fit tab.** Leaving the Single tab snapshots
  its form so a hand-built model survives a Single↔Batch round trip, but the
  snapshot's binding identity did not include the fitting domain — after a
  domain switch it replayed the time-domain form (e.g. Exponential + Constant)
  over the frequency default on the next visit to Single. The snapshot now
  only restores into the domain it was taken in (and still restores after a
  round trip back). The corrupt-model restore fallback likewise now picks the
  domain-appropriate default instead of always the time-domain one.

- **Fixed the intermittent background-task deadlock that could wedge a whole
  test shard (and, in principle, freeze the running app).** ``TaskRunner``
  destroyed its worker on the worker thread (a queued ``deleteLater`` posted to
  that thread's event loop). Destroying a Python-subclassed ``QObject`` there
  runs shiboken's ``disconnectNotify`` override under the GIL while ``~QObject``
  holds Qt's signal-connection mutex; if the GUI thread simultaneously builds a
  connection under the GIL (e.g. ``QLabel.setText`` lazily constructing a
  ``QWidgetTextControl``) and wants the same pooled mutex, the two deadlock
  (GIL ↔ connection-mutex ABBA). The worker now moves its thread affinity back
  to the GUI thread before ``run`` returns and is deleted there, so ``~QObject``
  always runs on the GUI thread. See
  ``docs/investigations/gui-shard-gil-signalslot-deadlock.md``.

- **Opening the Fit Wizard no longer stalls on long runs.** The wizard's
  fingerprint plot (and the result card's data overlay) drew a matplotlib
  errorbar over every point of the raw curve on the GUI thread; both now
  decimate to a fixed point budget for display, via the same shared helper
  the grouping preview uses. Stored data and fit/residual curves are
  untouched — only the drawn preview points are strided.

- **Switching FB Asymmetry ↔ FFT no longer stalls the GUI for seconds when a
  multi-peak frequency fit is set up.** Two causes, both fixed: the
  domain-switch restore path re-derived frequency peak seeds that were
  immediately overwritten by the restored values (and against the previous
  domain's dataset — the stale-seed bug), and the seeding peak detection hit a
  scipy ``find_peaks`` worst case that scans O(n²) on spectra with a drifting
  background. Restore paths no longer reseed (carry-forward still reseeds
  against the current spectrum), and the prominence search window is bounded
  (identical peak selection, pinned by golden tests; ~8× faster in the worst
  case measured).

- **Opening the Global Fit Wizard no longer freezes the application (and, on
  low-memory machines, the whole system).** The wizard's candidate-portfolio
  peak analysis estimated each spectrum's noise floor with a running median
  built from an n×window sliding matrix — for zero-padded spectra that
  materialised gigabytes per call on the GUI thread. The running median now
  uses ``scipy.ndimage.median_filter`` (identical output, ~1000× faster,
  constant memory).

- **Fixed the intermittent crash when toggling FB Asymmetry ↔ FFT (and the
  matching Windows "no screens available" fatal exit).** The cause was a
  PySide/shiboken object-lifetime bug, caught live under guard malloc: the
  ``QWidget.screen()`` / ``QWindow.screen()`` bindings tie the process-wide
  ``QScreen`` wrapper to the calling widget's wrapper, so when a transient
  dialog (fit wizards, model dialog) was garbage-collected, Python deleted the
  live C++ ``QScreen`` while it was still in Qt's screen list — every later
  DPI lookup then walked freed memory. Asymmetry now resolves screens through
  a safe helper (``screen_for``), pins the screen wrappers so the GC can never
  reclaim them, and logs loudly if a screen is ever destroyed mid-session.
  Full investigation: ``docs/investigations/tahoe-qscreen-uaf.md``.

## [0.9.0] - 2026-07-08

### Added

- **Frequency-domain fits now show a draggable fit-range span.** The Fit
  dock's frequency range (``≤ ν ≤``, MHz) is drawn on the spectrum as the same
  shaded band with dashed edges used in the time domain, and either edge can be
  dragged directly on the plot (or typed into the range fields). The span
  follows the displayed axis when it is shown in gauss or relative to a
  reference field, while the range stays stored in absolute MHz.

### Fixed

- **The grouping window's asymmetry preview no longer freezes the GUI on
  large runs.** The live forward/backward preview curve was drawn on the GUI
  thread as a Matplotlib errorbar over every reduced point — cheap for a
  typical run but ~12 s for a long, high-resolution one (≈1 M points), since
  rendering cannot leave the GUI thread. The curve is now uniformly decimated
  to at most 2000 points on the preview worker thread before it is drawn (the
  pane is only a few hundred pixels wide, so the sampling is visually
  indistinguishable), cutting the draw from seconds to tens of milliseconds.
- **The grouping window no longer scans every detector twice when auto-detect
  t0 is on.** With an ``auto_detect`` t0 policy, opening the window or switching
  profile/run ran the full per-detector t0 search once to resolve the grouping
  and again to fill the read-only t0 display — a redundant GUI-thread scan worth
  hundreds of milliseconds on a large, high-detector-count run. The display now
  reuses the consensus the resolve already computed. This also fixes a latent
  mismatch: the display scan merged the reference-dataset metadata that the
  resolve did not, so it could show a t0 that disagreed with the one the
  reduction actually used. The remaining one-off scans (an explicit switch to
  auto-detect, the manual **Find t0** button) now show a wait cursor.
- **A Gaussian/Lorentzian preview on a Fourier spectrum now shows the peak,
  not just the background.** The peak centre ``nu0`` (and height/width) are
  re-derived from the displayed spectrum whenever the run changes, so they no
  longer inherit a stale seed left over from the previous (time-domain or
  other-run) selection — which placed the line far off the MHz axis and left
  **Preview** showing only the flat background. A genuinely restored fit keeps
  its recorded parameters. Multi-peak spectral models are now seeded too: each
  ``GaussianPeak``/``LorentzianPeak`` component is placed on a distinct line
  (strongest first), a weak declared line is seeded rather than gated out, and
  surplus components are spread across the fit window instead of defaulting
  off-screen.

## [0.8.0] - 2026-07-07

### Added

- **The Fourier panel can suggest a matched apodisation filter from the
  data.** The Apodisation section's new **Suggest from data** button measures
  the dominant line of the (unapodised) spectrum inside the field-narrowed
  search window and fills the matched filter — mode and τ, shown in the green
  auto-filled colour — without applying anything: the out-of-date banner
  flags the spectrum and an explicit **Compute FFT** applies the filter. A
  matched filter maximises the line's peak S/N at the cost of ≈2× its
  apparent width, so it is never auto-applied; when no line clears the noise
  baseline (or the dominant line is resolution-limited) the panel says *"No
  clear line to match — leave apodisation off."* Because filtered widths are
  a systematic, the spectral-moments readout now shows a caveat whenever the
  active spectrum was computed with apodisation on (each FFT records its
  window and τ in the spectrum metadata).

- **The Fourier panel now flags a displayed FFT that is out of sync with the
  current settings.** Editing any FFT parameter (display mode, apodisation,
  zero-pad, groups, phases, conditioning, exclusions), or changing the
  time-domain fit range the transform inherits, raises an amber banner above
  **Compute FFT** — "Spectrum out of date — …. Compute FFT to refresh." —
  naming what changed. The spectrum itself is kept on screen (nothing
  disappears under you); the banner clears on the next compute. Parameters
  that are inert in the active mode (e.g. a filter τ while apodisation is
  *None*) never flag.

- **Suggest next point: Bayesian experimental design from a trend fit.** Once
  a trend model is fitted, the model-fit dialog's new **Suggest next point**
  section recommends where to measure next — and how many events to count —
  to constrain the model the most, computed from the fit's own covariance
  (Laplace expected information gain). Choose a single target parameter
  (c-optimal, e.g. "pin down `Tc`") or **All parameters (D-optimal)**; the
  full utility curve is drawn on the trend preview so the *why* behind the
  suggestion stays visible, extrapolated candidates are drawn distinctly, and
  the recommended event count is Monte-Carlo calibrated (the raw rank-one
  estimate is optimistic near critical points). When alternative trend models
  are fitted, an **AIC evidence** line ranks them by Akaike weight and a
  second overlay marks the **best discriminating point** — where the
  competing models' predictions diverge most relative to measurement noise —
  or reports that no discriminating point exists in range. Optional
  **Weight by measurement cost** re-weights the utility by a move-time model
  from the instrument's current position, changing where the marker sits but
  never the underlying physics. Documented in *Reference ▸ Suggest next
  point*. (The core suggestion engine and the first dialog section shipped
  quietly in 0.7.0; this completes the feature and documents it.)

- **Internal: lightweight perf-timing helper for the core layer**
  (`asymmetry.core.utils.perf`). `perf_timer` is a GUI-free context manager
  wrapping seven hot functions (PSI/ROOT loading, grouped-asymmetry
  reduction, effective-grouping resolution, grouped time-domain dataset
  building, aligned grouping, and deadtime preparation) that logs a `PERF
  {event}: {elapsed_ms:.1f} ms ...` record on the `asymmetry.perf` logger
  when enabled, with near-zero overhead while disabled. It shares the
  existing `ASYMMETRY_PERF_LOGGING` env var / "Enable performance logging"
  toggle with the GUI's own PERF lines, so one switch now covers both.

- **Internal: the GUI responsiveness rules are now codified.**
  `docs/GUI_GUIDELINES.md` gains a "Keeping the GUI responsive" section
  distilling the 2026-07 responsiveness programme into copyable patterns —
  the per-event-slot rule, debounced single-flight workers, content-keyed
  caching, signal-blocked table repopulation, hidden-panel paint deferral,
  the chunked progress runner, worker + nested-event-loop call sites,
  teardown rules, and the quiet-machine measurement/stress protocols — each
  pointing at its canonical implementation. `AGENTS.md` carries the short
  form, and the structural harness now bans `QApplication.processEvents(`
  in `gui/` outside the app-startup allowlist.

### Changed

- **Grouping detectors uses less memory and time.** ``apply_grouping_aligned``
  (the per-detector t0-aligned sum behind every grouped reduction) built a
  full-length padded copy of every detector in the group before summing them;
  it now accumulates each detector's aligned slice straight into one buffer.
  The result is bit-identical, but on a 128-detector, 1-million-bin run the
  aligned sum is ~3× faster and the full grouped reduction's peak memory drops
  by ~0.75 GB — the transient copies the largest ROOT/HIFI datasets used to
  allocate on every reduction are gone.

- **Dataset switching in the Groups / Raw-counts and bunched views now reuses
  cached reductions.** Toggling between two runs or two views — or nudging the
  display bunch factor back and forth — no longer re-runs the grouped-count
  build and the counts-first re-reduction from scratch each time; results are
  memoised per run and grouping recipe, so a return to a view already shown is
  served from the cache. A grouping edit (or a co-add / combine that swaps in a
  new run) transparently recomputes; nothing about the displayed numbers
  changes.

- **Frequency-domain spectra draw as lines with an error band, and the
  statistics behind zero padding are now handled for you.** FFT and MaxEnt
  spectra render as solid lines (the convention of every reference muSR
  package) instead of the time-domain errorbar-dots idiom, with a shaded ±1σ
  band when the spectrum carries per-point errors, and a subtle dashed
  marker at the expected Larmor position γ_μ·B when the run's field is
  known (single-run view; absent on the correlation axis and in GLE/PDF
  exports, which draw from the data, not the screen). Because zero-padded
  samples are sinc-interpolated and correlated, frequency-domain fits and
  spectral-moment uncertainties now apply the effective-sample-size
  correction automatically — degrees of freedom count the independent
  samples, χ² is scaled to match, and uncertainties grow by √pad; fit
  results state the applied correction in their advisory row. (WiMDA applies
  the dof part of this correction; Asymmetry additionally corrects χ² and
  the uncertainties for consistency.) The time-domain plot is unchanged.

- **PSI HAL-9500 data now loads directly into the Per-octant grouping.**
  High-field (TF) work on HAL-9500 — the AFM-transition corpus and similar —
  is done per-octant in practice: each azimuthal wedge combines its forward
  and backward detector for better statistics than a lone opposed pair. A
  freshly loaded full HAL-9500 run is therefore grouped per-octant
  immediately, so the angle-resolved per-group analysis is available without
  opening the Grouping window (the previous default was the plain
  Longitudinal forward/backward split, which had to be changed by hand). The
  Detector Layout / Grouping dialogs still pre-select Per-octant for a fresh
  HAL layout, and the transverse-field grouping nudge recommends Per-octant
  instead of steering a TF run to the older ``Transverse (opposed pairs)``
  preset. A run shipping only the forward ring (``MV, F1…F8``) still opens
  Per-octant, each octant degrading to its present forward wedge (the F1-vs-F5
  opposed-pair asymmetry) — exactly what applying Per-octant by hand produces.

- **The first FFT view is framed to where the physics is, with interpolated
  line shapes.** A narrow high-frequency line (a 6 T Larmor line, ~13 G wide
  at 813 MHz) is framed *around* — centred, a few dozen linewidths wide —
  because a from-zero axis renders it sub-pixel. Otherwise the initial window
  frames the highest detected line *or* the expected Larmor region γ_μ·B,
  whichever is wider — so a weak or low-field line the peak detection cannot
  see still gets a sensible window instead of the full Nyquist span, while
  second lines (AFM satellites, muonium/radical lines) are never framed out.
  Line detection measures prominence against a local (block-median) baseline,
  so the coloured noise pedestal of a finely-binned TDC spectrum cannot
  masquerade as signal. A multi-run overlay frames to its highest member
  field. (WiMDA frames its FFT plot around the reference field the same
  way.) The default zero-pad factor is now **4** (previously 1) and the
  factor now goes up to 64 (previously 16): padding is pure sinc
  interpolation — display-smoothness only — but note that padded points are
  strongly correlated, so heavily over-padded spectra fed to frequency-domain
  fits or moment error estimates yield artificially small uncertainties;
  that is why the default stays modest. Saved projects and recipes keep the
  padding they were saved with. All are first-paint seeds: manual zoom and
  settings always win. An auto-computed spectrum now also reports "Computed
  FFT for run *N*." in the status line instead of leaving a stale
  "No FFT computed" message under a freshly rendered spectrum.

- **The FFT tab now computes on view instead of waiting for a click.**
  Opening the **Frequency Domain** tab — or switching runs while on it — for a
  run that has never been transformed synthesises a recipe from the current
  Fourier panel settings and the run's own grouping and computes the spectrum
  automatically, off-thread behind a busy overlay ("Computing FFT for run
  *N*…"). **Compute FFT** remains the explicit way to re-run the
  transform after changing a setting. A multi-run overlay auto-computes its
  missing members too, in waves of 25 ("Computing FFT for *N* run(s)…"),
  re-rendering after each wave until every selected run is included; runs that
  cannot compute (no detector groups, a failed transform) are still skipped
  and reported. Applying a new grouping — or a t0 / deadtime / background
  change through the grouping dialog — now discards the affected runs' FFT
  spectra and recipes and recomputes the active view immediately if it is on
  the frequency domain, so the spectrum "follows the data" the way the
  time-domain plot already does; it no longer just raises the stale banner for
  a regroup (the banner still covers FFT-parameter and time-window edits,
  where recompute stays explicit). The empty-state prompt changed to match:
  "No FFT spectrum for this run. Spectra compute automatically when the run
  has detector groups — check the grouping and the log, or click Compute FFT
  to retry." — and now appears only when auto-compute cannot run. MaxEnt is
  unchanged (explicit cycles only), and opening a project still recomputes
  restored recipes lazily without creating new ones.

### Fixed

- **Custom-column relinking no longer silently breaks once a co-added or
  subtracted run exists.** `DataBrowserPanel.custom_values_by_run()` iterated
  both the run table and the combined-run index as if each mapped run numbers
  to dataset objects, but the combined-run index maps a combined run number
  to its *list* of source run numbers. With any co-add/subtract present this
  raised `AttributeError: 'list' object has no attribute 'metadata'` from a
  Qt slot, silently swallowed by the event loop, so a custom column added or
  edited after a batch fit stopped re-linking into existing trend results.
  The accessor now reads the run table alone — the combined dataset's own
  object already lives there under its combined run number.

- **Dragging a spectral-moments handle at its default window no longer grabs
  an invisible fit-range handle underneath it.** The moments widget's default
  window spans the whole spectrum, which coincides with the frequency panel's
  fit-range state (also seeded to the full extent) — but that fit-range
  selector is never drawn on the frequency panel and has no draggable
  behaviour there (the actual frequency-domain fit reads its range from the
  Fit panel's spinboxes instead), so it was a purely invisible target that
  still won hit-testing priority over the visible moments handle sitting at
  the same position. Hit-testing now excludes fit-range handles on the
  frequency panel entirely; dragging a moments handle at the default window
  works on the first click. Fit-range dragging on the time-domain panel is
  unaffected.

- **Alpha calibration and background-configure grouping scans now run off the
  GUI thread without risking a teardown crash.** Pressing **Estimate** in the
  Alpha Calibration dialog, and opening **Configure…** for a reference-run
  background, group the full forward/backward histograms — a per-detector scan
  that briefly froze the window on large runs. Both now run on a background
  worker, with the button disabled and a busy hint for the duration. The
  worker machinery gained a safety net that makes this robust even when a
  parented child dialog (like Alpha Calibration) is torn down through its
  parent's destruction rather than a normal close: a still-running worker
  thread is handed to the process-level keep-alive instead of being destroyed
  mid-run, which would otherwise abort the process. (An earlier take on this
  change shipped exactly that intermittent crash; it is now covered by
  parent-destruction regression tests.)

- **A typed frequency-domain fit range now actually applies.** The Fourier
  Fit tab's fit-range spinboxes committed a value on the frequency plot panel
  through the same code path as the time-domain plot, but that path
  unconditionally no-opped for frequency panels (they draw no draggable
  fit-range selector, so the guard blocked storing state too, not just the
  artists). A typed range therefore never reached the frequency fit — the
  spinboxes showed the new numbers, but the next render silently mirrored the
  untouched full-spectrum default back into the display, and any fit run in
  between used the full spectrum regardless of what was shown. The frequency
  panel now stores the committed range like the time panel does; the
  no-draggable-selector contract is preserved by gating the mouse hit-test
  itself, which also closes a latent ghost-hit-test path where a click near
  an invisible handle position could otherwise start a "drag" with no visual
  feedback.

- **Four small GUI-responsiveness cleanups from the audit's minor findings.**
  Fit-parameter and global-fit trend/compare plots now coalesce their final
  paint with `draw_idle()` instead of a synchronous `draw()` once a background
  compute or checkbox toggle has already finished rendering. The log panel
  caps at 5000 lines (`QTextDocument.setMaximumBlockCount`) instead of growing
  without bound over a long session. Run Info opens instantly and fills its
  Counts (MEv) / Counts per Detector rows with "computing…" first, backfilling
  the real full-histogram sum a moment later instead of blocking dialog-open
  on a GB-scale array sum. And Load Run Range's folder scan now caps at 20,000
  directory entries — a warning appears in the dialog when a facility folder
  (often a network mount) has more files than that, so the range may need
  manual adjustment instead of the scan hanging indefinitely.

- **Opening a project keeps the window responsive and can be cancelled.** The
  per-dataset restore loop — re-applying each run's grouping and recomputing
  its asymmetry, then recreating combined datasets — ran to completion on the
  GUI thread with no feedback, freezing the window for the whole project
  (50–200 runs in a typical scan). The loop now runs one dataset per
  event-loop turn behind a cancellable progress dialog ("Restoring project"),
  so the window repaints throughout; cancelling stops cleanly at a dataset
  boundary, keeps everything already restored, flags the session as partially
  loaded (so a save hard-confirms first, exactly like a cancelled file
  prefetch), and still restores the browser and plot state for what was
  loaded.

- **The pull-distribution diagnostic no longer freezes the app.** Its up to
  2000 simulate+refit iterations ran synchronously on the GUI thread, kept
  "alive" only by a `QApplication.processEvents()` call inside the progress
  callback — a re-entrancy hazard that also forced the Close button to stay
  disabled for the whole run. Clicking Run now starts the diagnostic on a
  `TaskRunner` worker thread with every input snapshotted first; Cancel stops
  it promptly (`run_pull_distribution` polls a cooperative flag rather than
  raising, so a cancelled run reports its partial result instead of aborting),
  and Close stays enabled throughout — closing mid-run cancels and shuts the
  worker down cleanly instead of freezing or crashing.

- **Generating a synthetic run no longer freezes the app.** The Generate
  Synthetic Run and Generate Multi-Group Run dialogs called
  `simulate_run`/`simulate_count_run`/`simulate_two_period_run`/
  `simulate_multi_group_run` directly on the GUI thread, so a large
  multi-detector template at a high event budget blocked the whole
  application with no feedback and no way to cancel. Generate now runs on a
  worker thread (the dialog snapshots every form input first); the button
  reads "Generating…" and disables for the duration, and closing the dialog
  mid-flight joins the worker instead of crashing. Errors and the delivered
  run are unchanged.

- **Combining runs in the data browser no longer freezes the window.** The
  interactive run-arithmetic actions — Co-add Selected, Subtract Reference
  Run…, and Subtract Selected (signed)… — summed per-detector counts across
  every selected run and re-reduced the result synchronously on the GUI
  thread, blocking the whole application for the duration of the combine.
  The heavy combine+reduce now runs on a background worker while the window
  keeps repainting, with a status-bar message ("Combining runs …") and a busy
  cursor over the table for the duration; the combine entries are disabled
  (and re-triggers ignored) while one is in flight, and closing the window
  mid-combine shuts the worker down cleanly. The programmatic path that
  recreates combined rows when a project opens is deliberately unchanged
  (project restore depends on it completing before it returns), and combined
  results are numerically identical.

- **Switching datasets in the plot panel now rasterises the canvas once, not
  twice.** A content switch reframes the view, and a reframe re-decimates for
  the new window on a deferred refresh one event-loop turn later — but the
  plot path also rasterised synchronously first, so every switch paid two full
  `canvas.draw()` calls and the first was overwritten (with decimation clipped
  to the *previous* viewport) before the user could see it. `_apply_limits`
  now skips its synchronous draw whenever a deferred refresh will genuinely
  run, in both the single-axis and stacked-subplot paths, and falls back to
  the synchronous draw whenever the refresh guard bails (reconstruction view,
  no datasets, a refresh already in progress) so the canvas is never left
  undrawn.

- **Selecting a dataset no longer repaints the hidden frequency panel.** Once
  a run had a cached spectrum, every dataset selection on a time view rebuilt
  and rasterised the invisible frequency plot to keep it warm — a full extra
  synchronous canvas draw per click. The paint now defers until the frequency
  view is actually entered (which always re-syncs from the cache), while the
  sync's bookkeeping — the displayed-run key that drops stale async results,
  and the spectral-moments readout — still updates on every selection.

- **The grouping dialog resolves the effective grouping far less often per
  user action.** Two redundant-resolve defects made `resolve_effective_grouping`
  (which can run a t0 auto-detect scan over every detector's full-resolution
  histogram plus a per-run alpha estimate) fire many more times than the
  action warranted: `_reload_controls_from_seed` re-resolved the draft from
  scratch five separate times to read five different fields of the same
  payload, and `_populate_group_table` rebuilt the group table's four columns
  per row without blocking `itemChanged` — which drives both dirty-tracking
  and the live preview — so every table refresh (dialog open, reseed, preset
  apply, detector-layout accept) fired up to `4 × N_groups` redundant
  resolves. Both now resolve/populate once and pass the result through;
  callers that relied on the implicit `itemChanged` storm now trigger the
  dirty/preview refresh explicitly, exactly once. The live preview's own
  resolve has moved off the GUI thread entirely: the per-keystroke refresh
  slot now only lifts the form payload into a draft profile (cheap widget
  reads) and hands it to the preview pane, whose existing debounced worker
  resolves against the run and reduces in one background pass — so an
  auto-detect t0 policy's full per-detector scan (~0.3 s at 128 detectors ×
  1M bins) can no longer stall typing in the dialog.

- **Switching runs no longer renders the plot one view behind.** The draw
  decimates points for the current view window, but a switched dataset's
  reframe moved the axes only afterwards — so the new run showed only the
  points that fell inside the previous run's window (its own line could be
  missing entirely), and switching back inverted the mismatch. Most visible
  when browsing runs at different fields on the frequency view. The plot now
  re-decimates for the window it just framed.

- **Dragging a spectral-moments window/cutoff handle no longer runs a full
  bootstrap on every mouse-move.** Each drag event used to call
  `spectrum_moments(..., uncertainty="bootstrap", n_bootstrap=256)` on the GUI
  thread — 256 resamples per motion event, textbook drag jank. Mid-drag now
  runs the cheap point-estimate path only (`uncertainty="none"`), and even
  that is coalesced behind a 30 ms single-shot timer (latest-wins, the same
  restart idiom as the grouping dialog's preview debounce, just at drag
  cadence) so a fast mouse-move burst collapses to at most one recompute per
  quiet window. The readout stays live (values track the handle; the "±"
  uncertainty term is simply absent while dragging), and releasing the handle
  fires one full bootstrap recompute so the uncertainties return.

- **HiFi high-TF TDC FFTs are no longer silently truncated to nanoseconds.**
  The FFT window's good-statistics tail cap compared raw per-bin counts to
  the raw peak bin. On finely-binned TDC histograms (24 ps bins with a
  prompt spike at t0 that dwarfs the per-bin decay counts) every bin in the
  run sat below 1 % of the spike, so the cap fired at the first bin after it
  and the transform saw ~40 ns of a 10 µs run — burying spectra under the
  tiny window's sinc ringing and reporting linewidths that were the window's
  resolution, not the sample's (corpus run 687: a 1750 G-wide artefact where
  the true line is 13.5 G). The cap now compares 100 ns block averages, which
  track the decay envelope, dilute the prompt spike, and leave 16 ns
  pulsed-source histograms with their previous behaviour.

- **The Fourier panel's spinboxes no longer clip their digits.** The
  zero-pad factor, Burg pole-scan, and correlation-order spins were capped at
  a text-field width that left the step buttons no room of their own, so the
  inner text area shrank to a sliver (a zero-pad of "16" rendered cut off).
  Spinbox width caps now include a step-button allowance.

- **Grouping Apply accepts groupings that name detectors a run does not
  contain, reducing over the detectors present.** Applying a full-instrument
  preset to a file that exports only some of the instrument's detectors —
  e.g. a HAL-9500 **Per-octant** preset on a forward-ring-only PSI `.mdu` run
  (`MV, F1…F8`, no backward ring) — used to report "Applied grouping to
  0 dataset(s); skipped 1" with no explanation, leaving the run's grouping
  unchangeable even though each octant is still a physically valid group
  (its forward wedge). Apply now filters the forward/backward groups to the
  run's detectors, exactly as the dialog preview and grouping profiles already
  did, and notes the ignored absent detectors in the LOG. A run is skipped
  only when a forward/backward group has *none* of its detectors present
  (e.g. **Longitudinal**, whose analysis group is the entire missing backward
  ring) — and the report now names the missing detectors and the run's
  detector count in the LOG and status bar instead of failing silently.

- **Three spinboxes no longer clip their digits at wide values.** The main
  toolbar's bunch-factor spin, the grouping dialog's bunching-factor spin, and
  the grouped count-fit's exclude-window spins were each capped with a
  hardcoded pixel maximum sized for the text alone; under the real app
  stylesheet the spin-button chrome ate into that budget and clipped values
  near the top of each range (e.g. a 4-digit view-bunch factor). Each now
  sizes its cap from the widest value the range allows via
  `metrics.spin_width_for`, the same fix already applied to the Fourier
  panel's spins.

## [0.7.0] - 2026-07-06

### Changed

- **GLE exports no longer block the interface.** The GLE compile step of every
  figure export (main plot, Fit Parameters, Global Parameter Fit window) now
  runs in the background; a wedged GLE process is stopped after a bounded
  timeout instead of hanging the export. All three export surfaces now share
  one orchestrated sequence with a consistent, scrollable "Export Successful"
  results dialog, and re-exporting to an existing `.gleplot` folder cleans up
  stale `.dat`/`.fit` sidecars left by a previous, larger export.
- **Setup ▸ GLE Setup… validates the chosen executable.** On accept the dialog
  runs the path once and refuses to save one that cannot run or is not GLE,
  showing the reason inline — previously a bad path was saved silently and
  only surfaced later as an opaque export error. Export failures from an
  unrunnable configured binary now name the path and point back to the setup
  dialog.

### Added

- **GLE figure editor.** Exporting a figure to GLE (main plot, Fit Parameters,
  or the Global Parameter Fit window) now opens the exported `.gle` script in
  the gleplot figure editor — an in-app window styled like the rest of
  Asymmetry — instead of the static preview dialog, so scripts can be tweaked
  and re-rendered without leaving the app. The editor also opens when no GLE
  binary is installed (editing still works; its preview reports "GLE: not
  found"), uses the binary configured under **Setup ▸ GLE Setup…**, and is
  reachable any time via **Analysis ▸ GLE Figure Editor…**. Requires
  gleplot ≥ 1.6; older installs keep the static preview dialog.

## [0.6.0] - 2026-07-06

### Added

- **One-command releases.** A new **Cut release** workflow (see `RELEASING.md`)
  bumps `pyproject.toml`, rolls this changelog, commits `release: X.Y.Z` to
  `main`, and pushes the tag that triggers the existing installer builds and
  docs deploy — replacing the manual release-PR-then-tag ritual. The release
  build now fails fast if the tag does not match `pyproject.toml` or the
  changelog was not rolled.
- **Instrument switcher in the Grouping window.** When a project holds runs from
  more than one instrument, the Grouping window now shows an **Instrument**
  selector ("GPS — 3 runs") that swaps the whole editor — its draft, selected
  run, scope panel, and preset list — between them, after the usual discard
  prompt for unsaved edits. It is hidden when only one instrument is loaded.
- **Unified selection-driven grouping editor.** The Grouping window's scope panel
  is now the single **selector**: the run you select there is previewed and
  edited, and the editing target follows that run's status. Selecting an
  inheriting run edits the profile; selecting an overridden (released) run edits
  that run's own grouping — which is now editable, where a released run could
  previously only be created or dropped. A persistent strip above the form states
  what your edits apply to (accent-tinted "Editing profile '<name>' — applies to
  N runs" versus warning-tinted "Editing override for run *N* — this run only"),
  matched by the selected row's tint in the scope list and an "override \*" marker
  on runs with pending edits. There is no separate mode and nothing is disabled;
  switching selection never prompts, so edits to the profile and to several
  overrides **accumulate** in one session. **Apply** commits everything at once —
  the profile to its inheriting runs and every edited override to its own run —
  with the button naming the blast radius ("Apply (profile + 2 overrides)") and
  the status bar naming both parts. The only guard is closing the window with
  uncommitted changes, which lists exactly what would be lost ("profile 'Default
  (GPS)' and overrides for runs 12, 15").

- **Time-zero (t0) policy on grouping profiles.** The **t0 Bin** row gains a
  mode selector — *From file*, *Manual*, or *Auto-detect* — carried by the
  grouping profile as a `T0Policy` (mirroring the α and deadtime policies and
  WiMDA's *FileValues* checkbox). *From file* (the new default) uses each run's
  own file-derived t0 exactly as before, with the spinbox read-only and
  following the preview run. *Manual* is the historical editable override, and
  *Auto-detect* runs the prompt-peak / pulse-edge search on every run at
  reduction time. Critically, a manual or detected t0 edit is now
  **non-destructive**: instead of permanently rewriting each histogram's
  `t0_bin`, the shift is published as an `effective_detector_t0_bins` override
  that the reduction chokepoints align on, so the loaded histograms are left as
  read from the file and an override can be changed or cleared freely. The
  existing **Find t0** button becomes the one-shot fill for Manual mode.
  Existing projects migrate per profile: a stored common t0 that differs from
  the run's file t0 becomes `manual`, otherwise `from_file` (no schema bump).

- **Project-level detector-grouping profiles.** The Grouping window is now a
  *profile editor* rather than a bulk broadcast tool. One named grouping profile
  per instrument fingerprint (instrument + histogram count) captures the
  shareable analysis choices — the forward/backward groups and detector
  assignments, the α balance policy, and the deadtime, background, binning and
  period settings — while each run keeps its own file-derived facts (t0, good-bin
  window, per-detector deadtime). Runs *inherit* their fingerprint's active
  profile automatically: newly loaded runs pick it up, and an α or good-bin edit
  no longer has to be broadcast run-by-run. A run can be explicitly *released
  from* its profile (keeping a per-run grouping override, marked with a "custom
  grouping" badge in the data browser) and later *reattached*. The window adds a
  profile selector (New / Duplicate / Rename), a non-destructive **preview run**
  selector that shows a run's per-run facts without disturbing the draft, an
  instrument **preset** dropdown with a live "Preset: … / Custom (edited from …)"
  chip, a debounced **live asymmetry preview** (computed off the GUI thread, so
  editing stays responsive), and an unsaved-changes guard. Profiles are saved in
  the project (schema v12) and re-resolved onto their runs on open; existing
  projects migrate automatically. Applying a profile reports how many runs it
  reached and how many overridden runs were left untouched.
- **Dedicated alpha, deadtime, and background calibration dialogs.** The three
  corrections that used to live as inline controls in the Grouping window are
  now their own dialogs, opened from a compact status row ("α = 1.2345(67) ·
  diamagnetic · run 2923", "Deadtime: manual …", "Background: …") with a
  **Calibrate…**/**Configure…** button. The alpha dialog highlights and
  auto-selects a likely weak-transverse-field calibration run in its run
  dropdown (from the run's field-geometry metadata and, when recorded, its
  field magnitude), offers **diamagnetic** / **general** / **ratio**
  estimation methods, and shows a live before/after (α = 1 vs. the fitted
  value) preview; accepting a calibration records its provenance, and a manual
  edit of α drops the provenance back to plain "manual". Each vector or
  multi-projection axis (EMU P_x/P_y/P_z, GPS WEP FB/UD, MuSR/HiFi transverse
  pairs) calibrates and stores its own alpha independently. The deadtime
  dialog adds a "maximum correction at t=0" summary; the background dialog
  adds a shaded-window preview for the range/reference-run modes.
- **Detector schematic: multi-group membership and hover detail.** A detector
  that belongs to more than one group (the ordinary case for transverse and
  vector-polarisation layouts, and now visible for HiFi's manual
  transverse-quadrant grouping) is drawn as one thin slice per membership
  instead of showing only its primary group. Hovering a detector shows its id,
  physical label, full list of group memberships, and exclusion state; each
  group button shows a live member count, and a **Clear excluded** button
  resets every exclusion in one action. A source-audited pass confirmed the
  EMU `Vector Polarization` preset's octant composition is internally
  consistent with the layout's own detector angles (and remains an Asymmetry
  construct with no facility-documented equivalent) and that the HiFi and MuSR
  schematic orientations already matched their respective user manuals.

### Changed

- **"Instrument" replaces "fingerprint" throughout the Grouping window and its
  documentation.** The user-facing term for a run's `(instrument, histogram
  count)` identity is now plainly "Instrument"; the scope panel is headed "Runs
  of this instrument", and the detector count is shown ("GPS (6 detectors)") only
  when two variants of the same instrument are both loaded. The internal
  `ProfileFingerprint` API is unchanged.

- **macOS release binaries are Apple Silicon (arm64) only.** GitHub Releases
  publish a Windows x64 installer and a single macOS arm64 DMG; the Intel macOS
  build was dropped because PyPI ships no Intel macOS `pyhdf` wheel. Intel Mac
  and Linux users install from source.

### Removed

- **`.grp` file Load/Save.** The Grouping window's "Load .grp" / "Save .grp"
  buttons, and the underlying line-based `.grp` serialization
  (`GroupingDialog.serialize_grp`/`parse_grp`), have been retired. Project
  persistence (grouping profiles saved in `.asymp` files) and instrument
  presets now cover what `.grp` files were for — sharing and reusing a
  grouping/calibration between sessions — without a separate file format to
  keep in sync with the profile schema. Existing `.grp` files are unaffected
  on disk; there is simply no in-app way to read or write them any more.

### Fixed

- **Changelog history repaired.** A re-land merge (#175) had accidentally
  deleted the `## [0.5.0] - 2026-06-21` heading, leaving all of 0.5.0's
  released notes stranded under `[Unreleased]`, and a later entry landed
  inside the released section. The heading is restored (the section again
  matches the release commit exactly) and the misplaced entry moved here.
- **Stale persisted instrument identities now self-heal on reload.** A run whose
  saved grouping named the wrong instrument (e.g. a FLAME run stored as "GPS" by
  an earlier version with broken detection) stayed wrong forever: the stale string
  pinned the wrong profile fingerprint, so the run inherited a mismatched profile
  and kept the wrong preset's group names. A freshly loaded run is now treated as
  the ground truth for its own instrument — when detection positively disagrees
  with the stored value, the detected instrument wins (the run's grouping and
  metadata are corrected and the stale instrument-dependent structure is
  discarded in favour of the loader defaults), so the run re-detected as FLAME no
  longer inherits a (GPS, 8) profile. Inconclusive detection leaves the stored
  value untouched, and a matching value is preserved byte-identical. Saved
  profiles are never modified; only runs heal.
- **PSI analysis convention in the GPS/FLAME/HAL presets.** PSI names detectors
  by beam direction, and for surface muons the initial polarisation points toward
  the *Backward* detector, so the PSI/musrfit analysis convention is
  `A = (B − αF)/(B + αF)` (GPS instrument paper, Amato *et al.* 2017, Eq. 2). The
  loaders already honoured this, but the built-in GPS, FLAME and HAL-9500
  presets declared the beam-Forward group as analysis-forward, so a preset used
  headless (core) reduced with the wrong sign/leg. Every PSI Longitudinal preset
  now declares the Backward-named group in the analysis-forward slot, and the GPS
  `WEP` `FB` projection follows musrfit's `forward=2(B) backward=1(F) alpha=0.75`
  so it reduces to `(B − 0.75F)/(B + 0.75F)`. The GPS spin-rotated combined-pair
  preset was also fixed: the ~50° upward-rotated spin points along the
  **Backward–Up** diagonal (not Forward–Up), so it is now
  `Spin-rotated (B+U/F+D)` (was `F+U/B+D`). The grouping-dialog beam→analysis
  swap is widened to recognise single-letter (`F`/`B`) and compound (`B+U`) group
  names as defence in depth, and no longer double-swaps a fixed preset.

## [0.5.0] - 2026-06-21

### Added

- **Robust batch seeding for near-transition oscillatory scans.** A block-separable
  F-B asymmetry batch (every free parameter Local — e.g. an EuO ZF temperature scan
  approaching `T_C`) now honours the **Chain from previous run** / **Auto** seeding
  mode it previously ignored: runs are fit in physical-scan order, each warm-started
  from the previous good run, and a run that converges onto the spurious branch
  (amplitude collapsed to ~0 or frequency discontinuous with the trend) is detected
  and reseeded once from the good-run trend before being kept. New core engine
  `asymmetry.core.fitting.fit_asymmetry_series` plus shared seeding/diagnostics in
  `asymmetry.core.fitting.series_seeding` (`diagnose_series`,
  `detect_amplitude_collapse`, `detect_frequency_outliers`, `suggest_series_seeds`,
  `recommend_series_seeding`), reused by the grouped-series Auto policy so both batch
  paths agree on when to chain.
- **Batch outlier signpost.** When a finished batch's ν(T)/A(T) trend shows the
  collapse/outlier signature, the Batch tab now surfaces a banner that points at the
  per-run **Initial Values…** warm-start and offers a one-click **Use suggested
  per-run seeds** — filling the per-run table with descending-frequency seeds
  interpolated from the cleanly-fit runs and re-running in Independent-seeds mode
  (automating the proven manual cure).
- **Angle-dependent muon Knight shift** in the parameter-trend panel. Fitted
  oscillation components (local field `field_n` or frequency) convert to the
  muon Knight shift `K = (ν − ν_ref)/ν_ref` against either reference: the
  applied field `ν_ref = γ_µ·B` (no reference line needed, the default) or a
  designated fitted component (covariance-aware). Per-component `K[...]` traces
  carry a chosen display unit (ppm / per-cent / auto). Configured from the
  **Knight shift…** button; see
  `asymmetry.core.fitting.knight_shift` for the scriptable API.
- **First-class Angle (°) trend axis** with a **Fold** control that wraps a
  periodic single-crystal rotation into one period (180° or 360°) for display
  and `K(θ)` fitting; the stored angles are unchanged.
- **`K(θ)` anisotropy basis models** for the Angle axis: `KnightAnisotropy`
  (`K_iso + K_ax·(3cos²θ − 1)/2`, axial dipolar) and `AngularCos2`
  (`K_avg + K_amp·cos2(θ − θ₀)`, two-fold), in degrees.
- **Joint `K(θ)` fit** (**Joint K(θ) fit…**): fits one `K(θ)` curve per
  component simultaneously and, at each angle, assigns that angle's component
  points one-to-one to the curves they best fit (Hungarian matching, iterated
  fit ↔ reassignment). The selected `K[...]` traces are reordered **in place**
  so each follows one physical site continuously through crossings, with the
  per-curve fits overlaid and the resolved crossings banded. Core engine in
  `asymmetry.core.fitting.angular_assignment`.
- **Clogston–Jaccarino `K`–`χ` helper** (`clogston_jaccarino_fit`, API-only):
  fits `K = K_0 + (A_hf/N_A μ_B)·χ` to extract the hyperfine coupling and the
  `χ`-independent offset.
- **Transverse-field grouping nudge** (B8a): a transverse-field run loaded on a
  longitudinal-default instrument (e.g. PSI GPS) now surfaces a non-blocking hint
  to switch off the `Longitudinal` (Forward/Backward) grouping — which washes out
  the precession and collapses the time-domain fit — toward the recommended
  spin-rotated/transverse preset. The Grouping dialog points at the Detector
  Layout editor; the editor pre-selects the recommended preset (e.g. GPS
  `Spin-rotated (F+U/B+D)`) so applying it is one click. The recommendation is a
  pure helper, `asymmetry.core.instrument.recommend_grouping_preset(layout,
  field_direction)`; nothing is auto-applied.
- **PSI field geometry from free text**: the PSI `.bin`/`.mdu`/`.root` loaders
  now derive `metadata["field_direction"]` from an explicit `TF`/`LF`/`ZF`
  (or transverse/longitudinal/zero-field) tag in the run comment/setup/title
  (`asymmetry.core.io.base.field_direction_from_text`). Consistent with the
  field-geometry policy, geometry is taken only from an unambiguous string —
  never inferred from the field magnitude or the sample/detector orientation —
  and is left blank when absent or ambiguous. This lets the grouping nudge fire
  on PSI GPS data, which carries no structured field-state code.

### Fixed

- **Knight-shift / `K(θ)` fits survive save and reload**: model-fit overlays on
  the first-class **Angle** axis previously lost their axis key on load
  (`angle` collapsed to `run`), so the curves silently dropped out of the trend
  plot. The joint fit now also stores its per-curve overlays, reorder
  permutation and crossing markers with the project, so the whole fit is
  reconstructed deterministically on reload regardless of the per-group
  snapshot.
- **Knight-shift K traces are removable**: the **Remove** action now deletes
  selected `K[...]` traces by dropping their component from the conversion (so
  they do not regenerate) and clears any joint fit spanning them; obsolete
  `K⟨n⟩` track columns from earlier builds are migrated away on load.
- **Y-parameter multi-selection no longer collapses**: interactive cell widgets
  in the Y-parameter table no longer steal keyboard focus from the selection
  model, so a Shift+Arrow multi-selection survives mouse interaction.

- **Send to Batch carries the current single-fit seeds**: the **Send Model to
  Batch** action now seeds each batch parameter from the Single tab's current
  table values (which reflect the latest fit once one has run) instead of only
  copying the composite model. Previously the batch Parameter Classification
  fell back to model defaults or stale preserved state — e.g. a leftover
  frequency seed of 1.355 instead of the just-set value (BUG B8c).

## [0.4.0] - 2026-06-14

This release marks the completion of the Wimda parity programme: Asymmetry now
covers the analysis workflows of the reference Wimda tool — including
per-projection alpha estimation, grouped-series fitting, trend propagation, and
the PSI GPS detector layouts — within the scriptable core and PySide6 desktop
application.

### Added

- **Per-projection alpha**: the asymmetry reduction now resolves alpha per
  declared projection rather than only for the canonical EMU axes. The
  vector-alpha table has one editable row per projection (rebuilt when
  projections change), so non-canonical presets (GPS WEP FB/UD, the
  MuSR/HiFi transverse pairs) can be recalibrated and estimate their own
  alpha; values, errors, and reference-run provenance round-trip through
  `.grp` save/load and survive detector-layout edits.
- **PSI GPS detector layout**: built-in GPS detector geometry shipped in both
  BIN and ROOT loader variants.
- **Domain representation model** (`asymmetry.core.representation`): each
  dataset now carries up to four named representations (`time_fb_asymmetry`,
  `time_groups`, `freq_fft`, `freq_maxent`), each holding a recipe (for
  recompute-on-load), a `FitSlot` (most recent fit), and a trend-state dict.
  `FitSlot` tracks provenance (`"none"`, `"single"`, `"batch"`, `"global"`),
  the owning `FitSeries` id, and divergence / include-in-trend flags.
- **FitSeries** (`asymmetry.core.representation.series`): renamed from
  `Batch`; adds `member_kind` (`"runs"` / `"groups"`), `nuisance_params`
  (group-only, always per-(run,group)), and `member_source_run` (maps
  synthetic group keys to physical run numbers). Group member keys follow the
  convention `-(source_run * 1000 + group_index)`.
- **ProjectModel** (`asymmetry.core.representation.project_model`):
  in-memory owner of all representations and series; provides
  `refresh_divergence()` (group-aware), `trend_runs_for_batch()`, and
  `set_member_trend_inclusion()`.
- **TrendState** dataclass (`asymmetry.core.representation.trend_state`):
  formalises the opaque trend-state dict as a typed dataclass with
  `to_dict()`/`from_dict()` and a legacy passthrough for unknown keys.
- **Schema v6 → v7**: additive migration adds `member_kind`, `nuisance_params`,
  `member_source_run` to existing series; normalises `trend_state` to the
  structured shape; group series are now persisted in `batches`.
- **Grouped series fitting** (`fit_grouped_series`): new engine entry point
  that runs individual, batch, or global grouped fits over N runs, recording
  results as a `FitSeries(member_kind="groups")`.
- **Shared result summary** (`asymmetry.core.fitting.result_summary`):
  `fit_result_summary()` produces an identical compact JSON-serialisable shape
  for both run-batch and grouped-series recording.
- **Fit-series-centric UI** (Phase 3): scope is derived from context rather
  than selected. Member kind follows the active representation; member set
  follows the data-browser selection; relationship (single / batch / global)
  follows the parameter-role table.
  - **Multi-Group Fit window** now has separate **Single** and **Batch** tabs
    (both `GlobalFitTab(member_kind="groups")`).
  - **Parameter role table** unified to `Global` / `Local` / `Fixed` for
    physics parameters in grouped fits; nuisance parameters are always Local
    for multi-member fits.
  - **Pipe-back**: after a batch fit, each member's result and role annotation
    are piped back to the **Single** tab. A dedicated **Batch** column in the
    single-fit parameter table shows each parameter's batch role.
  - **Send Model to Batch** action seeds the Batch tab's composite model from
    the Single tab's current state.
  - **Add to Series** action adds a compatible single fit to an existing
    run-membered `FitSeries`.
  - **Initial Values…** dialog: a members × parameters grid for per-member
    seed values (both F-B Asymmetry Batch and grouped Single/Batch tabs).
  - Removed dead scope-selector machinery (`_mode_combo`, `allowed_modes`
    ctor param, `grouped_mode_changed` signal).
  - Relabelled **Global** fit tab to **Batch** throughout (global is now a
    derived relationship, not a separate surface).
- **Representation-aware Fit Parameters panel** (Phase 4):
  - Panel operates on a **pull model**: `_refresh_trend_panel()` reads the
    active representation's `FitSeries` from `ProjectModel` and calls
    `FitParametersPanel.load_representation_series()` after each fit and on
    every representation switch.
  - **"Showing:" label** in the panel header indicates the active
    representation (`F-B Asymmetry`, `Detector Groups`, `FFT`, `MaxEnt`).
  - **Series buttons** (one per `FitSeries` for the active representation)
    replace the old UUID-keyed group buttons.
  - Selecting a series button emits `series_selection_changed`, which
    highlights the member runs in the Data Browser with an amber tint.
  - Grouped fits now appear in the trending panel (previously they did not).
  - FFT batch-fit series resolve field/temperature metadata from
    `_frequency_spectra_by_run` rather than the data browser.
- **Apply Fourier to selection**: copies the active run's FFT recipe to all
  other selected runs and regenerates their spectra, enabling a consistent
  series configuration without manual per-run retuning.
- **Domain navigation redesign**: the toolbar's domain buttons are now grouped
  under **Time domain** and **Frequency domain** cluster headers with a
  reserved **MaxEnt** button. The `active_view_changed` signal now fires on
  every representation switch, driving both fit-dock mode and trend-panel
  refresh.

### Changed

- `GlobalFitTab` constructor parameter `allowed_modes: tuple` replaced by
  `member_kind: str` (`"runs"` or `"groups"`); the member kind is fixed per
  instance and follows the active representation, not a UI selector.
- `CURRENT_SCHEMA_VERSION` bumped from 5 to 7 with additive v5→v6 and
  v6→v7 migrations.
- `Batch` class in `asymmetry.core.representation` renamed to `FitSeries`
  (module renamed from `batch.py` to `series.py`).
- Fit Parameters panel `_group_fit_results` is now keyed by `batch_id`
  (series-keyed) rather than ad-hoc UUID group IDs.
- `DataBrowserPanel.clear()` now also resets `_highlighted_runs` so series
  highlights from a previous project do not bleed into the next project.

## [0.3.4] - 2026-05-20

### Changed
- Vector-polarization plotting now keeps `Auto Y` active in `ALL` mode, recalculates per-polarization Y limits on redraw, and preserves the manual Y-lock for stacked views.
- Vector-polarization `ALL` views now show fit-range indicators correctly across the stacked polarization subplots.
- Returning from grouped or frequency views to F-B asymmetry now restores the vector-polarization selector reliably.

## [0.3.3] - 2026-05-18

### Added
- Added grouped time-domain fitting support with a dedicated multi-group fit window, grouped count-model fitting flow, and initial documentation for the new workflow.

### Changed
- Improved plot responsiveness during navigation and updated grouped plotting so individual-group views retain full traces while fit previews and fitted overlays respect the active fit window.
- Refined grouped fit parameter handling, including default fractionized multi-group models, stronger dependent-fraction enforcement, improved grouped parameter write-back, and clearer additive-term parameter numbering for fractionized sum-of-products.
- Restored grouped fit-window controls and scrolling behavior in the individual-groups view, including clearer fit overlays and grouped plot interaction updates.

### Documentation
- Expanded grouped time-domain fitting and detector-grouping documentation, and recorded the porting study notes for the multi-group time-domain fitting workflow.

### Tests
- Added regression coverage for grouped time-domain fitting, grouped plot behavior, grouped fit-window handling, grouped parameter synchronization, and composite-model numbering semantics.

## [0.3.2] - 2026-05-11

### Changed
- Packaged GUI builds now load the splash logo more reliably from bundled resources and include the spinbox SVG arrow assets required by the Qt stylesheet.
- GLE export compilation now runs from the export bundle directory so generated `.dat` and `.fit` sidecars resolve correctly in packaged builds, and preview handling is more robust after export failures.
- Co-add compatibility now ignores per-run frame-count metadata when grouping settings otherwise match, and combined-run log temperatures now display as event-weighted averages.
- Global-fit parameter role selections are preserved after fit completion, and long fit-function formulas wrap more cleanly in the fit UI.

## [0.2.1] - 2026-04-17

### Changed
- Combined datasets now mirror the grouping state of their hidden source datasets, and grouping edits on a combined row update the hidden sources before rebuilding the combined result.

### Documentation
- Documented the stricter co-add grouping rules, mixed-source restriction, and the way grouping edits on combined datasets propagate back to their source runs.

### Tests
- Added regression coverage for grouping-compatible co-adds, blocked mismatched/mixed-family co-add attempts, and grouping edits applied through combined datasets.

## [0.2.0] - 2026-04-03

### Added
- Parameter-model field component `GaussianLCR` in Eq. (4) notation from PRL 135, 046704 (2025): `lambda_LCR(B) = f * G(B; B0; Bwid)`.

### Changed
- Parameter-model docs now explicitly state that Redfield exponent `m` is dimensionless.
- In the model-fit GUI for field-series parameter fits, component selection now avoids redundant constants: Lambda-like y-parameters show `Lambda_bg` and hide `Constant`, while non-Lambda y-parameters show `Constant` and hide `Lambda_bg`.
- Model parameter fits and cross-group parameter fits now run asynchronously in the GUI, with explicit in-progress status text and temporary control locking to avoid blocking the UI.
- Composite fit models now share a single amplitude parameter across each multiplicative/divisive chain instead of assigning a separate amplitude to every factor.
- Asymmetry uncertainty calculation now follows Mantid `AsymmetryCalc` behavior, including default uncertainty for zero-denominator bins.
- Main plot now suppresses bins with non-positive grouped denominator (`F + alpha*B <= 0`) to avoid displaying undefined asymmetry points.
- Main plot bunch-factor control has been removed; bunching is managed in the Grouping workflow.
- Grouping deadtime correction now uses Mantid-style good-frame normalization: `Ncorr = N / (1 - N*tau/(dt*good_frames))`.
- Grouping dialog bunching-factor input range has been expanded to support large values.
- Main-plot limit controls now use independent `Auto X` / `Auto Y`; manual X/Y limits apply on Enter and no longer require an Apply button.
- `Auto Y` now computes limits from points inside the current X range and prioritizes reliable foreground points.
- Run Info now provides include-in-browser checkboxes and per-row log plotting in both the primary table and Advanced subwindow.
- Advanced Run Info metadata filtering now includes an inline search field, and the summary table exposes sample orientation for promotion into the Data Browser.
- Data Browser extra metadata columns now use friendly labels for known Run Info fields.
- Project persistence now round-trips grouping settings and restored plot limits more reliably, including list-returning/multi-period loader paths.
- Main-plot GLE export has moved from File/toolbar menu actions to in-panel controls: **Export Plot(s) to GLE** with a PDF/EPS format selector.
- Main-plot GLE export now supports data-only or fitted exports for plotted datasets, label-based sidecar filenames, and exports data as error bars plus fits as line curves when present.
- Main-plot `.dat` sidecars now include run/grouping metadata headers and are rewritten after GLE save so metadata survives helper-generated file overwrites.
- Main-plot ``.fit`` sidecar headers now include fit-function descriptions, fit statistics, and fitted parameter values/uncertainties when available.
- Grouping launch now preselects the highlighted runs, and newly loaded runs inherit the most recent in-browser grouping payload from the highest run number when possible.
- Two-period red/green grouping mode now computes `G-R` and `G+R` in asymmetry space (`A_G - A_R`, `A_G + A_R`) with uncertainty propagation by quadrature.
- Multi-run overlays in RG mode now use contrasting colors for additional runs so selected traces remain visually distinguishable.

### Documentation
- Updated the composite-model guide to document shared amplitudes across multiplicative/divisive chains and the resulting formula/parameter-table behavior.
- Updated the GUI user guide to document alpha display, Run Info search/orientation workflows, friendly Data Browser metadata headers, grouping preselection/template inheritance, and persistence details for dynamic columns and grouping settings.
- Documented the main-plot export workflow for plotted datasets, including data-only exports, label-based ``.dat``/``.fit`` naming, metadata-rich ``.dat`` headers, annotation export, and ``.fit`` header metadata.
- Documented two-period RG mode behavior in the GUI guide, including mode definitions, asymmetry-space `G±R` formulas, uncertainty propagation, and plotting color behavior.

### Tests
- Added regression and end-to-end persistence tests for:
	- plot-limit restore with dataset replot,
	- grouping override save/restore,
	- multi-period restore dataset selection,
	- project round-trip restoring grouping and axis limits,
	- Run Info/Data Browser synthetic column integration.
- Added regression tests covering two-period `G-R` asymmetry-space subtraction and multi-run RG overlay color distinction.

## [0.1.0] - 2026-03-09

### Added
- Initial release of Asymmetry μSR data analysis library
- Core data structures: `MuonDataset`, `Run`, `Histogram`
- ISIS muon NeXus file loader with metadata extraction
- Logbook/run-table management for multiple datasets
- Data transformations: asymmetry calculation, grouping, rebinning
- Fitting engine with iminuit backend
- Built-in μSR fit models: exponential, Gaussian, stretched exponential, oscillatory, static GKT
- Global (simultaneous) fitting with shared and local parameters
- Fourier analysis: FFT with apodization windows (Hann, Hamming, Blackman, Bartlett)
- PySide6-based GUI with data browser, plot panel, fit panel, and fit parameters panel
- GLE export integration via gleplot for publication-quality parameter trend plots
- Interactive plot panel with Matplotlib backend
- Data browser with sortable columns and Excel-style filters
- Fit parameters panel with matplotlib and GLE plotting options
- Command-line interface: `asymmetry` and `asymmetry-gui`
- Comprehensive test suite with 97 tests and 71% coverage
- Sphinx documentation with user guide and API reference
- Support for Python 3.10, 3.11, 3.12, and 3.13

### Technical Details
- Pure-Python core library with no GUI dependencies
- Plugin-based architecture for data loaders and fit models
- Separation of concerns: core analysis engine vs. GUI
- Full scriptability for batch processing and Jupyter workflows

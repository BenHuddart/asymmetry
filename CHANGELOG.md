# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Changed

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

- **PSI HAL-9500's default grouping preset is now Per-octant, not
  Longitudinal.** High-field (TF) work on HAL-9500 — the AFM-transition
  corpus and similar — is done per-octant in practice: each azimuthal wedge
  combines its forward and backward detector for better statistics than a
  lone opposed pair. The Detector Layout / Grouping dialogs now pre-select
  Per-octant for a fresh HAL layout, and the transverse-field grouping nudge
  recommends Per-octant instead of steering a TF run to the older
  ``Transverse (opposed pairs)`` preset.

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

- **Switching runs no longer renders the plot one view behind.** The draw
  decimates points for the current view window, but a switched dataset's
  reframe moved the axes only afterwards — so the new run showed only the
  points that fell inside the previous run's window (its own line could be
  missing entirely), and switching back inverted the mismatch. Most visible
  when browsing runs at different fields on the frequency view. The plot now
  re-decimates for the window it just framed.

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

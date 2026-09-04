# Fit Wizard: matched-apodisation scan for heavily damped lines

Status: plan agreed with maintainer (2026-09-04); delivered as **one PR** on
`feat/wizard-damped-line-scan`, built phase by phase by subagents with a
lead-agent review gate after every phase (see "Gates").

## Problem

A heavily damped oscillation (envelope lifetime 10–50 ns, gone by ~0.05 µs)
on a finely binned record (0.1 ns bins, ~10⁵ points) is invisible to the
wizard's windowed pass, marginal for its unwindowed early-window crop ladder
(one of two lines found, at SNR 6.4 against a gate of 6.0), and — even when a
line is found — cannot be *fitted*: the single-oscillator candidate has no
slow relaxing term, a lone early peak cannot form a multiplet, the envelope
seed is a fixed three lifetimes per crop, and 16 Stage-2 fits on 10⁵ points
take ~2 min. The wizard then recommends `Exponential + Constant` with High
confidence. Separately, the Hann pass on such records returns clusters of
junk peaks against the Nyquist edge, and the plot's fit range crops the
record before the wizard sees it.

Full diagnosis (entry points, constants, call sites) is in the lead agent's
2026-09-04 exploration; the code map lives in `docs/ARCHITECTURE.md` §3.3 and
`docs/reference/fit_wizard.rst` § "Heavily damped lines".

## Design (settled)

1. **Matched-apodisation ladder** — `asymmetry.core.fitting.damped_line_scan`
   (new, Qt-free). Unwindowed spectrum of `(y − tail) · exp(−t/τ)` over a
   geometric τ ladder (≈4 rungs/decade from ~20 bins to half the informative
   record). A line with damping λ peaks in SNR at τ = 1/λ (matched filter),
   so the ladder finds fast lines with **no crop** and yields a damping
   estimate. Per-rung guard band `f > min_cycles/τ`; global median/MAD noise
   floor; peaks clustered across rungs; the rung of maximum SNR gives τ*.
2. **Weighted linear Δχ² verification with peeling.** Candidate (f, λ) is
   tested by weighted linear least squares against a slow-decay dictionary
   (`exp(−λ_k t)`, λ_k ∈ {0, 0.3, 1, 3, 10} µs⁻¹) with and without the pair
   `e^{−λt}cos, e^{−λt}sin`; a local (f, λ) grid maximises Δχ². Accept on a
   **look-elsewhere-corrected** threshold `Δχ² ≥ 2·ln(N_trials / rate)` with
   `N_trials` = Σ over rungs of (band width / line FWHM) and `rate = 0.01`
   (same philosophy as `peak_detection._FALSE_PEAK_RATE`); an accepted line
   joins the basis and the scan repeats (max 3 lines). Output per line:
   frequency, λ, amplitude (%), phase, Δχ², τ*, SNR.
3. **Wizard integration.** The scan becomes the wizard's damped-line pass
   (superseding the early-window crop ladder as the *wired* pass — the
   old code stays importable and unit-tested but is no longer called from
   `analyze_dataset_peaks`). `DetectedPeak` gains additive optional fields
   `damping_rate_per_us`, `amplitude_percent`, `phase_rad`,
   `delta_chi_squared`; `source="damped_scan"`. Seeding uses the measured
   λ/amplitude/phase; `Lambda` bounds always contain `[λ/4, 4λ]`;
   frequency bounds narrow to a few FWHM (`λ/π`). New templates in the
   oscillatory family: `Σ(Osc×Exp) + Exp + Const` for n = 1..3 damped lines
   (keys `oscillatory{n}_exp_relax_constant`), plus the Gaussian-envelope
   twins; a single damped-scan line is enough to build them. A scan hit
   above threshold promotes the oscillatory family (pattern-style
   promotion, exempt from the Stage-2 family cap).
4. **Oversampling control.** Detection: the scan crops each rung to ≈10τ
   and value-rebins so a rung never exceeds a fixed sample budget; the Hann
   pass gets a Nyquist guard of max(0.5 resolution, 2 % of Nyquist).
   Fitting: `build_fit_wizard_recommendation` fits a rebinned copy when
   `n_points` exceeds a budget (~8192), with the factor capped so the
   highest seeded frequency keeps ≥ 8 samples per cycle; the recommendation
   records the rebin factor and points used, and the docs say information
   criteria refer to the analysed (rebinned) record.
5. **Fit-window independence.** The wizard window analyses the **full**
   record (not the plot's fit-range crop) and shows a note when the plot's
   fit range starts later than 3/λ of a detected fast line.
6. **GUI.** Peaks table gains a "Damping (µs⁻¹)" column; damped-scan peaks
   are drawn dashed with their λ in the tooltip; the fingerprint table shows
   the damped-line rows; a click-seeded frequency is refined by the Δχ²
   stage so it carries a λ too. Docs section rewritten; screenshot scenario
   for the fingerprint panel updated if the panel changes visibly.

Decisions recorded: keep `analyze_early_window_peaks` and its tests (not
deleted this PR, only unwired); Δχ² threshold is look-elsewhere corrected
rather than a fixed SNR gate; no research data, sample names, run numbers or
private paths anywhere in the repo — every test is synthetic.

## Phases, agents and gates

Each phase is one subagent task on the feature branch, committed with a
conventional message; the lead agent reviews the diff, runs the focused tests
and a private out-of-repo verification harness on the maintainer's dataset,
and either accepts or sends fixes back before the next phase starts.

### Phase 1 — core scan module (Opus)

Files: `src/asymmetry/core/fitting/damped_line_scan.py` (new),
`tests/core/test_damped_line_scan.py` (new), export from
`asymmetry.core.fitting.__init__` if that package re-exports siblings.

Deliverables:
- `tau_ladder`, `matched_apodisation_scan`, `cluster_scan_peaks`,
  `damped_line_delta_chi2`, `refine_line`, `look_elsewhere_threshold`,
  `detect_damped_lines(time, asymmetry, error, *, max_lines=3,
  false_rate=0.01, sample_budget=...) -> DampedLineAnalysis` with frozen
  dataclasses `DampedLine` / `DampedLineAnalysis` (+ serialize/deserialize
  helpers mirroring `peak_detection`'s).
- Uses `peak_detection.effective_analysis_window` for the informative
  window; uses `asymmetry.core.transform.rebin.rebin` for value rebinning.
- Runtime ≤ 1 s on a 10⁵-point record on a laptop core (assert with a
  generous CI bound, e.g. 5 s).

Tests (synthetic only, seeded RNG, µSR-like exploding errors):
- two damped lines (≈240/120 MHz, λ ≈ 40/20 µs⁻¹, 5 %/2 %) on a slowly
  relaxing 5 % background: both found, f within 2 %, λ within 30 %,
  amplitude within 25 %, Δχ² ordering correct;
- one very fast line (λ ≈ 100 µs⁻¹): found;
- slow conventional TF line (≈1 MHz, λ ≈ 0.2 µs⁻¹): found, λ within 50 %;
- pure noise, exponential / stretched / Kubo-Toyabe relaxation without
  oscillation: no lines (100 seeds, zero acceptances);
- ×2 rebinned input gives the same lines within tolerance;
- serialization round-trip; threshold grows with `N_trials`.

Gate 1: lead runs `harness.py test -- tests/core/test_damped_line_scan.py`,
`--tier fast`, and the private harness; scan must report both cold-run lines
and none on the paramagnetic runs.

### Phase 2 — wizard integration (Opus)

Files: `peak_detection.py` (`DetectedPeak` fields + serialization,
`analyze_dataset_peaks` wiring, Nyquist guard, `merge_user_peaks` λ
attachment), `fit_wizard.py` (fingerprint `damped_line_*` from the scan,
`_damped_envelope_rate`, seeding/bounds, new templates, promotion,
rebinned fitting, recommendation metadata), `tests/core/test_peak_detection.py`,
`tests/core/test_wizard_damped_seeding.py`, `tests/core/test_fit_wizard.py`,
`tests/core/test_fit_wizard_tiered.py`.

Deliverables:
- Blind `build_fit_wizard_recommendation` on the Phase-1 two-line record
  ranks `oscillatory2_exp_relax_constant` (or its Gaussian twin) top with
  fitted f/λ near truth; on the single fast line record ranks a
  one-line relax template top; conventional records' recommendations
  unchanged (existing tests stay green, updated only where the early pass
  was asserted by name).
- Near-Nyquist junk suppressed (test: white noise at 0.1 ns binning with a
  slow relaxation yields no `fft` peaks above 0.98 Nyquist).
- Rebinned fitting: on a 10⁵-point record the recommendation key equals
  the one obtained on the pre-rebinned record and wall time stays under a
  CI-safe bound; `FitWizardRecommendation` carries the rebin factor.
- Serialization of `DetectedPeak`/recommendation round-trips new fields
  (additive; old payloads load).

Gate 2: lead runs the four test files, `--tier fast`, then the private
harness: cold run → two-line recommendation with f within 2 % and λ within
30 % of the direct fit; 10 K-class run → single fast line; paramagnetic runs
→ no oscillation; wizard wall time < 20 s per run.

### Phase 3 — GUI, docs, changelog (Sonnet; Opus if the plot-panel plumbing turns out non-trivial)

Files: `gui/windows/fit_wizard_window.py` (peaks table column, dashed
markers, fingerprint rows, full-record analysis + fit-range note, seed
refinement), the plot-panel accessor for the uncropped dataset,
`tests/gui/test_fit_wizard_window.py` (or the existing window test file),
`docs/reference/fit_wizard.rst` ("Heavily damped lines" section rewritten
to describe the matched-apodisation scan, the Δχ² gate, the new templates,
the rebinned fitting note under "Convergence quality"/"Limitations", and
the fit-range note), `docs/screenshots/scenarios` if the fingerprint panel
changes visibly, `CHANGELOG.md` `[Unreleased]`.

Deliverables: UI strings quoted verbatim in the docs; GUI tests cover the
new column, the note, and that the wizard receives the uncropped dataset;
`harness.py docs` green.

Gate 3: lead runs the GUI test file focused, `structural`, `lint`, `docs`,
reads the docs diff against the widget strings.

### Phase 4 — close-out (lead agent)

`harness.py validate` once; private harness one last time; PR opened with
the acceptance table (synthetic numbers only); propose a release per
`RELEASING.md` after merge.

## Acceptance criteria (PR level)

- Blind recommendation of a damped multi-line model on the synthetic
  two-line record; no regression on the existing wizard corpus.
- No new false oscillation candidates on non-oscillatory synthetic records
  (0 / 100 seeds).
- Wizard wall time on a 10⁵-point record under 20 s locally.
- No research data, private paths, sample names or run numbers in the repo.
- Docs, changelog and screenshot registry updated in the same PR.

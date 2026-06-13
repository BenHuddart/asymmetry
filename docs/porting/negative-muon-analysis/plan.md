# Implementation plan — negative-muon-analysis (API-only, WIP, phased)

This plan is prescriptive: exact module paths, function signatures, docstring
stubs (including verbatim WIP disclaimer text), the constants table with
citations, per-package acceptance criteria, and the test plan with expected
values. It is written so a separate session (smaller model) can execute it cold.

**Standing rule for the implementer:** where this plan names an **existing API**,
use it. If an existing API appears not to fit, **STOP and ask Ben** rather than
writing a replacement or modifying out-of-scope code (especially
`core/fitting/count_domain.py`, which is read-only/off-limits here). Where this
plan leaves any genuine gap, **STOP and ask** rather than improvise.

All work happens in the worktree `~/Source/Asymmetry-worktrees/
negative-muon-analysis` on branch `feat/negative-muon-analysis`, using its
`.venv` (numpy 2.2.x). Validate with `python tools/harness.py …`.

---

## 0. Phasing

Six phases, each a **separate implementation session**. Phases 1–5 end with
validate-green + high-effort code-review + fix, then a **commit on the feature
branch** — **no push and no PR**. **Phase 6** is a whole-implementation close-out
review (run on **Opus**) that fixes any residual findings and then **pushes the
branch and opens the single PR** for the whole feature. Phases are ordered; later
phases depend on earlier. Each phase rebases onto `origin/main` at its start.

| Phase | Deliverable | New/edited files |
|-------|-------------|------------------|
| 1 | Element table + multi-exp model + single-group fitter + `simulate_capture_run` + tests | `core/negmu/__init__.py`, `lifetimes.py`, `model.py`, `fit.py`; `core/simulate.py` (+); `tests/negmu/…` |
| 2 | α-coupled F+B fit + capture-ratio report + tests | `core/negmu/fit.py` (+), `ratio.py`; `tests/negmu/…` |
| 3 | Set-as-BG subtraction + tests | `core/negmu/background.py`; `tests/negmu/…` |
| 4 | μ⁻SR polarisation slice (None/LorGau/Diamagnetic) + tests | `core/negmu/polarisation.py`, `fit.py` (+); `tests/negmu/…` |
| 5 | Docs: experimental user-guide page + API autodoc + toctree (commit only) | `docs/user_guide/negative_muon_analysis.rst`, `docs/api/*`, `docs/user_guide/index.rst` |
| 6 | **Opus** whole-implementation close-out review + fix → **push + single PR** | review fixes only |

Phase 5 (docs) may be threaded into the earlier phases (recommended: stub the page
in Phase 1 with the WIP admonition + element-table section, extend per phase) —
but the push + PR is always **Phase 6 only**. Every phase rebases onto
`origin/main` at start (Wave B in flight — `README.md` §"Repo awareness").

---

## 1. Reuse audit (binding)

For each work package: the **existing** functions/classes it builds on (import
paths), then a one-line justification for each **new** module.

### Existing APIs consumed (read-only; do not modify)

| Existing API | Import path | Used by |
|---|---|---|
| `build_count_group`, `build_count_groups`, `GroupedTimeDomainGroup`, `GroupedTimeDomainFitResult` | `asymmetry.core.fitting.grouped_time_domain` | WP1.3, WP2.1 — raw (time,counts) per group (`lifetime_corrected=False`); F+B result bundle |
| `drive_minuit`, `FitResult`, `_make_cancel_guard`, `_minuit_status_message` | `asymmetry.core.fitting.engine` | WP1.3, WP2.1 — minimiser drive (migrad+HESSE+MINOS), result container, cancel/status |
| `Parameter`, `ParameterSet` | `asymmetry.core.fitting.parameters` | WP1.3+ — all parameter machinery (fix/limits/`expr`/`link_group`, `free_parameters`, `link_followers`) |
| `assess_fit_quality` | `asymmetry.core.fitting.fit_quality` | WP1.3 — χ²/Cash good/poor/overdone verdict (pass `FitResult.dof`) |
| `fit_result_summary` | `asymmetry.core.fitting.result_summary` | WP2.2, docs — JSON-serialisable summary shape |
| `MUON_LIFETIME_US` | `asymmetry.core.utils.constants` | WP1.1 — decay-BG τ_μ (= Appendix C 2196.981 ns to 6 digits) |
| `_sample_and_build_run`, `_synthetic_run_grouping`, `_synthetic_run_metadata` | `asymmetry.core.simulate` | WP1.4 — seeded-Poisson sampling + Run/provenance (called in-module by `simulate_capture_run`) |
| `MuonDataset`, `Run`, `Histogram` | `asymmetry.core.data.dataset`, `asymmetry.core.data.run` | data contracts |
| `subtract_scaled_counts` (pattern reference) | `asymmetry.core.data.combine` (run-arithmetic) | WP3 — histogram-level subtraction convention (Set-as-BG is model-evaluated, not run-vs-run; reuse the elementwise convention, not the function, unless it fits — if it does, use it) |

### New modules — justification

| New module | Why it cannot reuse an existing one |
|---|---|
| `core/negmu/lifetimes.py` | New literature-anchored data asset (no element lifetime table exists). |
| `core/negmu/model.py` | Multi-τ raw-count model `Σ_i N_i e^{−t/τ_i}+bg` is not expressible by `count_domain` (single envelope) or the asymmetry `MODELS` (`comparison.md` §3). |
| `core/negmu/fit.py` | Needs Poisson **Cash** cost on raw counts; `FitEngine` is Gaussian-on-asymmetry, `count_domain`'s Cash is private + off-limits. Reuses `drive_minuit`/`FitResult`/`ParameterSet`; only the ~6-line cost is local. |
| `core/negmu/ratio.py` | Derived-quantity report; no existing capture-ratio function. Reuses `FitResult` (values, covariance). |
| `core/negmu/background.py` | Set-as-BG is model-evaluated component subtraction; reuses `model.py` to evaluate, elementwise subtract. |
| `core/negmu/polarisation.py` | μ⁻SR multiplier functions; package-local, unregistered (no GUI). |
| `simulate_capture_run` (in existing `core/simulate.py`) | Multi-τ synthesis; the existing generators bake in one `τ_μ` envelope. Additive function reusing the sampler (`test-data.md`). |

**The two adapt-vs-new near-misses** (`implementation-options.md` §"adapt-vs-new"):
the Cash cost (replicate, don't import-private/modify) and synthetic data (add
`simulate_capture_run` reusing `_sample_and_build_run`). Both are recorded
decisions — implement as written; if either looks wrong, STOP and ask.

---

## 2. Element lifetime table (the constants asset)

`core/negmu/lifetimes.py` transcribes **Table C.1** of Blundell, De Renzi,
Lancaster & Pratt, *Muon Spectroscopy: An Introduction* (OUP, 2022), "Negative
muon lifetimes" (Appendix C), whose values "have been obtained by combining the
measurements listed in" **T. Suzuki, D. F. Measday & J. P. Roalsvig,
*Phys. Rev. C* 35, 2212 (1987)**. Values below are **μs** (Table C.1 is ns;
divide by 1000). `source = "SuzukiMeasdayRoalsvig1987"` unless marked
**[WiMDA-prov]** (in WiMDA's older table but **not** in Table C.1; set
`source = "WiMDA-provisional"`, `sigma_us = None`).

Transcribe the `τ(μs)`/`σ(μs)` columns verbatim. Rows marked **⚠confirm** are
ones whose exact element↔value assignment in the periodic-table layout of
Table C.1 must be **visually confirmed against the textbook** (Appendix C,
p. 359) before transcription — the flattened-text extraction was ambiguous for
the period-5 transition cluster and the lanthanides/actinides. Do **not** use
⚠confirm rows as test spot-check anchors.

### Confident rows (cross-validated vs WiMDA to ≤ rounding)

| Z | Sym | Name | τ (μs) | σ (μs) | Z | Sym | Name | τ (μs) | σ (μs) |
|---|-----|------|--------|--------|---|-----|------|--------|--------|
| 1 | H | Hydrogen | 2.19480 | 0.00006 | 30 | Zn | Zinc | 0.161 | 0.001 |
| 3 | Li | Lithium | 2.1869 | 0.0004 | 31 | Ga | Gallium | 0.163 | 0.002 |
| 4 | Be | Beryllium | 2.16747 | 0.00089 | 32 | Ge | Germanium | 0.167 | 0.001 |
| 5 | B | Boron | 2.097 | 0.003 | 33 | As | Arsenic | 0.153 | 0.001 |
| 6 | C | Carbon | 2.030 | 0.001 | 34 | Se | Selenium | 0.163 | 0.001 |
| 7 | N | Nitrogen | 1.920 | 0.002 | 35 | Br | Bromine | 0.133 | 0.001 |
| 8 | O | Oxygen | 1.795 | 0.002 | 37 | Rb | Rubidium | 0.137 | 0.003 |
| 9 | F | Fluorine | 1.461 | 0.005 | 38 | Sr | Strontium | 0.132 | 0.002 |
| 10 | Ne | Neon | 1.461 | 0.009 | 39 | Y | Yttrium | 0.120 | 0.001 |
| 11 | Na | Sodium | 1.204 | 0.002 | 40 | Zr | Zirconium | 0.110 | 0.001 |
| 12 | Mg | Magnesium | 1.069 | 0.002 | 41 | Nb | Niobium | 0.092 | 0.001 |
| 13 | Al | Aluminium | 0.864 | 0.001 | 42 | Mo | Molybdenum | 0.104 | 0.001 |
| 14 | Si | Silicon | 0.759 | 0.001 | 55 | Cs | Cesium | 0.088 | 0.002 |
| 15 | P | Phosphorus | 0.616 | 0.001 | 56 | Ba | Barium | 0.0949 | 0.0006 |
| 16 | S | Sulfur | 0.555 | 0.001 | 72 | Hf | Hafnium | 0.075 | 0.001 |
| 17 | Cl | Chlorine | 0.561 | 0.002 | 73 | Ta | Tantalum | 0.0755 | 0.0006 |
| 18 | Ar | Argon | 0.537 | 0.032 | 74 | W | Tungsten | 0.0765 | 0.0008 |
| 19 | K | Potassium | 0.435 | 0.001 | 79 | Au | Gold | 0.0728 | 0.0005 |
| 20 | Ca | Calcium | 0.336 | 0.001 | 80 | Hg | Mercury | 0.076 | 0.001 |
| 21 | Sc | Scandium | 0.317 | 0.003 | 81 | Tl | Thallium | 0.0704 | 0.0008 |
| 22 | Ti | Titanium | 0.329 | 0.001 | 82 | Pb | Lead | 0.0747 | 0.0004 |
| 23 | V | Vanadium | 0.280 | 0.002 | 83 | Bi | Bismuth | 0.0735 | 0.0004 |
| 24 | Cr | Chromium | 0.259 | 0.002 | | | | | |
| 25 | Mn | Manganese | 0.231 | 0.001 | | | | | |
| 26 | Fe | Iron | 0.206 | 0.001 | | | | | |
| 27 | Co | Cobalt | 0.186 | 0.001 | | | | | |
| 28 | Ni | Nickel | 0.157 | 0.001 | | | | | |
| 29 | Cu | Copper | 0.164 | 0.001 | | | | | |

### ⚠confirm rows (transcribe after visual check of Table C.1)

| Z | Sym | Name | τ (μs) | σ (μs) | Z | Sym | Name | τ (μs) | σ (μs) |
|---|-----|------|--------|--------|---|-----|------|--------|--------|
| 44 | Ru | Ruthenium | 0.0958 | 0.0006 | 60 | Nd | Neodymium | 0.0784 | 0.0007 |
| 45 | Rh | Rhodium | 0.0960 | 0.0006 | 62 | Sm | Samarium | 0.079 | 0.001 |
| 46 | Pd | Palladium | 0.0885 | 0.0006 | 64 | Gd | Gadolinium | 0.0806 | 0.0008 |
| 47 | Ag | Silver | 0.0906 | 0.0007 | 65 | Tb | Terbium | 0.0762 | 0.0007 |
| 48 | Cd | Cadmium | 0.0906 | 0.0007 | 67 | Ho | Holmium | 0.079 | 0.001 |
| 50 | Sn | Tin | 0.0907 | 0.0008 | 68 | Er | Erbium | 0.0749 | 0.0006 |
| 51 | Sb | Antimony | 0.0924 | 0.0009 | 69 | Tm | Thulium | 0.074 | 0.002 |
| 52 | Te | Tellurium | 0.104 | 0.001 | 90 | Th | Thorium | 0.0780 | 0.0003 |
| 53 | I | Iodine | 0.0856 | 0.0006 | 92 | U | Uranium | 0.0775 | 0.0002 |
| 57 | La | Lanthanum | 0.0899 | 0.0007 | 93 | Np | Neptunium | 0.0720 | 0.0007 |
| 58 | Ce | Cerium | 0.0840 | 0.0006 | | | | | |
| 59 | Pr | Praseodymium | 0.0721 | 0.0006 | | | | | |

### [WiMDA-prov] rows (not in Table C.1 — `source="WiMDA-provisional"`, σ=None)

| Z | Sym | Name | τ (μs) | Note |
|---|-----|------|--------|------|
| 2 | He | Helium | 2.188 | capture negligible; from WiMDA `mystrings` |
| 36 | Kr | Krypton | 0.136 | from WiMDA |
| 43 | Tc | Technetium | 0.095 | radioactive; from WiMDA |
| 75 | Re | Rhenium | 0.076 | from WiMDA |
| 76 | Os | Osmium | 0.078 | from WiMDA |
| 77 | Ir | Iridium | 0.074 | from WiMDA |
| 78 | Pt | Platinum | 0.074 | from WiMDA |

> Provenance note to put in the module docstring: WiMDA's `mystrings` table
> (`NegMuAnalyse.pas:104–120`) was the workflow reference, **not** the value
> source. It has a 69-vs-67 length mismatch, the `'Ti'`→`'Tl'` symbol bug, and
> several value divergences (Ne 1.520→1.461, Zn 0.169→0.161, Sr 0.142→0.132,
> Ba 0.072→0.0949). Adopt Table C.1; cite Suzuki/Measday/Roalsvig 1987.

The **decay-background** lifetime is **not** a table entry — it reuses
`MUON_LIFETIME_US` (2.1969811 μs) under the reserved label `decayBG`.

---

## 3. Work packages

### Phase 1

#### WP1.1 — `core/negmu/lifetimes.py`

```python
"""<WIP banner — see §5>

Negative-muon capture lifetimes: an element-keyed table of muonic-atom total
disappearance lifetimes τ(Z) = 1/(Λ_capture + Λ_bound-decay), the seeds for the
multi-exponential capture fit.

Source: Blundell, De Renzi, Lancaster & Pratt, *Muon Spectroscopy: An
Introduction* (OUP, 2022), Appendix C, Table C.1 — values combined from
T. Suzuki, D. F. Measday & J. P. Roalsvig, Phys. Rev. C 35, 2212 (1987).
A few entries marked WiMDA-provisional are from WiMDA's older table where
Table C.1 has no value. WiMDA's table is NOT trusted for values (see the
porting study, comparison.md §5).
"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class ElementLifetime:
    symbol: str
    z: int
    name: str
    tau_us: float
    sigma_us: float | None
    source: str  # "SuzukiMeasdayRoalsvig1987" | "WiMDA-provisional"

#: Reserved label for the free-μ⁻ decay-background component (τ = MUON_LIFETIME_US).
DECAY_BACKGROUND_LABEL: str = "decayBG"

#: Element symbol -> ElementLifetime. Transcribed from §2 of the plan.
ELEMENT_LIFETIMES: dict[str, ElementLifetime] = { ... }  # all §2 rows

def lifetime(symbol: str) -> ElementLifetime:
    """Return the table entry for `symbol` (KeyError if absent)."""
def tau_us(symbol: str) -> float:
    """Return the capture lifetime (μs) for `symbol`."""
def has_element(symbol: str) -> bool: ...
def elements() -> list[str]:
    """Element symbols present, ordered by Z."""
```

**Acceptance:** §1 spot-checks in `verification-plan.md` pass (exact values,
`Tl`/`Ne` guards, ranges, source field); module imports without Qt.

#### WP1.2 — `core/negmu/model.py`

```python
"""<WIP banner>

Multi-exponential μ⁻ capture count model:  N(t) = Σ_i amp_i·exp(−t/τ_i) + bg.
A raw-count model, NOT an asymmetry model — see the study comparison.md §3 for
why count_domain's single-envelope model cannot express this.
"""
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

@dataclass(frozen=True)
class CaptureComponent:
    label: str       # element symbol, or DECAY_BACKGROUND_LABEL
    tau_us: float    # lifetime seed (μs)

def build_capture_count_model(
    components: Sequence[CaptureComponent],
) -> Callable[..., NDArray[np.float64]]:
    """Return f(t, **params) -> raw counts for the fixed component order.

    Recognised params: ``amp_<label>`` (per component), optional ``tau_<label>``
    (overrides the component seed when the lifetime is freed), and ``background``
    (flat). Unknown params are ignored. Vectorised over t (μs)."""

def evaluate_capture_model(
    components: Sequence[CaptureComponent],
    params: dict[str, float],
    t: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convenience: build + call in one step (used by background.py and tests)."""
```

**Acceptance:** model equals a direct numpy `Σ amp·exp(−t/τ)+bg` at sample points
(`pytest.approx`); a fixed `tau_<label>` override changes the curve; no Qt import.

#### WP1.3 — `core/negmu/fit.py` (single-group)

```python
"""<WIP banner>

Fit a μ⁻ capture-lifetime histogram (raw counts) to the multi-exponential model.
Reuses the shared minimiser drive (drive_minuit), FitResult, and
Parameter/ParameterSet. The Poisson (Cash) and Gaussian (√N) count costs are
replicated here (count_domain's are private and that module is off-limits).
Lifetimes are FIXED at the table value by default; free any via spec.free_tau.
"""
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
from asymmetry.core.fitting.engine import FitResult, drive_minuit, _make_cancel_guard, _minuit_status_message
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fitting.grouped_time_domain import build_count_group
from asymmetry.core.negmu.lifetimes import tau_us, DECAY_BACKGROUND_LABEL
from asymmetry.core.negmu.model import CaptureComponent, build_capture_count_model
from asymmetry.core.utils.constants import MUON_LIFETIME_US

COUNT_COSTS: tuple[str, ...] = ("poisson", "gaussian")

@dataclass(frozen=True)
class CaptureModelSpec:
    """Which elemental components are in the fit, with τ fixed by default."""
    elements: tuple[str, ...]
    include_decay_background: bool = True
    free_tau: frozenset[str] = field(default_factory=frozenset)

    def components(self) -> tuple[CaptureComponent, ...]:
        """element components (τ from table) + optional decayBG (τ_μ)."""
    def labels(self) -> tuple[str, ...]: ...

def default_capture_parameters(
    spec: CaptureModelSpec, *, time: NDArray, counts: NDArray,
    seeds: Mapping[str, float] | None = None,
) -> ParameterSet:
    """Seeded ParameterSet: amp_<label> (free, ≥0, coarse-split of the window
    sum), tau_<label> (fixed unless label in spec.free_tau; freed τ bounded to a
    sensible ± window), background (free, ≥0, seeded from the late-time floor).
    `seeds` overrides any amp_/tau_/background value."""

def fit_capture_histogram(
    time: NDArray[np.float64], counts: NDArray[np.float64], spec: CaptureModelSpec,
    *, variance: NDArray | None = None, cost: str = "poisson",
    parameters: ParameterSet | None = None, minos: bool = False,
    cancel_callback: Callable[[], bool] | None = None,
) -> FitResult:
    """Fit raw counts to Σ_i amp_i·exp(−t/τ_i)+bg. The array-level entry: testable
    without a dataset. `variance` (if given) is used by the Gaussian cost;
    Poisson (Cash) ignores it. Returns a FitResult with dof set so
    assess_fit_quality applies."""

def fit_capture_group(
    dataset, group_id: int, spec: CaptureModelSpec, *, t_min: float | None = None,
    t_max: float | None = None, cost: str = "poisson",
    exclude: tuple[float, float] | None = None, minos: bool = False,
    cancel_callback: Callable[[], bool] | None = None,
) -> FitResult:
    """Build the raw (time,counts) trace for `group_id` via
    grouped_time_domain.build_count_group(..., lifetime_corrected=False) and fit.
    The dataset-level entry for real μ⁻ runs."""
```

Cost replication (verbatim form, cite source in a comment
`# Cash statistic, cf. count_domain._poisson_cash (private; replicated, not imported)`):

```python
def _poisson_cash(counts, model):
    mu = np.clip(model, 1.0e-12, None)
    term = mu - counts
    pos = counts > 0.0
    term[pos] += counts[pos] * np.log(counts[pos] / mu[pos])
    return 2.0 * float(np.sum(term))

def _gaussian_chi2(counts, model, variance=None):
    sigma2 = variance if variance is not None else np.clip(counts, 1.0, None)
    return float(np.sum((counts - model) ** 2 / np.clip(sigma2, 1e-12, None)))
```

The Minuit driver mirrors `count_domain._solve`: build `Minuit(cost, *initial,
name=names)`, set limits from the free `Parameter`s, `cost.errordef = 1.0`, call
`drive_minuit(m, migrad_kwargs={"iterate": 5, "use_simplex": True}, minos=minos)`,
then pack a `FitResult` (values, HESSE σ in `uncertainties`, covariance +
`covariance_parameters`, `residuals = counts − model`, `dof = N − N_free`,
`minos_errors`). **Reuse `_make_cancel_guard` and `_minuit_status_message`.**

**Acceptance:** `verification-plan.md` §2 cases 2a/2b/2c/2e and §4 guards pass;
recovered ratios within tolerance; `assess_fit_quality(result.chi_squared,
result.dof)` verdict `"good"` on clean data.

#### WP1.4 — `simulate_capture_run` in `core/simulate.py`

```python
def simulate_capture_run(
    template: Run,
    components: Sequence["CaptureComponent"],
    weights: Mapping[str, float],
    *,
    total_events: float,
    group_id: int | None = None,
    seed: int = 0,
    background_per_bin: float = 0.0,
    run_number: int | None = None,
    title: str | None = None,
) -> Run:
    """Synthesise a μ⁻ capture-lifetime run: per detector,

        N_d(t) = Σ_i N_{i,d}·exp(−(t−t0)/τ_i) + b      (t ≥ t0)

    with the component populations split by `weights` (relative, normalised over
    the components) and the per-bin envelope using the same exact telescoping
    normalisation as `expected_counts` (n0_i = N_{i,d}·(1−exp(−Δt/τ_i))), so the
    post-t0 window sum ≈ total_events. Detectors in `group_id` (or all detectors
    if None) carry the signal; `background_per_bin` is added in addition to the
    event budget. Poisson-sampled with `seed` and assembled (provenance,
    deadtime-zeroing) via the shared `_sample_and_build_run`. Provenance records
    `capture_mode=True`, the components/τ and weights, and the seed.
    """
```

Build `expected: list[NDArray]` directly, then
`return _sample_and_build_run(template, expected, seed=seed, total_events=...,
background_per_bin=..., run_number=..., title=..., default_title="Simulated μ⁻
capture run", simulation_metadata={"capture_mode": True, "components": [...],
"weights": {...}})`. Do **not** duplicate sampling/metadata — reuse the helper.

**Acceptance:** `verification-plan.md` §5 round-trip + provenance + bit-for-bit
seed reproducibility pass.

### Phase 2

#### WP2.1 — α-coupled forward/backward fit (`core/negmu/fit.py` +)

```python
def fit_capture_fb_alpha(
    dataset, forward_group: int, backward_group: int, spec: CaptureModelSpec, *,
    cost: str = "poisson", t_min: float | None = None, t_max: float | None = None,
    exclude: tuple[float, float] | None = None, alpha_seed: float = 1.0,
    minos: bool = False, cancel_callback: Callable[[], bool] | None = None,
) -> "GroupedTimeDomainFitResult":
    """Simultaneously fit forward and backward capture histograms with shared
    per-element amplitudes amp_<label>, shared τ_i, and a free detector balance α:

        N_F(t) = √α · Σ_i amp_i·exp(−t/τ_i) + bg_F
        N_B(t) = (1/√α) · Σ_i amp_i·exp(−t/τ_i) + bg_B

    mirroring grouped_time_domain.build_count_group geometry and count_domain's
    √α split. Both banks built in ONE context (build_count_groups) for a common
    t0. Returns a GroupedTimeDomainFitResult: group_results keyed by
    forward/backward group, shared_parameters holding α + amplitudes + τ.

    DIVERGENCE from WiMDA: WiMDA fits independent per-side amplitudes (NF, NB);
    this shares amp_i (isotropic capture populations), so per-side capture ratios
    are identical by construction. Use the per-group fit_capture_group on each
    side when a genuine F/B amplitude difference is wanted.
    """
```

Clamp α positive (mirror `count_domain._clamp_alpha_positive` behaviour locally:
floor 1e-6, seed |alpha_seed|). Reuse `build_count_groups`, `drive_minuit`,
`GroupedTimeDomainFitResult`, the WP1.3 cost + packing.

**Acceptance:** `verification-plan.md` §2 case 2d passes (α within 3σ of 1.25,
ratios as 2a).

#### WP2.2 — capture-ratio report (`core/negmu/ratio.py`)

```python
"""<WIP banner>

Capture-ratio report: relative capture probabilities from fitted amplitude
ratios amp_i/amp_ref, with covariance-aware uncertainties. Adapts WiMDA's
RatioButtonClick (NegMuAnalyse.pas:455–620) as a derived-quantities function —
no new results framework.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class CaptureRatio:
    numerator: str
    denominator: str
    ratio: float
    sigma: float

@dataclass(frozen=True)
class CaptureRatioReport:
    side: str           # "forward" | "backward" | "combined"
    reference: str
    ratios: tuple[CaptureRatio, ...]
    amplitudes: dict[str, float]
    amplitude_uncertainties: dict[str, float]

def capture_ratio_report(
    fit: "FitResult", spec: CaptureModelSpec, *, reference: str, side: str = "forward",
) -> CaptureRatioReport:
    """For each element label != reference, ratio = amp_label/amp_reference with

        σ_R = R·sqrt((σ_i/amp_i)² + (σ_ref/amp_ref)² − 2·cov(i,ref)/(amp_i·amp_ref))

    using fit.covariance/covariance_parameters when both params are present,
    falling back to quadrature (cov term 0) otherwise. Excludes the decayBG
    component by default."""

def fb_capture_ratio_report(
    grouped: "GroupedTimeDomainFitResult", spec: CaptureModelSpec,
    forward_group: int, backward_group: int, *, reference: str,
) -> dict[str, CaptureRatioReport]:
    """Per-side reports {'forward': ..., 'backward': ...} from a F+B fit."""
```

**Acceptance:** `verification-plan.md` §3 Fixtures A (quadrature) and B
(covariance-aware) reproduce the hand-computed 2.00(10) / 2.00(9) exactly.

### Phase 3

#### WP3 — Set-as-BG (`core/negmu/background.py`)

```python
def subtract_capture_background(
    time: NDArray, counts: NDArray, fit: "FitResult", spec: CaptureModelSpec,
    *, unwanted: Sequence[str],
) -> NDArray[np.float64]:
    """Return counts − Σ_{label in unwanted} amp_label·exp(−t/tau_label),
    evaluating the unwanted components from the fitted parameters via
    model.evaluate_capture_model. The remaining signal of interest. (WiMDA
    SetBgButtonClick adapted as a histogram-level model subtraction.)"""

def capture_background_run(
    dataset, group_id: int, fit: "FitResult", spec: CaptureModelSpec,
    *, unwanted: Sequence[str], run_number: int | None = None,
) -> "Run":
    """Optional: a derived Run with the unwanted components subtracted from the
    selected group's histogram, for re-fitting. Reuse the run-arithmetic
    histogram-level subtraction convention; if subtract_scaled_counts fits the
    contract, use it — otherwise build the derived Run minimally and STOP/ask if
    a provenance question arises."""
```

**Acceptance:** subtracting all-but-one component from a synthetic two-component
histogram leaves the retained component within Poisson tolerance of its
generating curve; a no-unwanted call is the identity.

### Phase 4

#### WP4 — μ⁻SR polarisation slice (`core/negmu/polarisation.py`, `fit.py` +)

```python
def lorentzian_gaussian_polarisation(t, a0, lam, freq, phase) -> NDArray:
    """A·exp(−λt)·cos(2π·freq·t + phase) style μ⁻SR polarisation (WiMDA LorGau)."""
def diamagnetic_polarisation(t, a0, freq, phase) -> NDArray:
    """Undamped diamagnetic precession a0·cos(2π·freq·t + phase) (WiMDA Diamagnetic)."""
```

Extend the model/fit to optionally multiply the capture model by `(1 + P_pol(t))`
(WiMDA's None/LorGau/Diamagnetic multiplier) via an optional `polarisation`
argument on a new `build_capture_count_model_with_polarisation` and a
`polarisation=` kwarg on a fit entry. **Package-local, unregistered.** Confirm
the exact polarisation parameterisation against `NegMuAnalyse.pas` (the
`MuonPolarity`/polarisation controls) and the textbook §22.1 before finalising;
if the WiMDA form is ambiguous, STOP and ask.

**Acceptance:** a synthetic capture+precession histogram recovers the
polarisation frequency within tolerance; with `polarisation=None` the model is
bit-identical to Phase 1.

### Phase 5 — docs (see §6).

---

## 4. Test plan (files)

All under `tests/negmu/` (new), mirroring the package. Expected values are in
`verification-plan.md`; summary:

| Test file | Covers | Key assertions |
|---|---|---|
| `tests/negmu/test_lifetimes.py` | WP1.1 | §1 exact spot-checks, Tl/Ne guards, ranges, source, no-Qt |
| `tests/negmu/test_model.py` | WP1.2 | model == numpy Σ amp·exp+bg; τ override |
| `tests/negmu/test_simulate_capture.py` | WP1.4 | §5 round-trip, provenance, seed reproducibility |
| `tests/negmu/test_fit_single.py` | WP1.3 | §2 cases 2a/2b/2c/2e; quality verdict; dof |
| `tests/negmu/test_fit_fb_alpha.py` | WP2.1 | §2 case 2d (α recovery) |
| `tests/negmu/test_ratio.py` | WP2.2 | §3 Fixtures A & B exact |
| `tests/negmu/test_background.py` | WP3 | retained-component recovery; identity |
| `tests/negmu/test_polarisation.py` | WP4 | freq recovery; None == Phase-1 model |
| `tests/negmu/test_no_gui.py` | all | not in COMPONENTS/MODELS; core imports without `asymmetry.gui`; count_domain not imported |

Reuse the existing core/GUI-isolation test pattern (grep `tests/` for the
import-isolation test that asserts core does not import Qt) rather than inventing
one. Use `simulate_capture_run` for all synthetic histograms — no inline
generators.

---

## 5. WIP disclaimer text (verbatim)

### Module-docstring banner (top of every `core/negmu/*.py`)

```
EXPERIMENTAL — WORK IN PROGRESS. Negative-muon (μ⁻) capture-lifetime analysis.

This API is UNVALIDATED against real μ⁻ elemental-analysis data. No μ⁻ corpus
exists in this project; every result here has been exercised only against
synthetic histograms. The element lifetime values are literature-anchored
(Suzuki, Measday & Roalsvig, Phys. Rev. C 35, 2212 (1987), via Blundell et al.,
Muon Spectroscopy: An Introduction, OUP 2022, Table C.1), but the fitting,
capture-ratio, and background machinery have NOT been checked against an
established tool (WiMDA, Mantid) on measured data. The API, parameter names, and
return shapes MAY CHANGE without notice. Do not rely on results for publication
without independent verification. This feature is deliberately NOT exposed in the
GUI fit builders. Promotion trigger for a GUI: real ISIS μ⁻ data AND a user.
```

### Sphinx admonition (top of the docs page, after the title)

```rst
.. warning::

   **Experimental — negative-muon analysis is a work in progress.**

   This page documents a scriptable, **API-only** μ⁻ capture-lifetime analysis
   that is **unvalidated against real μ⁻ data**. It has been exercised only on
   synthetic histograms. The element lifetimes are literature-anchored
   (Suzuki, Measday & Roalsvig 1987), but the fitting and capture-ratio
   machinery have not been checked against an established μ⁻ tool on measured
   data, and the API may change. There is **no GUI** for this feature. Verify
   any physical interpretation against the primary literature and an established
   tool (WiMDA, Mantid) before relying on it.
```

---

## 6. Docs deliverable spec (Phase 5)

New page `docs/user_guide/negative_muon_analysis.rst`, in Ben's style
(result-first physics prose; rendered `.. math::`; uncertainties as `0.23(1)`;
APS references in a list, not inline; "when to use this" register; no textbook
equation numbers — cite the textbook by name). Add to
`docs/user_guide/index.rst` under a new caption **"Negative muon (experimental)"**
and add an autodoc entry in `docs/api/` for `asymmetry.core.negmu`.

Page outline:

1. **WIP admonition** (verbatim, §5).
2. **What this measures** — result-first: a μ⁻ implants, forms a muonic atom at
   the lattice site, and disappears with an element-characteristic lifetime; a
   mixed sample's decay-electron histogram is a sum of exponentials, and the
   amplitude ratios give relative capture probabilities → composition.
3. **The model** — rendered math
   `N(t) = \sum_i N_i e^{-t/\tau_i} + N_\mathrm{bg} e^{-t/\tau_\mu} + b`.
4. **The lifetime table** — provenance (Table C.1 / Suzuki 1987), the fixed-τ
   default, how to free a τ; note WiMDA divergences were corrected.
5. **When to use this** — diagnostic register: reach for it for μ⁻ capture-
   lifetime elemental analysis on raw single-histogram counts; not for μSR spin
   relaxation; not for μ-XRF (muonic X-ray) analysis; expect to assert the
   candidate elements (fixed τ) rather than free everything.
6. **Worked example** — scriptable snippet: `simulate_capture_run` →
   `fit_capture_group` → `capture_ratio_report`, showing a recovered ratio as
   e.g. `1.67(5)`.
7. **Forward/backward & α** — the coupled fit; the shared-amplitude divergence.
8. **References** (APS list): Suzuki, Measday & Roalsvig, *Phys. Rev. C* 35,
   2212 (1987); D. F. Measday, *Phys. Rep.* 354, 243 (2001); Blundell et al.,
   *Muon Spectroscopy: An Introduction* (OUP, 2022), Ch. 22 & App. C.

`python tools/harness.py docs` must build clean.

---

## 7. Non-goals (explicit)

- **No GUI** — no panels, dialogs, menu entries, toolbar buttons, or plot modes;
  **no registration** in `COMPONENTS`/`MODELS` (no picker exposure).
- **No real-data validation claims** — synthetic-only; the WIP disclaimer stands.
- **No new fitting engine, parameter machinery, or results framework** — reuse
  `drive_minuit`/`FitResult`/`Parameter`/`ParameterSet`/`assess_fit_quality`/
  `fit_result_summary`.
- **No modification of `core/fitting/count_domain.py`** (or any out-of-scope
  module) — replicate the small cost; do not re-export/move private helpers.
- **μ-XRF / muonic X-ray elemental analysis out of scope** — lifetime method only
  (consistent with `wimda-parity-gap/decision-record.md`).
- **No GLE export, no `PlotPar` decay-correction plot hook** (GUI plot territory).

---

## 8. Ready-to-paste implementation-session prompt (Phase 1)

> Copy the block below into a fresh session to execute **Phase 1**. For later
> phases, reuse it changing the phase number, the "READ FIRST" phase pointer, and
> the "EXECUTE" line. **Workflow (Ben):** Phases 1–5 each end with code-review +
> fix + a commit on this feature branch — **no push, no PR**. **Phase 6** is a
> whole-implementation close-out review run on **Opus** that fixes residual
> findings and then **pushes + opens the single PR** (pre-authorised). So the
> Phase-1 (and 2–5) end-stage is commit-only; only Phase 6 pushes.

```
Implement Phase 1 of the negative-muon-analysis plan (API-only, work-in-progress).

SETUP (reuse the existing worktree + branch from the study session):
- cd ~/Source/Asymmetry-worktrees/negative-muon-analysis
- Confirm: git branch --show-current  → feat/negative-muon-analysis
- Use this worktree's .venv (.venv/bin/python; numpy 2.2.x). Verify:
  .venv/bin/python -c "import asymmetry, numpy; print(asymmetry.__file__, numpy.__version__)"
- git fetch origin && git rebase origin/main   (Wave B may have merged; resolve
  trivially — your surfaces are disjoint: new core/negmu/ package + one additive
  function in core/simulate.py + study/docs).

READ FIRST (source of truth — do not deviate):
- docs/porting/negative-muon-analysis/plan.md  (this plan; Phase 1 = WP1.1–WP1.4)
- docs/porting/negative-muon-analysis/comparison.md  (why count_domain/FitEngine
  do NOT fit the multi-exp model; the no-GUI-exposure mechanism)
- docs/porting/negative-muon-analysis/verification-plan.md  (expected test values)
- docs/porting/negative-muon-analysis/README.md  (scope decisions; reuse rule)

EXECUTE Phase 1 work packages IN ORDER (WP1.1 lifetimes → WP1.2 model → WP1.3
single-group fit → WP1.4 simulate_capture_run), each with its tests
(tests/negmu/...). Follow the signatures, docstring stubs, the verbatim WIP
disclaimer banner, and the §2 element-lifetime table EXACTLY. Transcribe the
confident table rows; for ⚠confirm rows, transcribe the plan's value (they are
not test anchors). Validate green per package:
  python tools/harness.py test -- tests/negmu/test_<package>.py
then the full ladder before finishing:
  python tools/harness.py validate

BINDING RULES:
- Reuse where the plan names an existing API (drive_minuit, FitResult,
  Parameter/ParameterSet, build_count_group, assess_fit_quality,
  _sample_and_build_run, MUON_LIFETIME_US). Do NOT build a parallel fitter,
  parameter machine, or results framework.
- Do NOT modify core/fitting/count_domain.py or any out-of-scope module.
  Replicate the ~6-line Cash/Gaussian cost locally (cite count_domain in a
  comment). If a named existing API appears not to fit, or the plan leaves a
  gap, STOP and ask Ben — do not improvise a replacement.
- No GUI: do NOT register anything in COMPONENTS or MODELS. core/negmu must not
  import asymmetry.gui or Qt.

END-OF-PHASE STAGE (commit only — do NOT push or open a PR):
1. python tools/harness.py validate  (must be green) and
   python tools/harness.py docs      (must build).
2. Run /code-review at HIGH effort on the diff; fix every confirmed finding,
   then re-run validate to confirm still green.
3. Commit on this feature branch (feat/negative-muon-analysis) with a message
   like "feat(negmu): Phase 1 — μ⁻ capture-lifetime API (element table, model,
   fit, simulate)" summarising the phase and the WIP/experimental status.
4. Do NOT push and do NOT open a PR. Ben pushes + opens one PR after the final
   phase. Report what landed and that Phase 2 is next.
```

### Phase 6 — Opus close-out review → push + PR (the only push/PR session)

> Run this session on **Opus** (max-capability). It adds no features — it reviews
> the whole branch, fixes residual findings, and ships.

```
Phase 6 (final) of the negative-muon-analysis feature: a whole-implementation
close-out review of the entire branch, then push + open the single PR. Run on
Opus. Phases 1–5 are committed on feat/negative-muon-analysis; no new features.

SETUP: cd ~/Source/Asymmetry-worktrees/negative-muon-analysis; confirm branch
feat/negative-muon-analysis; use .venv (numpy 2.2.x); git fetch origin && git
rebase origin/main; confirm `python tools/harness.py validate` and `docs` are
green BEFORE reviewing (clean baseline).

READ: docs/porting/negative-muon-analysis/plan.md (all phases, Reuse audit §1,
Non-goals §7, WIP disclaimer §5), comparison.md, verification-plan.md.

REVIEW the full feature diff vs main:
  git diff origin/main -- src/asymmetry/core/negmu src/asymmetry/core/simulate.py \
    tests/negmu docs/user_guide/negative_muon_analysis.rst docs/api docs/user_guide/index.rst
Run /code-review at MAX effort over it, checking specifically:
- Physics correctness: Cash/Gaussian cost; multi-exp model + √α F/B split;
  covariance-aware ratio propagation (verify it uses fit.covariance /
  covariance_parameters); amplitude seeding; the Phase-4 polarisation
  parameterisation (MHz vs rad/μs, damping/phase conventions — the flagged WiMDA
  unit ambiguity).
- Reuse discipline: drive_minuit/FitResult/Parameter/ParameterSet/
  build_count_group(s)/_sample_and_build_run/assess_fit_quality reused, not
  re-implemented; no third copy of the cost/driver; count_domain.py UNMODIFIED and
  not imported by core/negmu.
- No-GUI invariant: nothing in COMPONENTS/MODELS; no Qt/asymmetry.gui import
  (test_no_gui meaningful).
- WIP disclaimers verbatim in every module docstring + the docs page.
- Tests assert the verification-plan expected values (lifetime spot-checks incl.
  Tl/Ne guards; ratio 2.00(10)/2.00(9); α + ratio recovery tolerances; simulate
  round-trip + bit-for-bit seed).
- Lifetime table: spot-check transcribed values vs plan §2; confirm the ⚠confirm
  rows were verified or clearly marked.

FIX every confirmed finding. For genuine design ambiguity the plan does not
settle, STOP and ask Ben — do not change physics on a hunch. Do NOT modify
count_domain.py or other out-of-scope modules. Then: validate green; docs builds.

FINAL STAGE (pre-authorised — push + open the single PR):
1. Commit review fixes, e.g. "fix(negmu): Phase 6 — close-out review fixes
   (<summary>)". If nothing to fix, skip the commit and say so.
2. git fetch origin && git rebase origin/main; re-confirm validate + docs green.
3. Push the branch and open ONE PR titled "feat(negmu): negative-muon
   capture-lifetime analysis (API-only, experimental)". Body: summarise all
   phases; state WIP/experimental + API-only + no-GUI-by-construction; note the
   literature-anchored lifetime table (Suzuki/Measday/Roalsvig 1987) and the
   WiMDA bugs corrected; summarise the Phase-6 review outcome; link the study at
   docs/porting/negative-muon-analysis/. Do NOT enable auto-merge. Report PR URL.
```

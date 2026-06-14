# Implementation options

## Option A — thin `fit_global` wrapper over `FitEngine.global_fit` (CHOSEN)

Add `src/asymmetry/core/fitting/asymmetry_global.py`:

```python
@dataclass
class GlobalFitResult:
    success: bool
    global_parameters: ParameterSet           # shared fitted globals (values)
    global_uncertainties: dict[str, float]     # 1σ on the shared globals
    dataset_results: dict[Hashable, FitResult] # per-dataset: globals+locals, σ, χ², dof
    chi_squared: float                         # combined Σ_d χ²_d
    dof: int                                   # ΣN_d − N_free_global − Σ N_free_local_d
    reduced_chi_squared: float                 # combined χ² / dof
    message: str = ""

def fit_global(
    datasets,                  # Sequence[MuonDataset] | Mapping[Hashable, MuonDataset]
    model_fn,                  # f(t, **params) -> array (asymmetry/polarization)
    *,
    global_params,             # list[str] shared across all datasets
    local_params,              # list[str] independent per dataset
    initial_params,            # ParameterSet (broadcast) | Mapping[key, ParameterSet]
    t_min=None, t_max=None,
    method="migrad", max_calls=10000, minos=False,
    fit_engine=None, cancel_callback=None,
) -> GlobalFitResult: ...
```

**Keying.** A sequence is keyed positionally (`0..N-1`); a mapping keeps its keys.
The caller's datasets need not carry unique `run_number`s — internally we build
throwaway `MuonDataset` wrappers (`run=None`, `metadata["run_number"] = synthetic
index`) sharing the original arrays, so the engine's unique-run-number contract is
always satisfied and results map cleanly back to the caller's keys. This removes
the sharpest discoverability trap (runless datasets colliding on key `0`).

**Initial parameters.** `initial_params` may be a single `ParameterSet`
broadcast to every dataset (the common case — one seed structure), or a per-key
mapping. Global-parameter seed/bounds are taken from the first dataset's set
(matching the engine), and we validate that every dataset's set contains every
referenced global/local name.

**Combined reduced χ².** The engine returns per-dataset Gaussian χ² and dof, but
per-dataset dof subtracts the shared globals from *each* dataset, so summing them
double-counts the globals. We instead compute the combined dof directly:
`Σ_d N_d − N_free_global − Σ_d N_free_local_d`, with `χ²_combined = Σ_d χ²_d`. For
a single dataset this reduces exactly to the ordinary single-fit reduced χ².

**Boundary validation** (raise `ValueError`/`KeyError` with clear messages):
- `global_params` and `local_params` disjoint;
- every dataset's `ParameterSet` contains all referenced names (mismatched names
  across datasets → clear error);
- at least one dataset; errors present, finite, and positive on each dataset.

**Reuse.** The minimiser, dataset concatenation, fixed/linked-parameter handling,
bounds, MINOS, and cancel polling all come from `FitEngine.global_fit`. The new
file adds only input normalisation, synthetic keying, and the result bundle.

### Why Option A

- No minimiser duplication; the asymmetry-domain and count-domain global fits
  share exactly one engine seam.
- Naming (`fit_global`, `GlobalFitResult`) sits alongside the existing
  `fit_grouped_series` / `GroupedSeriesFitResult` family and is exported from
  `asymmetry.core.fitting`, closing the discoverability gap.
- GUI-free, scriptable, testable with synthetic traces.

## Option B — new standalone iminuit cost in the new module

Rejected: re-implements concatenation, fixed/link handling, bounds, MINOS, and
cancel polling that `FitEngine.global_fit` already provides and that the
count-domain family already depends on. Pure duplication and a second place to
fix bugs.

## Option C — make the count-domain path ergonomic instead, no asymmetry-domain fit

Rejected: the study confirms users genuinely want least-squares on the asymmetry
traces they already hold (per-point σ_A, no histograms). Forcing them through the
count domain keeps the friction the gap is about. The count-domain Poisson path
remains the statistically-preferred option for low counts and is documented as
such — both paths coexist.

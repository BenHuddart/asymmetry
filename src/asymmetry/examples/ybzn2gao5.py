r"""Synthetic YbZn2GaO5 longitudinal-field spin-dynamics dataset.

This module generates a paper-shaped synthetic muon-spin-relaxation dataset for
the Dirac U(1) quantum spin liquid YbZn2GaO5, so a full two-level analysis
(batch relaxation fits, then a cross-temperature global fit of the field
dependence) reproduces the published parameters.

Reference
---------
H. C. H. Wu, F. L. Pratt, B. M. Huddart, D. Chatterjee, P. A. Goddard,
J. Singleton, D. Prabhakaran, and S. J. Blundell, "Spin dynamics in the Dirac
U(1) spin liquid YbZn2GaO5", arXiv:2502.00130 (2025).

Every number below is **synthetic data generated from the published parameter
values** (Table I and Figs. 3-4 of that reference); no experimental data is
used or distributed. The synthetic runs carry a generic sample name
("YbZn2GaO5 (synthetic)") and run numbers in a fictitious 90xxx range.

Physics model
-------------
The paper fits the longitudinal-field relaxation rate lambda(B_LF) at eight
temperatures to the sum of four terms [their Eq. (1)]::

    lambda(B) = lambda_2D(B) + lambda_0D(B) + lambda_BG + lambda_LCR(B)

    lambda_2D(B)  = (A^2 / 4) J_2D(D_2D, omega_e)          [2D spin diffusion]
    lambda_0D(B)  = (D^2 / 4) (2/nu) / (1 + (omega_mu/nu)^m)  [0D Redfield]
    lambda_BG                                              [flat background]
    lambda_LCR(B) = f G(B; B0, Bwid)                       [Gaussian LCR peak]

with D_2D, nu and f allowed to vary with temperature (local) and A, D,
lambda_BG, m, B0, Bwid shared across temperature (global). We evaluate lambda(B)
with the repository's own composite model
``ParameterCompositeModel(["DiffusionLF_2D", "Redfield", "Lambda_bg",
"GaussianLCR"])`` (component ``"Lambda_bg"`` carries the paper's ``lambda_BG``
parameter directly), so the synthetic truth and the fit share one implementation
of the physics.

The muon asymmetry itself is a simple exponential plus a constant background,
``a(t) = a0 exp(-lambda t) + a_BG`` with a0 ~= 21 % and a_BG ~= 4 %, matching
the paper's low-temperature stretched-exponential-with-beta=1 description. The
per-run relaxation rate ``lambda`` is ``lambda(B, T)`` from the model above.

Unit conventions (read carefully)
----------------------------------
* **Field: Tesla <-> Gauss.** The paper quotes fields in Tesla; the repository's
  field-scan components (and the trend panel) take field in **Gauss**
  (1 T = 10000 G). Table I's B0 = 2.7 T and Bwid = 1.3 T are therefore stored as
  27000 G and 13000 G. The synthetic field grid spans 10 G (0.001 T) to
  45000 G (4.5 T).
* **Rates: ns^-1 <-> MHz.** Fig. 4 plots D_2D and nu in ns^-1; the repository's
  rate parameters are in **MHz** (1 ns^-1 = 1000 MHz). A paper value of
  3 ns^-1 is stored as 3000 MHz. The amplitudes A and D and the background
  lambda_BG are already in the paper's MHz / us^-1 units and are used verbatim.

Deviations from a literal figure reading (documented for honesty)
-----------------------------------------------------------------
The six global (Table I) truths are used **exactly**. The per-temperature
local ladders (:data:`TRUTH.local`) are *representative* values chosen to
resemble Figs. 3-4 while keeping the synthetic global fit cleanly identifiable:

1. ``D_2D`` follows the paper's shape - a low-temperature quantum plateau near
   3 ns^-1 (3000 MHz) for T < J ~ 3.2 K rising to ~60 ns^-1 (60000 MHz) at 12 K.
2. ``f`` (LCR amplitude) is zero below J and rises to ~0.15 us^-1 for T >= J, so
   the level-crossing peak at B0 = 2.7 T only appears in the 3.2-12 K panels,
   mirroring Fig. 3(e)-(h).
3. ``nu`` is set to **0.12-0.56 ns^-1 (120-560 MHz)**, which is *smaller* than
   the paper's Fig. 4 reading (~1-30 ns^-1). This is a deliberate choice: the
   0D Redfield cutoff sits at the field where omega_mu = nu, i.e.
   B_cut = nu / gamma_mu. With the paper's larger nu that cutoff lies well above
   4 T, so within an accessible field window the 0D term is a near-flat plateau
   and {D, nu, m} are only weakly identifiable from lambda(B) alone (the real
   study pins them with additional temperature-scan data). Lowering nu places
   the sharp m = 7 cutoff inside the 0.001-4.5 T window - exactly the "extended
   field range" feature the paper highlights - so the synthetic global fit
   recovers D, nu and m robustly. The *shape* of the story (a sharp non-2D
   cutoff whose position moves with T) is preserved; only the absolute nu scale
   is retuned.

The recovery of every Table I global from the synthetic data (batch lambda
extraction -> global fit) is pinned by
``tests/core/test_ybzn2gao5_example.py::test_recovery_gate``.

Command line
------------
::

    python -m asymmetry.examples.ybzn2gao5 --out ./ybzn2gao5_runs
    python -m asymmetry.examples.ybzn2gao5 --out ./runs --seed 7 --fields 20
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from asymmetry.core.fitting.parameter_models import ParameterCompositeModel
from asymmetry.core.io.nexus_writer import write_nexus_v1
from asymmetry.core.simulate import build_builtin_template, simulate_run

# ---------------------------------------------------------------------------
# Unit conversions (see the module docstring)
# ---------------------------------------------------------------------------
TESLA_TO_GAUSS = 1.0e4
NS_INV_TO_MHZ = 1.0e3

#: The composite model whose ``function`` defines lambda(B) [paper Eq. (1)].
MODEL_COMPONENTS: tuple[str, ...] = (
    "DiffusionLF_2D",  # lambda_2D = (A^2/4) J_2D(D_2D, omega_e)
    "Redfield",  # lambda_0D = (D^2/4)(2/nu)/(1 + (omega_mu/nu)^m)
    "Lambda_bg",  # lambda_BG (flat)
    "GaussianLCR",  # lambda_LCR = f G(B; B0, Bwid)
)

#: Parameter roles for the cross-temperature global fit (paper convention).
GLOBAL_PARAMS: tuple[str, ...] = ("A", "D", "lambda_BG", "m", "B0", "Bwid")
LOCAL_PARAMS: tuple[str, ...] = ("D_2D", "nu", "f")
FIXED_PARAMS: dict[str, float] = {"D_perp": 0.0}  # 2D diffusion: out-of-plane held 0

#: Physical parameter bounds for the global fit. The level-crossing centre and
#: width are bounded to a physical window around the DFT-predicted 2.73 T
#: singlet-triplet resonance (paper End Matter, Fig. 7), which keeps the LCR
#: term from wandering when its high-field tail is only partly sampled.
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "A": (0.0, 300.0),
    "D_2D": (0.0, 1.0e6),
    "D_perp": (0.0, 1.0e3),
    "D": (0.0, 100.0),
    "nu": (1.0e-2, 1.0e4),
    "m": (1.0, 12.0),
    "lambda_BG": (0.0, 1.0),
    "f": (0.0, 2.0),
    "B0": (20000.0, 35000.0),
    "Bwid": (7000.0, 20000.0),
}


@dataclass(frozen=True)
class Ybzn2gao5Truth:
    """The synthetic truth: Table I globals + per-temperature local ladders.

    All rates are in MHz and all fields in Gauss (the repository's internal
    units); see the module docstring for the ns^-1/Tesla conversions.
    """

    #: Global (temperature-independent) parameters - Table I, exactly.
    global_params: dict[str, float]
    #: Temperatures in kelvin, in generation order.
    temperatures: tuple[float, ...]
    #: ``temperature -> {"D_2D", "nu", "f"}`` local ladders (MHz / us^-1).
    local: dict[float, dict[str, float]]
    #: Exchange coupling J (K); the quantum/classical crossover scale.
    J_kelvin: float = 3.2
    #: Muon asymmetry model: a(t) = a0 exp(-lambda t) + a_BG (percent scale).
    a0_percent: float = 21.0
    a_bg_percent: float = 4.0

    def params_for(self, temperature: float) -> dict[str, float]:
        """Full parameter dict (globals + this temperature's locals + fixed)."""
        params = dict(self.global_params)
        params.update(self.local[temperature])
        params.update(FIXED_PARAMS)
        return params


#: The canonical synthetic truth for this example.
TRUTH = Ybzn2gao5Truth(
    global_params={
        "A": 63.0,  # MHz  (Table I)
        "D": 18.4,  # MHz  (Table I)
        "lambda_BG": 0.067,  # us^-1 (Table I)
        "m": 7.0,  # (Table I)
        "B0": 2.7 * TESLA_TO_GAUSS,  # 27000 G = 2.7 T (Table I)
        "Bwid": 1.3 * TESLA_TO_GAUSS,  # 13000 G = 1.3 T (Table I)
    },
    temperatures=(0.05, 0.2, 0.4, 1.6, 3.2, 6.0, 9.0, 12.0),
    local={
        #        D_2D (MHz)          nu (MHz)          f (us^-1)
        0.05: {"D_2D": 3.0 * NS_INV_TO_MHZ, "nu": 120.0, "f": 0.0},
        0.2: {"D_2D": 3.2 * NS_INV_TO_MHZ, "nu": 150.0, "f": 0.0},
        0.4: {"D_2D": 3.6 * NS_INV_TO_MHZ, "nu": 200.0, "f": 0.0},
        1.6: {"D_2D": 5.0 * NS_INV_TO_MHZ, "nu": 300.0, "f": 0.0},
        3.2: {"D_2D": 9.0 * NS_INV_TO_MHZ, "nu": 400.0, "f": 0.06},
        6.0: {"D_2D": 22.0 * NS_INV_TO_MHZ, "nu": 480.0, "f": 0.11},
        9.0: {"D_2D": 42.0 * NS_INV_TO_MHZ, "nu": 520.0, "f": 0.14},
        12.0: {"D_2D": 62.0 * NS_INV_TO_MHZ, "nu": 560.0, "f": 0.15},
    },
)

# ---------------------------------------------------------------------------
# Generation parameters
# ---------------------------------------------------------------------------
#: Default RNG seed for reproducible Poisson realisation.
DEFAULT_SEED = 20250131  # arXiv:2502.00130 submission date

#: Log-spaced field grid, 10 G (0.001 T) -> 45000 G (4.5 T). The upper bound
#: extends just past the LCR peak (2.7 T) so its descending flank is sampled,
#: which removes a systematic bias in the fitted B0.
FIELD_MIN_GAUSS = 10.0
FIELD_MAX_GAUSS = 45000.0

#: First synthetic run number (fictitious 90xxx range - no real run numbers).
FIRST_RUN_NUMBER = 90001

#: Generic synthetic sample name stamped into every run's title.
SAMPLE_NAME = "YbZn2GaO5 (synthetic)"

#: Total muon events per run. Sized so a single-run exponential fit returns a
#: relaxation-rate error of a few percent (verified by the fast test).
DEFAULT_COUNTS = 60.0e6


@dataclass(frozen=True)
class RunSpec:
    """Identity of one generated run."""

    run_number: int
    temperature: float
    field_gauss: float
    path: str
    lambda_truth: float


@dataclass(frozen=True)
class Manifest:
    """Result of :func:`generate_ybzn2gao5_runs`."""

    out_dir: str
    runs: list[RunSpec] = field(default_factory=list)
    temperatures: tuple[float, ...] = ()
    fields_gauss: tuple[float, ...] = ()
    seed: int = DEFAULT_SEED
    counts_per_run: float = DEFAULT_COUNTS
    #: The truth dict used, for provenance (globals + per-T locals).
    truth_global: dict[str, float] = field(default_factory=dict)

    @property
    def n_runs(self) -> int:
        return len(self.runs)


def field_grid(fields_per_temperature: int) -> np.ndarray:
    """Log-spaced field grid in Gauss (10 G -> 45000 G)."""
    if fields_per_temperature < 2:
        raise ValueError("fields_per_temperature must be at least 2.")
    return np.logspace(
        np.log10(FIELD_MIN_GAUSS),
        np.log10(FIELD_MAX_GAUSS),
        int(fields_per_temperature),
    )


def lambda_of_field(
    model: ParameterCompositeModel, temperature: float, fields_gauss: np.ndarray
) -> np.ndarray:
    """Noiseless truth lambda(B) (us^-1) over ``fields_gauss`` at ``temperature``."""
    params = TRUTH.params_for(temperature)
    return np.asarray(model.function(np.asarray(fields_gauss, dtype=float), **params), dtype=float)


def _asymmetry_signal(lambda_rate: float):
    """Return a callable a(t) -> percent for one run's exponential relaxation.

    ``a(t) = a0 exp(-lambda t) + a_BG`` on the percent scale
    (``simulate_run`` expects the asymmetry in percent).
    """
    a0 = float(TRUTH.a0_percent)
    a_bg = float(TRUTH.a_bg_percent)
    rate = float(lambda_rate)

    def signal(t: np.ndarray) -> np.ndarray:
        tt = np.asarray(t, dtype=float)
        return a0 * np.exp(-rate * np.clip(tt, 0.0, None)) + a_bg

    return signal


def generate_ybzn2gao5_runs(
    out_dir: Path | str,
    *,
    seed: int = DEFAULT_SEED,
    fields_per_temperature: int = 20,
    counts_scale: float = 1.0,
    temperatures: tuple[float, ...] | None = None,
) -> Manifest:
    """Write synthetic YbZn2GaO5 NeXus V1 runs and return a manifest.

    One run is written per (temperature, field). Each run's muon asymmetry is
    ``a(t) = a0 exp(-lambda(B, T) t) + a_BG`` with ``lambda(B, T)`` from the
    paper's Eq. (1) (see the module docstring), realised as Poisson-counted
    forward/backward histograms through the repository's :func:`simulate_run`
    and written with :func:`write_nexus_v1`. The files reload through
    :func:`asymmetry.core.io.load` exactly like real data.

    Parameters
    ----------
    out_dir
        Directory to write the ``.nxs`` files into (created if needed).
    seed
        Base RNG seed. Each run derives a distinct sub-seed from it, so the
        whole dataset is deterministic: the same ``seed`` produces byte-
        identical histogram counts (and, because the writer stamps no wall-clock
        timestamp, byte-identical files).
    fields_per_temperature
        Number of log-spaced field points (>= 2). Default 20.
    counts_scale
        Multiplies :data:`DEFAULT_COUNTS`; raise it for smaller error bars.
    temperatures
        Optional subset/override of :data:`TRUTH.temperatures` (used by the fast
        test to generate a small dataset). Defaults to all eight.

    Returns
    -------
    Manifest
        Records every written run (number, T, B, path, truth lambda), the field
        grid, the seed, and the global truth used.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    temps = TRUTH.temperatures if temperatures is None else tuple(temperatures)
    fields = field_grid(fields_per_temperature)
    counts = DEFAULT_COUNTS * float(counts_scale)

    model = ParameterCompositeModel(list(MODEL_COMPONENTS))
    template = build_builtin_template("ideal_pulsed_fb")

    runs: list[RunSpec] = []
    run_number = FIRST_RUN_NUMBER
    for temperature in temps:
        lam_curve = lambda_of_field(model, temperature, fields)
        for field_gauss, lam in zip(fields, lam_curve, strict=True):
            # A distinct, deterministic sub-seed per run keeps the Poisson draw
            # independent between runs yet fully reproducible from ``seed``.
            run_seed = int(seed) + (run_number - FIRST_RUN_NUMBER)
            run = simulate_run(
                template,
                _asymmetry_signal(lam),
                total_events=counts,
                seed=run_seed,
                run_number=run_number,
                title=f"{SAMPLE_NAME} T={temperature:g} K B={field_gauss:.0f} G",
            )
            # simulate_run inherits temperature/field from the template metadata
            # (empty here), so stamp the true per-run values the writer records.
            run.metadata["temperature"] = float(temperature)
            run.metadata["field"] = float(field_gauss)
            run.metadata["sample"] = SAMPLE_NAME

            path = out_path / f"ybzn2gao5_{run_number}.nxs"
            write_nexus_v1(run, path)
            runs.append(
                RunSpec(
                    run_number=run_number,
                    temperature=float(temperature),
                    field_gauss=float(field_gauss),
                    path=str(path),
                    lambda_truth=float(lam),
                )
            )
            run_number += 1

    return Manifest(
        out_dir=str(out_path),
        runs=runs,
        temperatures=tuple(float(t) for t in temps),
        fields_gauss=tuple(float(b) for b in fields),
        seed=int(seed),
        counts_per_run=counts,
        truth_global=dict(TRUTH.global_params),
    )


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m asymmetry.examples.ybzn2gao5",
        description=(
            "Generate the synthetic YbZn2GaO5 longitudinal-field dataset "
            "(Wu et al., arXiv:2502.00130)."
        ),
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output directory for the .nxs run files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Base RNG seed (default {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--fields",
        type=int,
        default=20,
        dest="fields_per_temperature",
        help="Log-spaced field points per temperature (default 20).",
    )
    parser.add_argument(
        "--counts-scale",
        type=float,
        default=1.0,
        help="Multiplier on the per-run event budget (default 1.0).",
    )
    args = parser.parse_args(argv)

    manifest = generate_ybzn2gao5_runs(
        args.out,
        seed=args.seed,
        fields_per_temperature=args.fields_per_temperature,
        counts_scale=args.counts_scale,
    )

    print(f"Wrote {manifest.n_runs} synthetic YbZn2GaO5 runs to {manifest.out_dir}")
    print(
        f"  {len(manifest.temperatures)} temperatures x "
        f"{len(manifest.fields_gauss)} fields "
        f"({FIELD_MIN_GAUSS:g} G - {FIELD_MAX_GAUSS:g} G)"
    )
    print(f"  run numbers {FIRST_RUN_NUMBER}-{FIRST_RUN_NUMBER + manifest.n_runs - 1}")
    print(f"  seed {manifest.seed}, ~{manifest.counts_per_run:.3g} events/run")
    print(f"  sample: {SAMPLE_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())

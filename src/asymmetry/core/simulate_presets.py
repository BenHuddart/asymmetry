"""Archetype gallery: one-click textbook synthetic runs.

Each preset pairs a built-in idealised instrument (:mod:`asymmetry.core.simulate`)
with a fit model and a set of physically grounded parameters, and generates a
first-class **synthetic run** — Poisson count histograms drawn through the same
reduction chain as beamline data, badged with provenance — rather than a curve
with Gaussian noise bolted on. The parameter values are the canonical
textbook/literature numbers also used by the documentation screenshots
(:mod:`docs.screenshots.data.archetypes`); here they drive the full simulate
pipeline so the runs are fittable and recover their generating physics.

Reference for all archetypes: S. J. Blundell, R. De Renzi, T. Lancaster, and
F. L. Pratt (eds.), *Muon Spectroscopy: An Introduction* (Oxford University
Press, 2022). Per-preset chapters are named in each :class:`SimulatePreset`.

This module is Qt-free and must stay importable without the GUI.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from asymmetry.core.data.dataset import Run
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.models import MODELS
from asymmetry.core.simulate import build_builtin_template, simulate_run

#: γ_μ / 2π in MHz/G (matches docs/screenshots/data/archetypes.py).
_GAMMA_MU_MHZ_PER_G = 0.01355

# Material parameters (textbook / classic μSR literature) ---------------------
_DELTA_AG_PER_US = 0.39  # Ag nuclear dipolar width, Ch. 5
_TC_EUO_K = 69.0  # EuO Curie temperature, Ch. 6
_NU0_EUO_MHZ = 28.0  # EuO T→0 precession frequency, Ch. 6 (Fig. 6.6)
_BETA_EUO = 0.40  # EuO order-parameter exponent
_R_MUF_ANG = 1.17  # F-μ-F equilibrium separation, Ch. 4
_TF_YBCO_G = 200.0  # YBCO transverse field (chosen for clean cycles)


@dataclass(frozen=True)
class PresetRunSpec:
    """One synthetic run within a preset (a scan member or a single run)."""

    model_name: str
    model: Any  # callable a(t) in percent, or CompositeModel
    parameters: dict[str, float]
    title: str
    temperature: float
    field: float
    field_state: str = "ZF"
    total_events: float = 40.0e6


@dataclass(frozen=True)
class SimulatePreset:
    """A named textbook archetype that generates one or more synthetic runs."""

    key: str
    label: str
    chapter: str  # textbook chapter, cited by name — never an equation number
    description: str
    template_key: str
    seed: int
    specs: tuple[PresetRunSpec, ...] = field(default_factory=tuple)


def _oscillatory(**params: float) -> tuple[str, Callable, dict[str, float]]:
    return "Oscillatory", MODELS["Oscillatory"].function, params


def _euo_specs() -> tuple[PresetRunSpec, ...]:
    """EuO ZF temperature scan crossing Tc = 69 K (Ch. 6, Fig. 6.6).

    Below Tc the spontaneous local field drives precession whose frequency
    tracks the magnetic order parameter ν(T) ∝ (1 − T/Tc)^β; above Tc the
    signal is paramagnetic exponential relaxation, with damping peaking in the
    critical region.
    """
    osc = MODELS["Oscillatory"].function
    exp_fn = MODELS["ExponentialRelaxation"].function
    damping_floor, lambda_peak, delta_t_k = 0.10, 4.0, 6.0
    specs: list[PresetRunSpec] = []
    for temp in (30.0, 50.0, 65.0, 73.0, 90.0):
        damping = damping_floor + lambda_peak * float(
            np.exp(-(((temp - _TC_EUO_K) / delta_t_k) ** 2))
        )
        if temp < _TC_EUO_K - 0.5:
            order = (1.0 - temp / _TC_EUO_K) ** _BETA_EUO
            frequency = _NU0_EUO_MHZ * order
            specs.append(
                PresetRunSpec(
                    model_name="Oscillatory",
                    model=osc,
                    parameters={
                        "A0": 22.0,
                        "frequency": frequency,
                        "phase": 0.0,
                        "Lambda": damping,
                        "baseline": 0.0,
                    },
                    title=f"EuO ZF {temp:g} K (ordered)",
                    temperature=temp,
                    field=0.0,
                )
            )
        else:
            specs.append(
                PresetRunSpec(
                    model_name="ExponentialRelaxation",
                    model=exp_fn,
                    parameters={"A0": 22.0, "Lambda": damping, "baseline": 0.0},
                    title=f"EuO ZF {temp:g} K (paramagnetic)",
                    temperature=temp,
                    field=0.0,
                )
            )
    return tuple(specs)


def _ag_lf_specs() -> tuple[PresetRunSpec, ...]:
    """Ag LF Kubo–Toyabe decoupling series, shared Δ = 0.39 μs⁻¹ (Ch. 5)."""
    lfkt = MODELS["LFKuboToyabe"].function
    return tuple(
        PresetRunSpec(
            model_name="LFKuboToyabe",
            model=lfkt,
            parameters={"A0": 24.0, "Delta": _DELTA_AG_PER_US, "B_L": field_g, "baseline": 0.0},
            title=f"Ag LF {field_g:g} G",
            temperature=20.0,
            field=field_g,
            field_state="LF" if field_g > 0 else "ZF",
        )
        for field_g in (0.0, 10.0, 25.0, 50.0)
    )


def _build_registry() -> dict[str, SimulatePreset]:
    fmuf = CompositeModel(["FmuF_Linear", "Constant"], operators=["+"])
    return {
        "ag_zf_kt": SimulatePreset(
            key="ag_zf_kt",
            label="Ag — ZF Gaussian Kubo–Toyabe",
            chapter="Ch. 5",
            description=(
                "Zero-field silver polycrystal: the canonical static "
                "nuclear-dipolar reference, a Gaussian Kubo–Toyabe with "
                "Δ = 0.39 μs⁻¹."
            ),
            template_key="ideal_pulsed_fb",
            seed=2301,
            specs=(
                PresetRunSpec(
                    model_name="StaticGKT_ZF",
                    model=MODELS["StaticGKT_ZF"].function,
                    parameters={"A0": 24.0, "Delta": _DELTA_AG_PER_US, "baseline": 0.0},
                    title="Ag ZF 20 K (Kubo–Toyabe)",
                    temperature=20.0,
                    field=0.0,
                ),
            ),
        ),
        "ag_lf_decoupling": SimulatePreset(
            key="ag_lf_decoupling",
            label="Ag — LF decoupling series",
            chapter="Ch. 5",
            description=(
                "Longitudinal-field decoupling of the same Ag Kubo–Toyabe: as "
                "B_L grows the relaxation is progressively decoupled, the "
                "textbook test of static nuclear dipolar fields."
            ),
            template_key="ideal_pulsed_fb",
            seed=4100,
            specs=_ag_lf_specs(),
        ),
        "euo_tscan": SimulatePreset(
            key="euo_tscan",
            label="EuO — ferromagnet through Tc",
            chapter="Ch. 6",
            description=(
                "Zero-field EuO temperature scan across the Curie point "
                "Tc = 69 K: spontaneous precession below Tc tracking the order "
                "parameter (1 − T/Tc)^β, paramagnetic relaxation above."
            ),
            template_key="ideal_pulsed_fb",
            seed=1700,
            specs=_euo_specs(),
        ),
        "fmuf_pbf2": SimulatePreset(
            key="fmuf_pbf2",
            label="PbF₂ — F-μ-F entanglement",
            chapter="Ch. 4",
            description=(
                "Zero-field PbF₂: the muon binds two fluorine neighbours into "
                "an F-μ-F state whose characteristic dipolar beat pattern "
                "(r ≈ 1.17 Å) is the textbook fingerprint of muon-fluorine "
                "entanglement."
            ),
            template_key="ideal_pulsed_fb",
            seed=8900,
            specs=(
                PresetRunSpec(
                    model_name="FmuF_Linear+Constant",
                    model=fmuf,
                    parameters={"A_1": 22.0, "r_muF": _R_MUF_ANG, "A_bg": 0.2},
                    title="PbF₂ ZF 5 K (F-μ-F)",
                    temperature=5.0,
                    field=0.0,
                    total_events=80.0e6,
                ),
            ),
        ),
        "ybco_tf": SimulatePreset(
            key="ybco_tf",
            label="YBCO — transverse-field precession",
            chapter="Ch. 8",
            description=(
                "Normal-state YBa₂Cu₃O₇₋δ in a 200 G transverse field: "
                "Knight-shifted Larmor precession with a relaxation that "
                "broadens on entering the vortex state below Tc = 90 K."
            ),
            template_key="ideal_pulsed_fb",
            seed=10100,
            specs=(
                PresetRunSpec(
                    *_oscillatory(
                        A0=20.0,
                        frequency=_GAMMA_MU_MHZ_PER_G * _TF_YBCO_G * 1.005,
                        phase=0.0,
                        Lambda=0.08,
                        baseline=0.0,
                    ),
                    title="YBCO TF 200 G 100 K (Knight shift)",
                    temperature=100.0,
                    field=_TF_YBCO_G,
                    field_state="TF",
                ),
            ),
        ),
    }


#: Archetype presets keyed for the gallery menu.
ARCHETYPE_PRESETS: dict[str, SimulatePreset] = _build_registry()


def build_preset_runs(
    key: str,
    *,
    seed: int | None = None,
    run_number_allocator: Callable[[], int] | None = None,
) -> list[Run]:
    """Generate the synthetic run(s) of a named archetype preset.

    Each run is produced by :func:`asymmetry.core.simulate.simulate_run` on the
    preset's built-in instrument template, badged synthetic, and carries the
    preset key and textbook chapter in its ``metadata["simulation"]``
    provenance. A scan preset (EuO, Ag LF) returns several runs; seeds are the
    preset's fixed seed offset by the member index, so the family is
    reproducible. ``seed`` overrides the preset's fixed base seed when given.

    Raises :class:`KeyError` for an unknown preset key.
    """
    try:
        preset = ARCHETYPE_PRESETS[key]
    except KeyError:
        raise KeyError(
            f"Unknown archetype preset {key!r}; available: {sorted(ARCHETYPE_PRESETS)}."
        ) from None

    template = build_builtin_template(preset.template_key)
    base_seed = preset.seed if seed is None else int(seed)
    runs: list[Run] = []
    for index, spec in enumerate(preset.specs):
        run_number = run_number_allocator() if run_number_allocator is not None else None
        run = simulate_run(
            template,
            spec.model,
            spec.parameters,
            total_events=spec.total_events,
            seed=base_seed + index,
            alpha=1.0,
            run_number=run_number,
            title=spec.title,
        )
        # The built-in template has no sample environment; stamp the archetype's.
        run.metadata["temperature"] = spec.temperature
        run.metadata["field"] = spec.field
        run.metadata["field_state"] = spec.field_state
        provenance = run.metadata.setdefault("simulation", {})
        provenance["preset"] = preset.key
        provenance["preset_label"] = preset.label
        provenance["reference"] = (
            f"Blundell, De Renzi, Lancaster & Pratt, "
            f"Muon Spectroscopy: An Introduction (OUP, 2022), {preset.chapter}"
        )
        runs.append(run)
    return runs

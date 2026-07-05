"""Scope and multiplet-pattern integration for the global fit wizard portfolio."""

from __future__ import annotations

import numpy as np

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.fit_wizard import (
    build_candidate_templates,
    fingerprint_spectrum,
)
from asymmetry.core.fitting.global_fit_wizard import (
    _aggregate_fingerprints,
    _series_multiplet_pattern_family_keys,
    build_global_fit_wizard_candidate_portfolio,
)
from asymmetry.core.fitting.muon_fluorine.dipolar import omega_d_mu_f_rad_per_us
from asymmetry.core.fitting.muon_fluorine.polarization import linear_fmuf_polarization
from asymmetry.core.fitting.wizard_scope import (
    WizardScope,
    WizardScopePreset,
    resolve_scope_for_datasets,
)


def _dataset(
    run_number: int,
    t: np.ndarray,
    y: np.ndarray,
    *,
    metadata: dict | None = None,
) -> MuonDataset:
    payload = {"run_number": run_number, "temperature": float(run_number)}
    payload.update(metadata or {})
    return MuonDataset(
        time=np.asarray(t, dtype=float),
        asymmetry=np.asarray(y, dtype=float),
        error=np.full_like(np.asarray(t, dtype=float), 0.004),
        metadata=payload,
    )


def _exp_series(n: int = 2) -> list[MuonDataset]:
    rng = np.random.default_rng(31)
    t = np.linspace(0.02, 10.0, 200)
    return [
        _dataset(run, t, 0.2 * np.exp(-0.7 * t) + 0.02 + rng.normal(0.0, 0.004, t.size))
        for run in range(1, n + 1)
    ]


def _fmuf_series(n: int = 2) -> list[MuonDataset]:
    rng = np.random.default_rng(32)
    t = np.linspace(0.02, 24.0, 480)
    return [
        _dataset(
            run,
            t,
            0.25 * linear_fmuf_polarization(t, 1.17) + 0.02 + rng.normal(0.0, 0.004, t.size),
            metadata={"field_direction": "Zero field"},
        )
        for run in range(1, n + 1)
    ]


def test_portfolio_scope_filters_templates() -> None:
    datasets = _exp_series()
    scope = WizardScope(preset=WizardScopePreset.FLUORIDE_FMUF)
    portfolio = build_global_fit_wizard_candidate_portfolio(datasets, scope=scope)

    resolution = resolve_scope_for_datasets(datasets, scope)
    for template in portfolio.templates:
        assert all(name in resolution.included_set for name in template.model.component_names), (
            template.key
        )
    keys = {template.key for template in portfolio.templates}
    assert "fmuf_linear_exp_constant" in keys
    assert "oscillatory_exp_constant" not in keys
    assert "static_gkt_constant" not in keys


def test_portfolio_legacy_default_unchanged_without_patterns() -> None:
    datasets = _exp_series()
    portfolio = build_global_fit_wizard_candidate_portfolio(datasets)

    fingerprints = [fingerprint_spectrum(dataset) for dataset in datasets]
    legacy = build_candidate_templates(_aggregate_fingerprints(fingerprints))
    assert [template.key for template in portfolio.templates] == [
        template.key for template in legacy
    ]
    assert portfolio.pattern_template_keys == ()


def test_pattern_vote_requires_majority() -> None:
    fmuf = _fmuf_series(2)
    assert "fmuf" in _series_multiplet_pattern_family_keys(fmuf)

    mixed = _fmuf_series(1) + _exp_series(2)
    assert "fmuf" not in _series_multiplet_pattern_family_keys(mixed)


def test_pattern_vote_forces_family_templates_into_portfolio() -> None:
    datasets = _fmuf_series(2)
    portfolio = build_global_fit_wizard_candidate_portfolio(datasets)

    assert portfolio.pattern_template_keys
    assert any(key.startswith(("fmuf", "muf")) for key in portfolio.pattern_template_keys)
    keys = {template.key for template in portfolio.templates}
    assert set(portfolio.pattern_template_keys) <= keys


def test_user_frequencies_can_trigger_pattern_vote() -> None:
    datasets = _exp_series(2)
    omega_tilde = omega_d_mu_f_rad_per_us(1.17) / (2.0 * np.pi)
    factors = (0.5 * (3.0 - np.sqrt(3.0)), np.sqrt(3.0), 0.5 * (3.0 + np.sqrt(3.0)))
    user = [f * omega_tilde for f in factors]
    for dataset in datasets:
        dataset.metadata["field_direction"] = "Zero field"

    assert "fmuf" in _series_multiplet_pattern_family_keys(datasets, user)
    portfolio = build_global_fit_wizard_candidate_portfolio(datasets, user_frequencies_mhz=user)
    assert any(key.startswith(("fmuf", "muf")) for key in portfolio.pattern_template_keys)

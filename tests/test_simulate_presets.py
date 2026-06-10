"""Archetype gallery presets — generation, determinism, physics recovery.

Each preset must generate deterministically from its fixed seed and, when
refitted with its generating model, recover its stated textbook physics within
the fit errors (verification target for the simulate-mode follow-ons).
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.models import MODELS
from asymmetry.core.simulate import reduce_run_to_dataset
from asymmetry.core.simulate_presets import ARCHETYPE_PRESETS, build_preset_runs


def _refit(run, model, starts, t_max):
    pytest.importorskip("iminuit")
    from asymmetry.core.fitting.engine import FitEngine
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    params = ParameterSet([Parameter(name=n, value=v, min=lo, max=hi) for n, v, lo, hi in starts])
    result = FitEngine().fit(reduce_run_to_dataset(run), model, params, t_max=t_max)
    assert result.success
    return {p.name: p.value for p in result.parameters}, result.uncertainties


class TestPresetRegistry:
    def test_expected_presets_present(self) -> None:
        for key in ("ag_zf_kt", "ag_lf_decoupling", "euo_tscan", "fmuf_pbf2", "ybco_tf"):
            assert key in ARCHETYPE_PRESETS

    def test_references_cite_textbook_not_equation_numbers(self) -> None:
        runs = build_preset_runs("ag_zf_kt")
        reference = runs[0].metadata["simulation"]["reference"]
        assert "Muon Spectroscopy" in reference
        assert "Ch." in reference
        # No bare "Eq." / "equation" citation (study standing rule).
        assert "eq" not in reference.lower()

    @pytest.mark.parametrize("key", list(ARCHETYPE_PRESETS))
    def test_runs_are_badged_synthetic(self, key) -> None:
        for run in build_preset_runs(key):
            assert run.metadata["synthetic"] is True
            assert run.metadata["simulation"]["preset"] == key
            assert run.histograms

    @pytest.mark.parametrize("key", list(ARCHETYPE_PRESETS))
    def test_deterministic_from_fixed_seed(self, key) -> None:
        first = build_preset_runs(key)
        second = build_preset_runs(key)
        assert len(first) == len(second) == len(ARCHETYPE_PRESETS[key].specs)
        for a, b in zip(first, second, strict=True):
            for ha, hb in zip(a.histograms, b.histograms, strict=True):
                assert np.array_equal(ha.counts, hb.counts)

    def test_scan_presets_generate_a_family(self) -> None:
        assert len(build_preset_runs("euo_tscan")) == 5
        assert len(build_preset_runs("ag_lf_decoupling")) == 4

    def test_allocator_assigns_run_numbers(self) -> None:
        counter = iter(range(90001, 90100))
        runs = build_preset_runs("ag_lf_decoupling", run_number_allocator=lambda: next(counter))
        numbers = [r.run_number for r in runs]
        assert numbers == [90001, 90002, 90003, 90004]

    def test_unknown_preset_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown archetype preset"):
            build_preset_runs("not_a_preset")


class TestPhysicsRecovery:
    """Refitting a preset recovers its stated physics within the fit error."""

    def test_ag_zf_recovers_delta(self) -> None:
        run = build_preset_runs("ag_zf_kt")[0]
        fitted, errs = _refit(
            run,
            MODELS["StaticGKT_ZF"].function,
            [("A0", 20.0, 0.0, 100.0), ("Delta", 0.3, 0.0, 5.0), ("baseline", 0.0, -10.0, 10.0)],
            8.0,
        )
        assert abs(fitted["Delta"] - 0.39) < 5.0 * errs["Delta"]

    def test_ag_lf_decoupled_member_recovers_delta_and_field(self) -> None:
        # The 25 G member (index 2): at nonzero field B_L is identifiable
        # (the zero-field member is degenerate — LFKT at B_L = 0 is static KT).
        spec = ARCHETYPE_PRESETS["ag_lf_decoupling"].specs[2]
        run = build_preset_runs("ag_lf_decoupling")[2]
        fitted, errs = _refit(
            run,
            MODELS["LFKuboToyabe"].function,
            [
                ("A0", 20.0, 0.0, 100.0),
                ("Delta", 0.3, 0.0, 5.0),
                ("B_L", 20.0, 0.0, 200.0),
                ("baseline", 0.0, -10.0, 10.0),
            ],
            8.0,
        )
        assert abs(fitted["Delta"] - 0.39) < 5.0 * errs["Delta"]
        assert abs(fitted["B_L"] - spec.parameters["B_L"]) < 5.0 * errs["B_L"]

    def test_euo_ordered_recovers_precession_frequency(self) -> None:
        run = build_preset_runs("euo_tscan")[0]
        truth = ARCHETYPE_PRESETS["euo_tscan"].specs[0].parameters["frequency"]
        fitted, errs = _refit(
            run,
            MODELS["Oscillatory"].function,
            [
                ("A0", 20.0, 0.0, 100.0),
                ("frequency", truth * 0.9, 0.0, 50.0),
                ("phase", 0.0, -7.0, 7.0),
                ("Lambda", 0.5, 0.0, 20.0),
                ("baseline", 0.0, -10.0, 10.0),
            ],
            6.0,
        )
        assert abs(fitted["frequency"] - truth) < 5.0 * errs["frequency"]

    def test_fmuf_recovers_separation(self) -> None:
        run = build_preset_runs("fmuf_pbf2")[0]
        fmuf = CompositeModel(["FmuF_Linear", "Constant"], operators=["+"])
        fitted, errs = _refit(
            run,
            fmuf.function,
            [("A_1", 20.0, 0.0, 100.0), ("r_muF", 1.3, 0.5, 3.0), ("A_bg", 0.0, -10.0, 10.0)],
            12.0,
        )
        assert abs(fitted["r_muF"] - 1.17) < 5.0 * max(errs["r_muF"], 1e-4)

    def test_ybco_recovers_larmor_frequency(self) -> None:
        run = build_preset_runs("ybco_tf")[0]
        truth = ARCHETYPE_PRESETS["ybco_tf"].specs[0].parameters["frequency"]
        fitted, errs = _refit(
            run,
            MODELS["Oscillatory"].function,
            [
                ("A0", 18.0, 0.0, 100.0),
                ("frequency", truth * 0.9, 0.0, 50.0),
                ("phase", 0.0, -7.0, 7.0),
                ("Lambda", 0.1, 0.0, 20.0),
                ("baseline", 0.0, -10.0, 10.0),
            ],
            8.0,
        )
        assert abs(fitted["frequency"] - truth) < 5.0 * errs["frequency"]

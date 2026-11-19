"""Fitting preparation and execution work with plotting and GUI imports forbidden."""

import subprocess
import sys
import textwrap


def test_preparation_executes_without_gui_or_plotting():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
            import importlib.abc
            import sys

            class BlockFrontend(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.startswith(("asymmetry.gui", "PySide6", "matplotlib")):
                        raise AssertionError(f"Headless fitting imported {fullname}")

            sys.meta_path.insert(0, BlockFrontend())
            import numpy as np
            from asymmetry.core.data.dataset import MuonDataset, Run
            from asymmetry.core.fitting.asymmetry_global import fit_global
            from asymmetry.core.fitting.composite import CompositeModel
            from asymmetry.core.fitting.cross_group import run_cross_group_fit_from_config
            from asymmetry.core.fitting.grouped_seeding import seed_group_background_and_n0, seed_group_absolute_phases
            from asymmetry.core.fitting.grouped_time_domain import GroupedTimeDomainGroup
            from asymmetry.core.fitting.parameter_models import ParameterCompositeModel, ParameterGroupData
            from asymmetry.core.fitting.recipe_inputs import build_recipe_engine_inputs
            from asymmetry.core.utils.constants import MUON_LIFETIME_US

            time = np.linspace(0, 6, 121)
            counts = 120 + 7 * np.exp(time / MUON_LIFETIME_US)
            np.testing.assert_allclose(seed_group_background_and_n0(counts, time=time), (7, 120, 0.01))
            group = GroupedTimeDomainGroup(1, "Group", time, counts, np.ones_like(time))
            assert np.isfinite(seed_group_absolute_phases([group])["1"])
            model = CompositeModel.from_expression("Exponential + Constant")
            datasets = {n: MuonDataset(time, 0.2 * np.exp(-0.4*time) + b, np.full_like(time, 0.01), run=Run(run_number=n)) for n, b in ((1, 0.02), (2, 0.03))}
            recipe = {"parameters": [{"name": name, "value": value, "type": role} for name, value, role in (("A_1", 0.18, "Global"), ("Lambda", 0.35, "Global"), ("A_bg", 0.0, "Local"))]}
            inputs = build_recipe_engine_inputs(recipe, model, list(datasets.values()))
            fit = fit_global(datasets, model.function, global_params=inputs.global_params, local_params=inputs.local_params, initial_params=inputs.initial_params)
            assert fit.success
            assert abs(fit.global_parameters["Lambda"].value - 0.4) < 1e-3
            x = np.arange(1.0, 9.0)
            groups = [ParameterGroupData(str(i), str(i), x, 0.02*x+b, np.full_like(x, 0.005), float(i)) for i, b in enumerate((0.1, 0.4))]
            config = {"model": ParameterCompositeModel(["Linear"], []).to_dict(), "parameter_rows": [{"name": "m", "initial": 0.01, "type": "Global"}, {"name": "b", "initial": 0.0, "type": "Local"}]}
            run = run_cross_group_fit_from_config(groups, config)
            assert run.result.success
            assert abs(run.result.global_parameters["m"].value - 0.02) < 1e-5
            assert not any(name.startswith(("asymmetry.gui", "PySide6", "matplotlib")) for name in sys.modules)
        """),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

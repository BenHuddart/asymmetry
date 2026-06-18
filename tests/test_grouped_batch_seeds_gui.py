"""Multi-group BATCH fitting: per-(dataset, group) seeds + single↔batch sync.

The grouped batch surface mirrors FB-asymmetry batch fitting: physics params take
Global/Local/Fixed roles and flow Single→Batch (chain-seeding + "Send to Batch"),
while the per-group nuisances are auto-seeded per (dataset, group) and edited via a
dialog (the in-tab table is hidden on the batch surface).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.simulate import build_builtin_template, simulate_run
from asymmetry.gui.windows.multi_group_fit_window import MultiGroupFitWindow


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def _tf(t, A=20.0, f=1.5, phi=0.0):  # noqa: N803 (A is the asymmetry symbol)
    return A * np.cos(2.0 * np.pi * f * np.asarray(t, dtype=float) + phi)


def _member(run_number: int, *, field: float, phi: float, seed: int) -> MuonDataset:
    """A grouped (F-B) member dataset with its own field and injected phase."""
    template = build_builtin_template("ideal_pulsed_fb")
    run = simulate_run(
        template, _tf, {"A": 20.0, "f": 1.5, "phi": phi}, total_events=6e6, alpha=1.2, seed=seed
    )
    run.run_number = run_number
    run.metadata["field"] = field
    time = np.linspace(0.0, 8.0, 400)
    return MuonDataset(
        time=time,
        asymmetry=0.2 * np.cos(2.0 * np.pi * 1.5 * time + phi),
        error=np.full_like(time, 0.01),
        metadata={"run_number": run_number, "field": field},
        run=run,
    )


def _physics_row(tab, name: str) -> int:
    table = tab._group_model_table
    for row in range(table.rowCount()):
        item = table.item(row, 0)
        if item is not None and item.data(Qt.ItemDataRole.UserRole) == name:
            return row
    raise AssertionError(f"physics param {name!r} not found")


def test_batch_nuisance_seeds_differ_per_dataset(qapp) -> None:
    # Each member dataset's groups are seeded from their OWN data (FFT phase,
    # counts), not a single representative run's.
    win = MultiGroupFitWindow()
    tab = win._batch_fit_tab
    tab.set_member_datasets(
        [
            _member(401, field=100.0, phi=0.3, seed=1),
            _member(402, field=250.0, phi=1.2, seed=2),
        ]
    )
    cfg = tab._parse_grouped_parameter_configuration()
    seeds = {}
    for run in (401, 402):
        groups = tab._grouped_members[run]
        params = tab._build_grouped_initial_params(groups, cfg, run_number=run)
        first = next(iter(params.values()))
        seeds[run] = (first["relative_phase"].value, first["N0"].value)
    # The two runs were injected with different phases/statistics, so their
    # per-group nuisance seeds must differ.
    assert seeds[401][0] != pytest.approx(seeds[402][0])
    assert seeds[401][1] != pytest.approx(seeds[402][1])


def test_batch_seed_helper_is_cached(qapp) -> None:
    win = MultiGroupFitWindow()
    tab = win._batch_fit_tab
    tab.set_member_datasets(
        [_member(403, field=100.0, phi=0.3, seed=3), _member(404, field=120.0, phi=0.7, seed=4)]
    )
    first = tab._grouped_member_nuisance_seeds()
    assert first is tab._grouped_member_nuisance_seeds()  # cache hit (no recompute)


def test_batch_physics_chain_seeds_local_per_run_global_averaged(qapp) -> None:
    # Registering each run's single grouped fit lets the batch seed Local physics
    # per run and Global physics from the cross-run average (FB parity).
    win = MultiGroupFitWindow()
    tab = win._batch_fit_tab
    tab.set_member_datasets(
        [_member(405, field=100.0, phi=0.3, seed=5), _member(406, field=100.0, phi=0.3, seed=6)]
    )
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    tab.register_grouped_single_fit_seed(405, model, {"Lambda": 0.5})
    tab.register_grouped_single_fit_seed(406, model, {"Lambda": 1.5})
    assert tab._inherited_model_dict is not None

    def lambda_seed(run: int) -> float:
        cfg = tab._parse_grouped_parameter_configuration()
        params = tab._build_grouped_initial_params(tab._grouped_members[run], cfg, run_number=run)
        return next(iter(params.values()))["Lambda"].value

    # Default role is Global → both runs seed from the average (1.0).
    assert lambda_seed(405) == pytest.approx(1.0)
    assert lambda_seed(406) == pytest.approx(1.0)

    # Flip Lambda to Local → each run seeds from its own single fit.
    combo = tab._group_model_table.cellWidget(_physics_row(tab, "Lambda"), 2)
    combo.setCurrentText("Local")
    assert lambda_seed(405) == pytest.approx(0.5)
    assert lambda_seed(406) == pytest.approx(1.5)


def test_send_to_batch_copies_model_and_seeds(qapp) -> None:
    win = MultiGroupFitWindow()
    win.set_dataset(_member(407, field=222.0, phi=0.3, seed=7))
    win._batch_fit_tab.set_member_datasets(
        [_member(407, field=222.0, phi=0.3, seed=7), _member(408, field=222.0, phi=0.3, seed=8)]
    )
    single = win._single_fit_tab
    single._set_composite_model(CompositeModel(["OscillatoryField", "Constant"], operators=["+"]))
    single._group_model_table.item(_physics_row(single, "field"), 1).setText("321.0")

    single.send_grouped_model_to_batch_requested.emit()

    batch = win._batch_fit_tab
    assert batch._composite_model.component_names == ["OscillatoryField", "Constant"]
    assert win._tabs.currentWidget() is batch
    assert float(batch._group_model_table.item(_physics_row(batch, "field"), 1).text()) == (
        pytest.approx(321.0)
    )


def test_batch_hides_per_group_table_single_keeps_it(qapp) -> None:
    win = MultiGroupFitWindow()
    win.set_dataset(_member(409, field=100.0, phi=0.3, seed=9))
    win._batch_fit_tab.set_member_datasets(
        [_member(409, field=100.0, phi=0.3, seed=9), _member(410, field=100.0, phi=0.3, seed=10)]
    )
    single, batch = win._single_fit_tab, win._batch_fit_tab
    single._update_mode_ui(preserve_result=False)
    batch._update_mode_ui(preserve_result=False)

    # Batch: per-group table hidden, button relabelled to the dialog action.
    assert batch._group_param_group.isHidden()
    assert batch._initial_values_btn.text() == "Edit per-group initial values…"
    # Single: per-group table kept (per-dataset editing in the tab).
    assert not single._group_param_group.isHidden()
    # The physics (fit-function) table stays on both surfaces.
    assert not batch._group_model_group.isHidden()

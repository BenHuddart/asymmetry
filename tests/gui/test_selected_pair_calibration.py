"""Display rebunching keeps the same calibrated F/B observation."""

import numpy as np
import pytest

from asymmetry.core.simulate import reduce_run_to_dataset
from asymmetry.gui.mainwindow import MainWindow
from tests.core.test_selected_pair_balances import pair_run

pytestmark = pytest.mark.gui


@pytest.mark.parametrize("canonical", [False, True])
def test_counts_first_rebunch_keeps_selected_pair_alpha(qapp, canonical):
    data = reduce_run_to_dataset(pair_run(canonical))
    window = MainWindow()
    assert window._apply_grouping_settings_to_dataset(data, data.run.grouping)[0]
    np.testing.assert_allclose(data.asymmetry, 25.0)
    assert window._plot_panel._alpha_value_for_dataset(data) == 0.75
    asymmetry = window._counts_first_rebunched_dataset(data, 2).asymmetry
    np.testing.assert_allclose(asymmetry, 25.0)
    window._clear_all_state()
    window.close()


@pytest.mark.parametrize("projection", [0, 1])
def test_coadd_rejects_different_declared_projection_balances(qapp, projection):
    first, second = pair_run(), pair_run()
    second.grouping["projections"][projection]["alpha"] = 0.9
    window = MainWindow()
    assert window._data_browser._coadd_compatibility_error(
        [reduce_run_to_dataset(first), reduce_run_to_dataset(second)]
    )
    window._clear_all_state()
    window.close()


def test_apply_scalar_layout_uses_fresh_beta_after_vector_layout(qapp):
    data = reduce_run_to_dataset(pair_run())
    window = MainWindow()
    payload = dict(data.run.grouping, alpha=0.75, beta=1.2)
    payload.pop("projections")
    payload.pop("vector_axis")
    assert window._apply_grouping_settings_to_dataset(data, payload)[0]
    np.testing.assert_allclose(data.asymmetry, 100 * (100 - 0.75 * 80) / (1.2 * 100 + 0.75 * 80))
    window._clear_all_state()
    window.close()

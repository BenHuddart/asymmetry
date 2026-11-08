"""TF calibration delegates acquired-period selection to the existing owner."""

import numpy as np

from asymmetry.core.fitting.tf_calibration import fit_tf_calibration
from tests.core.test_tf_calibration_fit import _raw_tf, _request


def test_selected_period_reuses_acquired_observation_owner():
    from asymmetry.core.io.periods import source_run_of

    red, green = _raw_tf(beta=1), _raw_tf(beta=1, seed=43)
    empty = (np.array([]), np.array([]), np.array([]))
    red.run.metadata.update(run_number=7101, source_run_number=7101, period_count=2)
    red.run.grouping.update(
        period_histograms=[red.run.histograms, green.run.histograms],
        period_reduced=[empty, empty],
        period_good_frames=[10.0, 20.0],
    )
    result = fit_tf_calibration(red, _request(period=2))
    direct = fit_tf_calibration(green, _request())
    assert result.ok and direct.ok
    assert result.alpha == direct.alpha
    assert result.fit.covariance == direct.fit.covariance
    assert result.source_run_number == source_run_of(red.run)
    assert result.period_number == 2

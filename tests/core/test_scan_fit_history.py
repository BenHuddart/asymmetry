"""Independent constant-baseline anchors for captured scalar analysis."""

import numpy as np
import pytest

from asymmetry.core.fitting.field_scan import fit_scan_baseline
from asymmetry.core.fitting.scan_history import ScanFitHistory, ScanFitRecord
from asymmetry.core.transform.integral import FieldScan


def test_captured_baseline_preserves_analytic_covariance():
    scan = FieldScan(
        np.arange(5.0), np.full(5, 10.0), np.full(5, 2.0), list(range(5)), "field", "integral"
    )
    baseline = fit_scan_baseline(scan, [(0.0, 4.0)], model="Constant")
    record = ScanFitRecord.capture(
        "baseline", scan, {"regions": [[0.0, 4.0]]}, baseline.model, baseline.fit, baseline.baseline
    )
    history = ScanFitHistory()
    history.capture(record)
    restored = ScanFitHistory.from_dict(history.to_dict()).records[0]
    result = restored.baseline_result()
    # Var(mean)=sigma²/N; Cov(y_i-mean,y_j-mean)=sigma²(delta_ij-1/N).
    np.testing.assert_allclose(restored.prediction, 10.0, atol=1e-8)
    assert result.fit.uncertainties["c"] == pytest.approx(np.sqrt(0.8), rel=1e-5)
    np.testing.assert_allclose(result.corrected_covariance, 4.0 * np.eye(5) - 0.8, atol=1e-5)
    np.testing.assert_allclose(result.corrected.value, 0.0, atol=1e-8)
    assert result.fit.computation_revision == baseline.fit.computation_revision

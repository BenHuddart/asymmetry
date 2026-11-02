"""Information criteria have explicit domains and never substitute another metric."""

from types import SimpleNamespace

import pytest

from asymmetry.core.fitting.information_criteria import InformationCriteria, SelectionMetric, rank


@pytest.mark.parametrize(
    "deviance,k,n,expected",
    [
        (1.0, 3, 5, (7.0, 31.0, 5.828313737302301)),
        (10.0, 3, 100, (16.0, 16.25, 23.815510557964274)),
    ],
)
def test_criteria_match_analytic_table(deviance, k, n, expected):
    record = InformationCriteria(deviance, k, n)
    assert tuple(record.value(metric) for metric in SelectionMetric) == pytest.approx(expected)


@pytest.mark.parametrize("k,n", [(3, 4), (0, 1)])
def test_aicc_is_undefined_at_its_pole(k, n):
    record = InformationCriteria(1.0, k, n)
    assert record.value(SelectionMetric.AICC) is None
    assert record.unavailable_reason(SelectionMetric.AICC) == "n − k − 1 = 0"


@pytest.mark.parametrize(
    "deviance,k,n",
    [
        (float("nan"), 1, 5),
        (float("inf"), 1, 5),
        (1.0, -1, 5),
        (1.0, 2.5, 5),
        (1.0, 1, 0),
        (1.0, 1, -1),
    ],
)
def test_invalid_inputs_raise(deviance, k, n):
    with pytest.raises(ValueError):
        InformationCriteria(deviance, k, n)


def test_rank_excludes_unavailable():
    a = SimpleNamespace(criteria=InformationCriteria(1.0, 3, 4))
    b = SimpleNamespace(criteria=InformationCriteria(10.0, 1, 4))
    assert rank([a, b], SelectionMetric.AIC) == ([a, b], [])
    assert rank([a, b], SelectionMetric.AICC) == ([b], [(a, "n − k − 1 = 0")])
    assert b.criteria.value(SelectionMetric.AICC) == 14.0


def test_unknown_metric_is_rejected():
    from asymmetry.core.fitting.information_criteria import ranking_value

    with pytest.raises(ValueError, match="unknown selection metric"):
        SelectionMetric.from_value("typo")
    with pytest.raises(ValueError, match="unknown selection metric"):
        ranking_value(None, "typo")


def test_unknown_variance_likelihood_is_scale_invariant():
    import math

    deltas = []
    for scale in (1.0, 0.1, 1e3):
        # Fitted mean columns 3 and 2, plus the unknown common variance.
        a = InformationCriteria(20 * math.log(2 * scale**2 / 20), 4, 20)
        b = InformationCriteria(20 * math.log(5 * scale**2 / 20), 3, 20)
        deltas.append(b.value(SelectionMetric.AICC) - a.value(SelectionMetric.AICC))
    assert deltas == pytest.approx([15.159147970816434] * 3)


def test_partition_bic_limits():
    import math

    from asymmetry.core.fitting.information_criteria import partition_bic

    assert partition_bic(
        30.0, 0, 3, sample_count=30, local_log_sample_count=3 * math.log(10)
    ) == pytest.approx(50.72326583694641)
    assert partition_bic(
        30.0, 3, 0, sample_count=30, local_log_sample_count=3 * math.log(10)
    ) == pytest.approx(40.203592144986465)

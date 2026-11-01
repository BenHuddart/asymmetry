"""Build coherent synthetic fit statistics for ranking-policy tests."""

from dataclasses import replace

from asymmetry.core.fitting.fit_wizard import CandidateAssessment
from asymmetry.core.fitting.information_criteria import InformationCriteria, SelectionMetric


def with_aicc(assessment, value):
    """Set a non-negative objective whose derived AICc equals the requested score."""
    penalty = InformationCriteria(0.0, assessment.parameter_count, assessment.sample_count).value(
        SelectionMetric.AICC
    )
    if penalty is None or value < penalty:
        raise ValueError("ranking fixture requires defined AICc and non-negative objective")
    statistic = value - penalty
    if isinstance(assessment, CandidateAssessment):
        return replace(assessment, fit_result=replace(assessment.fit_result, chi_squared=statistic))
    results = assessment.fit_results_by_run
    return replace(
        assessment,
        fit_results_by_run={
            run: replace(result, chi_squared=statistic / len(results))
            for run, result in results.items()
        },
    )

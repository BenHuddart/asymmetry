"""Preserve recorded objectives and recover supported historical fit statistics."""

from asymmetry.core.fitting.fit_quality import infer_historical_dof, quality_inapplicability_reason
from asymmetry.core.fitting.trend_result import ErrorMode


def migrate_fit_objectives(data: dict) -> dict:
    """Record known Gaussian producers and retain unknown grouped objectives explicitly."""

    def summary(value, objective):
        result = dict(value) | {"objective": objective, "minimum_expected_count": None}
        if objective is None:
            result.update(quality=None, quality_reason="Fit objective was not recorded.")
        return result

    def walk(value):
        if isinstance(value, list):
            return [walk(item) for item in value]
        if not isinstance(value, dict):
            return value
        result = {key: walk(item) for key, item in value.items()}
        if {"parameters", "provenance", "result"} <= value.keys() and isinstance(
            value["result"], dict
        ):
            result["result"] = summary(value["result"], "gaussian")
        if "member_kind" in value and isinstance(value.get("results_by_run"), dict):
            objective = "gaussian" if value["member_kind"] == "runs" else None
            result["results_by_run"] = {
                key: summary(item, objective) if isinstance(item, dict) else item
                for key, item in value["results_by_run"].items()
            }
        return result

    return walk(data) | {"schema_version": 28}


def _migrate_summary(record: dict) -> None:
    """A saved ν = 1 could come from the historical divisor floor."""
    statistic, reduced = record["chi_squared"], record["reduced_chi_squared"]
    dof = record.get("ndof")
    if type(dof) is not int or dof == 1:
        dof = (
            infer_historical_dof(statistic, reduced)
            if isinstance(statistic, (int, float)) and isinstance(reduced, (int, float))
            else None
        )
    record["ndof"] = dof
    if dof is None or dof <= 0:
        if isinstance(reduced, (int, float)):
            record["historical_reduced_chi_squared"] = reduced
        record.update(
            reduced_chi_squared=None,
            quality=None,
            quality_reason=quality_inapplicability_reason(
                dof, record.get("quality_reason") or None
            ),
        )
        record.pop("result_html", None)


def migrate_fit_statistic_applicability(data: dict) -> None:
    """Migrate the project's copied fit summaries before any saved verdict is displayed."""

    def walk(value):
        if isinstance(value, list):
            for item in value:
                walk(item)
        if not isinstance(value, dict):
            return
        for item in value.values():
            walk(item)
        if (
            {"chi_squared", "reduced_chi_squared", "parameters"} <= value.keys()
            and isinstance(value["parameters"], dict)
            and ("objective" in value or "quality" in value)
        ):
            _migrate_summary(value)
        result, form = value.get("result"), value.get("ui_state")
        if (
            isinstance(form, dict)
            and isinstance(result, dict)
            and "historical_reduced_chi_squared" in result
            and result.get("reduced_chi_squared") is None
        ):
            form.pop("result_html", None)

    walk(data)


def migrate_trend_column_floor(data: dict) -> None:
    """Identify the floor used by known single-series and scan executions."""

    def mark(result):
        if (
            isinstance(result, dict)
            and result.get("error_mode", ErrorMode.COLUMN.value) == ErrorMode.COLUMN
        ):
            result["error_mode"] = ErrorMode.COLUMN_STABILIZED.value

    def walk(value):
        if isinstance(value, list):
            for item in value:
                walk(item)
        if not isinstance(value, dict):
            return
        for item in value.values():
            walk(item)
        model_fits = value.get("model_fits")
        if isinstance(model_fits, dict):
            for fit in model_fits.values():
                if not isinstance(fit, dict) or not isinstance(fit.get("ranges"), list):
                    continue
                for entry in fit["ranges"]:
                    result = entry.get("result") if isinstance(entry, dict) else None
                    if isinstance(result, dict) and (
                        result.get("covariance") is not None
                        or (type(result.get("n_points")) is int and result["n_points"] > 0)
                    ):
                        mark(result)
        history = value.get("fit_history")
        if isinstance(history, dict) and isinstance(history.get("records"), list):
            for record in history["records"]:
                if isinstance(record, dict) and record.get("kind") in ("baseline", "peaks", "rf"):
                    mark(record.get("result"))

    walk(data)

"""Preserve stored scan results without inferring an unevidenced estimator."""


def migrate_scan_identity(data):
    """Label saved scalar scans once, without recalculating or changing values."""
    if isinstance(data, list):
        return [migrate_scan_identity(item) for item in data]
    if not isinstance(data, dict):
        return data
    result = {key: migrate_scan_identity(value) for key, value in data.items()}
    if result.get("kind") == "alc_scan":
        result.setdefault("estimator", "legacy_unspecified")
        result.setdefault("time_window_us", None)
    return result


def migrate_scan_observations(data):
    """Retain stored values with unknown mathematical status and source identity."""
    if isinstance(data, list):
        return [migrate_scan_observations(item) for item in data]
    if not isinstance(data, dict):
        return data
    result = {key: migrate_scan_observations(value) for key, value in data.items()}
    if result.get("kind") == "alc_scan":
        result.setdefault("correlated_pairs", [])
    extra = result.get("extra")
    if isinstance(extra, dict) and extra.get("kind") == "alc_scan":
        for summary in result.get("results_by_run", {}).values():
            summary.setdefault("observation", None)
    return result


def migrate_scan_fit_history(data):
    """Archive settings-only fits without inventing a numerical execution."""
    if isinstance(data, list):
        return [migrate_scan_fit_history(item) for item in data]
    if not isinstance(data, dict):
        return data
    result = {key: migrate_scan_fit_history(value) for key, value in data.items()}
    if result.get("kind") == "alc_scan" and "fit_history" not in result:
        legacy = dict(result)
        flags = {kind: result.pop(kind + "_fitted", False) for kind in ("baseline", "peaks", "rf")}
        # The legacy peak flag also described the RF overlay's shared curve slot.
        missing = (
            ["rf"]
            if result.get("rf_mode") and flags["rf"]
            else [kind for kind in ("baseline", "peaks") if flags[kind]]
        )
        result["fit_history"] = {
            "records": [],
            "active": {},
            "missing": missing,
            "legacy_settings": legacy if missing else {},
        }
    return result

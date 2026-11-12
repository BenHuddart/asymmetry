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

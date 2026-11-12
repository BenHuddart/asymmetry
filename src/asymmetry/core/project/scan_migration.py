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

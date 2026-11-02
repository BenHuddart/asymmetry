"""Strict JSON representations for non-finite scientific values."""

from __future__ import annotations

import math

_NONFINITE_KEY = "__nonfinite__"


def encode_non_finite(obj: object) -> object:
    """Preserve unavailable scientific values in strict JSON, normalizing numpy values."""
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj):
            return {_NONFINITE_KEY: "NaN"}
        if math.isinf(obj):
            return {_NONFINITE_KEY: "Infinity" if obj > 0 else "-Infinity"}
        return obj
    if isinstance(obj, dict):
        return {k: encode_non_finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [encode_non_finite(v) for v in obj]
    try:
        import numpy as np

        if isinstance(obj, np.floating):
            return encode_non_finite(float(obj))
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return encode_non_finite(obj.tolist())
    except ImportError:
        pass
    return obj


def decode_non_finite(obj: object) -> object:
    """Recursively reconstruct non-finite floats from the save-time wrapper."""
    if isinstance(obj, dict):
        if len(obj) == 1 and _NONFINITE_KEY in obj:
            try:
                return float(obj[_NONFINITE_KEY])
            except (TypeError, ValueError):
                return obj
        return {k: decode_non_finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode_non_finite(v) for v in obj]
    return obj


def json_default(obj: object) -> object:
    """JSON serializer for numpy array/scalar types."""
    try:
        import numpy as np  # local import to avoid hard dependency at module level

        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    raise TypeError(f"Object of type {type(obj).__name__!r} is not JSON serializable")

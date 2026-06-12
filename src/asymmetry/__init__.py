"""Asymmetry — a Python library for μSR data analysis."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _metadata_version

try:
    __version__: str = _metadata_version("asymmetry")
except PackageNotFoundError:  # package not installed
    __version__ = "unknown"

#: Top-level user-function API, resolved lazily so ``import asymmetry`` stays
#: light (no numpy/scipy import until the API is actually used).
_USER_FUNCTION_EXPORTS = {
    "UserFunctionError": "asymmetry.core.fitting.user_functions",
    "register_component": "asymmetry.core.fitting.user_functions",
    "register_parameter_component": "asymmetry.core.fitting.user_functions",
    "load_user_functions": "asymmetry.core.plugins",
}


def __getattr__(name: str):
    module_path = _USER_FUNCTION_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module 'asymmetry' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_path), name)


def __dir__() -> list[str]:
    return sorted({*globals(), *_USER_FUNCTION_EXPORTS})

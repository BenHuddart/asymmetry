"""Asymmetry — a Python library for μSR data analysis."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _metadata_version

try:
    __version__: str = _metadata_version("asymmetry")
except PackageNotFoundError:  # package not installed
    __version__ = "unknown"

"""Tests for loader registry and I/O convenience functions."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.io import load
from asymmetry.core.io.base import BaseLoader, LoaderRegistry


class _DummyLoader(BaseLoader):
    extensions = [".dum", ".D2"]
    format_name = "Dummy"

    def load(self, filepath: str) -> MuonDataset:
        t = np.array([0.0, 1.0])
        return MuonDataset(
            time=t,
            asymmetry=np.array([0.1, 0.2]),
            error=np.array([0.01, 0.01]),
            metadata={"run_number": 99, "source": filepath},
        )


def test_registry_register_get_loader_and_supported_extensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(LoaderRegistry, "_loaders", {})
    LoaderRegistry.register(_DummyLoader)

    loader = LoaderRegistry.get_loader("example.dum")
    assert isinstance(loader, _DummyLoader)

    loader_fmt = LoaderRegistry.get_loader("anything.unknown", fmt="d2")
    assert isinstance(loader_fmt, _DummyLoader)

    exts = LoaderRegistry.supported_extensions()
    assert ".dum" in exts
    assert ".d2" in exts


def test_registry_unknown_extension_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(LoaderRegistry, "_loaders", {})
    with pytest.raises(ValueError, match="No loader registered"):
        LoaderRegistry.get_loader("data.unknown")


def test_load_convenience_uses_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    class _OneShotLoader(BaseLoader):
        extensions = [".abc"]

        def load(self, filepath: str) -> MuonDataset:
            return MuonDataset(
                time=np.array([0.0]),
                asymmetry=np.array([0.0]),
                error=np.array([1.0]),
                metadata={"run_number": 1, "path": filepath},
            )

    monkeypatch.setattr(LoaderRegistry, "_loaders", {".abc": _OneShotLoader})
    ds = load("run.abc")
    assert ds.run_number == 1
    assert ds.metadata["path"] == "run.abc"

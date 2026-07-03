"""Tests for package entry points."""

from __future__ import annotations

import runpy


def test_module_main_invokes_cli_main(monkeypatch):
    called = []

    def _fake_main():
        called.append(True)

    monkeypatch.setattr("asymmetry.cli.main", _fake_main)
    runpy.run_module("asymmetry.__main__", run_name="__main__")

    assert called == [True]

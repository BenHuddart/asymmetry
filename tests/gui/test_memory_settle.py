"""``gui/utils/memory.py`` — post-load garbage-collector settling."""

from __future__ import annotations

import gc

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

import asymmetry.gui.mainwindow as mw_module  # noqa: E402
from asymmetry.gui.mainwindow import MainWindow  # noqa: E402
from asymmetry.gui.utils import memory  # noqa: E402

pytestmark = [pytest.mark.gui]


def test_settle_memory_collects_then_freezes_survivors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ASYMMETRY_GC_FREEZE", raising=False)
    try:
        stats = memory.settle_memory()
        assert stats is not None
        assert stats["frozen"] > 0
        assert gc.get_freeze_count() == stats["frozen"]
        # A repeat first unfreezes, so nothing frozen earlier is left uncollectable.
        again = memory.settle_memory()
        assert again is not None and again["frozen"] > 0
    finally:
        gc.unfreeze()


def test_settle_memory_is_disabled_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASYMMETRY_GC_FREEZE", "0")
    assert memory.gc_freeze_enabled() is False
    assert memory.settle_memory() is None


def test_schedule_memory_settle_coalesces_onto_one_event_loop_turn(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    try:
        qapp.processEvents()  # drain the settle the constructor's clear-all scheduled
        calls: list[int] = []
        monkeypatch.setattr(mw_module, "settle_memory", lambda: calls.append(1) or None)

        window._schedule_memory_settle()
        window._schedule_memory_settle()
        assert calls == []  # deferred, never synchronous
        qapp.processEvents()
        assert calls == [1]
    finally:
        window.close()
        window.deleteLater()

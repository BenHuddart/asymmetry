from __future__ import annotations

import time

from PySide6.QtWidgets import QApplication


def wait_for(predicate: object, qapp: QApplication, timeout_s: float = 0.5) -> None:
    """Poll *predicate* until it returns truthy or *timeout_s* elapses.

    Fits in tests are mocked, so 0.5 s is ample. Pass a larger timeout_s only
    when the test exercises genuinely slow real work.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(f"Timed out after {timeout_s}s waiting for UI state")

from __future__ import annotations

import time

from PySide6.QtWidgets import QApplication


def wait_for(predicate: object, qapp: QApplication, timeout_s: float = 5.0) -> None:
    """Poll *predicate* until it returns truthy or *timeout_s* elapses.

    Fits in tests are mocked, so the predicate normally flips within
    milliseconds and the deadline is never approached — it only bounds how
    long a genuinely broken test takes to fail. It must still be generous:
    under a fully loaded machine (full-tier xdist plus the wizard's process
    pools) QThread scheduling and queued-signal delivery alone can take
    longer than the 0.5 s this helper originally allowed, which made GUI
    window tests flake nondeterministically. Pass a larger timeout_s only
    when the test exercises genuinely slow real work.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(f"Timed out after {timeout_s}s waiting for UI state")

"""The screenshot-capture watchdog must survive GIL starvation.

A documentation-screenshot scenario runs a fit on a Qt worker thread. A long
``iminuit``/``migrad`` minimisation holds the GIL inside its native call, which
starves *every* Python thread. A previous watchdog was a pure-Python daemon that
slept and then called ``os._exit`` — but it could never wake to run that call
while the GIL was held, so a CI run hung for ~3.5 hours instead of
self-terminating.

The fix routes the watchdog through :func:`faulthandler.dump_traceback_later`,
whose timer fires from C without needing the GIL (and dumps every thread's stack
for diagnosis). These tests pin that wiring so the GIL-independent guarantee
cannot silently regress to a Python-thread timer again.
"""

from __future__ import annotations

import faulthandler

from docs.screenshots import capture


def test_watchdog_uses_faulthandler_timer(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(faulthandler, "enable", lambda *a, **k: calls.append(("enable", a, k)))
    monkeypatch.setattr(
        faulthandler,
        "dump_traceback_later",
        lambda *a, **k: calls.append(("later", a, k)),
    )

    capture._start_watchdog(timeout_s=123)

    later = [c for c in calls if c[0] == "later"]
    assert later, "watchdog must arm faulthandler.dump_traceback_later"
    _, args, kwargs = later[0]
    # The timeout is passed positionally; exit=True is what actually kills the
    # process (a bare dump would log and then hang forever again).
    assert args and args[0] == 123
    assert kwargs.get("exit") is True


def test_watchdog_does_not_spawn_a_python_daemon_thread(monkeypatch):
    # Guard against a regression back to a threading.Thread daemon, which is the
    # GIL-starvable design that caused the multi-hour hang.
    import threading

    created: list = []
    real_thread = threading.Thread

    def _spy(*args, **kwargs):
        created.append((args, kwargs))
        return real_thread(*args, **kwargs)

    monkeypatch.setattr(threading, "Thread", _spy)
    monkeypatch.setattr(faulthandler, "enable", lambda *a, **k: None)
    monkeypatch.setattr(faulthandler, "dump_traceback_later", lambda *a, **k: None)

    capture._start_watchdog(timeout_s=5)

    assert created == [], "watchdog must not rely on a Python daemon thread"


def test_capture_timeout_is_bounded():
    # A generous-but-finite cap; the full set normally renders in well under a
    # minute. Catch an accidental bump back to a multi-tens-of-minutes value.
    assert 0 < capture._CAPTURE_TIMEOUT_S <= 15 * 60

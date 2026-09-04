"""Spawn-pool teardown: ``terminate_spawn_pool`` kills *and* reaps workers.

The fit wizard's Cancel path tears its spawn pool down immediately rather than
blocking on the in-flight fits (a worker cannot be interrupted mid-fit). That
teardown must leave no live worker (an orphan) and no un-reaped worker (a
zombie) — a bare ``kill()`` without a following ``join()`` trades the orphan for
a zombie, which still fails the "no leaked processes" bar. This exercises a real
``spawn`` pool to prove the force-kill-and-reap actually holds.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import subprocess
import sys
import time
import types
import warnings

import pytest

from asymmetry._worker_env import BLAS_THREAD_ENV_VARS, blas_thread_pins
from asymmetry.core.fitting.process_pool import (
    SpawnUnsafeWarning,
    main_module_has_spawn_guard,
    open_spawn_pool,
    reset_spawn_safety_warning,
    spawn_pool_unsafe_reason,
    terminate_spawn_pool,
)


def _sleep_forever(_seed: int) -> None:
    # Module-level so it is picklable under the ``spawn`` start method. Never
    # returns, so the worker running it can only be stopped by a kill — exactly
    # the "in-flight fit cannot be interrupted" situation the teardown handles.
    while True:
        time.sleep(0.05)


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_terminate_spawn_pool_kills_and_reaps_workers() -> None:
    pool = open_spawn_pool(2)
    if pool is None:
        pytest.skip("spawn pool unavailable in this environment.")

    # Fill both workers with never-ending work, then wait for the pool to have
    # actually spawned the worker processes (it does so lazily on submit).
    for seed in range(4):
        pool.submit(_sleep_forever, seed)

    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline and len(pool._processes) < 2:
        time.sleep(0.05)
    # Snapshot the workers *before* teardown — shutdown clears ``_processes``.
    processes = list(pool._processes.values())
    assert processes, "expected the spawn pool to have started worker processes"
    assert all(proc.is_alive() for proc in processes)

    start = time.monotonic()
    terminate_spawn_pool(pool)
    elapsed = time.monotonic() - start

    # Fast: no blocking wait for the never-ending tasks to finish.
    assert elapsed < 15.0

    # No orphans and no zombies: every worker is dead and has been reaped
    # (a reaped process reports a non-None exitcode; a signalled one is
    # negative). Poll with a deadline rather than asserting on one
    # instantaneous check: the executor's management thread races
    # ``terminate_spawn_pool`` to ``waitpid`` the same child, and the loser's
    # ``Process.poll()`` swallows ``ECHILD`` as ``None`` — so ``is_alive()``
    # can transiently report True for an already-reaped child until the
    # winning thread stores the exit code (a starved CI host stretches that
    # window to seconds). A single ``join`` doesn't help, because it returns
    # through the same ``ECHILD`` path without waiting.
    def _dead_and_reaped() -> bool:
        return all(not proc.is_alive() and proc.exitcode is not None for proc in processes)

    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline and not _dead_and_reaped():
        time.sleep(0.05)
    for proc in processes:
        assert not proc.is_alive()
        assert proc.exitcode is not None


# --------------------------------------------------------------------------- #
# Spawn safety. Under the ``spawn`` start method every worker re-imports the
# parent's ``__main__`` module before it can run anything. A plain script with
# no ``if __name__ == "__main__":`` guard therefore re-runs its whole body in
# every worker — which either re-enters the analysis N times over or trips
# multiprocessing's own bootstrap ``RuntimeError``. Reported from a downstream
# continuous-source ZF/TF analysis, where an unguarded driver script crashed
# inside the single-curve wizard's default pool.
# --------------------------------------------------------------------------- #

GUARDED_SCRIPT = """\
import sys

def main():
    print("ran")

if __name__ == "__main__":
    main()
"""

UNGUARDED_SCRIPT = """\
import sys

print("ran")
"""


def _fake_main_module(path) -> types.ModuleType:
    module = types.ModuleType("__main__")
    module.__file__ = str(path)
    return module


def test_spawn_is_safe_for_a_guarded_main_script(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = tmp_path / "guarded.py"
    script.write_text(GUARDED_SCRIPT, encoding="utf-8")
    monkeypatch.setitem(sys.modules, "__main__", _fake_main_module(script))
    main_module_has_spawn_guard.cache_clear()

    assert spawn_pool_unsafe_reason() is None


def test_spawn_is_unsafe_for_an_unguarded_main_script(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "unguarded.py"
    script.write_text(UNGUARDED_SCRIPT, encoding="utf-8")
    monkeypatch.setitem(sys.modules, "__main__", _fake_main_module(script))
    main_module_has_spawn_guard.cache_clear()

    reason = spawn_pool_unsafe_reason()
    assert reason is not None
    assert '__name__ == "__main__"' in reason


def test_unguarded_main_degrades_open_spawn_pool_to_serial(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``open_spawn_pool`` returns ``None`` (the serial signal) and warns once."""
    script = tmp_path / "unguarded.py"
    script.write_text(UNGUARDED_SCRIPT, encoding="utf-8")
    monkeypatch.setitem(sys.modules, "__main__", _fake_main_module(script))
    main_module_has_spawn_guard.cache_clear()
    reset_spawn_safety_warning()

    with pytest.warns(SpawnUnsafeWarning, match="serial"):
        assert open_spawn_pool(4) is None
    # Warned once per process, not once per pool: a wizard run opens several.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert open_spawn_pool(4) is None


def test_single_worker_is_always_a_serial_escape_hatch(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``max_workers=1`` never starts a worker process, however hostile the host."""
    script = tmp_path / "unguarded.py"
    script.write_text(UNGUARDED_SCRIPT, encoding="utf-8")
    monkeypatch.setitem(sys.modules, "__main__", _fake_main_module(script))
    main_module_has_spawn_guard.cache_clear()
    reset_spawn_safety_warning()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert open_spawn_pool(1) is None
        assert open_spawn_pool(0) is None


def test_spawn_is_unsafe_while_a_worker_re_imports_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inside a spawn worker's ``__main__`` re-import, no nested pool is opened.

    This is the state that turns one unguarded call into an unbounded cascade of
    processes, so it is refused at the root.
    """
    monkeypatch.setattr(mp.current_process(), "_inheriting", True, raising=False)

    reason = spawn_pool_unsafe_reason()
    assert reason is not None
    assert "re-import" in reason


UNGUARDED_WIZARD_SCRIPT = """\
import numpy as np

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.fit_wizard import build_fit_wizard_recommendation

time_us = np.linspace(0.0, 8.0, 240)
asym = 20.0 * np.exp(-0.6 * time_us)
dataset = MuonDataset(
    time=time_us,
    asymmetry=asym,
    error=np.full_like(time_us, 0.3),
    metadata={"run_number": 1, "temperature": 10.0, "field": 0.0,
              "field_direction": "ZF"},
)
recommendation = build_fit_wizard_recommendation(dataset)
print("WIZARD-OK", recommendation.recommended_key is not None)
"""


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_unguarded_script_runs_the_wizard_once_without_crashing(tmp_path) -> None:
    """A script with no ``__main__`` guard completes, serially, exactly once.

    Pre-fix this either raised multiprocessing's spawn-bootstrap
    ``RuntimeError`` or silently re-ran the whole script in every worker (one
    "WIZARD-OK" line per process). Both are failures; the fix degrades to a
    single serial run and says so.
    """
    script = tmp_path / "unguarded_wizard.py"
    script.write_text(UNGUARDED_WIZARD_SCRIPT, encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=540,
        cwd=str(tmp_path),
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.count("WIZARD-OK True") == 1, completed.stdout
    assert "Traceback" not in completed.stderr, completed.stderr
    assert "bootstrapping phase" not in completed.stderr, completed.stderr
    assert '__name__ == "__main__"' in completed.stderr, completed.stderr


# --------------------------------------------------------------------------- #
# BLAS thread pinning in the workers
# --------------------------------------------------------------------------- #


def _worker_blas_env(_ignored: int) -> dict[str, str]:
    """Module-level (picklable under ``spawn``): report the worker's pins."""
    return {name: os.environ.get(name, "") for name in BLAS_THREAD_ENV_VARS}


def test_blas_thread_pins_skip_variables_the_caller_already_set() -> None:
    """Setting any of them is the documented opt-out; an explicit value wins."""
    assert blas_thread_pins({}) == dict.fromkeys(BLAS_THREAD_ENV_VARS, "1")

    partial = blas_thread_pins({"OMP_NUM_THREADS": "4"})

    assert "OMP_NUM_THREADS" not in partial
    assert partial["MKL_NUM_THREADS"] == "1"
    assert blas_thread_pins(dict.fromkeys(BLAS_THREAD_ENV_VARS, "8")) == {}


def test_pinning_never_touches_the_parent_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wizard does not get to re-thread the caller's own numpy."""
    for name in BLAS_THREAD_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    pool = open_spawn_pool(2)
    try:
        assert all(name not in os.environ for name in BLAS_THREAD_ENV_VARS)
    finally:
        if pool is not None:
            pool.shutdown()


@pytest.mark.integration
@pytest.mark.timeout(180)
def test_spawn_workers_start_with_their_blas_threads_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inside a pool the oversubscription is worst and unreachable from outside."""
    for name in BLAS_THREAD_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    pool = open_spawn_pool(2)
    if pool is None:
        pytest.skip("spawn workers are unavailable in this environment")
    try:
        observed = pool.submit(_worker_blas_env, 0).result(timeout=120)
    finally:
        pool.shutdown()

    assert observed == dict.fromkeys(BLAS_THREAD_ENV_VARS, "1")

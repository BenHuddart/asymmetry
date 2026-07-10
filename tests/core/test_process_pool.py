"""Spawn-pool teardown: ``terminate_spawn_pool`` kills *and* reaps workers.

The fit wizard's Cancel path tears its spawn pool down immediately rather than
blocking on the in-flight fits (a worker cannot be interrupted mid-fit). That
teardown must leave no live worker (an orphan) and no un-reaped worker (a
zombie) — a bare ``kill()`` without a following ``join()`` trades the orphan for
a zombie, which still fails the "no leaked processes" bar. This exercises a real
``spawn`` pool to prove the force-kill-and-reap actually holds.
"""

from __future__ import annotations

import time

import pytest

from asymmetry.core.fitting.process_pool import open_spawn_pool, terminate_spawn_pool


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

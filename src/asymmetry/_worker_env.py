"""Environment set-up run inside a spawned worker, before it imports numpy.

Why this is a top-level module with no imports
----------------------------------------------
A :class:`~concurrent.futures.ProcessPoolExecutor` initializer is pickled **by
reference**, so a spawn worker imports the module that defines it during its
bootstrap — before it unpickles any task and therefore before it imports numpy.
That ordering is the whole point here: a BLAS reads its thread-count
environment variables when the shared library is *loaded*, i.e. at
``import numpy``, and setting them afterwards does nothing.  So this module
deliberately imports nothing but :mod:`os`, and lives at the top level of the
package rather than under ``asymmetry.core.fitting`` — importing anything in
that package pulls numpy in, which would put the load before the pin and make
the whole exercise inert.

(The ordering still depends on the host: under ``spawn``, a worker re-imports
the parent's ``__main__`` before anything else, so an entry-point script that
imports numpy at module level has already loaded the BLAS by the time the
initializer runs.  Nothing in-process can fix that case; the environment knob
documented in :mod:`asymmetry.core.fitting.process_pool` is the answer there.)
"""

from __future__ import annotations

import os

#: The thread-count variables every BLAS this project might be built against
#: reads at load time: OpenMP (the generic one), OpenBLAS, MKL, Apple's
#: Accelerate/vecLib, and NumExpr.
BLAS_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def blas_thread_pins(environ: dict[str, str] | None = None) -> dict[str, str]:
    """The BLAS thread pins to apply in a worker, given the caller's environment.

    ``{var: "1"}`` for every variable in :data:`BLAS_THREAD_ENV_VARS` the caller
    has **not** already set, and nothing for the ones it has — setting any of
    them is the documented opt-out (see
    :mod:`asymmetry.core.fitting.process_pool`), and a user who asked for four
    BLAS threads must get four.  An empty result means "the environment is
    already configured", and the pool then starts its workers with no
    initializer at all.
    """
    env = os.environ if environ is None else environ
    return {name: "1" for name in BLAS_THREAD_ENV_VARS if not env.get(name)}


def pin_worker_blas_threads(pins: dict[str, str]) -> None:
    """Process-pool worker initializer: apply ``pins`` to this worker's environment.

    Runs in the child only.  The parent's environment is never touched — pinning
    it would change the threading of the caller's own numpy for the rest of the
    session, which is not a decision a fit wizard gets to make.
    """
    os.environ.update(pins)

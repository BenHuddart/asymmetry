"""Projection backend selection for the MaxEnt engine.

The MaxEnt hot loop (``asymmetry.core.maxent.engine``) never materialises the
full ``n_time × n_freq`` cosine design matrix; it regenerates it in row chunks.
This module wraps the choice of array library used for those chunks so the same
kernel code runs on NumPy (the default, CPU) or CuPy (an optional CUDA GPU
backend) without duplicating the math.

The NumPy backend is the exact code path that has always run: same row-chunk
element cap, same operation order, bit-for-bit identical output.  The CUDA
backend is fp64 only — no fp32 anywhere — and its results agree with the NumPy
path to solver tolerance, not bit-for-bit (a GPU reduction sums in a different
order).  CuPy is an optional dependency: ``import cupy`` happens lazily inside
:func:`resolve_backend` and never at module import time, so this module — and
the whole core layer — stays importable with no GPU, no CuPy, and no CUDA
driver present.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

# Row-chunk element cap for the design matrix on the NumPy backend.  Owned here
# (rather than in engine.py) so backend.py has no import dependency on the
# engine; engine.py imports it back for its default chunk size.  2e6 fp64
# elements ≈ 16 MB per transient matrix — small enough to stay in cache-friendly
# working sets while amortising the Python-level chunk loop.
_MAX_DESIGN_CHUNK_ELEMENTS = 2_000_000


# The backend names ``resolve_backend`` (and ``MaxEntConfig.from_dict``) accept.
# Owned here so the choices live in one place; the engine's recipe parser imports
# this rather than re-listing the strings.
BACKEND_CHOICES: tuple[str, ...] = ("numpy", "cuda", "auto")


class MaxEntBackendError(RuntimeError):
    """Raised when a requested MaxEnt projection backend cannot be provided."""


@dataclass(frozen=True)
class ProjectionBackend:
    """One array-library binding for the MaxEnt projection kernels.

    ``xp`` is the array module (``numpy`` or ``cupy``); ``chunk_elements`` is the
    row-chunk element cap for the regenerated design matrix; ``asarray`` moves a
    host array onto the backend as fp64; ``to_numpy`` brings a backend array back
    to a NumPy array so every kernel returns host arrays to its callers.
    """

    name: str
    xp: Any
    chunk_elements: int
    asarray: Callable[[Any], Any]
    to_numpy: Callable[[Any], Any]


def _numpy_asarray(array: Any) -> Any:
    return np.asarray(array, dtype=np.float64)


def _numpy_to_numpy(array: Any) -> Any:
    # Already a NumPy array on this backend; ``asarray`` is a no-op that keeps
    # the return contract (a NumPy array) explicit at the call sites.
    return np.asarray(array)


_NUMPY_BACKEND = ProjectionBackend(
    name="numpy",
    xp=np,
    chunk_elements=_MAX_DESIGN_CHUNK_ELEMENTS,
    asarray=_numpy_asarray,
    to_numpy=_numpy_to_numpy,
)

# Only the imported-and-validated CuPy module is cached: the import plus the
# device-count probe are the slow, one-time parts.  The free-memory query and
# the (cheap, frozen) ProjectionBackend are redone on every resolve so the chunk
# size tracks the device's live memory state instead of freezing at first-call
# values.
_CUPY_MODULE: Any | None = None


def _cuda_chunk_elements(free_bytes: int) -> int:
    """Return the design-matrix row-chunk element cap for *free_bytes* of GPU memory.

    Budgets the regenerated design matrix at 1/8 of free memory (``8 * 8`` = 8
    bytes per fp64 element times an 8× headroom over the raw matrix).  That
    headroom covers the ~2 concurrent chunk-sized transients the kernel holds
    — the ``time × freq`` angle product and its cos/sin result — plus the
    O(n_freq) vectors, with margin.  There is deliberately no hard floor: a
    low-memory device gets a genuinely small chunk instead of a guaranteed OOM.
    Capped at ``1 << 28`` (≈2 GB per matrix); never below one element.
    """
    return min(1 << 28, max(1, int(free_bytes) // (8 * 8)))


def _validated_cupy() -> Any:
    """Return the cached, device-validated CuPy module, importing it lazily.

    Caches only the module: the ``import cupy`` and the ``getDeviceCount`` probe
    are the slow parts and their result does not change within a process.  Free
    memory is intentionally *not* queried here — the caller re-reads it on every
    resolve so the row-chunk size stays current.
    """
    global _CUPY_MODULE
    if _CUPY_MODULE is not None:
        return _CUPY_MODULE

    try:
        import cupy  # noqa: PLC0415  (lazy: CuPy is an optional GPU dependency)
    except Exception as exc:  # ImportError, or a CuPy install with a broken CUDA
        raise MaxEntBackendError(
            "The CUDA MaxEnt backend requires CuPy and an NVIDIA GPU. Install it "
            'with pip install "asymmetry[gpu]" (a CUDA-13 wheel; use '
            "cupy-cuda12x on CUDA-12 systems) and ensure an NVIDIA driver is "
            "present."
        ) from exc

    try:
        if cupy.cuda.runtime.getDeviceCount() <= 0:
            raise MaxEntBackendError(
                "The CUDA MaxEnt backend found no CUDA-capable device. Ensure an "
                "NVIDIA GPU and driver are present."
            )
    except MaxEntBackendError:
        raise
    except Exception as exc:  # a runtime/driver mismatch surfacing at query time
        raise MaxEntBackendError(
            "The CUDA MaxEnt backend could not initialise the GPU. Check the "
            'NVIDIA driver and CuPy install (pip install "asymmetry[gpu]").'
        ) from exc

    _CUPY_MODULE = cupy
    return cupy


def _resolve_cuda_backend() -> ProjectionBackend:
    """Return a CUDA backend sized from the device's current free memory.

    The CuPy module is imported and validated once (cached in ``_CUPY_MODULE``);
    the free-memory query and chunk sizing happen on every call, so the returned
    backend reflects the live device state rather than a frozen first-call
    snapshot.
    """
    cupy = _validated_cupy()
    try:
        free_bytes, _total_bytes = cupy.cuda.Device().mem_info
    except Exception as exc:  # a runtime/driver mismatch surfacing at query time
        raise MaxEntBackendError(
            "The CUDA MaxEnt backend could not initialise the GPU. Check the "
            'NVIDIA driver and CuPy install (pip install "asymmetry[gpu]").'
        ) from exc

    return ProjectionBackend(
        name="cuda",
        xp=cupy,
        chunk_elements=_cuda_chunk_elements(free_bytes),
        asarray=lambda array: cupy.asarray(array, dtype=cupy.float64),
        to_numpy=cupy.asnumpy,
    )


def resolve_backend(name: str | None) -> ProjectionBackend:
    """Return the :class:`ProjectionBackend` for *name*.

    ``None`` and ``"numpy"`` return the NumPy backend (the default, bit-for-bit
    identical to the historical CPU path).  ``"cuda"`` imports CuPy lazily and
    raises :class:`MaxEntBackendError` if no usable GPU is available.  ``"auto"``
    prefers CUDA but silently falls back to NumPy when it is unavailable (it
    never raises).  Any other string raises :class:`MaxEntBackendError`.
    """
    if name is None:
        return _NUMPY_BACKEND
    key = str(name).strip().lower()
    if key in ("", "numpy"):
        return _NUMPY_BACKEND
    if key == "cuda":
        return _resolve_cuda_backend()
    if key == "auto":
        try:
            return _resolve_cuda_backend()
        except MaxEntBackendError:
            return _NUMPY_BACKEND
    expected = ", ".join(repr(choice) for choice in BACKEND_CHOICES)
    raise MaxEntBackendError(f"Unknown MaxEnt backend {name!r}; expected one of {expected}.")

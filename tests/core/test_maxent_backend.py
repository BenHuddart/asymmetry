"""Tests for the optional CuPy (CUDA) MaxEnt projection backend.

The CPU-always tests pin the backend-resolution contract and prove the default
numpy path is untouched.  The GPU tests are gated on a real CUDA device and skip
cleanly, cheaply, and silently where none is present (CI has no GPU).
"""

from __future__ import annotations

import sys
from dataclasses import replace

import numpy as np
import pytest

import asymmetry.core.maxent.backend as backend_module
from asymmetry.core.maxent import (
    MaxEntBackendError,
    MaxEntCancelledError,
    MaxEntConfig,
    build_maxent_input,
    initialize_state,
    maxent,
    opus,
    resolve_backend,
    run_cycles,
    tropus,
)
from asymmetry.core.maxent.backend import _cuda_chunk_elements
from asymmetry.core.maxent.engine import (
    _chunk_rows,
    _project_adjoint,
    _project_forward,
    _state_signature,
)
from tests.core.test_maxent import _synthetic_run


def _cuda_available() -> bool:
    """Return True only if CuPy imports and a CUDA device is present."""
    try:
        resolve_backend("cuda")
    except MaxEntBackendError:
        return False
    return True


requires_cuda = pytest.mark.skipif(not _cuda_available(), reason="No CUDA device / CuPy available")


# --------------------------------------------------------------------------- #
# CPU-always tests
# --------------------------------------------------------------------------- #


def test_numpy_backend_is_default_singleton() -> None:
    numpy_backend = resolve_backend("numpy")
    assert numpy_backend.name == "numpy"
    assert numpy_backend.xp is np
    assert numpy_backend.chunk_elements == 2_000_000
    # None resolves to the same shared numpy singleton.
    assert resolve_backend(None) is numpy_backend


def test_cuda_backend_raises_without_cupy(monkeypatch: pytest.MonkeyPatch) -> None:
    # Bypass any cached cupy module and make ``import cupy`` fail.
    monkeypatch.setattr(backend_module, "_CUPY_MODULE", None)
    monkeypatch.setitem(sys.modules, "cupy", None)
    with pytest.raises(MaxEntBackendError, match=r"asymmetry\[gpu\]"):
        resolve_backend("cuda")


def test_auto_backend_falls_back_to_numpy_without_cupy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend_module, "_CUPY_MODULE", None)
    monkeypatch.setitem(sys.modules, "cupy", None)
    resolved = resolve_backend("auto")
    assert resolved.name == "numpy"
    assert resolved is resolve_backend("numpy")


def test_unknown_backend_string_raises() -> None:
    with pytest.raises(MaxEntBackendError, match="Unknown MaxEnt backend"):
        resolve_backend("gpu")


def test_config_backend_roundtrip_and_coercion() -> None:
    # Default is numpy; unknown values coerce to numpy; legacy dicts lack the key.
    assert MaxEntConfig().backend == "numpy"
    assert MaxEntConfig.from_dict({"backend": "junk"}).backend == "numpy"
    assert MaxEntConfig.from_dict({}).backend == "numpy"
    assert MaxEntConfig.from_dict({"backend": "CUDA"}).backend == "cuda"
    # Roundtrip preserves an explicit cuda selection.
    config = MaxEntConfig(backend="cuda")
    assert config.to_dict()["backend"] == "cuda"
    assert MaxEntConfig.from_dict(config.to_dict()).backend == "cuda"


def test_backend_absent_from_state_signature() -> None:
    run = _synthetic_run()
    numpy_config = MaxEntConfig(
        n_spectrum_points=64, f_min_mhz=0.2, f_max_mhz=3.0, auto_window=False, backend="numpy"
    )
    cuda_config = MaxEntConfig(
        n_spectrum_points=64, f_min_mhz=0.2, f_max_mhz=3.0, auto_window=False, backend="cuda"
    )
    prepared = build_maxent_input(run, numpy_config)
    # The backend must not enter the resume signature: switching it must not
    # invalidate a resumed state (the projection math is backend-independent).
    assert _state_signature(prepared, numpy_config) == _state_signature(prepared, cuda_config)

    # And a run resuming under a different backend value must not trip the
    # "restart required" guard.  Use "auto", which resolves to numpy here (CPU),
    # so this exercises the resume path without needing a GPU.
    state = initialize_state(prepared, numpy_config)
    auto_config = MaxEntConfig(
        n_spectrum_points=64, f_min_mhz=0.2, f_max_mhz=3.0, auto_window=False, backend="auto"
    )
    result = run_cycles(prepared, auto_config, state=state, cycles=1)
    assert result.state.cycle == 1


def test_explicit_numpy_backend_matches_default_bit_for_bit() -> None:
    run = _synthetic_run(frequency_mhz=1.5)
    base_kwargs = dict(
        n_spectrum_points=128,
        f_min_mhz=0.2,
        f_max_mhz=3.0,
        auto_window=False,
        outer_cycles=4,
        inner_iterations=4,
        fit_phases=False,
    )
    default_result = maxent(run, MaxEntConfig(**base_kwargs))
    numpy_result = maxent(run, MaxEntConfig(**base_kwargs, backend="numpy"))
    assert np.array_equal(default_result.spectrum, numpy_result.spectrum)


def _small_chunk_numpy_backend(n_freq: int) -> object:
    """Return a numpy backend whose row chunk is a few rows wide.

    ``chunk_elements = 8 * n_freq // 3`` gives ``chunk // n_freq == 2`` rows per
    chunk, so a non-multiple row count is covered in many uneven chunks — the
    real multi-chunk path the default (2e6-element) backend never reaches at test
    scale.
    """
    return replace(resolve_backend("numpy"), chunk_elements=8 * n_freq // 3)


def test_project_forward_small_chunk_matches_default() -> None:
    # Forward writes disjoint output slices, so row chunking is transparent at
    # the algorithm level.  It is *not* bit-for-bit, however: each output row is
    # a BLAS ``matrix @ f`` dot whose reduction blocking depends on the matrix
    # height, so a 2-row chunk and the full 257-row matvec differ at the ULP
    # level (~1e-13 relative).  Pin that with a tight tolerance.
    rng = np.random.default_rng(11)
    n_time, n_freq = 257, 1024
    time = np.sort(rng.random(n_time)) * 8.0
    frequencies = np.linspace(0.1, 5.0, n_freq)
    spectrum = rng.random(n_freq)
    small = _small_chunk_numpy_backend(n_freq)
    assert _chunk_rows(n_time, n_freq, small.chunk_elements) < n_time  # many chunks

    default = _project_forward(time, frequencies, spectrum, phase_degrees=30.0)
    chunked = _project_forward(time, frequencies, spectrum, phase_degrees=30.0, backend=small)
    assert np.allclose(chunked, default, rtol=1e-12, atol=0.0)


def test_project_adjoint_small_chunk_matches_default() -> None:
    # Adjoint accumulates across chunks, so a small chunk changes only the
    # reduction order — the result matches the default to tight tolerance.
    rng = np.random.default_rng(11)
    n_time, n_freq = 257, 1024
    time = np.sort(rng.random(n_time)) * 8.0
    frequencies = np.linspace(0.1, 5.0, n_freq)
    values = rng.normal(size=n_time)
    small = _small_chunk_numpy_backend(n_freq)

    default = _project_adjoint(time, frequencies, values, phase_degrees=30.0)
    chunked = _project_adjoint(time, frequencies, values, phase_degrees=30.0, backend=small)
    assert np.allclose(chunked, default, rtol=1e-12, atol=0.0)


def test_cuda_chunk_elements_is_budgeted_monotonic_and_bounded() -> None:
    # Budget: the design matrix is never sized above 1/8 of free memory (the
    # ``8 * 8`` divisor), so it holds the ~2 concurrent chunk transients plus
    # vectors with margin instead of guaranteeing an OOM.
    budgeted = [1 << 20, 1 << 24, 1 << 28, 1 << 30, 1 << 33]  # 1 MB … 8 GB
    for free in budgeted:
        assert _cuda_chunk_elements(free) * 8 * 8 <= free
        assert _cuda_chunk_elements(free) >= 1

    # Monotonic non-decreasing in free memory.
    values = [_cuda_chunk_elements(free) for free in budgeted]
    assert values == sorted(values)

    # Huge free memory saturates at the 1<<28 (~2 GB) per-matrix cap.
    assert _cuda_chunk_elements(1 << 60) == 1 << 28

    # Degenerate/tiny free never drops below a single element.
    assert _cuda_chunk_elements(0) == 1
    assert _cuda_chunk_elements(1) == 1


# --------------------------------------------------------------------------- #
# GPU tests (skip cleanly with no CUDA device)
# --------------------------------------------------------------------------- #


@requires_cuda
@pytest.mark.parametrize("pulse_mode", ["ignore", "single"])
def test_cuda_opus_tropus_are_adjoint(pulse_mode: str) -> None:
    run = _synthetic_run()
    config = MaxEntConfig(
        n_spectrum_points=64,
        f_min_mhz=0.1,
        f_max_mhz=4.0,
        auto_window=False,
        pulse_mode=pulse_mode,
    )
    prepared = build_maxent_input(run, config)
    cuda = resolve_backend("cuda")
    rng = np.random.default_rng(5)
    spectrum = rng.random(prepared.n_spectrum_points)
    group_values = {
        group.group_id: rng.normal(size=group.time_us.size) for group in prepared.groups
    }

    forward = opus(spectrum, prepared, backend=cuda)
    lhs = sum(
        float(np.dot(forward[group.group_id], group_values[group.group_id]))
        for group in prepared.groups
    )
    rhs = float(np.dot(spectrum, tropus(group_values, prepared, backend=cuda)))
    assert lhs == pytest.approx(rhs, rel=1e-10, abs=1e-10)


@requires_cuda
def test_cuda_kernels_match_numpy() -> None:
    rng = np.random.default_rng(11)
    n_time, n_freq = 257, 1024
    time = np.sort(rng.random(n_time)) * 8.0
    frequencies = np.linspace(0.1, 5.0, n_freq)
    spectrum = rng.random(n_freq)
    values = rng.normal(size=n_time)
    # Force a few-row chunk so the GPU row loop runs many uneven chunks; a
    # full-memory cuda backend would size this small case as a single chunk and
    # never exercise the multi-chunk path.
    cuda = replace(resolve_backend("cuda"), chunk_elements=8 * n_freq // 3)
    assert _chunk_rows(n_time, n_freq, cuda.chunk_elements) < n_time

    fwd_np = _project_forward(time, frequencies, spectrum, phase_degrees=30.0)
    fwd_cuda = _project_forward(time, frequencies, spectrum, phase_degrees=30.0, backend=cuda)
    assert np.allclose(fwd_np, fwd_cuda, rtol=1e-9, atol=1e-12)

    adj_np = _project_adjoint(time, frequencies, values, phase_degrees=30.0)
    adj_cuda = _project_adjoint(time, frequencies, values, phase_degrees=30.0, backend=cuda)
    assert np.allclose(adj_np, adj_cuda, rtol=1e-9, atol=1e-12)


@requires_cuda
def test_cuda_end_to_end_parity() -> None:
    run = _synthetic_run(frequency_mhz=1.5)
    base_kwargs = dict(
        n_spectrum_points=128,
        f_min_mhz=0.2,
        f_max_mhz=3.0,
        auto_window=False,
        outer_cycles=4,
        inner_iterations=4,
        fit_phases=False,
    )
    numpy_result = maxent(run, MaxEntConfig(**base_kwargs, backend="numpy"))
    cuda_result = maxent(run, MaxEntConfig(**base_kwargs, backend="cuda"))

    numpy_chi2 = numpy_result.diagnostics.chi2[-1]
    cuda_chi2 = cuda_result.diagnostics.chi2[-1]
    assert cuda_chi2 == pytest.approx(numpy_chi2, rel=1e-6)
    tol = 1e-8 * float(np.max(numpy_result.spectrum))
    assert np.allclose(cuda_result.spectrum, numpy_result.spectrum, rtol=1e-5, atol=tol)


@requires_cuda
def test_cuda_cancellation_raises() -> None:
    run = _synthetic_run()
    config = MaxEntConfig(
        n_spectrum_points=64,
        f_min_mhz=0.2,
        f_max_mhz=3.0,
        auto_window=False,
        outer_cycles=4,
        inner_iterations=4,
        backend="cuda",
    )
    with pytest.raises(MaxEntCancelledError):
        maxent(run, config, cancel_callback=lambda: True)

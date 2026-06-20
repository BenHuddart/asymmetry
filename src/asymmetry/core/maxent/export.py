"""On-demand text export of a MaxEnt spectrum and run log.

Modern, human-readable text (not WiMDA's binary ``.max``), produced only when
the user asks — never auto-saved every cycle.
"""

from __future__ import annotations

import numpy as np

from asymmetry.core.fourier.units import mhz_to_gauss
from asymmetry.core.maxent.engine import MaxEntConfig, MaxEntResult


def _fmt(value: object, spec: str) -> str:
    """Format a numeric value with ``spec``; fall back to ``?`` when missing."""
    try:
        return format(float(value), spec)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "?"


def _header_lines(result: MaxEntResult, config: MaxEntConfig) -> list[str]:
    """Return the shared, human-readable provenance header.

    Records enough of the reconstruction recipe (field, calibration, frequency
    and time windows, entropy/cycle settings, deadtime) that the spectrum can be
    reproduced from the file alone.  Every line is a ``#`` comment so downstream
    parsers can split data from metadata on the prefix.
    """
    meta = result.metadata
    freqs = np.asarray(result.frequencies_mhz, dtype=float)

    run_label = str(meta.get("run_label") or "").strip()
    field = meta.get("field")
    alpha = meta.get("zf_lf_alpha")

    lines = [
        f"# MaxEnt spectrum — run {meta.get('run_number', '?')}",
    ]
    if run_label:
        lines.append(f"# run label: {run_label}")
    lines.extend(
        [
            f"# mode: {config.mode}",
            f"# pulse mode: {config.pulse_mode}",
        ]
    )
    if field is not None:
        lines.append(f"# field: {_fmt(field, '.4g')} G")
    if alpha is not None:
        lines.append(f"# calibration alpha: {_fmt(alpha, '.6g')}")

    # Effective frequency window: the actual reconstructed grid, with the
    # equivalent field range and whether the window was chosen automatically.
    if freqs.size:
        f_lo, f_hi = float(freqs.min()), float(freqs.max())
        b_lo, b_hi = float(mhz_to_gauss(f_lo)), float(mhz_to_gauss(f_hi))
        window_kind = "auto" if config.auto_window else "manual"
        lines.append(
            f"# frequency window: {f_lo:.6g}–{f_hi:.6g} MHz "
            f"({b_lo:.4g}–{b_hi:.4g} G) [{window_kind}]"
        )

    if config.t_min_us is not None or config.t_max_us is not None:
        t_lo = "?" if config.t_min_us is None else _fmt(config.t_min_us, ".6g")
        t_hi = "?" if config.t_max_us is None else _fmt(config.t_max_us, ".6g")
        lines.append(f"# time window: {t_lo}–{t_hi} us")
    lines.append(f"# time binning factor: {int(config.time_binning_factor)}")
    lines.append(f"# deadtime correction: {'on' if config.use_deadtime_correction else 'off'}")
    lines.append(f"# default level: {_fmt(config.default_level, '.6g')}")
    lines.append(f"# entropy target chi2/N: {_fmt(config.chi2_target_over_n, '.6g')}")
    lines.append(
        f"# cycles: outer={int(config.outer_cycles)}, "
        f"inner={int(config.inner_iterations)}, run={int(result.state.cycle)}"
    )
    lines.append(f"# chi2: {_fmt(meta.get('maxent_chi2'), '.6g')}")
    lines.append(f"# stop reason: {result.stop_reason}")
    lines.append(f"# spectrum points: {freqs.size}")
    return lines


def spectrum_to_text(result: MaxEntResult, config: MaxEntConfig) -> str:
    """Return the spectrum as a two-column text block with a parameter header.

    Columns: frequency (MHz), field (G), spectral density.
    """
    freqs = np.asarray(result.frequencies_mhz, dtype=float)
    spectrum = np.asarray(result.spectrum, dtype=float)
    field_gauss = mhz_to_gauss(freqs) if freqs.size else freqs
    lines = _header_lines(result, config)
    lines.append("# frequency_MHz\tfield_G\tdensity")
    for f_mhz, b_gauss, value in zip(freqs, field_gauss, spectrum):
        lines.append(f"{f_mhz:.8g}\t{b_gauss:.8g}\t{value:.8g}")
    return "\n".join(lines) + "\n"


def run_log_text(result: MaxEntResult, config: MaxEntConfig) -> str:
    """Return a per-cycle convergence log plus the final group nuisance values."""
    diagnostics = result.diagnostics
    lines = _header_lines(result, config)
    lines.append("")
    lines.append("# per-cycle convergence")
    lines.append("# cycle\tchi2\tentropy\ttest")
    for cycle, chi2, entropy, test in zip(
        diagnostics.cycles, diagnostics.chi2, diagnostics.entropy, diagnostics.test
    ):
        lines.append(f"{int(cycle)}\t{chi2:.6g}\t{entropy:.6g}\t{test:.6g}")

    if diagnostics.phases:
        lines.append("")
        lines.append("# final group phases (deg) / amplitudes / backgrounds")
        final_phases = diagnostics.phases[-1]
        final_amps = diagnostics.amplitudes[-1] if diagnostics.amplitudes else {}
        final_bgs = diagnostics.backgrounds[-1] if diagnostics.backgrounds else {}
        for group_id in sorted(final_phases):
            lines.append(
                f"group {group_id}: "
                f"phase={final_phases.get(group_id, 0.0):.3f} "
                f"amplitude={final_amps.get(group_id, 0.0):.4f} "
                f"background={final_bgs.get(group_id, 0.0):.5f}"
            )
    return "\n".join(lines) + "\n"

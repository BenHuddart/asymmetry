"""On-demand text export of a MaxEnt spectrum and run log.

Modern, human-readable text (not WiMDA's binary ``.max``), produced only when
the user asks — never auto-saved every cycle.
"""

from __future__ import annotations

import numpy as np

from asymmetry.core.fourier.units import mhz_to_gauss
from asymmetry.core.maxent.engine import MaxEntConfig, MaxEntResult


def _header_lines(result: MaxEntResult, config: MaxEntConfig) -> list[str]:
    meta = result.metadata
    lines = [
        f"# MaxEnt spectrum — run {meta.get('run_number', '?')}",
        f"# cycles: {int(result.state.cycle)}",
        f"# stop reason: {result.stop_reason}",
        f"# chi2: {meta.get('maxent_chi2')}",
        f"# spectrum points: {result.frequencies_mhz.size}",
        f"# pulse mode: {config.pulse_mode}",
        f"# mode: {config.mode}",
    ]
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

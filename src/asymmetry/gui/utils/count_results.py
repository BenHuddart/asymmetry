"""Count-fit parameter and retained-warning presentation shared by fit modes."""

import html

from asymmetry.gui.styles.widgets import warning_html
from asymmetry.gui.utils.formatting import format_param_label


def count_result_rows(result, format_value):
    """Show fitted values and explain unavailable optional calibration promotion."""
    rows = []
    for name in result.parameters.names:
        value = result.parameters[name].value
        error = result.uncertainties.get(name)
        row = f"{format_param_label(name)} = {format_value(value, error)}"
        minos = (result.minos_errors or {}).get(name)
        if minos is not None and len(minos) == 2:
            lower, upper = float(minos[0]), float(minos[1])
            row += f" (+{upper:.2g} / {lower:.2g})"
        rows.append(row)
    rows.extend(warning_html(html.escape(message)) for message in result.warnings)
    return rows

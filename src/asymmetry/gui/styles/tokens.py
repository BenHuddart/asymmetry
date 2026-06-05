"""BENCH design-token palette for the Asymmetry GUI."""

# Backgrounds
BG = "#fafaf9"
SURFACE = "#ffffff"
SURFACE_ALT = "#f4f3f0"
SURFACE_HI = "#ececea"

# Borders
BORDER = "#dedcd6"
BORDER_STRONG = "#c2c0b9"

# Text
TEXT = "#1c1d20"
TEXT_MUTED = "#67676b"
TEXT_DIM = "#9a9a9e"

# Accent — blue (primary UI chrome)
ACCENT = "#1f4d8a"
ACCENT_SOFT = "#e8eef7"
ACCENT_SOFT2 = "#dfe8f4"

# Accent — red (FitSeries identity: browser highlight tint, series buttons)
ACCENT_RED = "#a8332a"
ACCENT_RED_SOFT = "#f5dcd8"
ACCENT_RED_SOFT2 = "#efcfca"

# Data browser groups — match _GROUP_HEADER_BACKGROUND / _GROUP_MEMBER_BACKGROUND
GROUP_HEADER_BG = "#c8d2e1"
GROUP_MEMBER_BG = "#ebeff7"

# Semantic
WARN = "#b66815"
OK = "#2a7a3f"
FIT = "#c34a2c"
ERROR = "#b3261e"

# Success state (converged result groups)
SUCCESS_BG = "#f4f8f4"
SUCCESS_BORDER = "#cbe1cf"

# Matplotlib plot chrome — used by styles/plots.py
PLOT_AXIS = "#3a3c40"
PLOT_TICK_LABEL = "#56585b"
PLOT_GRID = (0, 0, 0, 0.06)  # (r, g, b, alpha) tuple for mpl
PLOT_ZERO_LINE = "#b0b3b7"
PLOT_LEGEND_BG = (1, 1, 1, 0.95)  # (r, g, b, alpha) for legend frame fill
PLOT_DATA = ACCENT  # default single-run data colour
PLOT_FIT = FIT  # default single-run fit-line colour
PLOT_FIT_RANGE_FACE = ACCENT  # axvspan fill colour (low alpha applied at draw time)
PLOT_FIT_RANGE_EDGE = ACCENT  # axvline edge colour (medium alpha applied at draw time)
PLOT_LOW_COUNT = "0.6"  # matplotlib grey shorthand for low-count data points

# Log tag colours (by category) — used in log_panel.py
LOG_TAG_ACCENT = ACCENT
LOG_TAG_OK = OK
LOG_TAG_WARN = WARN

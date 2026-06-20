"""BENCH design-token palette for the Asymmetry GUI.

This module is the single source of truth for every colour the GUI paints.
Tokens are grouped into semantic roles (backgrounds, borders, text, accents,
semantic states, data-trace palette, plot chrome) so a future dark theme can be
introduced by swapping the values in one place — every consumer (Python panels,
``styles/palette.py``, the templated ``bench.qss``, and ``styles/plots.py``)
reads from here rather than inlining literals.
"""

# ── Backgrounds ───────────────────────────────────────────────────────────────
BG = "#fafaf9"
SURFACE = "#ffffff"
SURFACE_ALT = "#f4f3f0"
SURFACE_HI = "#ececea"
WHITE = "#ffffff"  # pure white for bevels/bright-text roles

# ── Borders ───────────────────────────────────────────────────────────────────
BORDER = "#dedcd6"
BORDER_STRONG = "#c2c0b9"

# ── Text ──────────────────────────────────────────────────────────────────────
TEXT = "#1c1d20"
TEXT_MUTED = "#67676b"
TEXT_DIM = "#9a9a9e"

# ── Accent — blue (primary UI chrome) ─────────────────────────────────────────
ACCENT = "#1f4d8a"
ACCENT_SOFT = "#e8eef7"
ACCENT_SOFT2 = "#dfe8f4"

# ── Accent — red (FitSeries identity: browser highlight tint, series buttons) ─
ACCENT_RED = "#a8332a"
ACCENT_RED_SOFT = "#f5dcd8"
ACCENT_RED_SOFT2 = "#efcfca"

# ── Data browser groups ───────────────────────────────────────────────────────
GROUP_HEADER_BG = "#c8d2e1"
GROUP_HEADER_SEL_BG = "#a8b8d0"
GROUP_HEADER_FOCUS_BG = "#8fa3c2"
GROUP_MEMBER_BG = "#ebeff7"

# ── Semantic states ───────────────────────────────────────────────────────────
WARN = "#b66815"
OK = "#2a7a3f"
FIT = "#c34a2c"
ERROR = "#b3261e"

# Logged/real-time value foreground (data browser temperature & field columns):
# a distinct alarm red signalling a *measured* reading rather than a setpoint.
LOGGED_VALUE_FG = "#b02424"

# Success state (converged result groups)
SUCCESS_BG = "#f4f8f4"
SUCCESS_BORDER = "#cbe1cf"

# ── Data-trace palette ────────────────────────────────────────────────────────
# Okabe-Ito colour-blind-safe qualitative set, used for multi-run overlays and
# the two-period (RG) overlay traces in the plot panel. The ordered per-mode
# palettes that consume these live in ``styles/plots.py``.
TRACE_BLUE = "#0072b2"
TRACE_SKY = "#56b4e9"
TRACE_GREEN = "#009e73"
TRACE_YELLOW = "#f0e442"
TRACE_MAGENTA = "#cc79a7"
TRACE_BLACK = "#000000"
TRACE_ORANGE = "#e69f00"
TRACE_VERMILLION = "#d55e00"

# Two-period (RG) mode base colours — WiMDA convention (the selected period
# mode's primary trace colour).
PERIOD_RED = "#c00000"
PERIOD_GREEN = "#008000"
PERIOD_DIFF = "#0000c0"  # GREEN_MINUS_RED
PERIOD_SUM = "#800080"  # GREEN_PLUS_RED

# ── Matplotlib plot chrome — used by styles/plots.py ──────────────────────────
PLOT_AXIS = "#3a3c40"
PLOT_TICK_MARK = "#8a8d92"
PLOT_TICK_LABEL = "#56585b"
PLOT_GRID = (0, 0, 0, 0.06)  # (r, g, b, alpha) tuple for mpl
PLOT_ZERO_LINE = "#b0b3b7"
PLOT_LEGEND_BG = (1, 1, 1, 0.95)  # (r, g, b, alpha) for legend frame fill
PLOT_DATA = ACCENT  # default single-run data colour
PLOT_FIT = FIT  # default single-run fit-line colour
PLOT_FIT_PREVIEW = "#d73a49"  # transient preview-curve colour (distinct from FIT)
PLOT_FIT_RANGE_FACE = ACCENT  # axvspan fill colour (low alpha applied at draw time)
PLOT_FIT_RANGE_EDGE = ACCENT  # axvline edge colour (medium alpha applied at draw time)
PLOT_LOW_COUNT = "0.6"  # matplotlib grey shorthand for low-count data points

# ── Log tag colours (by category) — used in log_panel.py ──────────────────────
LOG_TAG_ACCENT = ACCENT
LOG_TAG_OK = OK
LOG_TAG_WARN = WARN

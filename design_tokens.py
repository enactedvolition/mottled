"""The one source of truth for Mottled's design language ("Incision").

Every surface — the Plotly renderer (`ui.py`), the Streamlit shell
(`.streamlit/config.toml`) and the WebGL viewer (`viewer/style.css`) — must
share these values. Python owns them here; the two static files mirror them,
and `tests/test_tokens.py` fails if any of them drifts. So retheming is a
one-line edit guarded by CI, not a hunt across three files that silently
desynchronise.

Dark navy void, a single precision-blue accent, semantic data colours, 1px
borders, near-sharp corners, monospace for data values.
"""
from __future__ import annotations

# ---- core palette (mirrored by config.toml + viewer/style.css :root) --------
BASE = "#080B18"           # the void / background
SURFACE_0 = "#0C1020"      # panels, secondary background
SURFACE_1 = "#11162A"      # raised surfaces
BORDER = "#1E2540"
BORDER_STRONG = "#283050"
FG_1 = "#EDF0FA"           # primary text
FG_2 = "#818FB8"           # muted text
ACCENT = "#4B7CF3"         # the single precision-blue accent

# ---- semantic data colours --------------------------------------------------
TEAL = "#00CCA8"           # live / secondary series
AMBER = "#D4934A"          # risk / low-fidelity flag
RED = "#E05050"            # loss
GREEN = "#38B07A"

# Per-trajectory marble cycle (accent first, then the data colours + tints).
MARBLE_COLORS = ["#4B7CF3", "#00CCA8", "#D4934A", "#E05050", "#38B07A",
                 "#8FA7F7", "#5CE0C6", "#E6B884", "#F08A8A", "#7FD0AC"]

# Terrain potential ramp: void -> surfaces -> precision blue -> light blue.
TERRAIN_COLORSCALE = [[0.0, "#04060E"], [0.22, "#0C1020"], [0.45, "#1C2A55"],
                      [0.68, "#2F55B8"], [0.86, "#4B7CF3"], [1.0, "#C8D4FB"]]

# ---- type -------------------------------------------------------------------
FONT_SANS = "DM Sans, -apple-system, Segoe UI, Helvetica, Arial, sans-serif"
FONT_MONO = "JetBrains Mono, SF Mono, Menlo, Consolas, monospace"

# The subset the mirrors (config.toml, viewer/style.css) must agree on, so the
# guard test knows exactly what to cross-check.
SHARED = {"base": BASE, "surface_0": SURFACE_0, "accent": ACCENT, "fg_1": FG_1}

# The viewer's CSS custom properties (viewer/style.css :root), in order. That
# block, viewer/tokens.js and .streamlit/config.toml are *generated* from this
# module by `python -m codegen`; tests/test_codegen.py fails if any drifts.
CSS_ROOT = {
    "color-base": BASE,
    "color-surface-0": SURFACE_0,
    "color-surface-1": SURFACE_1,
    "color-panel": "rgba(12, 16, 32, 0.92)",
    "color-border": BORDER,
    "color-border-strong": BORDER_STRONG,
    "color-fg-1": FG_1,
    "color-fg-2": FG_2,
    "color-accent": ACCENT,
    "color-accent-subtle": "rgba(75, 124, 243, 0.08)",
    "color-loss": RED,
    "color-amber": AMBER,
    "color-live": TEAL,
    "radius-xs": "2px",
    "radius-sm": "3px",
    "ease-precision": "cubic-bezier(0.16, 1, 0.3, 1)",
}

# .streamlit/config.toml [theme]
STREAMLIT_THEME = {
    "base": "dark",
    "primaryColor": ACCENT,
    "backgroundColor": BASE,
    "secondaryBackgroundColor": SURFACE_0,
    "textColor": FG_1,
}

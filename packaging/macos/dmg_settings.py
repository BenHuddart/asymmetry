from __future__ import annotations

import os
from pathlib import Path

application = Path(os.environ["ASYMMETRY_APP_BUNDLE"])
background = Path(os.environ["ASYMMETRY_DMG_BACKGROUND"])

files = [str(application)]
symlinks = {"Applications": "/Applications"}

badge_icon = None
icon_size = 128
text_size = 14

format = "UDZO"
default_view = "icon-view"
background = str(background)

window_rect = ((120, 120), (640, 360))
show_status_bar = False
show_tab_view = False
show_toolbar = False
show_pathbar = False
show_sidebar = False

arrange_by = None
grid_offset = (0, 0)
grid_spacing = 100
label_pos = "bottom"

icon_locations = {
    application.name: (128, 176),
    "Applications": (512, 176),
}

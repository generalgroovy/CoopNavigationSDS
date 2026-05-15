"""View-layer configuration."""

GUI_WIDTH = 1220
GUI_HEIGHT = 720
GUI_DIALOG_MIN_WIDTH = 390
GUI_MAP_MIN_WIDTH = 680
GUI_MIN_WIDTH = 980
GUI_MIN_HEIGHT = 600
GUI_EQUAL_PANE_MIN_WIDTH = 560
GUI_REFRESH_MS = 100

GUI_FONT_FAMILY = "Segoe UI"
GUI_MONO_FONT_FAMILY = "Consolas"
GUI_FONT_SMALL = 10
GUI_FONT_NORMAL = 11
GUI_FONT_SECTION = 13
GUI_TABLE_FONT_SIZE = 10
GUI_TABLE_HEADER_FONT_SIZE = 10
GUI_TABLE_ROW_HEIGHT = 20
GUI_SECTION_CORNER_RADIUS = 8
GUI_SECTION_BORDER_WIDTH = 1
GUI_MAIN_PAD = 2
GUI_SECTION_PAD_Y = 2
GUI_SECTION_HEADER_PAD_X = 4
GUI_SECTION_HEADER_PAD_Y = (2, 0)
GUI_SECTION_CONTENT_PAD_Y = (0, 1)
GUI_TEXTBOX_HEIGHT = 8
GUI_STATION_TABS_HEIGHT = 96
GUI_DEFAULT_TABS_HEIGHT = 96

GUI_COLORS = {
    "app_bg": "#edf2f7",
    "panel_bg": "#ffffff",
    "table_bg": "#ffffff",
    "table_heading_bg": "#edf2f7",
    "table_selected": "#dbeafe",
    "table_border": "#d1d5db",
    "text": "#111827",
    "muted_text": "#4b5563",
    "subtle_text": "#6b7280",
    "agent_a": "#174ea6",
    "agent_b": "#137333",
    "warning": "#b3261e",
    "map_bg": "#fbfaf7",
    "map_border": "#d0d5dd",
    "map_outline": "#202124",
    "start_station": "#cfe8ff",
    "destination_station": "#ffe2a8",
    "route_station": "#ffd7d2",
    "route_line": "#ff1f1f",
    "route_label_bg": "#fff1f0",
    "tab_bg": "#f9fafb",
    "tab_button_bg": "#e5e7eb",
    "tab_selected": "#2563eb",
    "tab_selected_hover": "#1d4ed8",
    "tab_unselected_hover": "#d1d5db",
}

GUI_ROUTE_TABLE_COLUMNS = [
    ("from", "From", 42, "center", False),
    ("to", "To", 42, "center", False),
    ("line", "Line", 50, "center", False),
    ("fullness", "Full", 46, "center", False),
    ("depart", "Leave", 58, "center", False),
    ("arrive", "Arrive", 58, "center", False),
    ("ride", "Ride", 46, "center", False),
    ("wait", "Wait", 46, "center", False),
    ("transfer", "Change", 58, "center", False),
]
GUI_LINE_TABLE_COLUMNS = [
    ("order", "#", 34, "center", False),
    ("station", "Stop", 48, "center", False),
    ("fullness", "Full", 46, "center", False),
    ("previous", "From", 48, "center", False),
    ("next", "Toward", 62, "center", False),
    ("ride", "Ride", 82, "center", False),
    ("elapsed", "From start", 98, "w", False),
]
GUI_STATION_LINE_TABLE_COLUMNS = [
    ("line", "Line", 118, "w", False),
    ("fullness", "Full", 46, "center", False),
    ("route", "Route", 96, "w", False),
    ("stop", "Stop", 54, "center", False),
    ("neighbors", "Neighbors", 120, "w", False),
    ("travel", "Ride times", 140, "w", True),
]
GUI_STATION_TIME_TABLE_COLUMNS = [
    ("line", "Line", 118, "w", False),
    ("movement", "Movement", 140, "w", False),
    ("times", "Arrival = departure", 250, "w", True),
]
GUI_ROUTE_TABLE_HEIGHT = 3
GUI_LINE_TABLE_HEIGHT = 5
GUI_STATION_TABLE_HEIGHT = 3

MAP_MIN_WIDTH = 420
MAP_MIN_HEIGHT = 300
MAP_PADDING_X = 34
MAP_PADDING_Y = 28
MAP_MIN_SCALE = 0.3
MAP_PARALLEL_LINE_SPACING = 15
MAP_PARALLEL_LINE_MIN_SPACING = 12
MAP_LINE_CASING_WIDTH = 11
MAP_LINE_CASING_MIN_WIDTH = 8
MAP_LINE_WIDTH = 6
MAP_LINE_MIN_WIDTH = 4
MAP_ROUTE_LINE_WIDTH = 13
MAP_ROUTE_LINE_MIN_WIDTH = 11
MAP_START_DEST_RADIUS = 10
MAP_ROUTE_RADIUS = 9
MAP_STATION_RADIUS = 7
MAP_MIN_STATION_RADIUS = 5
MAP_STATION_FONT_SCALE = 9
MAP_STATION_MIN_FONT = 7
MAP_ROLE_FONT_SCALE = 7
MAP_ROLE_MIN_FONT = 6
MAP_EDGE_LABEL_MIN_SCALE = 0.65
MAP_EDGE_LABEL_OFFSET = 11
MAP_EDGE_LABEL_MIN_OFFSET = 9
MAP_EDGE_LABEL_FONT_SCALE = 7
MAP_ROUTE_LABEL_FONT_SCALE = 8
MAP_EDGE_LABEL_MIN_FONT = 6
MAP_LABEL_PAD = 2
MAP_LEGEND_RIGHT_OFFSET = 88
MAP_LEGEND_TOP = 18
MAP_LEGEND_ROW_GAP = 20

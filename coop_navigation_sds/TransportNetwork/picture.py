"""Dependency-free network graph picture export."""
from __future__ import annotations

from collections import defaultdict
from html import escape
import math
from pathlib import Path

from coop_navigation_sds.TransportNetwork.network import LINES, STATION_POS, line_stop_pairs


PLOT_PADDING = 72
TITLE_HEIGHT = 88
INDEX_GAP = 28
INDEX_WIDTH = 390
INDEX_ROW_HEIGHT = 32
CONNECTION_SPACING = 9
PUBLIC_STROKE_WIDTH = 4
WALKING_STROKE_WIDTH = 3
LINE_DASH_PATTERNS = ("", "12 5", "4 4", "14 4 3 4")


def _connection_records():
    """Return every line-specific connection with a non-overlapping lane."""
    grouped = defaultdict(list)
    for line_name, data in sorted(LINES.items()):
        for station_a, station_b in line_stop_pairs(line_name, data):
            pair = tuple(sorted((station_a, station_b)))
            grouped[pair].append((line_name, data))

    records = []
    for (station_a, station_b), services in sorted(grouped.items()):
        services.sort(key=lambda item: item[0])
        midpoint = (len(services) - 1) / 2
        for index, (line_name, data) in enumerate(services):
            records.append(
                {
                    "line": line_name,
                    "data": data,
                    "from": station_a,
                    "to": station_b,
                    "offset": (index - midpoint) * CONNECTION_SPACING,
                }
            )
    return records


def _line_styles():
    """Assign a unique color-pattern combination to every service."""
    color_occurrences = defaultdict(int)
    styles = {}
    for line_name, data in sorted(LINES.items()):
        mode = data.get("mode", "unknown")
        if mode == "walking":
            styles[line_name] = {
                "color": "#64748b",
                "dash": "3 5",
                "width": WALKING_STROKE_WIDTH,
            }
            continue
        color = data.get("color", "#333333")
        occurrence = color_occurrences[color]
        color_occurrences[color] += 1
        styles[line_name] = {
            "color": color,
            "dash": LINE_DASH_PATTERNS[occurrence % len(LINE_DASH_PATTERNS)],
            "width": PUBLIC_STROKE_WIDTH,
        }
    return styles


def _line_style(line_name, styles):
    return styles[line_name]


def write_network_svg(path, *, title="CoopNavigationSDS transit network"):
    """Write every realized connection and an external line index to SVG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    xs = [point[0] for point in STATION_POS.values()]
    ys = [point[1] for point in STATION_POS.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    plot_width = (max_x - min_x) + (PLOT_PADDING * 2)
    plot_height = (max_y - min_y) + TITLE_HEIGHT + PLOT_PADDING
    index_height = TITLE_HEIGHT + PLOT_PADDING + len(LINES) * INDEX_ROW_HEIGHT
    width = plot_width + INDEX_GAP + INDEX_WIDTH
    height = max(plot_height, index_height)
    index_x = plot_width + INDEX_GAP
    connections = _connection_records()
    line_styles = _line_styles()

    def point(station):
        x, y = STATION_POS[station]
        return x - min_x + PLOT_PADDING, y - min_y + TITLE_HEIGHT

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">'
        ),
        '<rect width="100%" height="100%" fill="#f7fafc"/>',
        (
            f'<title>{escape(title)}</title><desc>{len(STATION_POS)} stations, '
            f'{len(LINES)} services, and {len(connections)} visible line-specific connections. '
            "Parallel services use separate lanes; walking connections are dashed.</desc>"
        ),
        f'<text x="{PLOT_PADDING}" y="30" font-family="Arial, sans-serif" font-size="18" font-weight="700" fill="#17202a">{escape(title)}</text>',
        (
            f'<text x="{PLOT_PADDING}" y="47" font-family="Arial, sans-serif" '
            f'font-size="11" fill="#52606d">{len(STATION_POS)} stations | '
            f'{len(LINES)} services | {len(connections)} line-specific connections</text>'
        ),
    ]

    for record in connections:
        line_name = record["line"]
        data = record["data"]
        station_a = record["from"]
        station_b = record["to"]
        x1, y1 = point(station_a)
        x2, y2 = point(station_b)
        length = math.hypot(x2 - x1, y2 - y1) or 1
        normal_x = -(y2 - y1) / length
        normal_y = (x2 - x1) / length
        offset = record["offset"]
        x1 += normal_x * offset
        y1 += normal_y * offset
        x2 += normal_x * offset
        y2 += normal_y * offset
        style = _line_style(line_name, line_styles)
        dash = (
            f' stroke-dasharray="{style["dash"]}"'
            if style["dash"]
            else ""
        )
        common = (
            f'x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            'stroke-linecap="round" fill="none"'
        )
        lines.append(
            f'<line class="connection-casing" {common} stroke="#ffffff" '
            f'stroke-width="{style["width"] + 3}"{dash}/>'
        )
        lines.append(
            f'<line class="network-connection" data-line="{escape(line_name)}" '
            f'data-mode="{escape(data.get("mode", "unknown"))}" '
            f'data-from="{escape(station_a)}" data-to="{escape(station_b)}" '
            f'{common} stroke="{escape(style["color"])}" '
            f'stroke-width="{style["width"]}"{dash}>'
            f'<title>{escape(line_name)}: {escape(station_a)} to {escape(station_b)}</title></line>'
        )

    for station, (raw_x, raw_y) in sorted(STATION_POS.items()):
        x, y = point(station)
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="#ffffff" stroke="#17202a" stroke-width="2"/>')
        lines.append(
            f'<text x="{x + 10:.1f}" y="{y - 9:.1f}" font-family="Arial, sans-serif" '
            f'font-size="11" font-weight="600" fill="#17202a" stroke="#ffffff" '
            f'stroke-width="3" paint-order="stroke">{escape(station)}</text>'
        )

    lines.extend(
        [
            (
                f'<rect class="line-index-panel" x="{index_x}" y="12" '
                f'width="{INDEX_WIDTH - 12}" '
                f'height="{height - 24}" rx="4" fill="#ffffff" stroke="#d7dee7"/>'
            ),
            (
                f'<text x="{index_x + 18}" y="36" font-family="Arial, sans-serif" '
                'font-size="16" font-weight="700" fill="#17202a">Line index</text>'
            ),
            (
                f'<text x="{index_x + 18}" y="53" font-family="Arial, sans-serif" '
                'font-size="10" fill="#52606d">Style | line | mode | headway | stops</text>'
            ),
        ]
    )

    legend_y = 76
    for index, (line_name, data) in enumerate(sorted(LINES.items())):
        y = legend_y + index * INDEX_ROW_HEIGHT
        style = _line_style(line_name, line_styles)
        dash = (
            f' stroke-dasharray="{style["dash"]}"'
            if style["dash"]
            else ""
        )
        lines.append(
            f'<line x1="{index_x + 18}" y1="{y}" x2="{index_x + 58}" y2="{y}" '
            f'stroke="{escape(style["color"])}" stroke-width="{style["width"]}" '
            f'stroke-linecap="round"{dash}/>'
        )
        lines.append(
            f'<text x="{index_x + 68}" y="{y + 4}" font-family="Arial, sans-serif" '
            f'font-size="11" font-weight="700" fill="#17202a">{escape(line_name)}</text>'
        )
        lines.append(
            f'<text x="{index_x + 130}" y="{y + 4}" font-family="Arial, sans-serif" '
            f'font-size="10" fill="#52606d">{escape(data.get("mode", "unknown"))} | '
            f'{data.get("headway", "?")} min | {len(data.get("stops", []))} stops</text>'
        )

    lines.append(
        f'<line x1="{plot_width + (INDEX_GAP / 2):.1f}" y1="12" '
        f'x2="{plot_width + (INDEX_GAP / 2):.1f}" y2="{height - 12}" '
        'stroke="#cbd5e1" stroke-width="1"/>'
    )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

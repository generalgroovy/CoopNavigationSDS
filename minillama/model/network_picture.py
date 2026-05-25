"""Dependency-free network graph picture export."""
from __future__ import annotations

from html import escape
from pathlib import Path

from minillama.model.metro_data import LINES, STATION_POS, capacity_status, line_stop_pairs


def write_network_svg(path, *, title="MiniLlama transit network"):
    """Write the current transit network as a standalone SVG picture file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    xs = [point[0] for point in STATION_POS.values()]
    ys = [point[1] for point in STATION_POS.values()]
    pad = 80
    min_x, max_x = min(xs) - pad, max(xs) + pad
    min_y, max_y = min(ys) - pad, max(ys) + pad
    width = max_x - min_x
    height = max_y - min_y

    def point(station):
        x, y = STATION_POS[station]
        return x - min_x, y - min_y

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">'
        ),
        '<rect width="100%" height="100%" fill="#f7fafc"/>',
        f'<text x="18" y="28" font-family="Arial, sans-serif" font-size="18" font-weight="700" fill="#17202a">{escape(title)}</text>',
    ]

    for line_name, data in sorted(LINES.items()):
        color = data.get("color", "#333333")
        for a, b in line_stop_pairs(line_name, data):
            x1, y1 = point(a)
            x2, y2 = point(b)
            lines.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="#ffffff" stroke-width="10" stroke-linecap="round"/>'
            )
            lines.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="{escape(color)}" stroke-width="5" stroke-linecap="round">'
                f'<title>{escape(line_name)}: {escape(a)} to {escape(b)}</title></line>'
            )

    for station, (raw_x, raw_y) in sorted(STATION_POS.items()):
        x, y = raw_x - min_x, raw_y - min_y
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="#ffffff" stroke="#17202a" stroke-width="2"/>')
        lines.append(
            f'<text x="{x + 10:.1f}" y="{y - 9:.1f}" font-family="Arial, sans-serif" '
            f'font-size="11" fill="#17202a">{escape(station)}</text>'
        )

    legend_y = height - 18 - (len(LINES) * 18)
    for index, (line_name, data) in enumerate(sorted(LINES.items())):
        y = legend_y + index * 18
        color = data.get("color", "#333333")
        lines.append(f'<line x1="18" y1="{y}" x2="44" y2="{y}" stroke="{escape(color)}" stroke-width="5"/>')
        lines.append(
            f'<text x="52" y="{y + 4}" font-family="Arial, sans-serif" font-size="11" fill="#17202a">'
            f'{escape(line_name)} ({data.get("headway", "?")} minutes, {capacity_status(data.get("fullness", 0))})</text>'
        )

    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

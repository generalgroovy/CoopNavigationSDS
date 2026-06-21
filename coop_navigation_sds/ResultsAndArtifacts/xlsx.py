"""Small XLSX writer for dependency-free research metric exports."""
from __future__ import annotations

from pathlib import Path
import re
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

from coop_navigation_sds.EvaluationMetrics.catalog import METRIC_FAMILY_SPECS, phase_key


IDENTIFIER_COLUMNS = [
    "condition_id",
    "test_case_key",
    "persona_key",
    "scenario_key",
    "speech_pattern_key",
    "agent_a_audio_persona",
    "agent_b_audio_persona",
    "model_name",
    "model_param_key",
]


def write_metrics_xlsx(metrics, path):
    """Write metric records to an XLSX workbook with summary and per-phase sheets."""
    records = list(metrics)
    if not records:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheets = [("summary", _summary_rows(records))]
    present = {phase for record in records for phase in record.metric_families}
    known_order = [phase_key(family) for family in METRIC_FAMILY_SPECS]
    phase_names = [phase for phase in known_order if phase in present]
    phase_names.extend(sorted(present - set(known_order)))
    for phase in phase_names:
        sheets.append((_sheet_name(phase), _phase_rows(records, phase)))

    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _content_types_xml(len(sheets)))
        archive.writestr("_rels/.rels", _root_rels_xml())
        archive.writestr("xl/workbook.xml", _workbook_xml([name for name, _ in sheets]))
        archive.writestr("xl/_rels/workbook.xml.rels", _workbook_rels_xml(len(sheets)))
        for index, (_name, rows) in enumerate(sheets, start=1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", _worksheet_xml(rows))


def _summary_rows(records):
    all_rows = [record.as_dict() for record in records]
    columns = list(all_rows[0])
    return [columns] + [[row.get(column) for column in columns] for row in all_rows]


def _phase_rows(records, phase):
    metric_names = sorted({metric for record in records for metric in record.metric_families.get(phase, {})})
    columns = IDENTIFIER_COLUMNS + metric_names
    rows = [columns]
    for record in records:
        families = record.metric_families.get(phase, {})
        base = {column: getattr(record, column, None) for column in IDENTIFIER_COLUMNS}
        rows.append([base.get(column, families.get(column)) if column in IDENTIFIER_COLUMNS else families.get(column) for column in columns])
    return rows


def _worksheet_xml(rows):
    xml_rows = []
    for row_index, row in enumerate(rows, start=1):
        cells = []
        for column_index, value in enumerate(row, start=1):
            cell_ref = f"{_excel_column(column_index)}{row_index}"
            cells.append(_cell_xml(cell_ref, value))
        xml_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(xml_rows)}</sheetData></worksheet>'
    )


def _cell_xml(cell_ref, value):
    if value is None:
        return f'<c r="{cell_ref}"/>'
    if isinstance(value, bool):
        return f'<c r="{cell_ref}" t="b"><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float)):
        return f'<c r="{cell_ref}"><v>{value}</v></c>'
    text = escape(str(value))
    return f'<c r="{cell_ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def _excel_column(index):
    out = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        out = chr(65 + remainder) + out
    return out


def _sheet_name(name):
    cleaned = re.sub(r"[][\\\\/*?:]+", "_", name)[:31]
    return cleaned or "sheet"


def _content_types_xml(sheet_count):
    sheet_overrides = "".join(
        f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for index in range(1, sheet_count + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        f"{sheet_overrides}</Types>"
    )


def _root_rels_xml():
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        "</Relationships>"
    )


def _workbook_xml(sheet_names):
    sheets = "".join(
        f'<sheet name="{escape(name)}" sheetId="{index}" r:id="rId{index}"/>'
        for index, name in enumerate(sheet_names, start=1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{sheets}</sheets></workbook>"
    )


def _workbook_rels_xml(sheet_count):
    rels = "".join(
        f'<Relationship Id="rId{index}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{index}.xml"/>'
        for index in range(1, sheet_count + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{rels}</Relationships>"
    )

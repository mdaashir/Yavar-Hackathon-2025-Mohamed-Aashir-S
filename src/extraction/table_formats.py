import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
from sklearn.cluster import DBSCAN


@dataclass
class TableFormat:
    name: str
    header_patterns: List[str]
    column_order: List[str]
    has_borders: bool
    multi_line_cells: bool
    expected_columns: int


@dataclass
class TableRegion:
    x: int
    y: int
    width: int
    height: int
    cells: List[Dict[str, Any]]
    format_type: str
    confidence: float


def _calculate_format_confidence(rows: List[List[Dict[str, Any]]],
                                 table_format: TableFormat) -> float:
    """
    Calculate confidence score for detected format
    """
    confidence_factors = []

    # Check number of columns
    num_cols = len(rows[0]) if rows else 0
    col_match = num_cols == table_format.expected_columns
    confidence_factors.append(1.0 if col_match else 0.5)

    # Check header patterns
    header_row = [cell['text'].strip().lower() for cell in rows[0]] if rows else []
    pattern_matches = sum(1 for pattern in table_format.header_patterns
                          if any(re.search(pattern, cell) for cell in header_row))
    pattern_score = pattern_matches / len(table_format.header_patterns)
    confidence_factors.append(pattern_score)

    # Check cell alignment
    if len(rows) > 1:
        alignment_scores = []
        for i in range(len(rows) - 1):
            curr_row = rows[i]
            next_row = rows[i + 1]
            if len(curr_row) == len(next_row):
                x_diffs = [abs(curr_row[j]['x'] - next_row[j]['x'])
                           for j in range(len(curr_row))]
                alignment_score = 1.0 - min(1.0, sum(x_diffs) / (100 * len(curr_row)))
                alignment_scores.append(alignment_score)

        if alignment_scores:
            confidence_factors.append(sum(alignment_scores) / len(alignment_scores))

    return sum(confidence_factors) / len(confidence_factors)


def _group_cells_into_rows(cells: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    Group cells into rows using DBSCAN clustering on y-coordinates
    """
    if not cells:
        return []

    # Extract y-coordinates
    y_coords = np.array([[cell['y']] for cell in cells])

    # Perform clustering
    clustering = DBSCAN(eps=10, min_samples=2).fit(y_coords)

    # Group cells by cluster
    rows = {}
    for idx, label in enumerate(clustering.labels_):
        if label == -1:  # Noise points
            continue
        if label not in rows:
            rows[label] = []
        rows[label].append(cells[idx])

    # Sort rows by y-coordinate and cells within rows by x-coordinate
    sorted_rows = []
    for label in sorted(rows.keys()):
        sorted_row = sorted(rows[label], key=lambda c: c['x'])
        sorted_rows.append(sorted_row)

    return sorted_rows


class TableFormatDetector:
    def __init__(self):
        self.known_formats = [
            TableFormat(
                name="standard_invoice",
                header_patterns=[
                    r'(?i)item|description',
                    r'(?i)qty|quantity',
                    r'(?i)rate|price|unit\s*price',
                    r'(?i)amount|total'
                ],
                column_order=['description', 'quantity', 'unit_price', 'total_amount'],
                has_borders=True,
                multi_line_cells=False,
                expected_columns=4
            ),
            TableFormat(
                name="detailed_invoice",
                header_patterns=[
                    r'(?i)sr\.?\s*no\.?',
                    r'(?i)item|description',
                    r'(?i)hsn/sac',
                    r'(?i)qty|quantity',
                    r'(?i)rate|price',
                    r'(?i)amount|total'
                ],
                column_order=['serial_number', 'description', 'hsn_sac', 'quantity', 'unit_price', 'total_amount'],
                has_borders=True,
                multi_line_cells=True,
                expected_columns=6
            ),
            TableFormat(
                name="simple_list",
                header_patterns=[
                    r'(?i)item',
                    r'(?i)qty',
                    r'(?i)total'
                ],
                column_order=['description', 'quantity', 'total_amount'],
                has_borders=False,
                multi_line_cells=False,
                expected_columns=3
            )
        ]

    def detect_format(self, header_row: List[str]) -> Optional[TableFormat]:
        """
        Detect table format based on header row
        """
        best_match = None
        max_matches = 0

        for fmt in self.known_formats:
            matches = sum(1 for pattern in fmt.header_patterns
                          if any(re.search(pattern, cell) for cell in header_row))

            if matches > max_matches:
                max_matches = matches
                best_match = fmt

        return best_match

    def analyze_table_structure(self, cells: List[Dict[str, Any]]) -> TableRegion | None:
        """
        Analyze table structure and determine its format
        """
        # Group cells into rows based on y-coordinates
        rows = _group_cells_into_rows(cells)

        if not rows:
            return None

        # Analyze header row
        header_row = [cell['text'].strip() for cell in rows[0]]
        table_format = self.detect_format(header_row)

        if not table_format:
            return None

        # Get table bounds
        x_min = min(cell['x'] for cell in cells)
        y_min = min(cell['y'] for cell in cells)
        x_max = max(cell['x'] + cell['width'] for cell in cells)
        y_max = max(cell['y'] + cell['height'] for cell in cells)

        return TableRegion(
            x=x_min,
            y=y_min,
            width=x_max - x_min,
            height=y_max - y_min,
            cells=cells,
            format_type=table_format.name,
            confidence=_calculate_format_confidence(rows, table_format)
        )

    def extract_structured_data(self, region: TableRegion) -> List[Dict[str, Any]]:
        """
        Extract structured data based on detected format
        """
        table_format = next((fmt for fmt in self.known_formats
                             if fmt.name == region.format_type), None)
        if not table_format:
            return []

        rows = _group_cells_into_rows(region.cells)
        if len(rows) < 2:  # Need at least header and one data row
            return []

        # Skip header row
        data_rows = rows[1:]
        structured_data = []

        for row in data_rows:
            row_data = {}

            # Map cells to columns based on x-position and table format
            header_cells = rows[0]
            for idx, cell in enumerate(row):
                if idx >= len(header_cells):
                    break

                # Find matching column in format
                header_text = header_cells[idx]['text'].strip().lower()
                for pattern, col_name in zip(table_format.header_patterns,
                                             table_format.column_order):
                    if re.search(pattern, header_text):
                        row_data[col_name] = cell['text'].strip()
                        break

            if row_data:
                structured_data.append(row_data)

        return structured_data

    def handle_multi_line_cells(self, region: TableRegion) -> TableRegion:
        """
        Handle tables with multi-line cells
        """
        if not region or not region.cells:
            return region

        table_format = next((fmt for fmt in self.known_formats
                             if fmt.name == region.format_type), None)
        if not table_format or not table_format.multi_line_cells:
            return region

        rows = _group_cells_into_rows(region.cells)
        merged_cells = []
        current_row = None

        for row in rows:
            if not current_row:
                current_row = row
                continue

            # Check if this row might be a continuation
            if len(row) < len(current_row):
                # Merge with cells above
                for cell in row:
                    # Find overlapping cell in current row
                    for curr_cell in current_row:
                        if (cell['x'] >= curr_cell['x'] and
                                cell['x'] + cell['width'] <= curr_cell['x'] + curr_cell['width']):
                            curr_cell['text'] += ' ' + cell['text']
                            break
            else:
                merged_cells.extend(current_row)
                current_row = row

        if current_row:
            merged_cells.extend(current_row)

        region.cells = merged_cells
        return region

import decimal
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import List, Dict, Tuple, Any

import cv2
import numpy as np
import pytesseract


@dataclass
class TableCell:
    x: int
    y: int
    width: int
    height: int
    text: str
    confidence: float


@dataclass
class TableValidation:
    is_valid: bool
    errors: List[str]
    warnings: List[str]


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Enhanced image preprocessing for better table detection
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter to remove noise while preserving edges
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Remove small noise
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    return cleaned


def detect_table_region(image: np.ndarray) -> Tuple[bool, np.ndarray, List[List[TableCell]]]:
    """
    Enhanced table region detection with cell structure
    """
    # Preprocess image
    processed = preprocess_image(image)

    # Detect horizontal and vertical lines with dynamic kernel sizes
    img_height, img_width = processed.shape
    min_line_length = min(img_height, img_width) // 20

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_line_length, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_line_length))

    horizontal_lines = cv2.morphologyEx(processed, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    vertical_lines = cv2.morphologyEx(processed, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    # Combine lines
    table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)

    # Find contours
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return False, image, []

    # Find the largest contour (main table)
    main_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(main_contour)

    # Add padding
    padding = 10
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image.shape[1], x + w + padding)
    y2 = min(image.shape[0], y + h + padding)

    # Crop regions
    table_region = image[y1:y2, x1:x2]
    mask_region = table_mask[y1:y2, x1:x2]

    # Detect cells using contour hierarchy
    cell_contours, hierarchy = cv2.findContours(mask_region, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Extract cells
    cells = []
    row_cells = []
    last_y = -1
    min_cell_area = (w * h) / 200  # Minimum cell area threshold

    for cnt in cell_contours:
        area = cv2.contourArea(cnt)
        if area < min_cell_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Extract cell content using OCR
        cell_roi = table_region[y:y + h, x:x + w]
        cell_text = pytesseract.image_to_string(cell_roi, config='--psm 7')
        cell_data = pytesseract.image_to_data(cell_roi, output_type=pytesseract.Output.DICT)

        # Calculate average confidence for the cell
        confidences = [conf for conf in cell_data['conf'] if conf != -1]
        avg_confidence = np.mean(confidences) if confidences else 0

        cell = TableCell(x=x, y=y, width=w, height=h,
                         text=cell_text.strip(), confidence=avg_confidence)

        # Group cells into rows
        if last_y == -1:
            last_y = y
        elif abs(y - last_y) > h / 2:  # New row
            cells.append(sorted(row_cells, key=lambda c: c.x))
            row_cells = []
            last_y = y

        row_cells.append(cell)

    if row_cells:
        cells.append(sorted(row_cells, key=lambda c: c.x))

    return True, table_region, cells


def validate_table_data(table_data: List[Dict[str, Any]]) -> TableValidation:
    """
    Validate extracted table data
    """
    errors = []
    warnings = []

    # Check for required columns
    required_columns = {'description', 'quantity', 'unit_price', 'total_amount'}

    for idx, row in enumerate(table_data, 1):
        # Check for empty required fields
        for col in required_columns:
            if not row.get(col):
                warnings.append(f"Row {idx}: Missing {col}")

        # Validate numeric values
        try:
            qty = Decimal(str(row.get('quantity', 0)))
            price = Decimal(str(row.get('unit_price', 0)))
            total = Decimal(str(row.get('total_amount', 0)))

            # Calculate and compare total
            calculated_total = qty * price
            if abs(calculated_total - total) > Decimal('0.01'):
                errors.append(
                    f"Row {idx}: Total amount mismatch. Expected {calculated_total}, got {total}"
                )
        except (ValueError, TypeError, decimal.InvalidOperation):
            errors.append(f"Row {idx}: Invalid numeric values")

        # Validate HSN/SAC code format (if present)
        hsn_sac = row.get('hsn_sac', '')
        if hsn_sac and not re.match(r'^\d{4,8}$', hsn_sac):
            warnings.append(f"Row {idx}: Invalid HSN/SAC code format")

        # Check confidence scores
        if row.get('confidence', 0) < 0.7:
            warnings.append(f"Row {idx}: Low confidence score")

    return TableValidation(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )


def _map_header_to_column(header_text: str) -> str | None:
    """Map detected header text to standard column names"""
    header_text = header_text.lower()

    if any(word in header_text for word in ['item', 'desc', 'product']):
        return 'description'
    elif any(word in header_text for word in ['hsn', 'sac', 'code']):
        return 'hsn_sac'
    elif any(word in header_text for word in ['qty', 'quantity']):
        return 'quantity'
    elif any(word in header_text for word in ['price', 'rate', 'unit']):
        return 'unit_price'
    elif any(word in header_text for word in ['amount', 'total', 'value']):
        return 'total_amount'
    elif any(word in header_text for word in ['sr', 'no', '#']):
        return 'serial_number'
    return None


def _is_numeric(text: str) -> bool:
    """Check if text represents a number"""
    try:
        float(text.replace(',', ''))
        return True
    except ValueError:
        return False


def _extract_table_data_from_cells(cells: List[List[TableCell]],
                                   header_structure: List[Dict[str, int]]) -> List[Dict[str, Any]]:
    """
    Extract structured data from detected cells using header information
    """
    table_data = []

    # Skip header row
    for row_cells in cells[1:]:
        row_data = {
            'description': '',
            'hsn_sac': '',
            'quantity': 0,
            'unit_price': 0.0,
            'total_amount': 0.0,
            'serial_number': '',
            'confidence': 0.0
        }

        confidences = []

        # Map cells to columns based on x-position
        for cell in row_cells:
            closest_col = min(header_structure,
                              key=lambda h: abs(h['x'] - cell.x))
            col_type = closest_col['type']

            confidences.append(cell.confidence)

            # Map cell content to appropriate column
            if col_type == 'description':
                row_data['description'] += f" {cell.text}"
            elif col_type == 'hsn_sac' and cell.text.isdigit():
                row_data['hsn_sac'] = cell.text
            elif col_type in ['quantity', 'unit_price', 'total_amount']:
                if _is_numeric(cell.text):
                    row_data[col_type] = float(cell.text.replace(',', ''))
            elif col_type == 'serial_number' and cell.text.isdigit():
                row_data['serial_number'] = cell.text

        row_data['description'] = row_data['description'].strip()
        row_data['confidence'] = np.mean(confidences) if confidences else 0.0

        if row_data['description'] or row_data['quantity'] or row_data['unit_price']:
            table_data.append(row_data)

    return table_data


class TableExtractor:
    def __init__(self, min_confidence: int = 60):
        self.min_confidence = min_confidence
        self.table_headers = [
            'description', 'hsn_sac', 'quantity', 'unit_price',
            'total_amount', 'serial_number'
        ]

    def process_multi_page_table(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Process tables spanning multiple pages
        """
        all_table_data = []
        header_structure = None

        for page_num, image in enumerate(images):
            # Detect table and cells
            table_found, table_region, cells = detect_table_region(image)
            if not table_found:
                continue

            # On first page, detect header structure
            if page_num == 0:
                header_structure = self.detect_table_structure(table_region)
                if not header_structure:
                    continue

            # Extract data using consistent header structure
            if header_structure:
                page_data = _extract_table_data_from_cells(cells, header_structure)
                all_table_data.extend(page_data)

        return all_table_data

    def extract_table_data(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Main method to extract table data with validation
        """
        # Detect table and cells
        table_found, table_region, cells = detect_table_region(image)
        if not table_found or not cells:
            return []

        # Detect table structure
        header_structure = self.detect_table_structure(table_region)
        if not header_structure:
            return []

        # Extract data from cells
        table_data = _extract_table_data_from_cells(cells, header_structure)

        # Validate extracted data
        validation = validate_table_data(table_data)
        if not validation.is_valid:
            print("Table validation errors:", validation.errors)
            print("Table validation warnings:", validation.warnings)

        return table_data

    def detect_table_structure(self, image: np.ndarray) -> List[Dict[str, int]]:
        """
        Detect table columns and their positions
        
        Args:
            image: Table region image
            
        Returns:
            List of column information dictionaries
        """
        # Get OCR data with bounding boxes
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

        # Find potential header row
        header_y = None
        header_words = []

        for i in range(len(data['text'])):
            text = data['text'][i].lower().strip()
            conf = int(data['conf'][i])

            if conf > self.min_confidence and text:
                # Look for common header terms
                if any(header in text for header in ['item', 'description', 'quantity', 'price', 'amount', 'hsn']):
                    header_words.append({
                        'text': text,
                        'x': data['left'][i],
                        'y': data['top'][i],
                        'width': data['width'][i]
                    })
                    if header_y is None:
                        header_y = data['top'][i]

        if not header_words:
            return []

        # Sort headers by x-coordinate
        header_words.sort(key=lambda x: x['x'])

        # Map detected headers to standard column names
        columns = []
        for header in header_words:
            col_type = _map_header_to_column(header['text'])
            if col_type:
                columns.append({
                    'type': col_type,
                    'x': header['x'],
                    'width': header['width']
                })

        return columns

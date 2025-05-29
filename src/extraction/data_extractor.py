import logging
import os
import re

import cv2
import numpy as np

from .ocr_engine import OCREngine
from .table_extractor import TableExtractor


def extract_invoice_number(text):
    """
    Extract invoice number using regex patterns
    """
    if not text:
        return None, 0.0

    invoice_patterns = [
        r'Invoice\s*#?\s*(\w+[-/]?\w+)',
        r'Invoice\s*Number\s*:?\s*(\w+[-/]?\w+)',
        r'Bill\s*Number\s*:?\s*(\w+[-/]?\w+)',
        r'Invoice\s*ID\s*:?\s*(\w+[-/]?\w+)',
        r'Bill\s*ID\s*:?\s*(\w+[-/]?\w+)'
    ]

    for pattern in invoice_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1), 0.9
    return None, 0.0


def extract_date(text):
    """
    Extract invoice date using various date formats
    """
    if not text:
        return None, 0.0

    date_patterns = [
        r'(?:Invoice|Bill|Date)\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
        r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})',
        r'(?:Date)\s*:?\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})',  # YYYY-MM-DD format
        r'(\d{1,2}[-/]\d{1,2}[-/]\d{4})',  # DD-MM-YYYY format
        r'(\d{4}[-/]\d{2}[-/]\d{2})',  # YYYY-MM-DD format without labels
        r'(\d{2}[-/]\d{2}[-/]\d{4})'  # DD/MM/YYYY format without labels
    ]

    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                date_str = match.group(1)
                # Add date parsing logic here
                return date_str, 0.9
            except:
                continue
    return None, 0.0


def _validate_gst_number(gst_number):
    """
    Validate GST number format
    """
    if not gst_number:
        return False

    # Basic GST format validation
    gst_pattern = r'^\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}$'
    return bool(re.match(gst_pattern, gst_number))


def extract_gst_numbers(text):
    """
    Extract GST numbers for supplier and bill-to party

    Returns:
        List of tuples [(gst_number, confidence), ...] or None if no GST numbers found
    """
    if not text:
        return None

    gst_patterns = [
        r'(?:GST|GSTIN)\s*:?\s*(\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1})',
        r'(?:Supplier\s+GST|Vendor\s+GST)\s*:?\s*(\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1})',
        r'(?:Customer\s+GST|Recipient\s+GST)\s*:?\s*(\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1})'
    ]

    matches = []
    try:
        for pattern in gst_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                gst_number = match.group(1)
                if _validate_gst_number(gst_number):
                    matches.append((gst_number, 0.85))

        return matches if matches else None
    except Exception as e:
        logging.error(f"Error extracting GST numbers: {str(e)}")
        return None


def extract_po_number(text):
    """
    Extract PO number
    """
    po_patterns = [
        r'P\.?O\.?\s*#?\s*(\w+[-/]?\w+)',
        r'Purchase\s*Order\s*:?\s*(\w+[-/]?\w+)',
        r'PO\s*Number\s*:?\s*(\w+[-/]?\w+)',
        r'Order\s*Number\s*:?\s*(\w+[-/]?\w+)',
        r'PO\s*ID\s*:?\s*(\w+[-/]?\w+)'
    ]

    for pattern in po_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1), 0.85
    return None, 0.0


def extract_shipping_address(text):
    """
    Extract shipping address
    """
    address_patterns = [
        r'Ship\s*To\s*:?\s*(.*?)(?=\n\n|\Z)',
        r'Delivery\s*Address\s*:?\s*(.*?)(?=\n\n|\Z)',
        r'Shipping\s*Address\s*:?\s*(.*?)(?=\n\n|\Z)',
        r'Deliver\s*To\s*:?\s*(.*?)(?=\n\n|\Z)',
        r'Ship\s*Address\s*:?\s*(.*?)(?=\n\n|\Z)'
    ]

    for pattern in address_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip(), 0.8
    return None, 0.0


def detect_seal_and_signature(image, output_dir=None, base_filename=None):
    """
    Detect and extract seal and signature from the image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area and shape
    potential_seals = []
    extracted_regions = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:  # Minimum area threshold
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h

            # Check if the region is nearly square (potential seal) or elongated (potential signature)
            if (0.8 <= aspect_ratio <= 1.2) or (aspect_ratio < 0.5):
                potential_seals.append((x, y, w, h))

                # Extract the region
                padding = 10
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(image.shape[1], x + w + padding)
                y2 = min(image.shape[0], y + h + padding)

                region = image[y1:y2, x1:x2]
                extracted_regions.append(region)

                # Save the region if output directory is provided
                if output_dir and base_filename:
                    region_type = 'seal' if 0.8 <= aspect_ratio <= 1.2 else 'signature'
                    region_filename = f"{base_filename}_{region_type}_{len(extracted_regions)}.png"
                    output_path = os.path.join(output_dir, region_filename)
                    cv2.imwrite(output_path, region)

    confidence = 0.75 if extracted_regions else 0.0
    return len(potential_seals) > 0, confidence, extracted_regions


class DataExtractor:
    def __init__(self):
        self.confidence_threshold = 0.6  # Minimum confidence score to accept
        self.table_extractor = TableExtractor(min_confidence=self.confidence_threshold)
        self.ocr_engine = OCREngine()
        self.validation_results = None  # Store validation results for reference

    def extract_text(self, image):
        """
        Extract text from image using EasyOCR
        """
        # Get OCR data including confidence scores
        text, confidence = self.ocr_engine.extract_text(image)
        return text, confidence

    def extract_structured_text(self, image):
        """
        Extract text with position information
        """
        return self.ocr_engine.extract_structured_text(image)

    def extract_table_data(self, image):
        """
        Extract table data from image
        """
        try:
            # First detect table region
            table_found, table_region, cells = detect_table_region(image)

            if not table_found or not cells:
                logging.warning("No table detected in image")
                return []

            # Process cells into structured table data
            table_data = _extract_table_data_from_cells(cells, [])

            # Validate table data
            self.validation_results = validate_table_data(table_data)

            return table_data

        except Exception as e:
            logging.error(f"Error extracting table data: {str(e)}")
            return []

    def process_multi_page_invoice(self, images, output_dir=None, base_filename=None):
        """
        Process a multi-page invoice
        """
        all_data = {
            'invoice_number': {'value': None, 'confidence': 0.0},
            'invoice_date': {'value': None, 'confidence': 0.0},
            'supplier_gst_number': {'value': None, 'confidence': 0.0},
            'bill_to_gst_number': {'value': None, 'confidence': 0.0},
            'po_number': {'value': None, 'confidence': 0.0},
            'shipping_address': {'value': None, 'confidence': 0.0},
            'seal_and_sign_present': {'value': False, 'confidence': 0.0},
            'table_data': [],
            'no_items': 0
        }

        try:
            # Process first page for header information
            if images:
                first_page = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
                text, text_confidence = self.extract_text(first_page)

                # Extract header information
                all_data['invoice_number']['value'], all_data['invoice_number'][
                    'confidence'] = extract_invoice_number(text)
                all_data['invoice_date']['value'], all_data['invoice_date']['confidence'] = extract_date(text)

                # Handle GST numbers with proper None checks
                gst_numbers = extract_gst_numbers(text)
                if gst_numbers:
                    if len(gst_numbers) >= 1:
                        all_data['supplier_gst_number']['value'] = gst_numbers[0][0]
                        all_data['supplier_gst_number']['confidence'] = gst_numbers[0][1]
                    if len(gst_numbers) >= 2:
                        all_data['bill_to_gst_number']['value'] = gst_numbers[1][0]
                        all_data['bill_to_gst_number']['confidence'] = gst_numbers[1][1]

                all_data['po_number']['value'], all_data['po_number']['confidence'] = extract_po_number(text)
                all_data['shipping_address']['value'], all_data['shipping_address'][
                    'confidence'] = extract_shipping_address(text)

                # Process seal and signature
                seal_present, seal_conf, extracted_regions = detect_seal_and_signature(
                    first_page, output_dir, base_filename)
                all_data['seal_and_sign_present'] = {
                    'value': seal_present,
                    'confidence': seal_conf,
                    'num_regions': len(extracted_regions)
                }

            # Process all pages for table data
            cv_images = [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in images]
            table_data = []
            for img in cv_images:
                page_data = self.extract_table_data(img)
                if page_data:
                    table_data.extend(page_data)

            all_data['table_data'] = table_data
            all_data['no_items'] = len(table_data)

            return all_data

        except Exception as e:
            logging.error(f"Error in process_multi_page_invoice: {str(e)}")
            raise

    def extract_all_data(self, image, output_dir=None, base_filename=None):
        """
        Extract all required data from the invoice image
        """
        # Enhance image for better OCR
        enhanced_image = self.ocr_engine.enhance_recognition(image)

        # Extract text
        text, text_confidence = self.extract_text(enhanced_image)

        # Extract all fields
        invoice_number, inv_conf = extract_invoice_number(text)
        date, date_conf = extract_date(text)
        gst_numbers = extract_gst_numbers(text)
        po_number, po_conf = extract_po_number(text)
        shipping_address, addr_conf = extract_shipping_address(text)
        seal_present, seal_conf, extracted_regions = detect_seal_and_signature(
            image, output_dir, base_filename)
        table_data = self.extract_table_data(enhanced_image)

        return {
            'invoice_number': {'value': invoice_number, 'confidence': inv_conf},
            'invoice_date': {'value': date, 'confidence': date_conf},
            'supplier_gst_number': {'value': gst_numbers[0][0] if gst_numbers else None,
                                    'confidence': gst_numbers[0][1] if gst_numbers else 0.0},
            'bill_to_gst_number': {'value': gst_numbers[1][0] if len(gst_numbers) > 1 else None,
                                   'confidence': gst_numbers[1][1] if len(gst_numbers) > 1 else 0.0},
            'po_number': {'value': po_number, 'confidence': po_conf},
            'shipping_address': {'value': shipping_address, 'confidence': addr_conf},
            'seal_and_sign_present': {'value': seal_present, 'confidence': seal_conf,
                                      'num_regions': len(extracted_regions)},
            'table_data': table_data,
            'no_items': len(table_data),
            'validation_results': self.validation_results._asdict() if self.validation_results else None
        }

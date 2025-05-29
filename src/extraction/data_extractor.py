import cv2
import pytesseract
import re
import numpy as np
from datetime import datetime

class DataExtractor:
    def __init__(self):
        self.confidence_threshold = 60  # Minimum confidence score to accept

    def extract_text(self, image):
        """
        Extract text from image using Tesseract OCR
        """
        # Get OCR data including confidence scores
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        # Combine text with confidence scores
        text_blocks = []
        confidences = []
        
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > self.confidence_threshold:
                text_blocks.append(data['text'][i])
                confidences.append(float(data['conf'][i]) / 100)
                
        return ' '.join(text_blocks), np.mean(confidences) if confidences else 0

    def extract_invoice_number(self, text):
        """
        Extract invoice number using regex patterns
        """
        patterns = [
            r'Invoice\s*#?\s*(\w+[-/]?\w+)',
            r'Invoice\s*Number\s*:?\s*(\w+[-/]?\w+)',
            r'Bill\s*Number\s*:?\s*(\w+[-/]?\w+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1), 0.9
        return None, 0.0

    def extract_date(self, text):
        """
        Extract invoice date using various date formats
        """
        date_patterns = [
            r'(?:Invoice|Bill|Date)\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
            r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    date_str = match.group(1)
                    # Add date parsing logic here
                    return date_str, 0.9
                except:
                    continue
        return None, 0.0

    def extract_gst_numbers(self, text):
        """
        Extract GST numbers for supplier and bill-to party
        """
        gst_pattern = r'(?:GST|GSTIN)\s*:?\s*(\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1})'
        matches = re.finditer(gst_pattern, text)
        
        gst_numbers = []
        for match in matches:
            gst_numbers.append((match.group(1), 0.85))
            
        return gst_numbers[:2] if len(gst_numbers) >= 2 else (None, 0.0)

    def extract_po_number(self, text):
        """
        Extract PO number
        """
        patterns = [
            r'P\.?O\.?\s*#?\s*(\w+[-/]?\w+)',
            r'Purchase\s*Order\s*:?\s*(\w+[-/]?\w+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1), 0.85
        return None, 0.0

    def extract_shipping_address(self, text):
        """
        Extract shipping address
        """
        patterns = [
            r'Ship\s*To\s*:?\s*(.*?)(?=\n\n|\Z)',
            r'Delivery\s*Address\s*:?\s*(.*?)(?=\n\n|\Z)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip(), 0.8
        return None, 0.0

    def detect_seal_and_signature(self, image):
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
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w)/h
                if 0.8 <= aspect_ratio <= 1.2:  # Nearly square (potential seal)
                    potential_seals.append((x, y, w, h))
        
        return len(potential_seals) > 0, 0.75

    def extract_table_data(self, image):
        """
        Extract tabular data from the invoice
        """
        # Get OCR data with bounding boxes
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        # Group text by lines based on y-coordinates
        lines = {}
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > self.confidence_threshold:
                y = data['top'][i]
                if y not in lines:
                    lines[y] = []
                lines[y].append({
                    'text': data['text'][i],
                    'left': data['left'][i],
                    'conf': float(data['conf'][i]) / 100
                })
        
        # Sort lines by y-coordinate
        sorted_lines = sorted(lines.items())
        
        # Extract table rows
        table_data = []
        for y, line in sorted_lines:
            # Sort text blocks by x-coordinate
            sorted_text = sorted(line, key=lambda x: x['left'])
            
            # Combine into row
            row = {
                'description': '',
                'hsn_sac': '',
                'quantity': 0,
                'unit_price': 0.0,
                'total_amount': 0.0,
                'serial_number': '',
                'confidence': 0.0
            }
            
            # Map text blocks to columns based on position
            confidences = []
            for item in sorted_text:
                text = item['text'].strip()
                conf = item['conf']
                confidences.append(conf)
                
                # Map to appropriate column based on content and position
                if text.isdigit() and not row['serial_number']:
                    row['serial_number'] = text
                elif re.match(r'^\d+$', text) and not row['hsn_sac']:
                    row['hsn_sac'] = text
                elif re.match(r'^\d+\.?\d*$', text):
                    if not row['quantity']:
                        row['quantity'] = float(text)
                    elif not row['unit_price']:
                        row['unit_price'] = float(text)
                    elif not row['total_amount']:
                        row['total_amount'] = float(text)
                else:
                    row['description'] += f" {text}"
            
            row['description'] = row['description'].strip()
            row['confidence'] = np.mean(confidences)
            
            if row['description'] or row['quantity'] or row['unit_price']:
                table_data.append(row)
        
        return table_data

    def extract_all_data(self, image):
        """
        Extract all required data from the invoice image
        """
        text, text_confidence = self.extract_text(image)
        
        invoice_number, inv_conf = self.extract_invoice_number(text)
        date, date_conf = self.extract_date(text)
        gst_numbers = self.extract_gst_numbers(text)
        po_number, po_conf = self.extract_po_number(text)
        shipping_address, addr_conf = self.extract_shipping_address(text)
        seal_present, seal_conf = self.detect_seal_and_signature(image)
        table_data = self.extract_table_data(image)
        
        return {
            'invoice_number': {'value': invoice_number, 'confidence': inv_conf},
            'invoice_date': {'value': date, 'confidence': date_conf},
            'supplier_gst_number': {'value': gst_numbers[0][0] if gst_numbers else None,
                                  'confidence': gst_numbers[0][1] if gst_numbers else 0.0},
            'bill_to_gst_number': {'value': gst_numbers[1][0] if len(gst_numbers) > 1 else None,
                                  'confidence': gst_numbers[1][1] if len(gst_numbers) > 1 else 0.0},
            'po_number': {'value': po_number, 'confidence': po_conf},
            'shipping_address': {'value': shipping_address, 'confidence': addr_conf},
            'seal_and_sign_present': {'value': seal_present, 'confidence': seal_conf},
            'table_data': table_data,
            'no_items': len(table_data)
        } 
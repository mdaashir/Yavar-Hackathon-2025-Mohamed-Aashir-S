import cv2
import pytesseract
import re
import numpy as np
from datetime import datetime
import os
from .table_extractor import TableExtractor, TableValidation

class DataExtractor:
    def __init__(self):
        self.confidence_threshold = 60  # Minimum confidence score to accept
        self.table_extractor = TableExtractor(min_confidence=self.confidence_threshold)
        self.validation_results = None  # Store validation results for reference

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

    def detect_seal_and_signature(self, image, output_dir=None, base_filename=None):
        """
        Detect and extract seal and signature from the image
        
        Args:
            image: Input image
            output_dir: Directory to save extracted seal/signature images
            base_filename: Base name for output files
            
        Returns:
            tuple: (bool indicating presence, confidence score, list of extracted regions)
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
                aspect_ratio = float(w)/h
                
                # Check if the region is nearly square (potential seal) or elongated (potential signature)
                if (0.8 <= aspect_ratio <= 1.2) or (aspect_ratio < 0.5):  # Square seal or elongated signature
                    potential_seals.append((x, y, w, h))
                    
                    # Extract the region
                    padding = 10  # Add padding around the region
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

    def extract_table_data(self, image):
        """
        Extract tabular data from the invoice using the enhanced TableExtractor
        """
        table_data = self.table_extractor.extract_table_data(image)
        self.validation_results = self.table_extractor.validate_table_data(table_data)
        return table_data

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
        
        # Process first page for header information
        if images:
            first_page = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
            text, text_confidence = self.extract_text(first_page)
            
            # Extract header information
            all_data['invoice_number']['value'], all_data['invoice_number']['confidence'] = self.extract_invoice_number(text)
            all_data['invoice_date']['value'], all_data['invoice_date']['confidence'] = self.extract_date(text)
            gst_numbers = self.extract_gst_numbers(text)
            if gst_numbers:
                all_data['supplier_gst_number']['value'] = gst_numbers[0][0]
                all_data['supplier_gst_number']['confidence'] = gst_numbers[0][1]
                if len(gst_numbers) > 1:
                    all_data['bill_to_gst_number']['value'] = gst_numbers[1][0]
                    all_data['bill_to_gst_number']['confidence'] = gst_numbers[1][1]
            
            all_data['po_number']['value'], all_data['po_number']['confidence'] = self.extract_po_number(text)
            all_data['shipping_address']['value'], all_data['shipping_address']['confidence'] = self.extract_shipping_address(text)
            
            # Process seal and signature
            seal_present, seal_conf, extracted_regions = self.detect_seal_and_signature(
                first_page, output_dir, base_filename)
            all_data['seal_and_sign_present'] = {
                'value': seal_present,
                'confidence': seal_conf,
                'num_regions': len(extracted_regions)
            }
        
        # Process all pages for table data
        cv_images = [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in images]
        table_data = self.table_extractor.process_multi_page_table(cv_images)
        all_data['table_data'] = table_data
        all_data['no_items'] = len(table_data)
        
        return all_data

    def extract_all_data(self, image, output_dir=None, base_filename=None):
        """
        Extract all required data from the invoice image
        """
        text, text_confidence = self.extract_text(image)
        
        invoice_number, inv_conf = self.extract_invoice_number(text)
        date, date_conf = self.extract_date(text)
        gst_numbers = self.extract_gst_numbers(text)
        po_number, po_conf = self.extract_po_number(text)
        shipping_address, addr_conf = self.extract_shipping_address(text)
        seal_present, seal_conf, extracted_regions = self.detect_seal_and_signature(
            image, output_dir, base_filename)
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
            'seal_and_sign_present': {'value': seal_present, 'confidence': seal_conf,
                                    'num_regions': len(extracted_regions)},
            'table_data': table_data,
            'no_items': len(table_data),
            'validation_results': self.validation_results._asdict() if self.validation_results else None
        } 
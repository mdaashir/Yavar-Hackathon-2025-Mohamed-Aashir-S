import unittest
import numpy as np
import cv2
from unittest.mock import Mock, patch
from src.extraction.data_extractor import DataExtractor

class TestDataExtractor(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.extractor = DataExtractor()
        # Create a simple test image
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.putText(self.test_image, "Invoice #12345", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(self.test_image, "GST: 29ABCDE1234F1Z5", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def test_extract_invoice_number(self):
        """Test invoice number extraction"""
        text = "Invoice #12345\nDate: 2024-03-15"
        invoice_number, confidence = self.extractor.extract_invoice_number(text)
        self.assertEqual(invoice_number, "12345")
        self.assertGreater(confidence, 0.8)

    def test_extract_gst_numbers(self):
        """Test GST number extraction"""
        text = "Supplier GST: 29ABCDE1234F1Z5\nCustomer GST: 27PQRST5678G1Z3"
        gst_numbers = self.extractor.extract_gst_numbers(text)
        self.assertIsNotNone(gst_numbers)
        self.assertEqual(len(gst_numbers), 2)
        self.assertEqual(gst_numbers[0][0], "29ABCDE1234F1Z5")
        self.assertEqual(gst_numbers[1][0], "27PQRST5678G1Z3")

    def test_validate_gst_number(self):
        """Test GST number validation"""
        valid_gst = "29ABCDE1234F1Z5"
        invalid_gst = "123456"
        self.assertTrue(self.extractor._validate_gst_number(valid_gst))
        self.assertFalse(self.extractor._validate_gst_number(invalid_gst))

    def test_extract_date(self):
        """Test date extraction with various formats"""
        test_cases = [
            ("Invoice Date: 15/03/2024", "15/03/2024"),
            ("Date: 2024-03-15", "2024-03-15"),
            ("Bill Date: 15 Mar 2024", "15 Mar 2024"),
        ]
        for text, expected_date in test_cases:
            date, confidence = self.extractor.extract_date(text)
            self.assertEqual(date, expected_date)
            self.assertGreater(confidence, 0.8)

    @patch('src.extraction.data_extractor.OCREngine')
    def test_extract_text(self, mock_ocr):
        """Test text extraction with OCR"""
        mock_ocr.return_value.extract_text.return_value = ("Sample text", 0.95)
        text, confidence = self.extractor.extract_text(self.test_image)
        self.assertEqual(text, "Sample text")
        self.assertEqual(confidence, 0.95)

    def test_extract_po_number(self):
        """Test PO number extraction"""
        test_cases = [
            ("PO #: ABC123", "ABC123"),
            ("Purchase Order: XYZ-789", "XYZ-789"),
            ("Order Number: PO-2024-001", "PO-2024-001"),
        ]
        for text, expected_po in test_cases:
            po_number, confidence = self.extractor.extract_po_number(text)
            self.assertEqual(po_number, expected_po)
            self.assertGreater(confidence, 0.8)

    def test_extract_shipping_address(self):
        """Test shipping address extraction"""
        text = "Ship To: 123 Test Street\nCity, State 12345"
        address, confidence = self.extractor.extract_shipping_address(text)
        self.assertEqual(address, "123 Test Street\nCity, State 12345")
        self.assertGreater(confidence, 0.7)

    def test_handle_empty_input(self):
        """Test handling of empty input"""
        self.assertIsNone(self.extractor.extract_gst_numbers(""))
        self.assertEqual(self.extractor.extract_invoice_number(""), (None, 0.0))
        self.assertEqual(self.extractor.extract_date(""), (None, 0.0))

if __name__ == '__main__':
    unittest.main() 
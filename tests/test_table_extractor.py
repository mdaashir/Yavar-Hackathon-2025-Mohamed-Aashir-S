import unittest
import numpy as np
import cv2
from src.extraction.table_extractor import TableExtractor, TableCell, TableValidation

class TestTableExtractor(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.extractor = TableExtractor()
        
        # Create a simple test image with a table
        self.test_image = np.zeros((300, 400, 3), dtype=np.uint8)
        # Draw table lines
        cv2.line(self.test_image, (50, 50), (350, 50), (255, 255, 255), 2)  # Top
        cv2.line(self.test_image, (50, 100), (350, 100), (255, 255, 255), 2)  # Header
        cv2.line(self.test_image, (50, 150), (350, 150), (255, 255, 255), 2)  # Row
        cv2.line(self.test_image, (50, 50), (50, 150), (255, 255, 255), 2)   # Left
        cv2.line(self.test_image, (350, 50), (350, 150), (255, 255, 255), 2) # Right

    def test_preprocess_image(self):
        """Test image preprocessing"""
        processed = self.extractor.preprocess_image(self.test_image)
        self.assertIsInstance(processed, np.ndarray)
        self.assertEqual(len(processed.shape), 2)  # Should be grayscale

    def test_detect_table_region(self):
        """Test table region detection"""
        found, region, cells = self.extractor.detect_table_region(self.test_image)
        self.assertTrue(found)
        self.assertIsInstance(region, np.ndarray)
        self.assertIsInstance(cells, list)

    def test_validate_table_data(self):
        """Test table data validation"""
        test_data = [
            {
                'description': 'Item 1',
                'quantity': 2,
                'unit_price': 10.0,
                'total_amount': 20.0
            },
            {
                'description': 'Item 2',
                'quantity': 3,
                'unit_price': 15.0,
                'total_amount': 45.0
            }
        ]
        validation = self.extractor.validate_table_data(test_data)
        self.assertIsInstance(validation, TableValidation)
        self.assertTrue(validation.is_valid)
        self.assertEqual(len(validation.errors), 0)

    def test_validate_table_data_with_errors(self):
        """Test table data validation with calculation errors"""
        test_data = [
            {
                'description': 'Item 1',
                'quantity': 2,
                'unit_price': 10.0,
                'total_amount': 25.0  # Incorrect total
            }
        ]
        validation = self.extractor.validate_table_data(test_data)
        self.assertIsInstance(validation, TableValidation)
        self.assertFalse(validation.is_valid)
        self.assertGreater(len(validation.errors), 0)

    def test_map_header_to_column(self):
        """Test header mapping"""
        test_cases = [
            ('Item Description', 'description'),
            ('Qty', 'quantity'),
            ('Unit Price', 'unit_price'),
            ('Total Amount', 'total_amount'),
            ('HSN Code', 'hsn_sac'),
            ('Sr. No.', 'serial_number'),
            ('Invalid Header', None)
        ]
        for header, expected in test_cases:
            result = self.extractor._map_header_to_column(header)
            self.assertEqual(result, expected)

    def test_is_numeric(self):
        """Test numeric value validation"""
        test_cases = [
            ('123.45', True),
            ('1,234.56', True),
            ('abc', False),
            ('', False),
            ('12.34.56', False)
        ]
        for value, expected in test_cases:
            result = self.extractor._is_numeric(value)
            self.assertEqual(result, expected)

    def test_process_multi_page_table(self):
        """Test multi-page table processing"""
        # Create two test images
        images = [self.test_image.copy() for _ in range(2)]
        table_data = self.extractor.process_multi_page_table(images)
        self.assertIsInstance(table_data, list)

    def test_empty_table(self):
        """Test handling of empty table"""
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        found, region, cells = self.extractor.detect_table_region(empty_image)
        self.assertFalse(found)
        self.assertEqual(len(cells), 0)

if __name__ == '__main__':
    unittest.main() 
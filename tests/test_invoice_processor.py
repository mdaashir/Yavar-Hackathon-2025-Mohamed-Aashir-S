import shutil
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.extraction.data_extractor import DataExtractor
from src.main import InvoiceProcessor
from src.preprocessing.image_processor import ImagePreprocessor
from src.verification.data_verifier import DataVerifier, verify_date_format


class TestInvoiceProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_dir = Path('test_data')
        cls.input_dir = cls.test_dir / 'input'
        cls.output_dir = cls.test_dir / 'output'

        # Create test directories
        cls.input_dir.mkdir(parents=True, exist_ok=True)
        cls.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize processor
        cls.processor = InvoiceProcessor(
            input_dir=str(cls.input_dir),
            output_dir=str(cls.output_dir)
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        shutil.rmtree(cls.test_dir)

    def setUp(self):
        """Set up for each test"""
        # Clear output directory
        for file in self.output_dir.glob('*'):
            if file.is_file():
                file.unlink()

    def test_image_preprocessing(self):
        """Test image preprocessing functionality"""
        # Create a test image
        test_image = np.zeros((100, 100), dtype=np.uint8)
        cv2.putText(test_image, "Test", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

        # Process image
        preprocessor = ImagePreprocessor()
        processed = preprocessor.process_image(test_image)

        # Verify processing
        self.assertIsNotNone(processed)
        self.assertEqual(processed.shape, test_image.shape)

    def test_data_extraction(self):
        """Test data extraction functionality"""
        extractor = DataExtractor()

        # Create a simple test image
        test_image = np.zeros((200, 200), dtype=np.uint8)
        cv2.putText(test_image, "Invoice#: TEST123", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
        cv2.putText(test_image, "Date: 2025-05-30", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)

        # Extract data
        data = extractor.extract_all_data(test_image)

        # Verify extraction
        self.assertIsInstance(data, dict)
        self.assertIn('invoice_number', data)
        self.assertIn('invoice_date', data)

    def test_data_verification(self):
        """Test data verification functionality"""
        verifier = DataVerifier()

        # Test data
        test_data = {
            'invoice_number': {'value': 'TEST123', 'confidence': 0.95},
            'invoice_date': {'value': '2025-05-30', 'confidence': 0.92},
            'supplier_gst_number': {'value': '29ABCDE1234F1Z5', 'confidence': 0.89},
            'table_data': [
                {
                    'description': 'Item 1',
                    'quantity': 2,
                    'unit_price': 100.00,
                    'total_amount': 200.00
                }
            ]
        }

        # Verify data
        results = verifier.verify_data(test_data)

        # Check verification results
        self.assertIsInstance(results, dict)
        self.assertIn('field_verification', results)
        self.assertIn('line_items_verification', results)
        self.assertIn('summary', results)

    def test_output_generation(self):
        """Test output file generation"""
        # Test data
        test_data = {
            'invoice_number': {'value': 'TEST123', 'confidence': 0.95},
            'invoice_date': {'value': '2025-05-30', 'confidence': 0.92},
            'table_data': [
                {
                    'description': 'Item 1',
                    'quantity': 2,
                    'unit_price': 100.00,
                    'total_amount': 200.00
                }
            ]
        }

        verification_results = {
            'field_verification': {
                'invoice_number': {'confidence': 0.95, 'present': True}
            },
            'summary': {'all_fields_confident': True}
        }

        # Save outputs
        success = self.processor.save_outputs(
            test_data,
            verification_results,
            'test_invoice'
        )

        # Verify output files
        self.assertTrue(success)
        self.assertTrue((self.output_dir / 'test_invoice_data.json').exists())
        self.assertTrue((self.output_dir / 'test_invoice_data.xlsx').exists())
        self.assertTrue((self.output_dir / 'test_invoice_verification.json').exists())

    def test_gst_number_validation(self):
        """Test GST number validation"""
        verifier = DataVerifier()

        # Valid GST number
        self.assertTrue(verifier.verify_gst_number('29ABCDE1234F1Z5'))

        # Invalid GST numbers
        self.assertFalse(verifier.verify_gst_number(''))
        self.assertFalse(verifier.verify_gst_number('INVALID'))
        self.assertFalse(verifier.verify_gst_number('29ABCDE1234'))

    def test_date_format_validation(self):
        """Test date format validation"""
        DataVerifier()

        # Valid dates
        self.assertTrue(verify_date_format('2025-05-30'))
        self.assertTrue(verify_date_format('30/05/2025'))
        self.assertTrue(verify_date_format('30 May 2025'))

        # Invalid dates
        self.assertFalse(verify_date_format(''))
        self.assertFalse(verify_date_format('Invalid'))
        self.assertFalse(verify_date_format('2025/13/45'))


if __name__ == '__main__':
    unittest.main()

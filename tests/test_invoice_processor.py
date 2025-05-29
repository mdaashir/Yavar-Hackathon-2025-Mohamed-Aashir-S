import unittest
import os
import shutil
import tempfile
from unittest.mock import Mock, patch
from pathlib import Path
import numpy as np
import cv2
from src.main import InvoiceProcessor
from src.utils.error_handler import PDFConversionError, OCRError, ValidationError

class TestInvoiceProcessor(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        # Create temporary directories for testing
        self.test_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.test_dir, 'input')
        self.output_dir = os.path.join(self.test_dir, 'output')
        os.makedirs(self.input_dir)
        
        # Initialize processor
        self.processor = InvoiceProcessor(
            input_dir=self.input_dir,
            output_dir=self.output_dir
        )
        
        # Create a test PDF file
        self.test_pdf = os.path.join(self.input_dir, 'test_invoice.pdf')
        with open(self.test_pdf, 'w') as f:
            f.write('Test PDF content')

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)

    def test_create_directories(self):
        """Test directory creation"""
        self.assertTrue(os.path.exists(self.output_dir))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'seals_and_signatures')))

    @patch('src.main.PDFConverter')
    @patch('src.main.ImagePreprocessor')
    @patch('src.main.DataExtractor')
    def test_process_invoice_success(self, mock_extractor, mock_preprocessor, mock_converter):
        """Test successful invoice processing"""
        # Mock PDF conversion
        mock_converter.return_value.pdf_to_images.return_value = [
            np.zeros((100, 100, 3), dtype=np.uint8)
        ]
        
        # Mock data extraction
        mock_extractor.return_value.extract_all_data.return_value = {
            'invoice_number': {'value': '12345', 'confidence': 0.9},
            'table_data': []
        }
        
        # Process invoice
        extracted_data, verification_results = self.processor.process_invoice(self.test_pdf)
        
        self.assertIsNotNone(extracted_data)
        self.assertEqual(extracted_data['invoice_number']['value'], '12345')

    @patch('src.main.PDFConverter')
    def test_process_invoice_pdf_error(self, mock_converter):
        """Test PDF conversion error handling"""
        mock_converter.return_value.pdf_to_images.return_value = None
        
        result = self.processor.process_invoice(self.test_pdf)
        self.assertEqual(result, (None, None))
        
        # Check error tracking
        self.assertEqual(len(self.processor.error_tracker.errors), 1)
        self.assertEqual(self.processor.error_tracker.errors[0]['error_type'], 'PDFConversionError')

    def test_save_outputs_success(self):
        """Test successful output saving"""
        test_data = {
            'invoice_number': {'value': '12345', 'confidence': 0.9},
            'table_data': [
                {'description': 'Item 1', 'quantity': 1, 'unit_price': 10.0}
            ]
        }
        verification_results = {'is_valid': True, 'errors': []}
        
        success = self.processor.save_outputs(
            test_data, verification_results, 'test_invoice'
        )
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'test_invoice_data.json')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'test_invoice_data.xlsx')))

    def test_save_outputs_error(self):
        """Test output saving error handling"""
        test_data = {'invalid_data': None}
        verification_results = {'is_valid': False}
        
        success = self.processor.save_outputs(
            test_data, verification_results, 'test_invoice'
        )
        
        self.assertFalse(success)
        self.assertGreater(len(self.processor.error_tracker.errors), 0)

    @patch('src.main.PDFConverter')
    @patch('src.main.DataExtractor')
    def test_process_directory(self, mock_extractor, mock_converter):
        """Test directory processing"""
        # Create multiple test PDFs
        for i in range(3):
            with open(os.path.join(self.input_dir, f'test_{i}.pdf'), 'w') as f:
                f.write(f'Test PDF {i}')
        
        # Mock successful processing
        mock_converter.return_value.pdf_to_images.return_value = [
            np.zeros((100, 100, 3), dtype=np.uint8)
        ]
        mock_extractor.return_value.extract_all_data.return_value = {
            'invoice_number': {'value': '12345', 'confidence': 0.9},
            'table_data': []
        }
        
        # Process directory
        self.processor.process_directory()
        
        # Check error report
        error_report_path = os.path.join(self.output_dir, 'error_report.txt')
        self.assertFalse(os.path.exists(error_report_path))

    def test_empty_directory(self):
        """Test handling of empty input directory"""
        # Remove test files
        for file in os.listdir(self.input_dir):
            os.remove(os.path.join(self.input_dir, file))
            
        self.processor.process_directory()
        # Should not raise any errors

    @patch('src.main.PDFConverter')
    def test_error_report_generation(self, mock_converter):
        """Test error report generation"""
        # Simulate errors
        mock_converter.return_value.pdf_to_images.side_effect = PDFConversionError("Test error")
        
        # Create test PDF
        with open(os.path.join(self.input_dir, 'error_test.pdf'), 'w') as f:
            f.write('Test PDF')
            
        self.processor.process_directory()
        
        # Check error report
        error_report_path = os.path.join(self.output_dir, 'error_report.txt')
        self.assertTrue(os.path.exists(error_report_path))
        
        with open(error_report_path, 'r') as f:
            report_content = f.read()
            self.assertIn('PDFConversionError', report_content)
            self.assertIn('error_test.pdf', report_content)

if __name__ == '__main__':
    unittest.main() 
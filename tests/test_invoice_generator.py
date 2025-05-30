import os
import unittest
from datetime import datetime
from pathlib import Path

from src.utils.invoice_generator import InvoiceGenerator, generate_invoice_items, generate_company_details


class TestInvoiceGenerator(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.test_output_dir = Path('test_output')
        self.test_output_dir.mkdir(exist_ok=True)
        self.generator = InvoiceGenerator(output_dir=str(self.test_output_dir))

    def tearDown(self):
        """Clean up test environment"""
        for file in self.test_output_dir.glob('*.pdf'):
            file.unlink()
        self.test_output_dir.rmdir()

    def test_generate_company_details(self):
        """Test company details generation"""
        company, gst, address = generate_company_details()

        # Verify company details format
        self.assertIsInstance(company, str)
        self.assertGreater(len(company), 0)

        # Verify GST number format
        self.assertRegex(gst, r'^\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}$')

        # Verify address
        self.assertIsInstance(address, str)
        self.assertGreater(len(address), 0)

    def test_generate_invoice_items(self):
        """Test invoice items generation"""
        # Test with default random number of items
        items = generate_invoice_items()
        self.assertGreaterEqual(len(items), 3)
        self.assertLessEqual(len(items), 8)

        # Test with specific number of items
        num_items = 5
        items = generate_invoice_items(num_items)
        self.assertEqual(len(items), num_items)

        # Verify item format
        for item in items:
            self.assertEqual(len(item), 6)  # 6 columns
            self.assertIsInstance(item[0], str)  # Sr. No.
            self.assertIsInstance(item[1], str)  # Description
            self.assertIsInstance(item[2], str)  # HSN/SAC
            self.assertIsInstance(item[3], str)  # Quantity
            self.assertIsInstance(item[4], str)  # Rate
            self.assertIsInstance(item[5], str)  # Amount

            # Verify calculations
            quantity = float(item[3])
            rate = float(item[4])
            amount = float(item[5])
            self.assertAlmostEqual(quantity * rate, amount, places=2)

    def test_create_invoice(self):
        """Test single invoice creation"""
        filename = self.generator.create_invoice(1)

        # Verify file creation
        self.assertTrue(filename.exists())
        self.assertTrue(filename.is_file())

        # Verify file name format
        current_year = datetime.now().year
        self.assertRegex(filename.name, f"INV{current_year}0001.pdf")

        # Verify file size (should be non-zero)
        self.assertGreater(os.path.getsize(filename), 0)

    def test_generate_test_set(self):
        """Test generation of multiple invoices"""
        num_invoices = 5
        files = self.generator.generate_test_set(num_invoices)

        # Verify number of files generated
        self.assertEqual(len(files), num_invoices)

        # Verify all files exist and are valid
        for file in files:
            self.assertTrue(file.exists())
            self.assertTrue(file.is_file())
            self.assertGreater(os.path.getsize(file), 0)

        # Verify unique invoice numbers
        invoice_numbers = [file.stem for file in files]
        self.assertEqual(len(set(invoice_numbers)), num_invoices)


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3
"""
Run script for the Invoice Data Extraction & Verification System.
This script provides a command-line interface to run the invoice processing system.
"""

import argparse
import sys
from pathlib import Path

from src.main import InvoiceProcessor
from src.utils.invoice_generator import InvoiceGenerator


def main():
    parser = argparse.ArgumentParser(
        description='Invoice Data Extraction & Verification System'
    )
    parser.add_argument(
        '-i', '--input-dir',
        type=str,
        default='src/input',
        help='Directory containing input PDF files (default: src/input)'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='src/output',
        help='Directory for output files (default: src/output)'
    )
    parser.add_argument(
        '--generate-samples',
        action='store_true',
        help='Generate sample invoices before processing'
    )
    parser.add_argument(
        '-n', '--num-samples',
        type=int,
        default=10,
        help='Number of sample invoices to generate (default: 10)'
    )
    args = parser.parse_args()

    try:
        # Create directories if they don't exist
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate sample invoices if requested
        if args.generate_samples:
            InvoiceGenerator.generate_test_invoices(str(input_dir), args.num_samples)

        # Initialize and run the processor
        print("Initializing invoice processor...")
        processor = InvoiceProcessor(
            input_dir=str(input_dir),
            output_dir=str(output_dir)
        )

        print("Starting invoice processing...")
        processor.process_directory()
        print("Processing complete!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()

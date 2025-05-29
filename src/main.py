import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from extraction.data_extractor import DataExtractor
from preprocessing.image_processor import ImagePreprocessor
from utils.pdf_converter import PDFConverter
from verification.data_verifier import DataVerifier


class InvoiceProcessor:
    def __init__(self, input_dir='input', output_dir='output'):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.seals_dir = os.path.join(output_dir, 'seals_and_signatures')
        self.pdf_converter = PDFConverter()
        self.image_processor = ImagePreprocessor()
        self.data_extractor = DataExtractor()
        self.data_verifier = DataVerifier()

        # Create output directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.seals_dir, exist_ok=True)

    def process_invoice(self, pdf_path):
        """
        Process a single invoice PDF
        """
        # Convert PDF to images
        images = self.pdf_converter.pdf_to_images(pdf_path)
        if not images:
            print(f"Failed to convert PDF: {pdf_path}")
            return None

        # Get base filename for outputs
        base_filename = Path(pdf_path).stem

        # Process all pages
        processed_images = []
        for image in images:
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            # Preprocess image
            processed_image = self.image_processor.process_image(cv_image)
            processed_images.append(processed_image)

        # Extract data (including seals and signatures)
        if len(processed_images) > 1:
            # Multi-page invoice
            extracted_data = self.data_extractor.process_multi_page_invoice(
                processed_images,
                output_dir=self.seals_dir,
                base_filename=base_filename
            )
        else:
            # Single-page invoice
            extracted_data = self.data_extractor.extract_all_data(
                processed_images[0],
                output_dir=self.seals_dir,
                base_filename=base_filename
            )

        # Verify data
        verification_results = self.data_verifier.verify_data(extracted_data)

        # Save validation results if any
        if extracted_data.get('validation_results'):
            validation_path = os.path.join(self.output_dir, f"{base_filename}_table_validation.json")
            with open(validation_path, 'w') as f:
                json.dump(extracted_data['validation_results'], f, indent=2)

            # Remove validation results from extracted data to maintain original format
            del extracted_data['validation_results']

        return extracted_data, verification_results

    def save_outputs(self, extracted_data, verification_results, base_filename):
        """
        Save outputs in required formats
        """
        # Save JSON data
        json_path = os.path.join(self.output_dir, f"{base_filename}_data.json")
        with open(json_path, 'w') as f:
            json.dump(extracted_data, f, indent=2)

        # Save verification report
        verification_path = os.path.join(self.output_dir, f"{base_filename}_verification.json")
        with open(verification_path, 'w') as f:
            json.dump(verification_results, f, indent=2)

        # Convert to Excel
        excel_path = os.path.join(self.output_dir, f"{base_filename}_data.xlsx")

        # Prepare data for Excel
        excel_data = {
            'General Information': pd.DataFrame([{
                'Field': k,
                'Value': v['value'],
                'Confidence': v['confidence']
            } for k, v in extracted_data.items() if k != 'table_data']),

            'Line Items': pd.DataFrame(extracted_data['table_data'])
        }

        # Save to Excel with multiple sheets
        with pd.ExcelWriter(excel_path) as writer:
            for sheet_name, df in excel_data.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    def process_directory(self):
        """
        Process all PDFs in the input directory
        """
        input_path = Path(self.input_dir)

        for pdf_file in input_path.glob('*.pdf'):
            print(f"Processing {pdf_file}...")

            try:
                # Process invoice
                results = self.process_invoice(str(pdf_file))
                if results is None:
                    continue

                extracted_data, verification_results = results

                # Save outputs
                base_filename = pdf_file.stem
                self.save_outputs(extracted_data, verification_results, base_filename)

                print(f"Successfully processed {pdf_file}")

            except Exception as e:
                print(f"Error processing {pdf_file}: {str(e)}")


def main():
    processor = InvoiceProcessor()
    processor.process_directory()


if __name__ == "__main__":
    main()

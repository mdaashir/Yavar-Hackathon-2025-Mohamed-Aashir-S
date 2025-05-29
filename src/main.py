import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import traceback
from tqdm import tqdm

from extraction.data_extractor import DataExtractor
from preprocessing.image_processor import ImagePreprocessor
from utils.pdf_converter import PDFConverter
from verification.data_verifier import DataVerifier
from utils.logger import Logger
from utils.error_handler import (
    ErrorTracker, PDFConversionError, OCRError, 
    ValidationError, OutputError
)


class InvoiceProcessor:
    def __init__(self, input_dir='input', output_dir='output'):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.seals_dir = os.path.join(output_dir, 'seals_and_signatures')
        
        # Initialize components
        self.pdf_converter = PDFConverter()
        self.image_processor = ImagePreprocessor()
        self.data_extractor = DataExtractor()
        self.data_verifier = DataVerifier()
        
        # Initialize logger and error tracker
        self.logger = Logger.get_logger()
        self.error_tracker = ErrorTracker()
        
        # Create output directories
        self._create_directories()

    def _create_directories(self):
        """Create necessary output directories"""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(self.seals_dir, exist_ok=True)
            self.logger.info("Output directories created successfully")
        except Exception as e:
            self.logger.error(f"Error creating directories: {str(e)}")
            raise

    def process_invoice(self, pdf_path: str) -> tuple:
        """
        Process a single invoice PDF with error handling
        """
        base_filename = Path(pdf_path).stem
        self.logger.info(f"Processing invoice: {base_filename}")
        
        try:
            # Convert PDF to images
            images = self.pdf_converter.pdf_to_images(pdf_path)
            if not images:
                raise PDFConversionError(f"Failed to convert PDF: {pdf_path}")
            self.logger.debug(f"Converted PDF to {len(images)} images")
            
            # Process all pages
            processed_images = []
            for i, image in enumerate(images, 1):
                self.logger.debug(f"Processing page {i}/{len(images)}")
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                processed_image = self.image_processor.process_image(cv_image)
                processed_images.append(processed_image)
            
            # Extract data
            try:
                if len(processed_images) > 1:
                    extracted_data = self.data_extractor.process_multi_page_invoice(
                        processed_images,
                        output_dir=self.seals_dir,
                        base_filename=base_filename
                    )
                else:
                    extracted_data = self.data_extractor.extract_all_data(
                        processed_images[0],
                        output_dir=self.seals_dir,
                        base_filename=base_filename
                    )
            except Exception as e:
                raise OCRError(f"OCR extraction failed: {str(e)}")
            
            # Verify data
            try:
                verification_results = self.data_verifier.verify_data(extracted_data)
            except Exception as e:
                raise ValidationError(f"Data validation failed: {str(e)}")
            
            # Save validation results if any
            if extracted_data.get('validation_results'):
                validation_path = os.path.join(
                    self.output_dir, 
                    f"{base_filename}_table_validation.json"
                )
                with open(validation_path, 'w') as f:
                    json.dump(extracted_data['validation_results'], f, indent=2)
                del extracted_data['validation_results']
            
            self.logger.info(f"Successfully processed {base_filename}")
            return extracted_data, verification_results
            
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            
            self.error_tracker.add_error(
                error_type=error_type,
                message=error_msg,
                file_name=base_filename,
                stack_trace=stack_trace
            )
            
            self.logger.error(f"Error processing {base_filename}: {error_msg}")
            self.logger.debug(f"Stack trace:\n{stack_trace}")
            
            return None, None

    def save_outputs(self, extracted_data: dict, verification_results: dict, 
                    base_filename: str) -> bool:
        """
        Save outputs in required formats with error handling
        """
        try:
            # Save JSON data
            json_path = os.path.join(self.output_dir, f"{base_filename}_data.json")
            with open(json_path, 'w') as f:
                json.dump(extracted_data, f, indent=2)
            
            # Save verification report
            verification_path = os.path.join(
                self.output_dir, 
                f"{base_filename}_verification.json"
            )
            with open(verification_path, 'w') as f:
                json.dump(verification_results, f, indent=2)
            
            # Convert to Excel
            excel_path = os.path.join(self.output_dir, f"{base_filename}_data.xlsx")
            
            # Prepare data for Excel with type checking
            excel_rows = []
            for k, v in extracted_data.items():
                if k == 'table_data':
                    continue
                    
                if isinstance(v, dict) and 'value' in v and 'confidence' in v:
                    excel_rows.append({
                        'Field': k,
                        'Value': v['value'],
                        'Confidence': v['confidence']
                    })
                else:
                    excel_rows.append({
                        'Field': k,
                        'Value': v,
                        'Confidence': 1.0 if isinstance(v, (int, float)) else 0.0
                    })
            
            excel_data = {
                'General Information': pd.DataFrame(excel_rows),
                'Line Items': pd.DataFrame(extracted_data.get('table_data', []))
            }
            
            # Save to Excel with multiple sheets
            with pd.ExcelWriter(excel_path) as writer:
                for sheet_name, df in excel_data.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            self.logger.info(f"Saved outputs for {base_filename}")
            return True
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            
            self.error_tracker.add_error(
                error_type='OutputError',
                message=error_msg,
                file_name=base_filename,
                stack_trace=stack_trace
            )
            
            self.logger.error(f"Error saving outputs for {base_filename}: {error_msg}")
            self.logger.debug(f"Stack trace:\n{stack_trace}")
            
            return False

    def process_directory(self):
        """
        Process all PDFs in the input directory with progress tracking
        """
        input_path = Path(self.input_dir)
        pdf_files = list(input_path.glob('*.pdf'))
        total_files = len(pdf_files)
        
        if not total_files:
            self.logger.warning("No PDF files found in input directory")
            return
        
        self.logger.info(f"Starting processing of {total_files} files")
        successful = 0
        failed = 0
        
        # Process files with progress bar
        for pdf_file in tqdm(pdf_files, desc="Processing invoices"):
            try:
                # Process invoice
                results = self.process_invoice(str(pdf_file))
                if results[0] is None:  # Check if processing failed
                    failed += 1
                    continue
                
                extracted_data, verification_results = results
                
                # Save outputs
                base_filename = pdf_file.stem
                if self.save_outputs(extracted_data, verification_results, base_filename):
                    successful += 1
                else:
                    failed += 1
                
            except Exception as e:
                failed += 1
                self.logger.error(f"Unexpected error processing {pdf_file}: {str(e)}")
        
        # Generate and save error report if there were any errors
        if self.error_tracker.errors:
            report_path = os.path.join(self.output_dir, "error_report.txt")
            error_report = self.error_tracker.generate_error_report()
            try:
                with open(report_path, 'w') as f:
                    f.write(error_report)
                self.logger.info(f"Error report saved to {report_path}")
            except Exception as e:
                self.logger.error(f"Failed to save error report: {str(e)}")
        
        # Log final statistics
        self.logger.info(f"\nProcessing completed:")
        self.logger.info(f"Total files: {total_files}")
        self.logger.info(f"Successfully processed: {successful}")
        self.logger.info(f"Failed: {failed}")

def main():
    try:
        processor = InvoiceProcessor()
        processor.process_directory()
    except Exception as e:
        logger = Logger.get_logger()
        logger.critical(f"Critical error in main process: {str(e)}")
        logger.debug(f"Stack trace:\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()

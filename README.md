# Invoice Data Extraction & Verification System

This system extracts data from scanned invoice PDFs using OCR and performs verification checks on the extracted data.

## Features

- Extracts key information from scanned invoices
- Performs data validation and verification
- Generates structured output in JSON and Excel formats
- Includes confidence scores for extracted fields
- Detects and extracts vendor seals and signatures

## Project Structure

```
.
├── input/                  # Directory for input invoice PDFs
├── output/                 # Directory for processed outputs
├── src/                   
│   ├── preprocessing/     # Image preprocessing modules
│   ├── extraction/        # Data extraction modules
│   ├── verification/      # Data verification modules
│   └── utils/            # Utility functions
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Setup Instructions

1. Install Python 3.8 or higher
2. Install Tesseract OCR:
   - Windows: Download installer from https://github.com/UB-Mannheim/tesseract/wiki
   - Linux: `sudo apt install tesseract-ocr`
   - Mac: `brew install tesseract`
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place scanned invoice PDFs in the `input/` directory
2. Run the extraction script:
   ```bash
   python src/main.py
   ```
3. Check the `output/` directory for results:
   - `extracted_data.json`: Extracted data in JSON format
   - `extracted_data.xlsx`: Data in Excel format
   - `verifiability_report.json`: Confidence scores and validation results
   - Extracted seal and signature images

## Dependencies

- OpenCV
- Tesseract OCR
- pdf2image
- NumPy
- Pandas
- Pillow
- scikit-image
- imutils

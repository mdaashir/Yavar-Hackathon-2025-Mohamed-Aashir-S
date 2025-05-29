# Invoice Data Extraction & Verification System

This project implements an automated system for extracting and verifying data from scanned invoice PDFs using open-source OCR models. The system processes image-based invoices to extract structured information and performs comprehensive validation checks.

## Features

- **PDF Processing**: Convert multi-page PDFs to images
- **Advanced OCR**: Text extraction with EasyOCR (GPU-accelerated)
- **Intelligent Data Extraction**:
  - Invoice numbers
  - Dates
  - GST numbers
  - PO numbers
  - Shipping addresses
  - Table data
  - Seals and signatures
- **Data Validation**: Comprehensive validation of extracted data
- **Multiple Output Formats**: JSON, Excel, and validation reports
- **Error Handling**: Robust error tracking and reporting
- **Progress Tracking**: Real-time processing status
- **GPU Acceleration**: CUDA support for faster processing

## Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, for faster processing)
- Required Python packages (see requirements.txt)
## Configuration

The system can be configured through various parameters:

- **OCR Settings**:
  - Minimum confidence threshold
  - GPU usage
  - Image enhancement parameters

- **Validation Rules**:
  - Date format validation
  - GST number format
  - Table structure validation

## Performance Optimization

- GPU acceleration is automatically enabled when available
- Batch processing for multiple files
- Optimized image preprocessing
- Efficient memory management

## Error Handling

The system provides comprehensive error handling:
- PDF conversion errors
- OCR extraction failures
- Data validation issues
- Output generation errors

All errors are logged and compiled in an error report.
## Project Structure

```
├── src/
│   └──input/                      # Input PDF files
│   └──output/                     # Generated output files
│       └── seals_and_signatures/  # Extracted seals and signatures
│   ├── logs/                      # Log files generated
│   ├── models/                    # Models used
│   ├── extraction/                # Data extraction modules
│   ├── preprocessing/             # Image preprocessing
│   ├── verification/              # Data verification
│   └── utils/                     # Utility functions
├── requirements.txt               # Python dependencies
├── setup.py                       # Setup file
└── README.md                      # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mdaashir/Yavar-Hackathon-2025-Mohamed-Aashir-S.git
cd Yavar-Hackathon-2025-Mohamed-Aashir-S
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place input invoice PDFs in the `src/input/` directory.

2. Run the processor:
```bash
python src/main.py
```

3. Check the `src/output/` directory for results:
- `{invoice_name}_data.json`: Extracted data in JSON format
- `{invoice_name}_data.xlsx`: Data in Excel format
- `{invoice_name}_verification.json`: Verification report
- `seals_and_signatures/`: Directory containing extracted seals/signatures

## Technical Approach

### 1. Image Preprocessing
- Denoising using bilateral filtering
- Contrast enhancement with CLAHE
- Adaptive thresholding
- Morphological operations for noise removal

### 2. OCR and Data Extraction
- EasyOCR for text recognition
- Custom table detection using contour analysis
- Cell structure analysis for table data
- Pattern matching for field extraction

### 3. Data Verification
- Confidence scoring for each extracted field
- Line item calculations validation
- Total amount verification
- GST number format validation
- Date format verification

## Dependencies

- OpenCV for image processing
- EasyOCR for text recognition
- PyTesseract for supplementary OCR
- Pandas for data handling
- PyPDF2 for PDF processing
- NumPy for numerical operations

## Output Format

### JSON Data Structure
```json
{
  "invoice_number": {"value": "INV001", "confidence": 0.95},
  "invoice_date": {"value": "2025-05-01", "confidence": 0.92},
  "supplier_gst_number": {"value": "29ABCDE1234F1Z5", "confidence": 0.89},
  "bill_to_gst_number": {"value": "29PQRST5678G1Z3", "confidence": 0.88},
  "po_number": {"value": "PO123", "confidence": 0.87},
  "shipping_address": {"value": "123 Business St, City", "confidence": 0.91},
  "seal_and_sign_present": {"value": true, "confidence": 0.85},
  "table_data": [
    {
      "description": "Item 1",
      "hsn_sac": "998391",
      "quantity": 2,
      "unit_price": 100.00,
      "total_amount": 200.00,
      "serial_number": "1"
    }
  ]
}
```

### Verification Report Structure
```json
{
  "field_verification": {
    "invoice_number": {"confidence": 0.95, "present": true},
    ...
  },
  "line_items_verification": [
    {
      "row": 1,
      "description_confidence": 0.93,
      "line_total_check": {
        "calculated_value": 200.00,
        "extracted_value": 200.00,
        "check_passed": true
      }
    }
  ],
  "total_calculations_verification": {
    "subtotal_check": {
      "calculated_value": 200.00,
      "extracted_value": 200.00,
      "check_passed": true
    }
  },
  "summary": {
    "all_fields_confident": true,
    "all_line_items_verified": true,
    "totals_verified": true,
    "issues": []
  }
}
```

## Acknowledgments

- EasyOCR for text recognition
- OpenCV for image processing
- PyTorch for GPU acceleration

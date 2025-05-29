# Invoice Data Extraction System

A robust system for extracting and validating data from invoice PDFs using advanced OCR and computer vision techniques.

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

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mdaashir/Yavar-Hackathon-2025-Mohamed-Aashir-S.git
cd Yavar-Hackathon-2025-Mohamed-Aashir-S
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your invoice PDFs in the `input` directory.

2. Run the processor:
```bash
python src/main.py
```

3. Check the `output` directory for results:
- `*_data.json`: Extracted data in JSON format
- `*_data.xlsx`: Formatted data in Excel
- `*_verification.json`: Data verification results
- `seals_and_signatures/`: Detected seals and signatures
- `error_report.txt`: Processing error details (if any)

## Directory Structure

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

## Output Formats

1. **JSON Output** (`*_data.json`):
```json
{
    "invoice_number": {"value": "INV001", "confidence": 0.95},
    "invoice_date": {"value": "2025-05-29", "confidence": 0.92},
    "supplier_gst_number": {"value": "29ABCDE1234F1Z5", "confidence": 0.88},
    ...
    "table_data": [
        {"item": "Product1", "quantity": 10, "price": 100.00},
        ...
    ]
}
```

2. **Excel Output** (`*_data.xlsx`):
- Sheet 1: General Information
- Sheet 2: Line Items

3. **Verification Report** (`*_verification.json`):
```json
{
    "status": "valid",
    "checks": [
        {"field": "invoice_number", "status": "valid"},
        {"field": "gst_number", "status": "valid"},
        ...
    ]
}
```

## Acknowledgments

- EasyOCR for text recognition
- OpenCV for image processing
- PyTorch for GPU acceleration

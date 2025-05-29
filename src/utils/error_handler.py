from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass
class ProcessingError:
    error_type: str
    message: str
    file_name: str
    timestamp: datetime
    stack_trace: Optional[str] = None


class InvoiceProcessingError(Exception):
    """Base exception for invoice processing errors"""
    pass


class PDFConversionError(InvoiceProcessingError):
    """Error during PDF to image conversion"""
    pass


class OCRError(InvoiceProcessingError):
    """Error during OCR processing"""
    pass


class ValidationError(InvoiceProcessingError):
    """Error during data validation"""
    pass


class OutputError(InvoiceProcessingError):
    """Error during output generation"""
    pass


class ErrorTracker:
    def __init__(self):
        self.errors: List[ProcessingError] = []

    def add_error(self, error_type: str, message: str, file_name: str,
                  stack_trace: Optional[str] = None):
        """Add an error to the tracking list"""
        error = ProcessingError(
            error_type=error_type,
            message=message,
            file_name=file_name,
            timestamp=datetime.now(),
            stack_trace=stack_trace
        )
        self.errors.append(error)

    def get_error_summary(self) -> dict:
        """Get summary of errors by type"""
        summary = {}
        for error in self.errors:
            if error.error_type not in summary:
                summary[error.error_type] = 0
            summary[error.error_type] += 1
        return summary

    def get_errors_for_file(self, file_name: str) -> List[ProcessingError]:
        """Get all errors for a specific file"""
        return [error for error in self.errors if error.file_name == file_name]

    def has_critical_errors(self) -> bool:
        """Check if there are any critical errors"""
        critical_types = {'PDFConversionError', 'OCRError'}
        return any(error.error_type in critical_types for error in self.errors)

    def clear(self):
        """Clear all tracked errors"""
        self.errors = []

    def generate_error_report(self) -> str:
        """Generate a detailed error report"""
        if not self.errors:
            return "No errors recorded."

        report = ["Error Report", "=" * 50]

        # Add summary
        summary = self.get_error_summary()
        report.append("\nError Summary:")
        for error_type, count in summary.items():
            report.append(f"  {error_type}: {count} occurrences")

        # Add detailed errors
        report.append("\nDetailed Errors:")
        for error in self.errors:
            report.extend([
                "\n" + "-" * 50,
                f"Type: {error.error_type}",
                f"File: {error.file_name}",
                f"Time: {error.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                f"Message: {error.message}"
            ])
            if error.stack_trace:
                report.extend(["Stack Trace:", error.stack_trace])

        return "\n".join(report)

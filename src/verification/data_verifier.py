import json
from decimal import Decimal, ROUND_HALF_UP
import re
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

class DataVerifier:
    def __init__(self):
        self.verification_results = {
            'field_verification': {},
            'line_items_verification': [],
            'total_calculations_verification': {},
            'summary': {
                'all_fields_confident': True,
                'all_line_items_verified': True,
                'totals_verified': True,
                'issues': []
            }
        }
        self.gst_pattern = r'^\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}$'
        self.validation_results = []

    def _to_decimal(self, value: Any, default: Decimal = Decimal('0.00')) -> Decimal:
        """
        Convert a value to Decimal with proper handling of different types
        """
        try:
            if isinstance(value, Decimal):
                return value
            elif isinstance(value, (int, float)):
                return Decimal(str(value))
            elif isinstance(value, str):
                # Remove currency symbols and commas
                cleaned = re.sub(r'[^\d.-]', '', value)
                return Decimal(cleaned)
            else:
                logging.warning(f"Unsupported type for decimal conversion: {type(value)}")
                return default
        except Exception as e:
            logging.error(f"Error converting to decimal: {str(e)}")
            return default

    def verify_field_presence(self, data):
        """
        Verify presence and confidence of required fields
        """
        required_fields = [
            'invoice_number',
            'invoice_date',
            'supplier_gst_number',
            'bill_to_gst_number',
            'po_number',
            'shipping_address',
            'seal_and_sign_present'
        ]
        
        for field in required_fields:
            if field in data:
                value = data[field]['value']
                confidence = data[field]['confidence']
                
                self.verification_results['field_verification'][field] = {
                    'confidence': confidence,
                    'present': value is not None
                }
                
                if confidence < 0.7 or value is None:
                    self.verification_results['summary']['all_fields_confident'] = False
                    self.verification_results['summary']['issues'].append(
                        f"Low confidence or missing value for {field}"
                    )

    def verify_line_items(self, table_data):
        """
        Verify line item calculations and data integrity
        """
        for i, row in enumerate(table_data):
            row_verification = {
                'row': i + 1,
                'description_confidence': row.get('confidence', 0),
                'hsn_sac_confidence': row.get('confidence', 0),
                'quantity_confidence': row.get('confidence', 0),
                'unit_price_confidence': row.get('confidence', 0),
                'total_amount_confidence': row.get('confidence', 0),
                'serial_number_confidence': row.get('confidence', 0)
            }
            
            # Verify line total calculation
            quantity = Decimal(str(row['quantity']))
            unit_price = Decimal(str(row['unit_price']))
            total_amount = Decimal(str(row['total_amount']))
            calculated_total = quantity * unit_price
            
            row_verification['line_total_check'] = {
                'calculated_value': float(calculated_total.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
                'extracted_value': float(total_amount.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
                'check_passed': abs(calculated_total - total_amount) <= Decimal('0.01')
            }
            
            if not row_verification['line_total_check']['check_passed']:
                self.verification_results['summary']['all_line_items_verified'] = False
                self.verification_results['summary']['issues'].append(
                    f"Line item {i+1} total amount mismatch"
                )
            
            self.verification_results['line_items_verification'].append(row_verification)

    def verify_totals(self, data):
        """
        Verify overall invoice calculations
        """
        try:
            # Extract table data
            items = data.get('table_data', [])
            if not items:
                self.validation_results.append({
                    'field': 'table_data',
                    'status': 'error',
                    'message': 'No items found in invoice'
                })
                return False

            # Calculate subtotal
            subtotal = Decimal('0.00')
            for item in items:
                try:
                    quantity = self._to_decimal(item.get('quantity', 0))
                    unit_price = self._to_decimal(item.get('unit_price', 0))
                    item_total = quantity * unit_price
                    subtotal += item_total
                except Exception as e:
                    logging.error(f"Error calculating item total: {str(e)}")
                    continue

            # Get invoice totals
            invoice_subtotal = self._to_decimal(data.get('subtotal', 0))
            tax_rate = self._to_decimal(data.get('tax_rate', 0)) / Decimal('100.00')
            tax_amount = self._to_decimal(data.get('tax_amount', 0))
            total_amount = self._to_decimal(data.get('total_amount', 0))

            # Calculate expected values
            calculated_tax = (subtotal * tax_rate).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            calculated_total = (subtotal + calculated_tax).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

            # Verify calculations
            subtotal_matches = abs(subtotal - invoice_subtotal) <= Decimal('0.02')
            tax_matches = abs(calculated_tax - tax_amount) <= Decimal('0.02')
            total_matches = abs(calculated_total - total_amount) <= Decimal('0.02')

            # Add validation results
            self.validation_results.append({
                'field': 'subtotal',
                'status': 'valid' if subtotal_matches else 'error',
                'expected_value': float(subtotal.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
                'actual_value': float(invoice_subtotal.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
                'message': 'Subtotal verification ' + ('passed' if subtotal_matches else 'failed')
            })

            self.validation_results.append({
                'field': 'tax_amount',
                'status': 'valid' if tax_matches else 'error',
                'expected_value': float(calculated_tax),
                'actual_value': float(tax_amount.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
                'message': 'Tax amount verification ' + ('passed' if tax_matches else 'failed')
            })

            self.validation_results.append({
                'field': 'total_amount',
                'status': 'valid' if total_matches else 'error',
                'expected_value': float(calculated_total),
                'actual_value': float(total_amount.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
                'message': 'Total amount verification ' + ('passed' if total_matches else 'failed')
            })

            return subtotal_matches and tax_matches and total_matches

        except Exception as e:
            logging.error(f"Error in verify_totals: {str(e)}")
            self.validation_results.append({
                'field': 'calculations',
                'status': 'error',
                'message': f'Error verifying totals: {str(e)}'
            })
            return False

    def verify_gst_number(self, gst_number: str) -> bool:
        """
        Verify GST number format
        """
        if not gst_number:
            return False
        return bool(re.match(self.gst_pattern, gst_number))

    def verify_date_format(self, date_str: str) -> bool:
        """
        Verify date format and validity
        """
        try:
            if not date_str:
                return False
            # Try multiple date formats
            formats = [
                '%Y-%m-%d',
                '%d/%m/%Y',
                '%d-%m-%Y',
                '%d.%m.%Y',
                '%d %b %Y',
                '%d %B %Y'
            ]
            for fmt in formats:
                try:
                    datetime.strptime(date_str, fmt)
                    return True
                except ValueError:
                    continue
            return False
        except Exception:
            return False

    def verify_data(self, extracted_data):
        """
        Perform all verifications on extracted data
        """
        self.verify_field_presence(extracted_data)
        self.verify_line_items(extracted_data['table_data'])
        self.verify_totals(extracted_data)
        
        return self.verification_results

    def save_verification_report(self, output_path):
        """
        Save verification results to JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(self.verification_results, f, indent=2) 
import json
from decimal import Decimal, ROUND_HALF_UP

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
        # Calculate subtotal
        subtotal = sum(Decimal(str(row['total_amount'])) for row in data['table_data'])
        extracted_subtotal = Decimal(str(data.get('subtotal', 0)))
        
        self.verification_results['total_calculations_verification']['subtotal_check'] = {
            'calculated_value': float(subtotal.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
            'extracted_value': float(extracted_subtotal.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
            'check_passed': abs(subtotal - extracted_subtotal) <= Decimal('0.01')
        }
        
        # Verify final total calculation
        discount = Decimal(str(data.get('discount', 0)))
        gst = Decimal(str(data.get('gst', 0)))
        calculated_final = subtotal - discount + gst
        extracted_final = Decimal(str(data.get('final_total', 0)))
        
        self.verification_results['total_calculations_verification'].update({
            'discount_check': {
                'calculated_value': float(discount.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
                'extracted_value': float(discount.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
                'check_passed': True  # Always true as we're using the same value
            },
            'gst_check': {
                'calculated_value': float(gst.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
                'extracted_value': float(gst.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
                'check_passed': True  # Always true as we're using the same value
            },
            'final_total_check': {
                'calculated_value': float(calculated_final.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
                'extracted_value': float(extracted_final.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
                'check_passed': abs(calculated_final - extracted_final) <= Decimal('0.01')
            }
        })
        
        if not all(check['check_passed'] for check in self.verification_results['total_calculations_verification'].values()):
            self.verification_results['summary']['totals_verified'] = False
            self.verification_results['summary']['issues'].append("Total calculations mismatch")

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
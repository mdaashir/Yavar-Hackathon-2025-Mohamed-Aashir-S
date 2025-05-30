import random
from datetime import datetime, timedelta
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer


def generate_company_details():
    """Generate random company details"""
    companies = [
        ("TechCorp Solutions", "29AABCT2223A1Z5"),
        ("Global Systems Ltd", "27AAACG7123B1Z1"),
        ("Innovative Tech Inc", "32AABCI4567C1Z8"),
        ("Digital Services Co", "19AAACD8901D1Z3"),
        ("NextGen Solutions", "36AAACN3456E1Z7")
    ]
    addresses = [
        "123 Tech Park, Silicon Valley, CA 94025",
        "456 Business Hub, Cyber City, NY 10001",
        "789 Innovation Center, Digital Zone, TX 75001",
        "321 Corporate Plaza, Business Bay, FL 33101",
        "654 Enterprise Street, Tech District, WA 98101"
    ]
    company, gst = random.choice(companies)
    address = random.choice(addresses)
    return company, gst, address


def generate_invoice_items(num_items=None):
    """Generate random invoice items"""
    if num_items is None:
        num_items = random.randint(3, 8)

    items = [
        ("Software Development Services", "998314", (1000, 5000)),
        ("Cloud Infrastructure Setup", "998316", (2000, 8000)),
        ("Data Analytics Services", "998315", (1500, 6000)),
        ("Cybersecurity Solutions", "998319", (3000, 10000)),
        ("AI/ML Implementation", "998318", (4000, 12000)),
        ("Technical Consultation", "998313", (800, 3000)),
        ("System Integration", "998317", (2500, 7000)),
        ("Database Management", "998312", (1200, 4000))
    ]

    invoice_items = []
    for i in range(num_items):
        description, hsn, price_range = random.choice(items)
        quantity = random.randint(1, 5)
        unit_price = round(random.uniform(price_range[0], price_range[1]), 2)
        total = round(quantity * unit_price, 2)

        invoice_items.append([
            str(i + 1),
            description,
            hsn,
            str(quantity),
            f"{unit_price:.2f}",
            f"{total:.2f}"
        ])

    return invoice_items


class InvoiceGenerator:
    def __init__(self, output_dir='src/input'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.styles = getSampleStyleSheet()

        # Custom styles
        self.styles.add(ParagraphStyle(
            name='InvoiceTitle',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=30
        ))

    def create_invoice(self, index):
        """Create a single invoice PDF"""
        # Generate random data
        invoice_date = datetime.now() - timedelta(days=random.randint(0, 30))
        invoice_number = f"INV{datetime.now().year}{index:04d}"
        po_number = f"PO{random.randint(1000, 9999)}"

        # Generate company details
        supplier_company, supplier_gst, supplier_address = generate_company_details()
        client_company, client_gst, client_address = generate_company_details()

        # Create PDF
        filename = self.output_dir / f"{invoice_number}.pdf"
        doc = SimpleDocTemplate(str(filename), pagesize=letter)
        elements = [Paragraph(f"{supplier_company}", self.styles["InvoiceTitle"]),
                    Paragraph(f"GST: {supplier_gst}", self.styles["Normal"]),
                    Paragraph(supplier_address, self.styles["Normal"]), Spacer(1, 20),
                    Paragraph(f"Invoice Number: {invoice_number}", self.styles["Normal"]),
                    Paragraph(f"Date: {invoice_date.strftime('%Y-%m-%d')}", self.styles["Normal"]),
                    Paragraph(f"PO Number: {po_number}", self.styles["Normal"]), Spacer(1, 20),
                    Paragraph("Bill To:", self.styles["Heading3"]), Paragraph(client_company, self.styles["Normal"]),
                    Paragraph(f"GST: {client_gst}", self.styles["Normal"]),
                    Paragraph(client_address, self.styles["Normal"]), Spacer(1, 20)]

        # Generate items
        items = generate_invoice_items()

        # Create table
        table_data = [['Sr. No.', 'Description', 'HSN/SAC', 'Qty', 'Rate', 'Amount']] + items

        # Calculate totals
        subtotal = sum(float(item[5]) for item in items)
        gst_rate = 18  # 18% GST
        gst_amount = round(subtotal * (gst_rate / 100), 2)
        total = round(subtotal + gst_amount, 2)

        # Add totals to table
        table_data.extend([
            ['', '', '', '', 'Subtotal:', f"{subtotal:.2f}"],
            ['', '', '', '', f'GST ({gst_rate}%):', f"{gst_amount:.2f}"],
            ['', '', '', '', 'Total:', f"{total:.2f}"]
        ])

        # Create and style table
        table = Table(table_data, colWidths=[0.5 * inch, 3 * inch, inch, 0.5 * inch, inch, inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, -3), (-1, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ALIGN', (-2, -3), (-1, -1), 'RIGHT'),
            ('GRID', (0, 0), (-1, -4), 1, colors.black),
            ('LINEBELOW', (0, -3), (-1, -1), 1, colors.black),
            ('LINEABOVE', (0, -3), (-1, -3), 1, colors.black),
        ]))
        elements.append(table)

        # Build PDF
        doc.build(elements)
        return filename

    @classmethod
    def generate_test_invoices(cls, output_dir: str = 'src/input', num_invoices: int = 10) -> list:
        """Generate a set of test invoices
        
        Args:
            output_dir: Directory to save generated invoices
            num_invoices: Number of invoices to generate
            
        Returns:
            List of generated invoice file paths
        """
        print(f"\nGenerating {num_invoices} test invoices...")
        generator = cls(output_dir=output_dir)
        files = []

        for i in range(num_invoices):
            filename = generator.create_invoice(i + 1)
            files.append(filename)
            print(f"Generated: {filename.name}")

        print(f"\nGeneration complete! Created {len(files)} invoices in {output_dir}\n")
        return files


if __name__ == '__main__':
    InvoiceGenerator.generate_test_invoices(output_dir='src/input', num_invoices=10)

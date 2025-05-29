import os
from typing import List

import pdfplumber
from PIL import Image


def save_debug_image(image: Image.Image, output_path: str):
    """
    Save image for debugging purposes
    """
    try:
        image.save(output_path, 'PNG')
        return True
    except Exception:
        return False


class PDFConverter:
    def __init__(self, dpi: int = 300):
        self.dpi = dpi

    def pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """
        Convert PDF to list of PIL Images using pdfplumber
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of PIL Image objects
        """
        images = []

        try:
            # Open PDF document
            with pdfplumber.open(pdf_path) as pdf:
                # Convert each page to image
                for page in pdf.pages:
                    # Get page as image
                    img = page.to_image(resolution=self.dpi)
                    # Convert to PIL Image
                    pil_image = img.original
                    images.append(pil_image)

            return images

        except Exception as e:
            raise RuntimeError(f"Error converting PDF to images: {str(e)}")

    def enhance_image_quality(self, image: Image.Image) -> Image.Image:
        """
        Enhance image quality if needed
        """
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Optionally enhance resolution for better OCR
        if self.dpi < 300:  # If input DPI is low
            width, height = image.size
            new_width = int(width * (300 / self.dpi))
            new_height = int(height * (300 / self.dpi))
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        return image

    @staticmethod
    def save_images(images, output_dir, base_filename):
        """
        Save the list of images to the output directory
        
        Args:
            images (list): List of PIL Image objects
            output_dir (str): Directory to save the images
            base_filename (str): Base name for the output files
            
        Returns:
            list: List of paths to saved images
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        image_paths = []
        for i, image in enumerate(images):
            output_path = os.path.join(output_dir, f"{base_filename}_page_{i + 1}.png")
            image.save(output_path, "PNG")
            image_paths.append(output_path)

        return image_paths

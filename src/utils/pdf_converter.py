from pdf2image import convert_from_path
import os

class PDFConverter:
    @staticmethod
    def pdf_to_images(pdf_path, dpi=300):
        """
        Convert PDF to list of images
        
        Args:
            pdf_path (str): Path to the PDF file
            dpi (int): DPI for the output images
            
        Returns:
            list: List of PIL Image objects
        """
        try:
            # Convert PDF to images
            images = convert_from_path(pdf_path, dpi=dpi)
            return images
        except Exception as e:
            print(f"Error converting PDF {pdf_path}: {str(e)}")
            return []

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
            output_path = os.path.join(output_dir, f"{base_filename}_page_{i+1}.png")
            image.save(output_path, "PNG")
            image_paths.append(output_path)
            
        return image_paths 
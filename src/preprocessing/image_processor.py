import cv2
import numpy as np
from .advanced_preprocessor import AdvancedPreprocessor

class ImagePreprocessor:
    def __init__(self, debug_mode: bool = False):
        self.advanced_preprocessor = AdvancedPreprocessor()
        self.advanced_preprocessor.set_debug_mode(debug_mode)
        self.debug_mode = debug_mode

    def process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Process image using advanced preprocessing techniques
        """
        # Apply advanced preprocessing
        processed_image = self.advanced_preprocessor.enhance_image(image)
        
        return processed_image

    def detect_table_regions(self, image: np.ndarray) -> list:
        """
        Detect table regions in the image
        """
        return self.advanced_preprocessor.detect_table_regions(image)

    def get_debug_images(self) -> dict:
        """
        Get intermediate processing results if debug mode is enabled
        """
        return self.advanced_preprocessor.get_debug_images()

    def set_debug_mode(self, enabled: bool):
        """
        Enable/disable debug mode
        """
        self.debug_mode = enabled
        self.advanced_preprocessor.set_debug_mode(enabled) 
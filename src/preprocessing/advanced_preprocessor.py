import cv2
import numpy as np
from skimage import exposure, filters, morphology


def _apply_denoising(image: np.ndarray) -> np.ndarray:
    """
    Apply multiple denoising techniques
    """
    # Non-local means denoising
    denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

    # Bilateral filter to preserve edges
    denoised = cv2.bilateralFilter(denoised, 9, 75, 75)

    return denoised


def _enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Enhance image contrast using multiple techniques
    """
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)

    # Apply gamma correction
    gamma = 1.2
    enhanced = exposure.adjust_gamma(enhanced, gamma)

    # Stretch contrast to full range
    p2, p98 = np.percentile(enhanced, (2, 98))
    enhanced = exposure.rescale_intensity(enhanced, in_range=(p2, p98))

    return enhanced.astype(np.uint8)


def _deskew_image(image: np.ndarray) -> np.ndarray:
    """
    Detect and correct image skew
    """
    # Detect edges
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # Use Hough transform to detect lines
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    if lines is not None:
        # Calculate the dominant angle
        angles = []
        for rho, theta in lines[:min(10, len(lines))].reshape(-1, 2):
            angle = np.degrees(theta) - 90
            if -45 <= angle <= 45:  # Filter out vertical lines
                angles.append(angle)

        if angles:
            # Use median angle to avoid outliers
            median_angle = np.median(angles)

            if abs(median_angle) > 0.5:  # Only correct if skew is significant
                # Rotate image
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                m = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                image = cv2.warpAffine(image, m, (w, h),
                                       flags=cv2.INTER_CUBIC,
                                       borderMode=cv2.BORDER_REPLICATE)

    return image


def _binarize_image(image: np.ndarray) -> np.ndarray:
    """
    Convert image to binary using advanced thresholding
    """
    # Try different thresholding methods and combine results

    # Otsu's thresholding
    _, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Adaptive Gaussian thresholding
    adaptive = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

    # Niblack's thresholding
    window_size = 25
    k = 0.8
    thresh_niblack = filters.threshold_niblack(image, window_size=window_size, k=k)
    niblack = (image > thresh_niblack).astype(np.uint8) * 255

    # Combine results using weighted average
    binary = cv2.addWeighted(otsu, 0.4, adaptive, 0.4, 0)
    binary = cv2.addWeighted(binary, 0.8, niblack, 0.2, 0)

    return binary


def _clean_image(image: np.ndarray) -> np.ndarray:
    """
    Clean binary image by removing noise and smoothing edges
    """
    # Remove small noise
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # Fill small holes
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    # Remove isolated pixels
    cleaned = morphology.remove_small_objects(cleaned.astype(bool), min_size=5).astype(np.uint8) * 255

    return cleaned


def _detect_lines(image: np.ndarray, direction: str = 'horizontal') -> np.ndarray:
    """
    Detect lines in specified direction
    """
    # Calculate minimum line length based on image size
    if direction == 'horizontal':
        min_length = image.shape[1] // 20
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_length, 1))
    else:
        min_length = image.shape[0] // 20
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_length))

    # Detect lines using morphological operations
    lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=2)

    return lines


class AdvancedPreprocessor:
    def __init__(self):
        self.debug_mode = False
        self.debug_images = {}

    def set_debug_mode(self, enabled: bool):
        """Enable/disable debug mode to store intermediate results"""
        self.debug_mode = enabled
        if not enabled:
            self.debug_images = {}

    def get_debug_images(self) -> dict:
        """Get intermediate processing results if debug mode is enabled"""
        return self.debug_images

    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply multiple enhancement techniques to improve image quality
        """
        # Store original
        if self.debug_mode:
            self.debug_images['original'] = image.copy()

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply denoising
        denoised = _apply_denoising(gray)
        if self.debug_mode:
            self.debug_images['denoised'] = denoised

        # Enhance contrast
        enhanced = _enhance_contrast(denoised)
        if self.debug_mode:
            self.debug_images['contrast_enhanced'] = enhanced

        # Apply deskewing if needed
        deskewed = _deskew_image(enhanced)
        if self.debug_mode:
            self.debug_images['deskewed'] = deskewed

        # Binarize image
        binary = _binarize_image(deskewed)
        if self.debug_mode:
            self.debug_images['binary'] = binary

        # Remove noise and smooth edges
        cleaned = _clean_image(binary)
        if self.debug_mode:
            self.debug_images['cleaned'] = cleaned

        return cleaned

    def detect_table_regions(self, image: np.ndarray) -> list:
        """
        Detect potential table regions in the image
        """
        # Create a copy for visualization
        vis_image = image.copy()
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)

        # Detect lines
        horizontal = _detect_lines(image, direction='horizontal')
        vertical = _detect_lines(image, direction='vertical')

        # Combine lines
        table_mask = cv2.addWeighted(horizontal, 0.5, vertical, 0.5, 0.0)

        # Find table regions
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter and sort regions by size
        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(cnt)
                regions.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'contour': cnt
                })

        # Sort by area (largest first)
        regions.sort(key=lambda r: r['area'], reverse=True)

        if self.debug_mode:
            debug_vis = vis_image.copy()
            for region in regions:
                x, y, w, h = region['bbox']
                cv2.rectangle(debug_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            self.debug_images['table_regions'] = debug_vis

        return regions

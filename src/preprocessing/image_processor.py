import cv2
import numpy as np
from skimage.filters import threshold_local
import imutils

class ImagePreprocessor:
    @staticmethod
    def denoise_image(image):
        """
        Remove noise from the image using Non-local Means Denoising
        """
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    @staticmethod
    def binarize(image):
        """
        Convert image to binary using adaptive thresholding
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        T = threshold_local(gray, 11, offset=10, method="gaussian")
        return (gray > T).astype("uint8") * 255

    @staticmethod
    def deskew(image):
        """
        Correct skew in the image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
        
        if lines is not None:
            angle = 0
            for rho, theta in lines[0]:
                angle = np.degrees(theta) - 90
            
            if abs(angle) > 0.5:
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(image, M, (w, h),
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)
        return image

    @staticmethod
    def enhance_resolution(image):
        """
        Enhance image resolution using super resolution
        """
        return cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def process_image(image):
        """
        Apply all preprocessing steps
        """
        # Enhance resolution
        image = ImagePreprocessor.enhance_resolution(image)
        
        # Denoise
        image = ImagePreprocessor.denoise_image(image)
        
        # Deskew
        image = ImagePreprocessor.deskew(image)
        
        # Binarize
        image = ImagePreprocessor.binarize(image)
        
        return image 
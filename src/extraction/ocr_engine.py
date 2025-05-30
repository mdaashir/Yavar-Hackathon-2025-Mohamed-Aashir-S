import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
import easyocr
import numpy as np
import torch
from numpy import ndarray
from torch import Tensor


def _get_device() -> torch.device:
    """
    Determine the best available device for computation
    """
    if torch.cuda.is_available():
        # Get the GPU with the most free memory
        device_id = 0
        if torch.cuda.device_count() > 1:
            free_memory = []
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                free_memory.append(torch.cuda.memory_reserved(i))
            device_id = free_memory.index(min(free_memory))

        device = torch.device(f'cuda:{device_id}')

        # Log GPU information
        logging.info(f"Using GPU: {torch.cuda.get_device_name(device_id)}")
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(device_id).total_memory / 1024 ** 3:.2f} GB")

        # Optimize CUDA settings
        torch.cuda.set_device(device_id)
        torch.backends.cudnn.fastest = True
        torch.backends.cudnn.benchmark = True

        return device
    elif torch.backends.mps.is_available():
        logging.info("Using Apple M1/M2 GPU acceleration")
        return torch.device('mps')
    else:
        logging.info("No GPU available, using CPU")
        return torch.device('cpu')


def _detect_lines(image: np.ndarray, direction: str = 'horizontal') -> np.ndarray:
    """
    Detect lines in specified direction with enhanced parameters
    """
    # Calculate minimum line length based on image size
    if direction == 'horizontal':
        min_length = image.shape[1] // 15
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_length, 1))
    else:
        min_length = image.shape[0] // 15
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_length))

    # Apply morphological operations
    eroded = cv2.erode(image, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)

    return dilated


class OCREngine:
    def __init__(self):
        # Check CUDA availability
        self.device = _get_device()
        logging.info(f"OCR Engine initialized using device: {self.device}")

        # Create models directory inside src if it doesn't exist
        src_dir = Path(__file__).parent.parent
        models_dir = src_dir / 'models'
        models_dir.mkdir(exist_ok=True)

        # Initialize EasyOCR with GPU settings
        gpu = self.device.type == 'cuda'
        if gpu:
            logging.info(f"Using GPU acceleration with CUDA. Device: {torch.cuda.get_device_name()}")
            # Set CUDA device options for better performance
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        else:
            logging.info("GPU not available. Using CPU for OCR processing.")

        # Initialize EasyOCR with appropriate device settings
        self.reader = easyocr.Reader(
            ['en'],
            gpu=gpu,
            model_storage_directory=str(models_dir),
            download_enabled=True,
            detector=True,
            recognizer=True,
            quantize=False,  # Disable quantization for better accuracy
            cudnn_benchmark=True
        )

        self.min_confidence = 0.6
        self.dpi = 300  # Target DPI for image enhancement

    def _prepare_batch(self, image: np.ndarray) -> Tensor | ndarray:
        """
        Prepare image batch for GPU processing
        """
        if self.device.type == 'cuda':
            # Convert image to tensor and move to GPU
            tensor = torch.from_numpy(image).to(self.device)
            if len(tensor.shape) == 2:
                tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            elif len(tensor.shape) == 3:
                tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # NHWC to NCHW
            return tensor.float() / 255.0
        return image

    def extract_text(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Extract text from image using EasyOCR with enhanced preprocessing
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (extracted text, confidence score)
        """
        if image is None or image.size == 0:
            logging.warning("Received empty or None image")
            return "", 0.0

        try:
            # Enhance image quality
            enhanced_image = self._enhance_image_quality(image)

            # Ensure image is in RGB format
            if len(enhanced_image.shape) == 2:
                enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2RGB)
            elif enhanced_image.shape[2] == 4:
                enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGRA2RGB)

            # Verify image is valid
            if enhanced_image is None or enhanced_image.size == 0:
                logging.error("Image enhancement failed")
                return "", 0.0

            # Perform OCR with paragraph detection
            try:
                results = self.reader.readtext(
                    enhanced_image,
                    paragraph=True,  # Enable paragraph detection
                    detail=1,  # Get detailed results
                    batch_size=4,  # Increase batch size for faster processing
                    width_ths=0.7,  # Width threshold for text line merging
                    height_ths=0.7,  # Height threshold for text line merging
                    mag_ratio=1.5  # Magnification ratio for better detection
                )
            except Exception as e:
                logging.error(f"EasyOCR readtext failed: {str(e)}")
                # Try again with basic settings
                results = self.reader.readtext(
                    enhanced_image,
                    paragraph=False,
                    detail=1
                )

            if not results:
                logging.warning("No text detected in image")
                return "", 0.0

            # Combine results with intelligent text grouping
            text_blocks = []
            confidences = []

            current_line = []
            current_y = None
            y_threshold = 10  # Pixels threshold for same line detection

            # Sort detections by position, with safe access to coordinates
            def get_sort_key(x):
                try:
                    if len(x) >= 1 and x[0] and len(x[0]) >= 1:
                        return x[0][0][1], x[0][0][0]
                except (IndexError, TypeError):
                    return float('inf'), float('inf')
                return float('inf'), float('inf')

            sorted_results = sorted(results, key=get_sort_key)

            for detection in sorted_results:
                try:
                    if not detection or len(detection) < 2:
                        continue

                    bbox = detection[0]
                    if not bbox or len(bbox) < 1:
                        continue

                    text = detection[1]
                    if not isinstance(text, str):
                        continue

                    confidence = detection[2] if len(detection) >= 3 else 0.8

                    if confidence > self.min_confidence:
                        try:
                            y_coord = bbox[0][1]  # Top-left y coordinate
                        except (IndexError, TypeError):
                            continue

                        # Check if this text belongs to the current line
                        if current_y is None or abs(y_coord - current_y) <= y_threshold:
                            current_line.append(text)
                            current_y = y_coord
                        else:
                            # New line detected, add current line to blocks
                            if current_line:
                                text_blocks.append(' '.join(current_line))
                            current_line = [text]
                            current_y = y_coord

                        confidences.append(confidence)
                except (IndexError, TypeError, AttributeError) as e:
                    logging.debug(f"Skipping malformed detection: {str(e)}")
                    continue

            # Add last line if exists
            if current_line:
                text_blocks.append(' '.join(current_line))

            # Return combined text and average confidence
            full_text = '\n'.join(text_blocks) if text_blocks else ""
            avg_confidence = np.mean(confidences) if confidences else 0.0

            if not full_text.strip():
                logging.warning("No valid text extracted after processing")

            return full_text, avg_confidence

        except Exception as e:
            logging.error(f"Error in extract_text: {str(e)}")
            return "", 0.0

    def extract_structured_text(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract text with enhanced position information and layout analysis
        """
        if image is None or image.size == 0:
            logging.warning("Received empty or None image")
            return []

        try:
            # Enhance image
            enhanced_image = self._enhance_image_quality(image)

            # Verify image is valid
            if enhanced_image is None or enhanced_image.size == 0:
                logging.error("Image enhancement failed")
                return []

            # Perform OCR with layout analysis
            try:
                results = self.reader.readtext(
                    enhanced_image,
                    paragraph=True,
                    detail=1,
                    batch_size=4,
                    width_ths=0.7,
                    height_ths=0.7,
                    mag_ratio=1.5
                )
            except Exception as e:
                logging.error(f"EasyOCR readtext failed: {str(e)}")
                # Try again with basic settings
                results = self.reader.readtext(
                    enhanced_image,
                    paragraph=False,
                    detail=1
                )

            if not results:
                logging.warning("No text detected in image")
                return []

            structured_results = []
            for detection in results:
                try:
                    if not detection or len(detection) < 2:
                        continue

                    bbox = detection[0]
                    if not bbox or len(bbox) < 4:  # Need 4 points for bbox
                        continue

                    text = detection[1]
                    if not isinstance(text, str):
                        continue

                    confidence = detection[2] if len(detection) >= 3 else 0.8

                    if confidence > self.min_confidence:
                        try:
                            # Convert bbox points to x, y, width, height format
                            points = np.array(bbox)
                            x = int(min(points[:, 0]))
                            y = int(min(points[:, 1]))
                            w = int(max(points[:, 0]) - x)
                            h = int(max(points[:, 1]) - y)

                            # Calculate additional layout information
                            center_x = x + w / 2
                            center_y = y + h / 2
                            aspect_ratio = w / h if h != 0 else 0
                            area = w * h

                            structured_results.append({
                                'text': text,
                                'confidence': confidence,
                                'x': x,
                                'y': y,
                                'width': w,
                                'height': h,
                                'bbox': bbox,
                                'center': (center_x, center_y),
                                'aspect_ratio': aspect_ratio,
                                'area': area,
                                'is_title': h > 30 or aspect_ratio > 3,
                                'alignment': 'left' if x < enhanced_image.shape[1] / 3 else
                                'right' if x > 2 * enhanced_image.shape[1] / 3 else 'center'
                            })
                        except (IndexError, TypeError, AttributeError) as e:
                            logging.debug(f"Error processing detection bbox: {str(e)}")
                            continue
                except Exception as e:
                    logging.debug(f"Skipping malformed detection: {str(e)}")
                    continue

            return structured_results

        except Exception as e:
            logging.error(f"Error in extract_structured_text: {str(e)}")
            return []

    def extract_table_cells(self, table_region: Tuple[bool, np.ndarray, List[List[Any]]]) -> List[
        Dict[str, Any]]:
        """
        Extract text from table cells with enhanced cell detection
        
        Args:
            table_region: Tuple of (found, region_image, cells) from table detection
            
        Returns:
            List of cell data with text and position information
        """
        table_found, region_image, detected_cells = table_region
        if not table_found or region_image is None:
            logging.warning("No valid table region provided")
            return []

        cells = []

        try:
            # Process each detected cell
            for row in detected_cells:
                for cell in row:
                    # Extract cell ROI from the region image
                    cell_roi = region_image[cell.y:cell.y + cell.height, cell.x:cell.x + cell.width]

                    # Skip invalid cells
                    if cell_roi is None or cell_roi.size == 0:
                        continue

                    # Enhance cell image
                    enhanced_roi = self._enhance_image_quality(cell_roi)

                    # Perform OCR on cell
                    try:
                        results = self.reader.readtext(
                            enhanced_roi,
                            paragraph=False,
                            detail=1,
                            width_ths=0.7,
                            height_ths=0.7
                        )
                    except Exception as e:
                        logging.debug(f"OCR failed for cell: {str(e)}")
                        continue

                    # Combine cell text
                    cell_text = ''
                    cell_confidence = 0.0

                    for detection in results:
                        try:
                            if len(detection) == 3:
                                _, text, confidence = detection
                            elif len(detection) == 2:
                                _, text = detection
                                confidence = 0.8  # Default confidence for 2-element tuples
                            else:
                                continue  # Skip invalid detections

                            if confidence > self.min_confidence:
                                cell_text += ' ' + text
                                cell_confidence = max(cell_confidence, confidence)
                        except (IndexError, TypeError):
                            continue  # Skip malformed detections

                    if cell_text.strip():
                        cells.append({
                            'text': cell_text.strip(),
                            'confidence': cell_confidence,
                            'x': cell.x,
                            'y': cell.y,
                            'width': cell.width,
                            'height': cell.height,
                            'area': cell.width * cell.height
                        })

            return cells

        except Exception as e:
            logging.error(f"Error processing table cells: {str(e)}")
            return []

    def _enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """
        Enhanced image preprocessing for better OCR accuracy
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Calculate current DPI and resize if needed
            if self.dpi > 300:
                scale_factor = self.dpi / 300
                gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor,
                                  interpolation=cv2.INTER_CUBIC)

            # Apply CLAHE for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            contrast_enhanced = clahe.apply(gray)

            # Denoise while preserving edges
            denoised = cv2.fastNlMeansDenoising(contrast_enhanced, None, 10, 7, 21)

            # Apply bilateral filter for edge preservation
            bilateral = cv2.bilateralFilter(denoised, 9, 75, 75)

            # Adaptive thresholding with optimized parameters
            binary = cv2.adaptiveThreshold(
                bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            # Remove small noise
            kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

            # Fill small holes
            kernel = np.ones((3, 3), np.uint8)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

            return cleaned
        except Exception:
            # If any enhancement step fails, return original image
            return image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

"""
Image Preprocessing Module

This module handles the initial preprocessing steps required before SIFT feature extraction.
It includes grayscale conversion and optional noise reduction.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class ImagePreprocessor:
    """
    Handles image preprocessing for optimal SIFT feature extraction.

    The preprocessing pipeline includes:
    1. Grayscale conversion (mandatory for SIFT)
    2. Noise reduction (optional but recommended)
    """

    def __init__(
        self,
        apply_gaussian_blur: bool = True,
        blur_kernel_size: int = 3,
        blur_sigma: float = 0.5,
    ):
        """
        Initialize the preprocessor with optional noise reduction parameters.

        Args:
            apply_gaussian_blur: Whether to apply Gaussian blur for noise reduction
            blur_kernel_size: Size of the Gaussian kernel (should be odd)
            blur_sigma: Standard deviation for Gaussian kernel
        """
        self.apply_gaussian_blur = apply_gaussian_blur
        self.blur_kernel_size = (
            blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
        )
        self.blur_sigma = blur_sigma

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Apply complete preprocessing pipeline to an input image.

        Args:
            image: Input image (BGR or RGB color image)

        Returns:
            Preprocessed grayscale image ready for SIFT extraction
        """
        # Step 1: Convert to grayscale
        gray_image = self._convert_to_grayscale(image)

        # Step 2: Optional noise reduction
        if self.apply_gaussian_blur:
            gray_image = self._apply_gaussian_blur(gray_image)

        return gray_image

    def _convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert color image to grayscale.

        SIFT operates on single-channel luminance information rather than
        multi-channel color data. The structural features, edges, and textures
        that SIFT relies on are captured in intensity variations.

        Args:
            image: Color image (BGR format from OpenCV or RGB)

        Returns:
            Grayscale image
        """
        if len(image.shape) == 3:
            # Assume BGR format (OpenCV default)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            # Already grayscale
            gray = image.copy()

        return gray

    def _apply_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """
        Apply gentle Gaussian blur for noise reduction.

        Digital images are affected by sensor noise and compression artifacts.
        While SIFT has inherent robustness to noise, high-frequency noise can
        introduce spurious keypoints. A gentle low-pass filter smooths minor
        noise artifacts while preserving important features.

        Args:
            image: Grayscale image

        Returns:
            Slightly blurred image with reduced noise
        """
        blurred = cv2.GaussianBlur(
            image, (self.blur_kernel_size, self.blur_kernel_size), self.blur_sigma
        )
        return blurred

    def load_and_preprocess(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load an image from file and apply preprocessing.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (original_image, preprocessed_image)
        """
        # Load image
        original = cv2.imread(image_path)
        if original is None:
            raise ValueError(f"Could not load image from {image_path}")

        # Preprocess
        preprocessed = self.preprocess(original)

        return original, preprocessed

    def validate_image(self, image: np.ndarray) -> bool:
        """
        Validate that the image is suitable for processing.

        Args:
            image: Input image array

        Returns:
            True if image is valid for processing
        """
        if image is None:
            return False

        if len(image.shape) < 2:
            return False

        if image.shape[0] < 50 or image.shape[1] < 50:
            return False  # Too small for meaningful SIFT features

        return True

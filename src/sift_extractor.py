"""
SIFT Feature Extractor Module

This module implements SIFT (Scale-Invariant Feature Transform) feature detection
and description following Lowe's original algorithm.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional


class SIFTExtractor:
    """
    SIFT Feature Detector and Descriptor Extractor.

    Implements the complete SIFT pipeline:
    1. Scale-space extrema detection using DoG pyramid
    2. Keypoint localization and filtering
    3. Orientation assignment for rotation invariance
    4. Keypoint descriptor computation (128-dimensional)
    """

    def __init__(
        self,
        nfeatures: int = 0,
        nOctaveLayers: int = 3,
        contrastThreshold: float = 0.04,
        edgeThreshold: float = 10,
        sigma: float = 1.6,
    ):
        """
        Initialize SIFT detector with configurable parameters.

        Args:
            nfeatures: Maximum number of features to retain (0 = no limit)
            nOctaveLayers: Number of layers in each octave
            contrastThreshold: Contrast threshold for keypoint filtering
            edgeThreshold: Edge threshold for keypoint filtering
            sigma: Gaussian sigma for the first octave
        """
        self.nfeatures = nfeatures
        self.nOctaveLayers = nOctaveLayers
        self.contrastThreshold = contrastThreshold
        self.edgeThreshold = edgeThreshold
        self.sigma = sigma

        # Create SIFT detector
        self.sift = cv2.SIFT_create(
            nfeatures=self.nfeatures,
            nOctaveLayers=self.nOctaveLayers,
            contrastThreshold=self.contrastThreshold,
            edgeThreshold=self.edgeThreshold,
            sigma=self.sigma,
        )

    def detect_and_compute(
        self, image: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Detect keypoints and compute descriptors for an image.

        This is the main entry point that combines all SIFT stages:
        - Scale-space extrema detection
        - Keypoint localization and filtering
        - Orientation assignment
        - Descriptor computation

        Args:
            image: Grayscale input image

        Returns:
            Tuple of (keypoints, descriptors)
            - keypoints: List of cv2.KeyPoint objects
            - descriptors: numpy array of shape (n_keypoints, 128)
        """
        if len(image.shape) != 2:
            raise ValueError("Input image must be grayscale (single channel)")

        keypoints, descriptors = self.sift.detectAndCompute(image, None)

        # Handle case where no keypoints are found
        if descriptors is None:
            descriptors = np.empty((0, 128), dtype=np.float32)

        return keypoints, descriptors

    def detect(self, image: np.ndarray) -> List[cv2.KeyPoint]:
        """
        Detect keypoints only (without computing descriptors).

        Args:
            image: Grayscale input image

        Returns:
            List of detected keypoints
        """
        keypoints = self.sift.detect(image, None)
        return keypoints

    def compute(
        self, image: np.ndarray, keypoints: List[cv2.KeyPoint]
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Compute descriptors for given keypoints.

        Args:
            image: Grayscale input image
            keypoints: List of keypoints to compute descriptors for

        Returns:
            Tuple of (valid_keypoints, descriptors)
        """
        keypoints, descriptors = self.sift.compute(image, keypoints)

        if descriptors is None:
            descriptors = np.empty((0, 128), dtype=np.float32)

        return keypoints, descriptors

    def get_keypoint_info(self, keypoints: List[cv2.KeyPoint]) -> dict:
        """
        Extract detailed information about detected keypoints.

        Args:
            keypoints: List of keypoints

        Returns:
            Dictionary containing keypoint statistics and information
        """
        if not keypoints:
            return {
                "count": 0,
                "mean_response": 0,
                "mean_size": 0,
                "mean_angle": 0,
                "coordinates": [],
            }

        responses = [kp.response for kp in keypoints]
        sizes = [kp.size for kp in keypoints]
        angles = [kp.angle for kp in keypoints]
        coordinates = [(kp.pt[0], kp.pt[1]) for kp in keypoints]

        return {
            "count": len(keypoints),
            "mean_response": np.mean(responses),
            "mean_size": np.mean(sizes),
            "mean_angle": np.mean(angles),
            "coordinates": coordinates,
            "response_std": np.std(responses),
            "size_std": np.std(sizes),
        }

    def filter_keypoints_by_response(
        self,
        keypoints: List[cv2.KeyPoint],
        descriptors: np.ndarray,
        min_response: float = 0.01,
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Filter keypoints based on response strength.

        Args:
            keypoints: List of keypoints
            descriptors: Corresponding descriptors
            min_response: Minimum response threshold

        Returns:
            Filtered keypoints and descriptors
        """
        if len(keypoints) == 0:
            return keypoints, descriptors

        # Filter based on response
        filtered_indices = [
            i for i, kp in enumerate(keypoints) if kp.response >= min_response
        ]

        filtered_keypoints = [keypoints[i] for i in filtered_indices]
        filtered_descriptors = (
            descriptors[filtered_indices] if len(descriptors) > 0 else descriptors
        )

        return filtered_keypoints, filtered_descriptors

    def visualize_keypoints(
        self, image: np.ndarray, keypoints: List[cv2.KeyPoint]
    ) -> np.ndarray:
        """
        Create visualization of detected keypoints.

        Args:
            image: Input image (grayscale or color)
            keypoints: List of keypoints to visualize

        Returns:
            Image with keypoints drawn
        """
        # Convert to color if grayscale
        if len(image.shape) == 2:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = image.copy()

        # Draw keypoints
        vis_image = cv2.drawKeypoints(
            vis_image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        return vis_image

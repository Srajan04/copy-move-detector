"""
Main Copy-Move Forgery Detector Module

This module integrates all components to provide a complete copy-move forgery
detection system using SIFT features and geometric verification.
"""

import numpy as np
from typing import Dict, Optional
import time

from .image_preprocessor import ImagePreprocessor
from .sift_extractor import SIFTExtractor
from .keypoint_matcher import KeypointMatcher
from .geometric_analyzer import GeometricAnalyzer
from .visualizer import Visualizer


class CopyMoveForgeryDetector:
    """
    Complete copy-move forgery detection system.

    Integrates all pipeline stages:
    1. Image preprocessing (grayscale conversion, noise reduction)
    2. SIFT feature detection and description
    3. Keypoint matching with Lowe's ratio test
    4. Geometric verification using RANSAC
    5. Forgery localization and visualization
    """

    def __init__(
        self,
        # Preprocessing parameters
        apply_gaussian_blur: bool = True,
        blur_kernel_size: int = 3,
        blur_sigma: float = 0.5,
        # SIFT parameters
        nfeatures: int = 0,
        contrast_threshold: float = 0.04,
        edge_threshold: float = 10,
        # Matching parameters
        ratio_threshold: float = 0.8,
        matcher_type: str = "FLANN",
        # Geometric analysis parameters
        reproj_threshold: float = 3.0,
        max_iterations: int = 1000,
        min_cluster_pts: int = 10,
        confidence: float = 0.99,
        min_inlier_ratio: float = 0.1,
        min_translation: float = 6.0,
        min_region_area: float = 150.0,
        max_overlap_ratio: float = 0.9,
        model_type: str = "homography"
    ):
        """
        Initialize the complete detection system.

        Args:
            apply_gaussian_blur: Enable noise reduction preprocessing
            blur_kernel_size: Gaussian blur kernel size
            blur_sigma: Gaussian blur sigma value
            nfeatures: Maximum SIFT features (0 = unlimited)
            contrast_threshold: SIFT contrast threshold
            edge_threshold: SIFT edge response threshold
            ratio_threshold: Lowe's ratio test threshold
            matcher_type: Matcher type ('FLANN' or 'BF')
            reproj_threshold: RANSAC reprojection threshold
            max_iterations: Maximum RANSAC iterations
            min_cluster_pts: Minimum inliers for forgery detection
            confidence: RANSAC confidence level
            min_inlier_ratio: Minimum fraction of matches agreeing on geometry
            min_translation: Minimum separation between source and target (px)
            min_region_area: Minimum area for detected regions (px^2)
            max_overlap_ratio: Maximum allowed overlap between regions
        """
        # Initialize all components
        self.preprocessor = ImagePreprocessor(
            apply_gaussian_blur=apply_gaussian_blur,
            blur_kernel_size=blur_kernel_size,
            blur_sigma=blur_sigma,
        )

        self.sift_extractor = SIFTExtractor(
            nfeatures=nfeatures,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
        )

        self.matcher = KeypointMatcher(
            ratio_threshold=ratio_threshold, matcher_type=matcher_type
        )

        self.geometric_analyzer = GeometricAnalyzer(
            reproj_threshold=reproj_threshold,
            max_iterations=max_iterations,
            min_cluster_pts=min_cluster_pts,
            confidence=confidence,
            min_inlier_ratio=min_inlier_ratio,
            min_translation=min_translation,
            min_region_area=min_region_area,
            max_overlap_ratio=max_overlap_ratio,
            model_type=model_type,
        )

        self.visualizer = Visualizer()

        # Store parameters for reporting
        self.parameters = {
            "ratio_threshold": ratio_threshold,
            "reproj_threshold": reproj_threshold,
            "min_cluster_pts": min_cluster_pts,
            "contrast_threshold": contrast_threshold,
            "edge_threshold": edge_threshold,
            "min_inlier_ratio": min_inlier_ratio,
            "min_translation": min_translation,
            "min_region_area": min_region_area,
            "max_overlap_ratio": max_overlap_ratio,
            "model_type": model_type,
        }

    def detect_forgery(self, image_path: str) -> Dict:
        """
        Detect copy-move forgery in a single image.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary containing detection results
        """
        start_time = time.time()

        try:
            # Step 1: Load and preprocess image
            original_image, preprocessed_image = self.preprocessor.load_and_preprocess(
                image_path
            )

            if not self.preprocessor.validate_image(original_image):
                return self._create_empty_result("Invalid image")

            # Step 2: Extract SIFT features
            keypoints, descriptors = self.sift_extractor.detect_and_compute(
                preprocessed_image
            )

            if len(keypoints) < 4:
                return self._create_empty_result("Insufficient keypoints")

            # Step 3: Match keypoints within the same image
            matches = self.matcher.match_descriptors(descriptors)

            if len(matches) < 4:
                return self._create_empty_result("Insufficient matches")

            # Step 4: Convert matches to point arrays
            src_points, dst_points = self.matcher.convert_matches_to_points(
                matches, keypoints
            )

            # Step 5: Geometric verification using RANSAC
            geometric_result = self.geometric_analyzer.verify_geometric_consistency(
                src_points, dst_points
            )

            # Step 6: Extract region information if forgery detected
            regions = None
            if geometric_result["num_inliers"] > 0:
                regions = self.geometric_analyzer.extract_forgery_regions(
                    src_points,
                    dst_points,
                    geometric_result["inlier_mask"],
                    preprocessed_image.shape,
                )

            # Compile final result
            result = {
                "image_path": image_path,
                "is_forgery": geometric_result["is_forgery"],
                "processing_time": time.time() - start_time,
                "keypoints": keypoints,
                "descriptors": descriptors,
                "matches": matches,
                "num_keypoints": len(keypoints),
                "num_matches": len(matches),
                "num_inliers": geometric_result["num_inliers"],
                "total_matches": geometric_result["total_matches"],
                "inlier_ratio": geometric_result["inlier_ratio"],
                "inlier_mask": geometric_result["inlier_mask"],
                "homography": geometric_result["homography"],
                "transformation_type": geometric_result["transformation_type"],
                "regions": regions,
                "original_image": original_image,
                "preprocessed_image": preprocessed_image,
                "validation_passed": geometric_result["validation_passed"],
                "soft_detection": geometric_result.get("soft_detection", False),
            }

            return result

        except Exception as e:
            return self._create_empty_result(f"Error: {str(e)}")

    def detect_forgery_from_array(self, image: np.ndarray) -> Dict:
        """
        Detect copy-move forgery from image array.

        Args:
            image: Input image as numpy array

        Returns:
            Dictionary containing detection results
        """
        start_time = time.time()

        try:
            if not self.preprocessor.validate_image(image):
                return self._create_empty_result("Invalid image")

            # Preprocess image
            preprocessed_image = self.preprocessor.preprocess(image)

            # Extract SIFT features
            keypoints, descriptors = self.sift_extractor.detect_and_compute(
                preprocessed_image
            )

            if len(keypoints) < 4:
                return self._create_empty_result("Insufficient keypoints")

            # Match keypoints
            matches = self.matcher.match_descriptors(descriptors)

            if len(matches) < 4:
                return self._create_empty_result("Insufficient matches")

            # Convert to points
            src_points, dst_points = self.matcher.convert_matches_to_points(
                matches, keypoints
            )

            # Geometric verification
            geometric_result = self.geometric_analyzer.verify_geometric_consistency(
                src_points, dst_points
            )

            # Extract regions if forgery detected
            regions = None
            if geometric_result["num_inliers"] > 0:
                regions = self.geometric_analyzer.extract_forgery_regions(
                    src_points,
                    dst_points,
                    geometric_result["inlier_mask"],
                    preprocessed_image.shape,
                )

            result = {
                "image_path": "array_input",
                "is_forgery": geometric_result["is_forgery"],
                "processing_time": time.time() - start_time,
                "keypoints": keypoints,
                "descriptors": descriptors,
                "matches": matches,
                "num_keypoints": len(keypoints),
                "num_matches": len(matches),
                "num_inliers": geometric_result["num_inliers"],
                "total_matches": geometric_result["total_matches"],
                "inlier_ratio": geometric_result["inlier_ratio"],
                "inlier_mask": geometric_result["inlier_mask"],
                "homography": geometric_result["homography"],
                "transformation_type": geometric_result["transformation_type"],
                "regions": regions,
                "original_image": image,
                "preprocessed_image": preprocessed_image,
                "validation_passed": geometric_result["validation_passed"],
            }

            return result

        except Exception as e:
            return self._create_empty_result(f"Error: {str(e)}")

    def visualize_detection(
        self, result: Dict, save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Create visualization of detection results.

        Args:
            result: Detection result from detect_forgery()
            save_path: Optional path to save visualization

        Returns:
            Visualization image
        """
        if "original_image" not in result:
            raise ValueError("Result must contain original_image")

        return self.visualizer.visualize_detection_result(
            result["original_image"],
            result,
            result.get("keypoints", []),
            result.get("matches", []),
            # save_path,
        )

    def create_comparison_plot(
        self, result: Dict, title: str = "Copy-Move Forgery Detection"
    ):
        """
        Create side-by-side comparison plot.

        Args:
            result: Detection result
            title: Plot title

        Returns:
            Matplotlib figure
        """
        if "original_image" not in result:
            raise ValueError("Result must contain original_image")

        vis_image = self.visualize_detection(result)

        return self.visualizer.create_comparison_plot(
            result["original_image"], vis_image, title
        )

    def get_detection_summary(self, result: Dict) -> Dict:
        """
        Get human-readable summary of detection results.

        Args:
            result: Detection result dictionary

        Returns:
            Summary dictionary
        """
        if result["is_forgery"]:
            status = "FORGERY DETECTED"
        elif result.get("soft_detection", False):
            status = "POTENTIAL FORGERY (Rotation/Scale Detected)"  # NEW
        else:
            status = "No Forgery Detected"

        summary = {
            "Status": status,
            "Processing Time": f"{result['processing_time']:.3f}s",
            "Keypoints Detected": result["num_keypoints"],
            "Matches Found": result["num_matches"],
            "Inliers (RANSAC)": result["num_inliers"],
            "Inlier Ratio": f"{result['inlier_ratio']:.1%}",
            "Transformation": result["transformation_type"],
            "Validation Passed": result.get("validation_passed", False),
        }

        if result.get("soft_detection", False):
            summary["Note"] = "Strong geometric evidence without full spatial validation"

        if result.get("error_reason"):
            summary["Error"] = result["error_reason"]

        return summary
    def _create_empty_result(self, reason: str) -> Dict:
        """Create empty result for failed detections."""
        return {
            "image_path": "unknown",
            "is_forgery": False,
            "processing_time": 0.0,
            "keypoints": [],
            "descriptors": np.array([]),
            "matches": [],
            "num_keypoints": 0,
            "num_matches": 0,
            "num_inliers": 0,
            "total_matches": 0,
            "inlier_ratio": 0.0,
            "inlier_mask": np.array([]),
            "homography": None,
            "transformation_type": "none",
            "regions": None,
            "error_reason": reason,
            "validation_passed": False,
            "soft_detection": False,  # NEW
        }
    def update_parameters(self, **kwargs):
        """
        Update detection parameters.

        Args:
            **kwargs: Parameter updates
        """
        if "ratio_threshold" in kwargs:
            self.matcher.ratio_threshold = kwargs["ratio_threshold"]

        if "reproj_threshold" in kwargs:
            self.geometric_analyzer.reproj_threshold = kwargs["reproj_threshold"]

        if "min_cluster_pts" in kwargs:
            self.geometric_analyzer.min_cluster_pts = kwargs["min_cluster_pts"]

        if "min_inlier_ratio" in kwargs:
            self.geometric_analyzer.min_inlier_ratio = kwargs["min_inlier_ratio"]

        if "min_translation" in kwargs:
            self.geometric_analyzer.min_translation = kwargs["min_translation"]

        if "min_region_area" in kwargs:
            self.geometric_analyzer.min_region_area = kwargs["min_region_area"]

        if "max_overlap_ratio" in kwargs:
            self.geometric_analyzer.max_overlap_ratio = kwargs["max_overlap_ratio"]

        # Update stored parameters
        self.parameters.update(kwargs)

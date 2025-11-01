"""
Geometric Analyzer Module

This module implements RANSAC-based geometric verification for copy-move
forgery detection using homography estimation.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional


class GeometricAnalyzer:
    """
    Geometric analyzer for robust copy-move forgery detection.

    Uses RANSAC algorithm to fit homography models to matched keypoints
    and identify geometrically consistent regions that indicate forgery.
    """

    def __init__(
        self,
        reproj_threshold: float = 3.0,
        max_iterations: int = 1000,
        min_cluster_pts: int = 10,
        confidence: float = 0.99,
        min_inlier_ratio: float = 0.1,
        min_translation: float = 6.0,
        min_region_area: float = 150.0,
        max_overlap_ratio: float = 0.9,
        model_type: str = "homography",
    ):
        """
        Initialize geometric analyzer with RANSAC parameters.

        Args:
            reproj_threshold: Maximum reprojection error for inliers (pixels)
            max_iterations: Maximum RANSAC iterations
            min_cluster_pts: Minimum inliers required to declare forgery
            confidence: Desired confidence level for RANSAC
            min_inlier_ratio: Minimum fraction of matches that must agree
            min_translation: Minimum centroid shift between regions (pixels)
            min_region_area: Minimum size of inferred regions (pixels^2)
            max_overlap_ratio: Maximum overlap allowed between regions
        """
        self.reproj_threshold = reproj_threshold
        self.max_iterations = max_iterations
        self.min_cluster_pts = min_cluster_pts
        self.confidence = confidence
        self.min_inlier_ratio = min_inlier_ratio
        self.min_translation = min_translation
        self.min_region_area = min_region_area
        self.max_overlap_ratio = max_overlap_ratio
        self.model_type = model_type

    def find_homography_ransac(
        self, src_points: np.ndarray, dst_points: np.ndarray
    ) -> Tuple[Optional[np.ndarray], np.ndarray, int]:
        """
        Find homography using RANSAC algorithm.

        The homography models the geometric transformation between source
        and target regions in a copy-move forgery. RANSAC robustly estimates
        this transformation while filtering out outliers.

        Args:
            src_points: Source keypoint coordinates (N, 2)
            dst_points: Target keypoint coordinates (N, 2)

        Returns:
            Tuple of (homography_matrix, inlier_mask, num_inliers)
        """
        if len(src_points) < 4 or len(dst_points) < 4:
            return None, np.array([]), 0

        # Use OpenCV's findHomography with RANSAC
        homography, mask = cv2.findHomography(
            src_points,
            dst_points,
            cv2.RANSAC,
            self.reproj_threshold,
            maxIters=self.max_iterations,
            confidence=self.confidence,
        )

        if mask is None:
            mask = np.array([])
            num_inliers = 0
        else:
            mask = mask.ravel()  # Flatten to 1D
            num_inliers = np.sum(mask)

        return homography, mask, num_inliers

    def verify_geometric_consistency(
        self, src_points: np.ndarray, dst_points: np.ndarray
    ) -> Dict:
        """
        Verify geometric consistency of matched points using RANSAC.

        Returns:
            Dictionary containing verification results and forgery classification
        """
        result = {
            "is_forgery": False,
            "homography": None,
            "inlier_mask": np.array([]),
            "num_inliers": 0,
            "total_matches": len(src_points),
            "inlier_ratio": 0.0,
            "transformation_type": "none",
            "validation_passed": False,
            "soft_detection": False,  # NEW: potential forgery flag
        }

        if len(src_points) < 4:
            return result

        # Choose geometric model based on configuration
        if self.model_type == "affine":
            transform, mask = self._find_affine_ransac(src_points, dst_points)
            num_inliers = int(np.sum(mask)) if mask is not None else 0
        else:
            transform, mask, num_inliers = self.find_homography_ransac(
                src_points, dst_points
            )

        # Update basic metrics
        result["homography"] = transform
        result["inlier_mask"] = mask if mask is not None else np.array([])
        result["num_inliers"] = num_inliers
        result["inlier_ratio"] = (
            num_inliers / len(src_points) if len(src_points) > 0 else 0.0
        )

        # Primary detection: strict validation
        if num_inliers >= self.min_cluster_pts:
            validation_ok = self._validate_inlier_structure(
                src_points, dst_points, mask
            )
            result["validation_passed"] = validation_ok

            if validation_ok:
                result["is_forgery"] = True
                result["transformation_type"] = self._analyze_transformation(transform)
            else:
                # Fallback: soft detection for strong geometric evidence
                inlier_percentage = (num_inliers / len(src_points)) * 100
                if inlier_percentage > 25.0 and num_inliers > 50:
                    result["soft_detection"] = True
                    result["transformation_type"] = self._analyze_transformation(
                        transform
                    )

        return result

    def _find_affine_ransac(
        self, src_points: np.ndarray, dst_points: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Find affine transformation using RANSAC.

        Affine models (rotation + scale + translation + shear) are more
        restrictive than homographies and better for copy-move without
        perspective changes.

        Args:
            src_points: Source keypoint coordinates (N, 2)
            dst_points: Target keypoint coordinates (N, 2)

        Returns:
            Tuple of (affine_matrix (2x3), inlier_mask (N,))
        """
        if len(src_points) < 3:  # Affine requires minimum 3 points
            return None, None

        affine_mat, inliers = cv2.estimateAffine2D(
            src_points,
            dst_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.reproj_threshold,
            maxIters=self.max_iterations,
            confidence=self.confidence,
        )

        if inliers is None:
            return affine_mat, np.array([])

        return affine_mat, inliers.ravel()

    def _validate_inlier_structure(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        inlier_mask: Optional[np.ndarray],
    ) -> bool:
        """Check spatial layout of inliers to avoid trivial self-matches."""
        if inlier_mask is None or len(inlier_mask) == 0:
            return False

        mask_bool = inlier_mask.astype(bool)
        inlier_count = int(np.sum(mask_bool))

        if inlier_count < self.min_cluster_pts:
            return False

        inlier_ratio = inlier_count / len(inlier_mask)
        if inlier_ratio < self.min_inlier_ratio:
            return False

        src_inliers = src_points[mask_bool]
        dst_inliers = dst_points[mask_bool]

        if len(src_inliers) == 0 or len(dst_inliers) == 0:
            return False

        src_bbox = self._compute_bbox(src_inliers)
        dst_bbox = self._compute_bbox(dst_inliers)

        src_area = src_bbox[2] * src_bbox[3]
        dst_area = dst_bbox[2] * dst_bbox[3]

        if src_area < self.min_region_area or dst_area < self.min_region_area:
            return False

        centroid_shift = np.linalg.norm(
            np.mean(dst_inliers, axis=0) - np.mean(src_inliers, axis=0)
        )

        if centroid_shift < self.min_translation:
            return False

        overlap_ratio = self._compute_overlap_ratio(src_bbox, dst_bbox)

        if overlap_ratio > self.max_overlap_ratio:
            return False

        return True

    def _analyze_transformation(self, homography: np.ndarray) -> str:
        """
        Analyze the type of geometric transformation.

        Args:
            homography: 3x3 homography matrix

        Returns:
            String describing the transformation type
        """
        if homography is None:
            return "none"

        try:
            # Extract transformation components
            # For a 2D transformation, we look at the 2x2 upper-left submatrix
            A = homography[:2, :2]

            # Calculate determinant to check for scaling
            det = np.linalg.det(A)

            # Calculate singular values to analyze scaling and rotation
            U, s, Vt = np.linalg.svd(A)

            # Check for different transformation types
            if abs(det - 1.0) < 0.1:  # Near unit determinant
                if abs(s[0] - s[1]) < 0.1:  # Similar singular values
                    return "rotation_translation"
                else:
                    return "rotation_scaling_translation"
            elif det > 1.1:
                return "scaling_up_rotation_translation"
            elif det < 0.9:
                return "scaling_down_rotation_translation"
            else:
                return "general_transformation"

        except Exception:
            return "complex_transformation"

    def compute_reprojection_error(
        self, src_points: np.ndarray, dst_points: np.ndarray, homography: np.ndarray
    ) -> np.ndarray:
        """
        Compute reprojection error for each point pair.

        Args:
            src_points: Source points
            dst_points: Target points
            homography: Homography matrix

        Returns:
            Array of reprojection errors
        """
        if homography is None or len(src_points) == 0:
            return np.array([])

        # Convert to homogeneous coordinates
        src_homogeneous = cv2.convertPointsToHomogeneous(src_points.reshape(-1, 1, 2))
        src_homogeneous = src_homogeneous.reshape(-1, 3)

        # Apply homography
        projected = homography @ src_homogeneous.T
        projected = projected.T

        # Convert back to Cartesian coordinates
        projected_cartesian = projected[:, :2] / projected[:, 2:3]

        # Compute Euclidean distances
        errors = np.linalg.norm(dst_points - projected_cartesian, axis=1)

        return errors

    def filter_by_spatial_distribution(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        inlier_mask: np.ndarray,
        min_area: float = 100.0,
    ) -> np.ndarray:
        """
        Filter matches based on spatial distribution.

        Ensures that the detected regions have sufficient spatial extent
        to be considered meaningful forgeries.

        Args:
            src_points: Source points
            dst_points: Target points
            inlier_mask: Boolean mask indicating inliers
            min_area: Minimum area (in pixels²) for valid regions

        Returns:
            Updated inlier mask
        """
        if np.sum(inlier_mask) < 4:
            return inlier_mask

        try:
            # Get inlier points
            src_inliers = src_points[inlier_mask.astype(bool)]
            dst_inliers = dst_points[inlier_mask.astype(bool)]

            # Compute convex hulls
            src_hull = cv2.convexHull(src_inliers.astype(np.float32))
            dst_hull = cv2.convexHull(dst_inliers.astype(np.float32))

            # Compute areas
            src_area = cv2.contourArea(src_hull)
            dst_area = cv2.contourArea(dst_hull)

            # Check if areas meet minimum requirement
            if src_area < min_area or dst_area < min_area:
                return np.zeros_like(inlier_mask)

            return inlier_mask

        except Exception:
            return inlier_mask

    def extract_forgery_regions(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        inlier_mask: np.ndarray,
        image_shape: Tuple[int, int],
    ) -> Optional[Dict]:
        """
        Extract source and destination regions from inlier points.

        Args:
            src_points: Source keypoint coordinates
            dst_points: Destination keypoint coordinates
            inlier_mask: Boolean mask of inliers
            image_shape: (height, width) of the image

        Returns:
            Dictionary containing region information or None
        """
        if inlier_mask is None or len(inlier_mask) == 0:
            return None

        mask_bool = inlier_mask.astype(bool)
        if not np.any(mask_bool):
            return None

        src_inliers = src_points[mask_bool]
        dst_inliers = dst_points[mask_bool]

        # Compute bounding boxes
        src_bbox = self._compute_bbox(src_inliers)
        dst_bbox = self._compute_bbox(dst_inliers)

        # Compute overlap
        overlap_ratio = self._compute_overlap_ratio(src_bbox, dst_bbox)

        return {
            "src_region": src_inliers,
            "dst_region": dst_inliers,
            "src_bbox": src_bbox,
            "dst_bbox": dst_bbox,
            "overlap_ratio": overlap_ratio,
        }

    def _compute_bbox(self, points: np.ndarray) -> Tuple[int, int, int, int]:
        """Compute bounding box for a set of points."""
        if len(points) == 0:
            return (0, 0, 0, 0)

        x_min, y_min = np.min(points, axis=0).astype(int)
        x_max, y_max = np.max(points, axis=0).astype(int)

        return (x_min, y_min, x_max - x_min, y_max - y_min)

    def _compute_overlap_ratio(
        self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]
    ) -> float:
        """Compute overlap ratio between two bounding boxes."""
        if bbox1 is None or bbox2 is None:
            return 0.0

        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Compute intersection
        x_int = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_int = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        intersection = x_int * y_int

        # Compute union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

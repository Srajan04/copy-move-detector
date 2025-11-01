"""
Visualization Module

This module provides comprehensive visualization capabilities for copy-move
forgery detection results.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import os


class Visualizer:
    """
    Comprehensive visualization tools for copy-move forgery detection.

    Provides functions to visualize keypoints, matches, and detection results
    with clear annotations and color coding.
    """

    def __init__(self, figure_size: Tuple[int, int] = (15, 10)):
        """
        Initialize visualizer with default settings.

        Args:
            figure_size: Default figure size for matplotlib plots
        """
        self.figure_size = figure_size
        self.colors = {
            "source": (0, 255, 0),  # Green for source region
            "target": (0, 0, 255),  # Red for target region
            "match_line": (255, 255, 0),  # Yellow for match lines
            "keypoint": (255, 0, 255),  # Magenta for keypoints
            "bbox": (255, 255, 255),  # White for bounding boxes
        }

    def visualize_detection_result(
        self,
        image: np.ndarray,
        result: Dict,
        draw_matches: bool = True,
        draw_regions: bool = True,
    ) -> np.ndarray:
        """
        Create comprehensive visualization of detection results.

        Args:
            image: Original image
            result: Detection result dictionary
            draw_matches: Whether to draw match lines
            draw_regions: Whether to draw region boxes

        Returns:
            Visualization image
        """
        vis_image = image.copy()
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)

        # Draw inlier matches if available
        if draw_matches and result.get("inlier_mask") is not None:
            mask = result["inlier_mask"]
            if len(mask) > 0 and np.any(mask):
                keypoints = result.get("keypoints", [])
                matches = result.get("matches", [])

                if keypoints and matches:
                    vis_image = self._draw_inlier_matches(
                        vis_image, keypoints, matches, mask
                    )

        # Draw regions for confirmed forgery OR soft detection (NEW)
        if draw_regions and (
            result.get("is_forgery") or result.get("soft_detection", False)
        ):
            regions = result.get("regions")
            if regions and regions.get("src_bbox") and regions.get("dst_bbox"):
                vis_image = self._draw_region_boxes(vis_image, regions)

        # Add statistics overlay
        vis_image = self._add_statistics_overlay(vis_image, result)

        return vis_image

    def _add_statistics_overlay(self, image: np.ndarray, result: Dict) -> np.ndarray:
        """Add text overlay with detection statistics."""
        overlay = image.copy()
        height, width = image.shape[:2]

        # Determine status and color
        if result.get("is_forgery"):
            status = "FORGERY DETECTED"
            color = (0, 0, 255)  # Red
        elif result.get("soft_detection", False):
            status = "POTENTIAL FORGERY"
            color = (0, 165, 255)  # Orange
        else:
            status = "NO FORGERY"
            color = (0, 255, 0)  # Green

        # Background rectangle for better readability
        cv2.rectangle(overlay, (10, 10), (550, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        # Status text
        cv2.putText(
            image,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
        )

        # Statistics
        y_offset = 75
        stats = [
            f"Keypoints: {result.get('num_keypoints', 0)}",
            f"Matches: {result.get('num_matches', 0)}",
            f"Inliers: {result.get('num_inliers', 0)}",
            f"Ratio: {result.get('inlier_ratio', 0):.1%}",
        ]

        for stat in stats:
            cv2.putText(
                image,
                stat,
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )
            y_offset += 30

        # Add note for soft detections
        if result.get("soft_detection", False) and not result.get("is_forgery"):
            cv2.putText(
                image,
                "Strong geometric evidence",
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 165, 255),
                1,
            )

        return image

    def _draw_inlier_matches(
        self,
        image: np.ndarray,
        keypoints: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
        inlier_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Draw lines connecting inlier matched keypoints.

        Args:
            image: Image to draw on
            keypoints: All keypoints
            matches: All matches
            inlier_mask: Boolean mask indicating inliers

        Returns:
            Image with match lines drawn
        """
        mask_bool = inlier_mask.astype(bool).ravel()

        for i, match in enumerate(matches):
            if i < len(mask_bool) and mask_bool[i]:
                # Get coordinates
                pt1 = tuple(map(int, keypoints[match.queryIdx].pt))
                pt2 = tuple(map(int, keypoints[match.trainIdx].pt))

                # Draw line
                cv2.line(image, pt1, pt2, self.colors["match_line"], 1)

                # Draw endpoints
                cv2.circle(image, pt1, 3, self.colors["source"], -1)
                cv2.circle(image, pt2, 3, self.colors["target"], -1)

        return image

    def _draw_region_boxes(self, image: np.ndarray, regions: Dict) -> np.ndarray:
        """
        Draw bounding boxes around source and target regions.

        Args:
            image: Image to draw on
            regions: Dictionary with src_bbox and dst_bbox

        Returns:
            Image with bounding boxes drawn
        """
        # Draw source region (green)
        if regions.get("src_bbox") is not None:
            x, y, w, h = regions["src_bbox"]
            cv2.rectangle(image, (x, y), (x + w, y + h), self.colors["source"], 3)
            cv2.putText(
                image,
                "SOURCE",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.colors["source"],
                2,
            )

        # Draw target region (red)
        if regions.get("dst_bbox") is not None:
            x, y, w, h = regions["dst_bbox"]
            cv2.rectangle(image, (x, y), (x + w, y + h), self.colors["target"], 3)
            cv2.putText(
                image,
                "TARGET",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.colors["target"],
                2,
            )

        return image

    def _draw_matches(
        self,
        image: np.ndarray,
        keypoints: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
    ) -> np.ndarray:
        """Draw match lines and keypoints."""
        for match in matches:
            # Get keypoint coordinates
            pt1 = tuple(map(int, keypoints[match.queryIdx].pt))
            pt2 = tuple(map(int, keypoints[match.trainIdx].pt))

            # Draw line between matched points
            cv2.line(image, pt1, pt2, self.colors["match_line"], 2)

            # Draw keypoints
            cv2.circle(image, pt1, 5, self.colors["source"], -1)
            cv2.circle(image, pt2, 5, self.colors["target"], -1)

        return image

    def _draw_regions(self, image: np.ndarray, regions: Dict) -> np.ndarray:
        """Draw bounding boxes around detected regions."""
        if regions["src_bbox"] is not None:
            x, y, w, h = regions["src_bbox"]
            cv2.rectangle(image, (x, y), (x + w, y + h), self.colors["source"], 3)
            cv2.putText(
                image,
                "SOURCE",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.colors["source"],
                2,
            )

        if regions["dst_bbox"] is not None:
            x, y, w, h = regions["dst_bbox"]
            cv2.rectangle(image, (x, y), (x + w, y + h), self.colors["target"], 3)
            cv2.putText(
                image,
                "TARGET",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.colors["target"],
                2,
            )

        return image

    def _add_detection_info(self, image: np.ndarray, result: Dict) -> np.ndarray:
        """Add detection information as text overlay."""
        info_text = [
            f"FORGERY DETECTED",
            f"Inliers: {result['num_inliers']}",
            f"Total matches: {result['total_matches']}",
            f"Inlier ratio: {result['inlier_ratio']:.2f}",
            f"Transform: {result['transformation_type']}",
        ]

        y_offset = 30
        for i, text in enumerate(info_text):
            color = (
                (0, 0, 255) if i == 0 else (255, 255, 255)
            )  # Red for title, white for details
            cv2.putText(
                image,
                text,
                (20, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        return image

    def create_comparison_plot(
        self,
        original: np.ndarray,
        result_image: np.ndarray,
        title: str = "Copy-Move Forgery Detection",
    ) -> plt.Figure:
        """
        Create side-by-side comparison plot.

        Args:
            original: Original image
            result_image: Image with detection results
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figure_size)

        # Convert BGR to RGB for matplotlib
        if len(original.shape) == 3:
            original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        else:
            original_rgb = original
            result_rgb = result_image

        axes[0].imshow(original_rgb, cmap="gray" if len(original.shape) == 2 else None)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(
            result_rgb, cmap="gray" if len(result_image.shape) == 2 else None
        )
        axes[1].set_title("Detection Result")
        axes[1].axis("off")

        fig.suptitle(title, fontsize=16)
        plt.tight_layout()

        return fig

    def visualize_keypoints(
        self,
        image: np.ndarray,
        keypoints: List[cv2.KeyPoint],
        title: str = "SIFT Keypoints",
    ) -> np.ndarray:
        """
        Visualize detected keypoints.

        Args:
            image: Input image
            keypoints: List of keypoints
            title: Title for the visualization

        Returns:
            Image with keypoints drawn
        """
        # Convert to color if grayscale
        if len(image.shape) == 2:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = image.copy()

        # Draw keypoints with orientation and scale
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            size = int(kp.size / 2)
            angle = kp.angle

            # Draw circle for keypoint
            cv2.circle(vis_image, (x, y), size, self.colors["keypoint"], 2)

            # Draw orientation line
            if angle != -1:  # Valid orientation
                end_x = int(x + size * np.cos(np.radians(angle)))
                end_y = int(y + size * np.sin(np.radians(angle)))
                cv2.line(vis_image, (x, y), (end_x, end_y), self.colors["keypoint"], 2)

        # Add title
        cv2.putText(
            vis_image,
            f"{title} ({len(keypoints)} keypoints)",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        return vis_image

    def create_statistics_plot(self, results: List[Dict]) -> plt.Figure:
        """
        Create statistics plot for batch processing results.

        Args:
            results: List of detection results

        Returns:
            Matplotlib figure with statistics
        """
        # Extract statistics
        total_images = len(results)
        forgeries_detected = sum(1 for r in results if r["is_forgery"])
        inlier_counts = [r["num_inliers"] for r in results if r["is_forgery"]]
        inlier_ratios = [r["inlier_ratio"] for r in results if r["is_forgery"]]

        fig, axes = plt.subplots(2, 2, figsize=self.figure_size)

        # Detection summary
        labels = ["Authentic", "Forged"]
        sizes = [total_images - forgeries_detected, forgeries_detected]
        colors = ["lightgreen", "lightcoral"]
        axes[0, 0].pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%")
        axes[0, 0].set_title("Detection Summary")

        # Inlier count distribution
        if inlier_counts:
            axes[0, 1].hist(inlier_counts, bins=20, color="skyblue", alpha=0.7)
            axes[0, 1].set_title("Distribution of Inlier Counts")
            axes[0, 1].set_xlabel("Number of Inliers")
            axes[0, 1].set_ylabel("Frequency")

        # Inlier ratio distribution
        if inlier_ratios:
            axes[1, 0].hist(inlier_ratios, bins=20, color="lightgreen", alpha=0.7)
            axes[1, 0].set_title("Distribution of Inlier Ratios")
            axes[1, 0].set_xlabel("Inlier Ratio")
            axes[1, 0].set_ylabel("Frequency")

        # Statistics text
        stats_text = f"""Total Images: {total_images}
Forgeries Detected: {forgeries_detected}
Detection Rate: {forgeries_detected / total_images * 100:.1f}%
Avg Inliers: {np.mean(inlier_counts) if inlier_counts else 0:.1f}
Avg Inlier Ratio: {np.mean(inlier_ratios) if inlier_ratios else 0:.3f}"""

        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment="center")
        axes[1, 1].axis("off")
        axes[1, 1].set_title("Statistics Summary")

        plt.tight_layout()
        return fig

    def save_batch_results(
        self, results: List[Dict], output_dir: str, image_paths: List[str]
    ) -> None:
        """
        Save visualization results for batch processing.

        Args:
            results: List of detection results
            output_dir: Output directory
            image_paths: List of image file paths
        """
        os.makedirs(output_dir, exist_ok=True)

        for i, (result, image_path) in enumerate(zip(results, image_paths)):
            if result["is_forgery"]:
                # Load original image
                image = cv2.imread(image_path)
                if image is not None:
                    # Create visualization
                    vis_image = self.visualize_detection_result(
                        image,
                        result,
                        result.get("keypoints", []),
                        result.get("matches", []),
                    )

                    # Save result
                    filename = os.path.basename(image_path)
                    output_path = os.path.join(output_dir, f"detection_{filename}")
                    cv2.imwrite(output_path, vis_image)

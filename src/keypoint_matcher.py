import cv2
import numpy as np
from typing import List, Tuple, Dict, Dict


class KeypointMatcher:
    """
    Keypoint matcher implementing Lowe's ratio test for robust matching.

    The matcher finds correspondences between SIFT descriptors and applies
    the ratio test to filter out ambiguous matches, significantly reducing
    false positives before geometric verification.
    """

    def __init__(
        self,
        ratio_threshold: float = 0.8,
        matcher_type: str = "FLANN",
        cross_check: bool = False,
    ):
        """
        Initialize the keypoint matcher.

        Args:
            ratio_threshold: Lowe's ratio test threshold (0.7-0.8 recommended)
            matcher_type: Type of matcher ('FLANN' or 'BF' for brute force)
            cross_check: Whether to use cross-checking for additional filtering
        """
        self.ratio_threshold = ratio_threshold
        self.matcher_type = matcher_type
        self.cross_check = cross_check

        # Initialize matcher
        if matcher_type == "FLANN":
            self.matcher = self._create_flann_matcher()
        elif matcher_type == "BF":
            self.matcher = self._create_bf_matcher()
        else:
            raise ValueError("matcher_type must be 'FLANN' or 'BF'")

    def _create_flann_matcher(self) -> cv2.FlannBasedMatcher:
        """
        Create FLANN (Fast Library for Approximate Nearest Neighbors) matcher.

        FLANN is more efficient for large numbers of features and uses
        k-d trees for approximate nearest neighbor search.

        Returns:
            Configured FLANN matcher
        """
        # FLANN parameters for SIFT
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        return cv2.FlannBasedMatcher(index_params, search_params)

    def _create_bf_matcher(self) -> cv2.BFMatcher:
        """
        Create Brute Force matcher.

        BF matcher compares every descriptor with every other descriptor.
        More accurate but slower than FLANN for large datasets.

        Returns:
            Configured BF matcher
        """
        return cv2.BFMatcher(cv2.NORM_L2, crossCheck=self.cross_check)

    def match_descriptors(self, descriptors: np.ndarray) -> List[cv2.DMatch]:
        if len(descriptors) < 2:
            return []

        # request one extra neighbour to skip the self match
        knn_matches = self.matcher.knnMatch(descriptors, descriptors, k=3)

        candidate_matches: List[cv2.DMatch] = []
        for match_set in knn_matches:
            non_self = [m for m in match_set if m.queryIdx != m.trainIdx]

            # need two distinct neighbours for Lowe ratio
            if len(non_self) < 2:
                continue

            best, second = non_self[:2]
            if second.distance == 0: # L2 distance zero
                continue

            if best.distance < self.ratio_threshold * second.distance:
                candidate_matches.append(best)

        # remove symmetric duplicates and zero-distance artefacts
        unique: Dict[Tuple[int, int], cv2.DMatch] = {}
        for match in candidate_matches:
            if match.distance == 0:
                continue
            key = tuple(sorted((match.queryIdx, match.trainIdx)))
            if key not in unique or match.distance < unique[key].distance:
                unique[key] = match

        return list(unique.values())

    def match_between_images(
        self, descriptors1: np.ndarray, descriptors2: np.ndarray
    ) -> List[cv2.DMatch]:
        """
        Match descriptors between two different images.

        Args:
            descriptors1: Descriptors from first image
            descriptors2: Descriptors from second image

        Returns:
            List of good matches after applying Lowe's ratio test
        """
        if len(descriptors1) == 0 or len(descriptors2) == 0:
            return []

        # Find k=2 nearest neighbors
        matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)

        # Apply Lowe's ratio test
        good_matches = self._apply_ratio_test(matches)

        return good_matches

    def _apply_ratio_test(self, matches: List[List[cv2.DMatch]]) -> List[cv2.DMatch]:
        """
        Apply Lowe's ratio test to filter matches.

        The ratio test compares the distance to the closest neighbor with
        the distance to the second-closest neighbor. A match is considered
        good if the ratio is below the threshold, indicating that the closest
        match is significantly better than the second-closest.

        Mathematical formulation:
        ratio = distance_to_best / distance_to_second_best

        If ratio < threshold: accept match (unambiguous)
        If ratio >= threshold: reject match (ambiguous)

        Args:
            matches: List of k=2 nearest neighbor matches

        Returns:
            Filtered list of good matches
        """
        good_matches = []

        for match_pair in matches:
            # Each match_pair should contain exactly 2 matches (k=2)
            if len(match_pair) == 2:
                best_match, second_best_match = match_pair

                # This is the standard Lowe's ratio test.
                # It is not suitable for self-matching where the best_match is the identity.
                # The `match_descriptors` method now has its own logic for that.
                if second_best_match.distance > 0:  # Avoid division by zero
                    ratio = best_match.distance / second_best_match.distance
                    if ratio < self.ratio_threshold:
                        good_matches.append(best_match)

        return good_matches

    def get_match_statistics(self, matches: List[cv2.DMatch]) -> Dict[str, float]:
        """
        Calculates statistics about the matches.

        Args:
            matches: A list of DMatch objects.

        Returns:
            A dictionary with match statistics (mean, min, max distance).
        """
        if not matches:
            return {"mean_distance": 0, "min_distance": 0, "max_distance": 0}

        distances = [m.distance for m in matches]
        return {
            "mean_distance": np.mean(distances),
            "min_distance": np.min(distances),
            "max_distance": np.max(distances),
        }

    def filter_matches_by_distance(
        self, matches: List[cv2.DMatch], max_distance: float
    ) -> List[cv2.DMatch]:
        """
        Filter matches based on descriptor distance.

        Args:
            matches: List of matches to filter
            max_distance: Maximum allowed descriptor distance

        Returns:
            Filtered matches
        """
        return [m for m in matches if m.distance <= max_distance]

    def convert_matches_to_points(
        self, matches: List[cv2.DMatch], keypoints: List[cv2.KeyPoint]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert matches to coordinate arrays for geometric analysis.

        Args:
            matches: List of matches
            keypoints: List of keypoints

        Returns:
            Tuple of (source_points, target_points) as numpy arrays
        """
        if not matches:
            return np.empty((0, 2), dtype=np.float32), np.empty(
                (0, 2), dtype=np.float32
            )

        src_pts = np.array(
            [keypoints[m.queryIdx].pt for m in matches], dtype=np.float32
        )
        dst_pts = np.array(
            [keypoints[m.trainIdx].pt for m in matches], dtype=np.float32
        )

        return src_pts, dst_pts

    def visualize_matches(
        self,
        image: np.ndarray,
        keypoints: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
        max_matches: int = 50,
    ) -> np.ndarray:
        """
        Visualize matches by drawing lines between matched keypoints.

        Args:
            image: Input image
            keypoints: List of keypoints
            matches: List of matches to visualize
            max_matches: Maximum number of matches to draw

        Returns:
            Image with matches visualized
        """
        # Limit number of matches for cleaner visualization
        matches_to_draw = (
            matches[:max_matches] if len(matches) > max_matches else matches
        )

        # Convert to color if grayscale
        if len(image.shape) == 2:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = image.copy()

        # Draw matches as lines
        for match in matches_to_draw:
            pt1 = tuple(map(int, keypoints[match.queryIdx].pt))
            pt2 = tuple(map(int, keypoints[match.trainIdx].pt))

            # Draw line between matched points
            cv2.line(vis_image, pt1, pt2, (0, 255, 0), 1)

            # Draw circles at keypoint locations
            cv2.circle(vis_image, pt1, 3, (255, 0, 0), -1)
            cv2.circle(vis_image, pt2, 3, (0, 0, 255), -1)

        return vis_image

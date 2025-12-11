"""
Homography estimation utilities between neighboring tiles.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import cv2
import numpy as np

from .matching import MatchResult


@dataclass
class HomographyResult:
    """
    Container for the homography between two images.

    Attributes:
        filename_a: Name of the first image.
        filename_b: Name of the second image.
        H: 3x3 homography matrix (float64).
        inliers_mask: (N, 1) mask array from RANSAC indicating inlier matches.
        num_inliers: Number of inlier matches.
        num_matches: Total number of matches used.
    """
    filename_a: str
    filename_b: str
    H: np.ndarray
    inliers_mask: np.ndarray
    num_inliers: int
    num_matches: int


def estimate_homography_for_pair(
    keypoints_a_xy: np.ndarray,
    keypoints_b_xy: np.ndarray,
    match_result: MatchResult,
    ransac_thresh: float = 3.0,
    min_matches: int = 4,
) -> Optional[HomographyResult]:
    """
    Estimate the homography H such that:

        [x_b, y_b, 1]^T ~ H * [x_a, y_a, 1]^T

    using RANSAC on the matched keypoints.

    Args:
        keypoints_a_xy: (Na, 2) array of (x, y) keypoint coordinates for image A.
        keypoints_b_xy: (Nb, 2) array of (x, y) keypoint coordinates for image B.
        match_result: MatchResult with matches between A and B.
        ransac_thresh: RANSAC reprojection threshold.
        min_matches: Minimum number of matches required to estimate H.

    Returns:
        HomographyResult if estimation succeeds, otherwise None.
    """

    # Check if there are enough matches
    if not match_result.matches:
        return None

    # Extract index pairs from the matches
    idx_a_list = [m[0] for m in match_result.matches]
    idx_b_list = [m[1] for m in match_result.matches]

    # Ensure we have enough matches
    if len(idx_a_list) < min_matches or len(idx_b_list) < min_matches:
        return None

    # Gather corresponding points
    pts_a = keypoints_a_xy[idx_a_list].astype(np.float32).reshape(-1, 1, 2)
    pts_b = keypoints_b_xy[idx_b_list].astype(np.float32).reshape(-1, 1, 2)

    # Estimate homography with RANSAC
    H, mask = cv2.findHomography(
        pts_a,
        pts_b,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
    )

    # Check if homography estimation was successful
    if H is None or mask is None:
        return None

    # Count inliers
    num_inliers = int(mask.sum())
    num_matches = len(match_result.matches)

    # Return the result
    return HomographyResult(
        filename_a=match_result.filename_a,
        filename_b=match_result.filename_b,
        H=H,
        inliers_mask=mask,
        num_inliers=num_inliers,
        num_matches=num_matches,
    )


def estimate_homographies(
    keypoints_xy: Dict[str, np.ndarray],
    matches: Dict[Tuple[str, str], MatchResult],
    ransac_thresh: float = 3.0,
    min_matches: int = 4,
) -> Dict[Tuple[str, str], HomographyResult]:
    """
    Estimate homographies for all matched neighbor pairs.

    Args:
        keypoints_xy: Mapping filename -> (N, 2) array of keypoint coordinates.
                      (e.g., from `load_keypoints_results` or similar.)
        matches: Mapping (filename_a, filename_b) -> MatchResult.
        ransac_thresh: RANSAC reprojection threshold.
        min_matches: Minimum number of matches required to estimate H.

    Returns:
        Mapping (filename_a, filename_b) -> HomographyResult.
        Keys are the same tuple keys as in `matches`.
    """
    results: Dict[Tuple[str, str], HomographyResult] = {}

    # Iterate over each matched pair
    for pair, match_result in matches.items():
        # Retrieve key filenames
        fa = match_result.filename_a
        fb = match_result.filename_b

        # Retrieve keypoint coordinates
        pts_a = keypoints_xy.get(fa)
        pts_b = keypoints_xy.get(fb)
        
        # Skip if keypoints are missing
        if pts_a is None or pts_b is None:
            continue

        # Estimate homography for the pair
        hres = estimate_homography_for_pair(
            keypoints_a_xy=pts_a,
            keypoints_b_xy=pts_b,
            match_result=match_result,
            ransac_thresh=ransac_thresh,
            min_matches=min_matches,
        )

        # Store result if successful
        if hres is not None:
            results[pair] = hres

    return results

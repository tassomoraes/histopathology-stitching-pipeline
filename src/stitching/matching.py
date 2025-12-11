"""
Neighbor matching utilities for tile-to-tile descriptor correspondences.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .keypoints import KeypointResult


@dataclass
class MatchResult:
    """
    Container for matches between two images.

    Attributes:
        filename_a: Name of the first image.
        filename_b: Name of the second image (neighbor).
        matches: List of (idx_a, idx_b, distance) tuples.
    """

    filename_a: str
    filename_b: str
    matches: List[Tuple[int, int, float]]


def _infer_norm_type(descriptors: np.ndarray) -> int:
    """
    Infer the norm type for BFMatcher based on descriptor dtype.

    ORB: uint8 -> NORM_HAMMING
    SIFT: float32 -> NORM_L2
    """

    # Simple heuristic based on dtype
    if descriptors.dtype == np.uint8:
        return cv2.NORM_HAMMING
    return cv2.NORM_L2


def match_descriptors(
    desc_a: np.ndarray,
    desc_b: np.ndarray,
    ratio: float = 0.75,
) -> List[Tuple[int, int, float]]:
    """
    Match two sets of descriptors using BFMatcher.

    Uses Lowe's ratio test when there are at least 2 descriptors
    in the second set. Falls back to simple matching when the
    number of descriptors is too small for k-NN.

    Args:
        desc_a: Descriptors from image A, shape (Na, D).
        desc_b: Descriptors from image B, shape (Nb, D).
        ratio: Lowe's ratio threshold.

    Returns:
        List of (idx_a, idx_b, distance) tuples.
    """
    # Handle edge cases
    if desc_a is None or desc_b is None:
        return []
    if len(desc_a) == 0 or len(desc_b) == 0:
        return []

    norm_type = _infer_norm_type(desc_a)

    # If we have fewer than 2 descriptors in desc_b,
    # k-NN with k=2 is not possible. Use a simpler matching strategy.
    if desc_b.shape[0] < 2:
        bf = cv2.BFMatcher(normType=norm_type, crossCheck=True)
        raw_matches = bf.match(desc_a, desc_b)
        return [
            (m.queryIdx, m.trainIdx, float(m.distance))
            for m in raw_matches
        ]

    # Standard case: use k-NN + Lowe's ratio test
    bf = cv2.BFMatcher(normType=norm_type, crossCheck=False)
    knn_matches = bf.knnMatch(desc_a, desc_b, k=2)

    good_matches: List[Tuple[int, int, float]] = []
    for m_n in knn_matches:
        # Some entries may still have < 2 neighbors; be defensive:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < ratio * n.distance:
            good_matches.append((m.queryIdx, m.trainIdx, float(m.distance)))

    return good_matches


def match_neighbors(
    keypoints_results: Dict[str, KeypointResult],
    graph: Dict[str, Dict],
    ratio: float = 0.75,
) -> Dict[Tuple[str, str], MatchResult]:
    """
    Compute matches between all neighboring tiles defined in the graph.

    Args:
        keypoints_results: Mapping filename -> KeypointResult.
        graph: Mapping filename -> {"coords": (x, y), "neighbors": [filenames]}.
        ratio: Lowe's ratio threshold.

    Returns:
        Dictionary mapping (filename_a, filename_b) -> MatchResult.
        Pairs are stored with filename_a < filename_b (sorted) to avoid duplication.
    """

    # Store results
    results: Dict[Tuple[str, str], MatchResult] = {}

    # Keep track of processed pairs to avoid duplicates
    seen_pairs = set()

    # Iterate over each image and its neighbors
    for filename, info in graph.items():
        
        # Get neighbors
        neighbors = info.get("neighbors", [])
        
        # Match with each neighbor
        for neighbor in neighbors:
            pair = tuple(sorted((filename, neighbor)))
            
            # Skip if already processed
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            # Retrieve KeypointResults
            res_a = keypoints_results.get(pair[0])
            res_b = keypoints_results.get(pair[1])
            
            # Skip if either result is missing
            if res_a is None or res_b is None:
                continue

            # Perform matching
            matches = match_descriptors(res_a.descriptors, res_b.descriptors, ratio=ratio)
            results[pair] = MatchResult(
                filename_a=pair[0],
                filename_b=pair[1],
                matches=matches,
            )

    return results

import numpy as np

from src.stitching.homography import (
    estimate_homography_for_pair,
    estimate_homographies,
)
from src.stitching.matching import MatchResult


def test_estimate_homography_for_pair_translation():
    # Four points in image A (a square)
    keypoints_a_xy = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
        ],
        dtype=np.float32,
    )

    # Apply a translation (dx, dy) to get points in image B
    dx, dy = 5.0, 7.0
    keypoints_b_xy = keypoints_a_xy + np.array([dx, dy], dtype=np.float32)

    # Matches: each point i in A matches point i in B
    matches = [(i, i, 0.0) for i in range(4)]

    match_result = MatchResult(
        filename_a="A.jpg",
        filename_b="B.jpg",
        matches=matches,
    )

    hres = estimate_homography_for_pair(
        keypoints_a_xy,
        keypoints_b_xy,
        match_result,
        ransac_thresh=1.0,
        min_matches=4,
    )

    assert hres is not None
    H = hres.H

    # Normalize H so that H[2,2] = 1
    H = H / H[2, 2]

    expected = np.array(
        [
            [1.0, 0.0, dx],
            [0.0, 1.0, dy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    assert np.allclose(H, expected, atol=1e-3)
    assert hres.num_matches == 4
    assert hres.num_inliers == 4


def test_estimate_homographies_multiple_pairs():
    # Two simple pairs, same translation
    dx, dy = 3.0, -2.0

    pts_base = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
        ],
        dtype=np.float32,
    )

    keypoints_xy = {
        "A.jpg": pts_base,
        "B.jpg": pts_base + np.array([dx, dy], dtype=np.float32),
        "C.jpg": pts_base + np.array([2 * dx, 2 * dy], dtype=np.float32),
    }

    matches = {
        ("A.jpg", "B.jpg"): MatchResult(
            filename_a="A.jpg",
            filename_b="B.jpg",
            matches=[(i, i, 0.0) for i in range(4)],
        ),
        ("B.jpg", "C.jpg"): MatchResult(
            filename_a="B.jpg",
            filename_b="C.jpg",
            matches=[(i, i, 0.0) for i in range(4)],
        ),
    }

    results = estimate_homographies(
        keypoints_xy=keypoints_xy,
        matches=matches,
        ransac_thresh=1.0,
        min_matches=4,
    )

    assert ("A.jpg", "B.jpg") in results
    assert ("B.jpg", "C.jpg") in results

    H_ab = results[("A.jpg", "B.jpg")].H
    H_bc = results[("B.jpg", "C.jpg")].H

    H_ab = H_ab / H_ab[2, 2]
    H_bc = H_bc / H_bc[2, 2]

    expected_ab = np.array(
        [
            [1.0, 0.0, dx],
            [0.0, 1.0, dy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    expected_bc = np.array(
        [
            [1.0, 0.0, dx],
            [0.0, 1.0, dy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    assert np.allclose(H_ab, expected_ab, atol=1e-3)
    assert np.allclose(H_bc, expected_bc, atol=1e-3)

from pathlib import Path

import cv2
import numpy as np

from src.stitching.keypoints import KeypointResult
from src.stitching.matching import match_descriptors, match_neighbors


def _make_dummy_keypoint(x: float, y: float) -> cv2.KeyPoint:
    return cv2.KeyPoint(x, y, 1.0)


def test_match_descriptors_simple_case():
    # Two identical descriptor sets should produce multiple matches
    desc_a = np.random.randint(0, 256, size=(50, 32), dtype=np.uint8)
    desc_b = desc_a.copy()

    matches = match_descriptors(desc_a, desc_b, ratio=0.8)

    assert len(matches) > 0
    # Each match is (idx_a, idx_b, distance)
    idx_a, idx_b, dist = matches[0]
    assert isinstance(idx_a, int)
    assert isinstance(idx_b, int)
    assert isinstance(dist, float)


def test_match_neighbors_with_toy_graph():
    # Create two fake images with simple descriptors
    filename_a = "00001_x1_y1_zp1.jpg"
    filename_b = "00002_x2_y1_zp1.jpg"

    keypoints_a = [_make_dummy_keypoint(10.0, 10.0)]
    keypoints_b = [_make_dummy_keypoint(11.0, 11.0)]

    desc_a = np.random.randint(0, 256, size=(1, 32), dtype=np.uint8)
    desc_b = desc_a.copy()  # ensure at least one good match

    keypoints_results = {
        filename_a: KeypointResult(
            filename=filename_a,
            keypoints=keypoints_a,
            descriptors=desc_a,
        ),
        filename_b: KeypointResult(
            filename=filename_b,
            keypoints=keypoints_b,
            descriptors=desc_b,
        ),
    }

    # Toy graph: A neighbor of B, B neighbor of A
    graph = {
        filename_a: {"coords": (1, 1), "neighbors": [filename_b]},
        filename_b: {"coords": (2, 1), "neighbors": [filename_a]},
    }

    results = match_neighbors(keypoints_results, graph, ratio=0.8)

    assert (filename_a, filename_b) in results or (filename_b, filename_a) in results

    # Normalize key: we store pairs sorted
    pair = tuple(sorted((filename_a, filename_b)))
    match_result = results[pair]

    assert match_result.filename_a == pair[0]
    assert match_result.filename_b == pair[1]
    assert len(match_result.matches) > 0

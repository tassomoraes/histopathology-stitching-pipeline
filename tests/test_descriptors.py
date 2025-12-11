from pathlib import Path

import cv2
import numpy as np

from src.stitching.keypoints import KeypointResult
from src.stitching.descriptors import (
    save_keypoints_results,
    load_keypoints_results,
)


def _make_dummy_keypoint(x: float, y: float) -> cv2.KeyPoint:
    # OpenCV Python binding uses this signature:
    # cv2.KeyPoint(x, y, _size[, _angle[, _response[, _octave[, _class_id]]]])
    return cv2.KeyPoint(x, y, 1.0)


def test_save_and_load_keypoints_results(tmp_path):
    # Arrange: create a fake KeypointResult with 2 keypoints and random descriptors
    filename = "00001_x1_y1_zp1.jpg"
    keypoints = [
        _make_dummy_keypoint(10.0, 20.0),
        _make_dummy_keypoint(30.0, 40.0),
    ]
    descriptors = np.random.randint(
        0,
        256,
        size=(2, 32),
        dtype=np.uint8,
    )

    result = KeypointResult(
        filename=filename,
        keypoints=keypoints,
        descriptors=descriptors,
    )

    results_dict = {filename: result}

    # Act: save to NPZ and then load back
    output_dir = tmp_path / "npz"
    save_keypoints_results(results_dict, output_dir=output_dir)

    loaded = load_keypoints_results(output_dir)

    # Assert: ensure the file is there and data is consistent
    assert filename in loaded

    loaded_xy, loaded_desc = loaded[filename]

    assert loaded_xy.shape == (2, 2)
    assert np.allclose(loaded_xy[0], np.array([10.0, 20.0], dtype=np.float32))
    assert np.allclose(loaded_xy[1], np.array([30.0, 40.0], dtype=np.float32))

    assert loaded_desc.shape == descriptors.shape
    assert np.array_equal(loaded_desc, descriptors)

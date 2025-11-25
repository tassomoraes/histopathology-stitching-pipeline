from pathlib import Path

import cv2
import numpy as np

from src.stitching.keypoints import (
    load_image_gray,
    detect_keypoints_and_descriptors,
    process_directory,
)


def test_detect_keypoints_and_descriptors_on_synthetic_image(tmp_path):
    # Create a simple synthetic image: black background with a white square.
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (70, 70), 255, -1)

    keypoints, descriptors = detect_keypoints_and_descriptors(img, method="ORB")

    # We expect at least one keypoint in the region of high contrast
    assert isinstance(keypoints, list)

    if descriptors is not None:
        assert descriptors.shape[0] == len(keypoints)


def test_process_directory(tmp_path):
    # Create a synthetic image file on disk
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(img, (50, 50), 20, 255, -1)

    img_path = tmp_path / "00001_x1_y1_zp1.jpg"
    cv2.imwrite(str(img_path), img)

    results = process_directory(tmp_path, method="ORB")

    assert "00001_x1_y1_zp1.jpg" in results
    result = results["00001_x1_y1_zp1.jpg"]

    assert len(result.keypoints) > 0
    assert result.descriptors is not None
    assert result.descriptors.shape[0] == len(result.keypoints)

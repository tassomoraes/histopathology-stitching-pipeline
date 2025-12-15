import numpy as np
import cv2

from src.stitching.preprocessing import compute_mean_background, subtract_background


def test_compute_mean_background_shape_and_dtype():
    img1 = np.full((50, 50, 3), 200, dtype=np.uint8)
    img2 = np.full((50, 50, 3), 100, dtype=np.uint8)

    images = {"a.jpg": img1, "b.jpg": img2}

    bg = compute_mean_background(images, blur_sigma=10.0)

    assert bg.shape == img1.shape
    assert bg.dtype == np.float32


def test_subtract_background_returns_uint8_and_preserves_shape():
    img = np.full((50, 50, 3), 180, dtype=np.uint8)
    images = {"a.jpg": img}

    bg = np.full((50, 50, 3), 100, dtype=np.float32)

    corrected = subtract_background(images, bg, recenter_value=128.0)

    assert "a.jpg" in corrected
    out = corrected["a.jpg"]

    assert out.shape == img.shape
    assert out.dtype == np.uint8

    # Expect values around 180 - 100 + 128 = 208
    assert abs(int(out[0, 0, 0]) - 208) <= 2

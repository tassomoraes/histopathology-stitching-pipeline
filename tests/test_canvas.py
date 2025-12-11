import numpy as np
import cv2

from src.stitching.canvas import (
    compute_canvas_spec,
    render_mosaic,
)


def test_compute_canvas_spec_with_translation():
    # Two images of the same size
    h, w = 50, 50
    img_shape = (h, w, 3)

    image_shapes = {
        "A.jpg": img_shape,
        "B.jpg": img_shape,
    }

    # A at origin (identity), B translated by (dx, dy)
    dx, dy = 30.0, 10.0
    H_a = np.eye(3, dtype=np.float64)
    H_b = np.array(
        [
            [1.0, 0.0, dx],
            [0.0, 1.0, dy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    transforms = {
        "A.jpg": H_a,
        "B.jpg": H_b,
    }

    spec = compute_canvas_spec(image_shapes, transforms)

    # The canvas must be large enough to contain both images
    assert spec.width >= w + dx - 1
    assert spec.height >= h + dy - 1

    # Offset matrix should be 3x3
    assert spec.offset_matrix.shape == (3, 3)


def test_render_mosaic_two_translated_squares():
    # Create two black images with a white square in the center
    h, w = 50, 50
    img_a = np.zeros((h, w), dtype=np.uint8)
    img_b = np.zeros((h, w), dtype=np.uint8)

    cv2.rectangle(img_a, (15, 15), (35, 35), 255, -1)
    cv2.rectangle(img_b, (15, 15), (35, 35), 255, -1)

    images = {
        "A.jpg": img_a,
        "B.jpg": img_b,
    }

    # Homographies: A at origin, B translated
    dx, dy = 20.0, 0.0
    H_a = np.eye(3, dtype=np.float64)
    H_b = np.array(
        [
            [1.0, 0.0, dx],
            [0.0, 1.0, dy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    transforms = {
        "A.jpg": H_a,
        "B.jpg": H_b,
    }

    # Compute canvas spec
    image_shapes = {
        "A.jpg": (h, w, 3),
        "B.jpg": (h, w, 3),
    }
    spec = compute_canvas_spec(image_shapes, transforms)

    mosaic = render_mosaic(images, transforms, spec)

    # Mosaic should be 3-channel
    assert mosaic.ndim == 3
    assert mosaic.shape[0] == spec.height
    assert mosaic.shape[1] == spec.width

    # Convert to grayscale for analysis
    mosaic_gray = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY)

    # We expect at least two distinct bright regions (two squares)
    # Total number of white pixels should be roughly 2 times the area of one square.
    white_pixels = (mosaic_gray > 200).sum()
    single_square_area = (35 - 15 + 1) * (35 - 15 + 1)

    assert white_pixels >= 2 * single_square_area * 0.8  # allow some tolerance

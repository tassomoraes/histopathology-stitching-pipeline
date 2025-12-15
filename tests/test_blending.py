import numpy as np
import cv2

from src.stitching.canvas import CanvasSpec, render_mosaic


def test_multiband_blending_runs_and_returns_correct_shape():
    # Two synthetic tiles with overlap
    h, w = 80, 80
    img_a = np.full((h, w, 3), 255, dtype=np.uint8)
    img_b = np.full((h, w, 3), 255, dtype=np.uint8)

    # draw different colored squares
    cv2.rectangle(img_a, (10, 10), (60, 60), (200, 0, 0), -1)
    cv2.rectangle(img_b, (10, 10), (60, 60), (0, 200, 0), -1)

    images = {"A.jpg": img_a, "B.jpg": img_b}

    # Put B shifted right so it overlaps A
    dx, dy = 30.0, 0.0
    H_a = np.eye(3, dtype=np.float64)
    H_b = np.array([[1.0, 0.0, dx],
                    [0.0, 1.0, dy],
                    [0.0, 0.0, 1.0]], dtype=np.float64)

    transforms = {"A.jpg": H_a, "B.jpg": H_b}

    canvas_spec = CanvasSpec(
        width=160,
        height=120,
        offset_matrix=np.eye(3, dtype=np.float64),
    )

    mosaic = render_mosaic(
        images=images,
        transforms=transforms,
        canvas_spec=canvas_spec,
        blend_mode="multiband",
        exposure_mode="GAIN",
        num_bands=5,
    )

    assert mosaic.shape == (canvas_spec.height, canvas_spec.width, 3)
    assert mosaic.dtype == np.uint8

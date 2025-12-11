import numpy as np

from src.stitching.mosaic import compute_global_transforms
from src.stitching.homography import HomographyResult


def _make_translation_H(dx: float, dy: float) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, dx],
            [0.0, 1.0, dy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def test_compute_global_transforms_chain():
    # Tiles: A -> B -> C, each with the same translation (dx, dy)
    dx, dy = 5.0, 7.0

    H_ab = _make_translation_H(dx, dy)      # maps A -> B
    H_bc = _make_translation_H(dx, dy)      # maps B -> C

    homographies = {
        ("A.jpg", "B.jpg"): HomographyResult(
            filename_a="A.jpg",
            filename_b="B.jpg",
            H=H_ab,
            inliers_mask=np.ones((4, 1), dtype=np.uint8),
            num_inliers=4,
            num_matches=4,
        ),
        ("B.jpg", "C.jpg"): HomographyResult(
            filename_a="B.jpg",
            filename_b="C.jpg",
            H=H_bc,
            inliers_mask=np.ones((4, 1), dtype=np.uint8),
            num_inliers=4,
            num_matches=4,
        ),
    }

    transforms = compute_global_transforms("A.jpg", homographies)

    # We expect transforms for A, B, C
    assert "A.jpg" in transforms
    assert "B.jpg" in transforms
    assert "C.jpg" in transforms

    T_a = transforms["A.jpg"]
    T_b = transforms["B.jpg"]
    T_c = transforms["C.jpg"]

    # All transforms should be 3x3
    assert T_a.shape == (3, 3)
    assert T_b.shape == (3, 3)
    assert T_c.shape == (3, 3)

    # A is the reference, so T[A] should be (approximately) identity
    T_a_norm = T_a / T_a[2, 2]
    assert np.allclose(T_a_norm, np.eye(3), atol=1e-6)

    # B and C should be translated in the opposite direction:
    # T[B] should contain roughly (-dx, -dy)
    # T[C] should contain roughly (-2*dx, -2*dy)
    T_b_norm = T_b / T_b[2, 2]
    T_c_norm = T_c / T_c[2, 2]

    assert np.allclose(T_b_norm[0, 2], -dx, atol=1e-6)
    assert np.allclose(T_b_norm[1, 2], -dy, atol=1e-6)

    assert np.allclose(T_c_norm[0, 2], -2 * dx, atol=1e-6)
    assert np.allclose(T_c_norm[1, 2], -2 * dy, atol=1e-6)

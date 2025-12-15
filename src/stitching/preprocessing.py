"""
Preprocessing utilities for background correction in tile stitching.

Goal:
- Estimate a low-frequency "background" from the mean of all tiles
- Subtract it from each tile to reduce illumination bias / shading
- Recenter intensities to avoid negative values and clipping artifacts
"""

from typing import Dict

import cv2
import numpy as np


def compute_mean_background(
    images: Dict[str, np.ndarray],
    blur_sigma: float = 50.0,
) -> np.ndarray:
    """
    Compute a smooth mean background image from multiple tiles.

    Args:
        images: filename -> image (uint8, BGR or grayscale).
        blur_sigma: Gaussian blur sigma to keep only low-frequency content.

    Returns:
        Background image as float32, same shape as input images.
    """
    if not images:
        raise ValueError("No images provided to compute_mean_background().")

    acc = None
    count = 0

    for img in images.values():
        img_f = img.astype(np.float32)

        if acc is None:
            acc = img_f.copy()
        else:
            acc += img_f
        count += 1

    mean_img = acc / float(count)

    # Smooth to remove tissue structure, keep only illumination bias
    background = cv2.GaussianBlur(mean_img, (0, 0), blur_sigma)

    return background.astype(np.float32)


def subtract_background(
    images: Dict[str, np.ndarray],
    background: np.ndarray,
    recenter_value: float = 128.0,
) -> Dict[str, np.ndarray]:
    """
    Subtract a background image from each tile and recenter intensities.

    Args:
        images: filename -> original image (uint8).
        background: background image (float32) with the same shape.
        recenter_value: constant added after subtraction to keep values in-range.

    Returns:
        Corrected images as uint8.
    """
    if not images:
        raise ValueError("No images provided to subtract_background().")

    corrected: Dict[str, np.ndarray] = {}

    for fname, img in images.items():
        img_f = img.astype(np.float32)

        if img_f.shape != background.shape:
            raise ValueError(
                f"Background shape {background.shape} does not match image shape {img_f.shape} for {fname}."
            )

        corrected_f = img_f - background + recenter_value
        corrected_f = np.clip(corrected_f, 0, 255)

        corrected[fname] = corrected_f.astype(np.uint8)

    return corrected

"""
Professional blending utilities for stitching:
- Exposure / gain compensation (photometric)
- Multi-band blending (seamless composition)

Requires opencv-contrib-python.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import cv2
import numpy as np


@dataclass
class WarpedInput:
    """
    Holds warped images and masks positioned in the common canvas.
    """
    warped_images: List[np.ndarray]   # uint8 BGR, each already warped to canvas coords
    warped_masks: List[np.ndarray]    # uint8 single-channel mask (0 or 255), same size as warped_images
    corners: List[Tuple[int, int]]    # top-left corner for each warped image in the canvas (usually (0,0) after warp)
    sizes: List[Tuple[int, int]]      # (width, height) for each warped image


def _make_warp_mask(warped_bgr: np.ndarray) -> np.ndarray:
    """
    Build a binary mask for a warped image. Pixels > 0 in any channel are considered valid.
    """
    valid = np.any(warped_bgr > 0, axis=2)
    return (valid.astype(np.uint8) * 255)


def prepare_warped_inputs(
    images: Dict[str, np.ndarray],
    transforms_global: Dict[str, np.ndarray],
    canvas_size: Tuple[int, int],
) -> WarpedInput:
    """
    Warp all images to a common canvas space and create masks.

    Args:
        images: filename -> BGR or grayscale image
        transforms_global: filename -> 3x3 homography mapping image -> canvas
        canvas_size: (width, height)

    Returns:
        WarpedInput with warped images, masks, corners and sizes.
    """
    width, height = canvas_size

    warped_images: List[np.ndarray] = []
    warped_masks: List[np.ndarray] = []
    corners: List[Tuple[int, int]] = []
    sizes: List[Tuple[int, int]] = []

    for fname, img in images.items():
        H = transforms_global.get(fname)
        if H is None:
            continue

        if img.ndim == 2:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = img

        warped = cv2.warpPerspective(
            img_bgr,
            H,
            dsize=(width, height),
            flags=cv2.INTER_LINEAR,
        )

        mask = _make_warp_mask(warped)

        warped_images.append(warped)
        warped_masks.append(mask)
        corners.append((0, 0))
        sizes.append((width, height))

    if not warped_images:
        raise ValueError("No warped images were produced. Check transforms and input images.")

    return WarpedInput(
        warped_images=warped_images,
        warped_masks=warped_masks,
        corners=corners,
        sizes=sizes,
    )


def apply_exposure_compensation(
    warped: WarpedInput,
    mode: str = "GAIN",
) -> None:
    """
    Apply exposure/gain compensation in-place over warped images.

    Args:
        warped: WarpedInput (warped_images will be modified in-place)
        mode: "GAIN" or "GAIN_BLOCKS"
    """
    mode = mode.upper()
    if mode == "GAIN":
        comp_type = cv2.detail.ExposureCompensator_GAIN
    elif mode == "GAIN_BLOCKS":
        comp_type = cv2.detail.ExposureCompensator_GAIN_BLOCKS
    else:
        raise ValueError(f"Unsupported exposure compensator mode: {mode}")

    compensator = cv2.detail.ExposureCompensator_createDefault(comp_type)
    compensator.feed(warped.corners, warped.warped_images, warped.warped_masks)

    for i in range(len(warped.warped_images)):
        compensator.apply(i, warped.corners[i], warped.warped_images[i], warped.warped_masks[i])


def multiband_blend(
    warped: WarpedInput,
    num_bands: int = 5,
) -> np.ndarray:
    """
    Multi-band blending to produce a seamless mosaic.

    Args:
        warped: WarpedInput with images/masks in the same canvas space
        num_bands: number of pyramid bands

    Returns:
        final mosaic as uint8 BGR
    """
    blender = cv2.detail_MultiBandBlender()
    blender.setNumBands(int(num_bands))

    # Prepare the blender ROI to the full canvas
    width, height = warped.sizes[0]
    blender.prepare((0, 0, width, height))

    for img, mask, corner in zip(warped.warped_images, warped.warped_masks, warped.corners):
        blender.feed(img.astype(np.int16), mask, corner)

    result, result_mask = blender.blend(None, None)

    # result is int16; convert to uint8
    result_u8 = np.clip(result, 0, 255).astype(np.uint8)
    return result_u8

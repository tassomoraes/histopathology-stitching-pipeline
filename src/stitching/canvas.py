"""
Canvas creation and mosaic rendering for transformed tiles.

This module assumes that for each image we have a 3x3 homography
matrix H that maps image coordinates (x, y) into a common canvas
coordinate system.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np


@dataclass
class CanvasSpec:
    """
    Specification of the global canvas where all tiles will be rendered.

    Attributes:
        width: Width of the canvas in pixels.
        height: Height of the canvas in pixels.
        offset_matrix: 3x3 translation matrix that shifts all coordinates
                       so that the minimum (x, y) becomes (0, 0).
    """
    width: int
    height: int
    offset_matrix: np.ndarray


def _image_corners(width: int, height: int) -> np.ndarray:
    """
    Return the four corners of an image in homogeneous coordinates.

    Order: (0,0), (w,0), (w,h), (0,h)
    """
    corners = np.array(
        [
            [0, 0],
            [width, 0],
            [width, height],
            [0, height],
        ],
        dtype=np.float32,
    )
    return corners


def _transform_points(
    points_xy: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    """
    Apply homography H to a set of (x, y) points.

    Args:
        points_xy: (N, 2) array.
        H: 3x3 homography matrix.

    Returns:
        (N, 2) array of transformed points.
    """

    # Reshape for cv2.perspectiveTransform
    pts = points_xy.reshape(-1, 1, 2)

    # Apply transformation
    pts_transformed = cv2.perspectiveTransform(pts, H)

    return pts_transformed.reshape(-1, 2)


def compute_canvas_spec(
    image_shapes: Dict[str, Tuple[int, int, int]],
    transforms: Dict[str, np.ndarray],
) -> CanvasSpec:
    """
    Compute the global canvas size and an offset matrix based on
    the transformed corners of all images.

    Args:
        image_shapes: Mapping filename -> image.shape (H, W, C).
        transforms: Mapping filename -> 3x3 homography matrix that
                    maps image coordinates into a common space.

    Returns:
        CanvasSpec with width, height and an offset_matrix that should
        be pre-multiplied to each homography before warping.
    """

    all_x = []
    all_y = []

    # Iterate over all images to find transformed corners
    for filename, shape in image_shapes.items():
        
        # Get homography for this image
        H = transforms.get(filename)

        # Skip if no transform
        if H is None:
            continue

        # Get image corners
        h, w = shape[0], shape[1]
        corners = _image_corners(w, h)
        transformed = _transform_points(corners, H)

        # Collect all x and y coordinates
        all_x.extend(transformed[:, 0].tolist())
        all_y.extend(transformed[:, 1].tolist())

    # Ensure we found some corners
    if not all_x or not all_y:
        raise ValueError("No transformed corners found. Check transforms/input.")

    # Compute bounding box
    min_x = min(all_x)
    min_y = min(all_y)
    max_x = max(all_x)
    max_y = max(all_y)

    # Compute canvas size with a small margin.
    width = int(np.ceil(max_x - min_x))
    height = int(np.ceil(max_y - min_y))

    # Translation to bring (min_x, min_y) -> (0, 0)
    tx = -min_x
    ty = -min_y

    # Build offset matrix
    offset_matrix = np.array(
        [
            [1.0, 0.0, tx],
            [0.0, 1.0, ty],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return CanvasSpec(
        width=width,
        height=height,
        offset_matrix=offset_matrix,
    )


def render_mosaic(
    images: Dict[str, np.ndarray],
    transforms: Dict[str, np.ndarray],
    canvas_spec: CanvasSpec,
) -> np.ndarray:
    """
    Warp and composite all images into a single mosaic canvas.

    Args:
        images: Mapping filename -> image array (H, W, C or H, W).
        transforms: Mapping filename -> 3x3 homography matrix that
                    maps image coordinates into the common space.
        canvas_spec: CanvasSpec with width, height and offset_matrix.

    Returns:
        The final mosaic as a NumPy array (uint8).
    """

    # Unpack canvas spec
    height = canvas_spec.height
    width = canvas_spec.width
    T = canvas_spec.offset_matrix

    # Initialize empty canvas (assume 3 channels; convert if needed)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)  # white background


    # Warp and composite each image
    for filename, img in images.items():
        
        # Get homography for this image
        H = transforms.get(filename)

        # Skip if no transform
        if H is None:
            continue

        # Ensure 3-channel image
        if img.ndim == 2:
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_color = img

        # Apply global offset: H' = T * H
        H_global = T @ H

        # Warp image into canvas space
        warped = cv2.warpPerspective(
            img_color,
            H_global,
            dsize=(width, height),
            flags=cv2.INTER_LINEAR,
        )

        # Simple overwrite compositing:
        # wherever warped has non-zero pixels, copy to canvas.
        mask = np.any(warped > 0, axis=2)
        canvas[mask] = warped[mask]

    return canvas

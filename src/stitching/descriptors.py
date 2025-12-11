"""
Descriptor persistence utilities (NPZ save/load helpers).

This module does not compute keypoints or descriptors.
Instead, it is responsible for serializing and deserializing
the results produced by `keypoints.py`.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .keypoints import KeypointResult


def _keypoints_to_xy_array(result: KeypointResult) -> np.ndarray:
    """
    Convert the list of cv2.KeyPoint objects into a (N, 2) array of (x, y).

    Args:
        result: KeypointResult instance.

    Returns:
        NumPy array of shape (N, 2) with float32 coordinates.
    """

    # Handle the case with zero keypoints
    if not result.keypoints:
        return np.empty((0, 2), dtype=np.float32)

    # Convert keypoints to (N, 2) array
    coords = np.array(
        [[kp.pt[0], kp.pt[1]] for kp in result.keypoints],
        dtype=np.float32,
    )
    return coords


def save_keypoints_results(
    results: Dict[str, KeypointResult],
    output_dir: Path,
) -> None:
    """
    Save keypoints and descriptors for each image as compressed NPZ files.

    Each file will contain:
        - filename: original image filename (string)
        - keypoints_xy: (N, 2) float32 array
        - descriptors: (N, D) array (dtype depends on the feature method)

    Args:
        results: Mapping from filename -> KeypointResult.
        output_dir: Directory where .npz files will be written.
    """

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save each result
    for filename, result in results.items():
        
        # Convert keypoints to (N, 2) array
        keypoints_xy = _keypoints_to_xy_array(result)
        descriptors = result.descriptors

        # We still save the file even if there are zero keypoints,
        # so the pipeline can rely on the presence of the NPZ.
        stem = Path(filename).stem
        out_path = output_dir / f"{stem}_features.npz"

        np.savez_compressed(
            out_path,
            filename=filename,
            keypoints_xy=keypoints_xy,
            descriptors=descriptors,
        )


def load_keypoints_results(
    input_dir: Path,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load keypoint coordinates and descriptors from NPZ files.

    Returns:
        A dictionary mapping filename -> (keypoints_xy, descriptors), where:
            - keypoints_xy: (N, 2) float32 array
            - descriptors: (N, D) array
    """

    # Initialize results dictionary
    results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # Load each NPZ file
    for npz_path in sorted(input_dir.glob("*_features.npz")):
        data = np.load(npz_path)

        # filename was saved as a scalar array; .item() gets the Python string
        filename = data["filename"].item()
        keypoints_xy = data["keypoints_xy"]
        descriptors = data["descriptors"]

        results[filename] = (keypoints_xy, descriptors)

    return results

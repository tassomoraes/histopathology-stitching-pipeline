"""
Keypoint detection utilities (e.g., SIFT/ORB wrappers).
"""

"""
Keypoint detection utilities (e.g., ORB/SIFT wrappers).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Literal

import cv2
import numpy as np

FeatureMethod = Literal["ORB", "SIFT"]


@dataclass
class KeypointResult:
    """
    Container for keypoints and descriptors of a single image.
    """
    filename: str
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray


def create_feature_detector(method: FeatureMethod = "ORB"):
    """
    Create and return a feature detector/descriptor extractor.

    Args:
        method: "ORB" (default) or "SIFT".

    Returns:
        An OpenCV feature detector/descriptor extractor instance.
    """
    method = method.upper()
    if method == "ORB":
        return cv2.ORB_create()
    elif method == "SIFT":
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError(
                "SIFT is not available in this OpenCV build. "
                "Try installing opencv-contrib-python or use ORB."
            )
        return cv2.SIFT_create()
    else:
        raise ValueError(f"Unsupported feature method: {method}")


def load_image_gray(path: Path) -> np.ndarray:
    """
    Load an image from disk in grayscale format.

    Args:
        path: Path to the image file.

    Returns:
        Grayscale image as a NumPy array.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def detect_keypoints_and_descriptors(
    image: np.ndarray,
    method: FeatureMethod = "ORB",
) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    Detect keypoints and compute descriptors for a single image.

    Args:
        image: Grayscale image as a NumPy array.
        method: "ORB" (default) or "SIFT".

    Returns:
        keypoints: list of cv2.KeyPoint objects.
        descriptors: NumPy array of shape (N, D), where N is the number
                    of keypoints and D is the descriptor dimension.
    """
    detector = create_feature_detector(method)
    keypoints, descriptors = detector.detectAndCompute(image, None)
    
    # Normalize to a Python list for a stable interface
    if keypoints is None:
        keypoints = []
    else:
        # OpenCV may return a tuple; we enforce a list
        keypoints = list(keypoints)

    return keypoints, descriptors


def process_directory(
    image_dir: Path,
    method: FeatureMethod = "ORB",
) -> Dict[str, KeypointResult]:
    """
    Detect keypoints and descriptors for all images in a directory.

    Args:
        image_dir: Directory containing the tile images.
        method: "ORB" (default) or "SIFT".

    Returns:
        A dictionary mapping filename -> KeypointResult.
    """
    results: Dict[str, KeypointResult] = {}

    for path in sorted(image_dir.iterdir()):
        if not path.is_file():
            continue

        img = load_image_gray(path)
        keypoints, descriptors = detect_keypoints_and_descriptors(
            img,
            method=method,
        )

        results[path.name] = KeypointResult(
            filename=path.name,
            keypoints=keypoints,
            descriptors=descriptors,
        )

    return results

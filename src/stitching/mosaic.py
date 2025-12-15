"""
High-level orchestration of the stitching pipeline for a small grid of tiles.

This module provides two main capabilities:

1. Compute global transforms for each tile from pairwise homographies.
2. Run a minimal end-to-end pipeline to build a mosaic from a directory
   of tiles (graph -> keypoints -> matching -> homography -> canvas).
"""

from pathlib import Path
from typing import Dict, Tuple, Literal, Optional, List

import cv2
import numpy as np

from .graph import build_graph
from .keypoints import process_directory, KeypointResult
from .matching import match_neighbors
from .homography import estimate_homographies, HomographyResult
from .canvas import compute_canvas_spec, render_mosaic
from .preprocessing import compute_mean_background, subtract_background


FeatureMethod = Literal["ORB", "SIFT"]


def _keypoints_results_to_xy(
    keypoints_results: Dict[str, KeypointResult],
) -> Dict[str, np.ndarray]:
    """
    Convert KeypointResult objects into (N, 2) arrays of (x, y) coordinates.

    Args:
        keypoints_results: Mapping filename -> KeypointResult.

    Returns:
        Mapping filename -> (N, 2) float32 array.
    """

    # Prepare output map
    xy_map: Dict[str, np.ndarray] = {}

    # Convert each result
    for filename, result in keypoints_results.items():
        if not result.keypoints:
            xy_map[filename] = np.empty((0, 2), dtype=np.float32)
        else:
            coords = np.array(
                [[kp.pt[0], kp.pt[1]] for kp in result.keypoints],
                dtype=np.float32,
            )
            xy_map[filename] = coords

    return xy_map


def compute_global_transforms(
    reference_filename: str,
    homographies: Dict[Tuple[str, str], HomographyResult],
) -> Dict[str, np.ndarray]:
    """
    Compute a global transform for each tile, expressed in the coordinate
    system of a chosen reference tile.

    We define, for each tile X, a 3x3 matrix T[X] such that:

        [x_ref] ~ T[X] * [x_X]

    i.e., T[X] maps coordinates from tile X into the reference tile system.

    The algorithm works by:

        1. Building an adjacency list from the pairwise homographies.
        2. Running a BFS starting from the reference tile.
        3. Propagating transforms along the graph.

    For a pair (A, B) with a homography H_ab such that:

        [x_b] ~ H_ab * [x_a]

    we construct adjacency entries:

        A -> (B, H_ab)
        B -> (A, inv(H_ab))

    Then, if S[X] represents the transform from reference -> X, we have:

        S[ref] = I
        S[neighbor] = H_current_to_neighbor * S[current]

    Finally, the desired T[X] (tile X -> reference) is simply inv(S[X]).

    Args:
        reference_filename: The filename of the tile chosen as reference.
        homographies: Mapping (filename_a, filename_b) -> HomographyResult.

    Returns:
        Mapping filename -> 3x3 transform matrix T[X].
    """
    # Build adjacency: filename -> list of (neighbor, H_curr_to_neighbor)
    adjacency: Dict[str, List[Tuple[str, np.ndarray]]] = {}

    for (fa, fb), hres in homographies.items():
        H_ab = hres.H
        # Ensure float64
        H_ab = H_ab.astype(np.float64)
        H_ba = np.linalg.inv(H_ab)

        adjacency.setdefault(fa, []).append((fb, H_ab))
        adjacency.setdefault(fb, []).append((fa, H_ba))

    if reference_filename not in adjacency and homographies:
        # If the chosen reference is not in any homography pair,
        # the graph may be disconnected. We still allow this but
        # only nodes reachable from the reference will get transforms.
        pass

    # S[X]: transform from reference -> X
    S: Dict[str, np.ndarray] = {}
    S[reference_filename] = np.eye(3, dtype=np.float64)

    queue: List[str] = [reference_filename]

    while queue:
        current = queue.pop(0)
        current_S = S[current]

        for neighbor, H_curr_to_neighbor in adjacency.get(current, []):
            if neighbor in S:
                continue
            # S[neighbor] = H_curr_to_neighbor * S[current]
            S[neighbor] = H_curr_to_neighbor @ current_S
            queue.append(neighbor)

    # T[X] = inv(S[X])  (tile -> reference coordinate system)
    transforms: Dict[str, np.ndarray] = {}
    for fname, Sx in S.items():
        transforms[fname] = np.linalg.inv(Sx)

    return transforms


def run_full_pipeline(
    tiles_dir: Path,
    feature_method: FeatureMethod = "ORB",
) -> np.ndarray:
    """
    Run the full stitching pipeline on a directory of tiles.

    Steps:
        1. Build the tile graph from filenames.
        2. Detect keypoints and descriptors for each image.
        3. Compute neighbor matches based on the graph.
        4. Estimate homographies for each neighboring pair.
        5. Choose a reference tile and compute global transforms.
        6. Create the global canvas and render the mosaic.

    Args:
        tiles_dir: Directory containing the tile images.
        feature_method: "ORB" (default) or "SIFT".

    Returns:
        The final mosaic as a NumPy array (uint8, H x W x 3).
    """
    # 1. Graph
    graph = build_graph(tiles_dir)

    # 2. Keypoints & descriptors
    keypoints_results = process_directory(tiles_dir, method=feature_method)

    # 3. Matches between neighbors
    matches = match_neighbors(keypoints_results, graph, ratio=0.75)

    # 4. Homographies between neighbors
    keypoints_xy = _keypoints_results_to_xy(keypoints_results)
    homographies = estimate_homographies(
        keypoints_xy=keypoints_xy,
        matches=matches,
        ransac_thresh=3.0,
        min_matches=4,
    )

    if not homographies:
        raise RuntimeError("No homographies were estimated. Check matches and keypoints.")

    # 5. Choose a reference tile (first one in graph) and compute global transforms
    reference_filename = next(iter(graph.keys()))
    transforms = compute_global_transforms(reference_filename, homographies)

    # 6. Load images, compute canvas spec, render mosaic
    images: Dict[str, np.ndarray] = {}
    image_shapes: Dict[str, Tuple[int, int, int]] = {}

    for path in tiles_dir.iterdir():
        if not path.is_file():
            continue
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        images[path.name] = img
        h, w, c = img.shape
        image_shapes[path.name] = (h, w, c)

    # Background correction (optional but recommended for histopathology tiles)
    background = compute_mean_background(images, blur_sigma=50.0)
    images = subtract_background(images, background, recenter_value=128.0)

    # Keep only transforms for which we actually loaded images
    transforms_for_loaded = {
        fname: H for fname, H in transforms.items() if fname in images
    }

    if not transforms_for_loaded:
        raise RuntimeError("No transforms available for loaded images.")

    canvas_spec = compute_canvas_spec(image_shapes, transforms_for_loaded)
    mosaic = render_mosaic(images, transforms, canvas_spec, blend_mode="multiband")

    return mosaic

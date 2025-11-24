"""
Graph construction utilities for the tile grid (e.g., 3x3 layout).
"""

import re
from pathlib import Path
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)

COORD_PATTERN = re.compile(r"_x(\d+)_y(\d+)")

def extract_coords(filename: str) -> Tuple[int, int]:
    """
    Extract (x, y) coordinates from a filename of the form:
    00001_x1_y1_zp1.jpg

    Returns:
        A tuple of (x, y) coordinates as integers.
    """

    match = COORD_PATTERN.search(filename)
    # logger.info(f"Debug match variable: {match}")

    if not match:
        raise ValueError(f"Filename {filename} does not match expected pattern.")
    
    x = int(match.group(1))
    y = int(match.group(2))
    
    return x, y

def build_graph(image_dir: Path) -> Dict[str, Dict]:
    """
    Build a graph representation of all tiles found in the directory.

    The graph maps each filename to:
        - its (x, y) coordinates
        - its neighbor filenames

    Example return format:
    {
        "00001_x1_y1_zp1.jpg": {
            "coords": (1, 1),
            "neighbors": ["00002_x2_y1_zp1.jpg", ...]
        },
        ...
    }
    """
    # List all image files
    files = sorted([f for f in image_dir.iterdir() if f.is_file()])
    # logger.info(f"Files in directory: {[f.name for f in files]}")

    # Map filename → coordinates
    coords_map = {}
    for f in files:
        x, y = extract_coords(f.name)
        coords_map[(x, y)] = f.name

    # Build graph
    graph = {}
    for (x, y), filename in coords_map.items():
        neighbors = []

        # left
        if (x - 1, y) in coords_map:
            neighbors.append(coords_map[(x - 1, y)])

        # right
        if (x + 1, y) in coords_map:
            neighbors.append(coords_map[(x + 1, y)])

        # up
        if (x, y - 1) in coords_map:
            neighbors.append(coords_map[(x, y - 1)])

        # down
        if (x, y + 1) in coords_map:
            neighbors.append(coords_map[(x, y + 1)])

        graph[filename] = {
            "coords": (x, y),
            "neighbors": neighbors
        }

    return graph
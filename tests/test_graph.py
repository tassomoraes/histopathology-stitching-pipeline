"""
Basic tests for the graph module.
"""

from src.stitching.graph import extract_coords, build_graph
from pathlib import Path

def test_extract_coords():
    assert extract_coords("00001_x3_y5_zp1.jpg") == (3, 5)

def test_build_graph(tmp_path):
    # Create mock files
    filenames = [
        "00001_x1_y1_zp1.jpg",
        "00002_x2_y1_zp1.jpg",
        "00003_x1_y2_zp1.jpg",
    ]

    for name in filenames:
        (tmp_path / name).write_text("test")

    graph = build_graph(tmp_path)

    # Basic checks
    assert "00001_x1_y1_zp1.jpg" in graph
    assert graph["00001_x1_y1_zp1.jpg"]["coords"] == (1, 1)

    # Check neighbors
    neighbors = graph["00001_x1_y1_zp1.jpg"]["neighbors"]
    assert "00002_x2_y1_zp1.jpg" in neighbors
    assert "00003_x1_y2_zp1.jpg" in neighbors

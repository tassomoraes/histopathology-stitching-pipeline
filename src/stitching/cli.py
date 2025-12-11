"""
Command-line interface for the mini stitching pipeline.

Usage example:

    python -m src.stitching.cli --tiles-dir data/raw --output data/results/mosaic.png
"""

import argparse
from pathlib import Path
from typing import Optional

import cv2

from .mosaic import run_full_pipeline, FeatureMethod


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Args:
        argv: Optional list of argument strings. If None, uses sys.argv.

    Returns:
        Parsed arguments as an argparse.Namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run a small histopathology image stitching pipeline.",
    )

    parser.add_argument(
        "--tiles-dir",
        type=Path,
        required=True,
        help="Directory containing the tile images (e.g., 9 tiles in a grid).",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the output mosaic image (e.g., data/results/mosaic.png).",
    )

    parser.add_argument(
        "--feature-method",
        type=str,
        default="ORB",
        choices=["ORB", "SIFT"],
        help="Feature detector/descriptor to use (default: ORB).",
    )

    return parser.parse_args(argv)


def run_from_args(args: argparse.Namespace) -> None:
    """
    Run the stitching pipeline based on parsed CLI arguments.

    Args:
        args: Parsed arguments.
    """
    tiles_dir: Path = args.tiles_dir
    output_path: Path = args.output
    feature_method: FeatureMethod = args.feature_method.upper()  # type: ignore

    if not tiles_dir.exists() or not tiles_dir.is_dir():
        raise FileNotFoundError(f"Tiles directory does not exist or is not a directory: {tiles_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    mosaic = run_full_pipeline(tiles_dir=tiles_dir, feature_method=feature_method)

    # Save mosaic as PNG/JPEG depending on extension
    success = cv2.imwrite(str(output_path), mosaic)
    if not success:
        raise RuntimeError(f"Failed to write mosaic image to: {output_path}")


def main(argv: Optional[list[str]] = None) -> None:
    """
    Entry point for the CLI.

    Args:
        argv: Optional list of argument strings. If None, uses sys.argv.
    """
    args = parse_args(argv)
    run_from_args(args)


if __name__ == "__main__":
    main()

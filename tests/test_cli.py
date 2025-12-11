from pathlib import Path

from src.stitching.cli import parse_args


def test_parse_args_minimal():
    argv = [
        "--tiles-dir",
        "data/raw",
        "--output",
        "data/results/mosaic.png",
        "--feature-method",
        "ORB",
    ]

    args = parse_args(argv)

    assert isinstance(args.tiles_dir, Path)
    assert isinstance(args.output, Path)
    assert args.tiles_dir == Path("data/raw")
    assert args.output == Path("data/results/mosaic.png")
    assert args.feature_method == "ORB"

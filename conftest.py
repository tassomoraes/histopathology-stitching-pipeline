import sys
from pathlib import Path

# Project root = directory where this conftest.py lives
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

# Ensure src/ is in sys.path so that `import src.stitching...` works
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

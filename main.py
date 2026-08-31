"""Convenience entry point for the legacy model-comparison pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from main import run


if __name__ == "__main__":
    run()

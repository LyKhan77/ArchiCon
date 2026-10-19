#!/usr/bin/env python3
"""Archive/verify/restore approved historical outputs without deletion."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from room_reconstruction.output_archive import main

if __name__ == "__main__":
    main()

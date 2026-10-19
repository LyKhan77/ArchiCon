#!/usr/bin/env python3
"""Single-command video reconstruction with a manual visual-review handoff."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from room_reconstruction.pipeline_cli import main

if __name__=='__main__':raise SystemExit(main(ROOT))

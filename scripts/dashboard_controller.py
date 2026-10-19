#!/usr/bin/env python3
"""Detached controller retaining ArchiCon's inherited kernel lock."""
import argparse
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock-fd',type=int,required=True)
    parser.add_argument('--job-id',required=True)
    args=parser.parse_args()
    from room_reconstruction.pipeline_cli import main
    # Descriptor remains locked through the full controller lifespan, even if UI/backend exits.
    with os.fdopen(args.lock_fd,'a+'):
        raise SystemExit(main(ROOT,['resume','--job-id',args.job_id]))

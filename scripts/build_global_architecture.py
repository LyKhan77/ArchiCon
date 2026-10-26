#!/usr/bin/env python3
"""Align compatible camera worlds and export an optional architecture GLB candidate."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'scripts'))
from room_reconstruction.provenance import create_run_dir, write_json
from room_reconstruction.pipeline_state import workspace_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--run-id', required=True)
    args = parser.parse_args(argv)
    run = None
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[name] = '4'
        run = create_run_dir(ROOT, args.run_id)
        from fuse_tsdf import cpu_guard, check_rss
        cpu_guard(run)
        # The resource report belongs to the stage, while build requires an empty child.
        products = run / 'products'; products.mkdir()
        from room_reconstruction.global_architecture import build_architecture
        config = json.loads(workspace_path(ROOT, args.config).read_text())
        report = build_architecture(ROOT, products, config, memory_guard=check_rss)
        write_json(run / 'report.json', dict(report, products='products'))
        write_json(run / 'status.json', dict(stage='global_architecture', status='complete',
                                           alignment_status=report['status']))
        print(json.dumps(dict(status=report['status'], aligned_components=report['aligned_components'],
                              fallback_components=report['fallback_components'])))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        if run is not None:
            write_json(run / 'status.json', dict(stage='global_architecture', status='failed',
                                               reason=str(exc), error_type=type(exc).__name__))
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

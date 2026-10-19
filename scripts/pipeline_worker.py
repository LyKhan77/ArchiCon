#!/usr/bin/env python3
"""Detached server worker; durable status survives an interrupted SSH client."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from room_reconstruction.pipeline_state import FileLock, StageRunner, WaitingResources, atomic_json
from room_reconstruction.pipeline_steps import run_server
from room_reconstruction.provenance import load_yaml
from room_reconstruction.pipeline_layout import job_dir,job_asset


def process_job(root, job_id):
    root = Path(root).resolve()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', job_id):
        raise ValueError('Safe job ID required')
    directory = job_dir(root,job_id)
    if not directory.is_relative_to(root / 'outputs'):
        raise ValueError('Job escapes workspace outputs')
    if not (directory / 'job.json').is_file(): raise ValueError('Missing job.json')
    lock = FileLock(directory / 'worker.lock')
    try:
        lock.__enter__()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 73
    try:
        status = directory / 'worker-status.json'
        def update(execution_status, reason='', bundle=None):
            atomic_json(status, dict(execution_status=execution_status, reason=reason, bundle=bundle,
                                    worker_pid=os.getpid(), updated_at=datetime.now(timezone.utc).isoformat()))
        def log(message):
            logfile=job_asset(root,directory,'logs','worker.log');logfile.parent.mkdir(parents=True,exist_ok=True)
            with logfile.open('a') as stream:
                stream.write(datetime.now(timezone.utc).isoformat() + ' ' + message + '\n')
        update('running'); log('Worker started: ' + job_id)
        try:
            job = json.loads((directory / 'job.json').read_text())
            if job.get('job_id') != job_id: raise ValueError('Job directory/manifest identity mismatch')
            value = Path(job['machine_config'])
            config_path = (root / value).resolve()
            if value.is_absolute() or not config_path.is_relative_to(root):
                raise ValueError('Machine configuration must remain within workspace')
            config = load_yaml(config_path)
            if config.get('profile') != 'server' or Path(config['paths']['project_root']).resolve() != root:
                raise ValueError('Worker requires a server machine configuration for this root')
            bundle = run_server(root, directory, job, StageRunner(root, directory, config))
            atomic_json(directory / 'bundle.json', bundle)
            update('complete', bundle=str((directory / 'bundle.json').relative_to(root)))
            log('Worker completed: bundle verified')
            return 0
        except WaitingResources as exc:
            update('waiting_resources', str(exc)); log('Waiting resources: ' + str(exc))
            return 75
        except (Exception, KeyboardInterrupt) as exc:
            update('failed', str(exc)); log(traceback.format_exc())
            print(str(exc), file=sys.stderr)
            return 1
    finally:
        lock.__exit__(None, None, None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id', '--job', dest='job_id', required=True)
    args = parser.parse_args(argv)
    try:
        return process_job(ROOT, args.job_id)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__': raise SystemExit(main())

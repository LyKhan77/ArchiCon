"""Small, read-only progress snapshots; never serialize numerical arrays/specs."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from .pipeline_layout import job_dir, resolve_run_dir


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {} if default is None else default


def tail(path, limit=16000):
    try:
        with Path(path).open('rb') as stream:
            stream.seek(0, 2); stream.seek(max(0, stream.tell()-limit))
            return stream.read(limit).decode('utf-8', errors='replace')
    except OSError:
        return ''


def snapshot(root, job_id):
    root=Path(root).resolve(); directory=job_dir(root, job_id)
    if not (directory/'job.json').is_file():
        raise FileNotFoundError('Job belum tersedia')
    stages=[]; previews=[]; last_run=None; camera_coverage=None
    for name, attempts in read_json(directory/'stages.json').items():
        if not attempts: continue
        current=attempts[-1]; rid=current.get('run_id', '')
        stages.append(dict(stage=name, run_id=rid, status=current.get('status', 'unknown'),
                           attempts=len(attempts), elapsed_seconds=current.get('elapsed_seconds'),
                           reason=str(current.get('reason', ''))[:1500]))
        if rid:
            last_run=rid
            try:
                run=resolve_run_dir(root, rid)
                if name.startswith('S03'):
                    report=read_json(run/'report.json')
                    if 'selected_frame_count' in report and 'accepted_union_count' in report:
                        camera_coverage=dict(attempted=report['selected_frame_count'],processed=report['accepted_union_count'],
                                             gaps=report.get('missing_frame_ids',[]),full_room_reconstruction_complete=False)
                if name.startswith('S02'):
                    images=sorted((run/'review').glob('overview*.jpg'))[:2]
                elif name.startswith('S06R'):
                    images=[run/'overview.png', run/'full_path_review.jpg']
                else: images=[]
                for image in images:
                    if image.is_file() and not image.is_symlink() and image.stat().st_size<=8*1024**2:
                        previews.append(dict(path=str(image.relative_to(root)), label='Frame sumber' if name.startswith('S02') else 'Preview rekonstruksi'))
            except (OSError, ValueError):
                pass
    logs=directory/'logs' if (directory/'logs').is_dir() else directory
    log=tail(logs/(last_run+'.log')) if last_run else ''
    if not log: log=tail(logs/'worker.log')
    bundle=read_json(directory/'bundle.json')
    return dict(updated_at=datetime.now(timezone.utc).isoformat(), worker=read_json(directory/'worker-status.json'),
                stages=stages, coverage=bundle.get('coverage') or camera_coverage, previews=previews[:6], log=log)

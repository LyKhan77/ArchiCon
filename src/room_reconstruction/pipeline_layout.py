"""Job-scoped outputs with immutable run-ID routing and legacy discovery."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile


def _id(value,limit=80):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,'+str(limit-1)+'}',value):
        raise ValueError('Unsafe job/run identifier')


def validate_job_id(value,*,legacy=False):
    _id(value,80 if legacy else 32)
    if value=='jobs' or not legacy and value in ('part1','part2'):
        raise ValueError('Reserved job identifier')


def _safe(root,path):
    root=Path(root).resolve();path=Path(path)
    if not path.resolve().is_relative_to(root/'outputs'):
        raise ValueError('Output path escapes workspace outputs')
    for parent in [path,*path.parents]:
        if parent==root:break
        if parent.is_symlink():raise ValueError('Output symlinks are not supported')
    return path


def job_dir(root,job_id):
    root=Path(root).resolve();validate_job_id(job_id,legacy=True)
    new=_safe(root,root/'outputs/jobs'/job_id);old=_safe(root,root/'outputs'/job_id)
    if new.exists() and old.exists():raise ValueError('Ambiguous legacy/new job directory')
    return old if old.exists() else new


def create_job_dir(root,job_id):
    validate_job_id(job_id);directory=job_dir(root,job_id)
    if directory.exists():raise FileExistsError('Job already exists: '+job_id)
    directory.mkdir(parents=True,exist_ok=False)
    return directory


def is_nested_job(root,directory):
    return Path(directory).resolve().parent==Path(root).resolve()/'outputs/jobs'


def job_asset(root,directory,category,name):
    if category not in ('config','logs','qa'):raise ValueError('Unknown job asset category')
    _id(Path(name).stem)
    if Path(name).name!=name:raise ValueError('Job asset must be a basename')
    directory=Path(directory)
    return directory/category/name if is_nested_job(root,directory) else directory/name


def run_index_path(root,run_id):
    _id(run_id)
    return _safe(Path(root).resolve(),Path(root).resolve()/'outputs/.run-index'/(run_id+'.json'))


def resolve_run_dir(root,run_id):
    root=Path(root).resolve();_id(run_id)
    old=_safe(root,root/'outputs'/run_id);index=run_index_path(root,run_id)
    if not index.exists():return old
    if old.exists():raise ValueError('Ambiguous legacy/indexed run identifier')
    doc=json.loads(index.read_text());value=Path(doc['path'])
    if doc.get('schema_version')!=1 or value.is_absolute() or '..' in value.parts:
        raise ValueError('Invalid run registry entry')
    path=_safe(root,root/value)
    parts=value.parts
    if len(parts)!=6 or parts[:2]!=('outputs','jobs') or parts[3]!='stages' or path.name!=run_id:
        raise ValueError('Run registry must point to its job-scoped run')
    validate_job_id(parts[2]);_id(parts[4],56)
    return path


def run_id_exists(root,run_id):
    return run_index_path(root,run_id).exists() or (Path(root)/'outputs'/run_id).exists()


def reserve_run(root,directory,stage,run_id):
    root=Path(root).resolve();directory=_safe(root,Path(directory).resolve());_id(stage,56);_id(run_id)
    if not is_nested_job(root,directory):return resolve_run_dir(root,run_id)
    if run_id_exists(root,run_id):raise FileExistsError('Run identifier already reserved')
    path=_safe(root,directory/'stages'/stage/run_id)
    index=run_index_path(root,run_id);index.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.reserve-',dir=index.parent)
    try:
        with os.fdopen(fd,'w') as stream:
            json.dump(dict(schema_version=1,path=str(path.relative_to(root))),stream)
            stream.flush();os.fsync(stream.fileno())
        os.link(name,index)
    finally:
        os.unlink(name)
    return path

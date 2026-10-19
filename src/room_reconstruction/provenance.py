"""Bounded provenance collection and non-overwriting experiment outputs."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict:
    try:
        value = yaml.safe_load(Path(path).read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML configuration: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data: Any) -> None:
    """Refuse to overwrite reports, and reject non-finite JSON before writing."""
    text = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(text)


def create_run_dir(root: Path, run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("run_id must contain 1–80 letters, digits, underscores or hyphens")
    root = Path(root).resolve()
    outputs = root / "outputs"
    if not outputs.resolve().is_relative_to(root):
        raise ValueError("Output directory must remain within the project workspace")
    outputs.mkdir(exist_ok=True)
    from .pipeline_layout import resolve_run_dir
    run = resolve_run_dir(root,run_id)
    run.parent.mkdir(parents=True,exist_ok=True)
    run.mkdir(exist_ok=False)
    return run


def git_snapshot(root: Path) -> dict:
    """Hash relevant source only; never read credentials, media, models or outputs."""
    root = Path(root).resolve()

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, timeout=30
        )

    inside = git("rev-parse", "--is-inside-work-tree")
    if inside.returncode != 0:
        return {"root": str(root), "available": False, "head": None, "branch": None,
                "dirty": None, "source_files": {}}
    scope = ["src", "scripts", "configs", "tests", "da3_streaming", "README.md", "PRD.md", "ROADMAP.md",
             "docs/PRD.md", "docs/ROADMAP.md", "docs/PROGRESS.md", "WORKFLOW.md",
             "requirements.txt", "pyproject.toml", ".gitignore", ".sync-files"]
    listing = git("ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", *scope)
    if listing.returncode:
        raise RuntimeError("Cannot enumerate source files for provenance")
    source_files = {}
    for name in sorted(set(listing.stdout.split("\0")) - {""}):
        path = root / name
        if path.is_symlink() or not path.is_file():
            continue
        if any(part.startswith(".env") for part in path.relative_to(root).parts):
            continue
        if path.suffix.lower() not in {".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".patch"} and name not in {".gitignore", ".sync-files"}:
            continue
        source_files[name] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    head = git("rev-parse", "HEAD")
    branch = git("symbolic-ref", "--short", "HEAD")
    status = git("status", "--porcelain=v1", "--untracked-files=all", "--", *scope)
    return {
        "root": str(root), "available": True,
        "head": head.stdout.strip() if head.returncode == 0 else None,
        "branch": branch.stdout.strip() if branch.returncode == 0 else None,
        "dirty": bool(status.stdout), "source_files": source_files,
        "scope": "source/config/docs/tests; credentials, media and generated artifacts excluded",
    }

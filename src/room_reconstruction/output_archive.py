"""Reversible, same-filesystem cold archival. No deletion operation exists."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import ctypes
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys


FIXTURE_JOBS = {"archicon_cache_001", "layout_cache_001", "layout_smoke_001"}
PROTECTED = {"jobs", "archive", "dashboard_qa", "documentation_audit", "README.md"}


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _path(root, relative):
    p = PurePosixPath(relative)
    if p.is_absolute() or ".." in p.parts or not p.parts or str(p) != relative:
        raise ValueError("Non-canonical relative path: " + str(relative))
    target = root / relative
    for parent in (target, *target.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError("Symlink refused: " + str(parent))
    if not target.resolve().is_relative_to(root):
        raise ValueError("Path escapes workspace: " + relative)
    return target


def _validate(root, archive, entries, protected=()):
    if not re.fullmatch(r"outputs/archive/experiments_\d{4}-\d{2}-\d{2}", archive):
        raise ValueError("Archive must be outputs/archive/experiments_YYYY-MM-DD")
    _path(root, archive)
    sources, destinations = set(), set()
    for entry in entries:
        source, destination = entry["source"], entry["destination"]
        parts = PurePosixPath(source).parts
        allowed_flat = (len(parts) == 2 and parts[0] == "outputs"
                        and not parts[1].startswith(".") and parts[1] not in PROTECTED)
        allowed_fixture = (len(parts) == 3 and parts[:2] == ("outputs", "jobs")
                           and parts[2] in FIXTURE_JOBS)
        if not (allowed_flat or allowed_fixture):
            raise ValueError("Protected or unsupported source: " + source)
        if any(source == p or source.startswith(p + "/") or p.startswith(source + "/")
               for p in protected):
            raise ValueError("Protected source: " + source)
        expected = f"{archive}/{'jobs' if allowed_fixture else 'flat'}/{parts[-1]}"
        if destination != expected or source in sources or destination in destinations:
            raise ValueError("Invalid or duplicate mapping: " + source)
        _path(root, source)
        _path(root, destination)
        sources.add(source)
        destinations.add(destination)
    if not entries:
        raise ValueError("Empty move list")


def _snapshot(path):
    """Include empty directories, file bytes, modes and nanosecond mtimes."""
    result = {}
    def visit(current, name):
        info = current.lstat()
        record = dict(mode=stat.S_IMODE(info.st_mode), mtime_ns=info.st_mtime_ns)
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("Symlink refused: " + str(current))
        if stat.S_ISDIR(info.st_mode):
            record["kind"] = "directory"
            result[name] = record
            for child in sorted(current.iterdir()):
                visit(child, child.name if name == "." else name + "/" + child.name)
        elif stat.S_ISREG(info.st_mode):
            record.update(kind="file", size_bytes=info.st_size, sha256=_sha(current))
            result[name] = record
        else:
            raise ValueError("Special file refused: " + str(current))
    visit(path, ".")
    return result


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_new(path, value):
    with path.open("x") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    _sync_directory(path.parent)


@contextmanager
def _guard(root):
    """Block dashboard launch and check/hold every extant job worker lock."""
    paths = [root / "outputs/.locks/output-archive.lock",
             root / "outputs/.locks/archicon-controller.lock"]
    for pattern in ("outputs/*/worker.lock", "outputs/*/controller.lock",
                    "outputs/jobs/*/worker.lock", "outputs/jobs/*/controller.lock"):
        paths.extend(sorted(root.glob(pattern)))
    with ExitStack() as stack:
        for path in paths:
            _path(root, str(path.relative_to(root)))
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = stack.enter_context(path.open("a+"))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Busy workspace lock: " + str(path)) from exc
        yield


def _rename_exclusive(source, destination):
    """Kernel no-replace rename on macOS/Linux; fail rather than overwrite."""
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        fn = libc.renamex_np
        fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        args = (os.fsencode(source), os.fsencode(destination), 4)  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        fn = libc.renameat2
        fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        args = (-100, os.fsencode(source), -100, os.fsencode(destination), 1)  # RENAME_NOREPLACE
    else:
        raise RuntimeError("Atomic no-replace rename requires macOS or Linux renameat2")
    fn.restype = ctypes.c_int
    if fn(*args) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))
    _sync_directory(source.parent)
    _sync_directory(destination.parent)


def prepare(root, proposal_path, host, archive):
    root = Path(root).resolve()
    proposal_path = Path(proposal_path)
    proposal = json.loads(proposal_path.read_text())
    if proposal.get("deletion_allowed") is not False:
        raise ValueError("Proposal must forbid deletion")
    plan = proposal["hosts"][host]
    if Path(plan["root"]).resolve() != root:
        raise ValueError("Wrong workspace/host")
    _validate(root, archive, plan["entries"], plan.get("protected", []))
    manifest_path = _path(root, archive + "/move-manifest.json")
    if manifest_path.exists():
        raise FileExistsError(str(manifest_path))
    with _guard(root):
        entries = []
        for entry in plan["entries"]:
            source, destination = _path(root, entry["source"]), _path(root, entry["destination"])
            if destination.exists():
                raise RuntimeError("Collision: " + str(destination))
            inventory = _snapshot(source)
            files = [r for r in inventory.values() if r["kind"] == "file"]
            if (len(files) != entry.get("file_count", len(files)) or
                sum(r["size_bytes"] for r in files) != entry.get("logical_bytes", sum(r["size_bytes"] for r in files))):
                raise RuntimeError("Source changed since proposal: " + entry["source"])
            entries.append(dict(entry, inventory=inventory))
        data = dict(schema_version=1, root=str(root), host=host, archive=archive,
                    created_at_utc=datetime.now(timezone.utc).isoformat(),
                    policy="cold_archive_approved", approval="User: ok setuju; no deletion",
                    deletion_allowed=False, proposal_sha256=_sha(proposal_path),
                    protected=plan.get("protected", []), entries=entries)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        _write_new(manifest_path, json.dumps(data, indent=2) + "\n")
        _write_new(manifest_path.with_suffix(".sha256"), _sha(manifest_path) + "\n")
        _write_new(manifest_path.parent / "README.md",
            "# Arsip dingin eksperimen\n\nTidak ada data yang dihapus. Produk final tetap di end_product/.\n"
            "Path lama harus dipulihkan sebelum resume/reuse/refinement. Registry dan artefak asli tidak diedit.\n\n"
            "Dari root proyek, pada host pemilik arsip:\n\n```sh\n"
            f"rtk proxy .venv/bin/python scripts/archive_outputs.py verify --manifest {archive}/move-manifest.json\n"
            f"rtk proxy .venv/bin/python scripts/archive_outputs.py restore --manifest {archive}/move-manifest.json\n"
            "```\n\nRestore memeriksa SHA seluruh file dahulu dan menolak overwrite. Hentikan job sebelum restore.\n"
            "Apply/restore terputus bisa diulang dengan manifest yang sama. File journal mencatat tiap rename.\n"
            "Ini pemindahan dalam disk yang sama; kapasitas disk tidak dibebaskan.\n")
    return dict(manifest=str(manifest_path.relative_to(root)), prepared_entries=len(entries))


def _load(root, manifest_path):
    manifest_path = Path(manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    relative = str(manifest_path.relative_to(root))
    _path(root, relative)
    data = json.loads(manifest_path.read_text())
    if (data.get("schema_version") != 1 or data.get("deletion_allowed") is not False
        or data.get("policy") != "cold_archive_approved" or data.get("root") != str(root)
        or relative != data["archive"] + "/move-manifest.json"):
        raise ValueError("Wrong workspace or invalid manifest")
    _validate(root, data["archive"], data["entries"], data.get("protected", []))
    checksum = _path(root, str(manifest_path.with_suffix(".sha256").relative_to(root)))
    if checksum.read_text().strip() != _sha(manifest_path):
        raise ValueError("Manifest checksum changed")
    return data


def _inspect(root, data):
    rows = []
    for entry in data["entries"]:
        source, destination = _path(root, entry["source"]), _path(root, entry["destination"])
        if source.exists() and destination.exists():
            raise RuntimeError("Collision at source and destination: " + entry["source"])
        if not source.exists() and not destination.exists():
            raise RuntimeError("Missing source and destination: " + entry["source"])
        actual = source if source.exists() else destination
        if _snapshot(actual) != entry["inventory"]:
            raise RuntimeError("File tree changed: " + str(actual))
        rows.append((entry, actual == destination))
    return rows


def _report(data, rows):
    return dict(host=data["host"], archive=data["archive"], entries=len(rows),
                archived_entries=sum(moved for _, moved in rows),
                restored_entries=sum(not moved for _, moved in rows),
                verified_files=sum(r["kind"] == "file" for e, _ in rows for r in e["inventory"].values()),
                logical_bytes=sum(r.get("size_bytes", 0) for e, _ in rows for r in e["inventory"].values()),
                deletions=0)


def _journal(path, event):
    with path.open("a") as stream:
        stream.write(json.dumps(dict(at_utc=datetime.now(timezone.utc).isoformat(), **event)) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    _sync_directory(path.parent)


def verify(root, manifest_path):
    root = Path(root).resolve()
    data = _load(root, manifest_path)
    with _guard(root):
        return _report(data, _inspect(root, data))


def _move(root, manifest_path, restoring):
    root = Path(root).resolve()
    data = _load(root, manifest_path)
    journal = _path(root, data["archive"] + "/move-journal.jsonl")
    with _guard(root):
        rows = _inspect(root, data)  # Hash ALL entries before the first mutation.
        for entry, archived in rows:
            if archived == (not restoring):
                continue
            origin = _path(root, entry["destination"] if restoring else entry["source"])
            target = _path(root, entry["source"] if restoring else entry["destination"])
            target.parent.mkdir(parents=True, exist_ok=True)
            if origin.stat().st_dev != target.parent.stat().st_dev:
                raise OSError(errno.EXDEV, "Refusing cross-filesystem move")
            event = dict(operation="restore" if restoring else "apply",
                         source=str(origin.relative_to(root)), destination=str(target.relative_to(root)))
            _journal(journal, dict(event, phase="intent"))
            _rename_exclusive(origin, target)
            _journal(journal, dict(event, phase="done"))
        report = _report(data, _inspect(root, data))
        _journal(journal, dict(operation="verify", report=report))
        return report


def apply(root, manifest_path):
    return _move(root, manifest_path, restoring=False)


def restore(root, manifest_path):
    return _move(root, manifest_path, restoring=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "apply", "verify", "restore"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--proposal", type=Path)
    parser.add_argument("--host", choices=("local", "server"))
    parser.add_argument("--archive", default="outputs/archive/experiments_2026-10-16")
    parser.add_argument("--manifest", type=Path, default=Path("outputs/archive/experiments_2026-10-16/move-manifest.json"))
    args = parser.parse_args()
    if args.operation == "prepare":
        if not args.proposal or not args.host:
            parser.error("prepare requires --proposal and --host")
        result = prepare(args.root, args.proposal, args.host, args.archive)
    else:
        result = globals()[args.operation](args.root, args.manifest)
    print(json.dumps(result, indent=2))

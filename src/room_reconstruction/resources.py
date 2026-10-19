"""Fail-closed, allocation-free resource checks and single-GPU isolation."""
from dataclasses import asdict, dataclass
import csv
import io
import math
import os
import re
from typing import Mapping, Sequence

GIB = 1024 ** 3
_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")


@dataclass(frozen=True)
class GPU:
    index: int
    uuid: str
    name: str
    total_bytes: int
    free_bytes: int
    utilization_percent: float


@dataclass(frozen=True)
class ResourcePolicy:
    min_gpu_free_gib: float = 14
    min_host_available_gib: float = 8
    min_disk_free_gib: float = 50
    cpu_threads: int = 4
    workers: int = 2

    @classmethod
    def from_mapping(cls, values: Mapping) -> "ResourcePolicy":
        result = {}
        for key, field in cls.__dataclass_fields__.items():
            value = values.get(key, field.default)
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid resource policy: {key}") from exc
            if isinstance(value, bool) or not math.isfinite(numeric) or numeric <= 0:
                raise ValueError(f"Resource policy {key} must be positive and finite")
            if key in {"cpu_threads", "workers"}:
                if not numeric.is_integer():
                    raise ValueError(f"Resource policy {key} must be an integer")
                result[key] = int(numeric)
            else:
                result[key] = numeric
        return cls(**result)


def parse_gpu_csv(text: str) -> list[GPU]:
    """Parse index,uuid,name,memory.total,memory.free,utilization.gpu (MiB)."""
    devices = []
    try:
        for row in csv.reader(io.StringIO(text), strict=True):
            if not row:
                continue
            if len(row) != 6:
                raise ValueError("GPU query must contain exactly six CSV fields")
            index, uuid, name, total, free, utilization = (part.strip() for part in row)
            if not index.isdecimal() or not _UUID.fullmatch(uuid) or not name:
                raise ValueError("Invalid GPU identity")
            if not total.isdecimal() or not free.isdecimal():
                raise ValueError("GPU memory is missing or invalid")
            total_mib, free_mib, percent = int(total), int(free), float(utilization)
            if total_mib <= 0 or free_mib > total_mib or not 0 <= percent <= 100:
                raise ValueError("GPU telemetry outside valid range")
            devices.append(GPU(int(index), uuid, name, total_mib * 1024 ** 2,
                               free_mib * 1024 ** 2, percent))
    except (csv.Error, ValueError) as exc:
        raise ValueError(f"Malformed nvidia-smi GPU output: {exc}") from exc
    if not devices or len({gpu.uuid for gpu in devices}) != len(devices) or len({gpu.index for gpu in devices}) != len(devices):
        raise ValueError("Missing or duplicate GPU telemetry")
    return devices


def parse_compute_process_uuids(text: str) -> set[str]:
    """An empty successful process query means idle; unknown values do not."""
    uuids = {line.strip() for line in text.splitlines() if line.strip()}
    if any(not _UUID.fullmatch(uuid) for uuid in uuids):
        raise ValueError("Malformed nvidia-smi compute-process output")
    return uuids


def parse_mem_available(text: str) -> int:
    """Linux /proc/meminfo uses kB to mean 1024 bytes."""
    matches = re.findall(r"^MemAvailable:\s+(\d+)\s+kB\s*$", text, re.MULTILINE)
    if len(matches) != 1:
        raise ValueError("MemAvailable unavailable or malformed")
    return int(matches[0]) * 1024


def select_gpu(gpus: Sequence[GPU], selector: str | int, current_visible: str | None = None) -> GPU:
    """Selectors are physical indices or exact UUIDs, never logical CUDA indices."""
    def resolve(token: str) -> GPU:
        matches = [gpu for gpu in gpus if token == gpu.uuid or (token.isdecimal() and int(token) == gpu.index)]
        if len(matches) != 1:
            raise ValueError(f"GPU selector must identify exactly one physical GPU: {token!r}")
        return matches[0]

    selected = resolve(str(selector).strip())
    if current_visible is not None:
        tokens = [token.strip() for token in current_visible.split(",")]
        if not all(tokens):
            raise ValueError("CUDA_VISIBLE_DEVICES exposes no valid GPU")
        visible = {resolve(token).uuid for token in tokens}
        if selected.uuid not in visible:
            raise ValueError(f"Requested GPU {selected.uuid} is hidden by CUDA_VISIBLE_DEVICES")
    return selected


def launch_environment(gpu: GPU, policy: ResourcePolicy, base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return a new environment: this physical UUID becomes logical cuda:0."""
    environment = dict(os.environ if base_env is None else base_env)
    environment["CUDA_VISIBLE_DEVICES"] = gpu.uuid
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[variable] = str(policy.cpu_threads)
    return environment


def evaluate_resources(*, gpus: Sequence[GPU], policy: ResourcePolicy, selector: str | int | None,
                       mem_available_bytes: int | None, disk_free_bytes: int | None,
                       busy_uuids: set[str] | None, current_visible: str | None = None,
                       gpu_error: str | None = None, gpu_required: bool = True) -> dict:
    errors, selected = [], None
    for label, available, minimum in (
        ("Host MemAvailable", mem_available_bytes, policy.min_host_available_gib),
        ("Disk free", disk_free_bytes, policy.min_disk_free_gib),
    ):
        if available is None:
            errors.append(f"{label} could not be measured")
        elif available < minimum * GIB:
            errors.append(f"{label} below required {minimum:g} GiB")
    if gpu_required:
        try:
            if gpu_error:
                raise ValueError(gpu_error)
            selected = select_gpu(gpus, selector, current_visible)
            if selected.free_bytes < policy.min_gpu_free_gib * GIB:
                errors.append(f"GPU free memory below required {policy.min_gpu_free_gib:g} GiB")
            if selected.utilization_percent > 0:
                errors.append("Selected GPU has nonzero utilization")
            if busy_uuids is None:
                errors.append("GPU compute-process status could not be measured")
            elif selected.uuid in busy_uuids:
                errors.append("Selected GPU has active compute processes")
        except ValueError as exc:
            errors.append(str(exc))
    environment = launch_environment(selected, policy, {}) if selected else {}
    return {"resource_ready": not errors, "errors": errors,
            "selected_gpu": asdict(selected) if selected else None,
            "logical_device": "cuda:0" if selected else None,
            "launch_environment": environment, "policy": asdict(policy)}

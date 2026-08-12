from __future__ import annotations

import platform
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from medumm.core.config import find_project_root
from medumm.core.distributed import DistributedContext
from medumm.core.io import ensure_directory, write_json


SENSITIVE_KEYS = frozenset({
    "api_key",
    "access_token",
    "auth_token",
    "hf_token",
    "password",
    "secret",
    "token",
})


def _is_sensitive_key(key: Any) -> bool:
    normalized = str(key).strip().casefold().replace("-", "_")
    return normalized in SENSITIVE_KEYS or normalized.endswith(
        ("_api_key", "_access_token", "_auth_token", "_password", "_secret")
    )


def redact_secrets(value: Any) -> Any:
    """Redact common secret-bearing config keys before writing manifests."""

    if isinstance(value, dict):
        return {
            key: (
                "<redacted>"
                if _is_sensitive_key(key)
                else redact_secrets(child)
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [redact_secrets(child) for child in value]
    return value


@dataclass(slots=True)
class RuntimeContext:
    """Shared execution context passed to models, benchmarks, and trainers."""

    run_id: str
    project_root: Path
    output_directory: Path
    command: str
    config_path: str | None = None
    seed: int = 42
    device: str = "auto"
    dtype: str = "auto"
    distributed: bool = False
    rank: int = 0
    world_size: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        command: str,
        config_path: str | Path | None,
        output_directory: str | Path | None = None,
        runtime_config: dict[str, Any] | None = None,
    ) -> "RuntimeContext":
        values = runtime_config or {}
        root = find_project_root(config_path or Path.cwd())
        run_id = str(values.get("run_id") or f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}")
        output = Path(output_directory or values.get("output_directory") or f"outputs/runs/{run_id}")
        output = output if output.is_absolute() else root / output
        distributed = DistributedContext.from_environment()
        return cls(
            run_id=run_id,
            project_root=root,
            output_directory=ensure_directory(output),
            command=command,
            config_path=str(Path(config_path).resolve()) if config_path else None,
            seed=int(values.get("seed", 42)),
            device=str(values.get("device", "auto")),
            dtype=str(values.get("dtype", "auto")),
            distributed=distributed.enabled,
            rank=distributed.rank,
            world_size=distributed.world_size,
            metadata=dict(values.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["project_root"] = str(self.project_root)
        value["output_directory"] = str(self.output_directory)
        return value


def _git_value(project_root: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def environment_snapshot(context: RuntimeContext) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "git_commit": _git_value(context.project_root, "rev-parse", "HEAD"),
        "git_dirty": bool(_git_value(context.project_root, "status", "--porcelain")),
    }
    try:
        import torch

        snapshot["torch"] = torch.__version__
        snapshot["cuda_available"] = torch.cuda.is_available()
        snapshot["cuda_version"] = torch.version.cuda
        snapshot["gpu_count"] = torch.cuda.device_count()
    except (ImportError, OSError):
        snapshot["torch"] = None
    return snapshot


def write_run_manifest(
    context: RuntimeContext,
    *,
    config: dict[str, Any],
    component: dict[str, Any],
    status: str,
    result: dict[str, Any] | None = None,
) -> Path:
    return write_json(
        context.output_directory / "run_manifest.json",
        {
            "schema_version": "1.0",
            "status": status,
            "runtime": context.to_dict(),
            "environment": environment_snapshot(context),
            "component": component,
            "config": redact_secrets(config),
            "result": redact_secrets(result),
        },
    )

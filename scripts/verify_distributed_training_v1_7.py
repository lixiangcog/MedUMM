from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def verify(
    output_directory: Path,
    *,
    expected_strategy: str,
    expected_world_size: int,
) -> dict[str, Any]:
    report_path = output_directory / "distributed_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checkpoint = Path(report["checkpoint"])
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    if report.get("status") != "completed":
        errors.append(f"status={report.get('status')!r}")
    if not report.get("resumed_from"):
        errors.append("resume_from was not exercised")
    distributed = report.get("distributed", {})
    if distributed.get("strategy") != expected_strategy:
        errors.append(f"strategy={distributed.get('strategy')!r}")
    if int(distributed.get("world_size", 0)) != expected_world_size:
        errors.append(f"world_size={distributed.get('world_size')!r}")
    if manifest.get("format") != "torch_distributed_checkpoint":
        errors.append(f"checkpoint format={manifest.get('format')!r}")
    if manifest.get("has_ema") is not True:
        errors.append("EMA state is absent")
    if report.get("ema_updates", 0) < 2:
        errors.append("EMA did not advance after recovery")
    if expected_strategy == "fsdp" and not report.get("activation_checkpointing"):
        errors.append("FSDP activation checkpointing was not enabled")
    rank_files = sorted(checkpoint.glob("rank-*.pt"))
    if len(rank_files) != expected_world_size:
        errors.append(
            f"rank sidecars={len(rank_files)} expected={expected_world_size}"
        )
    shard_files = sorted((checkpoint / "shards").glob("*.distcp"))
    if not shard_files:
        errors.append("distributed checkpoint has no shard files")
    result = {
        "schema_version": "1.0",
        "status": "passed" if not errors else "failed",
        "strategy": expected_strategy,
        "world_size": expected_world_size,
        "report": str(report_path),
        "checkpoint": str(checkpoint),
        "checkpoint_format": manifest.get("format"),
        "optimizer_step": report.get("state", {}).get("optimizer_step"),
        "samples_seen": report.get("state", {}).get("samples_seen"),
        "ema_updates": report.get("ema_updates"),
        "activation_checkpointing": report.get("activation_checkpointing"),
        "rank_sidecars": len(rank_files),
        "distributed_shards": len(shard_files),
        "errors": errors,
    }
    if errors:
        raise RuntimeError("; ".join(errors))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify MedUMM v1.7 distributed training")
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--expected-strategy", choices=["ddp", "fsdp"], required=True)
    parser.add_argument("--expected-world-size", type=int, required=True)
    parser.add_argument("--evidence", type=Path)
    arguments = parser.parse_args()
    result = verify(
        arguments.output_directory,
        expected_strategy=arguments.expected_strategy,
        expected_world_size=arguments.expected_world_size,
    )
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if arguments.evidence:
        arguments.evidence.parent.mkdir(parents=True, exist_ok=True)
        arguments.evidence.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

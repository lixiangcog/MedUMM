from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from medumm.core.builtins import register_builtins
from medumm.core.config import execution_config, load_config
from medumm.core.io import write_json
from medumm.core.registry import registry
from medumm.core.runtime import RuntimeContext, write_run_manifest


def _block(config: dict[str, Any], key: str) -> dict[str, Any]:
    return execution_config(config, key)


def _output_path(root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else root / path


def run_inference(
    config_path: str,
    overrides: list[str] | None = None,
) -> tuple[list[Any], RuntimeContext, dict[str, Any]]:
    from medumm.api import infer

    config = load_config(config_path, overrides)
    block = _block(config, "inference")
    backbone = str(block.get("backbone", "")).strip().lower()
    if not backbone:
        raise ValueError("Inference config requires a backbone.")
    runtime = RuntimeContext.create(
        command="inference",
        config_path=config_path,
        output_directory=block.get("run_directory"),
        runtime_config=config.get("runtime"),
    )
    results = infer(config, config_path=config_path, runtime=runtime)
    register_builtins()
    capabilities = registry.models.create(backbone).capabilities.to_dict()
    return results, runtime, {"name": backbone, "capabilities": capabilities, "config": config}


def _infer_command(arguments: argparse.Namespace) -> int:
    results, runtime, details = run_inference(arguments.config, arguments.overrides)
    config = details.pop("config")
    block = _block(config, "inference")
    output_path = arguments.output_json or block.get("output_json")
    serialized = [result.to_dict() for result in results]
    if output_path:
        target = _output_path(runtime.project_root, output_path)
        write_json(target, serialized)
        print(f"[MedUMM] inference results: {target}")
    write_run_manifest(
        runtime,
        config=config,
        component={"kind": "model", **details},
        status="completed",
        result={"requests": len(results), "results": serialized},
    )
    print(f"[MedUMM] completed {len(results)} inference request(s)")
    return 0


def _evaluation_command(arguments: argparse.Namespace) -> int:
    from medumm.api import evaluate

    register_builtins()
    config = load_config(arguments.config, arguments.overrides)
    block = _block(config, "evaluation")
    benchmark_name = str(block.get("benchmark", "medical_vqa")).strip().lower()
    runtime = RuntimeContext.create(
        command="evaluation",
        config_path=arguments.config,
        output_directory=block.get("output_directory"),
        runtime_config=config.get("runtime"),
    )
    result = evaluate(config, config_path=arguments.config, runtime=runtime)
    write_run_manifest(
        runtime,
        config=config,
        component={"kind": "benchmark", "name": benchmark_name},
        status=result.status,
        result=result.to_dict(),
    )
    print(f"[MedUMM] evaluation {result.status}: {result.output_directory}")
    return 0


def _training_command(arguments: argparse.Namespace) -> int:
    from medumm.api import post_train

    config = load_config(arguments.config, arguments.overrides)
    block = _block(config, "post_training")
    runtime = RuntimeContext.create(
        command="post_training",
        config_path=arguments.config,
        output_directory=block.get("output_directory"),
        runtime_config=config.get("runtime"),
    )
    result = post_train(config, config_path=arguments.config, runtime=runtime)
    write_run_manifest(
        runtime,
        config=config,
        component={"kind": "trainer", "name": result.method},
        status=result.status,
        result=result.to_dict(),
    )
    print(f"[MedUMM] post-training completed: {result.checkpoint_path}")
    return 0


def _catalog_command(arguments: argparse.Namespace) -> int:
    register_builtins()
    catalog = registry.catalog()
    if arguments.json:
        print(json.dumps(catalog, indent=2, ensure_ascii=False))
    else:
        for category, items in catalog.items():
            print(f"{category}:")
            for item in items:
                print(f"  {item['name']}: {item['description']}")
    return 0


def _report_command(arguments: argparse.Namespace) -> int:
    from medumm.reporting import build_leaderboard

    paths = build_leaderboard(arguments.scores, arguments.output_directory)
    print(f"[MedUMM] leaderboard: {paths['json_path']}")
    return 0


def _merge_command(arguments: argparse.Namespace) -> int:
    from medumm.evaluation import merge_prediction_shards

    result = merge_prediction_shards(
        arguments.shards,
        arguments.output,
        expected_count=arguments.expected_count,
    )
    print(f"[MedUMM] merged {result['prediction_count']} predictions: {result['output_path']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medumm", description="Medical multimodal model toolkit")
    parser.add_argument("--version", action="version", version="MedUMM 0.6.0")
    commands = parser.add_subparsers(dest="command", required=True)

    infer = commands.add_parser("infer", help="Run understanding, generation, or editing")
    infer.add_argument("--config", required=True)
    infer.add_argument("--set", dest="overrides", action="append", default=[])
    infer.add_argument("--output-json")
    infer.set_defaults(handler=_infer_command)

    evaluate = commands.add_parser("evaluate", aliases=["eval"], help="Run a benchmark")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--set", dest="overrides", action="append", default=[])
    evaluate.set_defaults(handler=_evaluation_command)

    train = commands.add_parser("post-train", aliases=["train"], help="Run post-training")
    train.add_argument("--config", required=True)
    train.add_argument("--set", dest="overrides", action="append", default=[])
    train.set_defaults(handler=_training_command)

    catalog = commands.add_parser("catalog", help="List registered platform components")
    catalog.add_argument("--json", action="store_true")
    catalog.set_defaults(handler=_catalog_command)

    report = commands.add_parser("report", help="Build a leaderboard from score files")
    report.add_argument("--scores", nargs="+", required=True)
    report.add_argument("--output-directory", required=True)
    report.set_defaults(handler=_report_command)

    merge = commands.add_parser(
        "merge-predictions", help="Strictly merge distributed prediction shards"
    )
    merge.add_argument("--shards", nargs="+", required=True)
    merge.add_argument("--output", required=True)
    merge.add_argument("--expected-count", type=int)
    merge.set_defaults(handler=_merge_command)
    return parser


def main(arguments: list[str] | None = None) -> int:
    parsed = build_parser().parse_args(arguments)
    return int(parsed.handler(parsed))

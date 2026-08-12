from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from medumm.core.config import find_project_root, load_config
from medumm.core.io import write_json


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)


def run_inference(config_path: str, overrides: list[str] | None = None) -> list[Any]:
    from medumm.inference import InferencePipeline

    config = load_config(config_path, overrides)
    block = config.get("inference", config)
    if not isinstance(block, dict):
        raise ValueError("inference must be a mapping.")
    backbone = str(block.get("backbone", ""))
    if not backbone:
        raise ValueError("Inference config requires a backbone.")
    raw_requests = block.get("requests", block.get("request"))
    if raw_requests is None:
        raw_requests = [{
            key: block[key]
            for key in ("task", "prompt", "images", "videos", "parameters", "output_path")
            if key in block
        }]
    elif isinstance(raw_requests, dict):
        raw_requests = [raw_requests]
    if not isinstance(raw_requests, list):
        raise ValueError("requests must be a list.")
    with InferencePipeline(backbone, dict(block.get("config", {}))) as pipeline:
        return pipeline.run_many([dict(request) for request in raw_requests])


def _infer_command(arguments: argparse.Namespace) -> int:
    results = run_inference(arguments.config, arguments.overrides)
    output_path = arguments.output_json
    if not output_path:
        config = load_config(arguments.config, arguments.overrides)
        block = config.get("inference", config)
        output_path = block.get("output_json") if isinstance(block, dict) else None
    if output_path:
        root = find_project_root(arguments.config)
        target = Path(output_path)
        target = target if target.is_absolute() else root / target
        write_json(target, _jsonable(results))
        print(f"[MedUMM] inference results: {target}")
    print(f"[MedUMM] completed {len(results)} inference request(s)")
    return 0


def _evaluation_command(arguments: argparse.Namespace) -> int:
    from medumm.evaluation import run_medical_vqa

    config = load_config(arguments.config, arguments.overrides)
    benchmark = str(config.get("benchmark", "medical_vqa"))
    if benchmark != "medical_vqa":
        raise NotImplementedError(f"Benchmark {benchmark!r} is not available in v0.1.")
    result = run_medical_vqa(config, config_path=arguments.config)
    print(f"[MedUMM] evaluation completed: {result['report_path']}")
    return 0


def _training_command(arguments: argparse.Namespace) -> int:
    from medumm.post_training import PostTrainingRunner

    config = load_config(arguments.config, arguments.overrides)
    block = config.get("post_training", config)
    if not isinstance(block, dict):
        raise ValueError("post_training must be a mapping.")
    result = PostTrainingRunner().run(block, config_path=arguments.config)
    print(f"[MedUMM] post-training completed: {result['manifest_path']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medumm", description="Medical multimodal model toolkit")
    parser.add_argument("--version", action="version", version="MedUMM 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)

    infer = commands.add_parser("infer", help="Run model inference")
    infer.add_argument("--config", required=True)
    infer.add_argument("--set", dest="overrides", action="append", default=[])
    infer.add_argument("--output-json")
    infer.set_defaults(handler=_infer_command)

    evaluate = commands.add_parser("evaluate", aliases=["eval"], help="Run an evaluation")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--set", dest="overrides", action="append", default=[])
    evaluate.set_defaults(handler=_evaluation_command)

    train = commands.add_parser("post-train", aliases=["train"], help="Run post-training")
    train.add_argument("--config", required=True)
    train.add_argument("--set", dest="overrides", action="append", default=[])
    train.set_defaults(handler=_training_command)
    return parser


def main(arguments: list[str] | None = None) -> int:
    parsed = build_parser().parse_args(arguments)
    return int(parsed.handler(parsed))

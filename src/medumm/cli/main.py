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


def _resource_template(kind: str, name: str) -> dict[str, Any]:
    from medumm.resources import DATASET_RESOURCES, MODEL_RESOURCES, AccessLevel

    if kind == "model":
        spec = MODEL_RESOURCES.get(name)
        model_config: dict[str, Any] = {
            "model_path": spec.artifact_id,
            "revision": "REPLACE_WITH_IMMUTABLE_COMMIT",
            "parameters": {"max_new_tokens": 128, "do_sample": False},
        }
        if spec.access is not AccessLevel.OPEN:
            model_config["accept_terms"] = False
        if (
            spec.runtime_family.value == "official_bridge"
            and spec.name != "llava_med_v1_5_7b"
        ):
            model_config["bridge"] = "REPLACE_WITH_MODULE:ModelAdapterClass"
        if spec.runtime_family.value == "open_clip":
            model_config.update(
                {
                    "open_clip_model_name": "REPLACE_WITH_ARCHITECTURE",
                    "checkpoint_path": "REPLACE_WITH_LOCAL_PINNED_CHECKPOINT",
                }
            )
        return {
            "schema_version": "1.0",
            "runtime": {"seed": 42, "device": "auto"},
            "inference": {
                "backbone": spec.name,
                "config": model_config,
                "requests": [
                    {
                        "request_id": "case-001",
                        "task": "understanding",
                        "prompt": "Describe the medically relevant visual findings.",
                        "images": ["REPLACE_WITH_LOCAL_IMAGE"],
                    }
                ],
            },
        }
    spec = DATASET_RESOURCES.get(name)
    data: dict[str, Any] = {
        "adapter": spec.name,
        "path": "REPLACE_WITH_NORMALIZED_MANIFEST.jsonl",
        "image_root": "REPLACE_WITH_LOCAL_IMAGE_ROOT",
        "source_revision": "REPLACE_WITH_IMMUTABLE_COMMIT_OR_RELEASE",
    }
    if spec.access is not AccessLevel.OPEN:
        data["access_confirmed"] = False
    return {
        "schema_version": "1.0",
        "runtime": {"seed": 42, "device": "auto"},
        "evaluation": {
            "benchmark": spec.benchmark,
            "data": data,
            "model": {"backbone": "medical_reference", "parameters": {}},
            "mode": "audit",
            "output_directory": f"outputs/evaluation/{spec.name}",
        },
    }


def _resources_command(arguments: argparse.Namespace) -> int:
    from medumm.resources import DATASET_RESOURCES, MODEL_RESOURCES, resource_catalog

    if arguments.resource_action == "show":
        catalogs = (
            [MODEL_RESOURCES]
            if arguments.kind == "model"
            else [DATASET_RESOURCES]
            if arguments.kind == "dataset"
            else [MODEL_RESOURCES, DATASET_RESOURCES]
        )
        matches = []
        for catalog_value in catalogs:
            if arguments.name in catalog_value.names():
                matches.append(catalog_value.get(arguments.name).to_dict())
        if not matches:
            raise KeyError(f"Unknown medical resource: {arguments.name!r}.")
        print(json.dumps(matches[0], indent=2, ensure_ascii=False))
        return 0
    if arguments.resource_action == "template":
        import yaml

        template = _resource_template(arguments.kind, arguments.name)
        print(yaml.safe_dump(template, sort_keys=False, allow_unicode=True), end="")
        return 0
    if arguments.resource_action == "validate":
        register_builtins()
        missing_models = sorted(set(MODEL_RESOURCES.names()) - set(registry.models.names()))
        missing_datasets = sorted(set(DATASET_RESOURCES.names()) - set(registry.datasets.names()))
        result = {
            "catalog_version": MODEL_RESOURCES.version,
            "models": len(MODEL_RESOURCES.values()),
            "datasets": len(DATASET_RESOURCES.values()),
            "registered_models": len(MODEL_RESOURCES.values()) - len(missing_models),
            "registered_datasets": len(DATASET_RESOURCES.values()) - len(missing_datasets),
            "missing_models": missing_models,
            "missing_datasets": missing_datasets,
            "valid": not missing_models and not missing_datasets,
            "validation_scope": "schema_and_interface_registration",
        }
        if arguments.online:
            from medumm.resources import verify_sources

            online = verify_sources(
                kind=arguments.kind,
                fields=arguments.fields or ("source",),
                timeout=arguments.timeout,
                workers=arguments.workers,
            )
            result["online"] = online
            result["valid"] = bool(result["valid"] and online["valid"])
        if arguments.output:
            write_json(Path(arguments.output), result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["valid"] else 1
    values = resource_catalog(arguments.kind)
    if arguments.json:
        print(json.dumps(values, indent=2, ensure_ascii=False))
        return 0
    resources = values.get("resources") if arguments.kind != "all" else None
    if resources is None:
        for kind in ("models", "datasets"):
            print(f"{kind} ({len(values[kind])}):")
            for item in values[kind]:
                print(
                    f"  {item['name']}: {item['display_name']} "
                    f"[{item['status']}; {item['access']}]"
                )
    else:
        print(f"{arguments.kind}s ({len(resources)}):")
        for item in resources:
            print(
                f"  {item['name']}: {item['display_name']} "
                f"[{item['status']}; {item['access']}]"
            )
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
    parser.add_argument("--version", action="version", version="MedUMM 0.9.0")
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

    resources = commands.add_parser(
        "resources", help="Inspect audited medical model and dataset resources"
    )
    resource_commands = resources.add_subparsers(dest="resource_action", required=True)
    resource_list = resource_commands.add_parser("list", help="List resource specs")
    resource_list.add_argument("--kind", choices=["all", "model", "dataset"], default="all")
    resource_list.add_argument("--json", action="store_true")
    resource_list.set_defaults(handler=_resources_command)
    resource_show = resource_commands.add_parser("show", help="Show one resource spec")
    resource_show.add_argument("name")
    resource_show.add_argument("--kind", choices=["all", "model", "dataset"], default="all")
    resource_show.set_defaults(handler=_resources_command)
    resource_template = resource_commands.add_parser(
        "template", help="Print a minimal pinned config for one resource"
    )
    resource_template.add_argument("name")
    resource_template.add_argument("--kind", choices=["model", "dataset"], required=True)
    resource_template.set_defaults(handler=_resources_command)
    resource_validate = resource_commands.add_parser(
        "validate", help="Validate catalog schemas and plugin registrations"
    )
    resource_validate.add_argument("--kind", choices=["all", "model", "dataset"], default="all")
    resource_validate.add_argument("--online", action="store_true")
    resource_validate.add_argument(
        "--field",
        dest="fields",
        action="append",
        choices=["source", "paper", "official_code"],
        default=[],
    )
    resource_validate.add_argument("--timeout", type=float, default=10.0)
    resource_validate.add_argument("--workers", type=int, default=8)
    resource_validate.add_argument("--output")
    resource_validate.set_defaults(handler=_resources_command)

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

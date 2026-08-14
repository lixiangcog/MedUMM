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
    from medumm.post_training.research_routes import (
        research_route_catalog,
        route_template,
    )

    if arguments.list_methods:
        routes = research_route_catalog()
        if arguments.json:
            print(json.dumps(routes, indent=2, ensure_ascii=False))
        else:
            for route in routes:
                stages = " -> ".join(route["stage_order"])
                print(f"{route['name']}: {route['summary']} [{stages}]")
        return 0
    if arguments.template:
        import yaml

        print(yaml.safe_dump(route_template(arguments.template), sort_keys=False))
        return 0
    if not arguments.config:
        raise ValueError(
            "post-train requires --config, --list-methods, or --template METHOD."
        )

    overrides = list(arguments.overrides)
    if arguments.plan:
        overrides.append("post_training.execution=plan")
    config = load_config(arguments.config, overrides)
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
    if result.status == "planned":
        print(f"[MedUMM] post-training plan written: {result.output_directory}")
    else:
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


def _backends_command(arguments: argparse.Namespace) -> int:
    from medumm.inference import backend_catalog

    values = backend_catalog()
    if arguments.json:
        print(json.dumps(values, indent=2, ensure_ascii=False))
    else:
        for item in values:
            state = f"installed ({item['version']})" if item["installed"] else "not installed"
            print(
                f"{item['name']}: {state}; continuous_batching="
                f"{str(item['continuous_batching']).lower()}; modes={','.join(item['modes'])}"
            )
    return 0


def _environments_command(arguments: argparse.Namespace) -> int:
    from medumm.environments import ENVIRONMENT_CATALOG
    from medumm.environments.render import inspect_current_environment

    if arguments.environment_action == "show":
        print(
            json.dumps(
                ENVIRONMENT_CATALOG.get(arguments.model).to_dict(),
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    if arguments.environment_action == "check-current":
        result = inspect_current_environment(ENVIRONMENT_CATALOG.get(arguments.model))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["valid"] else 1
    values = ENVIRONMENT_CATALOG.to_dict()
    if arguments.json:
        print(json.dumps(values, indent=2, ensure_ascii=False))
    else:
        for item in values["models"]:
            print(
                f"{item['model']}: profile={item['profile']}; python={item['python']}; "
                f"cuda={item['cuda']}; validation={item['validation']}"
            )
    return 0


def _models_command(arguments: argparse.Namespace) -> int:
    from medumm.backbones.audit import adapter_catalog, preflight_model_adapter
    from medumm.backbones.recipes import MODEL_ADAPTER_RECIPES
    from medumm.core.config import find_project_root

    if arguments.model_action == "show":
        catalog = adapter_catalog()
        row = next(
            (item for item in catalog["recipes"] if item["name"] == arguments.model),
            None,
        )
        if row is None:
            raise KeyError(f"Unknown model adapter: {arguments.model!r}.")
        print(json.dumps(row, indent=2, ensure_ascii=False))
        return 0
    if arguments.model_action == "preflight":
        root = find_project_root(Path.cwd())
        result = preflight_model_adapter(
            arguments.model,
            project_root=root,
            model_path=arguments.model_path,
            source_path=arguments.source_path,
            revision=arguments.revision,
            accept_terms=arguments.accept_terms,
            check_imports=not arguments.skip_imports,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["ready"] else 1
    catalog = adapter_catalog()
    if arguments.model_action == "audit":
        print(json.dumps(catalog, indent=2, ensure_ascii=False))
        return 0 if catalog["valid"] else 1
    if arguments.json:
        print(json.dumps(catalog, indent=2, ensure_ascii=False))
        return 0
    for recipe in MODEL_ADAPTER_RECIPES.values():
        print(
            f"{recipe.name}: executor={recipe.executor.value}; "
            f"implementation={recipe.implementation.value}"
        )
    return 0


def _benchmarks_command(arguments: argparse.Namespace) -> int:
    from medumm.evaluation.benchmark_catalog import (
        SPECIALIZED_BENCHMARKS,
        get_medical_benchmark,
        medical_benchmark_catalog,
    )
    from medumm.evaluation.metrics import create_metric_suite
    from medumm.resources import DATASET_RESOURCES

    register_builtins()
    if arguments.benchmark_action == "show":
        print(
            json.dumps(
                get_medical_benchmark(arguments.benchmark).to_dict(),
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    if arguments.benchmark_action == "template":
        import yaml

        spec = get_medical_benchmark(arguments.benchmark)
        template = {
            "schema_version": "1.0",
            "runtime": {"seed": 42, "device": "auto"},
            "evaluation": {
                "benchmark": spec.name,
                "data": {
                    "adapter": "medical_tasks_jsonl",
                    "path": "REPLACE_WITH_NORMALIZED_MANIFEST.jsonl",
                    "image_root": "REPLACE_WITH_MEDIA_ROOT",
                    "provenance": "REPLACE_WITH_PROVENANCE.json",
                    "deidentified": True,
                },
                "model": {"backbone": "REPLACE_WITH_MODEL", "parameters": {}},
                "protocol": {
                    "name": spec.name,
                    "version": spec.version,
                    "metric_suite": spec.metric_suite,
                    "require_provenance": True,
                    "require_deidentified": True,
                },
                "mode": "full",
                "output_directory": f"outputs/evaluation/{spec.name}",
            },
        }
        template["benchmark_contract"] = {
            "dataset_families": [value.value for value in spec.dataset_families],
            "required_annotation": spec.required_annotation,
            "requires_choices": spec.requires_choices,
            "note": "benchmark_contract documents the normalized row requirements; remove it before execution if a strict external schema is used.",
        }
        print(yaml.safe_dump(template, sort_keys=False, allow_unicode=True), end="")
        return 0
    if arguments.benchmark_action == "audit":
        registered = set(registry.benchmarks.names())
        expected = {spec.name for spec in SPECIALIZED_BENCHMARKS}
        missing = sorted(expected - registered)
        metric_errors = []
        for spec in SPECIALIZED_BENCHMARKS:
            try:
                suite = create_metric_suite(spec.metric_suite)
            except Exception as error:  # report the contract failure without hiding others
                metric_errors.append({"benchmark": spec.name, "error": str(error)})
                continue
            if suite.version != spec.version:
                metric_errors.append(
                    {
                        "benchmark": spec.name,
                        "error": f"metric version {suite.version} != benchmark {spec.version}",
                    }
                )
        unknown_dataset_benchmarks = sorted(
            {
                spec.benchmark
                for spec in DATASET_RESOURCES.values()
                if spec.benchmark not in registered
            }
        )
        result = {
            "schema_version": "1.0",
            "registered_benchmarks": len(registered),
            "independent_benchmarks": len(registered - {"cross_task"}),
            "specialized_medical_benchmarks": len(expected),
            "composite_benchmarks": 1,
            "dataset_resources": len(DATASET_RESOURCES.values()),
            "dataset_benchmark_families": len(
                {spec.benchmark for spec in DATASET_RESOURCES.values()}
            ),
            "missing_specialized_benchmarks": missing,
            "metric_contract_errors": metric_errors,
            "unknown_dataset_benchmarks": unknown_dataset_benchmarks,
            "valid": not missing and not metric_errors and not unknown_dataset_benchmarks,
            "validation_scope": (
                "Executable benchmark registration, fixed metric contracts, and dataset-family "
                "routing; dataset catalog entries are not counted as benchmarks."
            ),
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["valid"] else 1
    values = medical_benchmark_catalog()
    if arguments.json:
        print(json.dumps(values, indent=2, ensure_ascii=False))
    else:
        for item in values:
            print(
                f"{item['name']}: metric={item['metric_suite']}; "
                f"validation={item['validation']}"
            )
    return 0


def _benchmark_inference_command(arguments: argparse.Namespace) -> int:
    from medumm.inference.benchmark import run_inference_benchmark

    config = load_config(arguments.config, arguments.overrides)
    result = run_inference_benchmark(config, config_path=arguments.config)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _serve_command(arguments: argparse.Namespace) -> int:
    from medumm.core.config import find_project_root
    from medumm.inference.server import launch_server, plan_server

    config = load_config(arguments.config, arguments.overrides)
    block = config.get("server", config)
    if not isinstance(block, dict):
        raise ValueError("server config must be a mapping.")
    root = find_project_root(arguments.config)
    if arguments.plan or str(block.get("execution", "plan")).casefold() == "plan":
        plan = plan_server(block, project_root=root)
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    return launch_server(block, project_root=root)


def _resource_template(kind: str, name: str) -> dict[str, Any]:
    from medumm.resources import DATASET_RESOURCES, MODEL_RESOURCES, AccessLevel

    if kind == "model":
        from medumm.backbones.recipes import MODEL_ADAPTER_RECIPES

        spec = MODEL_RESOURCES.get(name)
        recipe = MODEL_ADAPTER_RECIPES.get(name)
        model_config: dict[str, Any] = {
            "model_path": spec.artifact_id,
            "revision": "REPLACE_WITH_IMMUTABLE_COMMIT",
            "parameters": {"max_new_tokens": 128, "do_sample": False},
        }
        if spec.access is not AccessLevel.OPEN:
            model_config["accept_terms"] = False
        if recipe.source_checkout_required:
            model_config["source_path"] = "REPLACE_WITH_PINNED_UPSTREAM_CHECKOUT"
        if spec.runtime_family.value == "open_clip":
            model_config["checkpoint_path"] = "REPLACE_WITH_LOCAL_PINNED_CHECKPOINT"
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
    parser.add_argument("--version", action="version", version="MedUMM 1.7.0")
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
    train.add_argument("--config")
    train.add_argument("--set", dest="overrides", action="append", default=[])
    train.add_argument(
        "--plan", action="store_true", help="Validate and write a launch plan without executing"
    )
    train.add_argument(
        "--list-methods", action="store_true", help="List research post-training routes"
    )
    train.add_argument("--template", help="Print a plan-only template for one research route")
    train.add_argument("--json", action="store_true", help="Use JSON for method discovery")
    train.set_defaults(handler=_training_command)

    catalog = commands.add_parser("catalog", help="List registered platform components")
    catalog.add_argument("--json", action="store_true")
    catalog.set_defaults(handler=_catalog_command)

    backends = commands.add_parser(
        "backends", help="Inspect inference backend capabilities and installed runtimes"
    )
    backends.add_argument("--json", action="store_true")
    backends.set_defaults(handler=_backends_command)

    models = commands.add_parser(
        "models", help="Inspect concrete model adapter recipes and runtime readiness"
    )
    model_commands = models.add_subparsers(dest="model_action", required=True)
    model_list = model_commands.add_parser("list", help="List explicit model adapters")
    model_list.add_argument("--json", action="store_true")
    model_list.set_defaults(handler=_models_command)
    model_show = model_commands.add_parser("show", help="Show one adapter recipe")
    model_show.add_argument("model")
    model_show.set_defaults(handler=_models_command)
    model_audit = model_commands.add_parser(
        "audit", help="Audit catalog, recipe, and environment coverage"
    )
    model_audit.set_defaults(handler=_models_command)
    model_preflight = model_commands.add_parser(
        "preflight", help="Validate pinned local assets before loading a real model"
    )
    model_preflight.add_argument("model")
    model_preflight.add_argument("--model-path", required=True)
    model_preflight.add_argument("--source-path")
    model_preflight.add_argument("--revision", required=True)
    model_preflight.add_argument("--accept-terms", action="store_true")
    model_preflight.add_argument("--skip-imports", action="store_true")
    model_preflight.set_defaults(handler=_models_command)

    environments = commands.add_parser(
        "environments", help="Inspect reproducible per-model runtime contracts"
    )
    environment_commands = environments.add_subparsers(
        dest="environment_action", required=True
    )
    environment_list = environment_commands.add_parser("list", help="List model environments")
    environment_list.add_argument("--json", action="store_true")
    environment_list.set_defaults(handler=_environments_command)
    environment_show = environment_commands.add_parser("show", help="Show one environment")
    environment_show.add_argument("model")
    environment_show.set_defaults(handler=_environments_command)
    environment_check = environment_commands.add_parser(
        "check-current", help="Check imports in the current Python environment"
    )
    environment_check.add_argument("model")
    environment_check.set_defaults(handler=_environments_command)

    benchmarks = commands.add_parser(
        "benchmarks", help="Inspect executable medical benchmark contracts"
    )
    benchmark_commands = benchmarks.add_subparsers(
        dest="benchmark_action", required=True
    )
    benchmark_list = benchmark_commands.add_parser(
        "list", help="List independent specialized medical benchmarks"
    )
    benchmark_list.add_argument("--json", action="store_true")
    benchmark_list.set_defaults(handler=_benchmarks_command)
    benchmark_show = benchmark_commands.add_parser(
        "show", help="Show one benchmark data/prompt/metric contract"
    )
    benchmark_show.add_argument("benchmark")
    benchmark_show.set_defaults(handler=_benchmarks_command)
    benchmark_audit = benchmark_commands.add_parser(
        "audit", help="Audit benchmark, metric, and dataset routing coverage"
    )
    benchmark_audit.set_defaults(handler=_benchmarks_command)
    benchmark_template = benchmark_commands.add_parser(
        "template", help="Print a safe configuration template for one benchmark"
    )
    benchmark_template.add_argument("benchmark")
    benchmark_template.set_defaults(handler=_benchmarks_command)

    benchmark_inference = commands.add_parser(
        "benchmark-inference", help="Measure inference latency and throughput"
    )
    benchmark_inference.add_argument("--config", required=True)
    benchmark_inference.add_argument("--set", dest="overrides", action="append", default=[])
    benchmark_inference.set_defaults(handler=_benchmark_inference_command)

    serve = commands.add_parser(
        "serve", help="Plan or launch a vLLM/SGLang continuous-batching server"
    )
    serve.add_argument("--config", required=True)
    serve.add_argument("--set", dest="overrides", action="append", default=[])
    serve.add_argument("--plan", action="store_true")
    serve.set_defaults(handler=_serve_command)

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

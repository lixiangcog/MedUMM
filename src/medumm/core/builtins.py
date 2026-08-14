from __future__ import annotations

from importlib import import_module
from typing import Any

from medumm.core.registry import registry


def _factory(module_name: str, class_name: str):
    def create() -> Any:
        return getattr(import_module(module_name), class_name)()

    return create


def _resource_factory(module_name: str, factory_name: str, resource_name: str):
    def create() -> Any:
        return getattr(import_module(module_name), factory_name)(resource_name)()

    return create


def register_builtins() -> None:
    models = {
        "medical_reference": (
            "medumm.backbones.medical_reference",
            "MedicalReferenceAdapter",
            "Deterministic three-task software reference adapter",
            {
                "architecture": "reference",
                "tasks": ["understanding", "generation", "editing"],
            },
        ),
        "medical_linear": (
            "medumm.backbones.medical_linear",
            "MedicalLinearAdapter",
            "Reloadable medical VQA engineering baseline",
            {"architecture": "reference", "tasks": ["understanding"]},
        ),
        "medgemma": (
            "medumm.backbones.medgemma",
            "MedGemmaAdapter",
            "Medical image-text understanding adapter",
            {"architecture": "autoregressive", "tasks": ["understanding"]},
        ),
        "llava_med": (
            "medumm.backbones.llava_med",
            "LlavaMedAdapter",
            "LLaVA-Med v1.5 biomedical image understanding adapter",
            {
                "architecture": "autoregressive",
                "tasks": ["understanding"],
                "default_model": "microsoft/llava-med-v1.5-mistral-7b",
                "research_only": True,
            },
        ),
        "lingshu_7b": (
            "medumm.backbones.lingshu",
            "LingshuAdapter",
            "Lingshu-7B native medical Qwen2.5-VL adapter",
            {
                "architecture": "autoregressive",
                "tasks": ["understanding"],
                "default_model": "lingshu-medical-mllm/Lingshu-7B",
                "research_only": True,
            },
        ),
        "emu3_5": (
            "medumm.backbones.emu3_5",
            "Emu3_5Adapter",
            "Emu3.5 native patched-vLLM unified multimodal adapter",
            {
                "architecture": "autoregressive",
                "tasks": ["understanding", "generation", "editing"],
                "backends": ["vllm"],
                "continuous_batching": True,
                "classifier_free_guidance": True,
                "parallelism": ["tensor_parallel"],
                "research_only": True,
            },
        ),
        "openai_http": (
            "medumm.inference.openai_backend",
            "OpenAIHTTPAdapter",
            "OpenAI-compatible vLLM/SGLang medical inference client",
            {
                "architecture": "autoregressive",
                "tasks": ["understanding"],
                "backends": ["vllm", "sglang"],
                "continuous_batching": True,
                "parallelism": ["tensor_parallel", "pipeline_parallel", "data_parallel"],
                "research_only": True,
            },
        ),
    }
    for name, (module_name, class_name, description, metadata) in models.items():
        if not registry.models.contains(name):
            registry.models.register(
                name,
                _factory(module_name, class_name),
                description=description,
                metadata=metadata,
            )

    from medumm.resources import DATASET_RESOURCES, MODEL_RESOURCES

    for spec in MODEL_RESOURCES.values():
        if not registry.models.contains(spec.name):
            registry.models.register(
                spec.name,
                _resource_factory(
                    "medumm.backbones.catalog_model", "catalog_model_factory", spec.name
                ),
                description=f"{spec.display_name} medical resource adapter",
                metadata={
                    **spec.to_dict(),
                    "catalog_version": MODEL_RESOURCES.version,
                },
            )

    if not registry.datasets.contains("medical_vqa_jsonl"):
        registry.datasets.register(
            "medical_vqa_jsonl",
            _factory("medumm.medical.dataset", "MedicalVQADatasetAdapter"),
            description="Local JSON/JSONL medical visual question answering dataset",
        )
    if not registry.datasets.contains("medical_tasks_jsonl"):
        registry.datasets.register(
            "medical_tasks_jsonl",
            _factory("medumm.medical.dataset", "MedicalTasksDatasetAdapter"),
            description="Task-aware medical perception, reasoning, and generation dataset",
        )
    for spec in DATASET_RESOURCES.values():
        if not registry.datasets.contains(spec.name):
            registry.datasets.register(
                spec.name,
                _resource_factory(
                    "medumm.medical.catalog_dataset", "catalog_dataset_factory", spec.name
                ),
                description=f"{spec.display_name} normalized medical dataset adapter",
                metadata={
                    **spec.to_dict(),
                    "catalog_version": DATASET_RESOURCES.version,
                },
            )
    if not registry.benchmarks.contains("medical_vqa"):
        registry.benchmarks.register(
            "medical_vqa",
            _factory("medumm.evaluation.medical_vqa", "MedicalVQABenchmark"),
            description="Medical VQA generation and scoring benchmark",
        )
    if not registry.benchmarks.contains("medical_tasks"):
        registry.benchmarks.register(
            "medical_tasks",
            _factory("medumm.evaluation.medical_tasks", "MedicalTasksBenchmark"),
            description="Task-aware medical perception, reasoning, report, and communication benchmark",
        )
    if not registry.benchmarks.contains("cross_task"):
        registry.benchmarks.register(
            "cross_task",
            _factory("medumm.evaluation.cross_task", "CrossTaskBenchmark"),
            description="Composite evaluation over registered medical benchmarks",
        )
    from medumm.evaluation.benchmark_catalog import SPECIALIZED_BENCHMARKS

    for spec in SPECIALIZED_BENCHMARKS:
        if registry.benchmarks.contains(spec.name):
            continue
        registry.benchmarks.register(
            spec.name,
            _resource_factory(
                "medumm.evaluation.specialized",
                "specialized_benchmark_factory",
                spec.name,
            ),
            description=spec.description,
            metadata={
                "version": spec.version,
                "metric_suite": spec.metric_suite,
                "dataset_families": [value.value for value in spec.dataset_families],
                "validation": spec.validation,
                "composite": False,
            },
        )
    if not registry.trainers.contains("medical_sft"):
        registry.trainers.register(
            "medical_sft",
            _factory("medumm.post_training.medical_sft", "MedicalSFTTrainer"),
            description="Supervised training smoke implementation",
        )
    if not registry.trainers.contains("medical_alignment"):
        registry.trainers.register(
            "medical_alignment",
            _factory(
                "medumm.post_training.medical_alignment", "MedicalAlignmentTrainer"
            ),
            description="LoRA/QLoRA medical SFT, DPO, SimPO, ORPO, and relevance-weighted DPO",
            metadata={
                "objectives": ["sft", "dpo", "simpo", "orpo", "clinical_dpo"],
                "adapters": ["lora", "qlora"],
                "research_only": True,
            },
        )
    from medumm.post_training.research_routes import ROUTES

    for route in ROUTES.values():
        if registry.trainers.contains(route.name):
            continue

        def create_route_trainer(name: str = route.name):
            from medumm.post_training.research_routes import ResearchRouteTrainer

            return ResearchRouteTrainer(name)

        registry.trainers.register(
            route.name,
            create_route_trainer,
            description=route.summary,
            metadata={
                "display_name": route.display_name,
                "fidelity": route.fidelity,
                "stages": [stage.name for stage in route.stages],
                "paper_url": route.paper_url,
                "code_url": route.code_url,
                "research_only": True,
            },
        )

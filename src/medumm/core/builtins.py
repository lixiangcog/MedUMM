from __future__ import annotations

from importlib import import_module
from typing import Any

from medumm.core.registry import registry


def _factory(module_name: str, class_name: str):
    def create() -> Any:
        return getattr(import_module(module_name), class_name)()

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
    }
    for name, (module_name, class_name, description, metadata) in models.items():
        if not registry.models.contains(name):
            registry.models.register(
                name,
                _factory(module_name, class_name),
                description=description,
                metadata=metadata,
            )

    if not registry.datasets.contains("medical_vqa_jsonl"):
        registry.datasets.register(
            "medical_vqa_jsonl",
            _factory("medumm.medical.dataset", "MedicalVQADatasetAdapter"),
            description="Local JSON/JSONL medical visual question answering dataset",
        )
    if not registry.benchmarks.contains("medical_vqa"):
        registry.benchmarks.register(
            "medical_vqa",
            _factory("medumm.evaluation.medical_vqa", "MedicalVQABenchmark"),
            description="Medical VQA generation and scoring benchmark",
        )
    if not registry.benchmarks.contains("cross_task"):
        registry.benchmarks.register(
            "cross_task",
            _factory("medumm.evaluation.cross_task", "CrossTaskBenchmark"),
            description="Composite evaluation over registered medical benchmarks",
        )
    if not registry.trainers.contains("medical_sft"):
        registry.trainers.register(
            "medical_sft",
            _factory("medumm.post_training.medical_sft", "MedicalSFTTrainer"),
            description="Supervised training smoke implementation",
        )

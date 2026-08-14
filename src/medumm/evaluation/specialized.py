from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from medumm.core.builtins import register_builtins
from medumm.core.contracts import EvaluationMode
from medumm.core.interfaces import BenchmarkAdapter
from medumm.core.io import ensure_directory, write_json
from medumm.core.registry import registry
from medumm.core.results import Artifact, EvaluationResult
from medumm.core.runtime import RuntimeContext
from medumm.evaluation.benchmark_catalog import MedicalBenchmarkSpec, get_medical_benchmark
from medumm.evaluation.metrics import create_metric_suite
from medumm.evaluation.runner import EvaluationItem, EvaluationRunner
from medumm.inference import InferencePipeline
from medumm.medical.data import MedicalVQASample
from medumm.medical.tasks import MedicalTaskSample
from medumm.resources import DATASET_RESOURCES


SPECIALIZED_GROUPS = (
    "modality",
    "category",
    "answer_type",
    "language",
    "specialty",
    "anatomy",
    "medical_task",
)


@dataclass(frozen=True, slots=True)
class SpecializedBenchmarkProtocol:
    name: str
    version: str
    metric_suite: str
    metric_suite_version: str
    group_by: tuple[str, ...] = ("modality", "specialty")
    calibration_bins: int = 10
    selective_thresholds: tuple[float, ...] = (0.5, 0.7, 0.9)
    minimum_group_samples: int = 1
    minimum_samples: int = 1
    require_provenance: bool = False
    require_deidentified: bool = False
    require_complete_pairs: bool = True

    def __post_init__(self) -> None:
        if not self.name or not self.version:
            raise ValueError("Specialized benchmark protocol name/version cannot be empty.")
        unsupported = sorted(set(self.group_by) - set(SPECIALIZED_GROUPS))
        if unsupported:
            raise ValueError(f"Unsupported specialized benchmark groups: {unsupported}.")
        if self.calibration_bins < 2:
            raise ValueError("calibration_bins must be at least two.")
        if any(not 0 <= value <= 1 for value in self.selective_thresholds):
            raise ValueError("selective_thresholds must be between zero and one.")
        if self.minimum_group_samples < 1 or self.minimum_samples < 1:
            raise ValueError("Specialized benchmark minimums must be at least one.")

    @classmethod
    def from_config(
        cls,
        value: dict[str, Any] | None,
        *,
        spec: MedicalBenchmarkSpec,
        metric_suite_version: str,
    ) -> "SpecializedBenchmarkProtocol":
        config = dict(value or {})
        requested_suite = str(config.get("metric_suite", spec.metric_suite)).strip().lower()
        if requested_suite != spec.metric_suite:
            raise ValueError(
                f"{spec.name} fixes metric_suite={spec.metric_suite}; "
                f"received {requested_suite}."
            )
        raw_groups = config.get("group_by", ("modality", "specialty"))
        raw_thresholds = config.get("selective_thresholds", (0.5, 0.7, 0.9))
        if not isinstance(raw_groups, (list, tuple)):
            raise ValueError("evaluation.protocol.group_by must be a list.")
        if not isinstance(raw_thresholds, (list, tuple)):
            raise ValueError("evaluation.protocol.selective_thresholds must be a list.")
        return cls(
            name=str(config.get("name", spec.name)),
            version=str(config.get("version", spec.version)),
            metric_suite=spec.metric_suite,
            metric_suite_version=metric_suite_version,
            group_by=tuple(str(item) for item in raw_groups),
            calibration_bins=int(config.get("calibration_bins", 10)),
            selective_thresholds=tuple(float(item) for item in raw_thresholds),
            minimum_group_samples=int(config.get("minimum_group_samples", 1)),
            minimum_samples=int(config.get("minimum_samples", 1)),
            require_provenance=bool(config.get("require_provenance", False)),
            require_deidentified=bool(config.get("require_deidentified", False)),
            require_complete_pairs=bool(config.get("require_complete_pairs", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["group_by"] = list(self.group_by)
        result["selective_thresholds"] = list(self.selective_thresholds)
        return result


@dataclass(slots=True)
class _Sample:
    sample_id: str
    prompt: str
    images: list[str]
    volumes: list[str]
    videos: list[str]
    references: list[str]
    choices: dict[str, str]
    answer_type: str
    modality: str
    category: str
    language: str
    specialty: str
    anatomy: str
    medical_task: str
    annotations: dict[str, Any]
    metadata: dict[str, Any]


def _normalized_sample(sample: Any) -> _Sample:
    if isinstance(sample, MedicalVQASample):
        metadata = dict(sample.metadata)
        raw_annotations = metadata.get("annotations", {})
        return _Sample(
            sample_id=sample.sample_id,
            prompt=sample.question,
            images=list(sample.image_paths),
            volumes=list(sample.volume_paths),
            videos=list(sample.video_paths),
            references=list(sample.answers),
            choices=dict(sample.choices),
            answer_type=sample.answer_type,
            modality=sample.modality,
            category=sample.category,
            language=sample.language,
            specialty=str(metadata.get("specialty", "unknown")),
            anatomy=str(metadata.get("anatomy", "unknown")),
            medical_task=str(metadata.get("medical_task", "medical_vqa")),
            annotations=dict(raw_annotations) if isinstance(raw_annotations, dict) else {},
            metadata=metadata,
        )
    if isinstance(sample, MedicalTaskSample):
        return _Sample(
            sample_id=sample.sample_id,
            prompt=sample.prompt,
            images=list(sample.image_paths),
            volumes=list(sample.volume_paths),
            videos=list(sample.video_paths),
            references=list(sample.references),
            choices=dict(sample.choices),
            answer_type=sample.answer_type,
            modality=sample.modality,
            category=sample.task_family,
            language=sample.language,
            specialty=sample.specialty,
            anatomy=sample.anatomy,
            medical_task=sample.task.value,
            annotations=dict(sample.annotations),
            metadata=dict(sample.metadata),
        )
    raise TypeError(
        "Specialized medical benchmarks require MedicalVQASample or MedicalTaskSample data."
    )


def _resolved_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def _provenance(
    data_config: dict[str, Any], project_root: Path
) -> tuple[Path | None, dict[str, Any]]:
    raw = data_config.get("provenance")
    manifest = _resolved_path(str(data_config.get("path", "")), project_root)
    path = _resolved_path(str(raw), project_root) if raw else manifest.parent / "provenance.json"
    if not path.is_file():
        return None, {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Specialized benchmark provenance must be a JSON object.")
    return path, value


def _audit(
    samples: list[_Sample],
    *,
    data_config: dict[str, Any],
    project_root: Path,
    dataset_fingerprint: str,
    spec: MedicalBenchmarkSpec,
    protocol: SpecializedBenchmarkProtocol,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    manifest = _resolved_path(str(data_config.get("path", "")), project_root)
    provenance_path, provenance = _provenance(data_config, project_root)
    if protocol.require_provenance and provenance_path is None:
        errors.append("The specialized benchmark protocol requires provenance.")
    elif provenance_path is None:
        warnings.append("No dataset provenance document was discovered.")
    if protocol.require_deidentified and data_config.get("deidentified") is not True:
        errors.append("The specialized benchmark protocol requires deidentified: true.")
    if provenance.get("clinical_use") is True:
        errors.append("Clinical-use data cannot enter the research benchmark path.")
    if len(samples) < protocol.minimum_samples:
        errors.append(
            f"Dataset has {len(samples)} samples; protocol minimum is {protocol.minimum_samples}."
        )

    adapter = str(data_config.get("adapter", "medical_tasks_jsonl")).strip().lower()
    if adapter in DATASET_RESOURCES.names():
        resource = DATASET_RESOURCES.get(adapter)
        if resource.adapter_family not in spec.dataset_families:
            errors.append(
                f"Dataset {adapter} family {resource.adapter_family.value} is not valid for "
                f"{spec.name}."
            )
        if resource.benchmark != spec.name:
            errors.append(
                f"Dataset {adapter} is assigned to {resource.benchmark}, not {spec.name}."
            )

    missing_annotations = []
    missing_choices = []
    for sample in samples:
        if spec.required_annotation and spec.required_annotation not in sample.annotations:
            missing_annotations.append(sample.sample_id)
        if spec.requires_choices and not sample.choices:
            missing_choices.append(sample.sample_id)
    if missing_annotations:
        errors.append(
            f"{len(missing_annotations)} samples lack annotations.{spec.required_annotation}."
        )
    if missing_choices:
        errors.append(f"{len(missing_choices)} samples lack required choices.")

    if spec.name == "medical_robustness" and protocol.require_complete_pairs:
        pairs: dict[str, set[str]] = {}
        for sample in samples:
            annotation = sample.annotations.get("robustness", {})
            if isinstance(annotation, dict) and annotation.get("pair_id"):
                pairs.setdefault(str(annotation["pair_id"]), set()).add(
                    str(annotation.get("variant", "perturbed"))
                )
        incomplete = sorted(
            pair_id for pair_id, variants in pairs.items()
            if "baseline" not in variants or len(variants) < 2
        )
        if incomplete:
            errors.append(f"Robustness pairs are incomplete: {incomplete[:3]}.")
    if spec.name == "medical_fairness":
        groups = {
            str(sample.annotations.get("fairness", {}).get("group"))
            for sample in samples
            if isinstance(sample.annotations.get("fairness"), dict)
        }
        groups.discard("None")
        if len(groups) < 2:
            errors.append("Medical fairness requires at least two sensitive groups.")

    media = [
        path
        for sample in samples
        for path in (*sample.images, *sample.volumes, *sample.videos)
    ]
    missing_media = sorted(path for path in media if not Path(path).is_file())
    if missing_media:
        errors.append(f"{len(missing_media)} referenced media files are missing.")
    status = "failed" if errors else "warning" if warnings else "passed"
    return {
        "schema_version": "1.0",
        "benchmark": spec.name,
        "benchmark_version": spec.version,
        "status": status,
        "dataset_fingerprint": dataset_fingerprint,
        "sample_count": len(samples),
        "manifest": {
            "path": str(manifest),
            "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        },
        "media_count": len(media),
        "missing_media": missing_media,
        "answer_type_distribution": dict(
            sorted(Counter(sample.answer_type for sample in samples).items())
        ),
        "modality_distribution": dict(
            sorted(Counter(sample.modality for sample in samples).items())
        ),
        "required_annotation": spec.required_annotation,
        "missing_annotation_ids": missing_annotations,
        "missing_choice_ids": missing_choices,
        "governance": {
            "deidentified_declared": data_config.get("deidentified") is True,
            "research_only": provenance.get("clinical_use") is not True,
            "source": provenance.get("source"),
            "license": provenance.get("license"),
            "resolved_revision": provenance.get("resolved_revision"),
            "provenance_path": str(provenance_path) if provenance_path else None,
            "provenance_sha256": hashlib.sha256(provenance_path.read_bytes()).hexdigest()
            if provenance_path else None,
        },
        "warnings": warnings,
        "errors": errors,
    }


def _annotation_candidates(sample: _Sample, source: str | None) -> list[str]:
    if source == "choices":
        return list(sample.choices.values())
    if source == "retrieval":
        value = sample.annotations.get("retrieval", {})
        if isinstance(value, dict):
            raw = value.get("candidates", [])
            return [str(item) for item in raw] if isinstance(raw, list) else []
    return []


class SpecializedMedicalBenchmark(BenchmarkAdapter):
    """One executable benchmark bound to an immutable task/metric contract."""

    def __init__(self, name: str) -> None:
        self.spec = get_medical_benchmark(name)
        self.name = self.spec.name

    def run(
        self,
        config: dict[str, Any],
        *,
        config_path: str | Path,
        runtime: RuntimeContext,
    ) -> EvaluationResult:
        register_builtins()
        data_config = config.get("data")
        if not isinstance(data_config, dict):
            raise ValueError(f"{self.name} requires a data mapping.")
        mode = EvaluationMode(config.get("mode", "full"))
        raw_model = config.get("model", {})
        if mode is not EvaluationMode.AUDIT and not isinstance(raw_model, dict):
            raise ValueError(f"{self.name} requires a model mapping.")
        model_config = dict(raw_model) if isinstance(raw_model, dict) else {}
        dataset_name = str(data_config.get("adapter", "medical_tasks_jsonl")).strip().lower()
        dataset = registry.datasets.create(dataset_name)
        samples = [
            _normalized_sample(value)
            for value in dataset.load(data_config, runtime.project_root)
        ]
        benchmark_filter = str(data_config.get("benchmark_filter", "")).strip().lower()
        if benchmark_filter:
            samples = [
                sample
                for sample in samples
                if str(sample.metadata.get("benchmark", "")).strip().lower()
                == benchmark_filter
            ]
            if not samples:
                raise ValueError(
                    f"No samples match data.benchmark_filter={benchmark_filter!r}."
                )
        dataset_fingerprint = dataset.fingerprint(data_config, runtime.project_root)
        metric_suite = create_metric_suite(self.spec.metric_suite)
        raw_protocol = config.get("protocol")
        if raw_protocol is not None and not isinstance(raw_protocol, dict):
            raise ValueError("Evaluation protocol must be a mapping when provided.")
        protocol = SpecializedBenchmarkProtocol.from_config(
            raw_protocol,
            spec=self.spec,
            metric_suite_version=metric_suite.version,
        )
        output_directory = _resolved_path(
            config.get("output_directory", f"outputs/evaluation/{self.name}"),
            runtime.project_root,
        )
        ensure_directory(output_directory)
        audit = _audit(
            samples,
            data_config=data_config,
            project_root=runtime.project_root,
            dataset_fingerprint=dataset_fingerprint,
            spec=self.spec,
            protocol=protocol,
        )
        audit_path = write_json(output_directory / "dataset_audit.json", audit)
        if audit["status"] == "failed":
            raise ValueError(
                f"{self.name} dataset audit failed: " + "; ".join(audit["errors"])
            )
        if mode is EvaluationMode.AUDIT:
            return EvaluationResult(
                benchmark=self.name,
                mode=mode.value,
                status=str(audit["status"]),
                dataset_size=len(samples),
                output_directory=str(output_directory),
                metrics={"data_quality": audit},
                artifacts=[Artifact("dataset_audit", str(audit_path), "application/json")],
                metadata={
                    "dataset": dataset_name,
                    "dataset_fingerprint": dataset_fingerprint,
                    "benchmark_spec": self.spec.to_dict(),
                    "protocol": protocol.to_dict(),
                    "clinical_use": False,
                },
            )

        backbone = str(model_config.get("backbone", "")).strip().lower()
        if not backbone:
            raise ValueError(f"{self.name} model requires a backbone.")
        prompt_template = str(config.get("prompt_template", self.spec.prompt_template))
        if "{prompt}" not in prompt_template:
            raise ValueError(f"{self.name} prompt_template must contain {{prompt}}.")
        model_identity = {
            "backbone": backbone,
            "config": model_config.get("config", {}),
            "parameters": model_config.get("parameters", {}),
            "prompt_template": prompt_template,
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "dataset": dataset_fingerprint,
                    "benchmark": self.spec.to_dict(),
                    "protocol": protocol.to_dict(),
                    "model": model_identity,
                },
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        items = []
        for sample in samples:
            parameters = dict(model_config.get("parameters", {}))
            candidates = _annotation_candidates(sample, self.spec.candidate_source)
            if candidates and "candidates" not in parameters:
                parameters["candidates"] = candidates
            items.append(
                EvaluationItem(
                    sample_id=sample.sample_id,
                    request={
                        "request_id": sample.sample_id,
                        "task": "understanding",
                        "prompt": prompt_template.format(prompt=sample.prompt),
                        "images": sample.images,
                        "volumes": sample.volumes,
                        "videos": sample.videos,
                        "parameters": parameters,
                        "metadata": {
                            "benchmark": self.name,
                            "benchmark_version": self.spec.version,
                            "medical_task": sample.medical_task,
                            "specialty": sample.specialty,
                            "modality": sample.modality,
                        },
                    },
                    content={
                        "references": sample.references,
                        "choices": sample.choices,
                        "answer_type": sample.answer_type,
                        "modality": sample.modality,
                        "category": sample.category,
                        "language": sample.language,
                        "specialty": sample.specialty,
                        "anatomy": sample.anatomy,
                        "medical_task": sample.medical_task,
                        "annotations": sample.annotations,
                        "sample_metadata": sample.metadata,
                    },
                )
            )

        pipeline = None
        if mode is not EvaluationMode.SCORE:
            pipeline = InferencePipeline(
                backbone,
                dict(model_config.get("config", {})),
                runtime=runtime,
            )
        try:
            raw_shard = config.get("shard", {})
            if not isinstance(raw_shard, dict):
                raise ValueError("Evaluation shard must be a mapping when provided.")
            shard_count = int(
                raw_shard.get("count", runtime.world_size if runtime.distributed else 1)
            )
            shard_rank = int(raw_shard.get("rank", runtime.rank if shard_count > 1 else 0))
            runner = EvaluationRunner(
                benchmark=self.name,
                pipeline=pipeline,
                output_directory=output_directory,
                parser=lambda output: str(output.text or ""),
                scorer=metric_suite.score,
                summarizer=lambda rows: metric_suite.summarize(rows, protocol.to_dict()),
                mode=mode,
                resume=bool(config.get("resume", True)),
                fingerprint=fingerprint,
                metadata={
                    "dataset": dataset_name,
                    "dataset_fingerprint": dataset_fingerprint,
                    "model": backbone,
                    "benchmark_spec": self.spec.to_dict(),
                    "protocol": protocol.to_dict(),
                    "dataset_audit": {
                        "status": audit["status"],
                        "warnings": len(audit["warnings"]),
                    },
                    "clinical_use": False,
                },
                batch_size=int(config.get("batch_size", 1)),
                shard_rank=shard_rank,
                shard_count=shard_count,
                prediction_flush_interval=int(
                    config.get("checkpoint", {}).get("flush_every", config.get("batch_size", 1))
                ),
            )
            shard_items = items[runner.shard_rank :: runner.shard_count]
            if not shard_items:
                raise ValueError(f"Evaluation shard {runner.shard_rank} has no samples.")
            result = runner.run(shard_items)
            result.artifacts.append(
                Artifact("dataset_audit", str(audit_path), "application/json")
            )
            return result
        finally:
            if pipeline is not None:
                pipeline.close()


def specialized_benchmark_factory(name: str):
    def create() -> SpecializedMedicalBenchmark:
        return SpecializedMedicalBenchmark(name)

    return create

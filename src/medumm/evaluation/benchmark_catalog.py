from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from medumm.resources import DatasetAdapterFamily


@dataclass(frozen=True, slots=True)
class MedicalBenchmarkSpec:
    """Stable task, data, prompt, and metric contract for one benchmark family."""

    name: str
    version: str
    description: str
    metric_suite: str
    prompt_template: str
    dataset_families: tuple[DatasetAdapterFamily, ...]
    required_annotation: str | None = None
    requires_choices: bool = False
    candidate_source: str | None = None
    validation: str = "interface_validated"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["dataset_families"] = [value.value for value in self.dataset_families]
        return result


_VQA = DatasetAdapterFamily.VQA
_TASK = DatasetAdapterFamily.MEDICAL_TASK
_CLASSIFICATION = DatasetAdapterFamily.CLASSIFICATION
_REPORT = DatasetAdapterFamily.REPORT_GENERATION
_DETECTION = DatasetAdapterFamily.DETECTION_MEASUREMENT
_RETRIEVAL = DatasetAdapterFamily.RETRIEVAL
_VIDEO = DatasetAdapterFamily.VIDEO


SPECIALIZED_BENCHMARKS = (
    MedicalBenchmarkSpec(
        name="pathology_vqa",
        version="1.0",
        description="Pathology yes/no and free-form visual question answering",
        metric_suite="pathology_vqa",
        prompt_template="{prompt}\nAnswer the pathology question concisely.",
        dataset_families=(_VQA,),
    ),
    MedicalBenchmarkSpec(
        name="medical_mcqa",
        version="1.0",
        description="Expert medical multiple-choice visual reasoning",
        metric_suite="medical_mcqa",
        prompt_template="{prompt}\nReturn the single best option letter and answer.",
        dataset_families=(_VQA, _TASK),
        requires_choices=True,
        candidate_source="choices",
    ),
    MedicalBenchmarkSpec(
        name="medical_image_classification",
        version="1.0",
        description="Medical image classification with imbalance-aware scoring",
        metric_suite="medical_image_classification",
        prompt_template="{prompt}\nSelect exactly one medical class.",
        dataset_families=(_CLASSIFICATION,),
        requires_choices=True,
        candidate_source="choices",
    ),
    MedicalBenchmarkSpec(
        name="medical_multilabel_findings",
        version="1.0",
        description="Multilabel radiology and pathology finding recognition",
        metric_suite="medical_multilabel_findings",
        prompt_template="{prompt}\nReturn all supported findings and no unsupported findings.",
        dataset_families=(_CLASSIFICATION, _REPORT),
        required_annotation="multilabel",
    ),
    MedicalBenchmarkSpec(
        name="radiology_report_generation",
        version="1.0",
        description="Structured radiology report factuality and critical-finding coverage",
        metric_suite="medical_report_factuality",
        prompt_template="{prompt}\nWrite Findings and Impression sections.",
        dataset_families=(_REPORT,),
        required_annotation="report",
    ),
    MedicalBenchmarkSpec(
        name="medical_grounding",
        version="1.0",
        description="Medical box and point localization in normalized coordinates",
        metric_suite="medical_grounding",
        prompt_template=(
            "{prompt}\nReturn JSON with boxes and/or points in image coordinates."
        ),
        dataset_families=(_DETECTION, _REPORT),
        required_annotation="grounding",
    ),
    MedicalBenchmarkSpec(
        name="medical_measurement",
        version="1.0",
        description="Unit-aware quantitative medical measurement",
        metric_suite="medical_measurement",
        prompt_template="{prompt}\nReturn each measurement value and physical unit.",
        dataset_families=(_DETECTION,),
        required_annotation="measurements",
    ),
    MedicalBenchmarkSpec(
        name="medical_temporal_reasoning",
        version="1.0",
        description="Surgical and longitudinal phase/action sequence reasoning",
        metric_suite="medical_temporal_reasoning",
        prompt_template="{prompt}\nReturn the ordered sequence of phases or actions.",
        dataset_families=(_VIDEO,),
        required_annotation="temporal",
    ),
    MedicalBenchmarkSpec(
        name="medical_retrieval",
        version="1.0",
        description="Medical image-text retrieval with Recall@K and MRR",
        metric_suite="medical_retrieval",
        prompt_template="{prompt}",
        dataset_families=(_RETRIEVAL, _REPORT),
        required_annotation="retrieval",
        candidate_source="retrieval",
    ),
    MedicalBenchmarkSpec(
        name="medical_calibration",
        version="1.0",
        description="Probability calibration and selective medical prediction",
        metric_suite="medical_calibration",
        prompt_template="{prompt}\nSelect one answer and retain candidate confidence scores.",
        dataset_families=(_VQA, _TASK, _CLASSIFICATION),
        requires_choices=True,
        candidate_source="choices",
    ),
    MedicalBenchmarkSpec(
        name="medical_fairness",
        version="1.0",
        description="Subgroup performance and prediction-rate disparity",
        metric_suite="medical_fairness",
        prompt_template="{prompt}\nSelect the medically supported answer only.",
        dataset_families=(_CLASSIFICATION, _TASK),
        required_annotation="fairness",
        requires_choices=True,
        candidate_source="choices",
    ),
    MedicalBenchmarkSpec(
        name="medical_safety",
        version="1.0",
        description="Unsafe compliance, expected refusal, and over-refusal",
        metric_suite="medical_safety",
        prompt_template="{prompt}\nFollow the research safety policy and state limitations.",
        dataset_families=(_TASK,),
        required_annotation="safety",
    ),
    MedicalBenchmarkSpec(
        name="medical_robustness",
        version="1.0",
        description="Paired medical prompt/image perturbation consistency",
        metric_suite="medical_robustness",
        prompt_template="{prompt}\nAnswer using only visible evidence.",
        dataset_families=(_VQA, _TASK, _CLASSIFICATION),
        required_annotation="robustness",
    ),
)


_BY_NAME = {spec.name: spec for spec in SPECIALIZED_BENCHMARKS}
if len(_BY_NAME) != len(SPECIALIZED_BENCHMARKS):
    raise ValueError("Duplicate specialized medical benchmark names.")


def get_medical_benchmark(name: str) -> MedicalBenchmarkSpec:
    try:
        return _BY_NAME[name.strip().lower()]
    except KeyError as error:
        raise KeyError(f"Unknown specialized medical benchmark: {name!r}.") from error


def medical_benchmark_catalog() -> list[dict[str, Any]]:
    return [spec.to_dict() for spec in SPECIALIZED_BENCHMARKS]

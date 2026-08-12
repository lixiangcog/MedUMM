from __future__ import annotations

import math
import random
import re
from typing import Any

from medumm.medical.metrics import evaluate_answer, normalize_answer
from medumm.medical.tasks import MedicalTaskType


def _contains(text: str, concept: str) -> bool:
    normalized_text = f" {normalize_answer(text)} "
    normalized_concept = f" {normalize_answer(concept)} "
    return normalized_concept.strip() != "" and normalized_concept in normalized_text


def _affirmed(text: str, concept: str) -> bool:
    """Conservative negation guard for concept-level hallucination counts."""

    normalized_text = normalize_answer(text)
    normalized_concept = normalize_answer(concept)
    if not normalized_concept:
        return False
    escaped = re.escape(normalized_concept)
    for match in re.finditer(rf"(?<!\w){escaped}(?!\w)", normalized_text):
        preceding = normalized_text[: match.start()].split()[-4:]
        if not any(
            token in {"no", "not", "without", "absent", "negative", "denies", "deny"}
            for token in preceding
        ):
            return True
    return False


def _concept_scores(
    prediction: str,
    expected: list[str],
    vocabulary: list[str],
) -> dict[str, float | int | None]:
    expected_normalized = {
        normalize_answer(concept): concept for concept in expected if normalize_answer(concept)
    }
    matched = {
        normalized
        for normalized, original in expected_normalized.items()
        if _affirmed(prediction, original)
    }
    vocabulary_normalized = {
        normalize_answer(concept): concept
        for concept in vocabulary
        if normalize_answer(concept)
    }
    mentioned = {
        normalized
        for normalized, original in vocabulary_normalized.items()
        if _affirmed(prediction, original)
    }
    extras = mentioned - set(expected_normalized)
    recall = len(matched) / len(expected_normalized) if expected_normalized else None
    if vocabulary_normalized:
        precision = len(matched) / (len(matched) + len(extras)) if matched or extras else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if recall is not None and precision + recall > 0
            else 0.0
        )
        extra_rate = len(extras) / len(mentioned) if mentioned else 0.0
    else:
        precision = None
        f1 = recall
        extra_rate = None
    return {
        "expected_concept_count": len(expected_normalized),
        "matched_concept_count": len(matched),
        "extra_concept_count": len(extras),
        "concept_precision": precision,
        "concept_recall": recall,
        "concept_f1": f1,
        "extra_concept_rate": extra_rate,
    }


def evaluate_medical_task(prediction: str, content: dict[str, Any]) -> dict[str, Any]:
    task = MedicalTaskType(str(content["medical_task"]))
    base = evaluate_answer(
        prediction,
        list(content["references"]),
        dict(content.get("choices", {})),
    )
    concepts = [str(value) for value in content.get("concepts", [])]
    vocabulary = [
        str(value) for value in content.get("concept_vocabulary", [])
    ]
    concept_scores = _concept_scores(prediction, concepts, vocabulary)
    evidence = [str(value) for value in content.get("evidence", [])]
    evidence_hits = sum(_affirmed(prediction, value) for value in evidence)
    evidence_coverage = evidence_hits / len(evidence) if evidence else None

    strict_diagnosis = None
    if task is MedicalTaskType.FINDING_ASSESSMENT:
        success = float(base["exact_match"])
    elif task in {
        MedicalTaskType.ANATOMY_LOCALIZATION,
        MedicalTaskType.QUANTITATIVE_ASSESSMENT,
        MedicalTaskType.IMAGE_CONTEXT,
    }:
        success = float(
            base["exact_match"]
            if content.get("choices") or not concepts
            else concept_scores["concept_recall"] == 1.0
        )
    elif task is MedicalTaskType.DIAGNOSTIC_REASONING:
        concept_complete = (
            concept_scores["concept_recall"] == 1.0
            if concepts
            else bool(base["exact_match"])
        )
        no_extras = concept_scores["extra_concept_count"] == 0
        strict_diagnosis = float(concept_complete and no_extras)
        evidence_complete = evidence_coverage == 1.0 if evidence else False
        success = float(bool(strict_diagnosis) and evidence_complete)
    else:
        coverage = concept_scores["concept_recall"]
        success = float(coverage is not None and coverage >= 0.8)

    return {
        **base,
        **concept_scores,
        "evidence_count": len(evidence),
        "matched_evidence_count": evidence_hits,
        "evidence_coverage": evidence_coverage,
        "task_success": success,
        "strict_diagnostic_accuracy": strict_diagnosis,
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def _percent(value: float | None) -> float | None:
    return round(100 * value, 2) if value is not None else None


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    diagnostic = [row for row in rows if row.get("strict_diagnostic_accuracy") is not None]
    return {
        "total": len(rows),
        "task_success": _percent(_mean(rows, "task_success")),
        "exact_match": _percent(_mean(rows, "exact_match")),
        "token_f1": _percent(_mean(rows, "token_f1")),
        "concept_precision": _percent(_mean(rows, "concept_precision")),
        "concept_recall": _percent(_mean(rows, "concept_recall")),
        "concept_f1": _percent(_mean(rows, "concept_f1")),
        "evidence_coverage": _percent(_mean(rows, "evidence_coverage")),
        "extra_concept_rate": _percent(_mean(rows, "extra_concept_rate")),
        "strict_diagnostic_accuracy": _percent(
            _mean(diagnostic, "strict_diagnostic_accuracy")
        ),
        "abstention_rate": _percent(_mean(rows, "abstained")),
    }


def _wilson_interval(
    successes: int,
    total: int,
    confidence_level: float,
) -> dict[str, float | int] | None:
    if total <= 0:
        return None
    # Supported confidence levels cover the evaluation protocols shipped here.
    z_by_confidence = {0.9: 1.644854, 0.95: 1.959964, 0.99: 2.575829}
    z = z_by_confidence.get(round(confidence_level, 2))
    if z is None:
        raise ValueError("Wilson intervals support confidence levels 0.90, 0.95, or 0.99.")
    rate = successes / total
    denominator = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total**2)) / denominator
    return {
        "method": "wilson",
        "confidence_level": confidence_level,
        "lower": round(100 * max(0.0, centre - margin), 2),
        "upper": round(100 * min(1.0, centre + margin), 2),
        "total": total,
    }


def _bootstrap_interval(
    values: list[float],
    *,
    samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float | int | str] | None:
    if not values or samples <= 0:
        return None
    generator = random.Random(seed)
    means = sorted(
        sum(generator.choice(values) for _ in values) / len(values)
        for _ in range(samples)
    )
    tail = (1 - confidence_level) / 2
    lower = means[max(0, int(tail * samples))]
    upper = means[min(samples - 1, max(0, int((1 - tail) * samples) - 1))]
    return {
        "method": "bootstrap",
        "confidence_level": confidence_level,
        "lower": round(100 * lower, 2),
        "upper": round(100 * upper, 2),
        "bootstrap_samples": samples,
    }


def summarize_medical_tasks(
    rows: list[dict[str, Any]],
    *,
    group_by: tuple[str, ...],
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"overall": _aggregate(rows)}
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(str(row["medical_task"]), []).append(row)
    summary["macro_task_success"] = round(
        sum(_mean(values, "task_success") or 0.0 for values in by_task.values())
        / len(by_task)
        * 100,
        2,
    )
    uncertainty: dict[str, Any] = {}
    for offset, metric in enumerate(("token_f1", "concept_recall", "evidence_coverage")):
        interval = _bootstrap_interval(
            [float(row[metric]) for row in rows if row.get(metric) is not None],
            samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed + offset,
        )
        if interval is not None:
            uncertainty[metric] = interval
    task_successes = [float(row["task_success"]) for row in rows]
    uncertainty["task_success"] = _wilson_interval(
        sum(value >= 1.0 for value in task_successes),
        len(task_successes),
        confidence_level,
    )
    diagnosis = [
        float(row["strict_diagnostic_accuracy"])
        for row in rows
        if row.get("strict_diagnostic_accuracy") is not None
    ]
    if diagnosis:
        uncertainty["strict_diagnostic_accuracy"] = _wilson_interval(
            sum(value >= 1.0 for value in diagnosis), len(diagnosis), confidence_level
        )
    summary["uncertainty"] = uncertainty
    for field in group_by:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row.get(field, "unknown")), []).append(row)
        summary[f"by_{field}"] = {
            name: _aggregate(values) for name, values in sorted(grouped.items())
        }
    return summary

from __future__ import annotations

import json
import math
import re
from typing import Any, Iterable

from medumm.medical.metrics import evaluate_answer, normalize_answer


NEGATION_TERMS = {
    "no",
    "not",
    "without",
    "absent",
    "negative",
    "neither",
    "nor",
    "denies",
    "deny",
}
LENGTH_TO_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0}


def _values(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _strings(value: Any) -> list[str]:
    return [str(item).strip() for item in _values(value) if str(item).strip()]


def _annotation(content: dict[str, Any], name: str) -> Any:
    if name in content:
        return content[name]
    annotations = content.get("annotations", {})
    if isinstance(annotations, dict) and name in annotations:
        return annotations[name]
    metadata = content.get("sample_metadata", {})
    if isinstance(metadata, dict):
        nested = metadata.get("annotations", {})
        if isinstance(nested, dict) and name in nested:
            return nested[name]
        return metadata.get(name)
    return None


def _payload(prediction: str) -> dict[str, Any]:
    candidates = [prediction.strip()]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", prediction, re.DOTALL | re.I)
    if fenced:
        candidates.insert(0, fenced.group(1))
    embedded = re.search(r"\{.*\}", prediction, re.DOTALL)
    if embedded:
        candidates.append(embedded.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    return {}


def _concept_status(text: str, concept: str) -> str:
    normalized_concept = normalize_answer(concept)
    if not normalized_concept:
        return "absent"
    escaped = re.escape(normalized_concept)
    statuses: list[str] = []
    # Negation scope is deliberately limited to a punctuation/coordinating
    # clause so "No effusion. Impression: pneumonia" does not negate pneumonia.
    for clause in re.split(r"[.;!?\n]|\b(?:but|however|although)\b", text, flags=re.I):
        normalized_clause = normalize_answer(clause)
        for match in re.finditer(rf"(?<!\w){escaped}(?!\w)", normalized_clause):
            preceding = normalized_clause[: match.start()].split()[-5:]
            statuses.append(
                "negated"
                if any(token in NEGATION_TERMS for token in preceding)
                else "affirmed"
            )
    if "affirmed" in statuses:
        return "affirmed"
    return "negated" if statuses else "absent"


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate_report(prediction: str, content: dict[str, Any]) -> dict[str, Any]:
    """Score only explicitly annotated report facts; missing annotations stay unavailable."""

    raw = _annotation(content, "report")
    annotation = dict(raw) if isinstance(raw, dict) else {}
    positives = _strings(annotation.get("positive_findings", content.get("concepts", [])))
    negatives = _strings(annotation.get("negative_findings", []))
    critical = _strings(annotation.get("critical_findings", []))
    vocabulary = _strings(annotation.get("finding_vocabulary", content.get("concept_vocabulary", [])))
    required_sections = _strings(annotation.get("required_sections", []))
    available = bool(positives or negatives or critical or required_sections)
    if not available:
        return {"report_available": False}

    positive_hits = sum(_concept_status(prediction, concept) == "affirmed" for concept in positives)
    negative_hits = sum(_concept_status(prediction, concept) == "negated" for concept in negatives)
    contradictions = sum(_concept_status(prediction, concept) == "affirmed" for concept in negatives)
    affirmed = {
        normalize_answer(concept)
        for concept in vocabulary
        if _concept_status(prediction, concept) == "affirmed"
    }
    expected = {normalize_answer(concept) for concept in positives if normalize_answer(concept)}
    extras = affirmed - expected
    precision = _ratio(positive_hits, positive_hits + len(extras))
    recall = _ratio(positive_hits, len(expected))
    critical_hits = sum(_concept_status(prediction, concept) == "affirmed" for concept in critical)
    normalized_prediction = normalize_answer(prediction)
    section_hits = sum(
        bool(re.search(rf"(?:^|\s){re.escape(normalize_answer(section))}(?:\s|$)", normalized_prediction))
        for section in required_sections
    )
    return {
        "report_available": True,
        "report_positive_count": len(expected),
        "report_matched_positive_count": positive_hits,
        "report_extra_finding_count": len(extras),
        "report_contradiction_count": contradictions,
        "report_fact_precision": precision,
        "report_fact_recall": recall,
        "report_fact_f1": _f1(precision, recall),
        "report_negative_assertion_accuracy": _ratio(negative_hits, len(negatives)),
        "report_contradiction_rate": _ratio(contradictions, len(negatives)),
        "report_critical_recall": _ratio(critical_hits, len(critical)),
        "report_section_completeness": _ratio(section_hits, len(required_sections)),
    }


def _box(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, dict):
        value = [value.get(key) for key in ("x1", "y1", "x2", "y2")]
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _point(value: Any) -> tuple[float, float] | None:
    if isinstance(value, dict):
        value = [value.get("x"), value.get("y")]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        x, y = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    return (x, y) if math.isfinite(x) and math.isfinite(y) else None


def _image_size(annotation: dict[str, Any]) -> tuple[float, float] | None:
    value = annotation.get("image_size")
    if isinstance(value, dict):
        value = [value.get("width"), value.get("height")]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        width, height = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    return (width, height) if width > 0 and height > 0 else None


def _normalized_box(
    box: tuple[float, float, float, float], size: tuple[float, float] | None
) -> tuple[float, float, float, float]:
    if max(abs(value) for value in box) <= 1.0:
        return box
    if size is None:
        raise ValueError("Pixel-space grounding boxes require annotations.grounding.image_size.")
    width, height = size
    return box[0] / width, box[1] / height, box[2] / width, box[3] / height


def _normalized_point(
    point: tuple[float, float], size: tuple[float, float] | None
) -> tuple[float, float]:
    if max(abs(value) for value in point) <= 1.0:
        return point
    if size is None:
        raise ValueError("Pixel-space grounding points require annotations.grounding.image_size.")
    return point[0] / size[0], point[1] / size[1]


def _iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def evaluate_grounding(prediction: str, content: dict[str, Any]) -> dict[str, Any]:
    raw = _annotation(content, "grounding")
    annotation = dict(raw) if isinstance(raw, dict) else {}
    reference_boxes = [box for item in _values(annotation.get("boxes")) if (box := _box(item))]
    reference_points = [point for item in _values(annotation.get("points")) if (point := _point(item))]
    if not reference_boxes and not reference_points:
        return {"grounding_available": False}
    payload = _payload(prediction)
    raw_prediction = payload.get("grounding", payload)
    if not isinstance(raw_prediction, dict):
        raw_prediction = {}
    predicted_boxes = [
        box
        for item in _values(raw_prediction.get("boxes", raw_prediction.get("box")))
        if (box := _box(item))
    ]
    predicted_points = [
        point
        for item in _values(raw_prediction.get("points", raw_prediction.get("point")))
        if (point := _point(item))
    ]
    size = _image_size(annotation)
    refs = [_normalized_box(box, size) for box in reference_boxes]
    preds = [_normalized_box(box, size) for box in predicted_boxes]
    ref_points = [_normalized_point(point, size) for point in reference_points]
    pred_points = [_normalized_point(point, size) for point in predicted_points]

    best_ious = [max((_iou(reference, predicted) for predicted in preds), default=0.0) for reference in refs]
    distances = [
        min((math.dist(reference, predicted) / math.sqrt(2) for predicted in pred_points), default=1.0)
        for reference in ref_points
    ]
    point_hits = [
        any(box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3] for box in refs)
        for point in pred_points
    ]
    return {
        "grounding_available": True,
        "reference_box_count": len(refs),
        "predicted_box_count": len(preds),
        "reference_point_count": len(ref_points),
        "predicted_point_count": len(pred_points),
        "grounding_mean_iou": sum(best_ious) / len(best_ious) if best_ious else None,
        "grounding_iou_50_recall": _ratio(sum(value >= 0.5 for value in best_ious), len(best_ious)),
        "grounding_normalized_point_distance": sum(distances) / len(distances) if distances else None,
        "grounding_pointing_accuracy": _ratio(sum(point_hits), len(point_hits)),
    }


def _measurement(value: Any) -> dict[str, Any] | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return {"value": float(value), "unit": ""}
    if not isinstance(value, dict):
        return None
    try:
        number = float(value["value"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return {**value, "value": number, "unit": str(value.get("unit", "")).strip().casefold()}


def _parsed_measurements(prediction: str) -> list[dict[str, Any]]:
    payload = _payload(prediction)
    raw = payload.get("measurements", payload.get("measurement"))
    parsed = [item for value in _values(raw) if (item := _measurement(value))]
    if parsed:
        return parsed
    match = re.search(r"(?<!\w)([-+]?\d+(?:\.\d+)?)\s*(mm|cm|m)?\b", prediction, re.I)
    if not match:
        return []
    return [{"value": float(match.group(1)), "unit": (match.group(2) or "").casefold()}]


def _convert(value: float, source: str, target: str) -> float | None:
    if source == target or not source and not target:
        return value
    if source in LENGTH_TO_MM and target in LENGTH_TO_MM:
        return value * LENGTH_TO_MM[source] / LENGTH_TO_MM[target]
    return None


def evaluate_measurement(prediction: str, content: dict[str, Any]) -> dict[str, Any]:
    raw = _annotation(content, "measurements")
    if raw is None:
        raw = _annotation(content, "measurement")
    references = [item for value in _values(raw) if (item := _measurement(value))]
    if not references:
        return {"measurement_available": False}
    predictions = _parsed_measurements(prediction)
    used: set[int] = set()
    absolute_errors: list[float] = []
    relative_errors: list[float] = []
    tolerance_hits: list[bool] = []
    unit_errors = 0
    for index, reference in enumerate(references):
        name = normalize_answer(reference.get("name", ""))
        candidates = [
            (candidate_index, candidate)
            for candidate_index, candidate in enumerate(predictions)
            if candidate_index not in used
            and (
                not name
                or not normalize_answer(candidate.get("name", ""))
                or normalize_answer(candidate.get("name", "")) == name
            )
        ]
        if not candidates:
            continue
        candidate_index, candidate = candidates[0]
        used.add(candidate_index)
        converted = _convert(candidate["value"], candidate["unit"], reference["unit"])
        if converted is None:
            unit_errors += 1
            continue
        absolute = abs(converted - reference["value"])
        relative = absolute / abs(reference["value"]) if reference["value"] else float(absolute > 0)
        absolute_errors.append(absolute)
        relative_errors.append(relative)
        absolute_tolerance = float(reference.get("absolute_tolerance", 0.0))
        relative_tolerance = float(reference.get("relative_tolerance", 0.0))
        tolerance_hits.append(
            absolute <= absolute_tolerance
            or (relative_tolerance > 0 and relative <= relative_tolerance)
        )
    matched = len(absolute_errors)
    return {
        "measurement_available": True,
        "reference_measurement_count": len(references),
        "parsed_measurement_count": len(predictions),
        "matched_measurement_count": matched,
        "measurement_unit_error_count": unit_errors,
        "measurement_mae": sum(absolute_errors) / matched if matched else None,
        "measurement_mre": sum(relative_errors) / matched if matched else None,
        "measurement_within_tolerance": _ratio(sum(tolerance_hits), len(references)),
    }


def _reference_option(content: dict[str, Any]) -> str:
    choices = {str(key).upper(): str(value) for key, value in dict(content.get("choices", {})).items()}
    references = _strings(content.get("references", []))
    for reference in references:
        normalized = normalize_answer(reference)
        if normalized.upper() in choices:
            return normalized.upper()
        for key, option in choices.items():
            if normalized == normalize_answer(option):
                return key
    return ""


def _probabilities(scores: dict[str, Any], choices: dict[str, str]) -> dict[str, float]:
    values: dict[str, float] = {}
    for key, option in choices.items():
        raw = scores.get(key, scores.get(option))
        try:
            number = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and number >= 0:
            values[key] = number
    total = sum(values.values())
    return {key: value / total for key, value in values.items()} if total > 0 else {}


def evaluate_calibration(prediction: str, content: dict[str, Any]) -> dict[str, Any]:
    choices = {str(key).upper(): str(value) for key, value in dict(content.get("choices", {})).items()}
    scores = content.get("model_scores", {})
    if not choices or not isinstance(scores, dict):
        return {"calibration_available": False}
    probabilities = _probabilities(scores, choices)
    reference = _reference_option(content)
    if len(probabilities) != len(choices) or not reference:
        return {"calibration_available": False}
    base = evaluate_answer(prediction, _strings(content.get("references", [])), choices)
    confidence = max(probabilities.values())
    correct = float(base["exact_match"])
    epsilon = 1e-12
    return {
        "calibration_available": True,
        "calibration_confidence": confidence,
        "calibration_correct": correct,
        "calibration_brier": sum(
            (probability - float(key == reference)) ** 2
            for key, probability in probabilities.items()
        ),
        "calibration_nll": -math.log(max(probabilities[reference], epsilon)),
    }


def _mean(rows: Iterable[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def _percentage(value: float | None) -> float | None:
    return round(value * 100, 2) if value is not None else None


def summarize_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    available = [row for row in rows if row.get("report_available")]
    return {
        "total": len(rows),
        "annotated": len(available),
        "fact_precision": _percentage(_mean(available, "report_fact_precision")),
        "fact_recall": _percentage(_mean(available, "report_fact_recall")),
        "fact_f1": _percentage(_mean(available, "report_fact_f1")),
        "negative_assertion_accuracy": _percentage(_mean(available, "report_negative_assertion_accuracy")),
        "contradiction_rate": _percentage(_mean(available, "report_contradiction_rate")),
        "critical_recall": _percentage(_mean(available, "report_critical_recall")),
        "section_completeness": _percentage(_mean(available, "report_section_completeness")),
    }


def summarize_grounding(rows: list[dict[str, Any]]) -> dict[str, Any]:
    available = [row for row in rows if row.get("grounding_available")]
    return {
        "total": len(rows),
        "annotated": len(available),
        "mean_iou": _percentage(_mean(available, "grounding_mean_iou")),
        "iou_50_recall": _percentage(_mean(available, "grounding_iou_50_recall")),
        "normalized_point_distance": _mean(available, "grounding_normalized_point_distance"),
        "pointing_accuracy": _percentage(_mean(available, "grounding_pointing_accuracy")),
    }


def summarize_measurement(rows: list[dict[str, Any]]) -> dict[str, Any]:
    available = [row for row in rows if row.get("measurement_available")]
    return {
        "total": len(rows),
        "annotated": len(available),
        "mae": _mean(available, "measurement_mae"),
        "mean_relative_error": _percentage(_mean(available, "measurement_mre")),
        "within_tolerance": _percentage(_mean(available, "measurement_within_tolerance")),
        "unit_error_count": sum(int(row.get("measurement_unit_error_count", 0)) for row in available),
    }


def summarize_calibration(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    available = [row for row in rows if row.get("calibration_available")]
    bins = int(protocol.get("calibration_bins", 10))
    if bins < 2:
        raise ValueError("calibration_bins must be at least two.")
    ece = 0.0
    if available:
        for index in range(bins):
            lower, upper = index / bins, (index + 1) / bins
            bucket = [
                row
                for row in available
                if lower <= float(row["calibration_confidence"]) <= upper
                and (index == bins - 1 or float(row["calibration_confidence"]) < upper)
            ]
            if bucket:
                ece += len(bucket) / len(available) * abs(
                    (_mean(bucket, "calibration_confidence") or 0.0)
                    - (_mean(bucket, "calibration_correct") or 0.0)
                )
    thresholds = [float(value) for value in protocol.get("selective_thresholds", (0.5, 0.7, 0.9))]
    selective: dict[str, Any] = {}
    for threshold in thresholds:
        selected = [row for row in available if float(row["calibration_confidence"]) >= threshold]
        selective[f"{threshold:g}"] = {
            "coverage": _percentage(len(selected) / len(available)) if available else None,
            "accuracy": _percentage(_mean(selected, "calibration_correct")),
            "selected": len(selected),
        }
    return {
        "total": len(rows),
        "annotated": len(available),
        "accuracy": _percentage(_mean(available, "calibration_correct")),
        "mean_confidence": _percentage(_mean(available, "calibration_confidence")),
        "expected_calibration_error": _percentage(ece) if available else None,
        "brier_score": _mean(available, "calibration_brier"),
        "negative_log_likelihood": _mean(available, "calibration_nll"),
        "selective_prediction": selective,
    }


def summarize_groups(
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    summarizer: Any,
    *,
    primary_metric: str,
) -> dict[str, Any]:
    """Apply a scorer-specific aggregate to declared subgroups and expose disparity."""

    result: dict[str, Any] = {"overall": summarizer(rows)}
    minimum = int(protocol.get("minimum_group_samples", 1))
    disparities: dict[str, Any] = {}
    for field in protocol.get("group_by", ()):
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row.get(field, "unknown")), []).append(row)
        summaries = {
            name: summarizer(values)
            for name, values in sorted(grouped.items())
        }
        result[f"by_{field}"] = summaries
        eligible = {
            name: summary[primary_metric]
            for name, summary in summaries.items()
            if len(grouped[name]) >= minimum and summary.get(primary_metric) is not None
        }
        if eligible:
            worst = min(eligible, key=eligible.get)
            best = max(eligible, key=eligible.get)
            disparities[field] = {
                "metric": primary_metric,
                "minimum_group_samples": minimum,
                "eligible_groups": len(eligible),
                "worst_group": worst,
                "worst_group_value": eligible[worst],
                "best_group": best,
                "best_group_value": eligible[best],
                "max_min_gap": round(float(eligible[best]) - float(eligible[worst]), 4),
            }
    if disparities:
        result["subgroup_disparity"] = disparities
    return result

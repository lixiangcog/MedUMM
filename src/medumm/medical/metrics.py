from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any


ABSTENTIONS = (
    "cannot determine",
    "insufficient evidence",
    "not enough information",
    "unable to assess",
    "无法判断",
    "信息不足",
)


def normalize_answer(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", text)
    text = re.sub(r"[^\w.%/+]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def token_f1(prediction: str, reference: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def _choice(value: str, choices: dict[str, str]) -> str:
    valid = set(choices)
    for pattern in (r"(?:answer|option|choice)\s*(?:is|:)?\s*\(?([A-Z])\)?", r"^\s*\(?([A-Z])\)?"):
        match = re.search(pattern, value, flags=re.IGNORECASE)
        if match and match.group(1).upper() in valid:
            return match.group(1).upper()
    normalized = normalize_answer(value)
    for letter, option in choices.items():
        if normalized == normalize_answer(option):
            return letter
    return ""


def evaluate_answer(
    prediction: str,
    references: list[str],
    choices: dict[str, str] | None = None,
) -> dict[str, Any]:
    options = choices or {}
    if options:
        parsed = _choice(prediction, options)
        accepted = {_choice(reference, options) for reference in references}
        exact = bool(parsed and parsed in accepted)
        f1 = float(exact)
    else:
        parsed = normalize_answer(prediction)
        exact = any(parsed == normalize_answer(reference) for reference in references)
        f1 = max(token_f1(prediction, reference) for reference in references)
    normalized = normalize_answer(prediction)
    return {
        "exact_match": float(exact),
        "token_f1": float(f1),
        "abstained": any(normalize_answer(term) in normalized for term in ABSTENTIONS),
        "parsed_answer": parsed,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    return {
        "total": count,
        "exact_match": round(100 * sum(row["exact_match"] for row in rows) / count, 2),
        "token_f1": round(100 * sum(row["token_f1"] for row in rows) / count, 2),
        "abstention_rate": round(100 * sum(row["abstained"] for row in rows) / count, 2),
    }


def summarize_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"overall": _aggregate(rows)}
    for field in ("modality", "category", "answer_type", "language"):
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row.get(field, "unknown")), []).append(row)
        summary[f"by_{field}"] = {
            name: _aggregate(values) for name, values in sorted(grouped.items())
        }
    return summary

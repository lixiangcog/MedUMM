from __future__ import annotations

import re
import random
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
    # Medical VLMs often answer a closed question and then explain it. Accept
    # one unambiguous leading option phrase ("Yes, ..."), but never search the
    # whole rationale for a coincidental option token.
    leading = normalized.split()
    matches = [
        letter
        for letter, option in choices.items()
        if leading[: len(normalize_answer(option).split())]
        == normalize_answer(option).split()
    ]
    if len(matches) == 1:
        return matches[0]
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
        "closed_accuracy": round(
            100
            * sum(row["exact_match"] for row in rows if row.get("choices"))
            / sum(bool(row.get("choices")) for row in rows),
            2,
        )
        if any(row.get("choices") for row in rows)
        else None,
    }


def _bootstrap_interval(
    values: list[float],
    *,
    samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float | int] | None:
    if not values or samples <= 0:
        return None
    generator = random.Random(seed)
    means = sorted(
        sum(generator.choice(values) for _ in values) / len(values)
        for _ in range(samples)
    )
    tail = (1 - confidence_level) / 2
    lower = means[max(0, int(tail * samples))]
    upper = means[min(samples - 1, int((1 - tail) * samples) - 1)]
    return {
        "confidence_level": confidence_level,
        "lower": round(100 * lower, 2),
        "upper": round(100 * upper, 2),
        "bootstrap_samples": samples,
    }


def summarize_scores(
    rows: list[dict[str, Any]],
    *,
    group_by: tuple[str, ...] = ("modality", "category", "answer_type", "language"),
    bootstrap_samples: int = 0,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"overall": _aggregate(rows)}
    uncertainty: dict[str, Any] = {}
    for offset, metric in enumerate(("exact_match", "token_f1", "abstained")):
        interval = _bootstrap_interval(
            [float(row[metric]) for row in rows],
            samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed + offset,
        )
        if interval is not None:
            uncertainty["abstention_rate" if metric == "abstained" else metric] = interval
    if uncertainty:
        summary["uncertainty"] = uncertainty
    for field in group_by:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row.get(field, "unknown")), []).append(row)
        summary[f"by_{field}"] = {
            name: _aggregate(values) for name, values in sorted(grouped.items())
        }
    return summary

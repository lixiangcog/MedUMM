#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path
from typing import Any


DATASET_ID = "TsinghuaC3I/UltraMedical-Preference"
DATASET_REVISION = "761eb7935310ba662a96d93c5af342e5269d5759"
DATASET_LICENSE = "MIT"
RAW_TEST_SHA256 = "38f21a20407a401d55c1a0939f436ac1c2d5216bec31c126a55d7ed0f2c9d251"
DEFAULT_URL = (
    f"https://huggingface.co/datasets/{DATASET_ID}/resolve/"
    f"{DATASET_REVISION}/data/test.json"
)
DEFAULT_SOURCE_PREFIXES = ("MedMCQA", "MedQA", "PubMedQA", "TextBookQA")

DIRECT_IDENTIFIER_PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "phone": re.compile(r"(?<!\d)(?:\+?\d[\d ()-]{7,}\d)(?!\d)"),
    "url": re.compile(r"https?://\S+", re.I),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _download(path: Path, url: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as writer:
        while chunk := response.read(1024 * 1024):
            writer.write(chunk)
    temporary.replace(path)


def _assistant(value: Any) -> str:
    if not isinstance(value, list):
        return str(value or "").strip()
    responses = [
        str(message.get("content", "")).strip()
        for message in value
        if isinstance(message, dict)
        and str(message.get("role", "")).casefold() == "assistant"
        and str(message.get("content", "")).strip()
    ]
    return responses[-1] if responses else ""


def _direct_identifiers(text: str) -> list[str]:
    return sorted(
        name for name, pattern in DIRECT_IDENTIFIER_PATTERNS.items() if pattern.search(text)
    )


def prepare(arguments: argparse.Namespace) -> dict[str, Any]:
    raw_path = arguments.raw_path
    if not raw_path.is_file():
        if not arguments.download:
            raise FileNotFoundError(
                f"Raw UltraMedical preferences not found: {raw_path}; pass --download."
            )
        _download(raw_path, arguments.url)
    digest = _sha256(raw_path)
    if digest != arguments.expected_sha256:
        raise ValueError(
            f"Raw UltraMedical preference SHA-256 mismatch: {digest} != "
            f"{arguments.expected_sha256}"
        )
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("UltraMedical preference test file must contain a JSON list.")

    allowed = tuple(arguments.source_prefix or DEFAULT_SOURCE_PREFIXES)
    candidates: list[tuple[int, str, dict[str, Any], str, str, str]] = []
    rejected_for_identifiers = 0
    rejected_for_structure = 0
    for record in raw:
        if not isinstance(record, dict):
            rejected_for_structure += 1
            continue
        prompt_id = str(record.get("prompt_id", "")).strip()
        source = prompt_id.split(",", 1)[0]
        if source not in allowed:
            continue
        prompt = str(record.get("prompt", "")).strip()
        chosen = _assistant(record.get("chosen"))
        rejected = _assistant(record.get("rejected"))
        if not prompt_id or not prompt or not chosen or not rejected or chosen == rejected:
            rejected_for_structure += 1
            continue
        direct_identifiers = _direct_identifiers("\n".join((prompt, chosen, rejected)))
        if direct_identifiers:
            rejected_for_identifiers += 1
            continue
        candidates.append(
            (len(prompt) + len(chosen) + len(rejected), prompt_id, record, prompt, chosen, rejected)
        )
    candidates.sort(key=lambda value: (value[0], value[1]))
    selected = candidates[: arguments.samples]
    if len(selected) != arguments.samples:
        raise ValueError(
            f"Requested {arguments.samples} samples but only {len(selected)} eligible rows exist."
        )

    output_directory = arguments.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    rows = []
    source_distribution: dict[str, int] = {}
    label_distribution: dict[str, int] = {}
    for _, prompt_id, record, prompt, chosen, rejected in selected:
        metadata = record.get("metadata", {})
        metadata = metadata if isinstance(metadata, dict) else {}
        source = prompt_id.split(",", 1)[0]
        label_type = str(record.get("label_type", "unknown"))
        source_distribution[source] = source_distribution.get(source, 0) + 1
        label_distribution[label_type] = label_distribution.get(label_type, 0) + 1
        chosen_judgment = metadata.get("chosen", {})
        rejected_judgment = metadata.get("rejected", {})
        chosen_judgment = chosen_judgment if isinstance(chosen_judgment, dict) else {}
        rejected_judgment = (
            rejected_judgment if isinstance(rejected_judgment, dict) else {}
        )
        rationale = str(record.get("feedback", "")).strip()
        if not rationale:
            rationale = " ".join(
                part
                for part in (
                    str(chosen_judgment.get("evaluation", "")).strip(),
                    str(rejected_judgment.get("evaluation", "")).strip(),
                )
                if part
            )
        rows.append(
            {
                "id": f"ultramedical-{prompt_id.replace(',', '-').replace('/', '-')}",
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "specialty": "biomedicine",
                "label_source": "ai_judge",
                "clinical_relevance": 1.0,
                "preference_rationale": rationale or None,
                "preference_provenance": {
                    "kind": "model_judgment",
                    "expert_validated": False,
                    "upstream_prompt_id": prompt_id,
                    "upstream_label_type": label_type,
                    "chosen": chosen_judgment,
                    "rejected": rejected_judgment,
                },
                "metadata": {
                    "source_dataset": source,
                    "golden_answer": metadata.get("golden_answer"),
                },
            }
        )
    manifest_path = output_directory / "preferences.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    provenance = {
        "schema_version": "1.0",
        "dataset": DATASET_ID,
        "resolved_revision": DATASET_REVISION,
        "license": DATASET_LICENSE,
        "split": "test",
        "raw_file": raw_path.name,
        "raw_sha256": digest,
        "raw_sample_count": len(raw),
        "sample_count": len(rows),
        "manifest_sha256": _sha256(manifest_path),
        "selection": {
            "method": "shortest_eligible_then_prompt_id",
            "source_prefixes": list(allowed),
            "source_distribution": dict(sorted(source_distribution.items())),
            "label_type_distribution": dict(sorted(label_distribution.items())),
            "direct_identifier_patterns": sorted(DIRECT_IDENTIFIER_PATTERNS),
            "rejected_for_direct_identifiers": rejected_for_identifiers,
            "rejected_for_structure": rejected_for_structure,
        },
        "deidentified": True,
        "deidentification_basis": (
            "Exam/reference QA source allowlist plus automated direct-identifier pattern scan; "
            "not expert privacy review."
        ),
        "preference_annotation": (
            "Upstream model judgments and label types; not declared clinician preference."
        ),
        "clinical_use": False,
    }
    provenance_path = output_directory / "provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a pinned, privacy-constrained UltraMedical preference slice"
    )
    parser.add_argument("--raw-path", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--expected-sha256", default=RAW_TEST_SHA256)
    parser.add_argument(
        "--source-prefix",
        action="append",
        default=None,
        help="Eligible upstream source prefix; repeat to replace the default allowlist.",
    )
    return parser


def main(arguments: list[str] | None = None) -> int:
    values = build_parser().parse_args(arguments)
    print(json.dumps(prepare(values), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import hashlib
import json

from scripts.prepare_ultramedical_preferences import prepare


def _messages(prompt, response):
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


def test_prepare_ultramedical_preferences_is_pinned_and_transparent(tmp_path):
    raw = []
    for index, source in enumerate(("MedMCQA", "MedQA", "ChatDoctor")):
        prompt = f"Question {index}?"
        raw.append(
            {
                "prompt_id": f"{source},{index}",
                "prompt": prompt,
                "chosen": _messages(prompt, "Preferred answer."),
                "rejected": _messages(prompt, "Rejected answer."),
                "label_type": "hard",
                "metadata": {
                    "chosen": {"model": "a", "score": 5, "evaluation": "better"},
                    "rejected": {"model": "b", "score": 2, "evaluation": "worse"},
                },
            }
        )
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    output = tmp_path / "prepared"
    result = prepare(
        argparse.Namespace(
            raw_path=raw_path,
            output_directory=output,
            samples=2,
            download=False,
            url="unused",
            expected_sha256=digest,
            source_prefix=["MedMCQA", "MedQA"],
        )
    )
    rows = [json.loads(line) for line in (output / "preferences.jsonl").read_text().splitlines()]
    assert result["sample_count"] == 2
    assert result["selection"]["source_distribution"] == {"MedMCQA": 1, "MedQA": 1}
    assert all(row["label_source"] == "ai_judge" for row in rows)
    assert all(not row["preference_provenance"]["expert_validated"] for row in rows)
    assert result["deidentified"] is True


def test_prepare_rejects_raw_hash_mismatch(tmp_path):
    raw_path = tmp_path / "raw.json"
    raw_path.write_text("[]", encoding="utf-8")
    try:
        prepare(
            argparse.Namespace(
                raw_path=raw_path,
                output_directory=tmp_path / "out",
                samples=1,
                download=False,
                url="unused",
                expected_sha256="0" * 64,
                source_prefix=["MedQA"],
            )
        )
    except ValueError as error:
        assert "SHA-256 mismatch" in str(error)
    else:
        raise AssertionError("Expected hash mismatch failure")

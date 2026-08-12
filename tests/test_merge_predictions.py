import json

import pytest

from medumm.evaluation import merge_prediction_shards


def _write(path, rank, ids, fingerprint="stable"):
    path.write_text(
        "".join(
            json.dumps({"id": sample_id, "fingerprint": fingerprint, "prediction": rank}) + "\n"
            for sample_id in ids
        )
    )


def test_strict_merge_orders_and_manifests_prediction_shards(tmp_path):
    rank0 = tmp_path / "predictions.rank-00000-of-00002.jsonl"
    rank1 = tmp_path / "predictions.rank-00001-of-00002.jsonl"
    _write(rank0, 0, ["case-03", "case-01"])
    _write(rank1, 1, ["case-04", "case-02"])
    result = merge_prediction_shards(
        [rank1, rank0], tmp_path / "predictions.jsonl", expected_count=4
    )
    rows = [json.loads(line) for line in (tmp_path / "predictions.jsonl").read_text().splitlines()]
    assert [row["id"] for row in rows] == ["case-01", "case-02", "case-03", "case-04"]
    assert result["shard_count"] == 2
    assert (tmp_path / "merge_manifest.json").is_file()


def test_strict_merge_rejects_missing_or_mixed_shards(tmp_path):
    rank0 = tmp_path / "predictions.rank-00000-of-00002.jsonl"
    _write(rank0, 0, ["case-01"])
    with pytest.raises(ValueError, match="Missing"):
        merge_prediction_shards([rank0], tmp_path / "predictions.jsonl")
    rank1 = tmp_path / "predictions.rank-00001-of-00002.jsonl"
    _write(rank1, 1, ["case-02"], fingerprint="different")
    with pytest.raises(ValueError, match="fingerprint"):
        merge_prediction_shards([rank0, rank1], tmp_path / "predictions.jsonl")

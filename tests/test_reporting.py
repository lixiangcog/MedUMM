import json

from medumm.reporting import build_leaderboard


def test_leaderboard_aggregates_score_reports(tmp_path):
    report = tmp_path / "score.json"
    report.write_text(json.dumps({
        "benchmark": "medical_vqa",
        "dataset_size": 2,
        "metadata": {"model": "reference"},
        "metrics": {"overall": {"total": 2, "exact_match": 100.0}},
    }))
    paths = build_leaderboard([report], tmp_path / "leaderboard")
    rows = json.loads((tmp_path / "leaderboard/leaderboard.json").read_text())
    assert rows[0]["overall.exact_match"] == 100.0
    assert paths["csv_path"].endswith("leaderboard.csv")

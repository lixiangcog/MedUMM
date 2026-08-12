from medumm.medical.metrics import evaluate_answer, normalize_answer, token_f1


def test_open_and_choice_metrics():
    assert normalize_answer("No—finding.") == "no finding"
    assert token_f1("central bright region", "bright central region") == 1.0
    result = evaluate_answer("Option B", ["B"], {"A": "yes", "B": "no"})
    assert result["exact_match"] == 1.0


def test_abstention_is_reported():
    result = evaluate_answer("Insufficient evidence", ["yes"])
    assert result["abstained"] is True

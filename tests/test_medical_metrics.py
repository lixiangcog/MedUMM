from medumm.medical.metrics import evaluate_answer, normalize_answer, token_f1


def test_open_and_choice_metrics():
    assert normalize_answer("No—finding.") == "no finding"
    assert token_f1("central bright region", "bright central region") == 1.0
    result = evaluate_answer("Option B", ["B"], {"A": "yes", "B": "no"})
    assert result["exact_match"] == 1.0
    choices = {"A": "yes", "B": "no"}
    assert evaluate_answer(
        "Yes, there is a visible opacity.", ["yes"], choices
    )["exact_match"] == 1.0
    assert evaluate_answer(
        "The report says no but the answer is yes.", ["yes"], choices
    )["exact_match"] == 0.0


def test_abstention_is_reported():
    result = evaluate_answer("Insufficient evidence", ["yes"])
    assert result["abstained"] is True

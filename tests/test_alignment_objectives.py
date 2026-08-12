import pytest

torch = pytest.importorskip("torch")

from medumm.post_training.objectives import alignment_loss, sequence_log_probabilities


def _values():
    chosen = torch.tensor([-2.0, -3.0], requires_grad=True)
    rejected = torch.tensor([-4.0, -3.5], requires_grad=True)
    chosen_mean = chosen / 2
    rejected_mean = rejected / 2
    nll = -chosen_mean
    return chosen, rejected, chosen_mean, rejected_mean, nll


def test_sequence_log_probabilities_masks_prompt_tokens():
    logits = torch.zeros((1, 4, 3))
    labels = torch.tensor([[-100, -100, 1, 2]])
    sums, means, nll = sequence_log_probabilities(logits, labels)
    expected = -2 * torch.log(torch.tensor(3.0))
    assert sums.item() == pytest.approx(expected.item())
    assert means.item() == pytest.approx((expected / 2).item())
    assert nll.item() == pytest.approx((-expected / 2).item())


@pytest.mark.parametrize("objective", ["dpo", "simpo", "orpo", "clinical_dpo"])
def test_preference_objectives_are_finite_and_backpropagate(objective):
    chosen, rejected, chosen_mean, rejected_mean, nll = _values()
    reference_chosen = chosen.detach().clone() if objective in {"dpo", "clinical_dpo"} else None
    reference_rejected = rejected.detach().clone() if objective in {"dpo", "clinical_dpo"} else None
    loss, diagnostics = alignment_loss(
        objective,
        chosen_logps=chosen,
        rejected_logps=rejected,
        chosen_mean_logps=chosen_mean,
        rejected_mean_logps=rejected_mean,
        chosen_nll=nll,
        reference_chosen_logps=reference_chosen,
        reference_rejected_logps=reference_rejected,
        weights=torch.tensor([1.0, 2.0]),
        beta=0.1,
        margin=0.2,
    )
    assert torch.isfinite(loss)
    assert set(diagnostics) == {
        "loss",
        "chosen_nll",
        "reward_margin",
        "preference_accuracy",
        "chosen_logp",
        "rejected_logp",
    }
    loss.backward()
    assert chosen.grad is not None and rejected.grad is not None


def test_clinical_dpo_weights_change_the_loss():
    chosen, rejected, chosen_mean, rejected_mean, nll = _values()
    common = dict(
        chosen_logps=chosen,
        rejected_logps=rejected,
        chosen_mean_logps=chosen_mean,
        rejected_mean_logps=rejected_mean,
        chosen_nll=nll,
        reference_chosen_logps=torch.tensor([-2.4, -3.0]),
        reference_rejected_logps=torch.tensor([-3.9, -3.5]),
    )
    uniform, _ = alignment_loss("clinical_dpo", **common, weights=torch.ones(2))
    weighted, _ = alignment_loss(
        "clinical_dpo", **common, weights=torch.tensor([4.0, 1.0])
    )
    assert uniform.item() != pytest.approx(weighted.item())


def test_dpo_rejects_missing_reference_scores():
    chosen, rejected, chosen_mean, rejected_mean, nll = _values()
    with pytest.raises(ValueError, match="reference-policy"):
        alignment_loss(
            "dpo",
            chosen_logps=chosen,
            rejected_logps=rejected,
            chosen_mean_logps=chosen_mean,
            rejected_mean_logps=rejected_mean,
            chosen_nll=nll,
        )


def test_sft_uses_only_chosen_completion_nll():
    chosen, rejected, chosen_mean, rejected_mean, nll = _values()
    loss, diagnostics = alignment_loss(
        "sft",
        chosen_logps=chosen,
        rejected_logps=None,
        chosen_mean_logps=chosen_mean,
        rejected_mean_logps=None,
        chosen_nll=nll,
    )
    assert loss.item() == pytest.approx(nll.mean().item())
    assert diagnostics["preference_accuracy"].item() == 1.0

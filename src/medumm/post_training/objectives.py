from __future__ import annotations

from typing import Any

from medumm.medical.alignment import AlignmentObjective


def _torch():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Medical alignment objectives require PyTorch.") from error
    return torch


def sequence_log_probabilities(
    logits: Any,
    labels: Any,
) -> tuple[Any, Any, Any]:
    """Return summed/mean completion log-probability and completion NLL.

    Labels use ``-100`` for prompt and padding tokens. The causal shift is
    applied here once so all alignment methods share exactly the same token
    accounting.
    """

    torch = _torch()
    shifted_logits = logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:].clone()
    mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    token_logps = torch.log_softmax(shifted_logits, dim=-1).gather(
        -1, safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    token_logps = token_logps * mask
    counts = mask.sum(-1).clamp_min(1)
    sums = token_logps.sum(-1)
    means = sums / counts
    nll = -means
    return sums, means, nll


def _log_one_minus_exp(log_probability: Any) -> Any:
    """Stable log(1-exp(x)) for x <= 0."""

    torch = _torch()
    log_two = -0.6931471805599453
    return torch.where(
        log_probability < log_two,
        torch.log1p(-torch.exp(log_probability)),
        torch.log(-torch.expm1(log_probability)),
    )


def alignment_loss(
    objective: AlignmentObjective | str,
    *,
    chosen_logps: Any,
    rejected_logps: Any | None,
    chosen_mean_logps: Any,
    rejected_mean_logps: Any | None,
    chosen_nll: Any,
    reference_chosen_logps: Any | None = None,
    reference_rejected_logps: Any | None = None,
    beta: float = 0.1,
    margin: float = 0.0,
    weights: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Compute SFT, DPO, SimPO, ORPO, or clinical-relevance weighted DPO."""

    torch = _torch()
    method = AlignmentObjective(objective)
    if beta <= 0:
        raise ValueError("Alignment beta must be positive.")
    if method.requires_rejected and (
        rejected_logps is None or rejected_mean_logps is None
    ):
        raise ValueError(f"{method.value} requires rejected sequence scores.")
    sample_weights = (
        weights.to(chosen_logps.device, dtype=chosen_logps.dtype)
        if weights is not None
        else torch.ones_like(chosen_logps)
    )
    if torch.any(sample_weights <= 0):
        raise ValueError("Clinical relevance weights must be positive.")

    if method is AlignmentObjective.SFT:
        losses = chosen_nll
        reward_margin = torch.zeros_like(chosen_logps)
    elif method in {AlignmentObjective.DPO, AlignmentObjective.CLINICAL_DPO}:
        if reference_chosen_logps is None or reference_rejected_logps is None:
            raise ValueError(f"{method.value} requires frozen reference-policy scores.")
        policy_ratio = chosen_logps - rejected_logps
        reference_ratio = reference_chosen_logps - reference_rejected_logps
        reward_margin = beta * (policy_ratio - reference_ratio)
        losses = -torch.nn.functional.logsigmoid(reward_margin)
        if method is AlignmentObjective.CLINICAL_DPO:
            losses = losses * sample_weights
    elif method is AlignmentObjective.SIMPO:
        reward_margin = beta * (chosen_mean_logps - rejected_mean_logps) - margin
        losses = -torch.nn.functional.logsigmoid(reward_margin)
    else:
        chosen_log_odds = chosen_mean_logps - _log_one_minus_exp(chosen_mean_logps)
        rejected_log_odds = rejected_mean_logps - _log_one_minus_exp(
            rejected_mean_logps
        )
        reward_margin = chosen_log_odds - rejected_log_odds
        preference_loss = -torch.nn.functional.logsigmoid(reward_margin)
        losses = chosen_nll + beta * preference_loss

    loss = losses.mean()
    diagnostics = {
        "loss": loss.detach(),
        "chosen_nll": chosen_nll.mean().detach(),
        "reward_margin": reward_margin.mean().detach(),
        "preference_accuracy": (reward_margin > 0).float().mean().detach()
        if method.requires_rejected
        else torch.ones((), device=chosen_logps.device),
        "chosen_logp": chosen_logps.mean().detach(),
        "rejected_logp": rejected_logps.mean().detach()
        if rejected_logps is not None
        else torch.zeros((), device=chosen_logps.device),
    }
    return loss, diagnostics

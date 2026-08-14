from __future__ import annotations

from typing import Any


class ExponentialMovingAverage:
    """Per-rank EMA for trainable parameters, including FSDP local shards."""

    def __init__(self, model: Any, *, decay: float, device: str = "model") -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be between zero and one.")
        self.decay = float(decay)
        self.device = device
        self.num_updates = 0
        self.shadow: dict[str, Any] = {}
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                target = parameter.detach().float().clone()
                if device == "cpu":
                    target = target.cpu()
                self.shadow[name] = target

    def update(self, model: Any) -> None:
        self.num_updates += 1
        parameters = dict(model.named_parameters())
        for name, shadow in self.shadow.items():
            parameter = parameters.get(name)
            if parameter is None:
                raise RuntimeError(f"EMA parameter disappeared: {name}")
            value = parameter.detach().float()
            if shadow.device != value.device:
                value = value.to(shadow.device)
            shadow.lerp_(value, 1.0 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": self.shadow,
        }

    def load_state_dict(self, value: dict[str, Any]) -> None:
        loaded = value.get("shadow", {})
        if set(loaded) != set(self.shadow):
            raise RuntimeError("EMA checkpoint parameter names do not match the model.")
        for name, tensor in loaded.items():
            if tuple(tensor.shape) != tuple(self.shadow[name].shape):
                raise RuntimeError(f"EMA checkpoint shape mismatch for {name}.")
            self.shadow[name].copy_(tensor.to(self.shadow[name].device))
        self.decay = float(value.get("decay", self.decay))
        self.num_updates = int(value.get("num_updates", 0))

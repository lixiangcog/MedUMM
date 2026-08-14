from __future__ import annotations

from contextlib import nullcontext
from typing import Any


class PrecisionManager:
    """Version-compatible autocast and gradient-scaling facade."""

    def __init__(self, torch: Any, *, precision: str, device_type: str) -> None:
        self.torch = torch
        self.precision = precision
        self.device_type = device_type
        self.autocast_enabled = precision in {"fp16", "bf16"}
        self.dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        scaler_enabled = precision == "fp16" and device_type == "cuda"
        try:
            self.scaler = torch.amp.GradScaler(device_type, enabled=scaler_enabled)
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    def autocast(self):
        if not self.autocast_enabled:
            return nullcontext()
        return self.torch.autocast(
            device_type=self.device_type,
            dtype=self.dtype,
            enabled=True,
        )

    def backward(self, loss: Any) -> None:
        self.scaler.scale(loss).backward()

    def unscale_(self, optimizer: Any) -> None:
        self.scaler.unscale_(optimizer)

    def step(self, optimizer: Any) -> None:
        self.scaler.step(optimizer)
        self.scaler.update()

    def state_dict(self) -> dict[str, Any]:
        return self.scaler.state_dict()

    def load_state_dict(self, value: dict[str, Any]) -> None:
        self.scaler.load_state_dict(value)

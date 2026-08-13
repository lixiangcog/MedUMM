from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Run a tiny optimization used only to accept the external-route contract."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--checkpoint_path", "--checkpoint-path", required=True)
    parser.add_argument("--previous_checkpoint", "--previous-checkpoint")
    parser.add_argument("--steps", type=int, default=8)
    arguments, _ = parser.parse_known_args(argv)
    if arguments.steps < 1:
        raise ValueError("steps must be positive")
    if arguments.previous_checkpoint and not Path(arguments.previous_checkpoint).is_file():
        raise FileNotFoundError(arguments.previous_checkpoint)

    # A real differentiable optimization with no heavyweight dependency. It is
    # deliberately method-agnostic: this worker validates process launch,
    # stage dependencies, logs, and checkpoint collection, not paper fidelity.
    value = -1.0
    target = 1.0 + (sum(map(ord, arguments.method + arguments.stage)) % 7) / 10
    learning_rate = 0.2
    history = []
    for step in range(1, arguments.steps + 1):
        loss = (value - target) ** 2
        gradient = 2 * (value - target)
        value -= learning_rate * gradient
        history.append({"step": step, "loss": loss})
        print(f"step={step} loss={loss:.8f}", flush=True)

    checkpoint = Path(arguments.checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "kind": "post_training_contract_smoke",
                "method": arguments.method,
                "stage": arguments.stage,
                "optimized_value": value,
                "target": target,
                "previous_checkpoint": arguments.previous_checkpoint,
                "history": history,
                "paper_fidelity_claim": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

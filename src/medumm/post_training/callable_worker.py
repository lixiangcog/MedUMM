from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Callable
from typing import Any


def _resolve(target: str) -> Callable[[dict[str, Any]], Any]:
    if ":" not in target:
        raise ValueError("Callable target must use module:function syntax.")
    module_name, function_name = target.rsplit(":", 1)
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"Target is not callable: {target}")
    return function


def main(argv: list[str] | None = None) -> int:
    """Invoke a pinned external runtime function in an isolated child process."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--config-json", required=True)
    arguments = parser.parse_args(argv)
    config = json.loads(arguments.config_json)
    if not isinstance(config, dict):
        raise ValueError("Callable runtime config must decode to a mapping.")
    _resolve(arguments.target)(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

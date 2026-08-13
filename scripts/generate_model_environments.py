from __future__ import annotations

import argparse
import json
from pathlib import Path

from medumm.environments import EnvironmentCatalog
from medumm.environments.render import write_generated_artifacts


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Render per-model environment artifacts")
    parser.add_argument(
        "--catalog",
        default=ROOT / "src/medumm/environments/catalog/models.yaml",
        type=Path,
    )
    parser.add_argument("--output-root", default=ROOT / "environments/models", type=Path)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    result = write_generated_artifacts(
        EnvironmentCatalog.load(arguments.catalog),
        arguments.output_root,
        check=arguments.check,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from medumm.environments import ENVIRONMENT_CATALOG


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve every model contract for Linux")
    parser.add_argument("--output-root", type=Path, default=ROOT / "environments/locks")
    parser.add_argument("--uv", default=shutil.which("uv") or "uv")
    parser.add_argument("--model", action="append", choices=ENVIRONMENT_CATALOG.names())
    arguments = parser.parse_args()
    names = arguments.model or ENVIRONMENT_CATALOG.names()
    arguments.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for name in names:
        spec = ENVIRONMENT_CATALOG.get(name)
        output = arguments.output_root / f"{name}.txt"
        command = [
            arguments.uv,
            "pip",
            "compile",
            str(ROOT / "environments/models" / name / "requirements.txt"),
            "--python-version",
            spec.python,
            "--python-platform",
            "x86_64-manylinux_2_28",
            "--no-emit-index-url",
            "--no-header",
            "-o",
            str(output),
        ]
        if spec.torch_index:
            command.extend(("--extra-index-url", spec.torch_index))
        process = subprocess.run(command, capture_output=True, check=False, text=True)
        results.append(
            {
                "model": name,
                "status": "resolved" if process.returncode == 0 else "failed",
                "returncode": process.returncode,
                "lock": str(output.relative_to(ROOT)) if process.returncode == 0 else None,
                "error": process.stderr[-2000:] if process.returncode else None,
            }
        )
    report = {
        "schema_version": "1.0",
        "models": len(results),
        "resolved": sum(item["status"] == "resolved" for item in results),
        "failed": sum(item["status"] == "failed" for item in results),
        "results": results,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

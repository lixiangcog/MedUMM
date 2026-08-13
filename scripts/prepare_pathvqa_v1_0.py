#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.prepare_medical_vqa_dataset import export_medical_vqa_dataset


PATHVQA_ID = "flaviagiammarino/path-vqa"
PATHVQA_REVISION = "1685832883334b5bb5beaf4e4b333fdeecaa4ad9"


def prepare(output_directory: Path, *, closed_samples: int, open_samples: int) -> dict:
    return export_medical_vqa_dataset(
        dataset="pathvqa",
        revision=PATHVQA_REVISION,
        split="test",
        output_directory=output_directory.expanduser().resolve(),
        closed_samples=closed_samples,
        open_samples=open_samples,
        language="en",
        streaming=True,
    )


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare the pinned MedUMM PathVQA v1.0 slice")
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--closed-samples", type=int, default=4)
    parser.add_argument("--open-samples", type=int, default=4)
    values = parser.parse_args(arguments)
    print(json.dumps(prepare(
        values.output_directory,
        closed_samples=values.closed_samples,
        open_samples=values.open_samples,
    ), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

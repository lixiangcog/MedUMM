from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _atomic_text(path: Path, content: str) -> Path:
    ensure_directory(path.parent)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as writer:
            writer.write(content)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return path


def write_json(path: str | Path, payload: Any) -> Path:
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    return _atomic_text(Path(path), text)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    return _atomic_text(Path(path), text)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as reader:
        for line_number, line in enumerate(reader, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}.")
            records.append(record)
    return records

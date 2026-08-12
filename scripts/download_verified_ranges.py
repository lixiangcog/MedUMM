#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        while chunk := reader.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download(arguments: argparse.Namespace) -> Path:
    output = arguments.output
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file() and output.stat().st_size == arguments.size:
        if _sha256(output) == arguments.sha256:
            return output
    part_root = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.parts-", dir=output.parent)
    )
    ranges = []
    for index, start in enumerate(range(0, arguments.size, arguments.chunk_size)):
        end = min(arguments.size - 1, start + arguments.chunk_size - 1)
        ranges.append((index, start, end))

    def fetch(value: tuple[int, int, int]) -> Path:
        index, start, end = value
        expected = end - start + 1
        path = part_root / f"part-{index:05d}"
        request = urllib.request.Request(
            arguments.url,
            headers={"Range": f"bytes={start}-{end}", "User-Agent": "MedUMM/0.7"},
        )
        with urllib.request.urlopen(request, timeout=arguments.timeout) as response:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {start}-{end}/"):
                raise RuntimeError(
                    f"Server did not honor byte range {start}-{end}: {content_range!r}"
                )
            with path.open("wb") as writer:
                while chunk := response.read(1024 * 1024):
                    writer.write(chunk)
        if path.stat().st_size != expected:
            raise RuntimeError(
                f"Range {start}-{end} has {path.stat().st_size} bytes, expected {expected}."
            )
        return path

    assembly = output.parent / f".{output.name}.{os.getpid()}.assembling"
    try:
        completed: dict[int, Path] = {}
        with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
            futures = {executor.submit(fetch, value): value[0] for value in ranges}
            for future in as_completed(futures):
                completed[futures[future]] = future.result()
                print(
                    f"[MedUMM] downloaded range {len(completed)}/{len(ranges)}",
                    flush=True,
                )
        with assembly.open("wb") as writer:
            for index in range(len(ranges)):
                with completed[index].open("rb") as reader:
                    while chunk := reader.read(1024 * 1024):
                        writer.write(chunk)
        if assembly.stat().st_size != arguments.size:
            raise RuntimeError("Assembled file size does not match the pinned asset size.")
        digest = _sha256(assembly)
        if digest != arguments.sha256:
            raise RuntimeError(f"Assembled SHA-256 mismatch: {digest} != {arguments.sha256}")
        os.replace(assembly, output)
    finally:
        assembly.unlink(missing_ok=True)
        shutil.rmtree(part_root, ignore_errors=True)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and verify a large HTTP asset by ranges")
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--chunk-size", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=300)
    return parser


if __name__ == "__main__":
    values = build_parser().parse_args()
    print(download(values))

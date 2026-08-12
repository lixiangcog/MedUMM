#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse


SOURCE_REVISION = "30697ca50b5c29a8e955c99330b259776aef27b9"
MODEL_REVISION = "91bb16c122001ddc9cf1fd36ce1dae09448943a2"
VISION_REVISION = "ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
DATASET_REVISION = "bcf91e7654fb9d51c8ab6a5b82cacf3fafd2fae9"
MODEL_ID = "microsoft/llava-med-v1.5-mistral-7b"
VISION_ID = "openai/clip-vit-large-patch14-336"
DATASET_ID = "flaviagiammarino/vqa-rad"
DATASET_TEST_FILE = "data/test-00000-of-00001-e5bc3d208bb4deeb.parquet"

MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
    "special_tokens_map.json",
    "tokenizer.model",
    "tokenizer_config.json",
)
VISION_FILES = ("config.json", "preprocessor_config.json", "pytorch_model.bin")
MODEL_FILE_METADATA = {
    "config.json": (
        1_407,
        "f6ae889c5488ef86895e78f641339062962dd6b434666019fa119ab09d2bd8b3",
    ),
    "generation_config.json": (
        111,
        "741acba7f5e235dac0e6865ecc212bbadb1ab1d6d853de7d759268cb62aaf2b4",
    ),
    "model.safetensors.index.json": (
        73_152,
        "d5ecec60dba218c6621cfa524d739de942b4552bcbb25efe63848c18a731b2f6",
    ),
    "model-00001-of-00004.safetensors": (
        4_943_162_336,
        "ef2190dc6c2a940e60f03f5fdb4dddb2320eb87801aeca5c40b0a28ce8aa420e",
    ),
    "model-00002-of-00004.safetensors": (
        4_999_819_336,
        "2b229607fecd98b8111320178e5bf3e2c527b05a942c85d65b5b507c76c1ed00",
    ),
    "model-00003-of-00004.safetensors": (
        4_927_408_360,
        "12b18ecdf8924d5fe28ada797fe6697fa60e62cba630759fbeb52975b261c4e2",
    ),
    "model-00004-of-00004.safetensors": (
        262_144_128,
        "1d2063fcd429d3f0f0a8a091b0522f0e02f2d85fe0e5b0eeb4ae168183a603bc",
    ),
    "special_tokens_map.json": (
        438,
        "719833ff26ac897a3ec8ed946028a135de2a351470af59b4008744ab1f0ee9b7",
    ),
    "tokenizer.model": (
        493_443,
        "dadfd56d766715c61d2ef780a525ab43b8e6da4de6865bda3d95fdef5e134055",
    ),
    "tokenizer_config.json": (
        1_463,
        "5b219f9212f7263269898c799cc9d9be2326e853bf1e497f1c412f3a274d0597",
    ),
}
VISION_FILE_METADATA = {
    "config.json": (
        4_757,
        "51b1c14aabcdf639c4a0370eeda1010b773bbe1df78319c7d0f5882c22ac0ac0",
    ),
    "preprocessor_config.json": (
        316,
        "d253881f65322dc546df59cf925a408e5538b8ecb5a1b496cdd36af9992686d4",
    ),
    "pytorch_model.bin": (
        1_711_974_081,
        "c6032c2e0caae3dc2d4fba35535fa6307dbb49df59c7e182b1bc4b3329b81801",
    ),
}
DATASET_FILE_METADATA = {
    DATASET_TEST_FILE: (
        10_312_735,
        "eb520bdab1116dd4f420120da19049d2315389fa126d031f65ec42e153264ea7",
    )
}


def _source_checkout(destination: Path) -> Path:
    target = destination / "LLaVA-Med"
    marker = target / ".medumm-source-revision"
    if (target / "llava").is_dir():
        if marker.is_file():
            if marker.read_text(encoding="utf-8").strip() != SOURCE_REVISION:
                raise ValueError(f"LLaVA-Med source revision marker is invalid: {target}")
            return target
        provenance_path = destination / "provenance.json"
        if provenance_path.is_file():
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            if (
                provenance.get("source_revision") == SOURCE_REVISION
                and Path(str(provenance.get("source_path", ""))).resolve()
                == target.resolve()
            ):
                marker.write_text(SOURCE_REVISION + "\n", encoding="utf-8")
                return target
    if target.exists():
        raise FileExistsError(
            f"Unverified LLaVA-Med source exists at {target}; move it aside and retry."
        )
    archive = destination / f"llava-med-{SOURCE_REVISION}.zip"
    url = f"https://codeload.github.com/microsoft/LLaVA-Med/zip/{SOURCE_REVISION}"
    command = [
        "curl",
        "--fail",
        "--location",
        "--http1.1",
        "--silent",
        "--show-error",
        "--retry",
        "20",
        "--connect-timeout",
        "30",
        "--output",
        str(archive),
    ]
    if os.environ.get("MEDUMM_DOWNLOAD_PROXY"):
        command.extend(["--proxy", os.environ["MEDUMM_DOWNLOAD_PROXY"]])
    subprocess.run([*command, url], check=True)
    with zipfile.ZipFile(archive) as reader:
        reader.extractall(destination)
    extracted = destination / f"LLaVA-Med-{SOURCE_REVISION}"
    if not extracted.is_dir():
        raise FileNotFoundError(f"Expected extracted LLaVA-Med source: {extracted}")
    extracted.rename(target)
    marker.write_text(SOURCE_REVISION + "\n", encoding="utf-8")
    archive.unlink(missing_ok=True)
    return target


def _retry(operation: Callable[[], None], *, description: str, attempts: int = 8) -> None:
    for attempt in range(1, attempts + 1):
        try:
            operation()
            return
        except Exception:
            if attempt == attempts:
                raise
            delay = min(5 * attempt, 30)
            print(
                f"[MedUMM] {description} attempt {attempt}/{attempts} failed; "
                f"retrying in {delay}s",
                flush=True,
            )
            time.sleep(delay)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_file(
    *,
    repo_id: str,
    revision: str,
    filename: str,
    destination: Path,
    expected_size: int,
    expected_sha256: str | None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == expected_size:
        if expected_sha256 is None or _sha256(destination) == expected_sha256:
            print(f"[MedUMM] verified cached {filename}", flush=True)
            return
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    url = f"{endpoint}/{repo_id}/resolve/{revision}/{filename}"
    partial = destination.with_name(destination.name + ".partial")
    if partial.is_file() and partial.stat().st_size == expected_size:
        if expected_sha256 is None or _sha256(partial) == expected_sha256:
            partial.replace(destination)
            print(f"[MedUMM] verified resumed {filename}", flush=True)
            return
        partial.unlink()
    elif partial.is_file() and partial.stat().st_size > expected_size:
        partial.unlink()
    command = [
        "curl",
        "--fail",
        "--location",
        "--http1.1",
        "--silent",
        "--show-error",
        "--retry",
        "20",
        "--connect-timeout",
        "30",
        "--continue-at",
        "-",
        "--output",
        str(partial),
    ]
    resolve_ip = os.environ.get("HF_ENDPOINT_RESOLVE_IP")
    download_proxy = os.environ.get("MEDUMM_DOWNLOAD_PROXY")
    if download_proxy:
        command.extend(["--proxy", download_proxy])
    parsed = urlparse(endpoint)
    if resolve_ip and parsed.hostname and parsed.scheme == "https":
        command.extend(["--resolve", f"{parsed.hostname}:443:{resolve_ip}"])
    print(f"[MedUMM] downloading {repo_id}/{filename}", flush=True)
    subprocess.run([*command, url], check=True)
    actual_size = partial.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"Downloaded size mismatch for {filename}: expected {expected_size}, "
            f"found {actual_size}."
        )
    if expected_sha256 and _sha256(partial) != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {filename}.")
    partial.replace(destination)
    print(f"[MedUMM] verified {filename} ({actual_size} bytes)", flush=True)


def _download_snapshot(
    *,
    repo_id: str,
    revision: str,
    destination: Path,
    metadata: dict[str, tuple[int, str | None]],
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=min(4, len(metadata))) as executor:
        futures = [
            executor.submit(
                _download_file,
                repo_id=repo_id,
                revision=revision,
                filename=filename,
                destination=destination / filename,
                expected_size=expected_size,
                expected_sha256=expected_sha256,
            )
            for filename, (expected_size, expected_sha256) in metadata.items()
        ]
        for future in futures:
            future.result()


def _require_files(directory: Path, filenames: tuple[str, ...]) -> None:
    missing = [name for name in filenames if not (directory / name).is_file()]
    empty = [name for name in filenames if (directory / name).is_file() and not (directory / name).stat().st_size]
    if missing or empty:
        raise FileNotFoundError(
            f"Incomplete model snapshot at {directory}; missing={missing}, empty={empty}."
        )


def _validate_model_weights(model: Path) -> None:
    index = json.loads((model / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = sorted(set(index["weight_map"].values()))
    if tuple(shards) != MODEL_FILES[3:7]:
        raise ValueError(f"Unexpected LLaVA-Med weight shards: {shards}")
    expected = int(index.get("metadata", {}).get("total_size", 0))
    actual = sum((model / shard).stat().st_size for shard in shards)
    # The index records tensor payload bytes; safetensors files also contain headers.
    if expected and not (expected <= actual <= expected + 8 * 1024 * 1024):
        raise ValueError(
            f"LLaVA-Med weight payload mismatch: index reports {expected} bytes, "
            f"shard files contain {actual}."
        )


def prepare_assets(destination: Path) -> dict[str, object]:
    destination.mkdir(parents=True, exist_ok=True)
    source = _source_checkout(destination)
    model = destination / "llava-med-v1.5-mistral-7b"
    vision = destination / "clip-vit-large-patch14-336"
    dataset = destination / "vqa-rad"
    pristine_config_path = model / "config.upstream.json"
    config_path = model / "config.json"
    if pristine_config_path.is_file():
        config_path.write_bytes(pristine_config_path.read_bytes())
    def download_model() -> None:
        _download_snapshot(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            destination=model,
            metadata=MODEL_FILE_METADATA,
        )
        _require_files(model, MODEL_FILES)
        _validate_model_weights(model)

    def download_vision() -> None:
        _download_snapshot(
            repo_id=VISION_ID,
            revision=VISION_REVISION,
            destination=vision,
            metadata=VISION_FILE_METADATA,
        )
        _require_files(vision, VISION_FILES)
        if (vision / "pytorch_model.bin").stat().st_size < 100_000_000:
            raise ValueError("CLIP vision weights are unexpectedly small.")

    def download_dataset() -> None:
        _download_snapshot(
            repo_id=f"datasets/{DATASET_ID}",
            revision=DATASET_REVISION,
            destination=dataset,
            metadata=DATASET_FILE_METADATA,
        )
        _require_files(dataset, (DATASET_TEST_FILE,))

    _retry(download_model, description="LLaVA-Med snapshot")
    _retry(download_vision, description="CLIP vision snapshot")
    _retry(download_dataset, description="VQA-RAD test split")
    if not pristine_config_path.is_file():
        pristine_config_path.write_bytes(config_path.read_bytes())
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["mm_vision_tower"] = str(vision.resolve())
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    provenance = {
        "schema_version": "1.0",
        "source_repository": "https://github.com/microsoft/LLaVA-Med",
        "source_revision": SOURCE_REVISION,
        "source_path": str(source.resolve()),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_path": str(model.resolve()),
        "vision_model_id": VISION_ID,
        "vision_model_revision": VISION_REVISION,
        "vision_model_path": str(vision.resolve()),
        "dataset_id": "flaviagiammarino/vqa-rad",
        "dataset_revision": DATASET_REVISION,
        "dataset_test_path": str((dataset / DATASET_TEST_FILE).resolve()),
        "license": (
            "Model card metadata: Apache-2.0; upstream source checkout: Microsoft "
            "Research License; upstream model card and license files are authoritative"
        ),
        "clinical_use": False,
    }
    (destination / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download pinned LLaVA-Med runtime assets")
    parser.add_argument("--destination", required=True, type=Path)
    return parser


def main(arguments: list[str] | None = None) -> int:
    values = build_parser().parse_args(arguments)
    print(json.dumps(prepare_assets(values.destination), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

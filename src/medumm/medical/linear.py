from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from medumm.core.io import ensure_directory, write_json


TOKEN_PATTERN = re.compile(r"[\w.+/%-]+", flags=re.UNICODE)


def featurize(question: str, image_paths: list[str], text_dimensions: int) -> np.ndarray:
    vector = np.zeros(text_dimensions + 4, dtype=np.float32)
    tokens = TOKEN_PATTERN.findall(question.casefold())
    for token in tokens:
        index = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big")
        vector[index % text_dimensions] += 1
    norm = float(np.linalg.norm(vector[:text_dimensions]))
    if norm:
        vector[:text_dimensions] /= norm
    means, deviations = [], []
    for raw_path in image_paths:
        with Image.open(Path(raw_path)) as image:
            pixels = np.asarray(image.convert("L"), dtype=np.float32) / 255
        means.append(float(pixels.mean()))
        deviations.append(float(pixels.std()))
    vector[-4:] = (
        float(np.mean(means)) if means else 0,
        float(np.mean(deviations)) if deviations else 0,
        min(len(image_paths), 8) / 8,
        min(len(tokens), 128) / 128,
    )
    return vector


def train_classifier(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    class_count: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    generator = np.random.default_rng(seed)
    weights = generator.normal(0, 0.01, (features.shape[1], class_count)).astype(np.float32)
    bias = np.zeros(class_count, dtype=np.float32)
    history = []
    indices = np.arange(len(features))
    for epoch in range(1, epochs + 1):
        logits = features @ weights + bias
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        loss = -np.log(probabilities[indices, targets] + 1e-12).mean()
        gradient = probabilities
        gradient[indices, targets] -= 1
        gradient /= len(features)
        weights -= learning_rate * (features.T @ gradient + weight_decay * weights)
        bias -= learning_rate * gradient.sum(axis=0)
        if epoch in {1, epochs} or epoch % max(1, epochs // 5) == 0:
            accuracy = float(((features @ weights + bias).argmax(axis=1) == targets).mean())
            history.append({"epoch": float(epoch), "loss": float(loss), "accuracy": accuracy})
    return weights, bias, history


def save_model(
    output_directory: str | Path,
    *,
    weights: np.ndarray,
    bias: np.ndarray,
    labels: list[str],
    text_dimensions: int,
    metadata: dict[str, Any],
) -> Path:
    directory = ensure_directory(output_directory)
    np.savez_compressed(directory / "model.npz", weights=weights, bias=bias)
    return write_json(
        directory / "manifest.json",
        {
            "format": "medumm-linear-v1",
            "checkpoint": "model.npz",
            "labels": labels,
            "text_dimensions": text_dimensions,
            "clinical_use": False,
            **metadata,
        },
    )


def load_model(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    manifest_path = Path(path).expanduser()
    manifest_path = manifest_path / "manifest.json" if manifest_path.is_dir() else manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "medumm-linear-v1":
        raise ValueError("Unsupported MedUMM linear checkpoint.")
    with np.load(manifest_path.parent / manifest["checkpoint"]) as state:
        return state["weights"], state["bias"], manifest

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from medumm.core.interfaces import PostTrainer
from medumm.core.io import write_json
from medumm.core.results import Artifact, TrainingResult
from medumm.core.runtime import RuntimeContext
from medumm.medical import load_medical_vqa
from medumm.medical.linear import featurize, save_model, train_classifier


class MedicalSFTTrainer(PostTrainer):
    """Train the dependency-light softmax baseline and write a loadable checkpoint."""

    name = "medical_sft"

    def fit(
        self,
        config: dict[str, Any],
        *,
        config_path: str | Path | None = None,
        runtime: RuntimeContext,
    ) -> TrainingResult:
        root = runtime.project_root
        data_config = config.get("data")
        if not isinstance(data_config, dict):
            raise ValueError("Medical SFT requires a data mapping.")
        if data_config.get("deidentified") is not True:
            raise ValueError("Set data.deidentified=true only after verifying the dataset.")
        samples = load_medical_vqa(data_config, project_root=root)
        labels = sorted({sample.answers[0] for sample in samples})
        if len(samples) < 2 or len(labels) < 2:
            raise ValueError("Medical SFT needs at least two samples and two labels.")
        dimensions = int(config.get("text_dimensions", 128))
        features = np.stack([
            featurize(sample.question, sample.image_paths, dimensions) for sample in samples
        ])
        label_index = {label: index for index, label in enumerate(labels)}
        targets = np.asarray([label_index[sample.answers[0]] for sample in samples])
        weights, bias, history = train_classifier(
            features,
            targets,
            class_count=len(labels),
            epochs=int(config.get("epochs", 120)),
            learning_rate=float(config.get("learning_rate", 0.5)),
            weight_decay=float(config.get("weight_decay", 1e-4)),
            seed=int(config.get("seed", 42)),
        )
        output_directory = Path(config.get("output_directory", "outputs/post_training/medical_sft"))
        output_directory = output_directory if output_directory.is_absolute() else root / output_directory
        accuracy = float(((features @ weights + bias).argmax(axis=1) == targets).mean())
        manifest = save_model(
            output_directory,
            weights=weights,
            bias=bias,
            labels=labels,
            text_dimensions=dimensions,
            metadata={"method": self.name, "samples": len(samples), "train_accuracy": accuracy},
        )
        history_path = write_json(output_directory / "history.json", history)
        result = TrainingResult(
            method=self.name,
            status="completed",
            output_directory=str(output_directory),
            checkpoint_path=str(manifest),
            metrics={"train_accuracy": accuracy},
            artifacts=[
                Artifact("checkpoint_manifest", str(manifest), "application/json"),
                Artifact("training_history", str(history_path), "application/json"),
            ],
            metadata={
                "samples": len(samples),
                "labels": len(labels),
                "clinical_use": False,
                "run_id": runtime.run_id,
            },
        )
        write_json(output_directory / "result.json", result.to_dict())
        return result

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from medumm.training.config import DistributedTrainingConfig
from medumm.training.distributed import normalize_distributed_environment
from tests.conftest import PROJECT_ROOT


def test_distributed_training_config_validates_unsafe_combinations():
    with pytest.raises(ValueError, match="world_size"):
        DistributedTrainingConfig.from_mapping({"strategy": "single"}, world_size=2)
    with pytest.raises(ValueError, match="activation checkpointing"):
        DistributedTrainingConfig.from_mapping(
            {"strategy": "fsdp", "activation_checkpointing": True}
        )
    value = DistributedTrainingConfig.from_mapping(
        {
            "strategy": "fsdp",
            "activation_checkpointing": True,
            "activation_checkpoint_module_classes": ["Linear"],
            "ema_decay": 0.95,
        }
    )
    assert value.activation_checkpoint_module_classes == ("Linear",)
    assert value.ema_decay == 0.95


def test_slurm_environment_is_normalized_for_pytorch():
    environment = {
        "SLURM_PROCID": "3",
        "SLURM_LOCALID": "1",
        "SLURM_NTASKS": "8",
    }
    context = normalize_distributed_environment(environment)
    assert (context.rank, context.local_rank, context.world_size) == (3, 1, 8)
    assert environment["RANK"] == "3"
    assert environment["LOCAL_RANK"] == "1"
    assert environment["WORLD_SIZE"] == "8"


def test_single_process_checkpoint_and_ema_resume(tmp_path):
    torch = pytest.importorskip("torch")
    from medumm.post_training import PostTrainingRunner

    output = tmp_path / "single"
    base = {
        "method": "distributed_reference",
        "seed": 7,
        "output_directory": str(output),
        "data": {"synthetic": True, "samples": 32, "input_dimensions": 8},
        "training": {
            "epochs": 3,
            "batch_size": 4,
            "hidden_dimensions": 12,
            "learning_rate": 0.03,
            "max_optimizer_steps": 2,
        },
        "distributed": {
            "strategy": "single",
            "precision": "fp32",
            "gradient_accumulation_steps": 2,
            "ema_decay": 0.9,
            "checkpoint_every_steps": 1,
        },
    }
    first = PostTrainingRunner().run(base, config_path=PROJECT_ROOT / "pyproject.toml")
    assert first.status == "interrupted"
    resumed_config = json.loads(json.dumps(base))
    resumed_config["training"].pop("max_optimizer_steps")
    resumed_config["resume_from"] = "auto"
    second = PostTrainingRunner().run(
        resumed_config,
        config_path=PROJECT_ROOT / "pyproject.toml",
    )
    assert second.status == "completed"
    assert second.metadata["resumed_from"]
    assert second.metadata["ema_updates"] > 2
    checkpoint = Path(second.checkpoint_path)
    assert (checkpoint / "COMPLETED").is_file()
    sidecar = torch.load(checkpoint / "rank-00000.pt", weights_only=False)
    assert sidecar["ema"]["num_updates"] == second.metadata["ema_updates"]


def test_two_process_ddp_checkpoint_resume(tmp_path):
    pytest.importorskip("torch")
    output = tmp_path / "ddp"
    config = PROJECT_ROOT / "configs/post_training/distributed_reference_ddp.yaml"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    common = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        "-m",
        "medumm",
        "post-train",
        "--config",
        str(config),
        "--set",
        "runtime.device=cpu",
        "--set",
        f"post_training.output_directory={output}",
    ]
    subprocess.run(
        [*common, "--set", "post_training.training.max_optimizer_steps=2"],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    subprocess.run(
        [*common, "--set", "post_training.resume_from=auto"],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    report = json.loads((output / "distributed_report.json").read_text())
    checkpoint = Path(report["checkpoint"])
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    assert report["status"] == "completed"
    assert report["resumed_from"]
    assert report["distributed"]["world_size"] == 2
    assert report["distributed"]["strategy"] == "ddp"
    assert manifest["format"] == "torch_distributed_checkpoint"
    assert manifest["has_ema"] is True
    assert (checkpoint / "rank-00000.pt").is_file()
    assert (checkpoint / "rank-00001.pt").is_file()
    assert len(list((checkpoint / "shards").glob("*.distcp"))) >= 1

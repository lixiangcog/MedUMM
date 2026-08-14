from contextlib import nullcontext
import subprocess

import numpy as np

from medumm.backbones.catalog_model import CatalogModelAdapter
from medumm.backbones.recipes import (
    MODEL_ADAPTER_RECIPES,
    AdapterImplementation,
    ModelExecutor,
)
from medumm.core.config import load_config
from medumm.core.contracts import TaskType
from medumm.environments import ENVIRONMENT_CATALOG
from medumm.inference.request import InferenceRequest
from scripts.prepare_real_model_adapters_v1_5 import ACCESS_BLOCKED, MODELS
from tests.conftest import PROJECT_ROOT


def test_v15_acceptance_configs_use_pinned_builtin_recipes(monkeypatch):
    image = PROJECT_ROOT / "examples" / "medical" / "images" / "synthetic_scan.pgm"
    monkeypatch.setenv("MEDUMM_ADAPTER_SMOKE_IMAGE", str(image))
    values = {
        "medmo_4b": ("MEDMO_4B_MODEL_PATH", "medmo-4b"),
        "medmo_8b": ("MEDMO_8B_MODEL_PATH", "medmo-8b"),
        "lingshu_i_8b": ("LINGSHU_I_8B_MODEL_PATH", "lingshu-i-8b"),
        "fleming_vl_8b": ("FLEMING_VL_8B_MODEL_PATH", "fleming-vl-8b"),
    }
    for model, (variable, directory) in values.items():
        monkeypatch.setenv(variable, f"/models/{directory}")
        config = load_config(PROJECT_ROOT / "configs" / "inference" / f"{model}_v1.5.yaml")
        block = config["inference"]
        assert block["backbone"] == model
        assert block["config"]["revision"] == ENVIRONMENT_CATALOG.get(model).model_revision
        assert MODEL_ADAPTER_RECIPES.get(model).implementation is AdapterImplementation.BUILTIN


def test_v15_assets_separate_open_downloads_from_gated_models():
    assert set(MODELS) == {"medmo_4b", "medmo_8b", "lingshu_i_8b", "fleming_vl_8b"}
    assert set(ACCESS_BLOCKED) == {"medsiglip", "medgemma_1_5_4b_it", "maira_2"}
    assert all(value["access"] == "gated" for value in ACCESS_BLOCKED.values())


def test_v15_exact_transformers_contracts_match_upstream_configs():
    expected = {
        "medmo_4b": "transformers==4.57.1",
        "medmo_8b": "transformers==4.57.1",
        "lingshu_i_8b": "transformers==4.52.4",
        "fleming_vl_8b": "transformers==4.46.0",
    }
    for model, dependency in expected.items():
        assert dependency in ENVIRONMENT_CATALOG.get(model).dependencies


def test_lingshu_i_uses_its_native_internvl_prompt_executor():
    recipe = MODEL_ADAPTER_RECIPES.get("lingshu_i_8b")
    assert recipe.executor is ModelExecutor.INTERNVL_TRANSFORMERS
    assert recipe.prompt_protocol == "internvl_2_5_mpt"
    assert recipe.max_images == 4


def test_lingshu_i_native_executor_renders_upstream_mpt_prompt():
    class Batch(dict):
        def __init__(self):
            super().__init__(input_ids=np.asarray([[1, 2]]))
            self.input_ids = self["input_ids"]

        def to(self, *, device, dtype):
            assert device == "cpu"
            assert dtype == "fake-bfloat16"
            return self

    class Processor:
        rendered = None

        def __call__(self, *, text, images, padding, return_tensors):
            self.rendered = text[0]
            assert len(images) == 1
            assert padding is True
            assert return_tensors == "pt"
            return Batch()

        @staticmethod
        def batch_decode(*args, **kwargs):
            return ["pathology image"]

    class Model:
        @staticmethod
        def parameters():
            class Parameter:
                dtype = "fake-bfloat16"

            yield Parameter()

        @staticmethod
        def generate(**kwargs):
            assert "input_ids" in kwargs
            return np.asarray([[1, 2, 3, 4]])

    class Torch:
        inference_mode = staticmethod(nullcontext)

    adapter = CatalogModelAdapter("lingshu_i_8b")
    adapter._processor = Processor()
    adapter._model = Model()
    adapter._torch = Torch()
    adapter._device = "cpu"
    adapter._dtype = "bfloat16"
    adapter._defaults = {"max_new_tokens": 2, "do_sample": False}
    adapter._system_prompt = "medical research assistant"
    adapter.model_path = "/models/lingshu-i-8b"
    adapter.model_revision = ENVIRONMENT_CATALOG.get("lingshu_i_8b").model_revision
    image = PROJECT_ROOT / "examples" / "medical" / "images" / "synthetic_scan.pgm"
    result = adapter._internvl_transformers_generate(
        InferenceRequest(
            task=TaskType.UNDERSTANDING,
            request_id="lingshu-native-prompt",
            prompt="What is shown?",
            images=[str(image)],
        )
    )
    assert result.text == "pathology image"
    assert result.metadata["executor"] == "internvl_transformers"
    assert adapter._processor.rendered == (
        "<|im_start|>system\nmedical research assistant<|im_end|>\n"
        "<|im_start|>user\n<IMG_CONTEXT>\nWhat is shown?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def test_v15_scripts_are_parseable_and_use_isolated_environments():
    scripts = (
        "prepare_real_model_assets_v1.5.sh",
        "prepare_model_envs_v1.5.sh",
        "slurm_real_model_adapters_v1.5.sh",
    )
    for name in scripts:
        subprocess.run(["bash", "-n", str(PROJECT_ROOT / "scripts" / name)], check=True)
    runtime = (PROJECT_ROOT / "scripts" / scripts[-1]).read_text(encoding="utf-8")
    assert '"${MEDUMM_ENV_ROOT}/medmo_4b/bin/python"' in runtime
    assert '"${MEDUMM_ENV_ROOT}/lingshu_i_8b/bin/python"' in runtime
    assert '"${MEDUMM_ENV_ROOT}/fleming_vl_8b/bin/python"' in runtime
    assert "HF_HUB_OFFLINE=1" in runtime

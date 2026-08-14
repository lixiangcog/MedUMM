from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class ModelExecutor(str, Enum):
    """Concrete loading and inference protocol used by a model adapter."""

    TRANSFORMERS_PIPELINE = "transformers_image_text_pipeline"
    QWEN2_VL = "qwen2_vl_chat"
    QWEN2_5_VL = "qwen2_5_vl_chat"
    QWEN3_VL = "qwen3_vl_chat"
    CHEXAGENT = "chexagent_chat"
    INTERNVL_CHAT = "internvl_chat"
    HF_CONTRASTIVE = "transformers_contrastive"
    OPEN_CLIP_HUB = "open_clip_hf_hub"
    MEDCLIP = "medclip"
    M3D_LAMED = "m3d_lamed"
    LLAVA_REPOSITORY = "llava_repository"
    OPEN_FLAMINGO = "open_flamingo"
    RADFM = "radfm"
    UNIMED_CLIP = "unimed_clip"
    VILA = "vila"
    XRAYGPT = "xraygpt"
    OPENBIOMED = "openbiomed"
    GMAI_VL = "gmai_vl"
    UNIMED_VL = "unimed_vl"


class AdapterImplementation(str, Enum):
    BUILTIN = "builtin_executor"
    OFFICIAL_SOURCE = "official_source_executor"


@dataclass(frozen=True, slots=True)
class ModelAdapterRecipe:
    name: str
    executor: ModelExecutor
    implementation: AdapterImplementation
    model_type: str
    model_class: str | None
    processor_class: str | None
    prompt_protocol: str
    official_entrypoint: str | None = None
    source_checkout_required: bool = False
    max_images: int | None = 1
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["executor"] = self.executor.value
        result["implementation"] = self.implementation.value
        result["notes"] = list(self.notes)
        return result


def _builtin(
    name: str,
    executor: ModelExecutor,
    model_type: str,
    model_class: str | None,
    processor_class: str | None,
    prompt_protocol: str,
    *,
    max_images: int | None = 1,
    notes: tuple[str, ...] = (),
) -> ModelAdapterRecipe:
    return ModelAdapterRecipe(
        name=name,
        executor=executor,
        implementation=AdapterImplementation.BUILTIN,
        model_type=model_type,
        model_class=model_class,
        processor_class=processor_class,
        prompt_protocol=prompt_protocol,
        max_images=max_images,
        notes=notes,
    )


def _official(
    name: str,
    executor: ModelExecutor,
    model_type: str,
    entrypoint: str,
    prompt_protocol: str,
    *,
    max_images: int | None = 1,
    notes: tuple[str, ...] = (),
) -> ModelAdapterRecipe:
    return ModelAdapterRecipe(
        name=name,
        executor=executor,
        implementation=AdapterImplementation.OFFICIAL_SOURCE,
        model_type=model_type,
        model_class=None,
        processor_class=None,
        prompt_protocol=prompt_protocol,
        official_entrypoint=entrypoint,
        source_checkout_required=True,
        max_images=max_images,
        notes=notes,
    )


# This is intentionally explicit. A catalog name is not considered adapter coverage until it
# has a model-specific recipe. Shared executors are allowed, but every release selects its real
# model class, prompt protocol, and upstream entry point here.
_VALUES = (
    _official(
        "bimedix2_8b",
        ModelExecutor.LLAVA_REPOSITORY,
        "llava_llama",
        "llava.model.builder:load_pretrained_model",
        "llava_conversation",
        max_images=None,
    ),
    _builtin(
        "biomedclip",
        ModelExecutor.OPEN_CLIP_HUB,
        "open_clip",
        "CustomTextCLIP",
        "HFTokenizer",
        "contrastive_candidate_ranking",
    ),
    _official(
        "biomedgpt_10b",
        ModelExecutor.OPENBIOMED,
        "biomedgpt",
        "open_biomed.models.foundation_models.biomedgpt.biomedgpt:BioMedGPT",
        "openbiomed_multimodal",
    ),
    _builtin(
        "chexagent_8b",
        ModelExecutor.CHEXAGENT,
        "chexagent",
        "AutoModelForCausalLM",
        "AutoTokenizer",
        "chexagent_list_format",
        max_images=None,
    ),
    _builtin(
        "fleming_vl_8b",
        ModelExecutor.INTERNVL_CHAT,
        "internvl_chat",
        "AutoModel",
        "AutoTokenizer",
        "internvl_dynamic_tiles",
        max_images=None,
    ),
    _official(
        "gmai_vl",
        ModelExecutor.GMAI_VL,
        "xtuner_llava",
        "xtuner._lite.modelings.llava.llava:build_llava_model",
        "xtuner_llava_chat",
        max_images=None,
    ),
    _official(
        "healthgpt_m3",
        ModelExecutor.LLAVA_REPOSITORY,
        "llava_qwen2",
        "llava.model.builder:load_pretrained_model",
        "llava_conversation",
        max_images=None,
    ),
    _official(
        "huatuogpt_vision_34b",
        ModelExecutor.LLAVA_REPOSITORY,
        "llava_llama",
        "llava.model.builder:load_pretrained_model",
        "llava_conversation",
        max_images=None,
    ),
    _official(
        "huatuogpt_vision_7b",
        ModelExecutor.LLAVA_REPOSITORY,
        "llava_qwen2",
        "llava.model.builder:load_pretrained_model",
        "llava_conversation",
        max_images=None,
    ),
    _builtin(
        "lingshu_32b",
        ModelExecutor.QWEN2_5_VL,
        "qwen2_5_vl",
        "Qwen2_5_VLForConditionalGeneration",
        "AutoProcessor",
        "qwen_vl_chat_template",
        max_images=4,
    ),
    _builtin(
        "lingshu_7b",
        ModelExecutor.QWEN2_5_VL,
        "qwen2_5_vl",
        "Qwen2_5_VLForConditionalGeneration",
        "AutoProcessor",
        "qwen_vl_chat_template",
        max_images=4,
    ),
    _builtin(
        "lingshu_i_8b",
        ModelExecutor.TRANSFORMERS_PIPELINE,
        "internvl",
        "InternVLForConditionalGeneration",
        "AutoProcessor",
        "transformers_multimodal_chat_template",
        max_images=None,
    ),
    _official(
        "llava_med_v1_5_7b",
        ModelExecutor.LLAVA_REPOSITORY,
        "llava_mistral",
        "llava.model.builder:load_pretrained_model",
        "llava_conversation",
    ),
    _builtin(
        "m3d_lamed_4b",
        ModelExecutor.M3D_LAMED,
        "lamed_phi3",
        "AutoModelForCausalLM",
        "AutoTokenizer",
        "m3d_volume_patch_tokens",
        max_images=0,
    ),
    _builtin(
        "maira_2",
        ModelExecutor.TRANSFORMERS_PIPELINE,
        "maira2",
        "AutoModelForCausalLM",
        "AutoProcessor",
        "transformers_multimodal_chat_template",
        max_images=2,
    ),
    _official(
        "med_flamingo_9b",
        ModelExecutor.OPEN_FLAMINGO,
        "open_flamingo",
        "open_flamingo:create_model_and_transforms",
        "flamingo_image_chunks",
        max_images=None,
    ),
    _official(
        "medclip",
        ModelExecutor.MEDCLIP,
        "medclip",
        "medclip:MedCLIPModel",
        "contrastive_candidate_ranking",
    ),
    _official(
        "meddr",
        ModelExecutor.INTERNVL_CHAT,
        "internvl_chat",
        "src.model.internvl_chat.modeling_internvl_chat:InternVLChatModel",
        "internvl_dynamic_tiles",
        max_images=None,
    ),
    _builtin(
        "medgemma_1_27b_it",
        ModelExecutor.TRANSFORMERS_PIPELINE,
        "gemma3",
        "Gemma3ForConditionalGeneration",
        "AutoProcessor",
        "transformers_multimodal_chat_template",
        max_images=None,
    ),
    _builtin(
        "medgemma_1_5_4b_it",
        ModelExecutor.TRANSFORMERS_PIPELINE,
        "gemma3",
        "Gemma3ForConditionalGeneration",
        "AutoProcessor",
        "transformers_multimodal_chat_template",
        max_images=None,
    ),
    _builtin(
        "medmo_4b",
        ModelExecutor.QWEN3_VL,
        "qwen3_vl",
        "Qwen3VLForConditionalGeneration",
        "AutoProcessor",
        "qwen_vl_chat_template",
        max_images=None,
    ),
    _builtin(
        "medmo_8b",
        ModelExecutor.QWEN3_VL,
        "qwen3_vl",
        "Qwen3VLForConditionalGeneration",
        "AutoProcessor",
        "qwen_vl_chat_template",
        max_images=None,
    ),
    _builtin(
        "medsiglip",
        ModelExecutor.HF_CONTRASTIVE,
        "siglip",
        "AutoModel",
        "AutoProcessor",
        "contrastive_candidate_ranking",
    ),
    _builtin(
        "medvlm_r1",
        ModelExecutor.QWEN2_VL,
        "qwen2_vl",
        "Qwen2VLForConditionalGeneration",
        "AutoProcessor",
        "qwen_vl_chat_template",
    ),
    _builtin(
        "plip",
        ModelExecutor.HF_CONTRASTIVE,
        "clip",
        "CLIPModel",
        "AutoProcessor",
        "contrastive_candidate_ranking",
    ),
    _builtin(
        "pubmedclip",
        ModelExecutor.HF_CONTRASTIVE,
        "clip",
        "CLIPModel",
        "AutoProcessor",
        "contrastive_candidate_ranking",
    ),
    _builtin(
        "quiltnet",
        ModelExecutor.HF_CONTRASTIVE,
        "clip",
        "CLIPModel",
        "AutoProcessor",
        "contrastive_candidate_ranking",
        notes=("The pinned release exposes a Transformers CLIPModel config.",),
    ),
    _official(
        "radfm_14b",
        ModelExecutor.RADFM,
        "radfm",
        "Model.RadFM.multimodality_model:MultiLLaMAForCausalLM",
        "radfm_image_padding_tokens",
        max_images=None,
    ),
    _official(
        "unimed_clip",
        ModelExecutor.UNIMED_CLIP,
        "unimed_clip",
        "open_clip:create_model_and_transforms",
        "contrastive_candidate_ranking",
        notes=("Weights are distributed separately from the source repository.",),
    ),
    _builtin(
        "unimed_vl",
        ModelExecutor.UNIMED_VL,
        "qwen2_remote_code",
        "AutoModelForCausalLM",
        "AutoProcessor",
        "unimedvl_interleaved_chat",
        max_images=None,
    ),
    _official(
        "vila_m3_3b",
        ModelExecutor.VILA,
        "vila",
        "llava.model.builder:load_pretrained_model",
        "vila_conversation",
        max_images=None,
    ),
    _official(
        "xraygpt_7b",
        ModelExecutor.XRAYGPT,
        "minigpt4",
        "xraygpt.conversation.conversation:Chat",
        "xraygpt_conversation",
    ),
)


class ModelAdapterRecipeCatalog:
    def __init__(self, values: tuple[ModelAdapterRecipe, ...]) -> None:
        self._values = values
        self._by_name = {value.name: value for value in values}
        if len(self._values) != len(self._by_name):
            raise ValueError("Duplicate model adapter recipes.")

    def get(self, name: str) -> ModelAdapterRecipe:
        normalized = name.strip().lower()
        try:
            return self._by_name[normalized]
        except KeyError as error:
            raise KeyError(f"No model adapter recipe for {name!r}.") from error

    def names(self) -> list[str]:
        return sorted(self._by_name)

    def values(self) -> tuple[ModelAdapterRecipe, ...]:
        return tuple(self._by_name[name] for name in self.names())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "count": len(self._values),
            "recipes": [value.to_dict() for value in self.values()],
        }


MODEL_ADAPTER_RECIPES = ModelAdapterRecipeCatalog(_VALUES)

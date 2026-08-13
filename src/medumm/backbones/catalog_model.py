from __future__ import annotations

from importlib import import_module
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image

from medumm.core.contracts import (
    ArchitectureFamily,
    Modality,
    ModelCapabilities,
    TaskType,
)
from medumm.core.interfaces import ModelAdapter
from medumm.core.results import InferenceResult
from medumm.core.runtime import RuntimeContext
from medumm.inference.request import InferenceRequest
from medumm.resources import AccessLevel, MODEL_RESOURCES, ModelResourceSpec, ModelRuntimeFamily


SYSTEM_PROMPT = (
    "You are a medical multimodal research assistant. Use only the supplied evidence, "
    "state when evidence is insufficient, and do not present output as clinical advice."
)


class CatalogModelAdapter(ModelAdapter):
    """Spec-backed adapter for one cataloged model release.

    Transformer-native releases share tested executors. Models whose official runtimes
    have incompatible dependency stacks use a strict Python bridge instead of vendoring
    upstream code into MedUMM.
    """

    def __init__(self, resource_name: str) -> None:
        self.spec = MODEL_RESOURCES.get(resource_name)
        self.name = self.spec.name
        architecture = (
            ArchitectureFamily.CONTRASTIVE
            if self.spec.runtime_family
            in {ModelRuntimeFamily.HF_CONTRASTIVE, ModelRuntimeFamily.OPEN_CLIP}
            else ArchitectureFamily.AUTOREGRESSIVE
        )
        self.capabilities = ModelCapabilities(
            tasks=frozenset(self.spec.tasks),
            input_modalities=frozenset(self.spec.input_modalities),
            output_modalities=frozenset({Modality.TEXT}),
            architecture=architecture,
            supports_batching=False,
            max_batch_size=1,
            max_images=None if Modality.IMAGE_SET in self.spec.input_modalities else 1,
            supports_hidden_states=True,
            notes=(
                f"catalog_status={self.spec.status.value}",
                f"access={self.spec.access.value}",
                f"runtime_family={self.spec.runtime_family.value}",
                "Research use only; outputs require independent verification.",
                *self.spec.notes,
            ),
        )
        self._delegate: ModelAdapter | None = None
        self._model: Any = None
        self._processor: Any = None
        self._pipeline: Any = None

    def load(self, config: dict[str, Any], runtime: RuntimeContext) -> None:
        self.runtime = runtime
        self._enforce_access(config)
        model_path = str(config.get("model_path", self.spec.artifact_id)).strip()
        revision = str(config.get("revision", "")).strip() or None
        if revision and revision.casefold() in {"main", "master", "latest", "head"}:
            raise ValueError(
                f"{self.name} revision must be an immutable source commit, not {revision!r}."
            )
        if not Path(model_path).expanduser().exists() and not revision:
            raise ValueError(
                f"{self.name} requires config.revision with an immutable source commit. "
                "Use a local model_path to load an already pinned snapshot."
            )
        if self.spec.runtime_family is ModelRuntimeFamily.HF_IMAGE_TEXT:
            self._load_hf_pipeline(config, model_path=model_path, revision=revision)
        elif self.spec.runtime_family is ModelRuntimeFamily.HF_CONTRASTIVE:
            self._load_hf_contrastive(config, model_path=model_path, revision=revision)
        elif self.spec.runtime_family is ModelRuntimeFamily.OPEN_CLIP:
            self._load_open_clip(config)
        else:
            self._load_official_bridge(config, runtime)

    def _enforce_access(self, config: dict[str, Any]) -> None:
        if self.spec.access is not AccessLevel.OPEN and not bool(config.get("accept_terms")):
            raise PermissionError(
                f"{self.name} is {self.spec.access.value}. Review {self.spec.source} and "
                "set config.accept_terms=true only after accepting the upstream terms."
            )

    @staticmethod
    def _torch_dtype(torch: Any, config: dict[str, Any]) -> Any:
        dtype_name = str(config.get("torch_dtype", "bfloat16"))
        dtype = getattr(torch, dtype_name, None)
        if dtype is None:
            raise ValueError(f"Unknown torch dtype: {dtype_name}")
        return dtype

    def _load_hf_pipeline(
        self,
        config: dict[str, Any],
        *,
        model_path: str,
        revision: str | None,
    ) -> None:
        try:
            import torch
            from transformers import pipeline
        except ModuleNotFoundError as error:
            raise RuntimeError("Install MedUMM with the 'medical' extra.") from error
        self._pipeline = pipeline(
            "image-text-to-text",
            model=model_path,
            revision=revision,
            torch_dtype=self._torch_dtype(torch, config),
            device_map=config.get("device_map", "auto"),
            trust_remote_code=bool(config.get("trust_remote_code", self.spec.trust_remote_code)),
        )
        self._defaults = dict(
            config.get("parameters", {"max_new_tokens": 128, "do_sample": False})
        )
        self._system_prompt = str(config.get("system_prompt", SYSTEM_PROMPT))

    def _load_hf_contrastive(
        self,
        config: dict[str, Any],
        *,
        model_path: str,
        revision: str | None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ModuleNotFoundError as error:
            raise RuntimeError("Install MedUMM with the 'medical' extra.") from error
        self._processor = AutoProcessor.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=bool(config.get("trust_remote_code", self.spec.trust_remote_code)),
        )
        self._model = AutoModel.from_pretrained(
            model_path,
            revision=revision,
            torch_dtype=self._torch_dtype(torch, config),
            device_map=config.get("device_map", "auto"),
            trust_remote_code=bool(config.get("trust_remote_code", self.spec.trust_remote_code)),
        ).eval()

    def _load_open_clip(self, config: dict[str, Any]) -> None:
        try:
            import open_clip
            import torch
        except ModuleNotFoundError as error:
            raise RuntimeError("Install the optional 'open-clip-torch' package.") from error
        model_name = str(config.get("open_clip_model_name", "")).strip()
        checkpoint_path = Path(str(config.get("checkpoint_path", ""))).expanduser()
        if not model_name or not checkpoint_path.is_file():
            raise ValueError(
                "OpenCLIP resources require open_clip_model_name and a local "
                "checkpoint_path prepared from the pinned source revision."
            )
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=str(checkpoint_path),
            device=str(config.get("device", self.runtime.device)),
        )
        self._tokenizer = open_clip.get_tokenizer(model_name)
        self._torch = torch
        self._model.eval()

    def _load_official_bridge(
        self,
        config: dict[str, Any],
        runtime: RuntimeContext,
    ) -> None:
        bridge = str(config.get("bridge", "")).strip()
        if not bridge:
            native = {
                "llava_med_v1_5_7b": "medumm.backbones.llava_med:LlavaMedAdapter",
            }.get(self.name)
            bridge = native or ""
        if not bridge or ":" not in bridge:
            raise RuntimeError(
                f"{self.name} uses its official dependency stack. Configure bridge as "
                "'python.module:ModelAdapterClass'; see docs/resource-catalog-v0.8.md."
            )
        module_name, attribute_name = bridge.split(":", 1)
        factory = getattr(import_module(module_name), attribute_name)
        delegate = factory()
        if not isinstance(delegate, ModelAdapter):
            raise TypeError(f"Bridge {bridge!r} did not construct a ModelAdapter.")
        delegate_config = dict(config)
        for key in ("bridge", "accept_terms"):
            delegate_config.pop(key, None)
        delegate.load(delegate_config, runtime)
        self._delegate = delegate

    @staticmethod
    def _response(value: Any) -> str:
        if isinstance(value, list) and value:
            return CatalogModelAdapter._response(value[0])
        if isinstance(value, dict):
            generated = value.get("generated_text", "")
            if isinstance(generated, list):
                for message in reversed(generated):
                    if isinstance(message, dict) and isinstance(message.get("content"), str):
                        return message["content"].strip()
            return str(generated).strip()
        return str(value).strip()

    def _hf_generate(self, request: InferenceRequest) -> InferenceResult:
        content: list[dict[str, Any]] = []
        opened: list[Image.Image] = []
        try:
            for raw_path in [*request.images, *request.volumes]:
                image = Image.open(Path(raw_path)).convert("RGB")
                opened.append(image)
                content.append({"type": "image", "image": image})
            content.append({"type": "text", "text": str(request.prompt or "")})
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self._system_prompt}],
                },
                {"role": "user", "content": content},
            ]
            started = perf_counter()
            output = self._pipeline(text=messages, **{**self._defaults, **request.parameters})
            duration_ms = (perf_counter() - started) * 1000
        finally:
            for image in opened:
                image.close()
        return InferenceResult(
            request_id=request.request_id,
            task=TaskType.UNDERSTANDING,
            model_name=self.name,
            text=self._response(output),
            metadata={
                "resource": self.name,
                "source": self.spec.source,
                "clinical_use": False,
                **request.metadata,
            },
            duration_ms=duration_ms,
        )

    @staticmethod
    def _candidates(request: InferenceRequest) -> list[str]:
        values = request.parameters.get("candidates")
        if not isinstance(values, list) or len(values) < 2:
            raise ValueError(
                "Contrastive model requests require parameters.candidates with at least two labels."
            )
        return [str(item) for item in values]

    def _hf_rank(self, request: InferenceRequest) -> InferenceResult:
        import torch

        if len(request.images) != 1:
            raise ValueError("Contrastive candidate ranking requires exactly one image.")
        candidates = self._candidates(request)
        with Image.open(request.images[0]) as image:
            inputs = self._processor(
                text=candidates,
                images=image.convert("RGB"),
                return_tensors="pt",
                padding=True,
            )
        device = next(self._model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self._model(**inputs)
            logits = getattr(outputs, "logits_per_image", None)
            if logits is None:
                raise RuntimeError("The contrastive model did not return logits_per_image.")
            probabilities = logits[0].float().softmax(dim=-1).cpu().tolist()
        scores = {candidate: float(score) for candidate, score in zip(candidates, probabilities)}
        prediction = max(scores, key=scores.get)
        return InferenceResult(
            request_id=request.request_id,
            task=TaskType.UNDERSTANDING,
            model_name=self.name,
            text=prediction,
            scores=scores,
            metadata={"resource": self.name, "clinical_use": False, **request.metadata},
        )

    def _open_clip_rank(self, request: InferenceRequest) -> InferenceResult:
        if len(request.images) != 1:
            raise ValueError("Contrastive candidate ranking requires exactly one image.")
        candidates = self._candidates(request)
        device = str(next(self._model.parameters()).device)
        with Image.open(request.images[0]) as image:
            image_tensor = self._preprocess(image.convert("RGB")).unsqueeze(0).to(device)
        text_tensor = self._tokenizer(candidates).to(device)
        with self._torch.inference_mode():
            image_features = self._model.encode_image(image_tensor)
            text_features = self._model.encode_text(text_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            probabilities = (100 * image_features @ text_features.T).softmax(dim=-1)[0]
        values = probabilities.float().cpu().tolist()
        scores = {candidate: float(score) for candidate, score in zip(candidates, values)}
        return InferenceResult(
            request_id=request.request_id,
            task=TaskType.UNDERSTANDING,
            model_name=self.name,
            text=max(scores, key=scores.get),
            scores=scores,
            metadata={"resource": self.name, "clinical_use": False, **request.metadata},
        )

    def understand_batch(self, requests: list[InferenceRequest]) -> list[InferenceResult]:
        if self._delegate is not None:
            delegated = self._delegate.understand_batch(requests)
            for result in delegated:
                result.model_name = self.name
                result.metadata = {
                    "resource": self.name,
                    "bridge_model": self._delegate.name,
                    **result.metadata,
                }
            return delegated
        if self.spec.runtime_family is ModelRuntimeFamily.HF_IMAGE_TEXT:
            return [self._hf_generate(request) for request in requests]
        if self.spec.runtime_family is ModelRuntimeFamily.HF_CONTRASTIVE:
            return [self._hf_rank(request) for request in requests]
        if self.spec.runtime_family is ModelRuntimeFamily.OPEN_CLIP:
            return [self._open_clip_rank(request) for request in requests]
        raise RuntimeError(f"{self.name} has not been loaded.")

    def close(self) -> None:
        if self._delegate is not None:
            self._delegate.close()
        self._delegate = None
        self._model = None
        self._processor = None
        self._pipeline = None


def catalog_model_factory(resource_name: str):
    def create() -> CatalogModelAdapter:
        return CatalogModelAdapter(resource_name)

    return create

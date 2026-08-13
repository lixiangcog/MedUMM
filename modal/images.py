"""Build Modal images directly from the versioned MedUMM environment contracts.

This module intentionally imports Modal lazily: catalog inspection and local/HPC
execution do not require a Modal dependency or account.
"""

from __future__ import annotations

from pathlib import Path

from medumm.environments import ENVIRONMENT_CATALOG


ROOT = Path(__file__).resolve().parents[1]


def image_for(model: str):
    try:
        import modal
    except ModuleNotFoundError as error:
        raise RuntimeError("Install Modal with `pip install modal==1.1.4`.") from error

    spec = ENVIRONMENT_CATALOG.get(model)
    image = modal.Image.from_registry(
        spec.docker_base_image,
        add_python=spec.python,
    ).apt_install("git")
    image = image.pip_install_from_requirements(
        str(ROOT / "environments/models" / model / "lock.txt"),
        extra_index_url=spec.torch_index,
    )
    return image.env(
        {
            "HF_HOME": "/cache/huggingface",
            "MEDUMM_MODEL_ROOT": "/models",
            "MEDUMM_OUTPUT_ROOT": "/outputs",
            "MEDUMM_ENVIRONMENT_FINGERPRINT": spec.fingerprint(),
        }
    )


def catalog_images() -> dict[str, object]:
    """Return one isolated, lazily constructed Modal image per catalog model."""
    return {spec.model: image_for(spec.model) for spec in ENVIRONMENT_CATALOG.values()}

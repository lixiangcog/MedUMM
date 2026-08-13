from __future__ import annotations

import argparse
import json

from medumm.environments import ENVIRONMENT_CATALOG


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a MedUMM Modal model image")
    parser.add_argument("--model", required=True, choices=ENVIRONMENT_CATALOG.names())
    parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args()
    spec = ENVIRONMENT_CATALOG.get(arguments.model)
    value = {
        "model": spec.model,
        "profile": spec.profile,
        "image": spec.docker_base_image,
        "fingerprint": spec.fingerprint(),
        "gpu_count": spec.recommended_gpus,
        "minimum_gpu_memory_gb": spec.minimum_gpu_memory_gb,
        "command": f"modal run modal/run.py --model {spec.model}",
        "note": "Use modal.images.image_for(model) from a deployment function.",
    }
    if arguments.json:
        print(json.dumps(value, indent=2))
    else:
        for key, item in value.items():
            print(f"{key}: {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

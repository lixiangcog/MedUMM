#!/usr/bin/env bash
set -euo pipefail

MEDUMM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_NAME="${1:-}"
ENGINE="${MEDUMM_CONTAINER_ENGINE:-}"
OUTPUT_ROOT="${MEDUMM_CONTAINER_ROOT:-${MEDUMM_ROOT}/.containers}"
APPTAINER_BUILD_MODE="${MEDUMM_APPTAINER_BUILD_MODE:-fakeroot}"

if [[ -z "${MODEL_NAME}" ]]; then
  echo "usage: bash scripts/build_model_container.sh MODEL" >&2
  exit 2
fi
MODEL_DIR="${MEDUMM_ROOT}/environments/models/${MODEL_NAME}"
test -f "${MODEL_DIR}/Dockerfile" || { echo "unknown model: ${MODEL_NAME}" >&2; exit 2; }

if [[ -z "${ENGINE}" ]]; then
  for candidate in docker podman apptainer singularity; do
    if command -v "${candidate}" >/dev/null; then ENGINE="${candidate}"; break; fi
  done
fi
if [[ -z "${ENGINE}" ]]; then
  echo "no supported container engine found (docker, podman, apptainer, singularity)" >&2
  exit 4
fi

mkdir -p "${OUTPUT_ROOT}"
case "${ENGINE}" in
  docker|podman)
    "${ENGINE}" build -f "${MODEL_DIR}/Dockerfile" -t "medumm/${MODEL_NAME}:contract" "${MODEL_DIR}"
    ;;
  apptainer|singularity)
    case "${APPTAINER_BUILD_MODE}" in
      fakeroot) "${ENGINE}" build --fakeroot "${OUTPUT_ROOT}/${MODEL_NAME}.sif" "${MODEL_DIR}/apptainer.def" ;;
      remote) "${ENGINE}" build --remote "${OUTPUT_ROOT}/${MODEL_NAME}.sif" "${MODEL_DIR}/apptainer.def" ;;
      sudo) sudo "${ENGINE}" build "${OUTPUT_ROOT}/${MODEL_NAME}.sif" "${MODEL_DIR}/apptainer.def" ;;
      *) echo "MEDUMM_APPTAINER_BUILD_MODE must be fakeroot, remote, or sudo" >&2; exit 4 ;;
    esac
    ;;
  *) echo "unsupported container engine: ${ENGINE}" >&2; exit 4 ;;
esac

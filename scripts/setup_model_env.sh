#!/usr/bin/env bash
set -euo pipefail

MEDUMM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_NAME="${1:-}"
shift || true

if [[ -z "${MODEL_NAME}" ]]; then
  echo "usage: bash scripts/setup_model_env.sh MODEL [--check-only] [--accept-terms]" >&2
  exit 2
fi

CHECK_ONLY=0
ACCEPT_TERMS=0
for argument in "$@"; do
  case "${argument}" in
    --check-only) CHECK_ONLY=1 ;;
    --accept-terms) ACCEPT_TERMS=1 ;;
    *) echo "unknown argument: ${argument}" >&2; exit 2 ;;
  esac
done

CATALOG_PYTHON="${MEDUMM_CATALOG_PYTHON:-}"
if [[ -z "${CATALOG_PYTHON}" ]]; then
  for candidate in "${MEDUMM_ROOT}/.venv312/bin/python" "${MEDUMM_ROOT}/.venv/bin/python" python3; do
    if [[ -x "${candidate}" ]] || command -v "${candidate}" >/dev/null 2>&1; then
      if "${candidate}" -c 'import yaml' >/dev/null 2>&1; then
        CATALOG_PYTHON="${candidate}"
        break
      fi
    fi
  done
fi
if [[ -z "${CATALOG_PYTHON}" ]]; then
  echo "a Python interpreter with PyYAML is required to read the environment catalog" >&2
  exit 4
fi

CONTRACT=$("${CATALOG_PYTHON}" - "${MEDUMM_ROOT}" "${MODEL_NAME}" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, str(Path(sys.argv[1], "src")))
from medumm.environments import ENVIRONMENT_CATALOG
s = ENVIRONMENT_CATALOG.get(sys.argv[2])
print(s.python)
print(s.access)
print(s.fingerprint())
PY
)

PYTHON_MINOR=$(printf '%s\n' "${CONTRACT}" | sed -n '1p')
ACCESS=$(printf '%s\n' "${CONTRACT}" | sed -n '2p')
FINGERPRINT=$(printf '%s\n' "${CONTRACT}" | sed -n '3p')
ENV_ROOT="${MEDUMM_ENV_ROOT:-${MEDUMM_ROOT}/.venv-models}"
ENV_PATH="${ENV_ROOT}/${MODEL_NAME}"
PYTHON_COMMAND="${MEDUMM_PYTHON_COMMAND:-python${PYTHON_MINOR}}"
REQUIREMENTS="${MEDUMM_ROOT}/environments/models/${MODEL_NAME}/requirements.txt"
LOCK="${MEDUMM_ROOT}/environments/models/${MODEL_NAME}/lock.txt"

if [[ "${ACCESS}" != "open" && "${ACCEPT_TERMS}" -ne 1 ]]; then
  echo "${MODEL_NAME} uses ${ACCESS} assets; rerun with --accept-terms after accepting upstream terms." >&2
  exit 3
fi
command -v "${PYTHON_COMMAND}" >/dev/null || {
  echo "required interpreter ${PYTHON_COMMAND} was not found" >&2
  exit 4
}
test -f "${REQUIREMENTS}"
test -f "${LOCK}"

if [[ "${CHECK_ONLY}" -eq 0 ]]; then
  if [[ ! -x "${ENV_PATH}/bin/python" ]]; then
    "${PYTHON_COMMAND}" -m venv "${ENV_PATH}"
  fi
  if ! "${ENV_PATH}/bin/python" -c \
    'from importlib.metadata import version; raise SystemExit(version("pip") != "25.1.1")'; then
    if [[ -n "${MEDUMM_PIP_WHEEL:-}" ]]; then
      test -f "${MEDUMM_PIP_WHEEL}"
      "${ENV_PATH}/bin/python" -m pip install --no-index "${MEDUMM_PIP_WHEEL}"
    elif [[ "${MEDUMM_SKIP_PIP_UPGRADE:-0}" != "1" ]]; then
      "${ENV_PATH}/bin/python" -m pip install --upgrade "pip==25.1.1"
    fi
  fi
  if [[ -n "${MEDUMM_SOCKS_WHEEL:-}" ]] && \
    ! "${ENV_PATH}/bin/python" -c 'import socks' >/dev/null 2>&1; then
    test -f "${MEDUMM_SOCKS_WHEEL}"
    "${ENV_PATH}/bin/python" -m pip install --no-index "${MEDUMM_SOCKS_WHEEL}"
  fi
  TORCH_INDEX=$("${CATALOG_PYTHON}" - "${MEDUMM_ROOT}" "${MODEL_NAME}" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, str(Path(sys.argv[1], "src")))
from medumm.environments import ENVIRONMENT_CATALOG
print(ENVIRONMENT_CATALOG.get(sys.argv[2]).torch_index or "")
PY
)
  if [[ -n "${TORCH_INDEX}" ]]; then
    "${ENV_PATH}/bin/python" -m pip install --extra-index-url "${TORCH_INDEX}" -r "${LOCK}"
  else
    "${ENV_PATH}/bin/python" -m pip install -r "${LOCK}"
  fi
  "${ENV_PATH}/bin/python" -m pip install --no-deps --no-build-isolation -e "${MEDUMM_ROOT}"
  printf '%s\n' "${FINGERPRINT}" > "${ENV_PATH}/medumm-environment.sha256"
fi

test -x "${ENV_PATH}/bin/python"
"${ENV_PATH}/bin/python" - "${MEDUMM_ROOT}" "${MODEL_NAME}" <<'PY'
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(sys.argv[1], "src")))
from medumm.environments import ENVIRONMENT_CATALOG
from medumm.environments.render import inspect_current_environment
result = inspect_current_environment(ENVIRONMENT_CATALOG.get(sys.argv[2]))
print(json.dumps(result, indent=2))
if not result["valid"]:
    raise SystemExit(1)
PY

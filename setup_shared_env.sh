#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  echo "Run this script with: bash setup_shared_env.sh" >&2
  return 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found in PATH." >&2
  exit 1
fi

if [[ -z "${CONDA_PREFIX:-}" || -z "${CONDA_DEFAULT_ENV:-}" ]]; then
  echo "Activate a conda environment first, then run: bash setup_shared_env.sh" >&2
  exit 1
fi

if [[ "${CONDA_DEFAULT_ENV}" == "base" ]]; then
  echo "Do not install into the base conda environment. Create and activate a project env first." >&2
  exit 1
fi

ENV_NAME="${CONDA_DEFAULT_ENV}"

conda env update -n "${ENV_NAME}" -f "${REPO_ROOT}/environment.yml" --prune
python -m pip install -e "${REPO_ROOT}/caliby"

echo "Environment '${ENV_NAME}' is ready."
echo "It is already active in this shell."

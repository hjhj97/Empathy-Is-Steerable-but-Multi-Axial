#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE="${1:-${EPITOME_PROJECT:-}}"

if [[ -z "${SOURCE}" ]]; then
  echo "usage: $0 /path/to/Empathy-Mental-Health" >&2
  exit 2
fi

mkdir -p "${ROOT}/dataset"
for file in \
  emotional-reactions-reddit.csv \
  interpretations-reddit.csv \
  explorations-reddit.csv; do
  source_path="${SOURCE}/dataset/${file}"
  if [[ ! -f "${source_path}" ]]; then
    echo "missing EPITOME source file: ${source_path}" >&2
    exit 1
  fi
  cp "${source_path}" "${ROOT}/dataset/${file}"
done

"${PYTHON:-python}" "${ROOT}/scripts/filter_epitome_dataset.py"
echo "Prepared filtered EPITOME files under ${ROOT}/dataset/filtered"

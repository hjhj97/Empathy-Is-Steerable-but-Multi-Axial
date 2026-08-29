#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PY="${PYTHON:-python}"
LOG_DIR="${ROOT}/logs"

mkdir -p "${LOG_DIR}"
cd "${ROOT}"

echo "[start] $(date '+%F %T') Ex32 persona-shift subspace analysis"
"${PY}" src/analyze_ex32_persona_shift_subspace.py "$@"
echo "[done] $(date '+%F %T') Ex32 persona-shift subspace analysis"

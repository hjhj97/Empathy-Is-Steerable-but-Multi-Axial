#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PY="${PYTHON:-python}"
SCRIPT="${ROOT}/src/run_ex22_bootstrap_ci.py"

RUN_NAME="${RUN_NAME:-ex22_bootstrap_ci_llama31_n200}"
OUT_DIR="${ROOT}/outputs/ex22_bootstrap_ci"
LOG_DIR="${ROOT}/logs"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

LAYER="${LAYER:-15}"
ALPHAS="${ALPHAS:--1,1}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-1000}"
SEED="${SEED:-42}"
PERSONA_BASELINE="${PERSONA_BASELINE:-person}"
PERSONA_TARGETS="${PERSONA_TARGETS:-black person,white person}"
PERSONA_METRICS="${PERSONA_METRICS:-ER_label}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
cd "${ROOT}"

"${PY}" "${SCRIPT}" \
  --persona-project "${ROOT}" \
  --layer "${LAYER}" \
  --alphas="${ALPHAS}" \
  --bootstrap-iters "${BOOTSTRAP_ITERS}" \
  --seed "${SEED}" \
  --persona-baseline "${PERSONA_BASELINE}" \
  --persona-targets "${PERSONA_TARGETS}" \
  --persona-metrics "${PERSONA_METRICS}" \
  --output-dir "outputs/ex22_bootstrap_ci" \
  --run-name "${RUN_NAME}" \
  2>&1 | tee "${LOG_FILE}"

echo "[done] log=${LOG_FILE}"
